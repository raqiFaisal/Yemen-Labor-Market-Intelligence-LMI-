"""LSTM للتنبؤ بالطلب الشهري (حصة الإعلانات) لكل عائلة: مهارات / تخصصات / مسميات / محافظات — مع مقارنة بخطوط أساس.

    python -m src.forecasting.train_lstm [--families skills specializations job_titles governorates]

المنهجية (لكل عائلة):
  1) فحص الكفاية (series.build_family): تُدرَّب السلاسل المستوفية فقط؛ إن لم تستوفِ أي سلسلة → لا تدريب ولا أرقام.
  2) تقسيم زمني: [train | validation = 3 أشهر | test = آخر 6 أشهر]. النماذج لا ترى أشهر test أثناء التدريب أو الاختيار.
  3) القيم المفقودة (أشهر غير مرصودة) تُملأ بآخر قيمة سابقة فقط (forward-fill — لا تسرب من المستقبل).
  4) تطبيع لكل سلسلة بمتوسط/انحراف محسوبين من جزء train فقط؛ يُحفظ Scaler.
  5) LSTM عالمي (مشترك بين سلاسل العائلة) بنافذة lookback؛ اختيار hidden/lr على validation (MAE)؛ إيقاف مبكر.
  6) تقييم test بتنبؤ خطوة واحدة متدحرج (المدخلات قيم فعلية سابقة) مقابل Naive و Moving Average-3.
  7) توقع مستقبلي حقيقي (أشهر بعد آخر بيانات) بتدحرج تكراري — يُعلَّم «توقع» ولا يُقيَّم.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import CONFIG, DATA_DIR, MODELS_DIR, REPORTS_DIR, env_info, now, set_seed, write_json
from src.evaluation.metrics import forecast_metrics
from src.forecasting.series import build_family

L = CONFIG["lstm"]


def make_model(hidden, layers, dropout):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            o, _ = self.lstm(x)
            return self.head(self.drop(o[:, -1])).squeeze(-1)
    return Net()


def windows(z: np.ndarray, lb: int, t_from: int, t_to: int, target_observed=None):
    """Supervise only observed targets; never train/validate on imputed labels.

    target_observed is a bool (T,S) mask. Test can still return predictions
    for unobserved periods if mask is omitted (metrics discard their NaN actuals).
    """
    X, y, meta = [], [], []
    for t in range(max(t_from, lb), t_to):
        for s in range(z.shape[1]):
            if target_observed is not None and not target_observed[t, s]:
                continue
            X.append(z[t - lb:t, s])
            y.append(z[t, s])
            meta.append((t, s))
    return np.array(X, dtype=np.float32)[..., None], np.array(y, dtype=np.float32), meta


def fit(Xtr, ytr, Xva, yva, hidden, lr, seed):
    import torch
    torch.manual_seed(seed)
    m = make_model(hidden, L["num_layers"], L["dropout"])
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lossf = torch.nn.MSELoss()
    Xt, yt, Xv, yv = (torch.tensor(a) for a in (Xtr, ytr, Xva, yva))
    best, best_state, bad, hist = np.inf, None, 0, []
    g = torch.Generator().manual_seed(seed)
    for ep in range(L["epochs"]):
        m.train()
        perm = torch.randperm(len(Xt), generator=g)
        tl = 0.0
        for i in range(0, len(perm), L["batch_size"]):
            b = perm[i:i + L["batch_size"]]
            opt.zero_grad()
            loss = lossf(m(Xt[b]), yt[b])
            loss.backward()
            opt.step()
            tl += loss.item() * len(b)
        m.eval()
        with torch.no_grad():
            vl = lossf(m(Xv), yv).item() if len(Xv) else np.nan
        hist.append({"epoch": ep + 1, "train_mse": tl / len(Xt), "val_mse": vl})
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= L["early_stopping_patience"]:
                break
    m.load_state_dict(best_state)
    return m, best, hist


def run_family(family, data_dir, out_models, out_rep, seed, as_of=None):
    import torch
    fam = build_family(family, data_dir, as_of=as_of)
    out_rep.mkdir(parents=True, exist_ok=True)
    if fam.get("status") != "ok":
        if "sufficiency" in fam:
            fam["sufficiency"].to_csv(out_rep / "sufficiency.csv", index=False, encoding="utf-8-sig")
        res = {"family": family, "trained": False, "reason": f"البيانات الزمنية غير كافية ({fam.get('status')})",
               **{k: v for k, v in fam.items() if k in ("window", "months", "window_coverage", "months_total")},
               "criteria": {k: L[k] for k in ("min_months", "min_coverage", "min_mentions", "min_postings_per_month")}}
        write_json(out_rep / "metrics.json", res)
        print(f"[{family}] ✖ لا تدريب: {res['reason']}")
        return res
    S = fam["series"]
    fam["sufficiency"].to_csv(out_rep / "sufficiency.csv", index=False, encoding="utf-8-sig")
    T, items = len(S), list(S.columns)
    n_test, n_val = L["test_months"], L["val_months"]
    n_train = T - n_test - n_val
    lb = int(min(L["lookback"], max(3, n_train - 3)))
    observed = np.isfinite(S.values)
    raw = S.ffill().fillna(0.0).values.astype(float)          # missing input: only past observations; pre-first=0
    if not np.isfinite(raw).all():
        raise ValueError("Invalid nonfinite temporal feature values")
    # No unobserved synthetic target is used for training/validation.
    if not observed[:n_train].any(axis=0).all():
        raise ValueError("Some series have no observed training target")
    mu = raw[:n_train].mean(0)
    sd = raw[:n_train].std(0) + 1e-6                            # Scaler من train فقط
    z = (raw - mu) / sd
    Xtr, ytr, _ = windows(z, lb, 0, n_train, observed)
    Xva, yva, _ = windows(z, lb, n_train, n_train + n_val, observed)
    if len(Xtr) == 0 or len(Xva) == 0:
        raise ValueError("Insufficient observed targets for train/validation temporal windows")
    grid = []
    for hidden in (16, 32):
        for lr in (3e-3, 1e-3):
            m, vl, hist = fit(Xtr, ytr, Xva, yva, hidden, lr, seed)
            grid.append((vl, hidden, lr, m, hist))
    vbest, hb, lrb, model, hist = min(grid, key=lambda g: g[0])
    # test: تنبؤ خطوة واحدة متدحرج (المدخلات قيم فعلية حتى الشهر السابق)
    Xte, yte, meta = windows(z, lb, n_train + n_val, T)
    model.eval()
    with torch.no_grad():
        pz = model(torch.tensor(Xte)).numpy()
    act_full = S.values                                         # القيم الفعلية (NaN = غير مرصود → خارج التقييم)
    rows = []
    for (t, s), p in zip(meta, pz):
        hist_vals = raw[:t, s]
        rows.append({"month": str(S.index[t]), "item": items[s], "actual": act_full[t, s], "lstm": p * sd[s] + mu[s],
                     "naive": hist_vals[-1], "moving_avg_3": hist_vals[-3:].mean()})
    pr = pd.DataFrame(rows)
    pr["lstm"] = pr.lstm.clip(lower=0)
    comp = {k: forecast_metrics(pr.actual, pr[k]) for k in ("lstm", "naive", "moving_avg_3")}
    per = pr.dropna(subset=["actual"]).groupby("item").apply(
        lambda g: pd.Series({k + "_mae": np.mean(np.abs(g.actual - g[k])) for k in ("lstm", "naive", "moving_avg_3")}),
        include_groups=False).reset_index()
    # توقع مستقبلي حقيقي: بعد آخر شهر بيانات، تدحرج تكراري
    H = L["forecast_horizon"]
    zz = z.copy()
    fut = []
    with torch.no_grad():
        for h in range(H):
            x = torch.tensor(zz[-lb:].T[..., None].astype(np.float32))
            nxt = model(x).numpy()
            zz = np.vstack([zz, nxt])
            fut.append(nxt * sd + mu)
    fmonths = [str(S.index[-1] + h + 1) for h in range(H)]
    fut = pd.DataFrame(np.clip(np.array(fut), 0, None), index=fmonths, columns=items)
    recent = S.tail(6).mean()
    fmean = fut.mean()
    trend = pd.DataFrame({"item": items, "recent_6m_mean": recent.values.round(5), "forecast_mean": fmean.values.round(5)})
    trend["change"] = (trend.forecast_mean - trend.recent_6m_mean).round(5)
    trend["direction"] = np.where((trend.change > 0.005) & (trend.change / trend.recent_6m_mean.clip(lower=1e-6) > 0.10), "rising",
                                  np.where((trend.change < -0.005) & (trend.change / trend.recent_6m_mean.clip(lower=1e-6) < -0.10),
                                           "falling", "stable"))
    # حفظ
    out_models.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_models / "lstm.pt")
    write_json(out_models / "scaler.json", {"items": items, "mean": mu.tolist(), "std": sd.tolist(), "fit_on": "train months only"})
    cfg = {"family": family, "architecture": f"LSTM(1→{hb}, layers={L['num_layers']}) → Linear(1)", "lookback": lb,
           "hidden_size": hb, "learning_rate": lrb, "dropout": L["dropout"], "epochs_run": len(hist), "best_val_mse_scaled": vbest,
           "window": fam["window"], "months": T, "n_train_months": n_train, "n_val_months": n_val, "n_test_months": n_test,
           "train_period": [str(S.index[0]), str(S.index[n_train - 1])],
           "validation_period": [str(S.index[n_train]), str(S.index[n_train + n_val - 1])],
           "test_period": [str(S.index[n_train + n_val]), str(S.index[-1])], "series": len(items),
           "grid": [{"hidden": h, "lr": l, "val_mse": v} for v, h, l, _, _ in grid], "seed": seed, "trained_at": now(),
           "env": env_info(), "test_used_during_training": False,
           "target_policy": "Only observed monthly shares used as supervised labels; input forward-fill only, first missing=0",
           "feature_selection": "Train-only mention frequency and coverage; title ranking train-only",
           "feature_selection_cutoff": fam["train_feature_cutoff"],
           "as_of": fam["as_of"], "excluded_incomplete_month": fam["excluded_incomplete_month"],
           "test_evaluation_mode": "One-step ahead using previous observed TEST months as inputs, not recursive 6-month forecast"}
    write_json(out_models / "config.json", cfg)
    pd.DataFrame(hist).to_csv(out_rep / "training_history.csv", index=False)
    pr.to_csv(out_rep / "test_predictions.csv", index=False, encoding="utf-8-sig")
    per.to_csv(out_rep / "per_series_test_mae.csv", index=False, encoding="utf-8-sig")
    fut.index.name = "month"
    fut.round(5).to_csv(out_rep / "future_forecast.csv", encoding="utf-8-sig")
    trend.sort_values("change", ascending=False).to_csv(out_rep / "trend_forecast.csv", index=False, encoding="utf-8-sig")
    S.round(5).to_csv(out_rep / "series_used.csv", encoding="utf-8-sig")
    plot(S, pr, fut, items, family, out_rep)
    res = {"family": family, "trained": True, **cfg, "test": comp,
           "lstm_beats_naive_mae": bool(comp["lstm"]["mae"] is not None and comp["lstm"]["mae"] < comp["naive"]["mae"]),
           "lstm_beats_ma3_mae": bool(comp["lstm"]["mae"] is not None and comp["lstm"]["mae"] < comp["moving_avg_3"]["mae"]),
           "future_horizon_months": fmonths, "note": "future_forecast = توقعات وليست قيماً مرصودة"}
    write_json(out_rep / "metrics.json", res)
    print(f"[{family}] ✔ {len(items)} سلسلة | LSTM MAE={comp['lstm']['mae']} | Naive={comp['naive']['mae']} | MA3={comp['moving_avg_3']['mae']}")
    return res


def plot(S, pr, fut, items, family, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top = pr.groupby("item").actual.mean().sort_values(ascending=False).head(6).index
    fig, axes = plt.subplots(len(top), 1, figsize=(10, 2.4 * len(top)), squeeze=False)
    for ax, it in zip(axes[:, 0], top):
        x = [str(i) for i in S.index]
        ax.plot(x, S[it].values, color="#333", lw=1.4, label="actual")
        p = pr[pr.item == it]
        ax.plot(p.month, p.lstm, "o-", color="#d62728", ms=3, label="LSTM (test)")
        ax.plot(p.month, p.naive, "--", color="#7f7f7f", lw=1, label="naive")
        ax.plot(p.month, p.moving_avg_3, ":", color="#1f77b4", lw=1.2, label="MA-3")
        ax.plot(list(fut.index), fut[it].values, "s--", color="#ff7f0e", ms=3, label="forecast (future)")
        ax.set_title(str(it), fontsize=9)
        ax.tick_params(axis="x", labelrotation=90, labelsize=6)
        step = max(1, len(x) // 18)
        ax.set_xticks(x[::step] + list(fut.index)[::2])
    axes[0, 0].legend(fontsize=7, ncol=5)
    fig.tight_layout()
    fig.savefig(out / "actual_vs_predicted.png", dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*", default=["skills", "specializations", "job_titles", "governorates"])
    ap.add_argument("--data_dir", type=Path, default=DATA_DIR)
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])
    ap.add_argument("--as-of", type=str, default=None, help="YYYY-MM-DD audit date; exclude its unfinished month")
    ap.add_argument("--cpu-threads", type=int, default=2, help="Limit CPU threads for reproducible performance")
    a = ap.parse_args()
    from datetime import date
    import torch
    audit_date = date.fromisoformat(a.as_of) if a.as_of else date.today()
    if a.cpu_threads > 0:
        torch.set_num_threads(a.cpu_threads)
    set_seed(a.seed)
    allres = {}
    for f in a.families:
        allres[f] = run_family(f, a.data_dir, MODELS_DIR / "lstm" / f, REPORTS_DIR / "evaluation" / f"lstm_{f}", a.seed, as_of=audit_date)
    sf = REPORTS_DIR / "evaluation" / "lstm_summary.json"
    summ = json.loads(sf.read_text(encoding="utf-8")) if sf.exists() else {}     # تشغيل عائلة واحدة لا يمحو البقية
    summ.update({k: {kk: v.get(kk) for kk in ("trained", "reason", "series", "test", "test_period", "lstm_beats_naive_mae",
                                               "lstm_beats_ma3_mae")} for k, v in allres.items()})
    write_json(sf, summ)


if __name__ == "__main__":
    main()
