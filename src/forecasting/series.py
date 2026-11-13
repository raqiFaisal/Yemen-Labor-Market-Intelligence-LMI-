"""بناء السلاسل الزمنية الشهرية من الإعلانات الفريدة لـ 4 عائلات: المهارات، التخصصات، المسميات الوظيفية، المحافظات.

القيمة = حصة الإعلانات الفريدة (وظائف/تدريب عملي) التي تذكر البند في ذلك الشهر. شهر بإعلانات < الحد الأدنى = NaN
(«غير مرصود» — لا يُعامل كطلب صفري). الكفاية تُفحص لكل سلسلة قبل أي تدريب.
"""
from __future__ import annotations

from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

from src.common import CONFIG, DATA_DIR

L = CONFIG["lstm"]
LABOR = {"JOB", "INTERNSHIP"}


def load_unique(data_dir: Path = DATA_DIR, as_of: date | None = None) -> pd.DataFrame:
    u = pd.read_csv(Path(data_dir) / "unique_opportunities.csv", dtype=str, keep_default_na=False)
    u = u[u.opportunity_type.isin(LABOR)].copy()
    from src.integration.canon import canon_specs
    u["specializations"] = u.specializations.map(canon_specs)          # «محاسبة» = «المحاسبة» = «Accounting»
    # Never evaluate on an unfinished month or on records posted after the audit date.
    as_of = as_of or date.today()
    d = pd.to_datetime(u.posting_date, errors="coerce")
    last_completed_month = pd.Period(as_of, freq="M") - 1
    u["month"] = d.dt.to_period("M")
    return u[(u.month.notna()) & (d.dt.date <= as_of) & (u.month <= last_completed_month)]


def title_groups(u: pd.DataFrame, top: int, train_end=None) -> pd.Series:
    from preprocessing.dedup import norm_title
    t = u.job_title.map(norm_title)
    # Ranking must be computed without validation/test months.
    if train_end is None:
        raise ValueError("title_groups requires a training cutoff to prevent future leakage")
    tr = t[(u.month <= train_end) & (t.str.len() >= 3)]
    keep = tr.value_counts().head(top).index
    return t.where(t.isin(keep), "")


def family_long(u: pd.DataFrame, family: str, data_dir: Path, train_end=None) -> pd.DataFrame:
    """(job_id, month, item) — بند واحد أو أكثر لكل إعلان."""
    if family == "skills":
        s = pd.read_csv(Path(data_dir) / "skills" / "job_skills.csv", dtype=str, keep_default_na=False)
        s = s[s.decision == "accepted"][["job_id", "skill_id"]].drop_duplicates()
        return u[["job_id", "month"]].merge(s, on="job_id").rename(columns={"skill_id": "item"})
    if family == "specializations":
        x = u[["job_id", "month", "specializations"]].assign(item=u.specializations.str.split(";")).explode("item")
        return x[x.item.fillna("").str.len() > 0][["job_id", "month", "item"]]
    if family == "job_titles":
        x = u[["job_id", "month"]].assign(item=title_groups(u, L["top_titles"], train_end))
        return x[x.item != ""]
    if family == "governorates":
        x = u[["job_id", "month"]].assign(item=u.governorate)
        return x[x.item != ""]
    raise ValueError(family)


def build_family(family: str, data_dir: Path = DATA_DIR, *, as_of: date | None = None) -> dict:
    """Construct monthly shares with train-only item eligibility and month completeness.

    Item ranking, number of mentions, and coverage are computed on train only.
    Months with too few postings remain missing; they are never made into targets.
    """
    as_of = as_of or date.today()
    u = load_unique(data_dir, as_of=as_of)
    if u.empty:
        return {"family": family, "status": "no data", "as_of": str(as_of)}
    rng = pd.period_range(u.month.min(), u.month.max(), freq="M")
    total = u.groupby("month").size().reindex(rng, fill_value=0)
    obs = total >= L["min_postings_per_month"]
    # Window eligibility uses observed posting coverage, not a particular candidate feature.
    o = obs.values
    win = None
    if o.any():
        last = int(np.where(o)[0].max())
        for start in range(last, -1, -1):
            if o[start] and o[start:last + 1].mean() >= L["min_coverage"]:
                win = (start, last)
    if win is None:
        return {"family": family, "status": "no observed months", "months_total": len(rng), "as_of": str(as_of)}
    idx = rng[win[0]:win[1] + 1]
    n_train = len(idx) - L["test_months"] - L["val_months"]
    if len(idx) < L["min_months"] or n_train <= L["lookback"]:
        return {"family": family, "status": "insufficient train months", "months_total": len(idx), "as_of": str(as_of)}
    train_end = idx[n_train - 1]
    lg = family_long(u, family, data_dir, train_end=train_end).drop_duplicates(["job_id", "item"])
    cnt = lg.groupby(["month", "item"]).size().unstack(fill_value=0).reindex(idx, fill_value=0)
    share = cnt.div(total.reindex(idx).replace(0, np.nan), axis=0).where(obs.reindex(idx), axis=0)
    # Determine each series' eligibility strictly from the training window.
    rows, eligible = [], []
    for it in share.columns:
        cov = float(share[it].iloc[:n_train].notna().mean())
        men = int(cnt[it].iloc[:n_train].sum())
        good = cov >= L["min_coverage"] and men >= L["min_mentions"]
        rows.append({"item": it, "train_months": n_train, "train_coverage": round(cov, 3),
                     "train_mentions": men, "ready": bool(good)})
        if good:
            eligible.append(it)
    suff = pd.DataFrame(rows).sort_values("train_mentions", ascending=False) if rows else pd.DataFrame(columns=["item", "ready"])
    return {"family": family, "status": "ok" if eligible else "insufficient", "window": (str(idx[0]), str(idx[-1])),
            "months": len(idx), "window_coverage": round(float(obs.reindex(idx).mean()), 3),
            "series": share[eligible], "counts": cnt[eligible], "total": total.reindex(idx),
            "sufficiency": suff, "ready": eligible, "train_feature_cutoff": str(train_end),
            "as_of": str(as_of), "excluded_incomplete_month": str(pd.Period(as_of, freq="M"))}
