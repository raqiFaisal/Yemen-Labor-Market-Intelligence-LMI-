"""خط المشروع كاملاً: تحميل → توحيد → Embeddings (تزايدي) → استخراج المهارات → التسميات → FNN → LSTM → تقرير.

    python -m scripts.train_all                      # الوضع الحقيقي: data/raw فقط
    python -m scripts.train_demo                     # وضع العرض التجريبي المنفصل (data/demo → artifacts_demo)
    python -m scripts.train_all --freq W             # سلاسل أسبوعية لـ LSTM
    python -m scripts.train_all --allow-weak-labels  # تجريبي: تدريب المصنف على تسميات ضعيفة أيضاً (يُوسم Silver)

قواعد: لا يُدرَّب نموذج على بيانات غير كافية. إن لم تكفِ البيانات المعنونة يُعاد استخدام مصنف سابق فقط إذا كان
مدرباً على بيانات حقيقية وبنفس المُرمِّز (ويُوسم بمصدر تدريبه)، وإلا يُحذف ويُكتب السبب. كل شيء في data_report.json.
"""
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd

from core.artifacts import mark_not_trained, read_status
from core.config import (CLASSIFIER_FILE, CLASSIFIER_METRICS, CLF_MIN_CONFIDENCE, CLF_MIN_LABELED, CLF_MIN_PER_CLASS,
                         COUNTRY_YEMEN, DATA_REPORT, EMB_CACHE, EMB_IDS, IS_DEMO_MODE, LSTM_FILE, LSTM_FREQ,
                         LSTM_METRICS, LSTM_MIN_MENTIONS, LSTM_MIN_MONTHS, MODE, RAW_DIR)
from core.pipeline import (category_report, extract_job_skills_full, load_raw_sources, save_processed,
                           save_skill_decisions)
from models.classifier import JobClassifier, readiness, train_classifier
from models.embedding_store import cached_encode
from models.encoder import TextEncoder
from models.forecaster import monthly_share, series_sufficiency, time_readiness, train_lstm
from preprocessing.labeling import apply_labels, label_summary
from preprocessing.schema import job_text
from preprocessing.taxonomy_seed import DOMAINS, write_seed
from preprocessing.text import normalize_for_match

MARKETS = (("Yemen", lambda j: j[j.country == COUNTRY_YEMEN]), ("Global", lambda j: j[j.country != COUNTRY_YEMEN]))


def provenance(jobs: pd.DataFrame) -> dict:
    """مصدر البيانات وتاريخ جمعها لكل ملف/مصدر."""
    out = {}
    for src, g in jobs.groupby("source"):
        sc = pd.to_datetime(g.get("scraped_at"), errors="coerce")
        pdt = pd.to_datetime(g["posted_date"], errors="coerce")
        out[src] = {"rows": int(len(g)), "files": sorted(g["source_file"].dropna().unique().tolist()),
                    "market": g["market"].mode().iat[0],
                    "collected_from": None if sc.isna().all() else str(sc.min().date()),
                    "collected_to": None if sc.isna().all() else str(sc.max().date()),
                    "posted_from": None if pdt.isna().all() else str(pdt.min().date()),
                    "posted_to": None if pdt.isna().all() else str(pdt.max().date()),
                    "dated_pct": round(float(pdt.notna().mean() * 100), 1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default=None, help="مُهمل — استخدم scripts.train_demo لوضع العرض")
    ap.add_argument("--skip-lstm", action="store_true")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--allow-weak-labels", action="store_true", help="تجريبي: استخدام التسميات الضعيفة في التدريب")
    ap.add_argument("--no-weak-labels", action="store_true", help=argparse.SUPPRESS)  # توافق مع الإصدار السابق
    ap.add_argument("--no-semantic", action="store_true", help="تعطيل الاستخراج الدلالي للمهارات")
    ap.add_argument("--min-labeled", type=int, default=CLF_MIN_LABELED)
    ap.add_argument("--freq", default=LSTM_FREQ, choices=["M", "W"], help="M شهري | W أسبوعي")
    ap.add_argument("--min-periods", type=int, default=LSTM_MIN_MONTHS)
    ap.add_argument("--min-mentions", type=int, default=LSTM_MIN_MENTIONS)
    a = ap.parse_args()
    if a.demo:
        print("[i] --demo مُهمل: البيانات التجريبية منفصلة الآن. للعرض التجريبي: python -m scripts.train_demo")
    report = {"generated_at": time.strftime("%Y-%m-%d %H:%M"), "mode": MODE, "args": vars(a)}
    write_seed()

    print(f"[1/7] تحميل وتوحيد المصادر (الوضع: {MODE}، المجلد: {RAW_DIR}) ...")
    jobs, log = load_raw_sources()
    for l in log:
        print("     ", l)
    report["sources"] = log
    if jobs.empty:
        raise SystemExit(f"لا توجد بيانات في {RAW_DIR}. اجمع البيانات أولاً: python -m scripts.scrape_yemen")
    is_demo = IS_DEMO_MODE
    print(f"      {len(jobs)} إعلان | demo={is_demo}")
    report["data"] = {"n_jobs": int(len(jobs)), "is_demo": is_demo,
                      "by_market": {k: int(v) for k, v in jobs["market"].value_counts().items()},
                      "by_source": {k: int(v) for k, v in jobs["source"].value_counts().items()}}
    report["provenance"] = provenance(jobs)
    report["categories"] = category_report(jobs)

    print("[2/7] تحميل المُرمِّز (Transformer مجمَّد — لا تدريب ولا Fine-tuning) ...")
    enc = TextEncoder()
    texts = job_text(jobs).tolist()
    if enc.kind != "transformer":
        print(f"      ⚠ تعذر تحميل الـ Transformer ({enc.error}) — البديل TF-IDF+SVD (ليس Deep Learning)")
        enc.fit_fallback(texts)
    enc.save_info()

    print("[3/7] Embeddings للإعلانات (تزايدي: يُرمَّز الجديد فقط) ...")
    emb, cache = cached_encode(enc, texts, "jobs", show_progress=True)
    np.save(EMB_CACHE, emb)
    EMB_IDS.write_text(json.dumps(jobs["job_id"].tolist()))
    print(f"      embeddings: {emb.shape} via {enc.kind} | {cache}")
    report["encoder"] = enc.info() | {"shape": list(emb.shape), "cache": cache}

    print("[4/7] استخراج المهارات (قاموسي + دلالي) ...")
    semantic = enc.kind == "transformer" and not a.no_semantic
    job_skills, decisions, cands = extract_job_skills_full(jobs, enc, semantic=semantic)
    save_skill_decisions(decisions, cands)
    mc = job_skills["method"].value_counts().to_dict()
    dc = decisions["decision"].value_counts().to_dict()
    print(f"      مقبولة: {len(job_skills)} علاقة | {job_skills.skill_id.nunique()} مهارة مميزة | {mc}")
    print(f"      القرارات: {dc} | مرشحات خام: {len(cands)}")
    top_rej = decisions[decisions.decision == "rejected"]["reason"].str.split(":").str[0].value_counts().head(5).to_dict()
    report["skills"] = {"relations": int(len(job_skills)), "distinct": int(job_skills.skill_id.nunique()),
                        "by_method": {k: int(v) for k, v in mc.items()}, "semantic_enabled": semantic,
                        "decisions": {k: int(v) for k, v in dc.items()}, "top_rejection_reasons": top_rej,
                        "human_reviewed": int(decisions.get("human_verdict", pd.Series(dtype=object)).notna().sum()),
                        "extractor_version": "hybrid-3.0",
                        "jobs_without_skills": int((~jobs.job_id.isin(job_skills.job_id)).sum())}

    print("[5/7] التسميات ...")
    use_weak = a.allow_weak_labels and not a.no_weak_labels
    jobs = apply_labels(jobs, job_skills, use_weak=use_weak)
    ls = label_summary(jobs)
    print(f"      {ls}  (التسميات الضعيفة في التدريب: {'مفعلة ⚠' if use_weak else 'معطلة'})")
    report["labels"] = ls | {"weak_labels_used_for_training": use_weak,
                             "suggestions_available": int(jobs["suggested_label"].notna().sum())}
    jobs["dup_group"] = (jobs["title"].fillna("") + "|" + jobs["description"].fillna("").str.slice(0, 500)) \
        .map(lambda t: hashlib.md5(normalize_for_match(t).encode()).hexdigest()[:10])

    print("[6/7] مصنف المجالات (PyTorch FNN) ...")
    lab = jobs["label"].notna().to_numpy()
    rd = readiness(jobs.loc[lab, "label"], a.min_labeled, CLF_MIN_PER_CLASS, list(DOMAINS))
    report["classifier"] = rd
    for d, v in rd["per_class"].items():
        print(f"      {d:22s} لديه {v['have']:4d} | ينقص للحد الأدنى {v['missing_for_min']:3d} | للموصى به {v['missing_for_recommended']:3d}")
    jobs["domain_pred"], jobs["domain_conf"] = None, np.nan
    trained = False
    if rd["ok"]:
        try:
            m = train_classifier(emb[lab], jobs.loc[lab, "label"].tolist(), enc.info(),
                                 label_sources=jobs.loc[lab, "label_source"].tolist(),
                                 groups=jobs.loc[lab, "dup_group"].tolist(), epochs=a.epochs, is_demo=is_demo,
                                 min_labeled=a.min_labeled, all_classes=list(DOMAINS))
            print(f"      ✔ test acc={m['test']['accuracy']:.3f} F1={m['test']['f1_macro']:.3f} [تقييم: {m['eval_type']}] "
                  f"| خط الأساس (الفئة الأكثر): acc={m['baseline_majority']['accuracy']:.3f}")
            report["classifier"] |= {"trained": True, "eval_type": m["eval_type"], "test": m["test"],
                                     "baseline_majority": m["baseline_majority"]}
            trained = True
        except ValueError as e:
            rd = {**rd, "ok": False, "reason": str(e)}
    if not trained:
        prev = read_status(CLASSIFIER_METRICS)
        enc_ok = prev.get("encoder", {}).get("name") == enc.info()["name"] and \
            prev.get("encoder", {}).get("dim") == enc.dim
        if prev["status"] == "trained" and not prev.get("trained_on_demo_data") and enc_ok and CLASSIFIER_FILE.exists():
            prev["reused"] = {"at": report["generated_at"], "why": rd["reason"],
                              "note": f"نموذج مدرب سابقاً بتاريخ {prev.get('trained_at')} على {prev.get('n_train')} سجلاً حقيقياً"}
            CLASSIFIER_METRICS.write_text(json.dumps(prev, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"      ↺ البيانات الحالية غير كافية ({rd['reason']}) — إعادة استخدام المصنف السابق المدرب بتاريخ "
                  f"{prev.get('trained_at')}")
            report["classifier"] |= {"trained": False, "reused_previous": prev["reused"]}
            trained = True
        else:
            print(f"      ⚠ لم يُدرَّب المصنف: {rd['reason']}")
            mark_not_trained(CLASSIFIER_FILE, CLASSIFIER_METRICS, rd["reason"], report["classifier"])
            report["classifier"]["trained"] = False
    if trained and CLASSIFIER_FILE.exists():
        clf = JobClassifier()
        if clf.in_dim == emb.shape[1]:
            pred, conf = clf.predict(emb)
            jobs["domain_pred"], jobs["domain_conf"] = pred, conf

    # المجال النهائي: تسمية موثوقة ← تنبؤ النموذج بثقة ≥ العتبة ← "غير مصنف" (الاقتراحات الضعيفة لا تُعرض كمجال)
    confident = jobs["domain_conf"].fillna(0) >= CLF_MIN_CONFIDENCE
    model_dom = jobs["domain_pred"].where(confident)
    jobs["domain_final"] = jobs["label"].fillna(model_dom).fillna("unclassified")
    jobs["domain_source"] = np.where(jobs["label"].notna(), jobs["label_source"].fillna("none"),
                                     np.where(model_dom.notna(), "model", "none"))
    report["domain_final"] = {k: int(v) for k, v in jobs["domain_source"].value_counts().items()}
    save_processed(jobs, job_skills)

    print(f"[7/7] LSTM للتنبؤ بالطلب (وحدة: {a.freq}) — الأسواق منفصلة ...")
    tr = {mk: time_readiness(f(jobs), job_skills, a.freq, a.min_periods, a.min_mentions) for mk, f in MARKETS}
    report["time_series"] = {"by_market": tr}
    series, periods, skipped, reasons = {}, {}, 0, {}
    if not a.skip_lstm:
        for mk, f in MARKETS:
            share = monthly_share(f(jobs), job_skills, freq=a.freq)
            if share.empty:
                continue
            counts = share.attrs["counts"]
            for sid in share.columns:
                ok, why = series_sufficiency(share[sid], int(counts[sid].sum()), a.min_periods, a.min_mentions, a.freq)
                if ok:
                    series[f"{mk}|{sid}"] = (share[sid].to_numpy(), int(counts[sid].sum()))
                    periods[f"{mk}|{sid}"] = [str(p) for p in share.index]
                else:
                    skipped += 1
                    reasons[why.split(" — ")[0][:60]] = reasons.get(why.split(" — ")[0][:60], 0) + 1
        for mk, t in tr.items():
            print(f"      {mk}: {t['n_dated']} إعلان مؤرخ | المدى {t['span_periods']} {t['unit']} | "
                  f"مرصودة {t.get('observed_periods', 0)} | {t.get('blocking') or 'جاهز'}")
        print(f"      سلاسل صالحة: {len(series)} | مستبعدة: {skipped}")
        for r_, n_ in sorted(reasons.items(), key=lambda x: -x[1])[:5]:
            print(f"        سبب الاستبعاد ({n_} سلسلة): {r_}")
    report["time_series"].update(valid_series=len(series), skipped_series=skipped, skip_reasons=reasons)
    if series:
        m = train_lstm(series, is_demo=is_demo, freq=a.freq, periods=periods)
        m |= {"min_periods": max(a.min_periods, 0), "min_mentions": a.min_mentions}
        LSTM_METRICS.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
        t = m["test"]
        print(f"      ✔ LSTM MAE={t['lstm']['mae']:.4f} | Naive={t['naive']['mae']:.4f} | MA3={t['moving_avg_3']['mae']:.4f}")
        report["time_series"]["trained"] = True
    else:
        blocking = "; ".join(f"{k}: {v.get('blocking')}" for k, v in tr.items() if v.get("blocking"))
        reason = "تم تخطي LSTM (--skip-lstm)." if a.skip_lstm else (blocking or "لا توجد سلاسل زمنية كافية.")
        print(f"      ⚠ لم يُدرَّب LSTM: {reason}")
        mark_not_trained(LSTM_FILE, LSTM_METRICS, reason, {"by_market": tr, "skip_reasons": reasons})
        report["time_series"]["trained"] = False

    DATA_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"[✓] تقرير الجاهزية: {DATA_REPORT}")


if __name__ == "__main__":
    main()
