"""فحص استخراج المهارات على الإعلانات الحقيقية وتقييم جودته.

    python -m scripts.inspect_skills show --n 3       # أمثلة عربية/إنجليزية/مختلطة: الطريقة، الثقة، الدليل، السبب
    python -m scripts.inspect_skills export --n 40    # عينة للتعليم الكامل → data/labels/skills_to_annotate.csv
    python -m scripts.inspect_skills eval             # تقرير الجودة → reports/skill_extraction_report.md

مصدرا الحقيقة المرجعية (Ground Truth):
  1) تعليم كامل لعينة (true_skills = كل المهارات الصحيحة في الإعلان) ← يسمح بحساب Precision و Recall و F1.
  2) مراجعات بشرية لمهارات بعينها (صفحة «مراجعة التصنيفات» ← تبويب المهارات) ← تسمح بتقدير Precision فقط
     على العناصر المراجَعة (عينة غير عشوائية، تميل للحالات منخفضة الثقة).
بدون أيٍّ منهما لا تُحسب أي دقة، ويصرّح التقرير بذلك.
"""
import argparse
import json
from datetime import datetime

import pandas as pd

from core.config import ARTIFACTS_DIR, IS_DEMO_MODE, REPORTS_DIR, SKILL_ANNOTATIONS_FILE
from core.pipeline import load_processed, load_skill_decisions
from models.skill_extractor import load_taxonomy
from preprocessing.skill_review import load_reviews

OUT_JSON = ARTIFACTS_DIR / "skill_extraction_eval.json"
OUT_MD = REPORTS_DIR / ("skill_extraction_report_demo.md" if IS_DEMO_MODE else "skill_extraction_report.md")
MIN_ANNOTATED = 20   # أقل عدد إعلانات معلَّمة بالكامل لاعتبار المقاييس مؤشراً (لا يزال بتباين عالٍ)
METHOD_VIEWS = {"lexicon (قاموسي فقط)": lambda d: d.method.eq("lexicon"),
                "semantic (دلالي فقط)": lambda d: d.method.eq("semantic"),
                "lexicon+semantic": lambda d: d.method.eq("lexicon+semantic"),
                "كل ما يكتشفه القاموس (lexicon ∪ lexicon+semantic)": lambda d: d.method.str.contains("lexicon"),
                "النظام الهجين كاملاً": lambda d: d.method.notna()}


def _lang(jobs):
    if "lang" in jobs and jobs["lang"].notna().any():
        return jobs["lang"].fillna("en")
    ar = jobs["description"].fillna("").str.count(r"[؀-ۿ]")
    ln = jobs["description"].fillna("").str.len().clip(lower=1)
    return pd.Series(["ar" if a / l > 0.3 else "mixed" if a > 0 else "en" for a, l in zip(ar, ln)], index=jobs.index)


def show(n):
    jobs, _ = load_processed()
    dec = load_skill_decisions()
    _, by_id = load_taxonomy()
    jobs = jobs.assign(lang=_lang(jobs))
    for lg in ("ar", "en", "mixed"):
        sub = jobs[jobs.lang == lg].head(n)
        print(f"\n════ لغة الإعلان: {lg} ({int((jobs.lang == lg).sum())} إعلان)")
        for r in sub.itertuples():
            print(f"\n■ {r.title} — {r.company} [{r.source}]")
            d = dec[dec.job_id == r.job_id].sort_values(["decision", "confidence"], ascending=[True, False])
            for s in d.itertuples():
                mark = {"accepted": "✔", "review": "?", "rejected": "✖"}[s.decision]
                print(f"  {mark} {by_id[s.skill_id]['name_en']:24s} {s.method:16s} ثقة={s.confidence:.2f} «{str(s.evidence)[:70]}»")
                print(f"      السبب: {s.reason}")
            if d.empty:
                print("   (لا مرشحات)")


def export(n):
    jobs, _ = load_processed()
    dec = load_skill_decisions()
    jobs = jobs[~jobs.is_demo].assign(lang=_lang(jobs))
    sample = pd.concat([g.sample(min(len(g), max(1, n // 3)), random_state=0) for _, g in jobs.groupby("lang")])
    acc = dec[dec.decision == "accepted"].groupby("job_id").skill_id.apply(";".join)
    rev = dec[dec.decision == "review"].groupby("job_id").skill_id.apply(";".join)
    pd.DataFrame({"job_id": sample.job_id, "title": sample.title, "lang": sample.lang,
                  "text": sample.description.fillna("").str.slice(0, 2500),
                  "system_accepted (للمرجعية)": sample.job_id.map(acc).fillna(""),
                  "system_review (للمرجعية)": sample.job_id.map(rev).fillna(""), "true_skills": ""}) \
        .to_csv(SKILL_ANNOTATIONS_FILE, index=False, encoding="utf-8-sig")
    print(f"✔ {len(sample)} إعلاناً → {SKILL_ANNOTATIONS_FILE}")
    print("اكتب في true_skills كل معرفات المهارات الصحيحة المذكورة فعلاً في الإعلان مفصولة بـ ; — حتى غير المكتشفة.")


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p and r else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": None if p is None else round(p, 3),
            "recall": None if r is None else round(r, 3), "f1": None if f is None else round(f, 3)}


def evaluate():
    dec = load_skill_decisions()
    if dec is None:
        raise SystemExit("لا توجد قرارات مهارات — شغّل scripts.train_all أولاً.")
    _, by_id = load_taxonomy()
    acc = dec[dec.decision == "accepted"]
    res = {"generated_at": datetime.now().isoformat(timespec="minutes"), "mode": "demo" if IS_DEMO_MODE else "real",
           "decisions": {k: int(v) for k, v in dec.decision.value_counts().items()},
           "accepted_by_method": {k: int(v) for k, v in acc.method.value_counts().items()},
           "top_rejection_reasons": dec[dec.decision == "rejected"].reason.str.split("|").str[0].str.strip()
               .value_counts().head(10).to_dict()}

    # 1) تعليم كامل → P/R/F1
    full = None
    if SKILL_ANNOTATIONS_FILE.exists():
        ann = pd.read_csv(SKILL_ANNOTATIONS_FILE, dtype=str).fillna("")
        ann = ann[ann.true_skills.str.strip() != ""]
        if len(ann):
            full = {"n_annotated_jobs": int(len(ann)), "sufficient": len(ann) >= MIN_ANNOTATED, "by_method": {},
                    "false_positives": [], "false_negatives": []}
            truth = {r.job_id: {x.strip() for x in r.true_skills.split(";") if x.strip()} for r in ann.itertuples()}
            sub = acc[acc.job_id.isin(truth)]
            for name, f in METHOD_VIEWS.items():
                tp = fp = fn = 0
                pred_m = sub[f(sub)]
                for jid, t in truth.items():
                    p = set(pred_m[pred_m.job_id == jid].skill_id)
                    tp, fp, fn = tp + len(t & p), fp + len(p - t), fn + len(t - p)
                full["by_method"][name] = _prf(tp, fp, fn)
            for jid, t in truth.items():
                d = sub[sub.job_id == jid]
                for r in d[~d.skill_id.isin(t)].itertuples():
                    full["false_positives"].append({"job_id": jid, "skill": r.skill_id, "method": r.method,
                                                    "evidence": r.evidence, "reason": r.reason})
                for s in t - set(d.skill_id):
                    why = dec[(dec.job_id == jid) & (dec.skill_id == s)]
                    full["false_negatives"].append({"job_id": jid, "skill": s, "system_decision": why.decision.iloc[0]
                                                    if len(why) else "لم يُرشَّح", "reason": why.reason.iloc[0] if len(why) else ""})
    res["full_annotation"] = full

    # 2) مراجعات بشرية → Precision على العناصر المراجَعة فقط
    rv = load_reviews()
    partial = None
    if len(rv):
        m = rv.merge(dec[["job_id", "skill_id", "method", "decision"]], on=["job_id", "skill_id"], how="left")
        m["orig"] = m["decision"]
        partial = {"n_reviewed": int(len(m)), "correct": int((m.verdict == "correct").sum()),
                   "incorrect": int((m.verdict == "incorrect").sum()), "by_method": {}}
        for meth, g in m.groupby("method"):
            partial["by_method"][meth] = {"n": int(len(g)), "correct": int((g.verdict == "correct").sum()),
                                          "precision_on_reviewed": round(float((g.verdict == "correct").mean()), 3)}
        partial["errors"] = m[m.verdict == "incorrect"][["job_id", "skill_id", "method"]].to_dict("records")[:50]
    res["human_reviews"] = partial
    OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    OUT_MD.write_text(render_md(res, by_id), encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))


def render_md(res, by_id) -> str:
    L = [f"# تقرير جودة استخراج المهارات ({'تجريبي DEMO' if res['mode'] == 'demo' else 'بيانات حقيقية'})",
         f"تاريخ: {res['generated_at']}", "",
         "## مخرجات النظام (ليست مقاييس دقة)",
         f"- القرارات: {res['decisions']}", f"- المقبولة حسب الطريقة: {res['accepted_by_method']}",
         "- أكثر أسباب الرفض:"] + [f"  - {k}: {v}" for k, v in res["top_rejection_reasons"].items()] + [""]
    f = res["full_annotation"]
    L.append("## 1) التقييم على تعليم بشري كامل (Precision / Recall / F1)")
    if not f:
        L.append("**لا توجد Ground Truth.** لم يُعلَّم أي إعلان بالكامل، لذلك **لا توجد دقة مقاسة** لاستخراج المهارات. "
                 "أنشئ عينة: `python -m scripts.inspect_skills export --n 40` ثم املأ `true_skills`.")
    else:
        if not f["sufficient"]:
            L.append(f"⚠️ عدد الإعلانات المعلَّمة {f['n_annotated_jobs']} أقل من {MIN_ANNOTATED} — الأرقام مؤشر أولي غير مستقر.")
        L += ["", "Recall لكل صف = نسبة كل المهارات الصحيحة التي تكتشفها تلك الطريقة وحدها (مساهمتها)، لا دقة مستقلة.",
              "", "| الطريقة | TP | FP | FN | Precision | Recall | F1 |", "|---|---|---|---|---|---|---|"]
        for k, v in f["by_method"].items():
            L.append(f"| {k} | {v['tp']} | {v['fp']} | {v['fn']} | {v['precision']} | {v['recall']} | {v['f1']} |")
        L += ["", f"المهارات الصحيحة المكتشفة (TP للنظام كاملاً): {f['by_method']['النظام الهجين كاملاً']['tp']} • "
                  f"الخاطئة (FP): {f['by_method']['النظام الهجين كاملاً']['fp']} • الفائتة (FN): {f['by_method']['النظام الهجين كاملاً']['fn']}",
              "", "### تحليل الأخطاء — مقبولة خطأً (FP)"]
        fp = pd.DataFrame(f["false_positives"])
        if len(fp):
            for (sk, me), g in fp.groupby(["skill", "method"]):
                L.append(f"- **{by_id.get(sk, {}).get('name_en', sk)}** ({me}) × {len(g)} — مثال: «{g.evidence.iloc[0][:90]}»")
        L.append("### تحليل الأخطاء — فائتة (FN)")
        fn = pd.DataFrame(f["false_negatives"])
        if len(fn):
            for (sk, d), g in fn.groupby(["skill", "system_decision"]):
                L.append(f"- **{by_id.get(sk, {}).get('name_en', sk)}** × {len(g)} — قرار النظام: {d} {('— ' + g.reason.iloc[0][:80]) if g.reason.iloc[0] else ''}")
    p = res["human_reviews"]
    L += ["", "## 2) المراجعات البشرية لمهارات منفردة (Precision على العناصر المراجَعة فقط)"]
    if not p:
        L.append("لا توجد مراجعات بشرية بعد (صفحة «مراجعة التصنيفات» ← تبويب المهارات).")
    else:
        L.append(f"راجع المستخدم {p['n_reviewed']} مهارة: صحيحة {p['correct']} • خاطئة {p['incorrect']}. "
                 "العينة ليست عشوائية (تميل للحالات منخفضة الثقة) ولا تسمح بحساب Recall.")
        L += ["", "| الطريقة | المراجَعة | الصحيحة | Precision على المراجَع |", "|---|---|---|---|"]
        for k, v in p["by_method"].items():
            L.append(f"| {k} | {v['n']} | {v['correct']} | {v['precision_on_reviewed']} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["show", "export", "eval"])
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    {"show": lambda: show(a.n), "export": lambda: export(a.n), "eval": evaluate}[a.action]()
