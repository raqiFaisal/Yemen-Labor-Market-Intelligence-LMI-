"""التصنيف اليدوي (Gold Labels) لمجالات الوظائف — يمنح المصنف تقييماً حقيقياً.

1) تصدير عينة للتصنيف:
       python -m scripts.label_jobs export --n 200
   يُنشئ data/labels/to_label.csv فيه اقتراح آلي (suggested) وعمود domain فارغ.
   افتحه في Excel/Google Sheets واكتب المجال الصحيح في عمود domain لكل صف
   (بالمعرّف الإنجليزي أو الاسم العربي — القائمة في أعلى المخرجات). اترك الصف فارغاً إن لم تكن متأكداً.

2) استيراد التصنيفات:
       python -m scripts.label_jobs import
   تُضاف إلى data/labels/gold_labels.csv (لا تُحذف تصنيفات سابقة)، ثم أعد تشغيل scripts.train_all.

عند توفر 20 تصنيفاً يدوياً أو أكثر، يُقيَّم المصنف عليها كبيانات اختبار مستقلة لا تدخل التدريب.
"""
import argparse
import json

import pandas as pd

from core.config import DOMAINS_FILE, GOLD_LABELS_FILE, TO_LABEL_FILE
from core.pipeline import load_processed
from models.skill_extractor import load_taxonomy
from preprocessing.labeling import load_gold


def _domains():
    return json.loads(DOMAINS_FILE.read_text(encoding="utf-8"))


def export(n: int, seed: int = 0):
    jobs, js = load_processed()
    if jobs is None:
        raise SystemExit("شغّل python -m scripts.train_all أولاً.")
    gold = set(load_gold().job_id)
    trusted = jobs.get("label_source", pd.Series(None, index=jobs.index)).isin(["gold", "source"])
    pool = jobs[~jobs.job_id.isin(gold) & ~trusted]
    if pool.empty:
        raise SystemExit("لا توجد إعلانات حقيقية غير مصنفة.")
    # عينة متنوعة: موزعة على الاقتراحات الآلية بدل أخذ الأكثر شيوعاً فقط
    key = pool.get("domain_final", pd.Series("x", index=pool.index)).fillna("unclassified")
    per = max(1, n // max(key.nunique(), 1))
    sample = pool.groupby(key, group_keys=False).apply(lambda g: g.sample(min(len(g), per), random_state=seed))
    if len(sample) < n:
        rest = pool.drop(sample.index)
        sample = pd.concat([sample, rest.sample(min(len(rest), n - len(sample)), random_state=seed)])
    _, by_id = load_taxonomy()
    sk = js.groupby("job_id").skill_id.apply(lambda s: "، ".join(by_id[i]["name_ar"] for i in s if i in by_id))
    doms = _domains()
    out = pd.DataFrame({
        "job_id": sample.job_id, "title": sample.title, "company": sample.company,
        "description": sample.description.fillna("").str.slice(0, 400), "skills": sample.job_id.map(sk).fillna(""),
        # اقتراح آلي (قواعد/نموذج) لتسريع المراجعة فقط — ليس تسمية
        "suggested": sample.get("suggested_label", pd.Series(index=sample.index, dtype=object))
            .fillna(sample.get("domain_pred", pd.Series(index=sample.index, dtype=object)))
            .map(lambda d: f"{d} ({doms[d]['ar']})" if d in doms else ""),
        "source_category": sample.get("category_raw", pd.Series("", index=sample.index)),
        "domain": "",
    })
    out.to_csv(TO_LABEL_FILE, index=False, encoding="utf-8-sig")
    print(f"✔ {len(out)} إعلاناً → {TO_LABEL_FILE}")
    print("القيم المسموحة في عمود domain:")
    for k, v in doms.items():
        print(f"   {k:22s} | {v['ar']}")


def import_(path: str | None):
    f = path or TO_LABEL_FILE
    df = pd.read_csv(f, dtype=str).fillna("")
    doms = _domains()
    ar2id = {v["ar"]: k for k, v in doms.items()}
    en2id = {v["en"].lower(): k for k, v in doms.items()}

    def norm(v):
        v = v.strip()
        if not v:
            return None
        if v in doms:
            return v
        v0 = v.split("(")[0].strip()
        return doms.get(v0) and v0 or ar2id.get(v) or ar2id.get(v0) or en2id.get(v.lower()) or "INVALID"

    df["domain_id"] = df["domain"].map(norm)
    bad = df[df.domain_id == "INVALID"]
    if len(bad):
        print(f"⚠ {len(bad)} قيمة غير معروفة تم تجاهلها: {sorted(bad.domain.unique())[:10]}")
    ok = df[df.domain_id.notna() & (df.domain_id != "INVALID")][["job_id", "domain_id"]].rename(columns={"domain_id": "domain"})
    gold = pd.concat([load_gold(), ok]).drop_duplicates("job_id", keep="last")
    gold.to_csv(GOLD_LABELS_FILE, index=False, encoding="utf-8-sig")
    print(f"✔ أضيف {len(ok)} تصنيفاً — إجمالي التصنيفات اليدوية: {len(gold)} → {GOLD_LABELS_FILE}")
    print(gold.domain.value_counts().to_string())
    print("التالي: python -m scripts.train_all")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["export", "import"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--file")
    a = ap.parse_args()
    export(a.n) if a.action == "export" else import_(a.file)
