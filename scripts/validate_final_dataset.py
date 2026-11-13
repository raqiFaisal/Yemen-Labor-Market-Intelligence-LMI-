"""اختبارات ما قبل التدريب على FINAL_TRAINING_DATASET + تقرير الجودة HTML + dataset_manifest.json + data_dictionary.md.

    python -m scripts.validate_final_dataset [--dir DIR]      # رمز الخروج 1 إن فشل فحص حرج

كل فحص يُشغَّل فعلياً على الملفات النهائية (لا على الذاكرة). النتائج: dataset_statistics/quality_checks.json
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import ROOT
from preprocessing.dedup import norm_text_hash

SPLITS = ("train", "validation", "test")
TASKS = {"classification_opportunity_type": "label", "classification_opportunity_type/relevance_filter": "label",
         "classification_geographic_scope": "label", "skills/extraction": None}
REQUIRED = {
    "all_posts.csv": ["post_id", "source_channel", "source_message_id", "source_url", "message_date", "raw_text", "opportunity_type",
                      "type_status", "excluded_reason"],
    "all_opportunities.csv": ["job_id", "post_id", "canonical_job_id", "is_canonical", "dup_status", "source_url", "raw_text", "job_title",
                              "organization", "skills", "qualifications", "education", "experience_min_years", "specializations",
                              "location", "governorate", "employment_type", "salary", "application_deadline", "apply_url",
                              "opportunity_type", "geographic_scope", "scope_detail", "posting_date", "message_date"],
    "unique_opportunities.csv": ["job_id", "canonical_job_id", "n_records", "sources", "opportunity_type"],
    "duplicate_pairs.csv": ["job_id_a", "job_id_b", "evidence", "score"],
}
DESCRIPTIONS = {
    "post_id": "معرف المنشور: tg:<channel>:<message_id>", "job_id": "معرف الفرصة (…#p<n> للوظيفة رقم n داخل منشور متعدد)",
    "canonical_job_id": "معرف مجموعة التكرار — كل النسخ المكررة لنفس الإعلان تتشاركه", "is_canonical": "الممثل المختار للمجموعة",
    "dup_status": "unique | duplicate | possible_duplicate_review", "dup_reason": "دليل الربط بالتكرار",
    "source_channel": "قناة Telegram", "source_message_id": "رقم الرسالة في القناة", "source_url": "رابط المنشور الأصلي",
    "raw_text": "النص الأصلي حرفياً", "text_clean": "نص منظّف للنماذج (لا يغير المعنى)", "text_source": "post_text | ocr",
    "ocr_text": "نص مستخرج من صورة المنشور بـ OCR (قد يحوي أخطاء قراءة)", "telegraph_url": "صفحة التفاصيل الكاملة",
    "job_title": "المسمى الوظيفي كما ورد", "organization": "الجهة كما وردت", "organization_canonical": "اسم الجهة الموحد",
    "location": "الموقع كما ورد", "governorate": "المحافظة اليمنية (الاسم العربي المعتمد)", "governorates_all": "كل المحافظات المذكورة",
    "country": "الدولة", "work_location_type": "onsite | remote | hybrid | unknown",
    "description": "الوصف (نص telegra.ph الكامل إن وُجد)", "skills": "معرفات المهارات المقبولة مفصولة بـ ;",
    "qualifications": "قسم المؤهلات/المتطلبات كما ورد", "education": "مستويات المؤهل المذكورة", "education_min": "أدنى مؤهل مذكور",
    "experience_min_years": "أدنى سنوات خبرة (null إن لم تُذكر)", "experience_max_years": "أعلى سنوات خبرة",
    "specializations": "التخصصات المذكورة", "employment_type": "full_time;part_time;contract;…",
    "salary": "الراتب كما ورد", "salary_amount_min": "أدنى مبلغ مذكور", "salary_amount_max": "أعلى مبلغ مذكور",
    "salary_currency": "YER | USD | SAR (null إن لم تُذكر)", "salary_negotiable": "صياغة «بعد المقابلة/بالاتفاق»",
    "application_deadline": "آخر موعد (ISO) — null إن لم يُذكر أو كان غير منطقي", "deadline_raw_text": "نص الموعد كما ورد",
    "apply_url": "رابط التقديم (ليس روابط القنوات)", "apply_email": "بريد التقديم",
    "published_date": "تاريخ النشر المكتوب داخل الإعلان", "message_date": "تاريخ رسالة Telegram",
    "posting_date": "التاريخ المعتمد للسلاسل الزمنية", "posting_date_source": "in_text | message_date",
    "opportunity_type": "JOB | INTERNSHIP | TRAINING | SCHOLARSHIP | VOLUNTEERING | OTHER_OPPORTUNITY (| NOT_RELEVANT في all_posts)",
    "type_confidence": "ثقة استدلالية (ليست احتمالاً معايراً)", "type_status": "accepted | review | excluded",
    "type_reason": "الكلمات/القالب الذي حسم التصنيف", "geographic_scope": "YEMEN | REMOTE_GLOBAL | OUTSIDE_YEMEN | UNKNOWN",
    "scope_detail": "LOCAL | SINGLE_GOVERNORATE | MULTI_GOVERNORATE | NATIONWIDE | YEMEN_UNSPECIFIED | INTERNATIONAL | REMOTE | UNKNOWN",
    "split_random": "قسم التقسيم المجمّع (عام)", "excluded_reason": "سبب الاستبعاد من الفرص",
    "text": "نص الإدخال للنموذج", "label": "التسمية (Silver)", "label_id": "رقم التسمية حسب label_map.json",
    "confidence": "ثقة التسمية الآلية", "label_source": "silver_rules | silver_extractor", "group": "مفتاح التقسيم (لا يتكرر بين الأقسام)",
}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_any(p: Path):
    if p.suffix == ".csv":
        return pd.read_csv(p, dtype=str, keep_default_na=False, low_memory=False)
    if p.suffix == ".jsonl":
        return pd.read_json(p, lines=True, dtype=False) if p.stat().st_size else pd.DataFrame()
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    if p.suffix in (".conll", ".md", ".txt"):
        return p.read_text(encoding="utf-8")
    return None


class Checks:
    def __init__(self):
        self.items = []

    def add(self, name, ok, detail="", critical=True):
        self.items.append({"check": name, "passed": bool(ok), "critical": critical, "detail": str(detail)[:600]})
        print(("✔ " if ok else ("✖ " if critical else "⚠ ")) + name + (f" — {detail}" if detail and not ok else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path(os.environ.get("LMI_FINAL_DIR", ROOT / "FINAL_TRAINING_DATASET")))
    D = ap.parse_args().dir
    C = Checks()
    files = sorted(p for p in D.rglob("*") if p.is_file() and p.name not in ("dataset_manifest.json", "dataset_quality_report.html"))

    # 1) قابلية القراءة
    loaded, bad = {}, []
    for p in files:
        try:
            loaded[p] = read_any(p)
        except Exception as e:  # noqa: BLE001
            bad.append(f"{p.relative_to(D)}: {type(e).__name__}: {e}")
    C.add(f"قراءة جميع الملفات ({len(files)})", not bad, "; ".join(bad))
    df = lambda rel: loaded.get(D / rel)  # noqa: E731

    # 2) الأعمدة المطلوبة
    miss = {f: [c for c in cols if c not in df(f).columns] for f, cols in REQUIRED.items() if df(f) is not None}
    miss = {k: v for k, v in miss.items() if v}
    C.add("الأعمدة المطلوبة موجودة في الملفات الرئيسية", not miss and all(df(f) is not None for f in REQUIRED), miss)
    req_dirs = ["classification_opportunity_type", "classification_geographic_scope", "ner", "skills", "timeseries", "gold_annotation",
                "review_queue", "dataset_statistics"]
    C.add("كل المجلدات المطلوبة موجودة", all((D / d).is_dir() for d in req_dirs), [d for d in req_dirs if not (D / d).is_dir()])

    # 3) التسرب بين الأقسام + التسميات + التناقض + النصوص الفارغة
    split_sizes = []
    for task, lab in TASKS.items():
        parts = {sp: df(f"{task}/{sp}.csv") for sp in SPLITS}
        if any(v is None for v in parts.values()):
            C.add(f"[{task}] ملفات train/validation/test", False, "ملف مفقود")
            continue
        for sp, d in parts.items():
            split_sizes.append({"task": task, "split": sp, "rows": len(d),
                                **({"labels": json.dumps(d[lab].value_counts().to_dict(), ensure_ascii=False)} if lab else {})})
        for key in ("group", "id"):
            sets = {sp: set(d[key]) for sp, d in parts.items()}
            inter = sum(len(sets[a] & sets[b]) for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")])
            C.add(f"[{task}] لا تداخل في {key} بين الأقسام", inter == 0, f"{inter} متداخل")
        hs = {sp: set(h for h in d.text.map(norm_text_hash) if h) for sp, d in parts.items()}
        inter = sum(len(hs[a] & hs[b]) for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")])
        C.add(f"[{task}] لا نصوص متطابقة (بعد التطبيع) بين الأقسام", inter == 0, f"{inter} نص متكرر عبر الأقسام")
        allr = pd.concat(parts.values())
        C.add(f"[{task}] لا نصوص قصيرة/فارغة (<20 حرفاً)", (allr.text.str.len() >= 20).all(),
              f"{int((allr.text.str.len() < 20).sum())} صف")
        if lab:
            lm = json.loads((D / task / "label_map.json").read_text(encoding="utf-8"))
            outside = set(allr[lab]) - set(lm["label2id"])
            C.add(f"[{task}] كل التسميات ضمن الفئات المحددة", not outside, outside)
            ids_ok = (allr.label_id.astype(int) == allr[lab].map(lm["label2id"])).all()
            C.add(f"[{task}] label_id يطابق label_map", ids_ok)
            g = allr.assign(h=allr.text.map(norm_text_hash))
            g = g[g.h != ""].groupby("h")[lab].nunique()
            C.add(f"[{task}] لا تسميات متناقضة لنفس النص", (g <= 1).all(), f"{int((g > 1).sum())} نص بتسميتين")
            tr = parts["train"][lab].value_counts()
            small = [k for k in lm["label2id"] if tr.get(k, 0) < 5]
            C.add(f"[{task}] كل فئة لها ≥5 أمثلة تدريب", not small, f"فئات ناقصة: {small}", critical=False)
        else:
            vocab = set(json.loads((D / task / "skill_vocab.json").read_text(encoding="utf-8")))
            bad_sk = set()
            for d in parts.values():
                for v in d.skills:
                    bad_sk |= set(json.loads(v.replace("'", '"')) if v.startswith("[") else []) - vocab
            C.add(f"[{task}] كل المهارات ضمن القاموس", not bad_sk, list(bad_sk)[:10])

    # 4) NER / BIO
    lab_list = json.loads((D / "ner" / "label_list.json").read_text(encoding="utf-8"))
    bio, l2i = set(lab_list["bio_tags"]), lab_list["label2id"]
    errs, ner_rows, groups, ner_hash = [], {}, {}, {}
    for sp in SPLITS:
        recs = [json.loads(l) for l in open(D / "ner" / f"{sp}.jsonl", encoding="utf-8") if l.strip()]
        ner_rows[sp], groups[sp] = len(recs), {r["canonical_job_id"] for r in recs}
        ner_hash[sp] = {norm_text_hash(r["text"]) for r in recs} - {""}
        split_sizes.append({"task": "ner", "split": sp, "rows": len(recs)})
        for r in recs:
            n = len(r["tokens"])
            if not (n == len(r["ner_tags"]) == len(r["ner_tag_ids"]) == len(r["token_offsets"])):
                errs.append(f"{r['id']}: أطوال غير متساوية")
                continue
            prev = "O"
            for t, i, (s, e), tok in zip(r["ner_tags"], r["ner_tag_ids"], r["token_offsets"], r["tokens"]):
                if t not in bio or l2i[t] != i:
                    errs.append(f"{r['id']}: وسم غير صالح {t}")
                if t.startswith("I-") and prev[2:] != t[2:]:
                    errs.append(f"{r['id']}: {t} بعد {prev}")
                if r["text"][s:e] != tok:
                    errs.append(f"{r['id']}: موضع الرمز لا يطابق النص")
                prev = t
            for sp_ in r["spans"]:
                if r["text"][sp_["start"]:sp_["end"]] != sp_["text"]:
                    errs.append(f"{r['id']}: نص الكيان لا يطابق موضعه")
    C.add(f"NER: صحة BIO والأطوال والمواضع ({sum(ner_rows.values())} مثال)", not errs, "; ".join(errs[:5]))
    inter = len(groups["train"] & groups["validation"]) + len(groups["train"] & groups["test"]) + len(groups["validation"] & groups["test"])
    C.add("NER: لا تداخل في مجموعات الإعلانات بين الأقسام", inter == 0, f"{inter}")
    inter = len(ner_hash["train"] & ner_hash["test"]) + len(ner_hash["train"] & ner_hash["validation"])
    C.add("NER: لا نصوص متطابقة بين الأقسام", inter == 0, f"{inter}")

    # 5) السلاسل الزمنية
    ts = D / "timeseries"
    mu = pd.read_csv(ts / "monthly_unique_postings.csv")
    months = pd.PeriodIndex(mu.month, freq="M")
    cont = len(mu) == 0 or (months == pd.period_range(months.min(), months.max(), freq="M")).all()
    C.add("السلاسل: الأشهر متصلة ومرتبة بلا فجوات في الفهرس", cont)
    num = mu.drop(columns=["month", "observed"])
    C.add("السلاسل: لا أعداد سالبة", (num.values >= 0).all())
    sc = pd.read_csv(ts / "monthly_skill_counts.csv")
    C.add("السلاسل: عدد إعلانات المهارة ≤ إجمالي الإعلانات في الشهر", (sc.postings_with_skill <= sc.job_postings).all())
    C.add("السلاسل: الحصة بين 0 و1", sc.share.dropna().between(0, 1).all())
    wide = pd.read_csv(ts / "monthly_skill_share_wide.csv", index_col=0)
    unobs = mu.set_index("month").observed.astype(str).str.lower() != "true"
    C.add("السلاسل: الأشهر غير المرصودة = NaN (لا صفر وهمي)", wide[unobs.reindex(wide.index).fillna(True).values].isna().all().all())
    uo = df("unique_opportunities.csv")
    dated = pd.to_datetime(uo.posting_date, errors="coerce").notna().sum() if uo is not None else 0
    C.add("السلاسل: مجموع الإعلانات الشهرية = الإعلانات الفريدة ذات التاريخ", int(mu.unique_opportunities.sum()) == int(dated),
          f"{int(mu.unique_opportunities.sum())} مقابل {int(dated)}")
    suff = json.loads((ts / "lstm_sufficiency.json").read_text(encoding="utf-8"))
    C.add(f"LSTM: مهارات تستوفي الشروط = {suff.get('skills_ready', 0)} (نافذة {suff.get('window')})",
          suff.get("skills_ready", 0) > 0, "لا توجد مهارة تستوفي الشروط", critical=False)
    if (ts / "lstm_ready" / "skill_share_series.csv").exists():
        ls = pd.read_csv(ts / "lstm_ready" / "skill_share_series.csv")
        order_ok = list(ls.split) == sorted(ls.split, key=["train", "validation", "test"].index)
        C.add("LSTM: التقسيم زمني (train ثم validation ثم test)", order_ok)

    # 6) قيم منطقية وتواريخ
    ao = df("all_opportunities.csv")
    now = datetime.now()
    pdt = pd.to_datetime(ao.posting_date, errors="coerce")
    C.add("التواريخ: لا تاريخ نشر في المستقبل", not (pdt > now + timedelta(days=1)).any(), int((pdt > now + timedelta(days=1)).sum()))
    C.add("التواريخ: لا تاريخ نشر قبل 2010", not (pdt < "2010-01-01").any(), int((pdt < "2010-01-01").sum()))
    dl = pd.to_datetime(ao.application_deadline, errors="coerce")
    md = pd.to_datetime(ao.message_date, errors="coerce", utc=True).dt.tz_localize(None)
    C.add("التواريخ: آخر موعد ليس قبل الرسالة بأكثر من 7 أيام", not ((dl - md).dt.days < -7).any(), int(((dl - md).dt.days < -7).sum()))
    ex = pd.to_numeric(ao.experience_min_years, errors="coerce")
    ex2 = pd.to_numeric(ao.experience_max_years, errors="coerce")
    C.add("القيم: الخبرة بين 0 و30 سنة والحد الأدنى ≤ الأعلى", ex.dropna().between(0, 30).all() and not (ex > ex2).any())
    sal = pd.to_numeric(ao.get("salary_amount_min"), errors="coerce")
    C.add("القيم: مبالغ الرواتب موجبة", (sal.dropna() > 0).all())
    C.add("التكرار: معرف canonical فريد في unique_opportunities", uo.canonical_job_id.is_unique)
    C.add("التكرار: لكل مجموعة ممثل واحد", ao[ao.is_canonical.str.lower() == "true"].canonical_job_id.is_unique)
    lab_types = {"JOB", "INTERNSHIP", "TRAINING", "SCHOLARSHIP", "VOLUNTEERING", "OTHER_OPPORTUNITY"}
    C.add("الفرص: لا أنواع خارج الأنواع الستة في all_opportunities", set(ao.opportunity_type) <= lab_types, set(ao.opportunity_type) - lab_types)
    ap_ = df("all_posts.csv")
    C.add("المنشورات: معرف فريد لكل منشور", ap_.post_id.is_unique)
    C.add("المنشورات: كل فرصة مرتبطة بمنشور موجود", set(ao.post_id) <= set(ap_.post_id))
    g = df("gold_annotation/opportunity_gold_to_annotate.csv")
    if g is not None:
        n = int((g.gold_opportunity_type.str.strip() != "").sum())
        C.add(f"Gold: {len(g)} صفاً للتصنيف البشري (مصنّف فعلاً: {n}) — لا يُعامل كـ Gold قبل التعبئة",
              "opportunity_type" not in g.columns and len(g) > 0, "الملف يحتوي توقعات النظام" if "opportunity_type" in g.columns else "")

    # ---------- إحصاءات + manifest + قاموس + HTML
    st = D / "dataset_statistics"
    passed = sum(i["passed"] for i in C.items)
    crit_fail = [i for i in C.items if i["critical"] and not i["passed"]]
    (st / "quality_checks.json").write_text(json.dumps({"generated_at": now.isoformat(timespec="seconds"), "passed": passed,
                                                        "total": len(C.items), "critical_failures": len(crit_fail),
                                                        "checks": C.items}, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.DataFrame(split_sizes).to_csv(st / "split_sizes.csv", index=False, encoding="utf-8-sig")
    rows = []
    for p, obj in loaded.items():
        n = len(obj) if isinstance(obj, pd.DataFrame) else None
        if p.suffix == ".conll":
            n = obj.count("\n\n") + (1 if obj.strip() else 0)
        rows.append({"file": str(p.relative_to(D)), "rows": n, "bytes": p.stat().st_size})
    fr = pd.DataFrame(rows)
    fr.to_csv(st / "file_row_counts.csv", index=False, encoding="utf-8-sig")
    manifest = {"name": "FINAL_TRAINING_DATASET — AI Labor Market Intelligence (Yemen)", "generated_at": now.isoformat(timespec="seconds"),
                "label_quality": "SILVER (rule-based with reasons). Human Gold annotation pending — see gold_annotation/.",
                "summary": json.loads((st / "summary.json").read_text(encoding="utf-8")),
                "quality_checks": {"passed": passed, "total": len(C.items), "critical_failures": len(crit_fail)},
                "files": []}
    for p in files:
        obj = loaded.get(p)
        ent = {"path": str(p.relative_to(D)), "bytes": p.stat().st_size, "sha256": sha(p)}
        if isinstance(obj, pd.DataFrame):
            ent["rows"] = len(obj)
            ent["columns"] = {c: infer_dtype(obj[c]) for c in obj.columns}
        manifest["files"].append(ent)
    (D / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    write_dictionary(D, loaded)
    write_html(D, C.items, fr, pd.DataFrame(split_sizes), manifest["summary"])
    print(f"\nالنتيجة: {passed}/{len(C.items)} فحصاً ناجحاً | إخفاقات حرجة: {len(crit_fail)}")
    raise SystemExit(1 if crit_fail else 0)


def infer_dtype(s: pd.Series) -> str:
    v = s.dropna()
    v = v[v.astype(str) != ""] if v.dtype == object else v
    if v.empty:
        return "string (all null)"
    if v.map(lambda x: isinstance(x, (list, dict))).any():
        return "list/object"
    t = v.astype(str)
    if t.str.fullmatch(r"-?\d+").all():
        return "integer"
    if t.str.fullmatch(r"-?\d+(\.\d+)?").all():
        return "float"
    if t.str.lower().isin(["true", "false"]).all():
        return "boolean"
    if t.str.fullmatch(r"\d{4}-\d{2}(-\d{2})?([T ][\d:.+\-Z]+)?").mean() > 0.95:
        return "date/datetime (ISO)"
    return "string"


def write_dictionary(D: Path, loaded: dict):
    L = ["# قاموس البيانات — FINAL_TRAINING_DATASET", "",
         "القيم المفقودة: فارغة في CSV و`null` في JSONL. التسميات **Silver** (آلية) ما لم يُذكر غير ذلك.", ""]
    for p in sorted(loaded):
        obj = loaded[p]
        if not isinstance(obj, pd.DataFrame) or obj.empty and not len(obj.columns):
            continue
        rel = str(p.relative_to(D))
        if any(rel.endswith(f"{sp}.jsonl") for sp in SPLITS) and (D / rel.replace(".jsonl", ".csv")).exists():
            continue
        if rel.endswith(("validation.csv", "test.csv")):
            continue
        L += [f"## `{rel}` — {len(obj)} صف", "", "| العمود | النوع | الوصف |", "|---|---|---|"]
        for c in obj.columns:
            L.append(f"| `{c}` | {infer_dtype(obj[c])} | {DESCRIPTIONS.get(c, '')} |")
        L.append("")
    L += ["## ner/*.jsonl", "`id, canonical_job_id, text, tokens[], ner_tags[] (BIO), ner_tag_ids[], token_offsets[[start,end]], "
          "spans[{start,end,label,source,text}], label_quality` — الوسوم في `ner/label_list.json`. `*.conll`: رمز<TAB>وسم، سطر فارغ بين الأمثلة.", "",
          "## timeseries", "- `monthly_unique_postings.csv`: إعلانات فريدة شهرياً لكل نوع + `observed` (≥10 وظائف).",
          "- `monthly_skill_counts.csv`: (شهر، مهارة، عدد الإعلانات، الإجمالي، الحصة). `monthly_skill_share_wide.csv`: مصفوفة الحصص (NaN = شهر غير مرصود).",
          "- `lstm_sufficiency.json` + `lstm_sufficiency_by_skill.csv`: شروط LSTM ونتيجتها. `lstm_ready/skill_share_series.csv`: مدخلات LSTM مع عمود split الزمني."]
    (D / "data_dictionary.md").write_text("\n".join(L), encoding="utf-8")


def write_html(D: Path, checks, fr, ss, summary):
    e = html.escape

    def table(df_):
        return "<table><tr>" + "".join(f"<th>{e(str(c))}</th>" for c in df_.columns) + "</tr>" + "".join(
            "<tr>" + "".join(f"<td>{e(str(v))}</td>" for v in r) + "</tr>" for r in df_.itertuples(index=False)) + "</table>"

    def kv(d):
        return table(pd.DataFrame([{"البند": k, "القيمة": v} for k, v in d.items()]))
    ck = pd.DataFrame(checks)
    ck["الحالة"] = np.where(ck.passed, "✔ نجح", np.where(ck.critical, "✖ فشل", "⚠ تنبيه"))
    src = pd.read_csv(D / "dataset_statistics" / "source_statistics.csv")
    issues = []
    lstm = summary.get("lstm", {})
    if not lstm.get("skills_ready"):
        issues.append("لا توجد مهارة تستوفي شروط LSTM في النافذة الحديثة.")
    if not summary.get("gold_annotated_rows"):
        issues.append("عينة Gold غير مصنفة بشرياً بعد — لا أرقام دقة صالحة للنشر قبل تعبئتها.")
    issues += [f"فحص فاشل: {c['check']} — {c['detail']}" for c in checks if not c["passed"]]
    css = ("body{font-family:Tahoma,Arial,sans-serif;direction:rtl;margin:24px;background:#fafafa;color:#222}"
           "table{border-collapse:collapse;margin:8px 0 20px;background:#fff;font-size:13px}td,th{border:1px solid #ddd;padding:5px 9px;text-align:right}"
           "th{background:#eef2f7}h1{font-size:22px}h2{font-size:17px;margin-top:28px;border-bottom:2px solid #dde}"
           ".note{background:#fff7e0;border:1px solid #f0d58a;padding:10px;border-radius:6px}")
    body = [f"<h1>تقرير جودة FINAL_TRAINING_DATASET</h1><p>{e(summary.get('generated_at', ''))}</p>",
            "<p class=note>كل التسميات <b>Silver</b> (قواعد آلية مع السبب والثقة) وليست تصنيفاً بشرياً. "
            "الأرقام الصالحة لدقة النماذج تأتي فقط بعد تعبئة gold_annotation.</p>",
            "<h2>الملخص</h2>", kv({k: v for k, v in summary.items() if not isinstance(v, dict)}),
            "<h2>توزيع أنواع الفرص (فريدة)</h2>", kv(summary.get("type_distribution_unique", {})),
            "<h2>النطاق الجغرافي التفصيلي (فريدة)</h2>", kv(summary.get("scope_detail_unique", {})),
            "<h2>تغطية الحقول (وظائف فريدة)</h2>", kv(summary.get("field_coverage_unique_jobs", {})),
            "<h2>كيانات NER</h2>", kv(summary.get("ner_entity_counts", {})),
            "<h2>المصادر</h2>", table(src), "<h2>أحجام التقسيمات</h2>", table(ss.fillna("")),
            "<h2>كفاية بيانات LSTM</h2>", kv({k: v for k, v in lstm.items() if k != "criteria"}),
            "<h2>الفحوص</h2>", table(ck[["الحالة", "check", "detail"]]),
            "<h2>مشكلات لم تُحل / تحتاج تدخلاً</h2><ul>" + "".join(f"<li>{e(i)}</li>" for i in issues) + "</ul>",
            "<h2>الملفات</h2>", table(fr)]
    (D / "dataset_quality_report.html").write_text(f"<!doctype html><html lang=ar dir=rtl><head><meta charset=utf-8>"
                                                    f"<title>Dataset Quality Report</title><style>{css}</style></head><body>"
                                                    + "".join(body) + "</body></html>", encoding="utf-8")


if __name__ == "__main__":
    main()
