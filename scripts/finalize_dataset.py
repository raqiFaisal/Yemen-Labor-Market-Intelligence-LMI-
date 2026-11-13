"""تجميع FINAL_TRAINING_DATASET من مخرجات build_dataset — ملفات مستقلة لكل مهمة تدريب بتقسيم مجمّع حسب الإعلان الأصلي.

    python -m scripts.finalize_dataset            # → $LMI_FINAL_DIR أو <ROOT>/FINAL_TRAINING_DATASET
    python -m scripts.finalize_dataset --src DIR --out DIR

مبادئ:
  • لا حذف ولا تعديل للبيانات الخام. القيم المفقودة = null (فارغ في CSV، null في JSONL/Parquet).
  • التقسيم على مستوى مجموعة الإعلان (canonical_job_id) + دمج النصوص المتطابقة بعد التطبيع في نفس المجموعة،
    وتوزيع طبقي لكل فئة (70/15/15 بترتيب تجزئة ثابت) — لا يظهر إعلان أو نسخة منه في قسمين.
  • كل التسميات Silver (قواعد آلية). عينة Gold تُجهَّز بأعمدة فارغة ولا تُعامل كمصنفة.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import (LABELS_DIR, LSTM_MIN_COVERAGE, LSTM_MIN_MENTIONS, LSTM_MIN_MONTHS, LSTM_MIN_POSTINGS_PER_PERIOD,
                         LSTM_TEST_MONTHS, LSTM_VAL_MONTHS, PROCESSED_DIR, ROOT)
from preprocessing import job_fields as jf
from preprocessing.dedup import norm_text_hash
from preprocessing.opportunity import SCOPE_DETAIL_AR, SCOPE_DETAILS
from preprocessing.telegram_parse import body_text
from preprocessing.text import strip_diacritics
from preprocessing.text_normalize import clean_text

OPP_TYPES = ["JOB", "INTERNSHIP", "TRAINING", "SCHOLARSHIP", "VOLUNTEERING", "OTHER_OPPORTUNITY"]
LABOR = ["JOB", "INTERNSHIP"]
NER_LABELS = ["JOB_TITLE", "ORGANIZATION", "SKILL", "DEGREE", "MAJOR", "EXPERIENCE", "LOCATION", "SALARY", "DEADLINE"]
BIO = ["O"] + [f"{p}-{l}" for l in NER_LABELS for p in ("B", "I")]
SPLITS = ("train", "validation", "test")
TOKEN_RX = re.compile(r"[^\s.,:;!?؟،؛()\[\]{}\"«»|]+|[.,:;!?؟،؛()\[\]{}\"«»|]")
MIN_TYPE_CONF, MIN_SCOPE_CONF = 0.80, 0.70


def log(m):
    print(f"{datetime.now():%H:%M:%S} | {m}", flush=True)


def h(x: str) -> int:
    return int(hashlib.md5(str(x).encode()).hexdigest(), 16)


def stratified_group_split(df: pd.DataFrame, label: str, group: str, ratios=(0.70, 0.15, 0.15)) -> pd.Series:
    """كل مجموعة في قسم واحد؛ داخل كل فئة تُرتَّب المجموعات بتجزئة ثابتة وتُوزَّع بالنسب. الفئات الصغيرة جداً (<7 مجموعات)
    تُوضع كلها في train مع تسجيل ذلك (لا يمكن تمثيلها إحصائياً في val/test)."""
    out = pd.Series(index=df.index, dtype=object)
    gl = df.groupby(group)[label].agg(lambda x: x.mode().iloc[0])
    for lab, groups in gl.groupby(gl):
        gs = sorted(groups.index, key=h)
        n = len(gs)
        if n < 7:
            assign = {g: "train" for g in gs}
        else:
            a, b = int(round(n * ratios[0])), int(round(n * (ratios[0] + ratios[1])))
            assign = {g: ("train" if i < a else "validation" if i < b else "test") for i, g in enumerate(gs)}
        m = df[group].isin(groups.index)
        out[m] = df.loc[m, group].map(assign)
    return out


def merge_groups_by_text(df: pd.DataFrame, group_col: str, text_col: str) -> pd.Series:
    """مجموعة نهائية = اتحاد (مجموعة التكرار) و(النص المطابق بعد التطبيع) — يمنع تسرّب نسخ حرفية لم تُربط بالتكرار."""
    parent = {}

    def f(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    hashes = df[text_col].map(norm_text_hash)
    for g, hh in zip(df[group_col], hashes):
        if hh:
            a, b = f(g), f("t:" + hh)
            if a != b:
                parent[max(a, b)] = min(a, b)
    return df[group_col].map(f)


def write_split_files(d: pd.DataFrame, folder: Path, cols: list[str], label: str | None = None, labels: list | None = None):
    folder.mkdir(parents=True, exist_ok=True)
    for sp in SPLITS:
        part = d[d.split == sp][cols]
        part.to_csv(folder / f"{sp}.csv", index=False, encoding="utf-8-sig")
        part.to_json(folder / f"{sp}.jsonl", orient="records", lines=True, force_ascii=False)
    if label:
        lab2id = {l: i for i, l in enumerate(labels)}
        (folder / "label_map.json").write_text(json.dumps({"label2id": lab2id, "id2label": {i: l for l, i in lab2id.items()}},
                                                          ensure_ascii=False, indent=1), encoding="utf-8")
        dist = pd.crosstab(d[label], d.split).reindex(index=labels, columns=list(SPLITS), fill_value=0)
        dist.to_csv(folder / "class_distribution.csv", encoding="utf-8-sig")
        tr = dist["train"]
        w = (tr.sum() / (len(tr[tr > 0]) * tr.replace(0, np.nan))).round(4)
        (folder / "class_weights.json").write_text(json.dumps({k: (None if pd.isna(v) else float(v)) for k, v in w.items()},
                                                              ensure_ascii=False, indent=1), encoding="utf-8")


def nullify(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace({"": None}).where(pd.notna(df), None)


# ------------------------------------------------------------------ NER (Silver)
def locate(text_low: str, value: str, spans: list, label: str, src: str):
    if not value or len(str(value).strip()) < 2:
        return
    v = strip_diacritics(str(value).strip()).lower().translate(str.maketrans("أإآىة", "ااايه"))
    i = text_low.find(v)
    if i >= 0 and not any(s["start"] < i + len(v) and i < s["end"] for s in spans):
        spans.append({"start": i, "end": i + len(v), "label": label, "source": src})


def ner_record(r, skill_terms: dict, job_skills: dict) -> dict | None:
    text = strip_diacritics(r.ner_text or "")[:2500]
    low = text.lower().translate(str.maketrans("أإآىة", "ااايه"))
    spans: list = []
    locate(low, r.job_title, spans, "JOB_TITLE", "parsed_field")
    locate(low, r.organization, spans, "ORGANIZATION", "parsed_field")
    locate(low, (r.location or "").split("\n")[0], spans, "LOCATION", "parsed_field")
    locate(low, r.deadline_raw_text, spans, "DEADLINE", "parsed_field")
    if r.salary_amount_min is not None and not pd.isna(r.salary_amount_min):   # «بعد المقابلة» ليست كياناً للراتب
        locate(low, r.salary, spans, "SALARY", "parsed_field")
    if r.education:
        for _, _, rx in jf.EDU_RX:
            m = rx.search(text)
            if m:
                locate(low, m.group(0), spans, "DEGREE", "rule")
                break
    for major in (r.specializations or "").split(";"):
        locate(low, major, spans, "MAJOR", "rule")
    if pd.notna(r.experience_min_years):
        for rx in jf.EXP_RX:
            m = rx.search(text)
            if m:
                locate(low, m.group(0).strip(), spans, "EXPERIENCE", "rule")
                break
    for sid in job_skills.get(r.job_id, []):
        for t in skill_terms.get(sid, []):
            n = len(spans)
            locate(low, t, spans, "SKILL", "skill_extractor")
            if len(spans) > n:
                break
    if not spans:
        return None
    spans.sort(key=lambda s: s["start"])
    tokens, tags, offs = [], [], []
    for m in TOKEN_RX.finditer(text):
        sp = next((s for s in spans if s["start"] <= m.start() < s["end"]), None)
        if sp is None:
            tags.append("O")
        else:
            prev_in_same = bool(offs) and sp["start"] <= offs[-1][0] < sp["end"]
            tags.append(("I-" if prev_in_same else "B-") + sp["label"])
        tokens.append(m.group(0))
        offs.append((m.start(), m.end()))
    lab2id = {l: i for i, l in enumerate(BIO)}
    return {"id": r.job_id, "canonical_job_id": r.canonical_job_id, "text": text, "tokens": tokens, "ner_tags": tags,
            "ner_tag_ids": [lab2id[t] for t in tags], "token_offsets": offs,
            "spans": [{**s, "text": text[s["start"]:s["end"]]} for s in spans], "label_quality": "silver"}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=PROCESSED_DIR / "dataset")
    ap.add_argument("--labels", type=Path, default=LABELS_DIR)
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("LMI_FINAL_DIR", ROOT / "FINAL_TRAINING_DATASET")))
    ap.add_argument("--overwrite-with-backup", action="store_true", help="Explicitly permit rebuilding an existing dataset; backs it up first")
    a = ap.parse_args()
    src, out = a.src, a.out
    required_inputs = [src / "opportunities.pkl", src / "posts_all.pkl", src / "duplicate_pairs.csv",
                       src / "skill_decisions.csv"]
    missing_inputs = [str(p) for p in required_inputs if not p.is_file()]
    if missing_inputs:
        raise SystemExit(f"✖ Source files missing; refusing to touch existing output: {missing_inputs}")
    if out.exists() and any(out.iterdir()):
        if not a.overwrite_with_backup:
            raise SystemExit(f"✖ Refusing to overwrite {out}. Pass --overwrite-with-backup to save a snapshot first.")
        archive = shutil.make_archive(str(out) + "_before_rebuild", "zip", root_dir=str(out.parent), base_dir=out.name)
        print(f"Existing output backed up to {archive}")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    log(f"المصدر: {src} → {out}")
    opp = pd.read_pickle(src / "opportunities.pkl")
    opp = opp.join(pd.DataFrame([jf.salary_parts(x) for x in opp.salary.fillna("")], index=opp.index))
    posts = pd.read_pickle(src / "posts_all.pkl")
    pairs = pd.read_csv(src / "duplicate_pairs.csv", dtype=str, keep_default_na=False)
    skd = pd.read_csv(src / "skill_decisions.csv", dtype=str, keep_default_na=False)
    from models.skill_extractor import load_taxonomy
    skills_tax, by_id = load_taxonomy()

    # ---------- 1) الملفات الرئيسية
    log("1) all_posts / all_opportunities / unique_opportunities / duplicate_pairs")
    post_cols = ["post_id", "source_channel", "source_message_id", "source_url", "message_date", "raw_text", "text_source",
                 "ocr_text", "telegraph_url", "media_only", "forwarded_from", "template", "opportunity_type", "type_confidence",
                 "type_status", "type_reason", "geographic_scope", "scope_confidence", "scope_status", "excluded_reason",
                 "n_positions"]
    nullify(posts.reindex(columns=post_cols)).to_csv(out / "all_posts.csv", index=False, encoding="utf-8-sig")
    drop = ["model_text", "ner_text", "link_keys"]
    oppx = opp.drop(columns=[c for c in drop if c in opp], errors="ignore")
    nullify(oppx).to_csv(out / "all_opportunities.csv", index=False, encoding="utf-8-sig")
    grp = opp.groupby("canonical_job_id").agg(n_records=("job_id", "size"), sources=("source_channel", lambda x: ";".join(sorted(set(x)))),
                                              first_seen=("posting_date", "min"), last_seen=("posting_date", "max"))
    uni = opp[opp.is_canonical & opp.opportunity_type.isin(OPP_TYPES)].merge(grp, left_on="canonical_job_id", right_index=True)
    uni = uni.drop(columns=[c for c in drop if c in uni], errors="ignore")
    nullify(uni).to_csv(out / "unique_opportunities.csv", index=False, encoding="utf-8-sig")
    t = opp.set_index("job_id")
    pr = pairs.copy()
    for side in ("a", "b"):
        ids = pr[f"job_id_{side}"]
        ok = ids.isin(t.index)
        for c in ("job_title", "organization", "governorate", "posting_date", "source_url"):
            pr[f"{c}_{side}"] = None
            pr.loc[ok, f"{c}_{side}"] = t.loc[ids[ok], c].values
    nullify(pr).to_csv(out / "duplicate_pairs.csv", index=False, encoding="utf-8-sig")

    # ---------- 2) تصنيف نوع الفرصة (6 فئات) + مرشح الصلة
    log("2) classification_opportunity_type")
    first_rec = opp.sort_values("position_index").drop_duplicates("post_id").set_index("post_id")
    P = posts.copy()
    P["text"] = [clean_text((body_text(r.raw_text or r.ocr_text or "") + ("\n" + r.telegraph_text if r.telegraph_text else ""))[:4000])
                 for r in P.itertuples()]
    P["group0"] = P.post_id.map(first_rec.canonical_job_id).fillna("p:" + P.post_id)
    P["group"] = merge_groups_by_text(P, "group0", "text")
    base = P[(P.type_status == "accepted") & (P.type_confidence.astype(float) >= MIN_TYPE_CONF) & (P.text_source != "ocr")
             & (P.text.str.len() >= 20)]
    # تسميات متناقضة: نفس النص المطبّع بتسميتين مختلفتين → يُستبعد ويُسجَّل للمراجعة
    base = base.assign(th=base.text.map(norm_text_hash))
    contra_h = base[base.th != ""].groupby("th").opportunity_type.nunique()
    contra_h = set(contra_h[contra_h > 1].index)
    contradictions = base[base.th.isin(contra_h)]
    base = base[~base.th.isin(contra_h)]
    six = base[base.opportunity_type.isin(OPP_TYPES)].drop_duplicates("group").copy()
    six["label"], six["label_id"] = six.opportunity_type, six.opportunity_type.map({l: i for i, l in enumerate(OPP_TYPES)})
    six["split"] = stratified_group_split(six, "label", "group")
    six = six.rename(columns={"post_id": "id", "type_confidence": "confidence"}).assign(label_source="silver_rules")
    cls_cols = ["id", "text", "label", "label_id", "confidence", "label_source", "source_channel", "group"]
    write_split_files(six, out / "classification_opportunity_type", cls_cols, "label", OPP_TYPES)
    # نسخة تدريب متوازنة (اختيارية): الفئة الغالبة تُقلَّص إلى 4× ثاني أكبر فئة بعينة ثابتة. validation/test تبقى بتوزيعها الحقيقي
    trn = six[six.split == "train"]
    vc = trn.label.value_counts()
    if len(vc) > 1:
        cap = int(4 * vc.iloc[1])
        bal = pd.concat([g if len(g) <= cap else g.sample(cap, random_state=13) for _, g in trn.groupby("label")])
        bal[cls_cols].to_csv(out / "classification_opportunity_type" / "train_balanced.csv", index=False, encoding="utf-8-sig")
    rel = base.drop_duplicates("group").copy()
    rel["label"] = np.where(rel.opportunity_type == "NOT_RELEVANT", "NOT_RELEVANT", "RELEVANT")
    rel["label_id"] = (rel.label == "RELEVANT").astype(int)
    rel["split"] = stratified_group_split(rel, "label", "group")
    rel = rel.rename(columns={"post_id": "id", "type_confidence": "confidence"}).assign(label_source="silver_rules")
    write_split_files(rel, out / "classification_opportunity_type" / "relevance_filter", cls_cols, "label",
                      ["NOT_RELEVANT", "RELEVANT"])

    # ---------- 3) النطاق الجغرافي التفصيلي
    log("3) classification_geographic_scope")
    sc = opp[opp.is_canonical & opp.opportunity_type.isin(OPP_TYPES) & (opp.scope_status == "accepted")
             & (opp.scope_confidence.astype(float) >= MIN_SCOPE_CONF) & (opp.text_source != "ocr")].copy()
    sc["text"] = sc.text_clean.fillna("").str[:4000]
    sc = sc[sc.text.str.len() >= 20]
    sc["group"] = merge_groups_by_text(sc, "canonical_job_id", "text")
    sc = sc.drop_duplicates("group")
    sc["label"], sc["label_id"] = sc.scope_detail, sc.scope_detail.map({l: i for i, l in enumerate(SCOPE_DETAILS)})
    sc["split"] = stratified_group_split(sc, "label", "group")
    sc = sc.rename(columns={"job_id": "id", "scope_confidence": "confidence"}).assign(label_source="silver_rules")
    write_split_files(sc, out / "classification_geographic_scope",
                      ["id", "text", "label", "label_id", "confidence", "label_source", "geographic_scope", "governorate",
                       "governorates_all", "source_channel", "group"], "label", SCOPE_DETAILS)

    # ---------- 4) NER
    log("4) ner (BIO, 9 labels)")
    terms = {s["id"]: sorted({x for x in [s.get("name_en"), s.get("name_ar"), *s.get("aliases", [])] if x}, key=len, reverse=True)
             for s in skills_tax}
    acc = skd[skd.decision == "accepted"]
    job_sk = acc.groupby("job_id").skill_id.apply(list).to_dict()
    nr = opp[opp.is_canonical & opp.opportunity_type.isin(LABOR) & (opp.text_source != "ocr")]
    ner = [e for e in (ner_record(r, terms, job_sk) for r in nr.itertuples(index=False)) if e]
    nd = pd.DataFrame({"i": range(len(ner)), "group0": [e["canonical_job_id"] for e in ner], "text": [e["text"] for e in ner],
                       "lab": [max((s["label"] for s in e["spans"] if s["label"] != "JOB_TITLE"), default="JOB_TITLE") for e in ner]})
    # إعادة نشر نفس النص بعد أشهر = إعلان مستقل زمنياً، لكنه نفس النص للتدريب → نفس المجموعة (منع التسرب)
    nd["group"] = merge_groups_by_text(nd, "group0", "text")
    nd = nd.drop_duplicates("group")
    nd["split"] = stratified_group_split(nd, "lab", "group")
    (out / "ner").mkdir()
    conll = {sp: [] for sp in SPLITS}
    for sp in SPLITS:
        with open(out / "ner" / f"{sp}.jsonl", "w", encoding="utf-8") as fh:
            for i in nd[nd.split == sp].i:
                e = ner[i]
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
                conll[sp].append("\n".join(f"{t}\t{g}" for t, g in zip(e["tokens"], e["ner_tags"])))
        (out / "ner" / f"{sp}.conll").write_text("\n\n".join(conll[sp]) + "\n", encoding="utf-8")
    (out / "ner" / "label_list.json").write_text(json.dumps({
        "entity_labels": NER_LABELS, "bio_tags": BIO, "label2id": {l: i for i, l in enumerate(BIO)},
        "tokenization": "word-level (regex: non-space runs, punctuation split). For subword models align labels to first subtoken.",
        "quality": "SILVER: spans found by exact string match of rule/extractor outputs inside the original text. "
                   "Entities the rules missed are tagged O (recall-limited). Not human-annotated."},
        ensure_ascii=False, indent=1), encoding="utf-8")
    ner_label_counts = pd.Series([s["label"] for e in ner for s in e["spans"]]).value_counts().reindex(NER_LABELS, fill_value=0)

    # ---------- 5) المهارات
    log("5) skills")
    skd2 = skd.merge(opp[["job_id", "canonical_job_id", "is_canonical", "opportunity_type", "posting_date", "governorate",
                          "source_url"]], on="job_id", how="left")
    skd2["name_ar"] = skd2.skill_id.map(lambda s: by_id.get(s, {}).get("name_ar"))
    skd2["name_en"] = skd2.skill_id.map(lambda s: by_id.get(s, {}).get("name_en"))
    skd2["skill_type"] = skd2.skill_id.map(lambda s: by_id.get(s, {}).get("type"))
    (out / "skills").mkdir()
    sk_cols = ["job_id", "canonical_job_id", "is_canonical", "opportunity_type", "skill_id", "name_ar", "name_en", "skill_type",
               "method", "decision", "confidence", "evidence", "reason", "source_url", "posting_date", "governorate"]
    nullify(skd2[skd2.decision == "accepted"][sk_cols]).to_csv(out / "skills" / "job_skills.csv", index=False, encoding="utf-8-sig")
    lab_c = skd2[(skd2.decision == "accepted") & (skd2.is_canonical == True) & skd2.opportunity_type.isin(LABOR)]  # noqa: E712
    cat = lab_c.groupby("skill_id").agg(unique_jobs=("canonical_job_id", "nunique"),
                                        first_month=("posting_date", lambda x: str(x.min())[:7]),
                                        last_month=("posting_date", lambda x: str(x.max())[:7]))
    cat = pd.DataFrame([{"skill_id": s["id"], "name_ar": s["name_ar"], "name_en": s["name_en"], "skill_type": s["type"]}
                        for s in skills_tax]).merge(cat, left_on="skill_id", right_index=True, how="left")
    cat["unique_jobs"] = cat.unique_jobs.fillna(0).astype(int)
    cat.sort_values("unique_jobs", ascending=False).to_csv(out / "skills" / "skill_catalog.csv", index=False, encoding="utf-8-sig")
    sx = opp[opp.is_canonical & opp.opportunity_type.isin(LABOR) & (opp.text_source != "ocr")].copy()
    sx["text"] = sx.text_clean.fillna("").str[:4000]
    sx["skills"] = sx.job_id.map(lambda j: sorted(set(job_sk.get(j, []))))
    sx["n_skills"] = sx.skills.map(len)
    sx = sx[sx.text.str.len() >= 30]
    sx["group"] = merge_groups_by_text(sx, "canonical_job_id", "text")
    sx = sx.drop_duplicates("group")
    sx["has_skill"] = np.where(sx.n_skills > 0, "with_skills", "no_skills")
    sx["split"] = stratified_group_split(sx, "has_skill", "group")
    sx = sx.rename(columns={"job_id": "id"}).assign(label_source="silver_extractor")
    write_split_files(sx, out / "skills" / "extraction", ["id", "text", "skills", "n_skills", "label_source", "group"])
    (out / "skills" / "extraction" / "skill_vocab.json").write_text(json.dumps(sorted(by_id), ensure_ascii=False), encoding="utf-8")

    # ---------- 6) السلاسل الزمنية
    log("6) timeseries")
    ts = out / "timeseries"
    ts.mkdir()
    U = opp[opp.is_canonical & opp.opportunity_type.isin(OPP_TYPES)].copy()
    U["month"] = pd.to_datetime(U.posting_date, errors="coerce").dt.to_period("M")
    U = U[U.month.notna()]
    rng = pd.period_range(U.month.min(), U.month.max(), freq="M") if len(U) else pd.PeriodIndex([], freq="M")
    by_type = pd.crosstab(U.month, U.opportunity_type).reindex(rng, fill_value=0)
    jobs_m = U[U.opportunity_type.isin(LABOR)].groupby("month").size().reindex(rng, fill_value=0)
    mon = by_type.assign(unique_jobs_internships=jobs_m, unique_opportunities=by_type.sum(axis=1),
                         observed=jobs_m >= LSTM_MIN_POSTINGS_PER_PERIOD)
    mon.index = mon.index.astype(str)
    mon.index.name = "month"
    mon.to_csv(ts / "monthly_unique_postings.csv", encoding="utf-8-sig")
    L = U[U.opportunity_type.isin(LABOR)]
    sk = L[["job_id", "month"]].merge(acc[["job_id", "skill_id"]], on="job_id").drop_duplicates()
    cnt = sk.groupby(["month", "skill_id"]).size().unstack(fill_value=0).reindex(rng, fill_value=0)
    share = cnt.div(jobs_m.replace(0, np.nan), axis=0).where(jobs_m >= LSTM_MIN_POSTINGS_PER_PERIOD)   # فترة غير مرصودة = NaN
    long = cnt.stack().rename("postings_with_skill").reset_index().rename(columns={"level_0": "month"})
    long.columns = ["month", "skill_id", "postings_with_skill"]
    long = long.merge(jobs_m.rename("job_postings"), left_on="month", right_index=True)
    long["share"] = (long.postings_with_skill / long.job_postings.replace(0, np.nan)).round(5)
    long["observed"] = long.job_postings >= LSTM_MIN_POSTINGS_PER_PERIOD
    long["month"] = long.month.astype(str)
    long.to_csv(ts / "monthly_skill_counts.csv", index=False, encoding="utf-8-sig")
    sh = share.copy()
    sh.index = sh.index.astype(str)
    sh.round(5).to_csv(ts / "monthly_skill_share_wide.csv", encoding="utf-8-sig")
    gv = pd.crosstab(L.month, L.governorate.replace("", "غير محدد")).reindex(rng, fill_value=0)
    gv.index = gv.index.astype(str)
    gv.to_csv(ts / "monthly_jobs_by_governorate.csv", encoding="utf-8-sig")
    spz = L.assign(sp=L.specializations.fillna("").str.split(";")).explode("sp").reset_index(drop=True)
    spz = spz[spz.sp.str.len() > 0]
    sm = pd.crosstab(spz.month, spz.sp).reindex(rng, fill_value=0)
    sm.index = sm.index.astype(str)
    sm.to_csv(ts / "monthly_jobs_by_specialization.csv", encoding="utf-8-sig")
    src_m = pd.crosstab(L.month, L.source_channel).reindex(rng, fill_value=0)
    src_m.index = src_m.index.astype(str)
    src_m.to_csv(ts / "monthly_jobs_by_source.csv", encoding="utf-8-sig")
    # كفاية بيانات LSTM: أطول نافذة حديثة متصلة تقريباً (تغطية ≥ LSTM_MIN_COVERAGE) تنتهي بآخر شهر مرصود
    obs = (jobs_m >= LSTM_MIN_POSTINGS_PER_PERIOD).values
    win_start = None
    if obs.any():
        last = int(np.where(obs)[0].max())
        best = None
        for s in range(last, -1, -1):
            seg = obs[s:last + 1]
            if seg.mean() >= LSTM_MIN_COVERAGE and obs[s]:
                best = s
        win_start = best
    win = rng[win_start:int(np.where(obs)[0].max()) + 1] if win_start is not None else rng[:0]
    suff = {"criteria": {"min_months": LSTM_MIN_MONTHS, "min_coverage": LSTM_MIN_COVERAGE, "min_mentions": LSTM_MIN_MENTIONS,
                         "min_postings_per_month": LSTM_MIN_POSTINGS_PER_PERIOD, "val_months": LSTM_VAL_MONTHS,
                         "test_months": LSTM_TEST_MONTHS},
            "months_total": len(rng), "months_observed": int(obs.sum()),
            "window": [str(win[0]), str(win[-1])] if len(win) else None, "window_months": len(win),
            "window_coverage": round(float((jobs_m.reindex(win) >= LSTM_MIN_POSTINGS_PER_PERIOD).mean()), 3) if len(win) else 0}
    ready, rows = [], []
    for sid in cnt.columns:
        sw = share[sid].reindex(win)
        mentions = int(cnt[sid].reindex(win).sum())
        ok = len(win) >= LSTM_MIN_MONTHS and sw.notna().mean() >= LSTM_MIN_COVERAGE and mentions >= LSTM_MIN_MENTIONS
        rows.append({"skill_id": sid, "window_months": len(win), "observed_months": int(sw.notna().sum()),
                     "coverage": round(float(sw.notna().mean()), 3) if len(win) else 0, "mentions": mentions, "lstm_ready": bool(ok)})
        if ok:
            ready.append(sid)
    pd.DataFrame(rows).sort_values("mentions", ascending=False).to_csv(ts / "lstm_sufficiency_by_skill.csv", index=False,
                                                                       encoding="utf-8-sig")
    suff["skills_ready"] = len(ready)
    suff["skills_total"] = len(cnt.columns)
    lr = ts / "lstm_ready"
    lr.mkdir()
    if ready and len(win) >= LSTM_MIN_MONTHS:
        mat = share[ready].reindex(win)
        mat.index = mat.index.astype(str)
        mat.index.name = "month"
        n = len(mat)
        split = ["train"] * (n - LSTM_VAL_MONTHS - LSTM_TEST_MONTHS) + ["validation"] * LSTM_VAL_MONTHS + ["test"] * LSTM_TEST_MONTHS
        mat.round(5).assign(split=split, job_postings=jobs_m.reindex(win).values).to_csv(lr / "skill_share_series.csv",
                                                                                     encoding="utf-8-sig")
        suff["lstm_split"] = {"train": [mat.index[0], mat.index[n - LSTM_VAL_MONTHS - LSTM_TEST_MONTHS - 1]],
                              "validation": [mat.index[n - LSTM_VAL_MONTHS - LSTM_TEST_MONTHS], mat.index[n - LSTM_TEST_MONTHS - 1]],
                              "test": [mat.index[n - LSTM_TEST_MONTHS], mat.index[-1]]}
        suff["note"] = "share = postings mentioning skill / unique job postings that month; NaN = month with < min postings (not zero demand)"
    else:
        suff["lstm_split"] = None
        suff["note"] = "insufficient: no skill meets the LSTM criteria in the recent window"
    (ts / "lstm_sufficiency.json").write_text(json.dumps(suff, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------- 7) Gold (فارغ) + المراجعة
    log("7) gold_annotation / review_queue")
    gd = out / "gold_annotation"
    gd.mkdir()
    gsrc = a.labels / "opportunity_gold_to_annotate.csv"
    if gsrc.exists():
        g = pd.read_csv(gsrc, dtype=str, keep_default_na=False)
        for c in ("gold_opportunity_type", "gold_geographic_scope", "gold_scope_detail", "annotator", "notes"):
            if c not in g:
                g[c] = ""
        g.to_csv(gd / "opportunity_gold_to_annotate.csv", index=False, encoding="utf-8-sig")
        n_gold = int((g.gold_opportunity_type.str.strip() != "").sum())
    else:
        n_gold = None
    if (a.labels / "ner_gold_to_annotate.jsonl").exists():
        shutil.copy(a.labels / "ner_gold_to_annotate.jsonl", gd / "ner_gold_to_annotate.jsonl")
    (gd / "ANNOTATION_GUIDE.md").write_text(annotation_guide(n_gold), encoding="utf-8")
    rq = out / "review_queue"
    rq.mkdir()
    pv = ["post_id", "source_channel", "source_url", "opportunity_type", "type_confidence", "type_reason", "text_source"]
    nullify(posts[posts.type_status == "review"][pv]).to_csv(rq / "opportunity_type_review.csv", index=False, encoding="utf-8-sig")
    nullify(posts[posts.text_source == "ocr"][pv + ["ocr_text"]]).to_csv(rq / "ocr_based_posts.csv", index=False, encoding="utf-8-sig")
    sv = ["job_id", "source_url", "location", "geographic_scope", "scope_confidence", "scope_reason", "scope_detail", "scope_detail_reason"]
    nullify(opp[(opp.scope_status == "review") | opp.scope_detail_reason.fillna("").str.startswith("unlisted sub-area")][sv]) \
        .to_csv(rq / "geographic_scope_review.csv", index=False, encoding="utf-8-sig")
    nullify(pr[pr.evidence == "possible"]).to_csv(rq / "possible_duplicates.csv", index=False, encoding="utf-8-sig")
    nullify(skd2[skd2.decision == "review"][sk_cols]).to_csv(rq / "skill_review.csv", index=False, encoding="utf-8-sig")
    contradictions[["post_id", "source_channel", "opportunity_type", "type_reason", "text"]].to_csv(
        rq / "contradictory_labels.csv", index=False, encoding="utf-8-sig")
    issues = []
    for r in opp[opp.date_issue.fillna("") != ""].itertuples():
        issues.append({"record_id": r.job_id, "issue": "date", "detail": r.date_issue, "action": "deadline set to null"})
    for r in opp[opp.extraction_status != "complete"].itertuples():
        issues.append({"record_id": r.job_id, "issue": "incomplete_record", "detail": r.extraction_status,
                       "action": "kept in all_opportunities; excluded from text-classification if text < 20 chars"})
    pd.DataFrame(issues, columns=["record_id", "issue", "detail", "action"]).to_csv(rq / "data_quality_issues.csv", index=False,
                                                                                   encoding="utf-8-sig")
    # أسماء الجهات: متغيرات كتابة لنفس الجهة بعد التطبيع
    v = opp[opp.organization.fillna("") != ""].groupby("organization_canonical").organization.agg(lambda x: sorted(set(x)))
    v = v[v.map(len) > 1]
    pd.DataFrame({"organization_canonical": v.index, "n_variants": v.map(len).values, "variants": v.map(" | ".join).values}) \
        .sort_values("n_variants", ascending=False).to_csv(rq / "organization_name_variants.csv", index=False, encoding="utf-8-sig")

    # ---------- 8) إحصاءات أولية (الفحوص والتقرير والـ manifest في validate_final_dataset)
    st = out / "dataset_statistics"
    st.mkdir()
    lab = opp[opp.is_canonical & opp.opportunity_type.isin(LABOR)]
    cov = {c: round(float(lab[c].notna().mean() if c in ("experience_min_years", "salary_amount_min") else (lab[c].fillna("").astype(str).str.len() > 0).mean()), 3)
           for c in ["job_title", "organization", "skills", "qualifications", "education", "experience_min_years", "specializations",
                     "location", "governorate", "employment_type", "salary", "salary_amount_min", "application_deadline",
                     "apply_url", "apply_email"]}
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "posts_total": len(posts), "posts_by_text_source": posts.text_source.value_counts().to_dict(),
        "posts_excluded": int((posts.excluded_reason.fillna("") != "").sum()),
        "opportunity_records": len(opp), "unique_groups": int(opp.canonical_job_id.nunique()),
        "unique_opportunities_6types": len(uni), "duplicate_records": int((opp.dup_status == "duplicate").sum() - opp[opp.dup_status == "duplicate"].canonical_job_id.nunique()),
        "possible_duplicate_review": int((opp.dup_status == "possible_duplicate_review").sum()),
        "type_distribution_unique": uni.opportunity_type.value_counts().to_dict(),
        "type_distribution_posts": posts.opportunity_type.replace("", "(no text)").value_counts().to_dict(),
        "scope_detail_unique": uni.scope_detail.value_counts().to_dict(),
        "field_coverage_unique_jobs": cov, "unique_jobs_internships": len(lab),
        "skills_accepted_pairs": int((skd.decision == "accepted").sum()), "skills_distinct": int(acc.skill_id.nunique()),
        "skills_review_pairs": int((skd.decision == "review").sum()),
        "ner_examples": len(ner), "ner_entity_counts": ner_label_counts.to_dict(),
        "timeseries_months": len(rng), "timeseries_skill_month_rows": len(long), "lstm": suff,
        "gold_annotated_rows": n_gold, "contradictory_label_posts": len(contradictions),
    }
    (st / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    pd.DataFrame([{"field": k, "coverage": v} for k, v in cov.items()]).to_csv(st / "field_coverage.csv", index=False)
    src_stats = posts.groupby("source_channel").agg(posts=("post_id", "size"),
                                                    excluded=("excluded_reason", lambda x: int((x.fillna("") != "").sum())),
                                                    ocr_text=("text_source", lambda x: int((x == "ocr").sum()))) \
        .join(opp.groupby("source_channel").agg(records=("job_id", "size"), unique=("is_canonical", "sum"),
                                                duplicates=("dup_status", lambda x: int((x == "duplicate").sum()))))
    src_stats.to_csv(st / "source_statistics.csv", encoding="utf-8-sig")
    log(f"✔ الملفات في {out}")


def annotation_guide(n_gold) -> str:
    rows = "\n".join(f"| {k} | {v} |" for k, v in SCOPE_DETAIL_AR.items())
    return f"""# دليل التصنيف البشري (Gold)

**الحالة:** {'لم يُصنَّف أي صف بعد' if not n_gold else f'{n_gold} صفاً مصنفاً'} — الملف لا يُعتبر Gold قبل التعبئة الفعلية.

صنّف من **النص وحده**؛ لا تنظر إلى توقعات النظام (غير مضمّنة في الملف عمداً).

## gold_opportunity_type
JOB (وظيفة مدفوعة) · INTERNSHIP (تدريب عملي/متدرب/Intern) · TRAINING (دورة/دبلوم/ورشة لتعليم مهارة) · SCHOLARSHIP (منحة دراسية)
· VOLUNTEERING (تطوع) · OTHER_OPPORTUNITY (مسابقة/جائزة/زمالة/دعوة تقديم برنامج) · NOT_RELEVANT (إعلان تجاري، مناقصة، دعاء، ترويج قناة…)

## gold_scope_detail
| القيمة | المعنى |
|---|---|
{rows}

`gold_geographic_scope` (اختياري، للتوافق): YEMEN / REMOTE_GLOBAL / OUTSIDE_YEMEN / UNKNOWN، أو NA إن لم تكن فرصة.

## NER (ner_gold_to_annotate.jsonl)
املأ `gold_spans` بقائمة `{{"start","end","label"}}` على حقل `text` (مواضع حروف)، بالتسميات:
JOB_TITLE · ORGANIZATION · SKILL · DEGREE · MAJOR · EXPERIENCE · LOCATION · SALARY · DEADLINE — أو [] إن لم توجد كيانات.
`silver_spans` اقتراح آلي للمساعدة فقط.

بعد التعبئة: `python -m scripts.evaluate_dataset`
"""


if __name__ == "__main__":
    main()
