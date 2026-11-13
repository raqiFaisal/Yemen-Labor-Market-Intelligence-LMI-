"""المرحلة 5: بناء Dataset موحد وجاهز للتدريب من منشورات Telegram + صفحات telegra.ph (+ YemenHR كمرجع للتكرار).

    python -m scripts.build_dataset                      # كامل (المهارات: هجين إن توفر Transformer وإلا قاموسي)
    python -m scripts.build_dataset --skills lexicon     # أسرع، بلا Transformer
    python -m scripts.build_dataset --sample DIR --out DIR2   # تجربة على عينة (posts_sample.csv / telegraph_sample.csv)

لا يعدّل أي ملف خام. كل المخرجات في data/processed/dataset/ و labels/ و reports/.
التسميات هنا «Silver» آلية بقواعد معلّلة؛ التقييم الحقيقي يحتاج عينة Gold يصنفها إنسان (الملفات تُصدَّر تلقائياً).

الخطوات:
 1) تحميل المنشورات + إزالة أسطر التوقيع المتكررة لكل قناة (للتحليل فقط؛ raw_text محفوظ كما هو)
 2) دمج صفحة telegra.ph بكل منشور يشير إليها (النص الكامل)
 3) تحليل الحقول + تصنيف نوع الفرصة والنطاق الجغرافي (مع السبب والثقة) + تقسيم المنشورات متعددة الوظائف
 4) استخراج المؤهل/الخبرة/التخصص/نوع التوظيف + المهارات (المستخرج الهجين الموجود)
 5) كشف التكرار عبر المصادر (canonical_job_id) — بلا حذف
 6) بيانات التصنيف (نوع الفرصة، النطاق) + NER (Silver spans + BIO) + السلاسل الزمنية + التقسيم
 7) تقارير الجودة وملفات Gold للتصنيف اليدوي + ملف التدريب للخط الحالي
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import IS_DEMO_MODE, LABELS_DIR, PROCESSED_DIR, RAW_DIR, REPORTS_DIR, ROOT
from preprocessing import job_fields as jf
from preprocessing.dedup import find_duplicates, pick_representative
from preprocessing.opportunity import (channel_boilerplate, classify_scope, classify_type, reconcile_scope, scope_detail,
                                       strip_boilerplate)
from preprocessing.text_normalize import apply_links, clean_text
from preprocessing.telegram_parse import body_text, parse_post, split_positions
from preprocessing.text import strip_diacritics

TG_DIR = RAW_DIR / "telegram"
OUT = PROCESSED_DIR / "dataset"
MAX_POST_CHARS = 15000
RELEVANT = ["JOB", "INTERNSHIP", "TRAINING", "SCHOLARSHIP", "VOLUNTEERING", "OTHER_OPPORTUNITY"]
LABOR = ["JOB", "INTERNSHIP"]
SOURCE_PRIORITY = {"yemenhr_site": 0, "telegram+telegraph": 1, "telegram": 2}
FINAL_COLUMNS = [
    "job_id", "post_id", "canonical_job_id", "is_canonical", "dup_status", "dup_reason", "dup_group_size",
    "source", "source_type", "source_name", "source_channel", "source_message_id", "source_url", "position_index",
    "telegraph_url", "raw_text", "job_title", "organization", "location", "governorate", "country", "work_location_type",
    "description", "skills", "qualifications", "education", "education_min", "experience_min_years", "experience_max_years",
    "specializations", "employment_type", "salary", "published_date", "message_date", "application_deadline",
    "posting_date", "posting_date_source", "language", "collected_at", "opportunity_type", "type_confidence",
    "type_status", "type_reason", "geographic_scope", "scope_confidence", "scope_status", "scope_reason",
    "template", "hashtags", "extraction_status", "verification_status", "split_random", "split_temporal",
    "scope_detail", "scope_detail_reason", "governorates_all", "apply_url", "apply_email", "apply_urls_all",
    "deadline_raw_text", "date_issue", "text_source", "organization_canonical", "text_clean",
]


def log(msg):
    print(f"{datetime.now():%H:%M:%S} | {msg}", flush=True)


# ------------------------------------------------------------------ 1) التحميل
def load_posts(sample: Path | None) -> pd.DataFrame:
    if sample:
        d = pd.read_csv(sample / "posts_sample.csv", dtype=str, keep_default_na=False)
        for c in ("collected_at", "source_name"):
            d[c] = d.get(c, "")
        return d
    src = json.loads((ROOT / "data" / "sources.json").read_text(encoding="utf-8"))
    chans = [c["username"] if isinstance(c, dict) else c for c in src["telegram"]
             if not isinstance(c, dict) or c.get("enabled", True)]
    frames = []
    for ch in chans:
        f = TG_DIR / f"{ch}.csv"
        if not f.exists():
            log(f"⚠ لا يوجد ملف للقناة @{ch} — تُتخطى (لا بيانات بديلة)")
            continue
        frames.append(pd.read_csv(f, dtype=str, keep_default_na=False))
    if not frames:
        raise SystemExit(f"✖ لا توجد منشورات في {TG_DIR}")
    return pd.concat(frames, ignore_index=True)


def load_telegraph(sample: Path | None) -> tuple[pd.DataFrame, dict]:
    from scripts.collectors.telegraph import path_of
    f = (sample / "telegraph_sample.csv") if sample else (TG_DIR / "telegraph" / "pages.csv")
    pages = pd.read_csv(f, dtype=str, keep_default_na=False) if f.exists() else pd.DataFrame(
        columns=["telegraph_path", "telegraph_url", "title", "content_text"])
    return pages.set_index("telegraph_path"), {"path_of": path_of}


def load_yemenhr_site() -> pd.DataFrame:
    f = RAW_DIR / "yemen_yemenhr.csv"
    if not f.exists():
        return pd.DataFrame()
    d = pd.read_csv(f, dtype=str, keep_default_na=False)
    return d


# ------------------------------------------------------------------ 2-4) المعالجة لكل منشور
def yemenhr_key(url: str) -> str | None:
    from scripts.scrapers.yemenhr import job_key
    try:
        return job_key(url) if "yemenhr.com/jobs/" in url else None
    except Exception:  # noqa: BLE001
        return None


OCR: dict = {}
OCR_MIN_CHARS, OCR_MIN_CONF = 25, 55.0


def load_ocr(sample: Path | None) -> dict:
    """نصوص OCR لمنشورات الصور (من scripts.collectors.telegram_media_ocr). تُقبل فقط بطول وثقة كافيين."""
    f = (sample / "ocr_texts.csv") if sample else (TG_DIR / "ocr" / "ocr_texts.csv")
    if not f.exists():
        return {}
    d = pd.read_csv(f, dtype=str, keep_default_na=False)
    d = d[(d.ocr_text.str.len() >= OCR_MIN_CHARS) & (pd.to_numeric(d.ocr_conf, errors="coerce") >= OCR_MIN_CONF)]
    d = d.groupby(["source_channel", "source_message_id"]).ocr_text.apply(lambda x: "\n".join(x)).reset_index()
    return {(r.source_channel, r.source_message_id): r.ocr_text for r in d.itertuples()}


def sane_deadline(deadline: str, message_date: str) -> tuple[str, str]:
    """آخر موعد غير منطقي (قبل الرسالة بأكثر من 7 أيام أو بعدها بأكثر من سنة) → null + سبب. لا تصحيح تخميني."""
    if not deadline:
        return "", ""
    dl = pd.to_datetime(deadline, errors="coerce")
    md = pd.to_datetime(message_date, errors="coerce", utc=True)
    if pd.isna(dl) or pd.isna(md):
        return deadline, ""
    days = (dl - md.tz_localize(None)).days
    if days < -7:
        return "", f"deadline {deadline} is {-days} days before message date — dropped"
    if days > 366:
        return "", f"deadline {deadline} is {days} days after message date — dropped"
    return deadline, ""


def posting_date(published: str, message_date: str) -> tuple[str, str]:
    """تاريخ النشر داخل الإعلان إن كان معقولاً (±60 يوماً من تاريخ الرسالة)، وإلا تاريخ الرسالة. يُسجَّل المصدر."""
    md = pd.to_datetime(message_date, errors="coerce", utc=True)
    md = md.tz_localize(None) if pd.notna(md) else md
    pd_ = pd.to_datetime(published, errors="coerce")
    if pd.notna(pd_) and pd.notna(md) and abs((pd_ - md).days) <= 60:
        return pd_.date().isoformat(), "in_text"
    return (md.date().isoformat(), "message_date") if pd.notna(md) else ("", "none")


def process_posts(posts: pd.DataFrame, pages: pd.DataFrame, path_of) -> tuple[pd.DataFrame, pd.DataFrame]:
    """يعيد (posts_all: صف لكل منشور مع قراره، records: صف لكل فرصة/وظيفة بعد التقسيم)."""
    post_rows, recs = [], []
    for ch, g in posts.groupby("source_channel"):
        bp = channel_boilerplate(g.raw_text, min_share=0.01, min_count=5 if len(g) < 500 else 20)
        log(f"@{ch}: {len(g)} منشوراً | أسطر توقيع متكررة مستبعدة من التحليل: {len(bp)}")
        for r in g.itertuples(index=False):
            t_post = time.time()
            raw = r.raw_text or ""
            text_source = "post_text"
            if not raw.strip() and (ch, str(r.source_message_id)) in OCR:
                raw, text_source = OCR[(ch, str(r.source_message_id))], "ocr"     # منشور صورة: نص مستخرج آلياً من الصورة
            text = strip_boilerplate(raw, bp)[:MAX_POST_CHARS]       # للتحليل؛ raw_text يُحفظ كاملاً
            paths = [p for p in (path_of(u) for u in r.link_urls.split()) if p]
            tp = next((p for p in paths if p in pages.index), None)
            tg_text = pages.at[tp, "content_text"] if tp else ""
            tg_title = pages.at[tp, "title"] if tp else ""
            f = parse_post(text, ch, r.message_date)
            if tg_title and (not f["title"] or f["template"] == "free_text"):
                f["title"] = tg_title
            t = classify_type(text, f, ch, r.link_urls)
            if text_source == "ocr" and t["type_status"] == "accepted":
                t = {**t, "type_status": "review", "type_confidence": min(t["type_confidence"], 0.65),
                     "type_reason": "[OCR] " + t["type_reason"]}           # نص OCR قد يحوي أخطاء قراءة → مراجعة لا قبول آلي
            if not f["title"] and t["opportunity_type"] not in LABOR:
                # دورة/منحة/فرصة: أول سطر ذي معنى كعنوان. للوظائف يبقى فارغاً (لا نجعل «وظائف شاغرة» مسمى وظيفياً)
                f["title"] = next((l for l in body_text(text).splitlines() if len(l) >= 6), "")[:120]
            full = text + ("\n\n" + tg_text if tg_text else "")
            s = classify_scope(full, f)
            pid = f"tg:{ch}:{r.source_message_id}"
            keys = [f"telegraph:{p}" for p in paths] + [f"yemenhr:{k}" for k in
                                                         (yemenhr_key(u) for u in r.link_urls.split()) if k]
            excluded = ""
            if t["type_status"] == "excluded":
                excluded = t["type_reason"]
            elif t["opportunity_type"] == "NOT_RELEVANT" and t["type_status"] == "accepted":
                excluded = "not_relevant: " + t["type_reason"]
            if time.time() - t_post > 3:
                log(f"   ⚠ منشور بطيء التحليل {pid} ({len(raw)} حرفاً، {time.time() - t_post:.1f} ث)")
            post_rows.append({"post_id": pid, "text_source": text_source, "source_channel": ch, "source_message_id": r.source_message_id,
                              "source_url": r.source_url, "message_date": r.message_date, "media_only": r.media_only,
                              "forwarded_from": r.forwarded_from, "has_telegraph": bool(tg_text), "template": f["template"],
                              **t, **{k: s[k] for k in ("geographic_scope", "scope_confidence", "scope_status", "scope_reason")},
                              "excluded_reason": excluded, "n_positions": 0, "raw_text": r.raw_text,
                              "ocr_text": raw if text_source == "ocr" else "", "telegraph_url": f"https://telegra.ph/{tp}" if tp else "",
                              "telegraph_text": tg_text, "post_title": f["title"]})
            if excluded or t["opportunity_type"] == "NOT_RELEVANT":
                continue
            # تقسيم منشور متعدد الوظائف (فقط للوظائف ومن قنوات النص الحر)
            parts = split_positions(text, ch) if t["opportunity_type"] in LABOR and not tg_text else []
            post_rows[-1]["n_positions"] = len(parts) or 1
            units = [(0, f, full, None)]
            if parts:
                units = []
                for p in parts:
                    seg = p["header"] + "\n" + p["text"]
                    fp = parse_post(seg, ch, r.message_date)
                    fp["title"] = p["role_line"]
                    for k in ("organization", "published_text", "published_date"):
                        fp[k] = fp[k] or f[k]
                    units.append((p["n"], fp, seg, p))
            for n, fu, utext, part in units:
                su = classify_scope(utext, fu) if part else s
                sd = scope_detail(su, fu, utext, own_text=part["text"] if part else None)
                su = reconcile_scope(su, sd)
                if not su["governorate"] and sd["governorates_all"] and ";" not in sd["governorates_all"]:
                    su = {**su, "governorate": sd["governorates_all"]}       # محافظة من اسم مديرية معروفة (مثل المخا → تعز)
                ex = jf.extract_all(utext, fu["title"])
                pdate, psrc = posting_date(fu["published_date"], r.message_date)
                dl, dl_issue = sane_deadline(fu["application_deadline"], r.message_date)
                links = apply_links(utext + ("\n" + tg_text if tg_text else ""), r.link_urls if not part else "")
                recs.append({
                    "job_id": pid + (f"#p{n}" if part else ""), "post_id": pid, "position_index": n,
                    "source": f"telegram:{ch}", "source_type": "telegram", "source_kind": "telegram+telegraph" if tg_text else "telegram",
                    "source_name": getattr(r, "source_name", "") or ch, "source_channel": ch,
                    "source_message_id": r.source_message_id, "source_url": r.source_url,
                    "telegraph_url": f"https://telegra.ph/{tp}" if tp else "", "raw_text": raw if not part else part["text"],
                    "job_title": fu["title"], "organization": fu["organization"], "location": fu["location_text"],
                    "governorate": su["governorate"], "country": su["country"], "work_location_type": su["work_location_type"],
                    "description": (tg_text or body_text(utext))[:20000], "salary": fu["salary"],
                    "published_date": fu["published_date"], "message_date": r.message_date,
                    "application_deadline": dl, "deadline_raw_text": fu["deadline_text"], "date_issue": dl_issue,
                    "posting_date": pdate, "posting_date_source": psrc, "text_source": text_source,
                    **sd, **links,
                    "collected_at": r.collected_at, "template": fu["template"], "hashtags": fu["hashtags"],
                    **{k: v for k, v in t.items()}, **{k: su[k] for k in ("geographic_scope", "scope_confidence",
                                                                         "scope_status", "scope_reason")},
                    **ex, "link_keys": keys if not part else [],
                    "model_text": (fu["title"] + ".\n" + body_text(utext) + ("\n" + tg_text if tg_text else ""))[:6000],
                    "ner_text": (body_text(utext) + ("\n" + tg_text if tg_text else ""))[:6000],   # النص الأصلي بلا إضافات
                    "text_clean": clean_text((fu["title"] + "\n" if fu["title"] else "") + body_text(utext)
                                             + ("\n" + tg_text if tg_text else ""))[:8000],
                })
    return pd.DataFrame(post_rows), pd.DataFrame(recs)


def site_records(site: pd.DataFrame) -> pd.DataFrame:
    """إعلانات موقع YemenHR (مجمعة سابقاً) كمرجع للتكرار فقط — لا تدخل ملفات Telegram الناتجة."""
    if site.empty:
        return pd.DataFrame()
    from preprocessing.telegram_parse import match_governorate

    def col(*names):
        for n in names:
            if n in site:
                return site[n].fillna("").astype(str)
        return pd.Series("", index=site.index)
    key = col("source_job_id")
    return pd.DataFrame({
        "job_id": "site:yemenhr:" + key.where(key != "", site.index.astype(str)),
        "source_kind": "yemenhr_site", "job_title": col("title"), "organization": col("company"),
        "governorate": col("location_raw", "location").map(lambda x: match_governorate(x) or ""),
        "posting_date": col("posted_date"), "model_text": col("title") + "\n" + col("description"),
        "link_keys": key.map(lambda k: [f"yemenhr:{k.lower()}"] if k else []),
    })


# ------------------------------------------------------------------ 6) مساعدات المخرجات
def hash_split(key: str, train=70, val=15) -> str:
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    return "train" if h < train else "val" if h < train + val else "test"


def temporal_split(df: pd.DataFrame) -> pd.Series:
    """حسب أقدم تاريخ في كل مجموعة تكرار: أقدم 70% تدريب، 15% تحقق، أحدث 15% اختبار (لا تسرّب زمني ولا تكراري)."""
    first = pd.to_datetime(df.posting_date, errors="coerce").groupby(df.canonical_job_id).transform("min")
    q1, q2 = first.quantile(0.70), first.quantile(0.85)
    return np.where(first <= q1, "train", np.where(first <= q2, "val", "test"))


TOKEN_RX = re.compile(r"[^\s.,:;!?؟،؛()\[\]{}\"«»|]+|[.,:;!?؟،؛()\[\]{}\"«»|]")


def ner_example(r, skills_by_job: dict, terms: dict) -> dict | None:
    """Silver NER: تُحدَّد الكيانات بالبحث الحرفي عن القيم المستخرجة داخل النص (لا مواضع مخمّنة)."""
    text = strip_diacritics(r.ner_text)[:2500]
    low = text.lower().translate(str.maketrans("أإآىة", "ااايه"))
    spans = []

    def add(val, label, src):
        if not val or len(val) < 2:
            return
        v = strip_diacritics(str(val)).lower().translate(str.maketrans("أإآىة", "ااايه"))
        i = low.find(v)
        if i >= 0 and not any(s["start"] < i + len(v) and i < s["end"] for s in spans):
            spans.append({"start": i, "end": i + len(v), "label": label, "source": src})

    add(r.job_title, "JOB_TITLE", "parsed_field")
    add(r.organization, "ORG", "parsed_field")
    add(r.location, "LOCATION", "parsed_field")
    for rx in [x for _, _, x in jf.EDU_RX]:
        m = rx.search(text)
        if m and r.education:
            add(m.group(0), "EDUCATION", "rule")
            break
    if r.experience_evidence:
        for rx in jf.EXP_RX:
            m = rx.search(text)
            if m:
                add(m.group(0).strip(), "EXPERIENCE", "rule")
                break
    for sid in skills_by_job.get(r.job_id, []):
        for t in terms.get(sid, []):
            before = len(spans)
            add(t, "SKILL", "skill_extractor")
            if len(spans) > before:
                break
    if not spans:
        return None
    spans.sort(key=lambda s: s["start"])
    toks, tags, pos = [], [], 0
    for m in TOKEN_RX.finditer(text):
        lab = next((s["label"] for s in spans if s["start"] <= m.start() < s["end"]), None)
        first = lab and not any(s["start"] <= m.start() - 1 < s["end"] and s["label"] == lab and s["start"] < m.start()
                                for s in spans if s["start"] < m.start())
        toks.append(m.group(0))
        tags.append("O" if not lab else ("B-" if first else "I-") + lab)
    return {"id": r.job_id, "text": text, "spans": spans, "tokens": toks, "ner_tags": tags, "label_quality": "silver"}


def skill_terms() -> dict:
    from models.skill_extractor import load_taxonomy
    skills, _ = load_taxonomy()
    return {s["id"]: sorted({x for x in [s.get("name_en"), s.get("name_ar"), *s.get("aliases", [])] if x}, key=len, reverse=True)
            for s in skills}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills", choices=["auto", "hybrid", "lexicon", "none"], default="auto")
    ap.add_argument("--sample", type=Path, help="مجلد عينة (posts_sample.csv, telegraph_sample.csv) — للتجربة")
    ap.add_argument("--out", type=Path, help="مجلد المخرجات (افتراضياً data/processed/dataset)")
    a = ap.parse_args()
    if IS_DEMO_MODE:
        raise SystemExit("بناء الـ Dataset في الوضع الحقيقي فقط.")
    out = a.out or OUT
    labels_dir = (out / "labels") if a.sample else LABELS_DIR
    reports_dir = (out / "reports") if a.sample else REPORTS_DIR
    for d in (out, out / "classification", out / "ner", out / "timeseries", labels_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log("1) تحميل المنشورات وصفحات telegra.ph")
    posts = load_posts(a.sample)
    pages, helpers = load_telegraph(a.sample)
    log(f"   منشورات: {len(posts)} | صفحات telegra.ph: {len(pages)}")

    OCR.update(load_ocr(a.sample))
    log(f"   نصوص OCR مقبولة لمنشورات صور: {len(OCR)}")
    log("2-4) التحليل والتصنيف والاستخراج")
    posts_all, recs = process_posts(posts, pages, helpers["path_of"])
    log(f"   سجلات فرص (بعد تقسيم المنشورات متعددة الوظائف): {len(recs)}")
    if recs.empty:
        posts_all.to_csv(out / "posts_all.csv", index=False, encoding="utf-8-sig")
        log("✖ لا توجد سجلات فرص بعد التصنيف — لا ملفات تدريب. راجع posts_all.csv (أسباب الاستبعاد).")
        raise SystemExit(1)

    log("4b) المهارات")
    skills_df = pd.DataFrame(columns=["job_id", "skill_id", "method", "decision", "confidence", "evidence", "reason"])
    skill_mode = "none"
    if a.skills != "none" and len(recs):
        from models.skill_extractor import SkillExtractor
        enc = None
        if a.skills in ("auto", "hybrid"):
            try:
                from models.encoder import TextEncoder
                enc = TextEncoder()
                enc = enc if enc.kind == "transformer" else None
            except Exception as e:  # noqa: BLE001
                log(f"   تعذر تحميل Transformer ({e}) — قاموسي فقط")
            if a.skills == "hybrid" and enc is None:
                raise SystemExit("✖ --skills hybrid يتطلب Transformer")
        ex = SkillExtractor(enc)
        skill_mode = "hybrid (lexicon + frozen transformer)" if ex.semantic_available else "lexicon (rule-based)"
        target = recs[recs.opportunity_type.isin(RELEVANT)]
        skills_df, _ = ex.extract_all(list(target.job_id), list(target.model_text), semantic=ex.semantic_available)
        log(f"   الطريقة: {skill_mode} | قرارات: {len(skills_df)} | مقبولة: {(skills_df.decision == 'accepted').sum()}")
    acc = skills_df[skills_df.decision == "accepted"]
    by_job = acc.groupby("job_id").skill_id.apply(list).to_dict()
    recs["skills"] = recs.job_id.map(lambda j: ";".join(by_job.get(j, [])))

    log("5) كشف التكرار")
    site = site_records(load_yemenhr_site())
    dd = pd.concat([recs[["job_id", "source_kind", "job_title", "organization", "governorate", "posting_date", "model_text",
                          "link_keys"]], site], ignore_index=True)
    dd = dd.rename(columns={"job_title": "title", "model_text": "text"})
    dd["date"] = pd.to_datetime(dd.posting_date, errors="coerce")
    dd["governorate"] = dd.governorate.fillna("")
    assign, pairs = find_duplicates(dd)
    dd = dd.merge(assign, on="job_id")
    dd["is_canonical"] = pick_representative(dd.rename(columns={"title": "job_title"}), SOURCE_PRIORITY)
    recs = recs.merge(dd[["job_id", "canonical_job_id", "dup_status", "dup_reason", "dup_group_size", "is_canonical"]],
                      on="job_id", how="left")
    site_groups = set(dd.loc[dd.source_kind == "yemenhr_site", "canonical_job_id"])
    recs["dup_of_yemenhr_site"] = recs.canonical_job_id.isin(site_groups)
    log(f"   مجموعات فريدة: {recs.canonical_job_id.nunique()} من {len(recs)} سجل | مكرر: {(recs.dup_status == 'duplicate').sum()} | "
        f"مراجعة: {(recs.dup_status == 'possible_duplicate_review').sum()} | مطابق لموقع YemenHR: {recs.dup_of_yemenhr_site.sum()}")

    log("6) التقسيم والحقول النهائية")
    # توحيد أسماء الجهات: نفس الاسم بعد التطبيع (حالة/همزات/كلمات مثل منظمة-مؤسسة) → أكثر كتابة شيوعاً. الأصل يبقى في organization
    from preprocessing.dedup import NOISE_ORG, norm_org
    key = recs.organization.fillna("").map(norm_org)
    common = recs[key != ""].assign(k=key).groupby("k").organization.agg(lambda x: x.value_counts().index[0])
    recs["organization_canonical"] = key.map(lambda k: "" if k in NOISE_ORG else common.get(k, ""))
    recs["split_random"] = recs.canonical_job_id.map(hash_split)
    recs["split_temporal"] = temporal_split(recs) if len(recs) else []
    recs["extraction_status"] = np.where(recs.job_title.str.len() > 0,
                                         np.where(recs.description.str.len() > 80, "complete", "title_only"), "no_title")
    recs["verification_status"] = np.where((recs.type_status == "accepted") & (recs.scope_status == "accepted"),
                                           "auto_accepted_silver", "needs_review")
    recs["experience_min_years"] = pd.to_numeric(recs.experience_min_years, errors="coerce")
    recs["experience_max_years"] = pd.to_numeric(recs.experience_max_years, errors="coerce")
    final = recs.reindex(columns=FINAL_COLUMNS + ["experience_evidence", "education_evidence", "dup_of_yemenhr_site", "model_text",
                                                  "ner_text"])
    final.drop(columns=["model_text", "ner_text"]).to_csv(out / "opportunities.csv", index=False, encoding="utf-8-sig")
    final.to_pickle(out / "opportunities.pkl")
    posts_all.drop(columns=["telegraph_text"]).to_csv(out / "posts_all.csv", index=False, encoding="utf-8-sig")
    posts_all.to_pickle(out / "posts_all.pkl")
    pairs.to_csv(out / "duplicate_pairs.csv", index=False, encoding="utf-8-sig")
    skills_df.to_csv(out / "skill_decisions.csv", index=False, encoding="utf-8-sig")

    # --- بيانات التصنيف (Silver): المنشورات كلها (بما فيها NOT_RELEVANT) — نص المنشور + تفاصيل telegra.ph
    cls = posts_all.merge(recs.drop_duplicates("post_id")[["post_id", "model_text", "canonical_job_id"]], on="post_id", how="left")
    raw_text = posts.assign(post_id="tg:" + posts.source_channel + ":" + posts.source_message_id).set_index("post_id").raw_text
    cls["text"] = cls.model_text.fillna(cls.post_id.map(raw_text).map(body_text)).str[:3000]
    cls["group"] = cls.canonical_job_id.fillna(cls.post_id)
    cls["split"] = cls.group.map(hash_split)
    for task, lab, conf, st in [("opportunity_type", "opportunity_type", "type_confidence", "type_status"),
                                ("geographic_scope", "geographic_scope", "scope_confidence", "scope_status")]:
        sub = cls[(cls[st] == "accepted") & (cls[conf] >= 0.8) & (cls.text.str.len() >= 20)]
        if task == "geographic_scope":
            sub = sub[sub.opportunity_type.isin(RELEVANT)]
        sub = sub.drop_duplicates("group")          # نسخة واحدة لكل إعلان مكرر (منع تسرّب التكرار بين الأقسام)
        for sp in ("train", "val", "test"):
            sub[sub.split == sp][["post_id", "text", lab, conf, "source_channel"]].rename(columns={lab: "label", conf: "confidence"}) \
                .assign(label_source="silver_rules").to_csv(out / "classification" / f"{task}_{sp}.csv", index=False,
                                                            encoding="utf-8-sig")
    # --- NER (Silver)
    terms = skill_terms()
    ner = [e for e in (ner_example(r, by_job, terms) for r in final[final.is_canonical & final.opportunity_type.isin(LABOR)]
                       .itertuples(index=False)) if e]
    split_of = dict(zip(final.job_id, final.split_random))
    for sp in ("train", "val", "test"):
        with open(out / "ner" / f"silver_{sp}.jsonl", "w", encoding="utf-8") as fh:
            for e in ner:
                if split_of.get(e["id"]) == sp:
                    fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    labels = ["JOB_TITLE", "ORG", "LOCATION", "EDUCATION", "EXPERIENCE", "SKILL"]
    (out / "ner" / "label_scheme.json").write_text(json.dumps({
        "format": "spans (char offsets) + tokens/ner_tags (BIO, whitespace tokens)", "labels": labels,
        "bio_tags": ["O"] + [f"{p}-{l}" for l in labels for p in "BI"],
        "quality": "silver — spans located by exact string match of rule/extractor outputs; NOT human-annotated",
        "gold_file": str(labels_dir / "ner_gold_to_annotate.jsonl")}, ensure_ascii=False, indent=1), encoding="utf-8")

    # --- السلاسل الزمنية (إعلانات فريدة فقط)
    canon = final[final.is_canonical].copy()
    canon["month"] = pd.to_datetime(canon.posting_date, errors="coerce").dt.to_period("M").astype(str)
    canon = canon[canon.month != "NaT"]
    ts = out / "timeseries"
    pd.crosstab(canon.month, canon.opportunity_type).to_csv(ts / "monthly_by_type.csv")
    lab_c = canon[canon.opportunity_type.isin(LABOR)]
    pd.crosstab(lab_c.month, lab_c.geographic_scope).to_csv(ts / "monthly_jobs_by_scope.csv")
    pd.crosstab(lab_c.month, lab_c.source_channel).to_csv(ts / "monthly_jobs_by_source.csv")
    ym = lab_c[lab_c.geographic_scope == "YEMEN"]
    pd.crosstab(ym.month, ym.governorate.replace("", "غير محدد")).to_csv(ts / "monthly_yemen_jobs_by_governorate.csv")
    sk = lab_c[["job_id", "month"]].merge(acc[["job_id", "skill_id"]], on="job_id")
    tot = lab_c.groupby("month").size().rename("job_postings")
    skm = sk.groupby(["month", "skill_id"]).size().rename("postings_with_skill").reset_index().merge(tot, on="month")
    skm["share"] = (skm.postings_with_skill / skm.job_postings).round(4)
    skm["observed"] = skm.job_postings >= 10
    skm.to_csv(ts / "monthly_skill_demand.csv", index=False)
    tot.to_frame().assign(observed=lambda x: x.job_postings >= 10).to_csv(ts / "monthly_totals.csv")

    # --- ملف التدريب للخط الحالي (وظائف فريدة، غير مكررة مع إعلانات موقع YemenHR المجمعة مسبقاً)
    train_src = canon[canon.opportunity_type.isin(LABOR) & ~canon.dup_of_yemenhr_site & (canon.type_status == "accepted")]
    pipe = pd.DataFrame({
        "source_job_id": train_src.job_id, "title": train_src.job_title, "description": train_src.description,
        "description_raw": train_src.description, "requirements": train_src.qualifications, "company": train_src.organization,
        "location": train_src.governorate.where(train_src.governorate != "", train_src.location),
        "location_raw": train_src.location,
        "country": np.where(train_src.geographic_scope == "REMOTE_GLOBAL", "Remote", train_src.country.replace("", "Unknown")),
        "posted_date": train_src.posting_date, "deadline": train_src.application_deadline, "source": train_src.source,
        # الوظائف المقسّمة من منشور واحد تتشارك رابطه؛ unify يزيل التكرار بالرابط، لذا يُترك فارغاً لها (الرابط في source_job_id)
        "url": train_src.source_url.where(train_src.position_index.fillna(0).astype(int) == 0, ""), "experience_years": train_src.experience_min_years, "education": train_src.education_min,
        "employment_type": train_src.employment_type, "category_raw": train_src.hashtags.str.replace(" ", ";"),
        "scraped_at": train_src.collected_at})
    pipe.to_csv(out / "telegram_jobs_for_training.csv", index=False, encoding="utf-8-sig")

    # --- 7) ملفات Gold للتصنيف اليدوي (عينة طبقية ثابتة)
    gold_f = labels_dir / "opportunity_gold_to_annotate.csv"
    if not gold_f.exists():
        pool = cls[cls.text.str.len() >= 20].copy()
        pool["stratum"] = pool.source_channel + "|" + pool.opportunity_type.fillna("")
        gs = pool.sample(frac=1, random_state=7).groupby("stratum").head(12)       # حتى 12 من كل (قناة × نوع متوقع)
        gs = gs.sample(min(len(gs), 300), random_state=7)
        gs[["post_id", "source_channel", "source_url", "text"]].assign(
            gold_opportunity_type="", gold_geographic_scope="", annotator="", notes="") \
            .to_csv(gold_f, index=False, encoding="utf-8-sig")
    ner_gold = labels_dir / "ner_gold_to_annotate.jsonl"
    if not ner_gold.exists() and ner:
        rng = np.random.default_rng(7)
        pick = rng.choice(len(ner), size=min(100, len(ner)), replace=False)
        with open(ner_gold, "w", encoding="utf-8") as fh:
            for i in pick:
                e = ner[i]
                fh.write(json.dumps({"id": e["id"], "text": e["text"], "silver_spans": e["spans"], "gold_spans": None},
                                    ensure_ascii=False) + "\n")
    review = pd.concat([
        posts_all[posts_all.type_status == "review"].assign(review_kind="opportunity_type")[["post_id", "review_kind", "opportunity_type", "type_reason"]]
        .rename(columns={"opportunity_type": "predicted", "type_reason": "reason"}),
        recs[recs.scope_status == "review"].assign(review_kind="geographic_scope")[["job_id", "review_kind", "geographic_scope", "scope_reason"]]
        .rename(columns={"job_id": "post_id", "geographic_scope": "predicted", "scope_reason": "reason"}),
        recs[recs.dup_status == "possible_duplicate_review"].assign(review_kind="possible_duplicate", predicted="", reason="same title+location, org unknown")
        [["job_id", "review_kind", "predicted", "reason"]].rename(columns={"job_id": "post_id"}),
    ], ignore_index=True)
    review.to_csv(out / "review_queue.csv", index=False, encoding="utf-8-sig")

    write_report(posts_all, recs, final, acc, pairs, ner, skill_mode, out, reports_dir, labels_dir, len(pages), time.time() - t0,
                 bool(a.sample), len(site))
    log(f"✔ انتهى في {time.time() - t0:.0f} ثانية → {out}")


def write_report(posts_all, recs, final, acc, pairs, ner, skill_mode, out, reports_dir, labels_dir, n_pages, secs, is_sample, n_site):
    pa = posts_all
    per_src = []
    for ch, g in pa.groupby("source_channel"):
        r = recs[recs.source_channel == ch]
        per_src.append({
            "source": f"@{ch}", "posts_scanned": len(g), "media_only_no_text": int((g.type_status == "excluded").sum()),
            "not_relevant": int(((g.opportunity_type == "NOT_RELEVANT") & (g.type_status == "accepted")).sum()),
            "relevant_posts": int(g.opportunity_type.isin(RELEVANT).sum()),
            "type_needs_review": int((g.type_status == "review").sum()),
            "records_after_split": len(r), "jobs_internships": int(r.opportunity_type.isin(LABOR).sum()),
            "duplicates": int((r.dup_status == "duplicate").sum()), "unique_ads": int(r[r.is_canonical == True].shape[0]),  # noqa: E712
            "with_telegraph_details": int(g.has_telegraph.sum()),
            "first_post": str(pd.to_datetime(g.message_date, errors="coerce").min())[:10],
            "last_post": str(pd.to_datetime(g.message_date, errors="coerce").max())[:10]})
    lab = final[final.opportunity_type.isin(LABOR)]
    labc = lab[lab.is_canonical == True]  # noqa: E712
    cov = {c: round(float((labc[c].astype(str).str.len() > 0).mean()), 3) if len(labc) else 0 for c in
           ["job_title", "organization", "location", "governorate", "education", "specializations", "employment_type",
            "qualifications", "salary", "application_deadline", "skills"]}
    cov["experience_min_years"] = round(float(labc.experience_min_years.notna().mean()), 3) if len(labc) else 0
    months = pd.to_datetime(labc.posting_date, errors="coerce").dt.to_period("M").value_counts()
    rep = {
        "generated_at": datetime.now().isoformat(timespec="seconds"), "is_sample_run": is_sample, "runtime_sec": round(secs),
        "label_quality": "SILVER (rule-based, with reasons). Not human-verified. Evaluate with labels/opportunity_gold_to_annotate.csv",
        "skill_extraction_mode": skill_mode, "telegraph_pages_available": n_pages, "yemenhr_site_records_for_dedup": n_site,
        "posts_total": len(pa), "per_source": per_src,
        "opportunity_type_distribution_posts": pa.opportunity_type.replace("", "(no text)").value_counts().to_dict(),
        "type_status": pa.type_status.value_counts().to_dict(),
        "records_total": len(recs), "unique_groups": int(recs.canonical_job_id.nunique()) if len(recs) else 0,
        "jobs_internships_records": len(lab), "jobs_internships_unique": len(labc),
        "geographic_scope_unique_jobs": labc.geographic_scope.value_counts().to_dict(),
        "work_location_type_unique_jobs": labc.work_location_type.value_counts().to_dict(),
        "duplicate_evidence": pairs.evidence.str.split(":").str[0].value_counts().to_dict() if len(pairs) else {},
        "dup_with_yemenhr_site": int(recs.get("dup_of_yemenhr_site", pd.Series(dtype=bool)).sum()),
        "field_coverage_unique_jobs": cov,
        "skills_accepted_pairs": len(acc), "unique_jobs_with_skills": int(labc.skills.str.len().gt(0).sum()),
        "top_skills_unique_jobs": acc[acc.job_id.isin(set(labc.job_id))].skill_id.value_counts().head(20).to_dict(),
        "months_with_10plus_unique_jobs": int((months >= 10).sum()), "months_span": len(months),
        # آخر 24 شهراً تقويمياً كاملاً أو جارياً: (الشهر الحالي − 23) … الشهر الحالي
        "last_24_months_with_10plus": int((months[months.index > (pd.Timestamp.now().to_period("M") - 24)] >= 10).sum()),
        "ner_silver_examples": len(ner),
        "classification_files": {p.name: len(pd.read_csv(p)) for p in sorted((out / "classification").glob("*.csv"))},
        "split_random_unique_jobs": labc.split_random.value_counts().to_dict(),
        "posting_date_source": lab.posting_date_source.value_counts().to_dict(),
        "training_export_rows": len(pd.read_csv(out / "telegram_jobs_for_training.csv")),
    }
    (reports_dir / "dataset_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    ex = []
    for t in RELEVANT:
        s = final[(final.opportunity_type == t) & (final.is_canonical == True)].head(2)  # noqa: E712
        for r in s.itertuples():
            ex.append(f"- **{t}** [{r.source_channel}]({r.source_url}) — {str(r.job_title)[:70]} | {r.organization} | {r.governorate or r.location} "
                      f"| {r.geographic_scope} | تعليم: {r.education_min or '—'} | خبرة: {'' if pd.isna(r.experience_min_years) else int(r.experience_min_years)} "
                      f"| مهارات: {r.skills[:80] or '—'}  \n  _سبب التصنيف: {str(r.type_reason)[:100]}_")
    L = [f"# تقرير جودة الـ Dataset — {rep['generated_at']}" + (" (تشغيل على عينة)" if is_sample else ""), "",
         f"> التسميات **Silver** (قواعد معلّلة). لم يُتحقق منها بشرياً بعد. المهارات: {skill_mode}.", "",
         "## لكل مصدر", "| المصدر | فُحص | بلا نص | غير ذي صلة | ذو صلة | مراجعة | سجلات | وظائف/تدريب عملي | مكرر | فريد | مع telegra.ph | من | إلى |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in per_src:
        L.append("| " + " | ".join(str(s[k]) for k in ["source", "posts_scanned", "media_only_no_text", "not_relevant", "relevant_posts",
                                                         "type_needs_review", "records_after_split", "jobs_internships", "duplicates",
                                                         "unique_ads", "with_telegraph_details", "first_post", "last_post"]) + " |")
    L += ["", f"**المجموع:** {rep['posts_total']} منشوراً → {rep['records_total']} سجل فرصة → {rep['unique_groups']} مجموعة فريدة | "
              f"وظائف/تدريب عملي فريدة: **{rep['jobs_internships_unique']}**", "",
          "## التوزيعات", f"- نوع الفرصة (منشورات): {rep['opportunity_type_distribution_posts']}",
          f"- حالة التصنيف: {rep['type_status']}", f"- النطاق (وظائف فريدة): {rep['geographic_scope_unique_jobs']}",
          f"- مكان العمل: {rep['work_location_type_unique_jobs']}", f"- أدلة التكرار: {rep['duplicate_evidence']} | مطابق لموقع YemenHR: {rep['dup_with_yemenhr_site']}",
          "", "## تغطية الحقول (وظائف فريدة)", str(cov), "",
          f"## المهارات\n- أزواج مقبولة: {rep['skills_accepted_pairs']} | وظائف فريدة بمهارة واحدة على الأقل: {rep['unique_jobs_with_skills']}",
          f"- الأكثر طلباً: {rep['top_skills_unique_jobs']}", "",
          f"## الزمن\n- أشهر بها ≥10 وظائف فريدة: {rep['months_with_10plus_unique_jobs']} من {rep['months_span']} | في آخر 24 شهراً: {rep['last_24_months_with_10plus']}",
          f"- مصدر تاريخ النشر: {rep['posting_date_source']}", "",
          f"## ملفات التدريب\n- التصنيف: {rep['classification_files']}\n- NER (Silver): {rep['ner_silver_examples']} مثالاً\n"
          f"- التقسيم العشوائي المجمّع (وظائف فريدة): {rep['split_random_unique_jobs']}\n- ملف خط التدريب الحالي: {rep['training_export_rows']} وظيفة",
          "", "## أمثلة حقيقية مُعالجة", *ex, "",
          "## للتحقق البشري", f"- `{labels_dir / 'opportunity_gold_to_annotate.csv'}` — صنّف عموديْ gold_* ثم شغّل `python -m scripts.evaluate_dataset`",
          f"- `{labels_dir / 'ner_gold_to_annotate.jsonl'}` — صحّح الكيانات", f"- `{out / 'review_queue.csv'}` — حالات غير مؤكدة"]
    (reports_dir / "dataset_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
