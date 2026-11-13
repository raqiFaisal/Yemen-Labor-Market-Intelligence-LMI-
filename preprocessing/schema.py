"""طبقة توحيد البيانات: تحويل أي ملف وظائف (بأعمدة مختلفة) إلى مخطط موحد واحد.

المخطط الموحد (UNIFIED_COLUMNS) هو ما تستقبله كل النماذج والصفحات.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

UNIFIED_COLUMNS = [
    "job_id", "title", "description", "requirements", "company", "location",
    "country", "domain", "posted_date", "source", "url", "experience_years",
    "education", "is_demo",
    # حقول إضافية (اختيارية): النص الأصلي دون أي تعديل، ومعرف المصدر، وبيانات الجمع
    "description_raw", "source_job_id", "category_raw", "location_raw", "deadline", "employment_type",
    "experience_level", "scraped_at",
]

# مرادفات أسماء الأعمدة الشائعة في المصادر المختلفة (تُطابق بعد تحويلها لحروف صغيرة)
COLUMN_ALIASES = {
    "title": ["title", "job_title", "jobtitle", "position", "job", "المسمى", "المسمى الوظيفي", "الوظيفة", "عنوان الوظيفة"],
    "description": ["description", "job_description", "desc", "details", "summary", "text", "وصف", "الوصف", "وصف الوظيفة"],
    "requirements": ["requirements", "requirements_text", "qualifications", "job_requirements", "skills", "required_skills", "المتطلبات", "المؤهلات", "المهارات"],
    "company": ["company", "employer", "organization", "organisation", "company_name", "الشركة", "الجهة", "المنظمة", "جهة العمل"],
    "location": ["location", "city", "governorate", "region", "place", "المحافظة", "المدينة", "الموقع", "مكان العمل"],
    "country": ["country", "nation", "الدولة", "البلد"],
    # تصنيف المصدر الخام — يُحوَّل إلى أحد المجالات الثمانية عبر data/taxonomy/category_map.json
    "category_raw": ["category_raw", "category", "categories", "domain", "sector", "field", "industry", "job_category", "label",
                     "المجال", "القطاع", "التصنيف"],
    "description_raw": ["description_raw", "original_text"],
    "source_job_id": ["source_job_id", "job_key", "external_id"],
    "location_raw": ["location_raw"],
    "deadline": ["deadline", "closing_date", "آخر موعد"],
    "employment_type": ["employment_type", "job_type", "contract_type", "نوع العقد", "نوع الدوام"],
    "experience_level": ["experience_level", "seniority"],
    "scraped_at": ["scraped_at", "collected_at"],
    "posted_date": ["posted_date", "date", "post_date", "published", "date_posted", "created_at", "publish_date", "تاريخ النشر", "التاريخ"],
    "source": ["source", "site", "المصدر"],
    "url": ["url", "link", "href", "الرابط"],
    "experience_years": ["experience_years", "experience", "years_experience", "الخبرة", "سنوات الخبرة"],
    "education": ["education", "degree", "education_level", "المؤهل", "المؤهل العلمي"],
}

YEMEN_GOVERNORATES = {
    "Sana'a": ["صنعاء", "sanaa", "sana'a", "sana’a", "amanat al asimah", "أمانة العاصمة", "امانة العاصمة", "sana a", "sanaa city", "صنعاء الجديدة"],
    "Aden": ["عدن", "aden"],
    "Taiz": ["تعز", "taiz", "ta'iz", "ta'izz", "taizz"],
    "Hodeidah": ["الحديدة", "hodeidah", "al hudaydah", "hudaydah", "الحديده", "hodiedah", "hudeidah", "al-hudaydah", "al hodeidah", "hodeida", "الحُديدة"],
    "Ibb": ["إب", "اب", "ibb", "اب"],
    "Hadramawt": ["حضرموت", "المكلا", "سيئون", "hadramout", "hadramawt", "mukalla", "seiyun", "hadhramout", "hadramaut", "hadhramaut", "al mukalla", "tarim", "تريم", "الشحر", "شبام"],
    "Marib": ["مأرب", "مارب", "marib", "ma'rib", "mareb", "ma'rib"],
    "Dhamar": ["ذمار", "dhamar"],
    "Hajjah": ["حجة", "hajjah"],
    "Lahj": ["لحج", "lahj", "lahij", "lahej"],
    "Abyan": ["أبين", "ابين", "abyan"],
    "Al Mahrah": ["المهرة", "mahrah", "al mahra", "al-mahrah", "almahra"],
    "Shabwah": ["شبوة", "shabwah", "shabwa", "shabwa", "ataq", "عتق"],
    "Al Bayda": ["البيضاء", "bayda", "al bayda"],
    "Saada": ["صعدة", "saada", "sa'dah", "sa'ada", "sadah"],
    "Amran": ["عمران", "amran"],
    "Al Dhale'e": ["الضالع", "dhale", "al dhale", "al-dhala'e", "al dhalea", "al-dhale", "dhala", "aldhale", "al-dhalea", "ad dali", "الضالع"],
    "Raymah": ["ريمة", "raymah"],
    "Al Mahwit": ["المحويت", "mahwit"],
    "Al Jawf": ["الجوف", "jawf"],
    "Socotra": ["سقطرى", "socotra"],
}

_AR_DIACRITICS = re.compile(r"[ؗ-ًؚ-ْـ]")


def normalize_arabic(text: str) -> str:
    """توحيد الكتابة العربية: إزالة التشكيل والتطويل وتوحيد الألف والياء والتاء المربوطة."""
    if not isinstance(text, str):
        return ""
    t = _AR_DIACRITICS.sub("", text)
    t = re.sub("[إأآا]", "ا", t)
    t = t.replace("ى", "ي").replace("ة", "ه")
    return t


def clean_text(text) -> str:
    if not isinstance(text, str):
        return ""
    t = re.sub(r"<[^>]+>", " ", text)          # إزالة HTML
    t = re.sub(r"http\S+|www\.\S+", " ", t)     # روابط
    t = re.sub(r"\S+@\S+", " ", t)              # بريد
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def normalize_governorate(loc) -> str | None:
    if not isinstance(loc, str) or not loc.strip():
        return None
    low = loc.lower()
    norm = normalize_arabic(low)
    tokens = set(re.split(r"[\s,،/\-()]+", norm))
    for canon, aliases in YEMEN_GOVERNORATES.items():
        for a in aliases:
            na = normalize_arabic(a.lower())
            # الأسماء القصيرة (مثل "اب") تُطابق ككلمة كاملة فقط لتجنب المطابقات الخاطئة
            if (na in tokens) if len(na) <= 3 else (na in norm):
                return aliases[0]  # الاسم العربي المعتمد للعرض
    return loc.strip()


def _parse_experience(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"(\d+(?:\.\d+)?)", str(v))
    return float(m.group(1)) if m else np.nan


def map_columns(df: pd.DataFrame) -> dict:
    """يعيد قاموس {اسم العمود الأصلي: اسم العمود الموحد} — يستخدم في صفحة رفع الملفات لعرض المطابقة."""
    mapping = {}
    lower = {c: str(c).strip().lower() for c in df.columns}
    for target, aliases in COLUMN_ALIASES.items():
        al = [a.lower() for a in aliases]
        for col, lc in lower.items():
            if col in mapping:
                continue
            if lc in al:
                mapping[col] = target
                break
    return mapping


def unify(df: pd.DataFrame, source: str | None = None, country: str | None = None,
          mapping: dict | None = None, is_demo: bool = False) -> pd.DataFrame:
    """تحويل DataFrame خام إلى المخطط الموحد. يرفع ValueError إن لم يوجد نص قابل للتحليل."""
    mapping = mapping or map_columns(df)
    out = pd.DataFrame(index=df.index)
    for col, target in mapping.items():
        if target not in out.columns:
            out[target] = df[col]
    for c in UNIFIED_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan

    if out["title"].isna().all() and out["description"].isna().all():
        raise ValueError("لم يتم العثور على عمود للمسمى الوظيفي أو الوصف. حدّد الأعمدة يدوياً.")

    # النص الأصلي يُحفظ كما هو قبل أي تنظيف
    if out["description_raw"].isna().all():
        out["description_raw"] = out["description"]
    for c in ("title", "description", "requirements", "company"):
        out[c] = out[c].map(clean_text)
    if out["location_raw"].isna().all():
        out["location_raw"] = out["location"]
    out["location"] = out["location"].map(normalize_governorate)
    if country:
        out["country"] = out["country"].fillna(country).replace("", country)
    out["country"] = out["country"].fillna("Unknown")
    if source:
        out["source"] = out["source"].fillna(source).replace("", source)
    out["source"] = out["source"].fillna("unknown")
    out["posted_date"] = pd.to_datetime(out["posted_date"], errors="coerce", dayfirst=False)
    out["deadline"] = pd.to_datetime(out["deadline"], errors="coerce")
    out["experience_years"] = out["experience_years"].map(_parse_experience)
    out["is_demo"] = bool(is_demo)

    # حذف الصفوف الفارغة والمكررة: أولاً بالرابط/معرف المصدر، ثم بالمحتوى
    out["_text"] = (out["title"] + " " + out["description"]).str.strip()
    out = out[out["_text"].str.len() > 3]
    has_url = out["url"].notna() & (out["url"].astype(str).str.len() > 0)
    out = pd.concat([out[has_url].drop_duplicates(subset=["source", "url"]), out[~has_url]])
    out = out.drop_duplicates(subset=["_text", "company"]).drop(columns="_text")

    def _id(r):
        # معرف ثابت عبر التشغيلات: معرف المصدر ← الرابط ← المحتوى (تعتمد عليه التصنيفات اليدوية)
        if isinstance(r.source_job_id, str) and r.source_job_id:
            key = f"{r.source}|id|{r.source_job_id}"
        elif isinstance(r.url, str) and r.url:
            key = f"{r.source}|url|{r.url}"
        else:
            key = f"{r.source}|{r.title}|{r.company}|{r.posted_date}|{str(r.description)[:80]}"
        return hashlib.md5(key.encode()).hexdigest()[:12]
    out["job_id"] = out.apply(_id, axis=1)
    return out[UNIFIED_COLUMNS].reset_index(drop=True)


def job_text(df: pd.DataFrame) -> pd.Series:
    """النص الذي يُمرر للنماذج: المسمى + الوصف + المتطلبات بعد clean_for_model (يُزال قسم التقديم والضوضاء والتكرار)."""
    from preprocessing.text import clean_for_model
    src = df["description_raw"].where(df["description_raw"].notna(), df["description"]) if "description_raw" in df \
        else df["description"]
    body = src.fillna("").map(clean_for_model)
    req = df["requirements"].fillna("").map(clean_for_model)
    txt = df["title"].fillna("") + ".\n" + body + "\n" + req
    return txt.map(lambda t: "\n".join(dict.fromkeys(l for l in t.split("\n") if l.strip()))).str.strip()
