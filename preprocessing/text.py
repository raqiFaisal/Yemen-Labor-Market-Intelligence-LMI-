"""معالجة النصوص العربية/الإنجليزية/المختلطة.

مبدأ أساسي: النص الأصلي لا يُعدَّل أبداً (يُحفظ في description_raw). نُنتج منه نسختين مشتقتين:
  • clean_for_model(text)  → نص نظيف للـ Transformer: إزالة التشكيل والتطويل والضوضاء والتكرار،
                              مع إبقاء الحروف كما هي (لا توحيد للألف/الياء حتى لا يتشوه المعنى للنموذج).
  • normalize_for_match(t) → نسخة مطابقة فقط (للقاموس والبحث): توحيد أ/إ/آ→ا، ى→ي، ة→ه، الأرقام، الفواصل.
"""
from __future__ import annotations

import re
import unicodedata

_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_PHONE = re.compile(r"(?:\+?967|00967)?[\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{3}\b")
_ZW = re.compile(r"[​-‏‪-‮⁦-⁩﻿]")  # محارف خفية واتجاه

# أسطر ضوضاء متكررة في مواقع الوظائف (أزرار، مشاركة، تسجيل دخول ...)
_NOISE_LINE = re.compile(
    r"^(?:login to (?:mark|save)|save\s*&\s*share|share on|share this|apply now|back to jobs|related jobs|"
    r"track your application|time remaining|days?\s+hours?\s+minutes?|printed from|"
    r"تسجيل الدخول|شارك|مشاركة|قدم الآن|تقدم الآن|العودة للوظائف|وظائف ذات صلة)\b", re.I)
# أقسام لا تصف الوظيفة نفسها: يُقطع النص عندها لأغراض النموذج (تبقى في النص الأصلي)
_CUT_SECTIONS = re.compile(r"^(?:how to apply|important notes|كيفية التقديم|طريقة التقديم|ملاحظات هامة|للتقديم)\b", re.I)


def strip_diacritics(t: str) -> str:
    return _DIACRITICS.sub("", t).replace(_TATWEEL, "")


def normalize_letters(t: str) -> str:
    t = re.sub("[إأآٱا]", "ا", t)
    return t.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")


def normalize_for_match(t) -> str:
    """نسخة للمطابقة فقط — لا تُعرض ولا تُخزن بدل الأصل."""
    if not isinstance(t, str):
        return ""
    t = unicodedata.normalize("NFKC", _ZW.sub("", t)).translate(_DIGITS).lower()
    t = normalize_letters(strip_diacritics(t))
    t = re.sub(r"[_\-–—/\\|•·]+", " ", t)          # Power-BI / power_bi / power/bi → power bi
    return re.sub(r"\s+", " ", t).strip()


def dedupe_lines(lines: list[str]) -> list[str]:
    """إزالة الأسطر المكررة حرفياً أو بعد التطبيع (شائع في إعلانات ثنائية اللغة والصفحات المنسوخة)."""
    seen, out = set(), []
    for l in lines:
        k = normalize_for_match(l)
        if k and k not in seen:
            seen.add(k)
            out.append(l)
    return out


def clean_for_model(text, cut_apply_section: bool = True) -> str:
    """نص نظيف لمدخلات الـ Transformer مع الحفاظ على الكلمات الأصلية."""
    if not isinstance(text, str) or not text.strip():
        return ""
    t = unicodedata.normalize("NFKC", _ZW.sub("", text)).translate(_DIGITS)
    t = strip_diacritics(re.sub(r"<[^>]+>", " ", t))
    t = _EMAIL.sub(" ", _URL.sub(" ", t))
    t = _PHONE.sub(" ", t)
    lines = []
    for raw in re.split(r"[\r\n]+", t):
        l = re.sub(r"\s+", " ", raw).strip(" \t-*•·")
        if not l or _NOISE_LINE.match(l):
            continue
        if cut_apply_section and _CUT_SECTIONS.match(l):
            break
        lines.append(l)
    return "\n".join(dedupe_lines(lines))


_PHRASE_SPLIT = re.compile(r"[\n\.،,;؛:••|()\[\]]+|\s+-\s+|\s+(?:و|or|and)\s+(?=\S)")


def split_phrases(text: str, min_len: int = 3, max_words: int = 8) -> list[str]:
    """عبارات قصيرة للمطابقة الدلالية — العبارات الطويلة تُستبعد لأن تشابهها مع اسم مهارة غير دقيق."""
    out = []
    for p in _PHRASE_SPLIT.split(text or ""):
        p = p.strip(" -*")
        if len(p) >= min_len and 1 <= len(p.split()) <= max_words:
            out.append(p)
    return dedupe_lines(out)
