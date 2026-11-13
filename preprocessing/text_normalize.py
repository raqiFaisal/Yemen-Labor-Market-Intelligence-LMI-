"""تنظيف النص العربي لإدخال النماذج — دون تغيير المعنى. النص الأصلي يبقى في raw_text دائماً.

ما يُطبَّق (قابل للعكس دلالياً):
  • NFKC + إزالة المحارف غير المرئية (ZWJ/ZWNJ/RLM/LRM/BOM) والتطويل «ـ»
  • الأرقام العربية الهندية والفارسية → أرقام لاتينية (٣ → 3)
  • تقليص التكرار المبالغ للحروف (اعلاااان → اعلان، !!!! → !) — حرف مكرر ≥3 يصبح واحداً
  • توحيد علامات الترقيم المتشابهة والمسافات، وإزالة الرموز التعبيرية والزخارف
ما لا يُطبَّق عمداً: توحيد الألف/الياء/التاء المربوطة وإزالة التشكيل (تغيّر الكتابة الصحيحة) — هذا للمطابقة الداخلية فقط.
"""
from __future__ import annotations

import re
import unicodedata

INVISIBLE = re.compile("[​-‏‪-‮⁦-⁩﻿­]")
TATWEEL = re.compile("ـ+")
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
REPEAT_CHAR = re.compile(r"([؀-ۿA-Za-z!?؟.،,])\1{2,}")
EMOJI_DECOR = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️⃣←-⇿⌀-⏿■-◿"
                         "━─═▪◾◽☆★✪✓✔➖➡⬅❇✳❤♦♨⭕🔹🔸]+")
PUNCT_MAP = str.maketrans({"“": '"', "”": '"', "«": '"', "»": '"', "’": "'", "‘": "'", "–": "-", "—": "-", "ـ": ""})
MARKDOWN = re.compile(r"(?<!\w)[*_~]{1,3}|[*_~]{1,3}(?!\w)")


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = INVISIBLE.sub("", t)
    t = TATWEEL.sub("", t)
    t = t.translate(DIGITS).translate(PUNCT_MAP)
    t = EMOJI_DECOR.sub(" ", t)
    t = MARKDOWN.sub("", t)
    t = REPEAT_CHAR.sub(r"\1", t)
    lines = [re.sub(r"[ \t]+", " ", l).strip(" -|") for l in t.splitlines()]
    out, blank = [], False
    for l in lines:
        if l:
            out.append(l)
            blank = False
        elif not blank and out:
            out.append("")
            blank = True
    return "\n".join(out).strip()


URL_RX = re.compile(r"https?://[^\s<>\"')\]]+|www\.[^\s<>\"')\]]+", re.I)
EMAIL_RX = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
NOT_APPLY = re.compile(r"t\.me/|telegram\.(?:me|org)|telegra\.ph|graph\.org|whatsapp\.com/channel|facebook\.com/(?:groups|profile|pages)|"
                       r"instagram\.com|tiktok\.com|youtube\.com|youtu\.be|twitter\.com|x\.com/|linkedin\.com/(?:in|company)/|"
                       r"chat\.whatsapp\.com|joinchat", re.I)
APPLY_HINT = re.compile(r"forms\.gle|docs\.google\.com/forms|apply|career|jobs?\b|vacanc|recruit|kobotoolbox|ee\.kobo|"
                        r"yemenhr\.com/jobs|linkedin\.com/jobs|unjobs|reliefweb|impactpool|eoi|erp\.|hr\.", re.I)


def apply_links(text: str, link_urls: str = "") -> dict:
    """رابط/بريد التقديم الفعلي فقط (روابط قنوات التواصل والمجموعات ليست روابط تقديم). null عند الغياب."""
    urls = list(dict.fromkeys([u.rstrip(".,;") for u in (link_urls or "").split()] + URL_RX.findall(text or "")))
    cand = [u for u in urls if not NOT_APPLY.search(u)]
    best = next((u for u in cand if APPLY_HINT.search(u)), cand[0] if cand else None)
    emails = [e for e in dict.fromkeys(EMAIL_RX.findall(text or "")) if not e.lower().endswith((".png", ".jpg"))]
    return {"apply_url": best, "apply_email": emails[0] if emails else None,
            "apply_urls_all": " ".join(cand) if cand else None}
