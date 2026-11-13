"""توحيد تسميات الفئات القادمة من المصادر إلى المجالات الثمانية.

المطابقة صريحة فقط (لا تخمين بالكلمات المفتاحية): كل تسمية مصدر يجب أن تكون في data/taxonomy/category_map.json،
وقيمتها إما معرف مجال أو null (غامضة/خارج المجالات الثمانية → تحتاج مراجعة يدوية، ولا تُستخدم كتسمية).
التسميات غير الموجودة في الملف تُسجَّل في تقرير البيانات لمراجعتها وإضافتها.
"""
from __future__ import annotations

import json

from core.config import CATEGORY_MAP_FILE, DOMAINS_FILE
from preprocessing.text import normalize_for_match

# مطابقات افتراضية موثقة. null = غامض عمداً (مثلاً Development في YemenHR تعني قطاع التنمية/البرامج الإنسانية
# وتشمل وظائف من مجالات متعددة، فلا يصح ربطها بمجال واحد).
DEFAULT_MAP = {
    # YemenHR (نوفمبر 2026)
    "Computers/IT": "software_engineering",
    "Finance/Accounting": "accounting",
    "Engineering": "engineering",
    "Healthcare": "healthcare",
    "Sales/Marketing": "marketing",
    "Logistics": "management",
    "Administration": "management",
    "Human Resources": "human_resources",
    "Monitoring and Evaluation": "data_science",
    "Development": None,
    "Education/Training": None,
    "Consulting": None,
    "Legal": None,
    "Media/Communications": None,
    "Research": None,
    "Others": None,
}


def _domain_names() -> dict:
    """معرفات المجالات وأسماؤها العربية/الإنجليزية → المعرف (تسميات صريحة لا لبس فيها)."""
    doms = json.loads(DOMAINS_FILE.read_text(encoding="utf-8")) if DOMAINS_FILE.exists() else {}
    out = {}
    for k, v in doms.items():
        for n in (k, v.get("en", ""), v.get("ar", "")):
            if n:
                out[normalize_for_match(n)] = k
    return out


def load_map() -> dict:
    if not CATEGORY_MAP_FILE.exists():
        CATEGORY_MAP_FILE.write_text(json.dumps(DEFAULT_MAP, ensure_ascii=False, indent=1), encoding="utf-8")
    m = json.loads(CATEGORY_MAP_FILE.read_text(encoding="utf-8"))
    norm = {normalize_for_match(k): v for k, v in m.items()}
    for k, v in _domain_names().items():
        norm.setdefault(k, v)
    return norm


def map_category(raw, cmap: dict | None = None) -> tuple[str | None, str]:
    """(domain|None, status) — status: empty | mapped | ambiguous | unmapped | conflict"""
    if not isinstance(raw, str) or not raw.strip():
        return None, "empty"
    cmap = cmap if cmap is not None else load_map()
    vals, unknown, ambiguous = set(), False, False
    for part in [p for p in raw.split(";") if p.strip()]:
        k = normalize_for_match(part)
        if k not in cmap:
            unknown = True
        elif cmap[k] is None:
            ambiguous = True
        else:
            vals.add(cmap[k])
    if len(vals) == 1:
        return vals.pop(), "mapped"
    if len(vals) > 1:
        return None, "conflict"
    return None, "unmapped" if unknown else "ambiguous" if ambiguous else "empty"
