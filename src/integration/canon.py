"""توحيد صيغ التخصصات للتحليل فقط (لا يُعدّل ملفات البيانات).

البيانات الحقيقية تحوي 55 صيغة لنحو 30 تخصصاً: «محاسبة» / «المحاسبة» / «Accounting»، «إدارة الأعمال» / «ادارة اعمال» /
«Business Administration»... الخريطة صريحة وقابلة للمراجعة؛ الصيغة غير المعروفة تبقى كما هي.
"""
from __future__ import annotations

SPEC_CANON = {
    "محاسبة": "المحاسبة", "Accounting": "المحاسبة",
    "ادارة اعمال": "إدارة الأعمال", "إدارة اعمال": "إدارة الأعمال", "Business Administration": "إدارة الأعمال",
    "Marketing": "التسويق",
    "Engineering": "الهندسة (عام)",
    "Social Sciences": "العلوم الاجتماعية",
    "Computer Science": "علوم الحاسوب",
    "Public Health": "الصحة العامة",
    "Nursing": "التمريض",
    "Information Technology": "تقنية المعلومات",
    "Psychology": "علم النفس",
    "Journalism": "الإعلام",
    "Civil Engineering": "الهندسة المدنية", "هندسة مدنية": "الهندسة المدنية",
    "Electrical Engineering": "الهندسة الكهربائية",
    "Pharmacy": "الصيدلة",
    "Statistics": "الإحصاء",
    "Law": "القانون",
    "Finance": "العلوم المالية والمصرفية",
    "Social Work": "الخدمة الاجتماعية",
    "Economics": "الاقتصاد",
    "Nutrition": "التغذية",
    "Medicine": "الطب البشري",
    "Development Studies": "دراسات التنمية",
    "Agriculture": "الزراعة",
    "المختبرات": "المختبرات الطبية",
}


def canon_spec(s: str) -> str:
    s = (s or "").strip()
    return SPEC_CANON.get(s, s)


def canon_specs(cell: str) -> str:
    """'محاسبة;Accounting;التسويق' → 'المحاسبة;التسويق' (بلا تكرار، بنفس الترتيب)."""
    out = []
    for x in str(cell or "").split(";"):
        c = canon_spec(x)
        if c and c not in out:
            out.append(c)
    return ";".join(out)
