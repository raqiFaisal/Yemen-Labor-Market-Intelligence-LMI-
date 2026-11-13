"""استخراج المؤهل والخبرة والتخصص ونوع التوظيف من نص الإعلان (قواعد — كل قيمة معها دليل نصي حرفي).

لا افتراضات: إن لم يُذكر المؤهل يبقى فارغاً. الخبرة تُستخرج فقط عند وجود رقم/عدد مكتوب بجوار كلمة خبرة/experience.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

from core.config import TAXONOMY_DIR
from preprocessing.schema import normalize_arabic

EDU_LEVELS = [  # (المستوى، الترتيب، أنماط) — الترتيب يسمح بتحديد «أدنى مؤهل مطلوب»
    ("PhD", 6, r"دكتوراه|دكتوراة|ph\.?\s?d|doctorate"),
    ("Master", 5, r"ماجستير|ماجيستير|master'?s?|m\.?sc|mba|m\.?a\.?(?=\s)"),
    ("Bachelor", 4, r"بكالوريوس|بكالريوس|بكلاريوس|ليسانس|مؤهل جامعي|شهادة جامعية|درجة جامعية|bachelor'?s?|b\.?sc|b\.?a\.?(?=\s)|university degree"),
    ("Diploma", 3, r"دبلوم|diploma"),
    ("Secondary", 2, r"ثانوية عامة|الثانوية|ثانوي|high school|secondary school"),
    ("Basic", 1, r"يجيد القراءة والكتابة|القراءة والكتابة|الشهادة الأساسية|اساسي"),
]
EDU_RX = [(lvl, rank, re.compile(rf"(?<![\w])(?:ال|بال|وال|و|ب)?(?:{p})", re.I)) for lvl, rank, p in EDU_LEVELS]
EDU_CONTEXT_BAD = re.compile(r"دبلوم\s+(?:الرخصة|ICDL|برمجة|صيانة|التسويق|المحاسبة المالية الحديثة)|دورة|سجل|التسجيل", re.I)

AR_NUM = {"سنة": 1, "سنه": 1, "عام": 1, "سنتين": 2, "سنتان": 2, "عامين": 2, "عامان": 2, "ثلاث": 3, "ثلاثة": 3, "أربع": 4,
          "اربع": 4, "أربعة": 4, "اربعة": 4, "خمس": 5, "خمسة": 5, "ست": 6, "ستة": 6, "سبع": 7, "سبعة": 7, "ثمان": 8,
          "ثمانية": 8, "تسع": 9, "عشر": 10, "عشرة": 10, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "ten": 10}
AR_DIG = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
N = r"(\d{1,2}|" + "|".join(sorted(AR_NUM, key=len, reverse=True)) + r")"
EXP_RX = [
    re.compile(rf"(?:خبرة|خبره|الخبرة|الخبره)[^\n.،]{{0,40}}?(?:لا تقل عن|لاتقل عن|لا يقل عن|لايقل عن|من|مدة|مدتها)?\s*{N}\s*(?:[-–إلىالى ]+\s*{N}\s*)?(?:سنوات|سنة|سنه|أعوام|اعوام|عام|سنين)(?!\s*(?:أشهر|اشهر|شهور))", re.I),
    re.compile(rf"{N}\s*(?:[-–]\s*{N}\s*)?\+?\s*(?:سنوات|سنة|سنه|أعوام|اعوام|سنين)\s*(?:من\s*)?(?:ال)?(?:خبرة|خبره)", re.I),
    re.compile(rf"(?:minimum\s+(?:of\s+)?|at least\s+)?{N}\s*(?:\(\d+\)\s*)?(?:[-–to ]+\s*{N}\s*)?\+?\s*years?[’']?\s*(?:of\s+)?(?:[\w-]+\s+){{0,4}}experience", re.I),
    re.compile(rf"(?:at least|minimum(?: of)?|a minimum of|more than|over)\s+{N}\s*(?:\(\d+\)\s*)?\+?\s*years?(?![\w])", re.I),
    re.compile(rf"(?:سنتين|سنتان|عامين)\s*(?:من\s*)?(?:ال)?(?:خبرة|خبره)|(?:خبرة|خبره)\s*(?:لا تقل عن\s*)?(سنتين|سنتان|عامين|سنة|سنه)", re.I),
]
FRESH = re.compile(r"حديث(?:ي)? التخرج|حديثي التخرج|لا يشترط الخبرة|لايشترط الخبرة|بدون خبرة|بدون أي خبرة|fresh graduates?|no experience required|entry[- ]level", re.I)
EMP_TYPES = [
    ("full_time", r"دوام كامل|full[- ]time|دوام كلي"), ("part_time", r"دوام جزئي|part[- ]time|دوام جزئى|نصف دوام"),
    ("contract", r"عقد(?: عمل)? (?:لمدة|مؤقت)|contract (?:type|duration|length)|type of contract|fixed[- ]term|consultancy|consultant|استشاري|مستشار"), ("temporary", r"مؤقت|temporary|لمهمة محدودة"),
    ("daily", r"عامل يومي|daily worker|بالأجر اليومي|daily rate|اجر يومي|أجر يومي"), ("freelance", r"عمل حر|freelance|بالقطعة|بالمشروع"),
    ("internship", r"internship|تدريب ميداني|intern(?![a-z])"), ("volunteer", r"تطوع|volunteer"),
]
EMP_RX = [(k, re.compile(rf"(?<![\w])(?:{p})", re.I)) for k, p in EMP_TYPES]
QUAL_HEAD = re.compile(r"^\W*(?:المؤهلات(?: والخبرات| والمتطلبات)?|المتطلبات|الشروط|شروط الوظيفة|المؤهل العلمي|متطلبات الوظيفة|"
                       r"Qualifications|Requirements|Required Qualifications|Minimum Qualifications|Education and Experience|"
                       r"Skills and Experience|Knowledge,? Skills)\b.*$", re.I | re.M)
NEXT_HEAD = re.compile(r"^\W*(?:المهام|المسؤوليات|المزايا|طريقة التقديم|كيفية التقديم|للتقديم|للتواصل|الراتب|Responsibilities|"
                       r"Duties|How to apply|Benefits|Application|Submission)\b", re.I | re.M)


def _num(tok: str | None) -> int | None:
    if not tok:
        return None
    tok = tok.translate(AR_DIG).strip().lower()
    if tok.isdigit():
        v = int(tok)
        return v if 0 < v <= 30 else None
    return AR_NUM.get(tok)


def _ev(text: str, m: re.Match, w: int = 50) -> str:
    return text[max(0, m.start() - 15):m.end() + w].replace("\n", " ").strip()


def education(text: str) -> dict:
    found = []
    for lvl, rank, rx in EDU_RX:
        for m in rx.finditer(text):
            ctx = text[max(0, m.start() - 10):m.end() + 40]
            if lvl == "Diploma" and EDU_CONTEXT_BAD.search(ctx):
                continue          # «دبلوم الرخصة الدولية» دورة وليس مؤهلاً
            found.append((rank, lvl, _ev(text, m)))
            break
    if not found:
        return {"education": "", "education_min": "", "education_evidence": ""}
    found.sort()
    return {"education": ";".join(l for _, l, _ in sorted(found, reverse=True)), "education_min": found[0][1],
            "education_evidence": found[0][2][:160]}


def experience(text: str) -> dict:
    for i, rx in enumerate(EXP_RX):
        m = next((m for m in rx.finditer(text)
                  if i != 3 or re.search(r"experience|خبرة|خبره", text[max(0, m.start() - 60):m.end() + 80], re.I)), None)
        if not m:
            continue
        nums = [_num(g) for g in m.groups() if g]
        nums = [n for n in nums if n is not None]
        if not nums and re.search(r"سنتين|سنتان|عامين", m.group(0)):
            nums = [2]
        if nums:
            return {"experience_min_years": min(nums), "experience_max_years": max(nums),
                    "experience_evidence": _ev(text, m, 20)[:160]}
    m = FRESH.search(text)
    if m:
        return {"experience_min_years": 0, "experience_max_years": 0, "experience_evidence": _ev(text, m, 10)[:160]}
    return {"experience_min_years": None, "experience_max_years": None, "experience_evidence": ""}


@lru_cache(maxsize=1)
def _majors():
    try:
        m = json.loads((TAXONOMY_DIR / "majors.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        m = {}
    extra = {"المحاسبة": [], "محاسبة": [], "إدارة الأعمال": [], "ادارة اعمال": [], "الهندسة المدنية": [], "هندسة مدنية": [],
             "الهندسة الكهربائية": [], "الهندسة المعمارية": [], "الطب البشري": [], "التمريض": [], "الصيدلة": [], "المختبرات": [],
             "القانون": [], "الشريعة والقانون": [], "الاقتصاد": [], "التسويق": [], "اللغة الإنجليزية": [], "الإعلام": [],
             "علم الاجتماع": [], "الخدمة الاجتماعية": [], "علم النفس": [], "الصحة العامة": [], "التغذية": [], "الزراعة": [],
             "Accounting": [], "Finance": [], "Business Administration": [], "Civil Engineering": [], "Electrical Engineering": [],
             "Computer Science": [], "Information Technology": [], "Public Health": [], "Nursing": [], "Pharmacy": [], "Medicine": [],
             "Law": [], "Economics": [], "Social Sciences": [], "Social Work": [], "Psychology": [], "Agriculture": [],
             "Engineering": [], "Statistics": [], "Nutrition": [], "Marketing": [], "Journalism": [], "Development Studies": []}
    names = list(dict.fromkeys(list(m) + list(extra)))
    return [(n, re.compile(rf"(?<![\w]){re.escape(normalize_arabic(n.lower()))}(?![\w])")) for n in names]


MAJOR_CTX = re.compile(r"تخصص|تخصصات|بكالوريوس|بكالريوس|ليسانس|دبلوم|ماجستير|مؤهل|degree|bachelor|master|diploma|graduate|studies", re.I)


def majors(text: str) -> dict:
    norm = normalize_arabic(text.lower())
    hits, ev = [], ""
    for name, rx in _majors():
        m = rx.search(norm)
        if m and MAJOR_CTX.search(norm[max(0, m.start() - 60):m.start()]):
            hits.append(name)
            ev = ev or norm[max(0, m.start() - 40):m.end() + 10].replace("\n", " ")
    return {"specializations": ";".join(dict.fromkeys(hits)), "specialization_evidence": ev[:160]}


def employment_type(text: str, title: str = "") -> str:
    """تطوع/تدريب من المسمى فقط (كلمة volunteers شائعة في تعريف المنظمات)، والباقي من النص المعطى."""
    return ";".join(k for k, rx in EMP_RX if rx.search(title if k in ("volunteer", "internship") else f"{title}\n{text}"))


def qualifications_block(text: str, max_chars: int = 1500) -> str:
    """قسم المؤهلات/المتطلبات كما ورد في النص (أول قسم يطابق)، بلا إعادة صياغة."""
    m = QUAL_HEAD.search(text)
    if not m:
        return ""
    rest = text[m.end():]
    n = NEXT_HEAD.search(rest)
    return rest[:n.start() if n else max_chars].strip()[:max_chars]


def language(text: str) -> str:
    ar = len(re.findall(r"[؀-ۿ]", text or ""))
    la = len(re.findall(r"[A-Za-z]", text or ""))
    if ar + la == 0:
        return ""
    r = ar / (ar + la)
    return "ar" if r > 0.7 else "en" if r < 0.3 else "mixed"


MAX_ANALYSE = 12000


def extract_all(text: str, title: str = "") -> dict:
    """المؤهل/الخبرة/التخصص من قسم المؤهلات إن وُجد (وإلا النص كاملاً) — حتى لا تُلتقط من تعريف الجهة.
    نوع التوظيف من المسمى + أول 800 حرف (حيث يُذكر عادة)."""
    text = (text or "")[:MAX_ANALYSE]          # حد للتحليل فقط (يمنع التراجع المكلف للتعابير على منشورات ضخمة)؛ النص المحفوظ كامل
    qual = qualifications_block(text)
    scope = qual if len(qual) >= 60 else text
    exp = experience(scope)
    if exp["experience_min_years"] is None and scope is not text:
        exp = experience(text)
    edu = education(scope)
    if not edu["education"] and scope is not text:
        edu = education(text)
    return {**edu, **exp, **majors(scope), "employment_type": employment_type(text[:800], title),
            "qualifications": qual, "language": language(text)}


NEGOTIABLE = re.compile(r"بعد\s*المقابل|عند\s*المقابل|حسب\s*(?:الخبر|الاتفاق|الكفاء|المؤهل)|يحدد|بالاتفاق|قابل للتفاوض|negotiable|"
                        r"competitive|according to|بالنسب[ةه]|نسب[ةه]\s*من", re.I)
AMOUNT = re.compile(r"(\d[\d,٬.]*)\s*(الف|ألف|آلاف|الاف|k|K)?\s*(?:[-–إلى الى]+\s*(\d[\d,٬.]*)\s*(الف|ألف|k)?)?\s*"
                    r"(ريال\s*سعودي|ريال سعودى|سعودي|ر\.?\s*س|ريال\s*يمني|ريال|دولار|\$|usd|yer|sar|ر\.?\s*ي)?", re.I)


def salary_parts(salary_text: str | None) -> dict:
    """الراتب كما ورد + المبلغ والعملة إن ذُكر رقم صريح. لا تحويل عملات ولا تقدير. «بعد المقابلة» = قابل للتفاوض بلا مبلغ."""
    t = (salary_text or "").translate(AR_DIG)
    out = {"salary_amount_min": None, "salary_amount_max": None, "salary_currency": None,
           "salary_negotiable": bool(NEGOTIABLE.search(t)) if t else None}
    for m in AMOUNT.finditer(t):
        try:
            a = float(m.group(1).replace(",", "").replace("٬", "").rstrip("."))
        except ValueError:
            continue
        if a < 10 and not m.group(2):
            continue                       # «10%» أو أرقام صغيرة ليست مبلغاً
        if re.match(r"\s*%", t[m.end(1):m.end(1) + 2]):
            continue
        b = m.group(3)
        mult = 1000 if (m.group(2) or (b and m.group(4))) else 1      # «100-150 الف»: الألف تعود على الطرفين
        bv = float(b.replace(",", "")) * (1000 if (m.group(4) or m.group(2)) else 1) if b else None
        cur = (m.group(5) or "").lower()
        cur = ("SAR" if re.search(r"سعود|sar|ر\.?\s*س", cur) else "USD" if re.search(r"دولار|\$|usd", cur)
               else "YER" if cur else None)
        out.update(salary_amount_min=a * mult, salary_amount_max=bv or a * mult, salary_currency=cur)
        break
    return out
