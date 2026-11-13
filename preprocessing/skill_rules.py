"""قواعد قبول المهارات — السياق، النفي، تعدد المعاني، وشروط المطابقة الدلالية.

كل الأنماط تُطبَّق على النسخة المطبّعة من المقطع (normalize_for_match): أحرف صغيرة، ا/ي/ه موحدة، بلا تشكيل.
أُضيفت القواعد استجابةً لأخطاء ظهرت فعلياً في تشغيل Colab على إعلانات YemenHR الحقيقية، مثل:
  REST APIs في إعلان جراح، Machine Learning من "Learning Outcomes"، Leadership من "Admin"،
  Recruitment من "مكان العمل"، Cybersecurity من "ونظم الأمان"، Payroll من "الأرباح".
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------- عتبات المطابقة الدلالية (Transformer)
SEM_ACCEPT = 0.75        # قبول دلالي (مع شرط الكلمة الدالّة)
SEM_REVIEW = 0.65        # بين REVIEW و ACCEPT ← مراجعة بشرية بدل القبول
SEM_NO_ANCHOR_REVIEW = 0.85  # تشابه عالٍ جداً بلا كلمة دالّة ← مراجعة (قد تكون صياغة مختلفة صحيحة)، لا قبول
SEM_MARGIN = 0.03        # فرق أدنى عن المهارة الثانية (عبارات غامضة بين مهارتين تُرفض)
SOFT_SKILL_ACCEPT = 0.82 # المهارات الشخصية عامة الصياغة: عتبة أعلى

# ---------------------------------------------------------------- كلمات لا تثبت مهارة بمفردها
GENERIC_TOKENS = {
    # إنجليزية
    "learning", "learn", "management", "manage", "managing", "skills", "skill", "system", "systems", "work", "working",
    "experience", "knowledge", "ability", "abilities", "good", "strong", "excellent", "outcomes", "outcome", "officer",
    "admin", "administration", "administrative", "assistant", "staff", "data", "information", "services", "service",
    "support", "development", "program", "programme", "project", "level", "basic", "advanced", "proficiency", "use",
    "using", "tools", "tool", "and", "the", "of", "in", "with", "for", "to", "a", "an", "or", "on", "at", "as", "is",
    "be", "are", "will", "must", "should", "required", "preferred", "plus", "years", "year", "minimum", "at least",
    "safety", "security", "quality", "process", "processes", "planning", "team", "office", "field", "general",
    # عربية (مطبّعة)
    "العمل", "عمل", "مكان", "اداره", "اداري", "الاداري", "الاداريه", "نظم", "نظام", "الانظمه", "انظمه", "مهارات", "مهاره",
    "خبره", "الخبره", "القدره", "قدره", "معرفه", "المعرفه", "جيد", "جيده", "ممتاز", "ممتازه", "المسؤول", "مسؤول", "موظف",
    "الموظفين", "البيانات", "بيانات", "المعلومات", "معلومات", "الخدمات", "خدمات", "الدعم", "دعم", "التطوير", "تطوير",
    "البرنامج", "برنامج", "المشروع", "مشروع", "مستوي", "استخدام", "في", "من", "علي", "الي", "مع", "و", "او", "ان",
    "الامان", "امان", "السلامه", "الجوده", "الارباح", "ارباح", "سنوات", "سنه", "الاقل", "يفضل", "مطلوب", "شرط",
}

# ---------------------------------------------------------------- سياق لا يصف مهارات المرشح
# أسطر التقديم والتواصل والتعريف الإداري: لا تُستخرج منها مهارات
BOILERPLATE_LINE = re.compile(
    r"(how to apply|to apply|apply (now|online|through|via|by)|send (your )?(cv|resume)|\bcv\b|resume|deadline|"
    r"shortlisted|equal opportunit|only shortlisted|whatsapp|e ?mail|للتقديم|التقديم|السيره الذاتيه|ارسال السيره|"
    r"سيتم التواصل|المرشحين المؤهلين|تكافؤ الفرص|اخر موعد|للاستفسار|رقم الهاتف|واتساب|البريد الالكتروني|"
    r"^مكان العمل|^location|^duty station|^job title|^المسمي الوظيفي|^reports? to|^يرفع تقاريره|^contract|^نوع العقد|"
    r"^deadline|^salary|^الراتب)")

# النفي: يُرفض التطابق إن سبقه/تلاه نفي قريب
NEGATION_BEFORE = re.compile(r"(\bno\b|\bnot\b|without|لا يشترط|لا يلزم|غير مطلوب|ليس مطلوب|ليس شرط|دون|بدون|لا حاجه)[^.\n]{0,25}$")
NEGATION_AFTER = re.compile(r"^[^.\n]{0,20}(is not required|not required|not necessary|غير مطلوب|ليس شرطا|ليست شرطا|غير ضروري)")

# ---------------------------------------------------------------- قواعد خاصة بكل مهارة
# ambiguous: مرادفات متعددة المعاني تحتاج سياقاً (require) وإلا تُحال للمراجعة
# require:   أنماط سياق يجب وجود أحدها في المقطع لقبول المرادف الغامض
# exclude:   أنماط إن وُجدت في المقطع تُرفض المهارة (معنى آخر للكلمة)
# sem_accept: عتبة دلالية خاصة
SKILL_RULES: dict[str, dict] = {
    "api": {"exclude": [r"\brest(ing)?\b(?! ?api)", r"\b(rest|break) (period|day|time)"]},
    "machine_learning": {"ambiguous": ["ml"], "require": [r"machine", r"\bmodel", r"\bai\b", r"python", r"ذكاء", r"تعلم الاله"],
                         "exclude": [r"\d+ ?ml\b", r"learning outcomes?", r"\be ?learning\b", r"learning (and|&) development",
                                     r"lessons? learn"]},
    "leadership": {"exclude": [r"قياده (السيار|المركب|الدراج|الشاحن|الباص)", r"رخصه قياده", r"\bdriv(ing|er)\b"]},
    "recruitment": {"ambiguous": ["التوظيف"], "require": [r"استقطاب", r"مقابلات", r"اختيار المرشحين", r"recruit", r"talent",
                                                        r"الموارد البشريه", r"تعيين", r"عمليات التوظيف"],
                    "exclude": [r"اعلان (توظيف|وظيفه)", r"فرصه (توظيف|عمل)", r"طلب توظيف", r"نموذج (التوظيف|توظيف)",
                                r"recruitment process will", r"this recruitment"]},
    "cybersecurity": {"exclude": [r"الامن الغذائي", r"food security", r"security guard", r"حراسه", r"\bحارس", r"الامان",
                                  r"safety", r"امن وسلامه", r"السلامه والامن", r"security (and|&) safety", r"security officer",
                                  r"security situation", r"الوضع الامني", r"physical security"]},
    "payroll": {"exclude": [r"الارباح", r"profit"]},
    "networking": {"ambiguous": ["شبكات", "الشبكات", "networking"],
                   "require": [r"cisco", r"\blan\b", r"\bwan\b", r"ccna", r"router", r"tcp", r"حاسوب", r"كمبيوتر", r"تقنيه",
                               r"الانترنت", r"سيرفر", r"server", r"switch", r"network (admin|engineer|infrastructure)", r"\bit\b"]},
    "communication": {"ambiguous": ["الاتصال", "التواصل"],
                      "require": [r"مهارات", r"مهاره", r"القدره", r"فعال", r"كتابي", r"شفهي", r"skills", r"قوي"],
                      "exclude": [r"للاتصال", r"الاتصال (علي|ب|بنا|على)", r"سيتم التواصل", r"contact (us|person|details)"]},
    "auditing": {"ambiguous": ["مراجعه"], "require": [r"مالي", r"حسابات", r"داخلي", r"خارجي", r"تدقيق", r"audit", r"محاسب"]},
    "training_dev": {"ambiguous": ["التدريب"], "require": [r"تطوير", r"احتياجات تدريبيه", r"برامج تدريبيه", r"تيسير", r"facilitat",
                                                          r"training needs", r"مدرب", r"تصميم"],
                     "exclude": [r"سيتم تدريب", r"training will be provided", r"on the job training", r"تدريب علي راس العمل"]},
    "attention_detail": {"ambiguous": ["الدقه"], "require": [r"التفاصيل", r"الانتباه", r"detail"]},
    "report_writing": {"ambiguous": ["reporting"], "require": [r"writ", r"prepar", r"narrative", r"donor", r"كتابه", r"اعداد"],
                       "exclude": [r"reporting (to|line|directly)", r"reports? (to|directly)", r"يرفع تقاريره", r"يتبع (مباشره|ل)"]},
    "cash_management": {"ambiguous": ["cash", "الصندوق"],
                        "require": [r"cash (management|handling|box|flow)", r"petty", r"النقديه", r"امين (ال)?صندوق", r"cashier", r"خزينه"],
                        "exclude": [r"cash (assistance|transfer|distribution|programming|for work|based)", r"المساعدات النقديه",
                                    r"النقد مقابل العمل", r"صندوق (التنميه|الرعايه|الضمان|البريد|الشكاوي)"]},
    "sql": {"ambiguous": ["قواعد البيانات", "قواعد بيانات"],
            "require": [r"\bsql\b", r"mysql", r"oracle", r"تصميم قواعد", r"برمجه", r"\bdba\b", r"استعلام", r"database admin"]},
    "humanitarian": {"ambiguous": ["ngo", "المنظمات الانسانيه"],
                     "require": [r"experience", r"خبره", r"programm", r"working (with|in)", r"العمل (في|مع)"]},
    "strategic_planning": {"ambiguous": ["التخطيط"], "require": [r"استراتيج", r"strategic"]},
    "wash": {"case_sensitive": ["WASH"], "exclude": [r"wash (hands|your|the)", r"hand ?wash", r"غسل"]},
    "excel": {"exclude": [r"\bexcel(s|led|ling)? (in|at)\b"]},
    "ms_office": {"ambiguous": ["الحاسوب", "الحاسب الالي"],
                  "require": [r"مهارات", r"استخدام", r"برامج", r"office", r"اجاده", r"اتقان", r"skills"],
                  "exclude": [r"(علوم|هندسه|تخصص) (ال)?حاسوب", r"computer science"]},
    "me": {"ambiguous": ["meal"], "require": [r"monitoring", r"evaluation", r"accountability", r"رصد", r"تقييم"],
           "exclude": [r"meals?( allowance| provided| per)"]},
    "pharmacy": {"ambiguous": ["الادويه"], "require": [r"صيدل", r"pharmac", r"صرف", r"وصفات"]},
    "civil_eng": {"ambiguous": ["construction"], "require": [r"engineer", r"هندس", r"\bsite\b", r"موقع", r"بناء", r"civil"]},
    "data_analysis": {},
}
SOFT_SKILLS_THRESHOLD_IDS = {"communication", "teamwork", "leadership", "problem_solving", "time_management",
                             "critical_thinking", "negotiation", "attention_detail", "adaptability"}

_compiled: dict = {}


def rule(sid: str) -> dict:
    if sid not in _compiled:
        r = SKILL_RULES.get(sid, {})
        _compiled[sid] = {
            "ambiguous": set(r.get("ambiguous", [])),
            "require": [re.compile(p) for p in r.get("require", [])],
            "exclude": [re.compile(p) for p in r.get("exclude", [])],
            "case_sensitive": r.get("case_sensitive", []),
            "sem_accept": r.get("sem_accept", SOFT_SKILL_ACCEPT if sid in SOFT_SKILLS_THRESHOLD_IDS else SEM_ACCEPT),
        }
    return _compiled[sid]


def stem(tok: str) -> str:
    """تجذيع خفيف للمقارنة فقط: سوابق/لواحق عربية شائعة، وجمع إنجليزي."""
    t = tok
    for p in ("وال", "بال", "لل", "ال", "و", "ب", "ل"):
        if t.startswith(p) and len(t) - len(p) >= 3:
            t = t[len(p):]
            break
    for s in ("ات", "ين", "ون", "يه", "ه", "ي"):
        if t.endswith(s) and len(t) - len(s) >= 3:
            t = t[:-len(s)]
            break
    if re.fullmatch(r"[a-z]+", t) and len(t) > 4 and t.endswith("s"):
        t = t[:-1]
    return t[:5] if re.fullmatch(r"[a-z]+", t) else t[:4]


def content_tokens(norm_text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9#+&.]+|[؀-ۿ]+", norm_text) if w not in GENERIC_TOKENS and len(w) >= 2]
