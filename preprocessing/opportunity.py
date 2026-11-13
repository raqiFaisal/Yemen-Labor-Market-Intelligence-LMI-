"""تصنيف المنشورات: نوع الفرصة + النطاق الجغرافي + نوع مكان العمل (قواعد مُعلّلة — ليست نموذج تعلم آلي).

مخرجات كل قرار: القيمة، الثقة (درجة استدلالية وليست احتمالاً)، الحالة accepted|review، والسبب (الكلمات التي طابقت).
هذه التسميات «Silver» (آلية). تقييمها الحقيقي يتم على عينة يصنفها إنسان (labels/opportunity_gold.csv).

opportunity_type: JOB | INTERNSHIP | TRAINING | SCHOLARSHIP | VOLUNTEERING | OTHER_OPPORTUNITY | NOT_RELEVANT
geographic_scope: YEMEN | REMOTE_GLOBAL | OUTSIDE_YEMEN | UNKNOWN
work_location_type: onsite | remote | hybrid | unknown
"""
from __future__ import annotations

import re

from preprocessing.schema import normalize_arabic
from preprocessing.telegram_parse import PHONE, match_governorate

TYPES = ["JOB", "INTERNSHIP", "TRAINING", "SCHOLARSHIP", "VOLUNTEERING", "OTHER_OPPORTUNITY", "NOT_RELEVANT"]
SCOPES = ["YEMEN", "REMOTE_GLOBAL", "OUTSIDE_YEMEN", "UNKNOWN"]
ACCEPT_AT = 0.70


def _rx(words: list[str]) -> re.Pattern:
    """كلمات عربية/إنجليزية بحدود كلمة حقيقية (\\b لا يكفي مع العربية). تُطابق على النص المطبّع."""
    alts = "|".join(sorted((re.escape(normalize_arabic(w.lower())) for w in words), key=len, reverse=True))
    return re.compile(rf"(?<![\w])(?:و|ب|ل|ال|لل|وال|بال)?(?:{alts})(?![\w])")


SIGNALS = {
    "JOB": (3, ["وظيفة شاغرة", "وظائف شاغرة", "فرصة عمل", "فرص عمل", "فرصة وظيفية", "فرص وظيفية", "إعلان وظيفة", "وظيفة جديدة",
                "المسمى الوظيفي", "الوصف الوظيفي", "job title", "vacancy", "vacancies", "we are hiring", "job description",
                "position title", "duty station", "terms of reference", "توظيف", "للتوظيف", "تعلن عن حاجتها", "يعلن عن حاجته",
                "عن حاجتها لتوظيف", "مطلوب للعمل", "عن حاجته", "عن حاجتها", "عن احتياجها", "لشغل الوظائف",
                "عن توفر وظائف", "عن توفر فرصة عمل", "شاغرة", "شاغره"]),
    "JOB_WEAK": (1, ["مطلوب", "مطلوبة", "وظيفة", "وظائف", "الراتب", "السيرة الذاتية", "cv", "resume", "الشروط", "للعمل",
                     "سنوات خبرة", "خبرة", "دوام", "full-time", "part-time", "consultant", "officer", "coordinator",
                     "manager", "assistant", "engineer", "accountant", "محاسب", "مهندس", "مندوب", "سكرتير", "طبيب", "ممرض",
                     "معلم", "مدرس", "سائق", "عامل", "فني", "مشرف", "مدير", "موظف", "موظفة", "أخصائي", "اخصائي", "منسق"]),
    "INTERNSHIP": (3, ["internship", "intern", "interns", "تدريب ميداني", "تدريب عملي للخريجين", "برنامج التدريب للخريجين",
                       "متدرب", "متدربين", "متدربة", "تدريب مدفوع", "فرصة تدريب", "graduate trainee", "trainee"]),
    "TRAINING": (3, ["دورة", "دورات", "دورة تدريبية", "برنامج تدريبي", "ورشة عمل", "ورشة تدريبية", "دبلوم مهني", "الدبلوم",
                     "دبلوم", "كورس", "course", "workshop", "bootcamp", "training program", "التسجيل في دورة", "رسوم الدورة",
                     "شهادة معتمدة", "المقاعد محدودة", "مقاعد محدودة", "للحجز", "سجل الآن", "التسجيل مفتوح", "افتتاح الدورة"]),
    "SCHOLARSHIP": (4, ["منحة", "منحة دراسية", "منح دراسية", "المنح الدراسية", "برنامج المنح", "scholarship", "scholarships", "fully funded",
                        "ممولة بالكامل", "مقاعد مجانية للدراسة", "fellowship"]),
    "VOLUNTEERING": (3, ["تطوع", "تطوعي", "تطوعية", "متطوع", "متطوعين", "متطوعات", "volunteer", "volunteers", "volunteering",
                         "unv"]),
    "OTHER_OPPORTUNITY": (3, ["مسابقة", "مسابقات", "جائزة", "هاكاثون", "hackathon", "competition", "call for proposals",
                              "دعوة لتقديم مقترحات", "منحة صغيرة لمشاريع", "grant for", "conference", "مؤتمر", "قبول طلبات الالتحاق",
                              "فتح باب القبول", "القبول والتسجيل", "دعوة للتقديم", "برنامج القيادات"]),
    "NOT_RELEVANT": (4, ["مناقصة", "مناقصات", "tender", "tenders", "rfq", "request for quotation", "expression of interest",
                         "eoi", "توريد", "توريد وتركيب", "procurement", "supply and delivery", "invitation to bid",
                         "للبيع", "عروض", "عرض خاص", "خصم", "تخفيضات", "ريال فقط", "اشترك في القناة", "يرجى نشر",
                         "انضموا", "رابط القناة", "للتبادل", "تقبل الله", "رمضان كريم", "عيد مبارك", "مفقود", "فقدان",
                         "فقدت", "كشف بأسماء", "نتائج القبول", "أسماء المقبولين", "تهنئة", "عزاء", "استئجار", "للايجار",
                         "للإيجار", "خدماته", "نقدم لكم خدمات", "channel name was changed", "pinned", "النتيجة العامة",
                         "امتحانات القبول", "المفاضلة", "joinchat", "اشتركوا", "تابع قناتنا", "موجودات", "تم العثور",
                         "احذروا", "تحذير", "احتيال", "نصب", "روابط وهمية", "تحسين السيرة", "كتابة السيرة الذاتية",
                         "تصميم السيرة الذاتية", "نسبة التوافق"]),
}
PATTERNS = {k: (w, _rx(v)) for k, (w, v) in SIGNALS.items()}
# دلائل على أن «تدريب» داخل وصف وظيفة لا إعلان دورة
JOB_CONTEXT_TRAINING = _rx(["التدريب والتطوير", "تدريب الموظفين", "احتياجات تدريبية", "مدرب", "مدربة", "trainer",
                            "training officer", "capacity building"])

REMOTE = _rx(["عن بعد", "عن بُعد", "العمل عن بعد", "عمل عن بعد", "remote", "remotely", "work from home", "من المنزل",
              "freelance", "عمل حر", "اونلاين", "أونلاين", "online"])
HYBRID = _rx(["hybrid", "هجين", "مرن بين المكتب"])
FOREIGN = {
    "السعودية": ["السعودية", "المملكة العربية السعودية", "الرياض", "جدة", "الدمام", "مكة", "المدينة المنورة", "saudi", "ksa", "riyadh", "jeddah"],
    "الإمارات": ["الإمارات", "الامارات", "دبي", "أبوظبي", "ابوظبي", "الشارقة", "uae", "dubai", "abu dhabi"],
    "قطر": ["قطر", "الدوحة", "qatar", "doha"], "الكويت": ["الكويت", "kuwait"], "عُمان": ["سلطنة عمان", "مسقط", "oman", "muscat"],
    "البحرين": ["البحرين", "bahrain"], "مصر": ["مصر", "القاهرة", "egypt", "cairo"], "الأردن": ["الأردن", "الاردن", "jordan", "amman"],
    "تركيا": ["تركيا", "اسطنبول", "turkey", "istanbul"], "ماليزيا": ["ماليزيا", "malaysia"], "جيبوتي": ["جيبوتي", "djibouti"],
}
FOREIGN_RX = {c: _rx(v) for c, v in FOREIGN.items()}
YEMEN_WORD = _rx(["اليمن", "yemen", "يمني", "يمنية", "اليمنية"])


def channel_boilerplate(texts, min_share: float = 0.01, min_count: int = 5) -> set[str]:
    """أسطر تتكرر في منشورات القناة نفسها (توقيع/تذييل/وسوم ثابتة) — تُستبعد قبل التصنيف حتى لا تعطي إشارة زائفة
    (مثال حقيقي: تذييل «#وظائف #عمل #منح #دورات» في كل منشورات @forus_taiz). النص الخام لا يتغير."""
    from collections import Counter
    c = Counter()
    n = 0
    for t in texts:
        n += 1
        c.update({normalize_arabic(l.strip().lower()) for l in (t or "").splitlines() if len(l.strip()) >= 4})
    thr = max(min_count, int(n * min_share))
    return {l for l, k in c.items() if k >= thr and _signature_like(l)}


_PROMO = re.compile(r"(?:https?://|t\.me/|@\w|واتساب|واتس|تيليجرام|تلجرام|فيسبوك|فيس بوك|انستجرام|لينكد|شارك|تابع|انضم|"
                    r"اشترك|الاعجاب|قناه|قناة|لنشر|للاعلان|للإعلان|whatsapp|telegram|facebook|instagram|linkedin|follow|join)")


def _signature_like(line: str) -> bool:
    """سطر توقيع/ترويج (يُحذف إذا تكرر) — لا تُحذف الأسطر المتكررة ذات المحتوى مثل «المهام الوظيفية»."""
    words = line.split()
    tags = sum(w.startswith("#") for w in words)
    return bool(_PROMO.search(line)) or (words and tags / len(words) >= 0.6) or not re.search(r"[\w\u0600-\u06FF]{3}", line)


def strip_boilerplate(text: str, bp: set[str]) -> str:
    if not bp:
        return text or ""
    return "\n".join(l for l in (text or "").splitlines() if normalize_arabic(l.strip().lower()) not in bp)


def _hits(rx: re.Pattern, norm: str) -> list[str]:
    return list(dict.fromkeys(m.group(0) for m in rx.finditer(norm)))


JOB_TEMPLATES = ("hr_yemen_labelled", "yemenhr_repost", "yemengate")
PLATFORM_HEADER = re.compile(r"Yemen\s*HR[^\n]{0,15}?Jobs\s*(?:&|and)\s*Tenders(?:\s+Platform|\s+in\s+Yemen)?|Jobs\s*(?:&|and)\s*Tenders", re.I)


def classify_type(text: str, fields: dict, channel: str, links: str = "") -> dict:
    """نوع الفرصة. الرأس (أول 250 حرفاً) يُضاعف وزنه لأن النوع يُعلن عادة في البداية."""
    if not (text or "").strip():
        return {"opportunity_type": "", "type_confidence": 0.0, "type_status": "excluded", "type_reason": "no_text (media only)"}
    tpl = fields.get("template", "")
    if re.search(r"yemenhr\.com/tenders|via yemen hr\s*-\s*tenders", (links + " " + text).lower()):
        return {"opportunity_type": "NOT_RELEVANT", "type_confidence": 0.95, "type_status": "accepted",
                "type_reason": "tender (YemenHR tenders link/feed)"}
    # ترويسة المنصة «Jobs & Tenders Platform/in Yemen» ليست دليلاً على مناقصة (تظهر في كل إعادة نشر من YemenHR)
    text = PLATFORM_HEADER.sub(" ", text)
    norm = normalize_arabic(text.lower())
    head = normalize_arabic(((fields.get("title") or "") + " " + text[:250]).lower())
    scores, why = {}, {}
    for k, (w, rx) in PATTERNS.items():
        h_all, h_head = _hits(rx, norm), _hits(rx, head)
        if k == "TRAINING" and h_all and JOB_CONTEXT_TRAINING.search(norm) and not h_head:
            h_all = []                      # «التدريب والتطوير» في وصف وظيفة
        sc = w * (len(h_all) + len(h_head))
        if sc:
            scores[k], why[k] = sc, h_head + [x for x in h_all if x not in h_head]
    job = scores.pop("JOB", 0) + min(scores.pop("JOB_WEAK", 0), 6)
    why_job = why.get("JOB", []) + why.get("JOB_WEAK", [])
    if tpl in JOB_TEMPLATES:
        job += 6
        why_job = [f"template:{tpl}"] + why_job
    if job:
        scores["JOB"], why["JOB"] = job, why_job
    if not scores:
        return {"opportunity_type": "NOT_RELEVANT", "type_confidence": 0.55, "type_status": "review",
                "type_reason": "no opportunity keywords"}
    # المتدرب/الإنترن داخل قالب وظيفة = فرصة تدريب عملي (INTERNSHIP) لا وظيفة
    if "INTERNSHIP" in scores and re.search(r"(?<![\w])(?:intern|internship|متدرب|trainee)", head):
        scores["INTERNSHIP"] += scores.get("JOB", 0)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    top, s1 = ranked[0]
    # قالب إعلان وظيفة («وظيفة جديدة»/حقول المسمى والجهة) يحسم النوع ما لم يظهر نوع آخر في العنوان/الرأس نفسه:
    # مثال حقيقي: «انستراكتور كاتب محتوى» في @TheYemenGate وظيفة مدرّب، وكلمات «كورس/دورات» في وصفها لا تجعلها دورة
    if tpl in JOB_TEMPLATES and top in ("TRAINING", "SCHOLARSHIP", "OTHER_OPPORTUNITY") and "JOB" in scores \
            and not _hits(PATTERNS[top][1], head):
        why["JOB"] = why["JOB"] + [f"template overrides body keywords of {top}"]
        scores["JOB"] = s1 + 1
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        top, s1 = ranked[0]
    s2 = ranked[1][1] if len(ranked) > 1 else 0
    margin = s1 - s2
    if tpl in JOB_TEMPLATES and top in ("JOB", "INTERNSHIP", "VOLUNTEERING", "NOT_RELEVANT"):
        conf = 0.92 if margin >= 3 else 0.75
    elif s1 >= 6 and margin >= 4:
        conf = 0.85
    elif s1 >= 4 and margin >= 2:
        conf = 0.72
    elif s1 >= 2 and margin >= 1:
        conf = 0.6
    else:
        conf = 0.45
    reason = f"{top}: {', '.join(why[top][:6])}" + (f" | ثانياً {ranked[1][0]}={s2}" if s2 else "")
    return {"opportunity_type": top, "type_confidence": conf, "type_status": "accepted" if conf >= ACCEPT_AT else "review",
            "type_reason": reason}


def classify_scope(text: str, fields: dict) -> dict:
    norm = normalize_arabic((text or "").lower())
    loc = fields.get("location_text") or ""
    loc_norm = normalize_arabic(loc.lower())
    remote = _hits(REMOTE, normalize_arabic((fields.get("title", "") + " " + loc + " " + (text or "")[:300]).lower()))
    hybrid = _hits(HYBRID, norm)
    wlt = "hybrid" if hybrid else "remote" if remote else "onsite" if loc else "unknown"
    gov_loc = match_governorate(loc)
    foreign_loc = [c for c, rx in FOREIGN_RX.items() if rx.search(loc_norm)]
    gov_text = match_governorate(text or "")
    foreign_text = [c for c, rx in FOREIGN_RX.items() if rx.search(norm)]
    yemen_word = bool(YEMEN_WORD.search(loc_norm)) or bool(YEMEN_WORD.search(norm))
    phone = bool(PHONE.search(text or ""))
    country, gov = "", gov_loc or ""
    if remote and not gov_loc:
        r = ("REMOTE_GLOBAL", 0.85 if len(remote) > 1 or "عن بعد" in " ".join(remote) else 0.7, f"remote: {', '.join(remote[:3])}")
    elif gov_loc or (YEMEN_WORD.search(loc_norm) and not foreign_loc):
        r, country = ("YEMEN", 0.95, f"location field: {loc}"), "Yemen"
    elif foreign_loc:
        r, country = ("OUTSIDE_YEMEN", 0.9, f"location field: {loc}"), foreign_loc[0]
    elif gov_text and not foreign_text:
        r, country, gov = ("YEMEN", 0.8, f"governorate in text: {gov_text}"), "Yemen", gov_text
    elif foreign_text and not gov_text and not yemen_word:
        r, country = ("OUTSIDE_YEMEN", 0.75, f"country in text: {', '.join(foreign_text[:2])}"), foreign_text[0]
    elif gov_text and foreign_text:
        r = ("UNKNOWN", 0.4, f"mixed: {gov_text} + {', '.join(foreign_text[:2])}")
    elif yemen_word or phone:
        r, country = ("YEMEN", 0.65, "Yemen mention" if yemen_word else "Yemeni phone number only"), "Yemen"
    else:
        r = ("UNKNOWN", 0.3, "no location signal")
    scope, conf, reason = r
    return {"geographic_scope": scope, "scope_confidence": conf, "scope_status": "accepted" if conf >= ACCEPT_AT else "review",
            "scope_reason": reason, "work_location_type": wlt, "country": country, "governorate": gov}


# ------------------------------------------------------------------ النطاق الجغرافي التفصيلي (لمهمة التصنيف B)
SCOPE_DETAILS = ["LOCAL", "SINGLE_GOVERNORATE", "MULTI_GOVERNORATE", "NATIONWIDE", "YEMEN_UNSPECIFIED", "INTERNATIONAL", "REMOTE",
                 "UNKNOWN"]
SCOPE_DETAIL_AR = {"LOCAL": "محلي (مديرية/مدينة داخل محافظة)", "SINGLE_GOVERNORATE": "محافظة محددة",
                   "MULTI_GOVERNORATE": "عدة محافظات", "NATIONWIDE": "اليمن بالكامل (نص صريح)",
                   "YEMEN_UNSPECIFIED": "داخل اليمن دون تحديد محافظة", "INTERNATIONAL": "دولي (خارج اليمن)",
                   "REMOTE": "عن بُعد", "UNKNOWN": "غير معروف"}
NATIONWIDE_RX = _rx(["جميع المحافظات", "كافة المحافظات", "مختلف المحافظات", "جميع محافظات الجمهورية", "كل المحافظات",
                     "all governorates", "nationwide", "across yemen", "various governorates", "multiple locations"])
SUBGOV_RX = re.compile(r"مديري[ةه]|عزل[ةه]|مدين[ةه]|حي\s|شارع|جول[ةه]|سوق|مخيم|قري[ةه]|district|city|camp|village", re.I)


# مديريات/مدن معروفة → محافظتها (مستوى محلي). قائمة قصيرة لأشهر المواقع في الإعلانات؛ غير الموجود يبقى غير مُطابق
DISTRICTS = {"المخا": "تعز", "al-mukha": "تعز", "mokha": "تعز", "التربة": "تعز", "الحجرية": "تعز", "الحوبان": "تعز", "مقبنة": "تعز",
             "زنجبار": "أبين", "zinjibar": "أبين", "الغيضة": "المهرة", "al ghaydah": "المهرة", "الحوطة": "لحج", "طور الباحة": "لحج",
             "سعوان": "صنعاء", "حزيز": "صنعاء", "شارع حدة": "صنعاء", "السبعين": "صنعاء", "بني الحارث": "صنعاء", "خمر": "عمران",
             "حبيش": "إب", "يريم": "إب", "جبلة": "إب", "رداع": "البيضاء", "باجل": "الحديدة", "زبيد": "الحديدة", "حيس": "الحديدة",
             "الخوخة": "الحديدة", "قعطبة": "الضالع", "qataba": "الضالع", "قطابة": "الضالع", "دمت": "الضالع", "مودية": "أبين",
             "لودر": "أبين", "بيحان": "شبوة", "عبس": "حجة", "حرض": "حجة", "المعلا": "عدن", "كريتر": "عدن", "خور مكسر": "عدن",
             "الشيخ عثمان": "عدن", "المنصورة": "عدن", "البريقة": "عدن", "دار سعد": "عدن"}
_DIST_RX = [(re.compile(rf"(?<![\w]){re.escape(normalize_arabic(k.lower()))}(?![\w])"), v) for k, v in DISTRICTS.items()]


def all_governorates(text: str, with_districts: bool = True) -> list[str]:
    from preprocessing.schema import YEMEN_GOVERNORATES
    norm = normalize_arabic((text or "").lower())
    tokens = set(re.split(r"[\s,،/\-()\[\]:.'’]+", norm))
    found = []
    for aliases in YEMEN_GOVERNORATES.values():
        for a in aliases:
            na = normalize_arabic(a.lower())
            if (na in tokens) if len(na) <= 4 else re.search(rf"(?<![\w]){re.escape(na)}(?![\w])", norm):
                found.append(aliases[0])
                break
    if with_districts:
        for rx, gov in _DIST_RX:
            if rx.search(norm) and gov not in found:
                found.append(gov)
    return found


def has_district(text: str) -> bool:
    norm = normalize_arabic((text or "").lower())
    return any(rx.search(norm) for rx, _ in _DIST_RX)


def scope_detail(scope: dict, fields: dict, text: str, own_text: str | None = None) -> dict:
    """تفصيل النطاق: محلي/محافظة/عدة محافظات/اليمن كاملاً/دولي/عن بعد. منطقة البحث = حقل الموقع + المسمى + أول 300 حرف
    (لا الوصف الطويل: تعريف المنظمات يذكر محافظات عملها لا مكان الوظيفة)."""
    loc = fields.get("location_text") or ""
    loc = "" if re.fullmatch(r"\s*(?:غير محدد|غير محددة|n/?a|-|—|tbd|not specified)\s*", loc, re.I) else loc
    # وظيفة مقسّمة من منشور: محافظتها في سطرها هي (own_text) تتقدم على مدينة الترويسة العامة
    zone_src = own_text if own_text and all_governorates(own_text) else text
    zone = f"{loc}\n{fields.get('title', '')}\n{(zone_src or '')[:300]}"
    gs = scope["geographic_scope"]
    if gs == "REMOTE_GLOBAL":
        return {"scope_detail": "REMOTE", "governorates_all": None, "scope_detail_reason": scope["scope_reason"]}
    if gs == "OUTSIDE_YEMEN":
        return {"scope_detail": "INTERNATIONAL", "governorates_all": None, "scope_detail_reason": scope["scope_reason"]}
    govs_loc, govs_zone = all_governorates(loc), all_governorates(zone)
    if NATIONWIDE_RX.search(normalize_arabic(zone.lower())):
        return {"scope_detail": "NATIONWIDE", "governorates_all": ";".join(govs_zone) or None,
                "scope_detail_reason": "explicit nationwide wording"}
    # حقل موقع موجود لكنه غير مُطابق → لا نأخذ محافظة من نص آخر (مثال حقيقي: «Al-Dhala'e» صار «عدن» من وسم في النص)
    govs = govs_loc if loc.strip() else govs_zone
    if not govs and scope.get("governorate"):
        if loc.strip():   # موقع فرعي غير معروف في القائمة (مثل «الحصبة») + محافظة مذكورة في النص → محلي، بسبب صريح للمراجعة
            return {"scope_detail": "LOCAL", "governorates_all": scope["governorate"],
                    "scope_detail_reason": f"unlisted sub-area «{loc[:40]}» + governorate in text: {scope['governorate']}"}
        govs = [scope["governorate"]]
    if len(govs) >= 2:
        return {"scope_detail": "MULTI_GOVERNORATE", "governorates_all": ";".join(govs),
                "scope_detail_reason": f"{len(govs)} governorates: {', '.join(govs)}"}
    if len(govs) == 1:
        parts = [p.strip() for p in re.split(r"\s[-–]\s|[–/،,|]", loc) if p.strip()]    # «Al-Dhala'e» اسم واحد
        extra = [p for p in parts if not all_governorates(p, with_districts=False)
                 and not re.fullmatch(r"(?:اليمن|yemen|محافظ[ةه].*|.*governorate)", p, re.I)]
        local = bool(SUBGOV_RX.search(loc)) or bool(extra) or has_district(loc)
        return {"scope_detail": "LOCAL" if local else "SINGLE_GOVERNORATE", "governorates_all": govs[0],
                "scope_detail_reason": f"{govs[0]}" + (f" + sub-area: {loc}" if local else "")}
    if gs == "YEMEN":       # «Location: Yemen» أو رقم يمني فقط: داخل اليمن، المحافظة غير مذكورة — ليس «اليمن بالكامل»
        return {"scope_detail": "YEMEN_UNSPECIFIED", "governorates_all": None, "scope_detail_reason": scope["scope_reason"]}
    return {"scope_detail": "UNKNOWN", "governorates_all": None, "scope_detail_reason": scope["scope_reason"]}


def reconcile_scope(scope: dict, detail: dict) -> dict:
    """تفصيل يمني صريح (محافظات/جميع المحافظات) يجعل النطاق الرئيسي YEMEN إن كان UNKNOWN."""
    if scope["geographic_scope"] == "UNKNOWN" and detail["scope_detail"] in ("LOCAL", "SINGLE_GOVERNORATE", "MULTI_GOVERNORATE", "NATIONWIDE"):
        scope = {**scope, "geographic_scope": "YEMEN", "country": "Yemen", "scope_confidence": 0.75, "scope_status": "accepted",
                 "scope_reason": scope["scope_reason"] + " → " + detail["scope_detail_reason"]}
    return scope
