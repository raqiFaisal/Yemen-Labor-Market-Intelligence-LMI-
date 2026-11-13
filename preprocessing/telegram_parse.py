"""تحليل منشورات Telegram إلى حقول (المرحلة 5). قواعد مبنية على صيغ مرصودة في عينة حقيقية من القنوات الخمس.

كل حقل مستخرج = نص موجود فعلاً في المنشور (لا قيم مستنتجة). ما لا يوجد يبقى فارغاً.
القوالب المرصودة:
  @hr_yemen       قالب مرقّم: «📋 المسمى الوظيفي / 🏢 الجهة / 📍 الموقع / 📅 تاريخ النشر / ⏰ آخر موعد / 📋 الوصف الوظيفي»
                  + إعادة نشر YemenHR: «Yemen HR - …\\n<المسمى>\\n#yemenhr\\n\\n<الموقع>\\n\\nsource\\n(author: <الجهة>)»
                    أو «<المسمى>\\n\\nvia Yemen HR - Jobs (author: <الجهة>)»
  @TheYemenGate   «وظيفة جديدة … YemenHR\\n<المسمى>\\n<المدينة>\\n<رابط>\\n#وسوم» أو «وظيفة جديدة … عمل عن بعد\\n<المسمى>\\n<الوصف>»
  @job_yemen / @jobsforyemeni   نص حر، وكثيراً قوائم مرقّمة بعدة وظائف في منشور واحد («1- …»، «1⃣ …»)
  @forus_taiz     إعلانات مختلطة (دورات، عروض تجارية، تنبيهات) — الوظائف أقلية
"""
from __future__ import annotations

import re
from datetime import datetime

from preprocessing.schema import YEMEN_GOVERNORATES, normalize_arabic
from scripts.scrapers.base import parse_date_text

EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️⃣‍←-⇿⌀-⏿━─_*~•●▪◾◽☆★✪✓✔➖➡⬅]+")
URL = re.compile(r"https?://\S+|www\.\S+|t\.me/\S+")
HASHTAG = re.compile(r"#([\w؀-ۿ]+)")
PHONE = re.compile(r"(?:\+?967[\s-]?)?\b7[01378]\d{7}\b")
SEP_LINE = re.compile(r"^[\s━─\-_=*•.]{5,}$")
FOOTER = re.compile(r"(قناة وظائف اليمن|تابعونا على|نتمنى لكم التوفيق|t\.me/hr_yemen)")

HR_LABELS = {
    "title": r"المسمى الوظيفي", "organization": r"الجهة", "location": r"الموقع",
    "published_text": r"تاريخ النشر", "deadline_text": r"آخر موعد للتقديم|اخر موعد للتقديم|آخر موعد",
    "description": r"الوصف الوظيفي", "how_to_apply": r"كيفية التقديم",
}
HR_ANY_LABEL = re.compile(r"^\W*(" + "|".join(HR_LABELS.values()) + r")\s*[:：]?", re.M)
YHR_REPOST = re.compile(r"Yemen HR\s*-\s*Yemen'?s Premier Jobs (?:&|and) Tenders Platform", re.I)
VIA_YHR = re.compile(r"via\s+Yemen HR[^\n(]*(?:\(author:\s*([^)]+)\))?", re.I)
AUTHOR = re.compile(r"\(author:\s*([^)]+)\)", re.I)
# تغذية YemenHR القديمة (RSS/Facebook): ترويسة ثم «منذ X» ثم المسمى ثم الموقع ثم تاريخ ثم الفئة — معظمها مناقصات
FEED_NOISE = re.compile(r"^(?:Yemen HR\b.*|\d+\s+(?:minutes?|hours?|days?|weeks?)\s+ago|Posted\b.*ago|source|\.?[\w-]+\s*\{.*|.*;\s*)$", re.I)
FEED_HEAD = re.compile(r"Yemen HR\s*(?:\||:|-)\s*Jobs\s*(?:&|and)\s*Tenders in Yemen|Yemen HR \((?:RSS|Facebook)\)|^\d+\s+(?:minutes?|hours?)\s+ago", re.I | re.M)
GATE_HEAD = re.compile(r"^\s*وظيفة جديدة[^\n]*", re.M)

DEADLINE_RX = re.compile(r"(?:آخر موعد(?: للتقديم)?|اخر موعد(?: للتقديم)?|الموعد النهائي|يستمر التسجيل حتى|التسجيل مستمر حتى|"
                         r"Closing date|Deadline|Application deadline|Apply before)\s*[:：\-–]?\s*([^\n]{3,60})", re.I)
PUBLISHED_RX = re.compile(r"(?:تاريخ النشر|بتاريخ|Posted on|Date posted|Publish(?:ed)? date)\s*[:：\-–]?\s*([^\n]{3,40})", re.I)
TITLE_INLINE = [
    re.compile(r"(?<![\w-])(?:Job Title|Position Title|Position|المسمى الوظيفي|الاسم الوظيفي|مسمى الوظيفة|اسم الوظيفة)\s*[:：]\s*([^\n;{}]{2,120})(?:\n|$)", re.I),
    re.compile(r"(?<![\w])الوظيفة\s*[:：]\s*([^\n]{2,80})"),
    re.compile(r"إعلان وظيفة(?: شاغرة)?\s*[:：\-–]\s*([^\n]{2,80})"),
    re.compile(r"(?<![\w])(?:تبحث عن|بحاجة إلى|بحاجة الى|بحاجه الى|تطلب|يطلب)[ \t]+\(?\s*([^\n)]{2,60})\)?"),
    re.compile(r"^\W*مطلوب(?:ة)?[ \t]+(?!للعمل)([^\n#]{2,70})", re.M),
]
LOCATION_INLINE = re.compile(r"(?<![\w])(?:Location|Duty Station|مكان العمل|الموقع|المكان|المحافظة|مقر العمل)\s*[:：➖\-–]\s*([^\n]{2,60})", re.I)
ORG_INLINE = re.compile(r"(?<![\w])(?:Organization|Organisation|Company|Employer|الجهة|الشركة|شركة|المنظمة|جهة العمل)\s*[:：]\s*([^\n]{2,80})", re.I)
ANNOUNCE_ORG = re.compile(r"(?:تعلن|يعلن|يسر|تسر)\s+(?!إحدى|احدى|أحد|احد)([^\n]{3,60}?)\s+(?:عن|في|بالإعلان|الإعلان)")
GENERIC_ORG = {"الشركة", "شركة", "الجهة", "المنظمة", "المؤسسة", "إحدى الشركات", "احدى الشركات", "غير محدد"}
SALARY_RX = re.compile(r"(?:الراتب|المرتب|Salary)\s*[:：➖\-–]?\s*([^\n]{2,60})", re.I)

NUM_ITEM = re.compile(r"^\s*(?:([0-9٠-٩]{1,2})\s*[-–.)\]]\s*|([0-9])️?⃣\s*)(.+)$")
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
SPLIT_HEADER = re.compile(r"(الوظائف التالية|وظائف شاغرة|وظائف شاغره|عن توفر الوظائف|فرص وظيفية|مطلوب للعمل)")
NOT_ROLE_START = re.compile(r"^(?:الاسم الوظيفي|الهدف|المهام|المسؤوليات|الشروط|المؤهل|المتطلبات|المزايا|طريقة|للتواصل|"
                            r"Responsibilities|Requirements|Qualifications)", re.I)


def clean_line(s: str) -> str:
    s = EMOJI.sub(" ", s)
    s = re.sub(r"\[(?:email|link|whatsapp)\]", " ", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip(" :：-–|*_")


def body_text(text: str) -> str:
    """النص للتحليل والنماذج: بلا روابط ووسوم وتذييل القناة وخطوط الفصل. النص الأصلي يبقى في raw_text."""
    lines = []
    for ln in (text or "").splitlines():
        if SEP_LINE.match(ln) or FOOTER.search(ln):
            continue
        ln = HASHTAG.sub(" ", URL.sub(" ", ln))
        ln = clean_line(ln)
        if ln:
            lines.append(ln)
    return "\n".join(dict.fromkeys(lines))


def match_governorate(text: str) -> str | None:
    """اسم المحافظة اليمنية (العربي المعتمد) إن وُجد صراحة في النص، وإلا None."""
    if not text:
        return None
    norm = normalize_arabic(text.lower())
    tokens = set(re.split(r"[\s,،/\-()\[\]:.'’]+", norm))
    for aliases in YEMEN_GOVERNORATES.values():
        for a in aliases:
            na = normalize_arabic(a.lower())
            hit = (na in tokens) if len(na) <= 4 else (re.search(rf"(?<![\w]){re.escape(na)}(?![\w])", norm) is not None)
            if hit:
                return aliases[0]
    return None


def _date(s: str, ref: datetime | None):
    d = parse_date_text(s or "", now=ref)
    return d.isoformat() if d else ""


def _labelled_block(text: str) -> dict:
    """قالب @hr_yemen: القيمة على نفس السطر أو الأسطر التالية حتى التسمية/الفاصل التالي."""
    out = {}
    marks = [(m.start(), m.end(), m.group(1)) for m in HR_ANY_LABEL.finditer(text)]
    for i, (s, e, lab) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        val_lines = []
        for ln in text[e:end].splitlines():
            if SEP_LINE.match(ln) or FOOTER.search(ln):
                break
            if clean_line(ln):
                val_lines.append(clean_line(ln))
        key = next(k for k, v in HR_LABELS.items() if re.fullmatch(v, lab))
        out.setdefault(key, "\n".join(val_lines).strip())
    return out


def parse_post(text: str, channel: str, message_date: str = "") -> dict:
    """الحقول المستخرجة + template (القالب الذي طابق). القيم نصوص من المنشور كما هي بعد إزالة الرموز فقط."""
    ref = None
    try:
        ref = datetime.fromisoformat(message_date.replace("Z", "+00:00")).replace(tzinfo=None) if message_date else None
    except ValueError:
        pass
    text = text or ""
    f = {"template": "free_text", "title": "", "organization": "", "location_text": "", "published_text": "",
         "deadline_text": "", "description": "", "salary": "", "hashtags": " ".join(dict.fromkeys(HASHTAG.findall(text)))}
    if len({m.group(1) for m in HR_ANY_LABEL.finditer(text)}) >= 3:
        f["template"] = "hr_yemen_labelled"
        for k, v in _labelled_block(text).items():
            if k != "description":                       # الحقول القصيرة سطر واحد (لا يلتصق بها تاريخ أو سطر لاحق)
                v = v.split("\n")[0].strip()
            if k in f:
                f[k] = v
            elif k == "location":
                f["location_text"] = v
    elif YHR_REPOST.search(text) or VIA_YHR.search(text):
        f["template"] = "yemenhr_repost"
        lines = [clean_line(l) for l in text.splitlines()]
        lines = [l for l in lines if l and not YHR_REPOST.search(l) and l.lower() not in ("source", "#yemenhr")
                 and not l.lower().startswith("via yemen hr") and not l.startswith("(author")]
        if lines:
            f["title"] = lines[0]
        if len(lines) > 1 and len(lines[1]) <= 40:
            f["location_text"] = lines[1]
        m = AUTHOR.search(text)
        f["organization"] = m.group(1).strip() if m else ""
    elif FEED_HEAD.search(text):
        f["template"] = "yemenhr_feed"
        lines = [clean_line(URL.sub(" ", l)) for l in text.splitlines()]
        lines = [l for l in lines if l and not FEED_NOISE.match(l)]
        if lines:
            f["title"] = lines[0][:150]
        for l in lines[1:4]:
            if match_governorate(l) or l.lower() in ("yemen", "اليمن", "remote"):
                f["location_text"] = l
                break
        for l in lines[1:6]:
            d = parse_date_text(l)
            if d:
                f["published_text"] = l
                break
        f["description"] = "\n".join(lines[1:])
    elif GATE_HEAD.search(text):
        f["template"] = "yemengate"
        lines = [clean_line(HASHTAG.sub(" ", URL.sub(" ", l))) for l in text.splitlines()]
        lines = [l for l in lines if l]
        head = lines[0] if lines else ""
        rest = lines[1:]
        if rest:
            t = rest[0]
            if ":" in t and len(t.split(":", 1)[0]) <= 40:          # «جريدة الرحلة : أخصائية تسويق»
                f["organization"], t = (x.strip() for x in t.split(":", 1))
            f["title"] = t
        if "YemenHR" in head and len(rest) > 1 and len(rest[1]) <= 30 and ":" not in rest[1]:
            f["location_text"] = rest[1]
        m = re.search(r"^\s*Posted\s*:\s*(.+)$", text, re.M)
        f["published_text"] = clean_line(m.group(1)) if m else ""
        f["description"] = "\n".join(rest[1:]) if "YemenHR" not in head else ""
    # حقول عامة لكل القوالب (لا تستبدل ما وجده القالب)
    if not f["title"]:
        for rx in TITLE_INLINE:
            m = rx.search(text)
            t = clean_line(m.group(1)) if m else ""
            if len(t) >= 3 and not t.endswith(("؟", "?")) and not t.startswith(("ات ", "عمل")):
                f["title"] = t[:120]
                break
    if f["template"] == "free_text" and f["title"]:
        f["title"] = re.split(r"\s+(?:للعمل|براتب|الراتب|بدوام|بشرط|لدى|في)(?![\w])", f["title"])[0].strip(" ،,.-")
    if f["organization"] in GENERIC_ORG:
        f["organization"] = ""
    if not f["organization"]:
        m = ORG_INLINE.search(text) or ANNOUNCE_ORG.search(text)
        if m:
            f["organization"] = clean_line(m.group(1))[:80]
    if not f["location_text"]:
        m = LOCATION_INLINE.search(text)
        if m:
            f["location_text"] = clean_line(m.group(1))[:60]
    if not f["deadline_text"]:
        m = DEADLINE_RX.search(text)
        f["deadline_text"] = clean_line(m.group(1)) if m else ""
    if not f["published_text"]:
        m = PUBLISHED_RX.search(text)
        f["published_text"] = clean_line(m.group(1)) if m else ""
    if f["organization"] in GENERIC_ORG:
        f["organization"] = ""
    m = SALARY_RX.search(text)
    f["salary"] = clean_line(m.group(1)) if m else ""
    f["published_date"] = _date(f["published_text"], ref)
    f["application_deadline"] = _date(f["deadline_text"], ref)
    if not f["description"]:
        f["description"] = body_text(text)
    return f


def split_positions(text: str, channel: str) -> list[dict]:
    """منشور بعدة وظائف مرقّمة → قائمة مقاطع [{n, role_line, text}]. يُرجع [] إن لم تتحقق الشروط كلها:
    قناة نص حر، ترويسة تدل على قائمة وظائف، ≥2 بند مرقّم متتالٍ يبدأ من 1، وسطر كل بند قصير ولا يبدأ بعنوان قسم."""
    if channel in ("hr_yemen", "TheYemenGate") or not SPLIT_HEADER.search(text or ""):
        return []
    lines = text.splitlines()
    starts = []
    for i, ln in enumerate(lines):
        m = NUM_ITEM.match(ln)
        if m:
            n = int((m.group(1) or m.group(2)).translate(AR_DIGITS))
            role = clean_line(m.group(3))
            starts.append((i, n, role))
    seq, expect = [], 1
    for i, n, role in starts:
        if n == expect:
            seq.append((i, n, role))
            expect += 1
    if len(seq) < 2 or any(len(r) > 70 or len(r) < 2 or NOT_ROLE_START.match(r) for _, _, r in seq):
        return []
    header = "\n".join(lines[:seq[0][0]])
    out = []
    for k, (i, n, role) in enumerate(seq):
        end = seq[k + 1][0] if k + 1 < len(seq) else len(lines)
        seg = "\n".join(lines[i:end])
        out.append({"n": n, "role_line": role, "text": seg, "header": header})
    return out
