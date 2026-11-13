"""جامع YemenHR (https://yemenhr.com).

البنية المرصودة للموقع (نوفمبر 2026):
  • صفحة القائمة /jobs: جدول بأعمدة Posted | Organization | Title | Location | Deadline، والتاريخ بصيغة "04 Nov, 26".
  • رابط التفاصيل: /jobs/{slug}-{8 hex}  ← الجزء الست عشري هو معرف الإعلان (source_job_id).
  • ترقيم الصفحات: "First Prev 1 of N pages Next" — نتبع رابط Next الفعلي بدل افتراض صيغة ?page=.
  • التصنيفات: روابط /categories/{slug} بنص "Development 21" — نزورها لربط كل إعلان بتصنيف الموقع (تسمية حقيقية).
  • صفحة الإعلان: H1 = المسمى، ثم الجهة ثم الموقع، "Posted:" و "Deadline:"، وقسم "Job Description /الوصف الوظيفي"
    حتى "How to Apply /كيفية التقديم".
الموقع يعرض الإعلانات النشطة فقط، لذلك يُبنى التاريخ بالتشغيل الدوري (الاستكمال يتخطى ما جُمع).
"""
from __future__ import annotations

import re
from datetime import datetime
from urllib.parse import urljoin, urlparse

import pandas as pd
from bs4 import BeautifulSoup

from preprocessing.schema import YEMEN_GOVERNORATES, normalize_governorate
from scripts.scrapers.base import (PARSER_VERSION, Fetcher, RunStats, SourceStore, make_logger, parse_date_text,
                                   summarize)

SOURCE = "yemenhr"
BASE = "https://yemenhr.com"
LIST_URL = BASE + "/jobs"
PAGE_TEMPLATE = BASE + "/jobs?page={p}"  # احتياطي فقط إن لم يوجد رابط Next
# صيغتا رابط الإعلان المرصودتان فعلياً (تم التحقق من الموقع):
#   صفحة /jobs:            /jobs/finance-officer-ado-aden-1ef02a5e
#   صفحات /categories/*:   /jobs/1ef02a5e-ed38-4e8c-bc95-617a6c6ef025   (UUID كامل)
# معرف الإعلان المشترك = أول 8 خانات ست عشرية من الـ UUID = لاحقة الرابط المختصر.
SLUG_RX = re.compile(r"^/jobs/[a-z0-9\-%]+-([0-9a-f]{8})/?$", re.I)
UUID_RX = re.compile(r"^/jobs/([0-9a-f]{8})-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/?$", re.I)


class _DetailRx:
    """توافق مع الاستخدام السابق DETAIL_RX.match(path).group(1) لكلتا الصيغتين."""

    @staticmethod
    def match(path):
        return UUID_RX.match(path) or SLUG_RX.match(path)


DETAIL_RX = _DetailRx()


def job_key(url: str) -> str | None:
    m = DETAIL_RX.match(urlparse(url).path)
    return m.group(1).lower() if m else None
CATEGORY_RX = re.compile(r"^/categories/[a-z0-9\-%]+/?$", re.I)
GOVS = {v[0] for v in YEMEN_GOVERNORATES.values()}

EMPLOYMENT_PATTERNS = [
    ("Internship", r"\bintern(ship)?\b|متدرب|تدريب عملي"),
    ("Daily Worker", r"\bdaily worker\b|عامل يومي|بالأجر اليومي"),
    ("Consultancy", r"\bconsultan(t|cy)\b|استشاري"),
    ("Part-time", r"\bpart[\s-]?time\b|دوام جزئي"),
    ("Full-time", r"\bfull[\s-]?time\b|دوام كامل"),
    ("Volunteer", r"\bvolunteer\b|متطوع"),
    ("Temporary", r"\btemporary\b|مؤقت"),
    ("Contract", r"\b(fixed[\s-]term|contract type|type of contract|contract duration)\b|نوع العقد|مدة العقد"),
]
LEVEL_PATTERNS = [("Senior", r"\bsenior\b|\bsr\.?\b|أول\b|اول\b"), ("Junior", r"\bjunior\b|مبتدئ"),
                  ("Lead/Manager", r"\b(lead|head of|manager|director|coordinator)\b|مدير|رئيس قسم|منسق"),
                  ("Entry/Intern", r"\bentry[\s-]level\b|\bintern\b|حديث التخرج|حديثي التخرج"),
                  ("Assistant", r"\bassistant\b|مساعد")]
EDU_PATTERNS = [("PhD", r"\bph\.?d\b|doctorate|دكتوراه"), ("Master", r"\bmaster'?s?\b|\bm\.?sc\b|\bmba\b|ماجستير"),
                ("Bachelor", r"\bbachelor'?s?\b|\bb\.?sc\b|\bb\.?a\b|university degree|بكالوريوس|شهادة جامعية"),
                ("Diploma", r"\bdiploma\b|دبلوم"), ("Secondary", r"\bhigh school\b|secondary school|ثانوية")]
EXP_RX = [re.compile(r"(\d{1,2})\s*\+?\s*(?:-\s*\d{1,2}\s*)?(?:years?|yrs)\b[^.\n]{0,40}?experience", re.I),
          re.compile(r"experience[^.\n]{0,30}?(?:of|at least|minimum|min\.?)\s*(\d{1,2})\s*\+?\s*years?", re.I),
          re.compile(r"خبرة[^.\n]{0,25}?(\d{1,2})\s*(?:سنوات|سنة|سنه|أعوام|اعوام)"),
          re.compile(r"(\d{1,2})\s*(?:سنوات|سنة)\s*(?:من\s*)?(?:ال)?خبرة")]
REQ_HEAD = re.compile(r"^(requirements?|qualifications?|required skills|skills|competenc|experience and qualif|"
                      r"المتطلبات|المؤهلات|المهارات|الشروط)", re.I)
END_HEAD = re.compile(r"^(how to apply|important notes|related jobs|job application|كيفية التقديم|ملاحظات هامة)", re.I)
START_HEAD = re.compile(r"^(job description|الوصف الوظيفي)", re.I)


FOREIGN = {"saudi arabia": "Saudi Arabia", "السعودية": "Saudi Arabia", "ksa": "Saudi Arabia", "riyadh": "Saudi Arabia",
           "jeddah": "Saudi Arabia", "uae": "United Arab Emirates", "dubai": "United Arab Emirates",
           "abu dhabi": "United Arab Emirates", "الإمارات": "United Arab Emirates", "oman": "Oman", "عمان": "Oman",
           "qatar": "Qatar", "قطر": "Qatar", "kuwait": "Kuwait", "الكويت": "Kuwait", "egypt": "Egypt", "مصر": "Egypt",
           "jordan": "Jordan", "الأردن": "Jordan", "djibouti": "Djibouti", "جيبوتي": "Djibouti", "kenya": "Kenya"}


def infer_country(loc_raw: str, gov: str | None) -> str:
    """YemenHR ينشر أحياناً وظائف خارج اليمن (مثل Saudi Arabia) — لا تُحسب ضمن السوق اليمني."""
    if gov in GOVS:
        return "Yemen"
    low = (loc_raw or "").lower()
    for k, v in FOREIGN.items():
        if k in low:
            return v
    return "Yemen"   # مثل "Multiple Cities" أو مواقع يمنية غير معروفة للقاموس


def _clean_cat(t: str) -> str:
    return re.sub(r"\s*\(?\d+\)?\s*$", "", t).strip()


# ------------------------------------------------------------------ القائمة
def parse_listing(html: str) -> tuple[list[dict], str | None, list[tuple[str, str]], int | None]:
    """(rows, next_url, categories[(name,url)], total_reported)"""
    soup = BeautifulSoup(html, "html.parser")
    rows, seen = [], set()
    for tr in soup.find_all("tr"):
        a = next((x for x in tr.find_all("a", href=True) if DETAIL_RX.match(urlparse(urljoin(BASE, x["href"])).path)), None)
        if not a:
            continue
        url = urljoin(BASE, a["href"]).split("?")[0]
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        dates = [d for d in (parse_date_text(c) for c in cells) if d]
        title_i = next((i for i, c in enumerate(tr.find_all(["td", "th"])) if a in c.find_all("a")), None)
        org = cells[title_i - 1] if title_i and title_i >= 1 else ""
        loc = cells[title_i + 1] if title_i is not None and title_i + 1 < len(cells) else ""
        if url not in seen:
            seen.add(url)
            rows.append({"url": url, "key": job_key(url), "listing_title": a.get_text(" ", strip=True),
                         "listing_org": "" if parse_date_text(org) else org,
                         "listing_location": "" if parse_date_text(loc) else loc,
                         "listing_posted": dates[0].isoformat() if dates else "",
                         "listing_deadline": dates[-1].isoformat() if len(dates) > 1 else ""})
    # روابط بلا جدول (عرض الجوال) — نضيفها دون حقول القائمة
    for a in soup.find_all("a", href=True):
        url = urljoin(BASE, a["href"]).split("?")[0]
        if DETAIL_RX.match(urlparse(url).path) and url not in seen:
            seen.add(url)
            rows.append({"url": url, "key": job_key(url), "listing_title": a.get_text(" ", strip=True)})
    nxt = None
    for a in soup.find_all("a", href=True):
        if a.get("rel") == ["next"] or a.get_text(strip=True).lower() in ("next", "التالي", "›", "»"):
            h = a["href"].strip()
            if h and h != "#" and not h.startswith("javascript"):
                nxt = urljoin(BASE, h)
                break
    cats = []
    for a in soup.find_all("a", href=True):
        u = urljoin(BASE, a["href"]).split("?")[0]
        if CATEGORY_RX.match(urlparse(u).path):
            name = _clean_cat(a.get_text(" ", strip=True))
            if name and (name, u) not in cats:
                cats.append((name, u))
    m = re.search(r"Total\s*:?\s*(\d+)", soup.get_text(" ", strip=True))
    return rows, nxt, cats, (int(m.group(1)) if m else None)


# ------------------------------------------------------------------ التفاصيل
def _first(patterns, text):
    return next((name for name, rx in patterns if re.search(rx, text, re.I)), "")


def _all(patterns, text):
    return [name for name, rx in patterns if re.search(rx, text, re.I)]


def parse_detail(html: str, url: str, listing: dict | None = None) -> dict:
    """يرفع ValueError بسبب واضح إن فشل الاستخراج."""
    listing = listing or {}
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript", "svg", "form"]):
        t.decompose()
    h1 = soup.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else listing.get("listing_title", "")
    if not title:
        raise ValueError("no title (h1)")
    main = soup.find("main") or soup.body or soup
    lines = [l.strip() for l in main.get_text("\n").split("\n") if l.strip()]
    ti = next((i for i, l in enumerate(lines) if l == title), None)
    after = lines[ti + 1: ti + 6] if ti is not None else []
    after = [l for l in after if not re.match(r"^(new|posted|deadline)\b", l, re.I)]
    company = listing.get("listing_org") or (after[0] if after else "")
    loc_raw = listing.get("listing_location") or (after[1] if len(after) > 1 else "")

    def labeled(label):
        rx = re.compile(rf"^{label}\s*:?\s*(.*)$", re.I)
        for i, l in enumerate(lines):
            m = rx.match(l)
            if m:
                return m.group(1).strip() or (lines[i + 1] if i + 1 < len(lines) else "")
        return ""
    posted_raw, deadline_raw = labeled("posted"), labeled("deadline")
    posted = parse_date_text(posted_raw) or (pd.to_datetime(listing["listing_posted"]).date()
                                             if listing.get("listing_posted") else None)
    deadline = parse_date_text(deadline_raw) or (pd.to_datetime(listing["listing_deadline"]).date()
                                                 if listing.get("listing_deadline") else None)

    s = next((i for i, l in enumerate(lines) if START_HEAD.match(l)), None)
    if s is None:  # احتياطي: من بعد سطر الموعد حتى قسم التقديم
        s = next((i for i, l in enumerate(lines) if l.lower().startswith("deadline")), ti or 0)
    e = next((i for i, l in enumerate(lines) if i > s and END_HEAD.match(l)), len(lines))
    body_lines = lines[s + 1:e]
    description = "\n".join(body_lines).strip()
    if len(description) < 60:
        raise ValueError(f"description too short ({len(description)} chars) — page structure may have changed")
    r = next((i for i, l in enumerate(body_lines) if REQ_HEAD.match(l)), None)
    requirements = "\n".join(body_lines[r:]) if r is not None else ""
    apply_text = "\n".join(lines[e:e + 40]) if e < len(lines) else ""

    blob = f"{title}\n{description}"
    exp = None
    for rx in EXP_RX:
        m = rx.search(description)
        if m:
            exp = int(m.group(1))
            break
    gov = normalize_governorate(loc_raw)
    return {
        "source": SOURCE, "source_job_id": DETAIL_RX.match(urlparse(url).path).group(1), "url": url,
        "title": title, "company": company, "location_raw": loc_raw,
        "governorate": gov if gov in GOVS else "", "country": infer_country(loc_raw, gov),
        "posted_date": posted.isoformat() if posted else "", "posted_date_raw": posted_raw or listing.get("listing_posted", ""),
        "deadline": deadline.isoformat() if deadline else "", "deadline_raw": deadline_raw,
        "category": listing.get("category", ""),
        "employment_type": _first(EMPLOYMENT_PATTERNS, title) or _first(EMPLOYMENT_PATTERNS, description[:1500]),
        "experience_level": _first(LEVEL_PATTERNS, title),
        "experience_years": "" if exp is None else exp,
        "education": ";".join(_all(EDU_PATTERNS, requirements or description)),
        "requirements_text": requirements,
        "description": description,          # النص الأصلي لقسم الوصف كما هو
        "how_to_apply": apply_text[:2000],
        "lang": "ar" if len(re.findall(r"[؀-ۿ]", description)) > len(description) * 0.3 else
                ("mixed" if re.search(r"[؀-ۿ]", description) else "en"),
        "scraped_at": datetime.now().isoformat(timespec="seconds"), "parser_version": PARSER_VERSION,
    }


# ------------------------------------------------------------------ التشغيل
def run(max_pages: int = 50, limit: int = 0, delay: float = 3.0, categories: bool = True, retry_failed: bool = False,
        refresh: bool = False, debug: bool = False) -> RunStats:
    log = make_logger(SOURCE)
    st, store = RunStats(SOURCE), SourceStore(SOURCE)
    f = Fetcher(delay, log)
    done = set() if refresh else store.downloaded_keys()
    perm_failed = set() if retry_failed else store.permanent_failures()
    disc_old = store.load_discovered()
    known = set(disc_old["key"]) if "key" in disc_old else set(job_key(u) for u in disc_old.get("url", []))
    found: dict[str, dict] = {}
    cats, total, cat_counts = [], None, {}

    # 1) صفحات القائمة مع تتبع Next الحقيقي
    url, p, visited = LIST_URL, 1, set()
    while url and p <= max_pages and url not in visited:
        visited.add(url)
        html, why, code = f.get(url)
        st.pages_visited += 1
        if not html:
            st.pages_failed += 1
            store.log_failure(url, "listing", why, code)
            log.warning("فشل صفحة القائمة %s: %s", url, why)
            break
        rows, nxt, c, tot = parse_listing(html)
        total = total or tot
        cats = cats or c
        new = 0
        for r in rows:
            if r["key"] in found:
                st.duplicate_links += 1
            else:
                found[r["key"]] = r
                new += 1
        log.info("صفحة %d: %d رابط (%d جديد في هذا التشغيل) | الموقع يذكر الإجمالي: %s", p, len(rows), new, total)
        if new == 0:
            log.info("لا روابط جديدة — نهاية الترقيم.")
            break
        url = nxt or (PAGE_TEMPLATE.format(p=p + 1) if total and len(found) < total else None)
        p += 1

    # 2) التصنيفات: تسمية المصدر الحقيقية لكل إعلان
    if categories:
        for name, cu in cats:
            cu_page, cp, seen_pages = cu, 1, set()
            while cu_page and cp <= max_pages and cu_page not in seen_pages:
                seen_pages.add(cu_page)
                html, why, code = f.get(cu_page)
                st.pages_visited += 1
                if not html:
                    st.pages_failed += 1
                    store.log_failure(cu_page, "category", why, code)
                    break
                rows, nxt, _, _ = parse_listing(html)
                newc = 0
                for r in rows:
                    # المطابقة بمعرف الإعلان لا بالرابط (صفحات التصنيفات تستخدم رابط UUID مختلفاً)
                    if r["key"] not in found:
                        found[r["key"]] = r     # إعلان ظهر في التصنيف فقط — يُنزَّل برابطه كما هو
                    base = found[r["key"]]
                    prev = base.get("category", "")
                    if name not in prev.split(";"):
                        base["category"] = f"{prev};{name}".strip(";")
                        newc += 1
                cat_counts[name] = cat_counts.get(name, 0) + len(rows)
                if not rows:
                    store.log_failure(cu_page, "category", "category page parsed but 0 job links matched", code)
                if not rows or newc == 0:
                    break
                cu_page, cp = nxt, cp + 1
            log.info("تصنيف «%s»: %d إعلان مرتبط", name, sum(name in (v.get("category") or "").split(";") for v in found.values()))
        linked = sum(bool(v.get("category")) for v in found.values())
        st_cats = {"categories_found": len(cats), "jobs_with_category": linked}
        log.info("التصنيفات: %d تصنيفاً | إعلانات ارتبطت بتصنيف: %d من %d", len(cats), linked, len(found))
        if cats and linked == 0:
            log.error("⚠ لم يرتبط أي إعلان بتصنيف — بنية صفحات التصنيف تغيرت؟ راجع logs/yemenhr_failures.csv")

    st.links_discovered = len(found)
    st.links_new = len(set(found) - known)
    st.jobs_with_category = sum(bool(v.get("category")) for v in found.values())
    # حفظ الروابط المكتشفة (حالة فقط — ليست إعلانات)
    now = datetime.now().isoformat(timespec="seconds")
    disc = pd.DataFrame(found.values())
    if len(disc):
        disc["last_seen"] = now
        if len(disc_old):
            if "key" not in disc_old:
                disc_old["key"] = disc_old["url"].map(job_key)
            first = disc_old.drop_duplicates("key").set_index("key").get("first_seen", pd.Series(dtype=str))
            disc["first_seen"] = disc["key"].map(first).fillna(now)
            disc = pd.concat([disc, disc_old[~disc_old.key.isin(disc.key)]], ignore_index=True)
        else:
            disc["first_seen"] = now
        store.save_discovered(disc)

    # 2-ب) ربط التصنيفات بالإعلانات المحفوظة مسبقاً (استكمال دون إعادة تنزيل)
    upd = store.update_categories({k: v["category"] for k, v in found.items() if v.get("category")})
    if upd:
        log.info("حُدّث تصنيف %d إعلاناً محفوظاً سابقاً", upd)

    # 3) صفحات التفاصيل — يُحسب الإعلان منزّلاً فقط بعد نجاح الاستخراج والحفظ
    todo = [k for k in found if k not in done and found[k]["url"] not in perm_failed]
    st.already_downloaded = sum(k in done for k in found)
    if limit:
        todo = todo[:limit]
    existing = store.load_data()
    sig = set((existing["title"] + "|" + existing["company"] + "|" + existing["posted_date"]).fillna("")) \
        if len(existing) else set()
    batch = []
    for i, k in enumerate(todo, 1):
        u = found[k]["url"]
        st.details_attempted += 1
        html, why, code = f.get(u)
        if not html:
            st.failed += 1
            store.log_failure(u, "detail", why, code)
            log.warning("[%d/%d] ✖ %s — %s", i, len(todo), u, why)
            continue
        try:
            job = parse_detail(html, u, found[k])
        except Exception as e:  # noqa: BLE001
            st.failed += 1
            store.log_failure(u, "parse", f"{type(e).__name__}: {e}", code)
            log.warning("[%d/%d] ✖ تحليل %s — %s", i, len(todo), u, e)
            continue
        k = f"{job['title']}|{job['company']}|{job['posted_date']}"
        if k in sig:
            st.content_duplicates += 1
            job["possible_duplicate"] = "same title+company+posted_date as an existing record"
        sig.add(k)
        batch.append(job)
        st.downloaded_ok += 1
        if debug:
            log.info("[%d/%d] ✔ %s | %s | %s | نشر %s | تصنيف %s", i, len(todo), job["title"][:50], job["company"][:25],
                     job["location_raw"], job["posted_date"], job["category"] or "—")
        if len(batch) >= 10:  # حفظ تدريجي
            store.append_rows(batch)
            batch = []
    data = store.append_rows(batch) if batch else store.load_data()
    st.total_in_file = len(data)
    if total is not None and len(found) < total:
        log.warning("الموقع يذكر %d إعلاناً واكتُشف %d — قد تكون بنية الترقيم تغيرت.", total, len(found))
    summarize(st, log)
    store.write_run(st)
    return st
