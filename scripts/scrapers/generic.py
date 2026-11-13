"""جامع تكيّفي للمصادر اليمنية الأخرى (Yemen ShortList / Yemen Board) — تجريبي.

⚠️ بنية هذه المواقع لم تُفحص مباشرة؛ الاستخراج تكيّفي (h1، أكبر كتلة نصية، حقول موسومة).
كل مصدر يُحفظ في ملف منفصل data/raw/yemen_<source>.csv مع عمود source، ونفس قواعد السجل والاستكمال.
ابدأ دائماً بـ: python -m scripts.scrape_yemen --site shortlist --limit 5 --debug
"""
from __future__ import annotations

import argparse
import re
import time
from datetime import datetime, timedelta
from urllib import robotparser
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from preprocessing.schema import YEMEN_GOVERNORATES, normalize_governorate

from scripts.scrapers.base import RunStats, SourceStore, make_logger, summarize
from scripts.scrapers.base import Fetcher as PoliteFetcher

UA = "Mozilla/5.0 (compatible; LMI-GraduationProject/2.0; academic labor-market research)"

SITES = {
    "shortlist": {"base": "https://yemenshortlist.com", "list": ["https://yemenshortlist.com/jobs?page={p}", "https://yemenshortlist.com/page/{p}/"]},
    "yemenboard": {"base": "https://yemenboard.com", "list": ["https://yemenboard.com/jobs?page={p}", "https://yemenboard.com/page/{p}"]},
}
# روابط صفحات تفاصيل الوظائف (تعديلها عند تغير بنية الموقع)
LINK_PATTERNS = {
    "shortlist": re.compile(r"/(jobs?|vacanc\w*|\d{4}/\d{2})/[\w\-%]+", re.I),
    "yemenboard": re.compile(r"/(jobs?|job-details|vacanc\w*)/[\w\-%]+", re.I),
}
MONTHS = {
    **{m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)},
    "يناير": 1, "فبراير": 2, "مارس": 3, "ابريل": 4, "أبريل": 4, "مايو": 5, "يونيو": 6, "يوليو": 7, "اغسطس": 8,
    "أغسطس": 8, "سبتمبر": 9, "اكتوبر": 10, "أكتوبر": 10, "نوفمبر": 11, "ديسمبر": 12,
    "كانون الثاني": 1, "شباط": 2, "آذار": 3, "اذار": 3, "نيسان": 4, "أيار": 5, "ايار": 5, "حزيران": 6, "تموز": 7,
    "آب": 8, "اب": 8, "أيلول": 9, "ايلول": 9, "تشرين الأول": 10, "تشرين الاول": 10, "تشرين الثاني": 11, "كانون الأول": 12,
    "كانون الاول": 12,
}
_MON = "|".join(sorted((re.escape(k) for k in MONTHS), key=len, reverse=True))
DATE_RX = [
    (re.compile(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})"), lambda m: (int(m[1]), int(m[2]), int(m[3]))),
    (re.compile(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})"), lambda m: (int(m[3]), int(m[2]), int(m[1]))),
    # 12 Nov 2025 | 12 نوفمبر 2025
    (re.compile(rf"(\d{{1,2}})[\s\-/]*({_MON})[a-z]*\.?,?[\s\-/]+(\d{{4}})", re.I),
     lambda m: (int(m[3]), MONTHS[m[2].lower()[:3]] if m[2][0].isascii() else MONTHS[m[2]], int(m[1]))),
    # Nov 12, 2025
    (re.compile(rf"({_MON})[a-z]*\.?\s+(\d{{1,2}}),?\s+(\d{{4}})", re.I),
     lambda m: (int(m[3]), MONTHS[m[1].lower()[:3]] if m[1][0].isascii() else MONTHS[m[1]], int(m[2]))),
]
# تاريخ نسبي فقط مع "منذ" أو "ago" — حتى لا تُفسَّر "خبرة 6 أشهر" كتاريخ نشر
REL_RX = re.compile(r"(?:منذ\s*(\d+)\s*(يوم|أيام|ايام|أسبوع|اسبوع|أسابيع|اسابيع|شهر|أشهر|اشهر)|"
                    r"(\d+)\s*(day|days|week|weeks|month|months)\s+ago)", re.I)
POSTED_LABELS = r"(?:posted|published|date posted|post date|تاريخ النشر|نشر في|تاريخ الإعلان|تاريخ الاعلان|نُشر)"
DEADLINE_LABELS = r"(?:deadline|closing date|apply before|expires?|آخر موعد|اخر موعد|تاريخ الانتهاء|تاريخ الإغلاق|ينتهي)"
CATEGORY_LABELS = (r"(?:job category|category|categories|sector|industry|functional area|job field|field|"
                   r"المجال|القطاع|التصنيف|الفئة|مجال العمل|التخصص المطلوب)")


def parse_date_text(text: str):
    """يحلل أول تاريخ صريح في النص (رقمي أو بأسماء الأشهر العربية/الإنجليزية)."""
    if not text:
        return None
    for rx, f in DATE_RX:
        for m in rx.finditer(text):
            try:
                y, mo, d = f(m)
                if 2000 <= y <= datetime.now().year + 1:
                    return datetime(y, mo, d).date()
            except (ValueError, KeyError):
                continue
    m = REL_RX.search(text)
    if m:
        n = int(m[1] or m[3])
        u = (m[2] or m[4]).lower()
        days = n * (7 if u.startswith(("week", "أسبوع", "اسبوع", "أسابيع", "اسابيع")) else
                    30 if u.startswith(("month", "شهر", "أشهر", "اشهر")) else 1)
        return (datetime.now() - timedelta(days=days)).date()
    return None


def labeled_field(body: str, labels: str) -> str | None:
    """قيمة حقل موسوم: 'Category: IT' أو العنوان في سطر والقيمة في السطر التالي."""
    lines = [l.strip() for l in body.split("\n") if l.strip()]
    rx = re.compile(rf"^{labels}\s*[:：\-]?\s*(.*)$", re.I)
    for i, l in enumerate(lines[:120]):
        m = rx.match(l)
        if m:
            v = m.group(1).strip(" :：-")
            if not v and i + 1 < len(lines):
                v = lines[i + 1]
            if v and len(v) <= 120:
                return v
    return None


class Fetcher:
    def __init__(self, delay: float):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept-Language": "ar,en;q=0.8"})
        self.delay = delay
        self.robots = {}
        self.last = 0.0

    def allowed(self, url: str) -> bool:
        host = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        if host not in self.robots:
            rp = robotparser.RobotFileParser()
            try:
                rp.parse(self.s.get(host + "/robots.txt", timeout=15).text.splitlines())
            except Exception:  # noqa: BLE001
                rp = None
            self.robots[host] = rp
        rp = self.robots[host]
        return True if rp is None else rp.can_fetch(UA, url)

    def get(self, url: str) -> str | None:
        if not self.allowed(url):
            print(f"   ⛔ robots.txt يمنع: {url}")
            return None
        wait = self.delay - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        self.last = time.time()
        try:
            r = self.s.get(url, timeout=25)
            return r.text if r.ok else None
        except Exception as e:  # noqa: BLE001
            print(f"   ⚠ {url}: {e}")
            return None


def _date(soup: BeautifulSoup, body: str):
    """الأولوية: حقل "تاريخ النشر" الموسوم ← meta/time ← أول تاريخ في النص ليس ضمن سطر "آخر موعد"."""
    v = labeled_field(body, POSTED_LABELS)
    if v and (d := parse_date_text(v)):
        return d
    for meta in ("article:published_time", "datePosted", "date", "og:updated_time"):
        m = soup.find("meta", attrs={"property": meta}) or soup.find("meta", attrs={"name": meta}) or \
            soup.find(attrs={"itemprop": meta})
        val = m and (m.get("content") or m.get("datetime"))
        if val:
            d = pd.to_datetime(val, errors="coerce", utc=True)
            if pd.notna(d):
                return d.date()
    for t in soup.find_all("time"):
        par = t.parent.get_text(" ", strip=True) if t.parent else ""
        if re.search(DEADLINE_LABELS, par, re.I):
            continue
        d = pd.to_datetime(t.get("datetime"), errors="coerce", utc=True) if t.get("datetime") else None
        d = d.date() if d is not None and pd.notna(d) else parse_date_text(t.get_text(" ", strip=True))
        if d:
            return d
    clean = "\n".join(l for l in body.split("\n") if not re.search(DEADLINE_LABELS, l, re.I))
    return parse_date_text(clean)


def parse_job(html: str, url: str, site: str) -> dict | None:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    h1 = soup.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else (soup.title.get_text(strip=True) if soup.title else "")
    blocks = sorted((b for b in soup.find_all(["article", "main", "section", "div"])),
                    key=lambda b: len(b.get_text(" ", strip=True)), reverse=True)
    body = blocks[0].get_text("\n", strip=True) if blocks else soup.get_text("\n", strip=True)
    if len(body) < 80 or not title:
        return None
    company = ""
    og = soup.find("meta", attrs={"property": "og:site_name"})
    for lab in ("Company", "Organization", "الجهة", "الشركة", "المنظمة"):
        m = re.search(rf"{lab}\s*[:：]\s*(.+)", body)
        if m:
            company = m.group(1).split("\n")[0][:120]
            break
    loc = None
    for line in body.split("\n")[:40]:
        g = normalize_governorate(line)
        if g in {v[0] for v in YEMEN_GOVERNORATES.values()}:
            loc = g
            break
    dl = labeled_field(body, DEADLINE_LABELS)
    return {"title": title[:200], "description": body[:6000], "company": company or (og["content"] if og else ""),
            "location": loc, "country": "Yemen", "category": labeled_field(body, CATEGORY_LABELS),
            "posted_date": _date(soup, body), "deadline": parse_date_text(dl) if dl else None,
            "source": site, "url": url, "scraped_at": datetime.now().strftime("%Y-%m-%d")}


def scrape(site: str, pages: int, limit: int, delay: float, debug: bool, skip_urls: set | None = None,
           start_page: int = 1) -> pd.DataFrame:
    cfg, pat, f = SITES[site], LINK_PATTERNS[site], Fetcher(delay)
    skip_urls = skip_urls or set()
    links, rows, empty_streak, known = [], [], 0, 0
    for p in range(start_page, start_page + pages):
        got = False
        for tmpl in cfg["list"]:
            html = f.get(tmpl.format(p=p))
            if not html:
                continue
            soup = BeautifulSoup(html, "html.parser")
            found = [urljoin(cfg["base"], a["href"]) for a in soup.find_all("a", href=True) if pat.search(a["href"])]
            found = [u for u in dict.fromkeys(found) if urlparse(u).netloc.endswith(urlparse(cfg["base"]).netloc)]
            new = [u for u in found if u not in links and u not in skip_urls]
            known += sum(u in skip_urls for u in found)
            if found:
                got = True
                links += new
                print(f"   صفحة {p}: {len(found)} رابط ({len(new)} جديد) — {tmpl}")
                cfg["list"] = [tmpl]  # تثبيت القالب الذي نجح لبقية الصفحات
                break
        empty_streak = 0 if got else empty_streak + 1
        if empty_streak >= 2:
            print("   ⏹ لا توجد صفحات إضافية — توقف التصفح.")
            break
        if limit and len(links) >= limit:
            break
    links = links[:limit] if limit else links
    if known:
        print(f"   ↺ تخطي {known} رابط مجموع سابقاً")
    for i, u in enumerate(links, 1):
        html = f.get(u)
        job = parse_job(html, u, site) if html else None
        if job:
            rows.append(job)
            if debug:
                print(f"   [{i}] {job['title'][:55]} | {job['location']} | نشر: {job['posted_date']} | "
                      f"تصنيف: {job['category']}")
    df = pd.DataFrame(rows)
    if len(df):
        print(f"   📅 نسبة الإعلانات ذات تاريخ نشر: {df.posted_date.notna().mean() * 100:.0f}% | "
              f"ذات تصنيف: {df.category.notna().mean() * 100:.0f}%")
    return df




def run(site: str, max_pages: int = 30, limit: int = 0, delay: float = 3.0, retry_failed: bool = False,
        refresh: bool = False, debug: bool = False) -> RunStats:
    """نفس منطق الحالة في yemenhr: مكتشف ≠ منزّل، سجل فشل، استكمال، عدادات."""
    log = make_logger(site)
    st, store = RunStats(site), SourceStore(site)
    f = PoliteFetcher(delay, log)
    cfg, pat = SITES[site], LINK_PATTERNS[site]
    done = set() if refresh else store.downloaded_urls()
    perm = set() if retry_failed else store.permanent_failures()
    links, empty = [], 0
    for p in range(1, max_pages + 1):
        got = False
        for tmpl in cfg["list"]:
            u = tmpl.format(p=p)
            html, why, code = f.get(u)
            st.pages_visited += 1
            if not html:
                st.pages_failed += 1
                store.log_failure(u, "listing", why, code)
                continue
            soup = BeautifulSoup(html, "html.parser")
            fl = [urljoin(cfg["base"], a["href"]) for a in soup.find_all("a", href=True) if pat.search(a["href"])]
            fl = [x for x in dict.fromkeys(fl) if urlparse(x).netloc.endswith(urlparse(cfg["base"]).netloc)]
            new = [x for x in fl if x not in links]
            st.duplicate_links += len(fl) - len(new)
            if new:
                got = True
                links += new
                cfg["list"] = [tmpl]
                log.info("صفحة %d: %d رابط جديد", p, len(new))
                break
        empty = 0 if got else empty + 1
        if empty >= 2:
            break
    st.links_discovered = len(links)
    st.already_downloaded = sum(u in done for u in links)
    todo = [u for u in links if u not in done and u not in perm]
    if limit:
        todo = todo[:limit]
    rows = []
    for i, u in enumerate(todo, 1):
        st.details_attempted += 1
        html, why, code = f.get(u)
        if not html:
            st.failed += 1
            store.log_failure(u, "detail", why, code)
            continue
        job = parse_job(html, u, site)
        if not job:
            st.failed += 1
            store.log_failure(u, "parse", "no title or body < 80 chars", code)
            continue
        job["parser_version"] = "generic-1"
        rows.append(job)
        st.downloaded_ok += 1
        if debug:
            log.info("[%d/%d] ✔ %s | %s | %s", i, len(todo), job["title"][:50], job["location"], job["posted_date"])
    data = store.append_rows(rows) if rows else store.load_data()
    st.total_in_file = len(data)
    summarize(st, log)
    store.write_run(st)
    return st
