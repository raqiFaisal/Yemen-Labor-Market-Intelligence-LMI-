"""أساس الجامعات: طلبات مهذبة، حالة الاستكمال، سجل الفشل، العدادات.

قواعد صارمة:
  • الرابط "المكتشف" ليس إعلاناً منزّلاً. يُعتبر الإعلان منزّلاً فقط بعد نجاح استخراج بياناته وحفظه في CSV.
  • كل فشل يُسجَّل بسببه (HTTP، مهلة، robots، فشل تحليل) في logs/<source>_failures.csv.
  • لا قيم مخترعة: أي حقل غير موجود صراحة في الصفحة يبقى فارغاً.
"""
from __future__ import annotations

import csv
import json
import logging
import random
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from urllib import robotparser
from urllib.parse import urlparse

import pandas as pd
import requests

from core.config import LOGS_DIR, RAW_DIR, STATE_DIR

UA = "Mozilla/5.0 (compatible; LMI-GraduationProject/2.0; academic labor-market research)"
PARSER_VERSION = "2.0"
TRANSIENT = {429, 500, 502, 503, 504}

# ------------------------------------------------------------------ التواريخ
MONTHS = {
    **{m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)},
    "يناير": 1, "فبراير": 2, "مارس": 3, "ابريل": 4, "أبريل": 4, "مايو": 5, "يونيو": 6, "يوليو": 7, "اغسطس": 8,
    "أغسطس": 8, "سبتمبر": 9, "اكتوبر": 10, "أكتوبر": 10, "نوفمبر": 11, "ديسمبر": 12,
    "كانون الثاني": 1, "شباط": 2, "آذار": 3, "اذار": 3, "نيسان": 4, "أيار": 5, "ايار": 5, "حزيران": 6, "تموز": 7,
    "آب": 8, "أيلول": 9, "ايلول": 9, "تشرين الأول": 10, "تشرين الاول": 10, "تشرين الثاني": 11, "كانون الأول": 12,
    "كانون الاول": 12,
}
_MON = "|".join(sorted((re.escape(k) for k in MONTHS), key=len, reverse=True))


def _mon(tok: str) -> int:
    return MONTHS[tok.lower()[:3]] if tok[0].isascii() else MONTHS[tok]


def _yy(y: str) -> int:
    y = int(y)
    return 2000 + y if y < 100 else y


DATE_RX = [
    (re.compile(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})"), lambda m: (int(m[1]), int(m[2]), int(m[3]))),
    (re.compile(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})"), lambda m: (int(m[3]), int(m[2]), int(m[1]))),
    # 04 Nov, 26 | 12 Nov 2025 | 12-سبتمبر-2025
    (re.compile(rf"(\d{{1,2}})[\s\-/]*({_MON})[a-z]*\.?,?[\s\-/]+(\d{{4}}|\d{{2}})(?!\d)", re.I),
     lambda m: (_yy(m[3]), _mon(m[2]), int(m[1]))),
    # Nov 11, 2026
    (re.compile(rf"({_MON})[a-z]*\.?\s+(\d{{1,2}}),?\s+(\d{{4}})", re.I), lambda m: (int(m[3]), _mon(m[1]), int(m[2]))),
]
# التاريخ النسبي يُقبل فقط مع "منذ" أو "ago" — حتى لا تُفسَّر "خبرة 6 أشهر" كتاريخ نشر
REL_RX = re.compile(r"(?:منذ\s*(\d+)\s*(يوم|أيام|ايام|أسبوع|اسبوع|أسابيع|اسابيع|شهر|أشهر|اشهر)|"
                    r"(\d+)\s*(day|days|week|weeks|month|months)\s+ago)", re.I)


def parse_date_text(text: str, now: datetime | None = None):
    """أول تاريخ صريح في النص (رقمي/بأسماء الأشهر/سنة بخانتين) أو None — لا افتراضات."""
    if not text:
        return None
    now = now or datetime.now()
    for rx, f in DATE_RX:
        for m in rx.finditer(text):
            try:
                y, mo, d = f(m)
                if 2000 <= y <= now.year + 1:
                    return datetime(y, mo, d).date()
            except (ValueError, KeyError):
                continue
    m = REL_RX.search(text)
    if m:
        n = int(m[1] or m[3])
        u = (m[2] or m[4]).lower()
        days = n * (7 if u.startswith(("week", "أسبوع", "اسبوع", "أسابيع", "اسابيع")) else
                    30 if u.startswith(("month", "شهر", "أشهر", "اشهر")) else 1)
        return (now - timedelta(days=days)).date()
    return None


# ------------------------------------------------------------------ الطلبات
class Fetcher:
    def __init__(self, delay: float, log: logging.Logger, retries: int = 3, timeout: int = 30):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept-Language": "ar,en;q=0.8"})
        self.delay, self.log, self.retries, self.timeout = delay, log, retries, timeout
        self.robots: dict = {}
        self.last = 0.0
        self.requests_made = 0

    def allowed(self, url: str) -> bool:
        host = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        if host not in self.robots:
            rp = robotparser.RobotFileParser()
            try:
                r = self.s.get(host + "/robots.txt", timeout=15)
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except Exception as e:  # noqa: BLE001
                self.log.warning("تعذر قراءة robots.txt لـ %s (%s) — المتابعة بحذر", host, e)
                rp = None
            self.robots[host] = rp
        rp = self.robots[host]
        return True if rp is None else rp.can_fetch(UA, url)

    def get(self, url: str) -> tuple[str | None, str | None, int | None]:
        """(html, reason_if_failed, http_status)"""
        if not self.allowed(url):
            return None, "robots.txt disallow", None
        status = None
        for attempt in range(1, self.retries + 1):
            wait = self.delay + random.uniform(0, self.delay * 0.4) - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            self.requests_made += 1
            try:
                r = self.s.get(url, timeout=self.timeout)
                status = r.status_code
                if r.ok:
                    r.encoding = r.encoding or r.apparent_encoding
                    return r.text, None, status
                if status in TRANSIENT and attempt < self.retries:
                    back = min(60, self.delay * 2 ** attempt)
                    if status == 429 and r.headers.get("Retry-After", "").isdigit():
                        back = max(back, int(r.headers["Retry-After"]))
                    self.log.info("HTTP %s — إعادة المحاولة بعد %.0f ث: %s", status, back, url)
                    time.sleep(back)
                    continue
                return None, f"HTTP {status}", status
            except requests.RequestException as e:
                if attempt < self.retries:
                    time.sleep(min(60, self.delay * 2 ** attempt))
                    continue
                return None, f"{type(e).__name__}: {str(e)[:150]}", status
        return None, "retries exhausted", status


# ------------------------------------------------------------------ الحالة والسجلات
@dataclass
class RunStats:
    source: str
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    pages_visited: int = 0
    pages_failed: int = 0
    links_discovered: int = 0          # روابط إعلانات فريدة رُصدت في هذا التشغيل
    links_new: int = 0                 # لم تكن مكتشفة في تشغيلات سابقة
    already_downloaded: int = 0        # موجودة في CSV فتم تخطيها (استكمال)
    duplicate_links: int = 0           # تكرار نفس الإعلان داخل الصفحات/التصنيفات
    jobs_with_category: int = 0        # إعلانات ارتبطت بتصنيف من صفحات التصنيفات
    details_attempted: int = 0
    downloaded_ok: int = 0             # إعلانات استُخرجت بياناتها وحُفظت فعلاً
    failed: int = 0
    content_duplicates: int = 0        # إعلان بمعرف جديد ونفس المحتوى (إعادة نشر محتملة)
    total_in_file: int = 0
    finished_at: str | None = None


class SourceStore:
    """ملفات مصدر واحد: البيانات، الروابط المكتشفة، الفشل، سجل التشغيلات."""

    def __init__(self, source: str):
        self.source = source
        self.data_file = RAW_DIR / f"yemen_{source}.csv"
        self.discovered_file = STATE_DIR / f"{source}_discovered.csv"
        self.failures_file = LOGS_DIR / f"{source}_failures.csv"
        self.runs_file = LOGS_DIR / f"{source}_runs.jsonl"

    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.data_file, dtype=str) if self.data_file.exists() else pd.DataFrame()

    def downloaded_keys(self) -> set:
        """معرفات الإعلانات المحفوظة فعلاً (source_job_id، أو المعرف المشتق من الرابط)."""
        d = self.load_data()
        if d.empty:
            return set()
        ids = d.get("source_job_id", pd.Series("", index=d.index)).fillna("")
        der = d.get("url", pd.Series("", index=d.index)).fillna("").str.extract(
            r"/jobs/(?:[a-z0-9\-%]+-)?([0-9a-f]{8})(?:-[0-9a-f\-]{27})?/?$", flags=re.I, expand=False).fillna("")
        return {x.lower() for x in ids.where(ids != "", der) if x}

    def update_categories(self, key_to_cat: dict) -> int:
        """تحديث تصنيف الإعلانات المحفوظة سابقاً دون إعادة تنزيلها (التصنيف معلومة من صفحة التصنيفات لا من الإعلان)."""
        d = self.load_data()
        if d.empty or not key_to_cat:
            return 0
        if "category" not in d:
            d["category"] = ""
        ids = d.get("source_job_id", pd.Series("", index=d.index)).fillna("").str.lower()
        new = ids.map(key_to_cat)
        changed = new.notna() & (new != d["category"].fillna(""))
        d.loc[changed, "category"] = new[changed]
        if changed.any():
            d.to_csv(self.data_file, index=False, encoding="utf-8-sig")
        return int(changed.sum())

    def downloaded_urls(self) -> set:
        d = self.load_data()
        return set(d["url"].dropna()) if "url" in d else set()

    def load_discovered(self) -> pd.DataFrame:
        return pd.read_csv(self.discovered_file, dtype=str) if self.discovered_file.exists() else pd.DataFrame()

    def save_discovered(self, df: pd.DataFrame):
        df.to_csv(self.discovered_file, index=False, encoding="utf-8-sig")

    def permanent_failures(self) -> set:
        if not self.failures_file.exists():
            return set()
        f = pd.read_csv(self.failures_file, dtype=str)
        return set(f.loc[f["reason"].str.contains("HTTP 404|HTTP 410|robots", na=False), "url"])

    def log_failure(self, url: str, stage: str, reason: str, status=None):
        new = not self.failures_file.exists()
        with open(self.failures_file, "a", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["at", "source", "stage", "url", "reason", "http_status"])
            w.writerow([datetime.now().isoformat(timespec="seconds"), self.source, stage, url, reason, status or ""])

    def append_rows(self, rows: list[dict]) -> pd.DataFrame:
        """إلحاق الإعلانات الناجحة فوراً (حتى لا يضيع شيء إن انقطع التشغيل) مع إزالة تكرار الرابط/المعرف."""
        old = self.load_data()
        df = pd.concat([old, pd.DataFrame(rows).astype(str).replace({"None": "", "nan": "", "NaT": ""})],
                       ignore_index=True).fillna("")
        if "source_job_id" not in df:
            df["source_job_id"] = ""
        # سجلات من إصدار سابق بلا معرف: يُشتق المعرف من الرابط إن أمكن
        miss = df["source_job_id"] == ""
        df.loc[miss, "source_job_id"] = df.loc[miss, "url"].str.extract(
            r"/jobs/(?:[a-z0-9\-%]+-)?([0-9a-f]{8})(?:-[0-9a-f\-]{27})?/?$", flags=re.I, expand=False).fillna("").str.lower()
        key = df["source_job_id"].where(df["source_job_id"] != "", df["url"])
        df = df.loc[~key.duplicated(keep="last")]   # الأحدث (إصدار المحلل الأحدث) يحل محل القديم
        df.to_csv(self.data_file, index=False, encoding="utf-8-sig")
        return df

    def write_run(self, st: RunStats):
        st.finished_at = datetime.now().isoformat(timespec="seconds")
        with open(self.runs_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(st), ensure_ascii=False) + "\n")


def make_logger(source: str) -> logging.Logger:
    log = logging.getLogger(f"scrape.{source}")
    if not log.handlers:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
        h = logging.StreamHandler()
        h.setFormatter(fmt)
        fh = logging.FileHandler(LOGS_DIR / f"{source}_{datetime.now():%Y%m%d}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(h)
        log.addHandler(fh)
    return log


def summarize(st: RunStats, log: logging.Logger):
    log.info("─" * 60)
    log.info("ملخص [%s]: صفحات مُزارة %d (فشل %d) | روابط مكتشفة %d (جديدة %d) | مكررة %d",
             st.source, st.pages_visited, st.pages_failed, st.links_discovered, st.links_new, st.duplicate_links)
    log.info("        إعلانات مرتبطة بتصنيف: %d", st.jobs_with_category)
    log.info("        موجودة مسبقاً (تخطي) %d | محاولات تنزيل %d | ✔ نجحت %d | ✖ فشلت %d | تكرار محتوى %d",
             st.already_downloaded, st.details_attempted, st.downloaded_ok, st.failed, st.content_duplicates)
    log.info("        إجمالي الإعلانات الحقيقية المحفوظة في الملف: %d", st.total_in_file)
