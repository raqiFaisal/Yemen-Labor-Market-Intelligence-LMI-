"""جامع صفحات telegra.ph المرتبطة بمنشورات Telegram (تفاصيل الإعلانات الكاملة).

المصدر: واجهة Telegraph العامة الرسمية  https://api.telegra.ph/getPage/<path>?return_content=true
        (JSON منظم: العنوان، الكاتب، المحتوى كعقد). عند فشلها: صفحة HTML ‎telegra.ph/<path>.

الملفات (خارج المستوى الأعلى لـ data/raw — لا يقرؤها خط التدريب قبل مرحلة التوحيد):
    data/raw/telegram/telegraph/pages.csv   صفحة واحدة لكل path: النص الكامل كما هو + العنوان + الروابط
    data/raw/telegram/telegraph/links.csv   ربط كل منشور Telegram (channel, message_id) بكل صفحة يشير إليها
    data/raw/logs/telegraph_failures.csv    كل فشل وسببه (PAGE_NOT_FOUND دائم؛ الشبكة/5xx مؤقت يعاد لاحقاً)
    data/raw/logs/telegraph_runs.jsonl      عدادات كل تشغيل

الاستكمال: الصفحات المحفوظة بنجاح لا تُطلب مجدداً؛ الفشل الدائم يُتخطى إلا مع --retry-failed؛
الفشل المؤقت يعاد تلقائياً في التشغيل التالي. الحفظ تدريجي كل SAVE_EVERY صفحة (لا يضيع شيء عند الانقطاع).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from urllib.parse import unquote, urlparse

import pandas as pd
from bs4 import BeautifulSoup

from core.config import LOGS_DIR
from scripts.collectors.telegram_web import TG_DIR
from scripts.scrapers.base import Fetcher, make_logger

API = "https://api.telegra.ph/getPage/{}?return_content=true"
HTML = "https://telegra.ph/{}"
HOSTS = {"telegra.ph", "www.telegra.ph", "graph.org", "te.legra.ph"}
OUT_DIR = TG_DIR / "telegraph"
PAGE_COLS = ["telegraph_path", "telegraph_url", "title", "author_name", "author_url", "content_text", "content_links",
             "text_hash", "views", "fetch_method", "fetched_at", "first_channel", "first_message_id"]
LINK_COLS = ["source_channel", "source_message_id", "message_date", "telegraph_path"]
PERMANENT = ("PAGE_NOT_FOUND", "HTTP 404", "HTTP 410", "robots.txt")
SAVE_EVERY = 25
BLOCK = {"p", "h3", "h4", "li", "blockquote", "aside", "figure", "figcaption", "pre", "hr", "br", "ul", "ol"}


def path_of(url: str) -> str | None:
    """مسار صفحة telegra.ph من الرابط؛ None لغير ذلك أو لروابط الملفات/الصور."""
    u = urlparse(url.strip())
    if u.netloc.lower() not in HOSTS:
        return None
    p = unquote(u.path).strip("/")
    if not p or p.startswith(("file/", "api", "upload")) or "/" in p:
        return None
    return p


def nodes_to_text(nodes, links: list) -> str:
    """تحويل عقد Telegraph إلى نص بأسطر، مع جمع الروابط. لا يحذف أي نص."""
    out = []

    def walk(n):
        if isinstance(n, str):
            out.append(n)
            return
        tag = n.get("tag", "")
        href = (n.get("attrs") or {}).get("href")
        if tag == "a" and href and href.startswith("http"):
            links.append(href)
        if tag == "br":
            out.append("\n")
            return
        if tag == "li":
            out.append("\n- ")
        elif tag in BLOCK:
            out.append("\n")
        for c in n.get("children") or []:
            walk(c)
        if tag in BLOCK:
            out.append("\n")

    for n in nodes or []:
        walk(n)
    text = "".join(out)
    lines = [ln.rstrip() for ln in text.splitlines()]
    cleaned, blank = [], False
    for ln in lines:
        if ln.strip():
            cleaned.append(ln)
            blank = False
        elif not blank and cleaned:
            cleaned.append("")
            blank = True
    return "\n".join(cleaned).strip()


def parse_api(body: str) -> tuple[dict | None, str | None]:
    try:
        j = json.loads(body)
    except ValueError:
        return None, "invalid JSON"
    if not j.get("ok"):
        return None, str(j.get("error") or "API error")
    r = j["result"]
    links: list = []
    text = nodes_to_text(r.get("content"), links)
    return {"title": r.get("title", ""), "author_name": r.get("author_name", ""), "author_url": r.get("author_url", ""),
            "content_text": text, "content_links": " ".join(dict.fromkeys(links)), "views": str(r.get("views", ""))}, None


def parse_html(html: str) -> tuple[dict | None, str | None]:
    soup = BeautifulSoup(html, "html.parser")
    art = soup.select_one("article#_tl_editor, article.tl_article_content, .tl_article_content")
    if not art:
        return None, "no article in HTML"
    h1 = art.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else ""
    if h1:
        h1.decompose()
    addr = art.find("address")
    author = addr.get_text(" ", strip=True) if addr else ""
    if addr:
        addr.decompose()
    for br in art.find_all("br"):
        br.replace_with("\n")
    for b in art.find_all(list(BLOCK - {"br"})):
        b.insert_before("\n")
        b.insert_after("\n")
    links = [a["href"] for a in art.find_all("a", href=True) if a["href"].startswith("http")]
    text = "\n".join(ln.strip() for ln in art.get_text("").splitlines() if ln.strip())
    return {"title": title, "author_name": author, "author_url": "", "content_text": text,
            "content_links": " ".join(dict.fromkeys(links)), "views": ""}, None


@dataclass
class TPStats:
    channels: list
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    posts_scanned: int = 0
    posts_with_telegraph: int = 0
    unique_pages_linked: int = 0
    already_saved: int = 0
    skipped_permanent_failures: int = 0
    attempted: int = 0
    saved_ok: int = 0
    via_api: int = 0
    via_html: int = 0
    empty_content: int = 0
    failed: int = 0
    remaining: int = 0
    finished_at: str | None = None


def build_links(channels: list[str]) -> pd.DataFrame:
    rows = []
    for ch in channels:
        f = TG_DIR / f"{ch}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, dtype=str, keep_default_na=False)
        for r in d.itertuples(index=False):
            for u in r.link_urls.split():
                p = path_of(u)
                if p:
                    rows.append({"source_channel": ch, "source_message_id": r.source_message_id,
                                 "message_date": r.message_date, "telegraph_path": p})
    return pd.DataFrame(rows, columns=LINK_COLS).drop_duplicates()


def _load(f, cols):
    return pd.read_csv(f, dtype=str, keep_default_na=False) if f.exists() else pd.DataFrame(columns=cols)


def _fail(path, url, reason, status):
    f = LOGS_DIR / "telegraph_failures.csv"
    pd.DataFrame([{"at": datetime.now().isoformat(timespec="seconds"), "telegraph_path": path, "url": url,
                   "reason": reason, "http_status": status or "",
                   "permanent": any(reason.startswith(p) for p in PERMANENT)}]) \
        .to_csv(f, mode="a", header=not f.exists(), index=False, encoding="utf-8-sig")


def permanent_failures() -> set:
    f = LOGS_DIR / "telegraph_failures.csv"
    if not f.exists():
        return set()
    d = pd.read_csv(f, dtype=str, keep_default_na=False)
    return set(d.loc[d.permanent == "True", "telegraph_path"])


def fetch_page(f: Fetcher, path: str) -> tuple[dict | None, str | None, int | None, str]:
    body, why, code = f.get(API.format(path))
    if body:
        rec, err = parse_api(body)
        if rec:
            return rec, None, code, "api"
        if err == "PAGE_NOT_FOUND":
            return None, err, code, "api"
        why = f"API: {err}"
    html, why2, code2 = f.get(HTML.format(path))
    if html:
        rec, err = parse_html(html)
        if rec:
            return rec, None, code2, "html"
        return None, f"{why} | HTML: {err}", code2, "html"
    return None, why2 or why, code2 or code, "html"


def run(channels: list[str], limit: int | None = None, delay: float = 3.0, retry_failed: bool = False) -> TPStats:
    log = make_logger("telegraph")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    pages_f, links_f = OUT_DIR / "pages.csv", OUT_DIR / "links.csv"
    st = TPStats(channels)
    missing = [ch for ch in channels if not (TG_DIR / f"{ch}.csv").exists()]
    if missing:
        raise SystemExit(f"✖ ملفات المنشورات غير موجودة: {[str(TG_DIR / f'{c}.csv') for c in missing]}\n"
                         "  تأكد من ربط Drive (data/raw → MyDrive/lmi_data/raw) — لا تُعد جمع القنوات.")
    links = build_links(channels)
    for ch in channels:
        f = TG_DIR / f"{ch}.csv"
        # عدد المنشورات = صفوف CSV (لا أسطر الملف: النصوص متعددة الأسطر)
        st.posts_scanned += len(pd.read_csv(f, usecols=["source_message_id"], dtype=str)) if f.exists() else 0
    st.posts_with_telegraph = links[["source_channel", "source_message_id"]].drop_duplicates().shape[0]
    old_links = _load(links_f, LINK_COLS)
    pd.concat([old_links, links]).drop_duplicates().to_csv(links_f, index=False, encoding="utf-8-sig")

    # ترتيب: الأحدث أولاً (أقرب للسوق الحالي)، وأول منشور يشير لكل صفحة
    first = links.sort_values("message_date", ascending=False).drop_duplicates("telegraph_path")
    st.unique_pages_linked = len(first)
    saved = set(_load(pages_f, PAGE_COLS).telegraph_path)
    perm = set() if retry_failed else permanent_failures()
    st.already_saved = len(set(first.telegraph_path) & saved)
    todo = first[~first.telegraph_path.isin(saved)]
    st.skipped_permanent_failures = int(todo.telegraph_path.isin(perm).sum())
    todo = todo[~todo.telegraph_path.isin(perm)]
    if limit is not None:
        todo = todo.head(limit)
    log.info("منشورات مفحوصة %d | تشير لـ telegra.ph %d | صفحات فريدة %d | محفوظة سابقاً %d | فشل دائم متخطى %d | للجلب الآن %d",
             st.posts_scanned, st.posts_with_telegraph, st.unique_pages_linked, st.already_saved,
             st.skipped_permanent_failures, len(todo))
    f = Fetcher(delay, log)
    buf = []

    def flush():
        if buf:
            pd.DataFrame(buf, columns=PAGE_COLS).to_csv(pages_f, mode="a", header=not pages_f.exists(), index=False,
                                                        encoding="utf-8-sig")
            buf.clear()

    try:
        for i, r in enumerate(todo.itertuples(index=False), 1):
            st.attempted += 1
            rec, why, code, method = fetch_page(f, r.telegraph_path)
            if not rec:
                st.failed += 1
                _fail(r.telegraph_path, HTML.format(r.telegraph_path), why or "unknown", code)
                log.warning("✖ %s — %s", r.telegraph_path, why)
                continue
            st.saved_ok += 1
            st.via_api += method == "api"
            st.via_html += method == "html"
            st.empty_content += not rec["content_text"]
            buf.append({**rec, "telegraph_path": r.telegraph_path, "telegraph_url": HTML.format(r.telegraph_path),
                        "text_hash": hashlib.sha1(rec["content_text"].encode()).hexdigest()[:16],
                        "fetch_method": method, "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "first_channel": r.source_channel, "first_message_id": r.source_message_id})
            if len(buf) >= SAVE_EVERY:
                flush()
            if i % 100 == 0:
                log.info("تقدم: %d/%d (نجاح %d، فشل %d)", i, len(todo), st.saved_ok, st.failed)
    finally:
        flush()
        if pages_f.exists():  # حماية من تكرار ناتج عن انقطاع أثناء الكتابة
            p = pd.read_csv(pages_f, dtype=str, keep_default_na=False).drop_duplicates("telegraph_path", keep="first")
            p.to_csv(pages_f, index=False, encoding="utf-8-sig")
        total = len(_load(pages_f, PAGE_COLS))
        st.remaining = st.unique_pages_linked - len(set(first.telegraph_path) & set(_load(pages_f, PAGE_COLS).telegraph_path))
        st.finished_at = datetime.now().isoformat(timespec="seconds")
        with open(LOGS_DIR / "telegraph_runs.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(st), ensure_ascii=False) + "\n")
        log.info("─" * 60)
        log.info("ملخص telegra.ph: حاولت %d | حُفظت %d (API %d، HTML %d، محتوى فارغ %d) | فشل %d",
                 st.attempted, st.saved_ok, st.via_api, st.via_html, st.empty_content, st.failed)
        log.info("        المحفوظ فعلاً: %d صفحة | المتبقي من صفحات هذه القنوات: %d (منها فشل دائم %d)",
                 total, st.remaining, st.skipped_permanent_failures)
    return st


def status():
    pages_f, links_f = OUT_DIR / "pages.csv", OUT_DIR / "links.csv"
    p, l = _load(pages_f, PAGE_COLS), _load(links_f, LINK_COLS)
    linked = l.telegraph_path.nunique() if len(l) else 0
    fails = LOGS_DIR / "telegraph_failures.csv"
    fd = pd.read_csv(fails, dtype=str, keep_default_na=False) if fails.exists() else pd.DataFrame(columns=["permanent", "reason"])
    print(f"صفحات مرتبطة بمنشورات: {linked} | محفوظة: {len(p)} | فشل دائم: {fd[fd.permanent == 'True'].shape[0]} | "
          f"سجلات فشل مؤقت: {fd[fd.permanent != 'True'].shape[0]}")
    if len(p):
        ln = p.content_text.str.len()
        print(f"طول النص: وسيط {int(ln.median())} حرفاً | فارغ {int((ln == 0).sum())} | عبر API {int((p.fetch_method == 'api').sum())}")
    if len(fd):
        print("أكثر أسباب الفشل:", fd.reason.str[:60].value_counts().head(5).to_dict())
