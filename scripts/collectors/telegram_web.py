"""جامع قنوات Telegram العامة عبر صفحات المعاينة t.me/s/<channel> (لا يحتاج مفاتيح API).

المرحلة 2 من خطة المصادر المتعددة. يحفظ **المنشورات الخام** فقط — لا يقرر أنها وظائف (ذلك في المرحلة 5).

الملفات (لكل قناة على حدة، داخل RAW_DIR/telegram/ — لا يقرؤها خط التدريب الحالي حتى مرحلة التوحيد):
    data/raw/telegram/<channel>.csv            منشور واحد لكل (channel, message_id)
    data/raw/telegram/<channel>_edits.csv      النسخ السابقة لأي منشور تغيّر نصه
    data/raw/telegram/state/<channel>.json     أدنى/أعلى رقم رسالة جُمع + هل اكتمل الأرشيف
    data/raw/logs/tg_<channel>_failures.csv    كل فشل وسببه
    data/raw/logs/tg_<channel>_runs.jsonl      عدادات كل تشغيل

الأوضاع:
    update    من الأحدث نزولاً حتى الوصول لآخر رسالة محفوظة (+ إعادة فحص أحدث صفحتين لالتقاط التعديلات)
    backfill  من أقدم رسالة محفوظة نزولاً نحو بداية القناة، بحد أقصى --max-pages في التشغيل
    all       update ثم backfill
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlparse

import pandas as pd
from bs4 import BeautifulSoup

from core.config import LOGS_DIR, RAW_DIR
from scripts.scrapers.base import Fetcher, make_logger

BASE = "https://t.me"
PARSER_VERSION = "tg-web-1.0"
TG_DIR = RAW_DIR / "telegram"
STATE_DIR = TG_DIR / "state"
COLUMNS = ["source", "source_type", "source_name", "source_channel", "source_message_id", "source_url", "message_date",
           "raw_text", "text_hash", "edited", "forwarded_from", "has_photo", "has_video", "has_document", "media_only",
           "link_urls", "views", "collected_at", "last_seen_at", "collector", "parser_version"]
EDIT_RECHECK_PAGES = 2


def _hash(t: str) -> str:
    return hashlib.sha1((t or "").encode("utf-8")).hexdigest()[:16]


def parse_page(html: str, channel: str) -> tuple[list[dict], int | None, dict]:
    """(posts, oldest_id_on_page, channel_info). يقرأ فقط ما في HTML — لا قيم مستنتجة."""
    soup = BeautifulSoup(html, "html.parser")
    info = {}
    t = soup.select_one(".tgme_channel_info_header_title")
    info["title"] = t.get_text(" ", strip=True) if t else None
    for c in soup.select(".tgme_channel_info_counter"):
        v, k = c.select_one(".counter_value"), c.select_one(".counter_type")
        if v and k:
            info[k.get_text(strip=True)] = v.get_text(strip=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    posts = []
    for m in soup.select("div.tgme_widget_message[data-post]"):
        ch, _, mid = m["data-post"].rpartition("/")
        if not mid.isdigit() or ch.lower() != channel.lower():
            continue
        tm = m.select_one(".tgme_widget_message_date time[datetime]") or m.select_one("time[datetime]")
        txt_el = m.select_one(".tgme_widget_message_text")
        for br in (txt_el.find_all("br") if txt_el else []):
            br.replace_with("\n")
        text = txt_el.get_text("", strip=False).strip() if txt_el else ""
        links = [a["href"] for a in (txt_el.find_all("a", href=True) if txt_el else [])
                 if urlparse(a["href"]).scheme in ("http", "https")]
        meta = m.select_one(".tgme_widget_message_meta")
        fwd = m.select_one(".tgme_widget_message_forwarded_from_name")
        views = m.select_one(".tgme_widget_message_views")
        photo = bool(m.select_one(".tgme_widget_message_photo_wrap, .tgme_widget_message_grouped_wrap"))
        video = bool(m.select_one(".tgme_widget_message_video_player, .tgme_widget_message_roundvideo, video"))
        doc = bool(m.select_one(".tgme_widget_message_document"))
        posts.append({
            "source": f"telegram:{channel}", "source_type": "telegram", "source_name": info.get("title") or channel,
            "source_channel": channel, "source_message_id": int(mid), "source_url": f"{BASE}/{channel}/{mid}",
            "message_date": tm["datetime"] if tm else "", "raw_text": text, "text_hash": _hash(text),
            "edited": bool(meta and "edited" in meta.get_text(" ", strip=True).lower()),
            "forwarded_from": fwd.get_text(" ", strip=True) if fwd else "",
            "has_photo": photo, "has_video": video, "has_document": doc, "media_only": not text,
            "link_urls": " ".join(dict.fromkeys(links)), "views": views.get_text(strip=True) if views else "",
            "collected_at": now, "last_seen_at": now, "collector": "web_preview", "parser_version": PARSER_VERSION,
        })
    ids = [p["source_message_id"] for p in posts]
    return posts, (min(ids) if ids else None), info


@dataclass
class TGRunStats:
    channel: str
    mode: str
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    pages_visited: int = 0
    pages_failed: int = 0
    posts_seen: int = 0          # منشورات ظهرت في الصفحات (بما فيها المحفوظة سابقاً)
    new_posts: int = 0           # حُفظت لأول مرة
    already_had: int = 0         # موجودة مسبقاً — لم تُكرر
    edited_detected: int = 0     # تغيّر نصها منذ آخر جمع (النسخة القديمة في _edits.csv)
    media_only: int = 0          # من الجديدة: وسائط بلا نص
    total_in_file: int = 0
    oldest_id: int | None = None
    newest_id: int | None = None
    archive_complete: bool = False
    finished_at: str | None = None


class ChannelStore:
    def __init__(self, channel: str):
        self.channel = channel
        for d in (TG_DIR, STATE_DIR, LOGS_DIR):
            d.mkdir(parents=True, exist_ok=True)
        self.data_file = TG_DIR / f"{channel}.csv"
        self.edits_file = TG_DIR / f"{channel}_edits.csv"
        self.state_file = STATE_DIR / f"{channel}.json"
        self.fail_file = LOGS_DIR / f"tg_{channel}_failures.csv"
        self.runs_file = LOGS_DIR / f"tg_{channel}_runs.jsonl"

    def load(self) -> pd.DataFrame:
        if not self.data_file.exists():
            return pd.DataFrame(columns=COLUMNS)
        d = pd.read_csv(self.data_file, dtype=str, keep_default_na=False)
        d["source_message_id"] = d["source_message_id"].astype(int)
        return d

    def state(self) -> dict:
        return json.loads(self.state_file.read_text()) if self.state_file.exists() else {}

    def save_state(self, s: dict):
        self.state_file.write_text(json.dumps(s, ensure_ascii=False, indent=1))

    def log_failure(self, url, reason, status=None):
        new = not self.fail_file.exists()
        pd.DataFrame([{"at": datetime.now().isoformat(timespec="seconds"), "channel": self.channel, "url": url,
                       "reason": reason, "http_status": status or ""}]) \
            .to_csv(self.fail_file, mode="a", header=new, index=False, encoding="utf-8-sig")

    def merge(self, posts: list[dict], st: TGRunStats) -> pd.DataFrame:
        """دمج بلا تكرار على (channel, message_id). المنشور الذي تغيّر نصه: تُحفظ النسخة القديمة ويُحدَّث."""
        old = self.load()
        if not posts:
            return old
        new = pd.DataFrame(posts).astype({c: str for c in COLUMNS if c != "source_message_id"})
        new = new.drop_duplicates("source_message_id", keep="first")
        idx = old.set_index("source_message_id") if len(old) else pd.DataFrame(columns=COLUMNS).set_index("source_message_id")
        edits, rows_new = [], []
        for r in new.itertuples(index=False):
            mid = r.source_message_id
            if mid in idx.index:
                st.already_had += 1
                prev = idx.loc[mid]
                if prev["text_hash"] != r.text_hash and r.raw_text:
                    st.edited_detected += 1
                    edits.append({"source_message_id": mid, "replaced_at": r.collected_at,
                                  "previous_text": prev["raw_text"], "previous_hash": prev["text_hash"]})
                    for c in ("raw_text", "text_hash", "edited", "link_urls", "media_only"):
                        idx.at[mid, c] = getattr(r, c)
                idx.at[mid, "last_seen_at"] = r.last_seen_at
                idx.at[mid, "views"] = r.views
            else:
                rows_new.append(r._asdict())
                st.new_posts += 1
                st.media_only += r.media_only == "True"
        out = pd.concat([idx.reset_index(), pd.DataFrame(rows_new)], ignore_index=True)
        out = out[COLUMNS].sort_values("source_message_id", ascending=False)
        out.to_csv(self.data_file, index=False, encoding="utf-8-sig")
        if edits:
            pd.DataFrame(edits).to_csv(self.edits_file, mode="a", header=not self.edits_file.exists(), index=False,
                                       encoding="utf-8-sig")
        return out


def _crawl(f: Fetcher, store: ChannelStore, st: TGRunStats, log, start_before: int | None, stop_at_or_below: int | None,
           max_pages: int, min_pages: int = 0) -> tuple[list[dict], int | None, bool, dict]:
    """يتنقل للخلف عبر ?before=. يتوقف عند: max_pages، أو بلوغ stop_at_or_below (بعد min_pages)، أو نهاية القناة."""
    posts, info, reached_start = [], {}, False
    before, last_oldest = start_before, None
    for p in range(max_pages):
        url = f"{BASE}/s/{store.channel}" + (f"?before={before}" if before else "")
        html, why, code = f.get(url)
        st.pages_visited += 1
        if not html:
            st.pages_failed += 1
            store.log_failure(url, why, code)
            log.warning("✖ %s — %s", url, why)
            break
        page, oldest, inf = parse_page(html, store.channel)
        info = info or inf
        if not page:
            if p == 0 and before is None:
                store.log_failure(url, "0 posts parsed (private channel, preview disabled, or layout changed)", code)
                log.error("لم يُقرأ أي منشور من %s", url)
            else:
                reached_start = True
            break
        posts += page
        st.posts_seen += len(page)
        log.info("صفحة %d: رسائل %d–%d", p + 1, oldest, max(x["source_message_id"] for x in page))
        if oldest <= 1 or oldest == last_oldest:
            reached_start = True
            break
        last_oldest = oldest
        if stop_at_or_below is not None and oldest <= stop_at_or_below and p + 1 >= min_pages:
            break
        before = oldest
    return posts, info, reached_start, info


def run(channel: str, mode: str = "all", max_pages: int = 300, delay: float = 3.0) -> TGRunStats:
    log = make_logger(f"tg_{channel}")
    store, st = ChannelStore(channel), TGRunStats(channel, mode)
    f = Fetcher(delay, log)
    s = store.state()
    have = store.load()
    max_have = int(have["source_message_id"].max()) if len(have) else None
    min_have = int(have["source_message_id"].min()) if len(have) else None
    pages_left = max_pages
    if mode in ("update", "all"):
        # من الأحدث حتى آخر محفوظ؛ وأحدث صفحتين دائماً لالتقاط التعديلات
        posts, info, done, _ = _crawl(f, store, st, log, None, max_have, pages_left, min_pages=EDIT_RECHECK_PAGES)
        pages_left -= st.pages_visited
        s["channel_info"] = info or s.get("channel_info", {})
        have = store.merge(posts, st)
        if max_have is None and done:
            s["archive_complete"] = True
    if mode in ("backfill", "all") and not s.get("archive_complete") and pages_left > 0 and len(have):
        start = int(have["source_message_id"].min())
        before_pages = st.pages_visited
        posts, _, done, _ = _crawl(f, store, st, log, start, None, pages_left)
        have = store.merge(posts, st)
        if done:
            s["archive_complete"] = True
        log.info("استكمال الأرشيف: %d صفحة", st.pages_visited - before_pages)
    st.total_in_file = len(have)
    if len(have):
        st.oldest_id, st.newest_id = int(have.source_message_id.min()), int(have.source_message_id.max())
        d = pd.to_datetime(have.message_date, errors="coerce", utc=True)
        s.update(oldest_id=st.oldest_id, newest_id=st.newest_id,
                 oldest_date=str(d.min()) if d.notna().any() else None,
                 newest_date=str(d.max()) if d.notna().any() else None)
    st.archive_complete = bool(s.get("archive_complete"))
    s["last_run"] = datetime.now().isoformat(timespec="seconds")
    store.save_state(s)
    st.finished_at = datetime.now().isoformat(timespec="seconds")
    with open(store.runs_file, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(st), ensure_ascii=False) + "\n")
    log.info("─" * 60)
    log.info("ملخص [%s/%s]: صفحات %d (فشل %d) | منشورات ظهرت %d | جديدة %d | موجودة %d | معدلة %d | وسائط بلا نص %d",
             channel, mode, st.pages_visited, st.pages_failed, st.posts_seen, st.new_posts, st.already_had,
             st.edited_detected, st.media_only)
    log.info("        المحفوظ فعلاً: %d منشوراً (رسائل %s–%s) | الأرشيف مكتمل: %s", st.total_in_file, st.oldest_id,
             st.newest_id, "نعم" if st.archive_complete else "لا — أعد التشغيل للاستكمال")
    return st
