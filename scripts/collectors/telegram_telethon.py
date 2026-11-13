"""جامع Telegram عبر Telethon (اختياري) — يعطي تاريخ التعديل الدقيق ونوع الوسائط والمعاد توجيهه.

الأسرار لا تُكتب في أي ملف: TG_API_ID و TG_API_HASH من متغيرات البيئة أو Colab Secrets.
ملف الجلسة (session) يُحفظ في .secrets/ (مستثنى في .gitignore) — لا ترفعه إلى GitHub.
أول تشغيل يطلب رقم الهاتف ورمز التحقق تفاعلياً في Colab.

المخرجات بنفس مخطط telegram_web وفي نفس الملف، مع collector=telethon و edit_date.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from core.config import ROOT
from scripts.collectors.telegram_web import PARSER_VERSION, ChannelStore, TGRunStats, _hash

SESSION = os.environ.get("TG_SESSION_PATH", str(ROOT / ".secrets" / "lmi_collector"))


def credentials() -> tuple[str | None, str | None]:
    api_id, api_hash = os.environ.get("TG_API_ID"), os.environ.get("TG_API_HASH")
    if not (api_id and api_hash):
        try:
            from google.colab import userdata  # type: ignore
            api_id, api_hash = userdata.get("TG_API_ID"), userdata.get("TG_API_HASH")
        except Exception:  # noqa: BLE001
            pass
    return api_id, api_hash


def message_to_row(m, channel: str, title: str) -> dict:
    """تحويل رسالة Telethon إلى صف المخطط الموحد للمنشورات الخام (دالة مستقلة لتسهيل الاختبار)."""
    text = m.message or ""
    urls = []
    for e in (m.entities or []):
        u = getattr(e, "url", None)
        if u:
            urls.append(u)
        elif type(e).__name__ == "MessageEntityUrl":
            urls.append(text[e.offset:e.offset + e.length])
    media = type(m.media).__name__ if m.media else ""
    fwd = ""
    if m.fwd_from:
        fwd = getattr(m.fwd_from, "from_name", None) or str(getattr(getattr(m.fwd_from, "from_id", None), "channel_id", "") or "")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "source": f"telegram:{channel}", "source_type": "telegram", "source_name": title, "source_channel": channel,
        "source_message_id": int(m.id), "source_url": f"https://t.me/{channel}/{m.id}",
        "message_date": m.date.isoformat() if m.date else "", "raw_text": text, "text_hash": _hash(text),
        "edited": bool(m.edit_date), "forwarded_from": fwd,
        "has_photo": media == "MessageMediaPhoto", "has_video": media == "MessageMediaDocument" and bool(getattr(m, "video", None)),
        "has_document": media == "MessageMediaDocument" and not getattr(m, "video", None), "media_only": not text,
        "link_urls": " ".join(dict.fromkeys(urls)), "views": str(m.views or ""), "collected_at": now, "last_seen_at": now,
        "collector": "telethon", "parser_version": PARSER_VERSION,
    }


def run(channel: str, limit: int | None = None) -> TGRunStats:
    api_id, api_hash = credentials()
    if not (api_id and api_hash):
        raise SystemExit("TG_API_ID / TG_API_HASH غير متوفرة. أضفها في Colab Secrets (🔑) — لا تكتبها في الكود.")
    from telethon.sync import TelegramClient
    os.makedirs(os.path.dirname(SESSION), exist_ok=True)
    store, st = ChannelStore(channel), TGRunStats(channel, "telethon")
    have = store.load()
    max_have = int(have.source_message_id.max()) if len(have) else 0
    rows = []
    with TelegramClient(SESSION, int(api_id), api_hash) as client:
        ent = client.get_entity(channel)
        title = getattr(ent, "title", channel)
        # الجديد فقط بعد آخر محفوظ (min_id)، ثم الأقدم من أصغر محفوظ إن لم يكتمل الأرشيف
        for m in client.iter_messages(ent, min_id=max_have, limit=limit):
            rows.append(message_to_row(m, channel, title))
        if len(have):
            for m in client.iter_messages(ent, offset_id=int(have.source_message_id.min()), limit=limit):
                rows.append(message_to_row(m, channel, title))
    st.posts_seen = len(rows)
    out = store.merge(rows, st)
    st.total_in_file = len(out)
    return st
