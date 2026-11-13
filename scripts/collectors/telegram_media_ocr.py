"""OCR لمنشورات Telegram المصوّرة التي لا نص لها (media_only) — لا يعيد جمع المنشورات، يجلب صورها فقط.

الآلية (مقتصدة في الطلبات):
  1) من ملفات المنشورات المحفوظة: معرفات المنشورات بلا نص ولها صورة (has_photo=True) لكل قناة.
  2) صفحة معاينة واحدة t.me/s/<ch>?before=<id+1> تعرض حتى 20 منشوراً → روابط صور كل المنشورات المطلوبة فيها دفعة واحدة.
  3) تنزيل الصورة (CDN) ثم OCR فوراً (Tesseract ara+eng) — روابط CDN قد تنتهي صلاحيتها، فلا تُؤجَّل.
الحفظ تدريجي وآمن للانقطاع:
  data/raw/telegram/ocr/ocr_texts.csv    صف لكل صورة: النص، متوسط ثقة Tesseract، المحرك، الرابط
  data/raw/telegram/ocr/state.json       المنشورات المنتهية (نجاح أو فشل دائم) — لا تُعاد
  data/raw/logs/ocr_failures.csv         كل فشل وسببه
نص OCR يُعلَّم text_source=ocr في الـ Dataset ولا يُقبل تصنيفه آلياً (مراجعة)، لأن OCR قد يخطئ في القراءة.
"""
from __future__ import annotations

import io
import json
import re
import time
from datetime import datetime, timezone

import pandas as pd
from bs4 import BeautifulSoup

from core.config import LOGS_DIR
from scripts.collectors.telegram_web import BASE, TG_DIR
from scripts.scrapers.base import Fetcher, make_logger

OCR_DIR = TG_DIR / "ocr"
COLS = ["source_channel", "source_message_id", "image_index", "image_url", "ocr_text", "ocr_conf", "ocr_engine", "fetched_at"]
BG_URL = re.compile(r"background-image:\s*url\(['\"]?([^'\")]+)['\"]?\)")


def photo_urls(html: str, channel: str) -> dict[int, list[str]]:
    """{message_id: [روابط الصور]} من صفحة معاينة القناة (الصور تظهر كـ background-image في tgme_widget_message_photo_wrap)."""
    soup = BeautifulSoup(html, "html.parser")
    out = {}
    for m in soup.select("div.tgme_widget_message[data-post]"):
        ch, _, mid = m["data-post"].rpartition("/")
        if not mid.isdigit() or ch.lower() != channel.lower():
            continue
        urls = []
        for a in m.select(".tgme_widget_message_photo_wrap"):
            g = BG_URL.search(a.get("style", ""))
            if g and g.group(1).startswith("http"):
                urls.append(g.group(1))
        if urls:
            out[int(mid)] = list(dict.fromkeys(urls))
    return out


def binarize(gray):
    """عتبة Otsu → نص أسود على خلفية بيضاء (يُعكس إن كانت الخلفية داكنة)."""
    import numpy as np
    from PIL import Image
    a = np.asarray(gray, dtype=np.uint8)
    hist = np.bincount(a.ravel(), minlength=256).astype(float)
    p = hist / hist.sum()
    omega, mu = np.cumsum(p), np.cumsum(p * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega))
    t = int(np.nanargmax(sb))
    bw = np.where(a > t, 255, 0).astype(np.uint8)
    if bw.mean() < 127:                    # خلفية داكنة ونص فاتح
        bw = 255 - bw
    return Image.fromarray(bw)


GOOD_WORD = re.compile(r"^(?:[\u0600-\u06FF]{2,}|[A-Za-z]{3,}|\d{2,}|[\w.+-]+@[\w.-]+|https?://\S+)$")


def _ocr_once(img, psm: int = 3) -> tuple[str, float, int]:
    """(النص، متوسط الثقة، عدد الكلمات «السليمة»). كلمات بعلامات اتجاه \u200e/\u200f = قراءة خاطئة غالباً ولا تُحسب."""
    import pytesseract
    d = pytesseract.image_to_data(img, lang="ara+eng", config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
    lines, cur, key, confs = [], [], None, []
    for i, w in enumerate(d["text"]):
        c = float(d["conf"][i])
        if not str(w).strip() or c < 0:
            continue
        k = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        if k != key and cur:
            lines.append(" ".join(cur))
            cur = []
        key = k
        cur.append(str(w))
        confs.append(c)
    if cur:
        lines.append(" ".join(cur))
    text = "\n".join(lines).strip()
    good = sum(1 for w in text.split() if GOOD_WORD.match(w) and "\u200e" not in w and "\u200f" not in w)
    return text, (round(sum(confs) / len(confs), 1) if confs else 0.0), good


def ocr_image(data: bytes) -> tuple[str, float]:
    """(النص، متوسط الثقة 0–100). Tesseract غير مستقر مع مقياس الصورة ونمط التقسيم (اختبارات: سطر أول يسقط عند ×2.14،
    وخلفية ملونة تتشوه مع psm 3 وتُقرأ صحيحة مع psm 6) → ثلاث قراءات واختيار معلّل. لا تصحيح لغوي للنص."""
    from PIL import Image, ImageOps
    img = binarize(ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("L"))
    fa, fb = (1.5, 2.5) if min(img.size) < 900 else (1.0, 1.5)
    results = []
    for f, psm in ((fa, 3), (fa, 6), (fb, 6)):
        im = img.resize((int(img.width * f), int(img.height * f))) if f != 1.0 else img
        results.append(_ocr_once(ImageOps.expand(im, border=80, fill=255), psm))  # الهامش بعد التحويل لأبيض/أسود
    mx = max(r[2] for r in results)
    if mx == 0:
        return "", 0.0
    # المتغير الأعلى ثقة من بين ما استرجع ≥70% من أكبر عدد كلمات سليمة (يمنع اختيار قراءة ناقصة أو مشوهة)
    best = max((r for r in results if r[2] >= 0.7 * mx), key=lambda r: r[1])
    return best[0], best[1]


def targets(channels: list[str]) -> dict[str, list[int]]:
    out = {}
    for ch in channels:
        f = TG_DIR / f"{ch}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, dtype=str, keep_default_na=False, usecols=lambda c: c in
                        {"source_message_id", "raw_text", "has_photo", "media_only"})
        photo = d.has_photo == "True" if "has_photo" in d else d.get("media_only", pd.Series("", index=d.index)) == "True"
        d = d[(d.raw_text.str.strip() == "") & photo]
        out[ch] = sorted(d.source_message_id.astype(int), reverse=True)
    return out


def run(channels: list[str], budget_minutes: float = 120, delay: float = 3.0, limit: int | None = None) -> dict:
    log = make_logger("ocr")
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_f, state_f, fail_f = OCR_DIR / "ocr_texts.csv", OCR_DIR / "state.json", LOGS_DIR / "ocr_failures.csv"
    state = json.loads(state_f.read_text()) if state_f.exists() else {"done": {}}
    done = {ch: set(v) for ch, v in state["done"].items()}
    f = Fetcher(delay, log)
    cdn = Fetcher(max(1.0, delay / 3), log)          # ملفات ثابتة على CDN: تأخير أقل، مع احترام نفس آلية الإعادة
    t_end = time.time() + budget_minutes * 60
    stats = {"posts_targeted": 0, "pages_fetched": 0, "images_ocr": 0, "posts_with_text": 0, "posts_failed": 0,
             "remaining": 0, "stopped_by_budget": False}

    def fail(ch, mid, reason, permanent=True):
        pd.DataFrame([{"at": datetime.now().isoformat(timespec="seconds"), "source_channel": ch, "source_message_id": mid,
                       "reason": reason, "permanent": permanent}]).to_csv(fail_f, mode="a", header=not fail_f.exists(),
                                                                          index=False, encoding="utf-8-sig")
        if permanent:
            done.setdefault(ch, set()).add(int(mid))

    def save_state():
        state_f.write_text(json.dumps({"done": {k: sorted(v) for k, v in done.items()},
                                       "updated": datetime.now().isoformat(timespec="seconds")}))

    tg = targets(channels)
    n_attempt = 0
    for ch, ids in tg.items():
        todo = [i for i in ids if i not in done.get(ch, set())]
        stats["posts_targeted"] += len(ids)
        log.info("@%s: منشورات صور بلا نص %d | منتهية سابقاً %d | للمعالجة %d", ch, len(ids), len(ids) - len(todo), len(todo))
        while todo:
            if time.time() > t_end or (limit is not None and n_attempt >= limit):
                stats["stopped_by_budget"] = True
                break
            top = todo[0]
            url = f"{BASE}/s/{ch}?before={top + 1}"
            html, why, code = f.get(url)
            stats["pages_fetched"] += 1
            if not html:
                fail(ch, top, f"page: {why}", permanent=code in (404, 410))
                todo.pop(0)
                continue
            found = photo_urls(html, ch)
            on_page = [i for i in todo if i in found] or [top]
            rows = []
            for mid in on_page:
                n_attempt += 1
                if mid not in found:
                    fail(ch, mid, "no photo url in preview (deleted, album layout or video)")
                    continue
                texts = []
                for k, iu in enumerate(found[mid][:4]):          # حتى 4 صور لكل منشور (ألبوم)
                    data_ok = False
                    try:
                        r = cdn.s.get(iu, timeout=30)
                        data_ok = r.status_code == 200 and r.content
                        if data_ok:
                            txt, conf = ocr_image(r.content)
                            stats["images_ocr"] += 1
                            rows.append({"source_channel": ch, "source_message_id": mid, "image_index": k, "image_url": iu,
                                         "ocr_text": txt, "ocr_conf": conf, "ocr_engine": "tesseract ara+eng psm3",
                                         "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                            texts.append(txt)
                        else:
                            fail(ch, mid, f"image HTTP {r.status_code}", permanent=False)
                    except Exception as e:  # noqa: BLE001
                        fail(ch, mid, f"image/ocr: {type(e).__name__}: {str(e)[:120]}", permanent=False)
                    time.sleep(cdn.delay)
                if texts:
                    stats["posts_with_text"] += any(len(t) >= 25 for t in texts)
                    done.setdefault(ch, set()).add(mid)
                else:
                    stats["posts_failed"] += 1
            if rows:
                pd.DataFrame(rows, columns=COLS).to_csv(out_f, mode="a", header=not out_f.exists(), index=False,
                                                        encoding="utf-8-sig")
            done_set = done.get(ch, set())
            # المنشور الذي فشل مؤقتاً يُزال من قائمة هذا التشغيل (يعاد في تشغيل لاحق)
            todo = [i for i in todo if i not in done_set and i not in on_page]
            save_state()
            if stats["pages_fetched"] % 20 == 0:
                log.info("تقدم @%s: صفحات %d | صور OCR %d | منشورات بنص %d | متبقٍ في القناة %d",
                         ch, stats["pages_fetched"], stats["images_ocr"], stats["posts_with_text"], len(todo))
        if stats["stopped_by_budget"]:
            break
    stats["remaining"] = sum(len([i for i in ids if i not in done.get(ch, set())]) for ch, ids in tg.items())
    save_state()
    log.info("ملخص OCR: %s", stats)
    return stats
