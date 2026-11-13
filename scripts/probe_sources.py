"""المرحلة 1 — فحص قابلية الجمع لكل مصدر (لا يحفظ أي بيانات جمع؛ يكتب تقريراً فقط).

    python -m scripts.probe_sources                    # كل المصادر في data/sources.yaml
    python -m scripts.probe_sources --pages 3          # صفحات أقدم لكل قناة (عينة تاريخية)
    python -m scripts.probe_sources --only hr_yemen

لكل قناة Telegram عامة يقرأ صفحات المعاينة t.me/s/<channel> (HTML حقيقي):
  data-post (رقم الرسالة)، <time datetime> (تاريخ النشر الدقيق)، النص، إعادة التوجيه، الوسائط، الروابط الخارجية.
وإن وُجدت TG_API_ID و TG_API_HASH (متغيرات بيئة أو Colab Secrets) يختبر Telethon: عدد الرسائل وتاريخ أول رسالة.
لكل موقع: robots.txt، sitemap، wp-json، RSS، وروابط الصفحات التي تشبه الوظائف.

الناتج: reports/source_probe.json و reports/source_probe.md
تصنيف نوع المنشور هنا تقديري بالكلمات المفتاحية — لأغراض الفحص فقط، وليس المصنّف النهائي.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from core.config import REPORTS_DIR, ROOT

UA = "Mozilla/5.0 (compatible; LMI-GraduationProject/2.0; academic labor-market research)"
DELAY = 3.0
SOURCES_FILE = ROOT / "data" / "sources.json"

DEFAULT_SOURCES = {
    "telegram": ["hr_yemen", "job_yemen", "jobsforyemeni", "TheYemenGate", "forus_taiz"],
    "websites": ["https://yemenforus.com"],
}

# تقدير أولي لنوع المنشور (للفحص فقط)
TYPE_HINTS = [
    ("SCHOLARSHIP", r"منح(ه|ة) |منحة|scholarship|fellowship|زمال(ه|ة)"),
    ("INTERNSHIP", r"\bintern(ship)?\b|تدريب ميداني|تدريب عملي|متدرب"),
    ("TRAINING", r"دور(ه|ة) تدريبي|دورة|ورش(ه|ة) عمل|\bcourse\b|\btraining\b|\bworkshop\b|دبلوم"),
    ("VOLUNTEERING", r"volunteer|تطوع|متطوع"),
    ("JOB", r"مطلوب|وظيف|فرص(ه|ة) عمل|vacanc|\bjob\b|position|hiring|officer|manager|engineer|مسؤول|محاسب|مهندس|سائق|معلم"),
    ("NOT_RELEVANT", r"مفقود|ضائع|عثر|فقدان|للبيع|tiktok|إعلان تجاري"),
]
OUTSIDE = r"السعودي|الرياض|جدة|الدمام|الإمارات|دبي|ابوظبي|أبوظبي|قطر|الدوحة|الكويت|عمان|مسقط|البحرين|مصر|القاهرة|الأردن|عمّان|saudi|riyadh|jeddah|dubai|uae|qatar|kuwait|oman|bahrain|egypt|jordan"
REMOTE = r"\bremote\b|عن بعد|عن بُعد|من المنزل|work from home|أونلاين|اونلاين"
YEMEN = r"صنعاء|عدن|تعز|الحديدة|إب|اب |المكلا|حضرموت|مأرب|مارب|ذمار|لحج|أبين|شبوة|المهرة|حجة|عمران|البيضاء|الضالع|سيئون|sana'?a|aden|taiz|hodeidah|marib|mukalla|hadramout|ibb|lahj|abyan|shabwa|yemen|اليمن"


def guess_type(t: str) -> str:
    if not t.strip():
        return "MEDIA_OR_EMPTY"
    low = t.lower()
    for name, rx in TYPE_HINTS:
        if re.search(rx, low):
            return name
    return "OTHER"


def guess_scope(t: str) -> str:
    low = t.lower()
    if re.search(REMOTE, low):
        return "REMOTE_GLOBAL"
    if re.search(OUTSIDE, low):
        return "OUTSIDE_YEMEN" if not re.search(YEMEN, low) else "MIXED_CHECK"
    if re.search(YEMEN, low):
        return "YEMEN"
    return "UNKNOWN"


class Polite:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept-Language": "ar,en;q=0.8"})
        self.last = 0.0

    def get(self, url, **kw):
        wait = DELAY - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        self.last = time.time()
        try:
            r = self.s.get(url, timeout=30, **kw)
            return r, None
        except requests.RequestException as e:
            return None, f"{type(e).__name__}: {str(e)[:160]}"


# ------------------------------------------------------------------ Telegram (صفحات المعاينة العامة)
def parse_tg_page(html: str, channel: str) -> tuple[list[dict], int | None, dict]:
    """(posts, before_id_for_older_page, channel_info)"""
    soup = BeautifulSoup(html, "html.parser")
    info = {}
    t = soup.select_one(".tgme_channel_info_header_title")
    info["title"] = t.get_text(" ", strip=True) if t else None
    for c in soup.select(".tgme_channel_info_counter"):
        v, k = c.select_one(".counter_value"), c.select_one(".counter_type")
        if v and k:
            info[k.get_text(strip=True)] = v.get_text(strip=True)
    posts = []
    for m in soup.select("div.tgme_widget_message[data-post]"):
        dp = m["data-post"]
        mid = int(dp.split("/")[-1]) if dp.split("/")[-1].isdigit() else None
        tm = m.select_one("time[datetime]")
        txt_el = m.select_one(".tgme_widget_message_text")
        text = txt_el.get_text("\n", strip=True) if txt_el else ""
        links = [a["href"] for a in (txt_el.find_all("a", href=True) if txt_el else [])]
        ext = [urlparse(u).netloc for u in links if urlparse(u).netloc and "t.me" not in urlparse(u).netloc]
        posts.append({
            "message_id": mid, "url": f"https://t.me/{dp}",
            "date": tm["datetime"] if tm else None,
            "edited": bool(m.select_one(".tgme_widget_message_meta .tgme_widget_message_edited") or
                           "edited" in (m.select_one(".tgme_widget_message_meta").get_text() if m.select_one(".tgme_widget_message_meta") else "")),
            "forwarded_from": (m.select_one(".tgme_widget_message_forwarded_from_name") or {}).get_text(strip=True)
                              if m.select_one(".tgme_widget_message_forwarded_from_name") else None,
            "has_photo": bool(m.select_one(".tgme_widget_message_photo_wrap")),
            "has_video": bool(m.select_one(".tgme_widget_message_video_player, .tgme_widget_message_roundvideo")),
            "has_document": bool(m.select_one(".tgme_widget_message_document")),
            "text_len": len(text), "text_sample": text[:160].replace("\n", " "),
            "link_domains": sorted(set(ext)), "type_guess": guess_type(text), "scope_guess": guess_scope(text),
        })
    ids = [p["message_id"] for p in posts if p["message_id"]]
    return posts, (min(ids) if ids else None), info


def probe_telegram(ch: str, pages: int, http: Polite) -> dict:
    out = {"source": f"telegram:{ch}", "public_preview": None, "errors": [], "pages_read": 0}
    url, all_posts, info = f"https://t.me/s/{ch}", [], {}
    for i in range(pages):
        r, err = http.get(url)
        if err or r is None or r.status_code != 200:
            out["errors"].append({"url": url, "error": err or f"HTTP {r.status_code}"})
            break
        posts, before, inf = parse_tg_page(r.text, ch)
        info = info or inf
        out["pages_read"] += 1
        if not posts:
            out["errors"].append({"url": url, "error": "page loaded but 0 posts parsed (private, no preview, or layout changed)"})
            break
        all_posts += posts
        if not before or before <= 1:
            break
        # الصفحة الأولى = الأحدث؛ الصفحات التالية تقفز بعيداً للخلف لأخذ عينة تاريخية
        jump = before if i == 0 else max(1, before - before // 3)
        url = f"https://t.me/s/{ch}?before={jump}"
    out["public_preview"] = bool(all_posts)
    out["channel_info"] = info
    if all_posts:
        dates = sorted(p["date"] for p in all_posts if p["date"])
        newest = max(p["message_id"] for p in all_posts if p["message_id"])
        out.update({
            "posts_sampled": len(all_posts), "newest_message_id": newest,
            "approx_total_messages": newest, "dates_in_sample": {"oldest": dates[0] if dates else None,
                                                                  "newest": dates[-1] if dates else None},
            "date_attribute_present_pct": round(100 * sum(bool(p["date"]) for p in all_posts) / len(all_posts), 1),
            "type_guess": dict(Counter(p["type_guess"] for p in all_posts)),
            "scope_guess": dict(Counter(p["scope_guess"] for p in all_posts)),
            "media_only": sum(p["text_len"] == 0 for p in all_posts),
            "with_photo": sum(p["has_photo"] for p in all_posts), "with_document": sum(p["has_document"] for p in all_posts),
            "forwarded": sum(bool(p["forwarded_from"]) for p in all_posts), "edited": sum(p["edited"] for p in all_posts),
            "link_domains": dict(Counter(d for p in all_posts for d in p["link_domains"]).most_common(8)),
            "examples": [{k: p[k] for k in ("message_id", "date", "type_guess", "scope_guess", "text_sample")}
                         for p in all_posts[:3]],
        })
        last = dates[-1] if dates else None
        if last:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(last.replace("Z", "+00:00"))).days
            out["days_since_newest_post"] = age
            out["active"] = age <= 30
    return out


def probe_telethon(channels: list[str]) -> dict:
    """يتطلب TG_API_ID و TG_API_HASH و (اختياري) TG_SESSION — لا تُطبع ولا تُحفظ أي قيمة سرية."""
    api_id, api_hash = os.environ.get("TG_API_ID"), os.environ.get("TG_API_HASH")
    if not (api_id and api_hash):
        try:
            from google.colab import userdata  # type: ignore
            api_id, api_hash = userdata.get("TG_API_ID"), userdata.get("TG_API_HASH")
        except Exception:  # noqa: BLE001
            pass
    if not (api_id and api_hash):
        return {"status": "skipped", "reason": "TG_API_ID / TG_API_HASH غير متوفرة (Colab Secrets أو متغيرات البيئة)"}
    try:
        from telethon.sync import TelegramClient  # noqa: F401
    except ImportError:
        return {"status": "skipped", "reason": "telethon غير مثبتة: pip install telethon"}
    from telethon.sync import TelegramClient
    sess = os.environ.get("TG_SESSION_PATH", str(ROOT / ".secrets" / "lmi_probe"))
    os.makedirs(os.path.dirname(sess), exist_ok=True)
    out = {"status": "ok", "channels": {}}
    with TelegramClient(sess, int(api_id), api_hash) as client:
        for ch in channels:
            try:
                ent = client.get_entity(ch)
                first = next(iter(client.iter_messages(ent, reverse=True, limit=1)), None)
                last = next(iter(client.iter_messages(ent, limit=1)), None)
                total = client.get_messages(ent, limit=0).total
                out["channels"][ch] = {"total_messages": total,
                                       "first_message_date": first.date.isoformat() if first else None,
                                       "last_message_date": last.date.isoformat() if last else None}
            except Exception as e:  # noqa: BLE001
                out["channels"][ch] = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
    return out


# ------------------------------------------------------------------ مواقع
def probe_website(base: str, http: Polite) -> dict:
    out = {"source": base, "errors": []}
    r, err = http.get(base)
    if err or r is None:
        out.update(reachable=False, errors=[{"url": base, "error": err}])
        return out
    out.update(reachable=r.status_code == 200, http_status=r.status_code, final_url=r.url,
               server=r.headers.get("server"), content_type=r.headers.get("content-type"))
    html = r.text
    soup = BeautifulSoup(html, "html.parser")
    out["title"] = soup.title.get_text(strip=True) if soup.title else None
    out["generator"] = (soup.find("meta", attrs={"name": "generator"}) or {}).get("content") if soup.find("meta", attrs={"name": "generator"}) else None
    out["looks_js_rendered"] = len(soup.get_text(" ", strip=True)) < 500 and bool(soup.find("script"))
    links = {urljoin(r.url, a["href"]).split("#")[0] for a in soup.find_all("a", href=True)}
    host = urlparse(r.url).netloc
    internal = sorted(u for u in links if urlparse(u).netloc == host)
    job_like = [u for u in internal if re.search(r"job|vacanc|career|وظ|فرص|opportunit|tender|منح|scholar|training", u, re.I)]
    out["internal_links"] = len(internal)
    out["job_like_links_sample"] = job_like[:20]
    for path, key in (("/robots.txt", "robots"), ("/sitemap.xml", "sitemap"), ("/sitemap_index.xml", "sitemap_index"),
                      ("/wp-json/wp/v2/types", "wp_json"), ("/feed", "rss"), ("/terms", "terms"), ("/privacy-policy", "privacy")):
        rr, e = http.get(urljoin(r.url, path))
        out[key] = {"status": rr.status_code if rr is not None else None, "error": e,
                    "sample": rr.text[:600] if rr is not None and rr.status_code == 200 else None}
    # صفحة واحدة تشبه قائمة وظائف: عدد الروابط وبنية الترقيم
    if job_like:
        jr, e = http.get(job_like[0])
        if jr is not None and jr.status_code == 200:
            js = BeautifulSoup(jr.text, "html.parser")
            jl = {urljoin(jr.url, a["href"]) for a in js.find_all("a", href=True)}
            out["first_job_page"] = {"url": jr.url, "links": len(jl),
                                     "pagination_links": sorted(u for u in jl if re.search(r"page[=/]\d+|paged=|/p/\d+", u))[:5],
                                     "has_time_tags": bool(js.find("time")),
                                     "date_like_strings": re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\d{4}-\d{2}-\d{2}", js.get_text(" "))[:5]}
    return out


def render_md(rep: dict) -> str:
    L = [f"# تقرير قابلية الجمع (المرحلة 1) — {rep['generated_at']}", "",
         "أعداد وتواريخ من HTML الحقيقي وقت الفحص. تصنيف الأنواع تقديري بالكلمات المفتاحية.", "",
         "| المصدر | معاينة عامة | آخر رقم رسالة | أحدث منشور | نشطة | عينة | أنواع (تقدير) | نطاق (تقدير) | وسائط بلا نص | معاد توجيهه | روابط خارجية |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in rep["telegram"]:
        if not t.get("public_preview"):
            L.append(f"| {t['source']} | ✖ | — | — | — | — | — | — | — | — | {t['errors'][:1]} |")
            continue
        L.append(f"| {t['source']} | ✔ | {t['newest_message_id']} | {t['dates_in_sample']['newest']} | "
                 f"{'نعم' if t.get('active') else 'لا'} | {t['posts_sampled']} | {t['type_guess']} | {t['scope_guess']} | "
                 f"{t['media_only']} | {t['forwarded']} | {t['link_domains']} |")
    L += ["", "## Telethon", "```", json.dumps(rep["telethon"], ensure_ascii=False, indent=1), "```", "", "## المواقع"]
    for w in rep["websites"]:
        L += [f"### {w['source']}", "```", json.dumps({k: v for k, v in w.items() if k != "source"}, ensure_ascii=False,
                                                       indent=1)[:3000], "```"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=3, help="صفحات لكل قناة (الأولى الأحدث ثم عينات أقدم)")
    ap.add_argument("--only", help="قناة أو موقع واحد")
    a = ap.parse_args()
    src = json.loads(SOURCES_FILE.read_text(encoding="utf-8")) if SOURCES_FILE.exists() else DEFAULT_SOURCES
    tg = [c["username"] if isinstance(c, dict) else c for c in src["telegram"]]
    web = [w["url"] if isinstance(w, dict) else w for w in src["websites"]]
    tg = [c for c in tg if not a.only or c == a.only]
    web = [w for w in web if not a.only or a.only in w]
    http = Polite()
    rep = {"generated_at": datetime.now().isoformat(timespec="minutes"), "telegram": [], "websites": []}
    for ch in tg:
        print(f"══ telegram:{ch}")
        res = probe_telegram(ch, a.pages, http)
        print(json.dumps({k: res.get(k) for k in ("public_preview", "newest_message_id", "dates_in_sample", "active",
                                                  "type_guess", "errors")}, ensure_ascii=False))
        rep["telegram"].append(res)
    rep["telethon"] = probe_telethon(tg)
    print("telethon:", rep["telethon"].get("status"), rep["telethon"].get("reason", ""))
    for w in web:
        print(f"══ {w}")
        res = probe_website(w, http)
        print(json.dumps({k: res.get(k) for k in ("reachable", "http_status", "title", "generator", "looks_js_rendered",
                                                  "job_like_links_sample")}, ensure_ascii=False)[:800])
        rep["websites"].append(res)
    (REPORTS_DIR / "source_probe.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    (REPORTS_DIR / "source_probe.md").write_text(render_md(rep), encoding="utf-8")
    print(f"\n✔ التقرير: {REPORTS_DIR / 'source_probe.md'}")


if __name__ == "__main__":
    main()
