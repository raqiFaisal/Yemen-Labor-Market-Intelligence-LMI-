"""جمع إعلانات الوظائف اليمنية — واجهة التشغيل.

    python -m scripts.scrape_yemen --site yemenhr --limit 5 --debug   # اختبار سريع
    python -m scripts.scrape_yemen --site yemenhr                     # جمع كل الإعلانات النشطة + التصنيفات
    python -m scripts.scrape_yemen --site all                         # + المصادر الأخرى (تجريبية)
    python -m scripts.scrape_yemen --site yemenhr --retry-failed      # إعادة محاولة الفشل النهائي (404...)
    python -m scripts.scrape_yemen --status                           # ملخص الملفات دون أي طلبات

الملفات (لكل مصدر على حدة، مع عمود source في كل سجل):
    data/raw/yemen_<source>.csv              الإعلانات المنزّلة بنجاح فقط
    data/raw/state/<source>_discovered.csv   الروابط المكتشفة (ليست إعلانات)
    data/raw/logs/<source>_failures.csv      كل فشل وسببه
    data/raw/logs/<source>_runs.jsonl        عدادات كل تشغيل
احترام الموقع: robots.txt، تأخير 3 ثوانٍ + عشوائية، طلب واحد في كل مرة، إعادة محاولة متدرجة عند 429/5xx.
شروط استخدام YemenHR (yemenhr.com/terms) لا تتناول الجمع الآلي — استخدم البيانات لأغراض بحثية مع ذكر المصدر.
"""
import argparse
import json

import pandas as pd

from core.config import IS_DEMO_MODE, LOGS_DIR, RAW_DIR, STATE_DIR
from scripts.scrapers import generic, yemenhr

OTHER = list(generic.SITES)  # shortlist, yemenboard


def status():
    files = sorted(RAW_DIR.glob("yemen_*.csv"))
    if not files:
        print("لا توجد ملفات مجمعة بعد في", RAW_DIR)
    for f in files:
        d = pd.read_csv(f, dtype=str)
        dt = pd.to_datetime(d.get("posted_date"), errors="coerce")
        cat = d["category"].fillna("").str.len().gt(0).mean() * 100 if "category" in d else 0
        rng = f"{dt.min():%Y-%m-%d} → {dt.max():%Y-%m-%d}" if dt.notna().any() else "بلا تواريخ"
        print(f"{f.name}: {len(d)} إعلان حقيقي | مؤرخ {dt.notna().mean() * 100:.0f}% ({rng}) | مصنف {cat:.0f}%")
        src = f.stem.replace("yemen_", "")
        disc, fail = STATE_DIR / f"{src}_discovered.csv", LOGS_DIR / f"{src}_failures.csv"
        if disc.exists():
            print(f"   روابط مكتشفة (تراكمي): {len(pd.read_csv(disc))}")
        if fail.exists():
            fl = pd.read_csv(fail, dtype=str)
            print(f"   سجلات فشل: {len(fl)} | روابط لم تنجح بعد: {len(set(fl.url) - set(d.get('url', [])))}")
        runs = LOGS_DIR / f"{src}_runs.jsonl"
        if runs.exists():
            last = json.loads(runs.read_text(encoding="utf-8").strip().split("\n")[-1])
            print(f"   آخر تشغيل {last['started_at']}: نجح {last['downloaded_ok']} | فشل {last['failed']}")


if __name__ == "__main__":
    if IS_DEMO_MODE:
        raise SystemExit("الجمع متاح في الوضع الحقيقي فقط (أزل LMI_MODE=demo).")
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", choices=["yemenhr", *OTHER, "all"], default="yemenhr")
    ap.add_argument("--max-pages", "--pages", type=int, default=50, dest="max_pages")
    ap.add_argument("--limit", type=int, default=0, help="أقصى عدد إعلانات تُنزَّل في هذا التشغيل (0 = الكل)")
    ap.add_argument("--delay", type=float, default=3.0, help="ثوانٍ بين الطلبات (حد أدنى 2)")
    ap.add_argument("--no-categories", action="store_true", help="عدم زيارة صفحات التصنيفات")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="إعادة تنزيل الموجود (لتحديث المحلل)")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.status:
        status()
        raise SystemExit
    delay = max(a.delay, 2.0)
    for s in (["yemenhr", *OTHER] if a.site == "all" else [a.site]):
        print(f"\n══ {s}")
        if s == "yemenhr":
            yemenhr.run(a.max_pages, a.limit, delay, not a.no_categories, a.retry_failed, a.refresh, a.debug)
        else:
            generic.run(s, a.max_pages, a.limit, delay, a.retry_failed, a.refresh, a.debug)
    print()
    status()
