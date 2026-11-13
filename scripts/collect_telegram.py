"""واجهة تشغيل جامع Telegram (المرحلة 2).

    python -m scripts.collect_telegram --channels hr_yemen job_yemen          # تحديث + استكمال الأرشيف
    python -m scripts.collect_telegram --channels hr_yemen --mode update      # الجديد فقط
    python -m scripts.collect_telegram --all-enabled                          # كل القنوات المفعّلة في data/sources.json
    python -m scripts.collect_telegram --backend telethon --channels hr_yemen # عبر Telethon (يتطلب Colab Secrets)
    python -m scripts.collect_telegram --status
    python -m scripts.collect_telegram --telegraph --channels hr_yemen --limit 1500   # تفاصيل telegra.ph (استكمال تلقائي)

يحفظ منشورات خاماً فقط. العدد المعروض = منشورات محفوظة فعلاً، لا روابط مكتشفة ولا وظائف (التصنيف في مرحلة لاحقة).
"""
import argparse
import json

import pandas as pd

from core.config import IS_DEMO_MODE, ROOT
from scripts.collectors import telegram_web
from scripts.collectors.telegram_web import STATE_DIR, TG_DIR

SOURCES = ROOT / "data" / "sources.json"


def enabled_channels() -> list[str]:
    src = json.loads(SOURCES.read_text(encoding="utf-8"))
    return [c["username"] if isinstance(c, dict) else c for c in src["telegram"]
            if not isinstance(c, dict) or c.get("enabled", True)]


def status():
    files = sorted(p for p in TG_DIR.glob("*.csv") if not p.stem.endswith("_edits"))
    if not files:
        print("لا توجد منشورات Telegram مجمعة بعد في", TG_DIR)
    for f in files:
        d = pd.read_csv(f, dtype=str, keep_default_na=False)
        dt = pd.to_datetime(d.message_date, errors="coerce", utc=True)
        s = json.loads((STATE_DIR / f"{f.stem}.json").read_text()) if (STATE_DIR / f"{f.stem}.json").exists() else {}
        ed = TG_DIR / f"{f.stem}_edits.csv"
        print(f"@{f.stem}: {len(d)} منشوراً محفوظاً | {dt.min():%Y-%m-%d} → {dt.max():%Y-%m-%d} | "
              f"بلا نص {int((d.media_only == 'True').sum())} | معاد توجيهه {int((d.forwarded_from != '').sum())} | "
              f"تعديلات مسجلة {len(pd.read_csv(ed)) if ed.exists() else 0} | الأرشيف مكتمل: {'نعم' if s.get('archive_complete') else 'لا'}")


if __name__ == "__main__":
    if IS_DEMO_MODE:
        raise SystemExit("الجمع متاح في الوضع الحقيقي فقط.")
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", nargs="*")
    ap.add_argument("--all-enabled", action="store_true")
    ap.add_argument("--mode", choices=["update", "backfill", "all"], default="all")
    ap.add_argument("--max-pages", type=int, default=300, help="حد الصفحات لكل قناة في هذا التشغيل (20 منشوراً/صفحة)")
    ap.add_argument("--delay", type=float, default=3.0)
    ap.add_argument("--backend", choices=["web", "telethon"], default="web")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--telegraph", action="store_true", help="جلب صفحات telegra.ph المرتبطة بمنشورات القنوات المحددة")
    ap.add_argument("--limit", type=int, help="حد صفحات telegra.ph في هذا التشغيل")
    ap.add_argument("--retry-failed", action="store_true")
    a = ap.parse_args()
    if a.status:
        status()
        from scripts.collectors import telegraph
        print()
        telegraph.status()
        raise SystemExit
    if a.telegraph:
        from scripts.collectors import telegraph
        telegraph.run(a.channels or ["hr_yemen"], a.limit, max(a.delay, 2.0), a.retry_failed)
        print()
        telegraph.status()
        raise SystemExit
    chans = enabled_channels() if a.all_enabled else (a.channels or [])
    if not chans:
        raise SystemExit("حدد --channels أو --all-enabled")
    for ch in chans:
        print(f"\n══ @{ch}")
        if a.backend == "telethon":
            from scripts.collectors import telegram_telethon
            telegram_telethon.run(ch)
        else:
            telegram_web.run(ch, a.mode, a.max_pages, max(a.delay, 2.0))
    print()
    status()
