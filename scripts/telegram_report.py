"""تقرير جودة منشورات Telegram المجمعة (حقائق محسوبة من الملفات فقط — لا تصنيف ولا تقدير).

    python -m scripts.telegram_report            → reports/telegram_collection.md + .json
يعرض لكل قناة: الحجم، المدى الزمني، التوزيع السنوي والشهري (آخر 24 شهراً)، الوسائط بلا نص، المعاد توجيهه،
أكثر نطاقات الروابط، النصوص المتطابقة حرفياً داخل القناة، والتطابق الحرفي بين القنوات.
التطابق هنا حرفي (text_hash) فقط؛ كشف التكرار الحقيقي بين المصادر في المرحلة 5.
"""
import json
from collections import Counter
from datetime import datetime
from urllib.parse import urlparse

import pandas as pd

from core.config import REPORTS_DIR
from scripts.collectors.telegram_web import TG_DIR


def channel_stats(d: pd.DataFrame) -> dict:
    dt = pd.to_datetime(d.message_date, errors="coerce", utc=True)
    texty = d[d.raw_text.str.strip() != ""]
    dups = texty.text_hash.duplicated(keep="first")
    doms = Counter(urlparse(u).netloc.lower().removeprefix("www.") for s in d.link_urls for u in s.split() if u)
    m = dt.dt.tz_localize(None).dt.to_period("M")
    last = pd.period_range(end=pd.Timestamp(datetime.now()).to_period("M"), periods=24, freq="M")
    monthly = m.value_counts().reindex(last, fill_value=0)
    return {
        "posts": len(d), "with_text": len(texty), "media_only": int((d.media_only == "True").sum()),
        "forwarded": int((d.forwarded_from != "").sum()),
        "first_date": str(dt.min().date()) if dt.notna().any() else None,
        "last_date": str(dt.max().date()) if dt.notna().any() else None,
        "per_year": {str(k): int(v) for k, v in dt.dt.year.value_counts().sort_index().items()},
        "last_24_months": {str(k): int(v) for k, v in monthly.items()},
        "months_with_10plus_last_24": int((monthly >= 10).sum()),
        "exact_duplicate_texts": int(dups.sum()),
        "top_link_domains": dict(doms.most_common(8)),
        "top_forwarded_from": dict(Counter(x for x in d.forwarded_from if x).most_common(5)),
    }


def main():
    files = sorted(p for p in TG_DIR.glob("*.csv") if not p.stem.endswith("_edits"))
    data = {f.stem: pd.read_csv(f, dtype=str, keep_default_na=False) for f in files}
    rep = {"generated_at": datetime.now().isoformat(timespec="seconds"), "channels": {}}
    for ch, d in data.items():
        rep["channels"][ch] = channel_stats(d)
    # تطابق حرفي بين القنوات
    owners: dict[str, set] = {}
    for ch, d in data.items():
        for h in set(d.loc[d.raw_text.str.strip() != "", "text_hash"]):
            owners.setdefault(h, set()).add(ch)
    pairs = Counter(tuple(sorted(s)) for s in owners.values() if len(s) > 1)
    rep["cross_channel_exact_matches"] = {" + ".join(k): v for k, v in pairs.items()}

    L = [f"# تقرير جمع Telegram — {rep['generated_at']}",
         "حقائق محسوبة من الملفات المحفوظة. منشورات خام وليست وظائف مصنفة.", "",
         "| القناة | منشورات | بنص | وسائط بلا نص | معاد توجيهه | من | إلى | أشهر ≥10 منشورات (آخر 24) | نصوص متطابقة حرفياً |",
         "|---|---|---|---|---|---|---|---|---|"]
    for ch, s in rep["channels"].items():
        L.append(f"| @{ch} | {s['posts']} | {s['with_text']} | {s['media_only']} | {s['forwarded']} | {s['first_date']} | "
                 f"{s['last_date']} | {s['months_with_10plus_last_24']}/24 | {s['exact_duplicate_texts']} |")
    for ch, s in rep["channels"].items():
        L += ["", f"## @{ch}", f"- حسب السنة: {s['per_year']}",
              "- آخر 24 شهراً: " + " ".join(f"{k[2:]}:{v}" for k, v in s["last_24_months"].items()),
              f"- نطاقات الروابط: {s['top_link_domains']}", f"- أكثر المصادر المعاد توجيهها: {s['top_forwarded_from']}"]
    L += ["", "## تطابق حرفي بين القنوات", str(rep["cross_channel_exact_matches"] or "لا يوجد")]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "telegram_collection.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    (REPORTS_DIR / "telegram_collection.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
