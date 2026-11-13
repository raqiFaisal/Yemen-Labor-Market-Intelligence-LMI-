"""تنزيل المصادر الخارجية المفتوحة إلى data/raw/.

    python -m scripts.fetch_external arabjobs      # ArabJobs من Hugging Face
    python -m scripts.fetch_external onet  --path db_29_0_text.zip   # بعد تنزيل O*NET يدوياً
    python -m scripts.fetch_external esco  --path skills_ar.csv      # بعد تنزيل ESCO يدوياً

O*NET و ESCO لا تُضاف كوظائف، بل تُوسِّع قاموس المهارات (data/taxonomy/skills.json)،
مع الاحتفاظ فقط بالمهارات التي تظهر فعلاً في نصوص الإعلانات (--min-freq) حتى لا يتضخم القاموس.
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile

import pandas as pd

from core.config import RAW_DIR, SKILLS_TAXONOMY_FILE


def fetch_arabjobs():
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("ثبّت المكتبة: pip install datasets")
    ds = load_dataset("drelhaj/ArabJobs")
    frames = [ds[s].to_pandas() for s in ds]
    df = pd.concat(frames, ignore_index=True)
    out = RAW_DIR / "arabjobs.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"✔ ArabJobs: {len(df)} سجل • الأعمدة: {list(df.columns)} → {out}")
    print("  راجع أسماء الأعمدة؛ إن لم تُطابق تلقائياً أضفها إلى COLUMN_ALIASES في preprocessing/schema.py")


def _corpus_text() -> str:
    from core.pipeline import load_processed
    jobs, _ = load_processed()
    if jobs is None:
        raise SystemExit("شغّل scripts.train_all مرة أولاً لبناء نصوص الإعلانات.")
    return " ".join((jobs.title.fillna("") + " " + jobs.description.fillna("") + " " + jobs.requirements.fillna("")).str.lower())


def _merge(new_terms: list[tuple[str, str, str]], min_freq: int):
    """new_terms: (id, english, arabic_or_empty)"""
    corpus = _corpus_text()
    skills = json.loads(SKILLS_TAXONOMY_FILE.read_text(encoding="utf-8"))
    known = {t.lower() for s in skills for t in [s["name_en"], s["name_ar"], *s["aliases"]]}
    added = 0
    for sid, en, ar in new_terms:
        if en.lower() in known or len(en) < 3:
            continue
        freq = len(re.findall(rf"(?<![a-z]){re.escape(en.lower())}(?![a-z])", corpus))
        if freq >= min_freq:
            skills.append({"id": sid, "name_en": en, "name_ar": ar or en, "type": "technical", "domains": [], "aliases": []})
            known.add(en.lower())
            added += 1
    SKILLS_TAXONOMY_FILE.write_text(json.dumps(skills, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"✔ أضيفت {added} مهارة للقاموس (المجموع {len(skills)}). أعد تشغيل scripts.train_all.")


def import_onet(path: str, min_freq: int):
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith("Technology Skills.txt"))
        df = pd.read_csv(z.open(name), sep="\t")
    ex = df["Example"].dropna().unique()
    slug = lambda e: "onet_" + re.sub(r"\W+", "_", e.lower())[:40]
    _merge([(slug(e), e, "") for e in ex], min_freq)


def import_esco(path: str, min_freq: int):
    df = pd.read_csv(path)
    col = "preferredLabel"
    _merge([(f"esco_{i}", str(l), "") for i, l in enumerate(df[col].dropna())], min_freq)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["arabjobs", "onet", "esco"])
    ap.add_argument("--path")
    ap.add_argument("--min-freq", type=int, default=5)
    a = ap.parse_args()
    if a.source == "arabjobs":
        fetch_arabjobs()
    elif a.source == "onet":
        import_onet(a.path, a.min_freq)
    else:
        import_esco(a.path, a.min_freq)
