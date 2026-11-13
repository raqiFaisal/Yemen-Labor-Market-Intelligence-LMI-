"""خط المعالجة: تحميل المصادر الخام → التوحيد → توحيد الفئات → استخراج المهارات → الحفظ.

اصطلاح أسماء الملفات في RAW_DIR (data/raw في الوضع الحقيقي):
    yemen_<source>.csv  إعلانات يمنية (scripts/scrape_yemen.py) → country=Yemen، ومصدر كل ملف منفصل
    arabjobs*.csv       مجموعة ArabJobs (scripts/fetch_external.py)
    global_*.csv        أي بيانات عالمية بأعمدة مختلفة
الوضع الحقيقي يرفض أي ملف demo_* أو أي سجل source=DEMO_SYNTHETIC — DEMO يعيش في data/demo ووضع LMI_MODE=demo فقط.
"""
from __future__ import annotations

import os

import pandas as pd

from core.config import (COUNTRY_YEMEN, DEMO_SOURCE, IS_DEMO_MODE, JOB_SKILLS_CSV, JOB_SKILLS_FILE, JOBS_FILE,
                         PROCESSED_DIR, RAW_DIR, SKILL_CANDIDATES_CSV, SKILL_DECISIONS_FILE)
from preprocessing.categories import load_map, map_category
from preprocessing.schema import job_text, unify


def map_domain(label) -> str | None:
    """توافق مع الإصدارات السابقة: تسمية مصدر → مجال (مطابقة صريحة فقط)."""
    return map_category(label)[0]


def load_raw_sources(include_demo: str = "auto") -> tuple[pd.DataFrame, list[dict]]:
    """include_demo: مُهمل (يبقى للتوافق) — الفصل يتم بوضع التشغيل LMI_MODE."""
    frames, log = [], []
    files = sorted(RAW_DIR.glob("*.csv")) + sorted(RAW_DIR.glob("*.xlsx"))
    for f in files:
        is_demo_file = f.name.startswith("demo_")
        if is_demo_file and not IS_DEMO_MODE:
            log.append({"file": f.name, "status": "rejected: demo file in real data folder", "rows": 0})
            continue
        try:
            raw = pd.read_excel(f) if f.suffix == ".xlsx" else pd.read_csv(f, encoding="utf-8-sig", low_memory=False)
            name = f.stem.lower()
            if not IS_DEMO_MODE and "source" in raw and (raw["source"] == DEMO_SOURCE).any():
                n = int((raw["source"] == DEMO_SOURCE).sum())
                raw = raw[raw["source"] != DEMO_SOURCE]
                log.append({"file": f.name, "status": f"dropped {n} DEMO rows", "rows": 0})
            country = COUNTRY_YEMEN if name.startswith("yemen") else None
            source = DEMO_SOURCE if is_demo_file else ("ArabJobs" if name.startswith("arabjobs") else
                                                        name.replace("yemen_", ""))
            u = unify(raw, source=source, country=country, is_demo=IS_DEMO_MODE)
            u["source_file"] = f.name
            frames.append(u)
            log.append({"file": f.name, "status": "ok", "rows": len(u)})
        except Exception as e:  # noqa: BLE001
            log.append({"file": f.name, "status": f"error: {type(e).__name__}: {e}", "rows": 0})
    # وظائف Telegram الموحدة (المرحلة 5): فريدة، غير مكررة مع إعلانات موقع YemenHR المجمعة — تُضاف في الوضع الحقيقي فقط
    tg = PROCESSED_DIR / "dataset" / "telegram_jobs_for_training.csv"
    if tg.exists() and not IS_DEMO_MODE and os.environ.get("LMI_SKIP_TELEGRAM") != "1":
        try:
            raw = pd.read_csv(tg, encoding="utf-8-sig", low_memory=False)
            u = unify(raw, is_demo=False)
            u["source_file"] = "dataset/" + tg.name
            frames.append(u)
            log.append({"file": "dataset/" + tg.name, "status": "ok (telegram, deduplicated)", "rows": len(u)})
        except Exception as e:  # noqa: BLE001
            log.append({"file": "dataset/" + tg.name, "status": f"error: {type(e).__name__}: {e}", "rows": 0})
    if not frames:
        return pd.DataFrame(), log
    df = pd.concat(frames, ignore_index=True).drop_duplicates("job_id")
    cmap = load_map()
    mapped = df["category_raw"].map(lambda c: map_category(c, cmap))
    df["domain"] = mapped.map(lambda x: x[0])           # تسمية المصدر بعد التوحيد (أو None)
    df["category_status"] = mapped.map(lambda x: x[1])
    # إعلان Telegram بلا موقع معروف لا يُنسب لأي سوق (يظهر في «الكل» فقط) — بدل نسبته خطأً للسوق العالمي
    tg_unknown = df["source"].astype(str).str.startswith("telegram:") & df["country"].isin(["Unknown", "", None])
    df["market"] = df["country"].map(lambda c: "اليمن" if c == COUNTRY_YEMEN else "عالمي/عربي")
    df.loc[tg_unknown, "market"] = "غير محدد"
    return df.reset_index(drop=True), log


def category_report(df: pd.DataFrame) -> dict:
    """ملخص توحيد الفئات: ما طُبق وما يحتاج مراجعة (لإضافته إلى category_map.json)."""
    if "category_raw" not in df:
        return {}
    exploded = df.assign(cat=df["category_raw"].fillna("").str.split(";")).explode("cat")
    exploded["cat"] = exploded["cat"].str.strip()
    exploded = exploded[exploded["cat"] != ""]
    cmap = load_map()
    rows = {}
    for c, n in exploded["cat"].value_counts().items():
        d, s = map_category(c, cmap)
        rows[c] = {"count": int(n), "domain": d, "status": s}
    return {"by_raw_category": rows,
            "status_counts": {k: int(v) for k, v in df["category_status"].value_counts().items()}}


def extract_job_skills(df: pd.DataFrame, encoder=None, semantic: bool = True) -> pd.DataFrame:
    """المهارات المقبولة فقط: job_id, skill_id, method, confidence, evidence, reason — صف واحد لكل (وظيفة، مهارة)."""
    from models.skill_extractor import SkillExtractor
    ex = SkillExtractor(encoder)
    return ex.extract_batch(df["job_id"].tolist(), job_text(df).tolist(), semantic=semantic)


def extract_job_skills_full(df: pd.DataFrame, encoder=None, semantic: bool = True):
    """(المقبولة بعد المراجعة البشرية، كل القرارات، كل المرشحات الخام)"""
    from models.skill_extractor import COLUMNS, SkillExtractor
    from preprocessing.skill_review import apply_reviews
    ex = SkillExtractor(encoder)
    dec, cands = ex.extract_all(df["job_id"].tolist(), job_text(df).tolist(), semantic=semantic)
    dec = apply_reviews(dec)
    return dec[dec.decision == "accepted"][COLUMNS].reset_index(drop=True), dec, cands


def save_skill_decisions(dec: pd.DataFrame, cands: pd.DataFrame):
    dec.to_pickle(SKILL_DECISIONS_FILE)
    cands.to_csv(SKILL_CANDIDATES_CSV, index=False, encoding="utf-8-sig")


def load_skill_decisions() -> pd.DataFrame | None:
    return pd.read_pickle(SKILL_DECISIONS_FILE) if SKILL_DECISIONS_FILE.exists() else None


def save_processed(jobs: pd.DataFrame, job_skills: pd.DataFrame):
    jobs.to_pickle(JOBS_FILE)
    job_skills.to_pickle(JOB_SKILLS_FILE)
    try:
        from models.skill_extractor import load_taxonomy
        _, by_id = load_taxonomy()
        out = job_skills.assign(skill_en=job_skills.skill_id.map(lambda s: by_id.get(s, {}).get("name_en")),
                                skill_ar=job_skills.skill_id.map(lambda s: by_id.get(s, {}).get("name_ar")))
        out.to_csv(JOB_SKILLS_CSV, index=False, encoding="utf-8-sig")
    except Exception:  # noqa: BLE001
        pass


def load_processed():
    if not JOBS_FILE.exists():
        return None, None
    jobs, js = pd.read_pickle(JOBS_FILE), pd.read_pickle(JOB_SKILLS_FILE)
    for c, v in (("method", "lexicon"), ("confidence", 0.85), ("evidence", ""), ("reason", "")):
        if c not in js:
            js[c] = v
    return jobs, js
