"""المراجعة البشرية لقرارات المهارات — تُطبَّق فوق قرارات المستخرج دون تعديل منطقه.

verdict = correct   → تُقبل المهارة (حتى لو كانت "review")، ويُضاف للسبب: مؤكدة بشرياً
verdict = incorrect → تُرفض المهارة (حتى لو كانت "accepted")، ويُضاف للسبب: رفضها المراجع
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from core.config import SKILL_REVIEWS_FILE

VERDICTS = {"correct", "incorrect"}


def load_reviews() -> pd.DataFrame:
    if not SKILL_REVIEWS_FILE.exists():
        return pd.DataFrame(columns=["job_id", "skill_id", "verdict", "reviewed_at"])
    r = pd.read_csv(SKILL_REVIEWS_FILE, dtype=str).dropna(subset=["job_id", "skill_id", "verdict"])
    return r[r.verdict.isin(VERDICTS)].drop_duplicates(["job_id", "skill_id"], keep="last")


def save_reviews(rows: list[dict]):
    new = pd.DataFrame(rows).assign(reviewed_at=datetime.now().isoformat(timespec="seconds"))
    pd.concat([load_reviews(), new]).drop_duplicates(["job_id", "skill_id"], keep="last") \
        .to_csv(SKILL_REVIEWS_FILE, index=False, encoding="utf-8-sig")


def apply_reviews(decisions: pd.DataFrame) -> pd.DataFrame:
    rv = load_reviews()
    if rv.empty or decisions.empty:
        return decisions.assign(human_verdict=None) if "human_verdict" not in decisions else decisions
    d = decisions.merge(rv[["job_id", "skill_id", "verdict"]], on=["job_id", "skill_id"], how="left")
    ok, bad = d.verdict == "correct", d.verdict == "incorrect"
    d.loc[ok, "decision"] = "accepted"
    d.loc[ok, "reason"] = d.loc[ok, "reason"] + " | مؤكدة بشرياً"
    d.loc[bad, "decision"] = "rejected"
    d.loc[bad, "reason"] = d.loc[bad, "reason"] + " | رفضها المراجع البشري"
    return d.rename(columns={"verdict": "human_verdict"})
