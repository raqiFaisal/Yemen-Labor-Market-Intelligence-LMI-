"""إدارة حالة النماذج المدربة.

القاعدة: ملف المقاييس هو مصدر الحقيقة. إذا لم يُدرَّب النموذج في آخر تشغيل يُحذف ملف الأوزان القديم
ويُكتب ملف حالة يشرح السبب — حتى لا تظهر نماذج قديمة (مثلاً مدربة على DEMO) كأنها نماذج البيانات الحالية.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


def mark_not_trained(weights: Path, metrics: Path, reason: str, details: dict | None = None):
    if weights.exists():
        weights.unlink()
    metrics.write_text(json.dumps({"status": "not_trained", "reason": reason, "details": details or {},
                                   "at": time.strftime("%Y-%m-%d %H:%M")}, ensure_ascii=False, indent=1), encoding="utf-8")


def read_status(metrics: Path) -> dict:
    """{'status': trained|not_trained|missing, ...}"""
    if not metrics.exists():
        return {"status": "missing", "reason": "لم يُشغَّل التدريب بعد (python -m scripts.train_all)."}
    try:
        m = json.loads(metrics.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return {"status": "missing", "reason": f"ملف المقاييس تالف: {e}"}
    if m.get("status") == "not_trained" or "test" not in m:
        m.setdefault("status", "not_trained")
        m.setdefault("reason", "بيانات غير كافية")
        return m
    m["status"] = "trained"
    return m
