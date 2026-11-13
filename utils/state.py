"""تحميل الموارد مع التخزين المؤقت (caching) — النماذج تُحمّل مرة واحدة لكل جلسة خادم."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from core.artifacts import read_status
from core.config import (CLASSIFIER_FILE, CLASSIFIER_METRICS, DATA_REPORT, DOMAINS_FILE, EMB_CACHE, EMB_IDS,
                         ENCODER_INFO, IS_DEMO_MODE, JOBS_FILE, LSTM_FILE, LSTM_METRICS, MAJORS_FILE, RAW_DIR)
from core.pipeline import load_processed
from models.skill_extractor import load_taxonomy
from utils.ui import note


def _mtime(p) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner="تحميل البيانات ...")
def _load_data(version: float):
    jobs, js = load_processed()
    if jobs is not None:
        # توافق مع ملفات معالجة من إصدار سابق
        if "domain_final" not in jobs:
            jobs["domain_final"] = jobs.get("domain")
        jobs["domain_final"] = jobs["domain_final"].fillna("unclassified")
        if "domain_source" not in jobs:
            jobs["domain_source"] = np.where(jobs["domain"].notna(), "source", "model")
    return jobs, js


def get_data():
    """يُعاد التحميل تلقائياً عند تغير ملف البيانات (بعد إعادة التدريب)."""
    return _load_data(_mtime(JOBS_FILE))


@st.cache_data
def get_domains() -> dict:
    return json.loads(DOMAINS_FILE.read_text(encoding="utf-8")) if DOMAINS_FILE.exists() else {}


@st.cache_data
def get_majors() -> dict:
    return json.loads(MAJORS_FILE.read_text(encoding="utf-8")) if MAJORS_FILE.exists() else {}


def domain_ar(d) -> str:
    if not isinstance(d, str) or d == "unclassified":
        return "غير مصنف"
    return get_domains().get(d, {}).get("ar", d)


DOMAIN_SOURCE_AR = {"gold": "تصنيف يدوي", "source": "من المصدر", "weak_title": "قاعدة المسمى (ضعيف)",
                    "weak_skills": "تصويت المهارات (ضعيف)", "model": "تنبؤ النموذج", "none": "غير مصنف"}


def taxonomy():
    return load_taxonomy()


def read_json(p):
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except Exception:  # noqa: BLE001
        return None


def encoder_info():
    return read_json(ENCODER_INFO)


@st.cache_resource(show_spinner="تحميل نموذج Transformer ...")
def get_encoder():
    """يحمّل نفس نوع المُرمِّز الذي حُسبت به الـ embeddings المخزنة لضمان الاتساق."""
    from models.encoder import TextEncoder
    info = encoder_info()
    prefer = (info or {}).get("kind", "transformer") == "transformer"
    enc = TextEncoder(prefer_transformer=prefer)
    return enc if enc.ready else None


@st.cache_resource
def get_extractor():
    from models.skill_extractor import SkillExtractor
    return SkillExtractor(get_encoder())


@st.cache_resource
def _load_classifier(version: float):
    if read_status(CLASSIFIER_METRICS)["status"] != "trained":
        return None
    try:
        from models.classifier import JobClassifier
        clf = JobClassifier()
    except FileNotFoundError:
        return None
    # التحقق من التوافق مع المُرمِّز الحالي: نموذج مدرب على مُرمِّز آخر لا يُستخدم
    info = encoder_info() or {}
    if clf.encoder_kind != info.get("kind") or (info.get("dim") and clf.in_dim != info["dim"]):
        return None
    return clf


def get_classifier():
    return _load_classifier(_mtime(CLASSIFIER_FILE) + _mtime(ENCODER_INFO))


@st.cache_resource
def _load_forecaster(version: float):
    if read_status(LSTM_METRICS)["status"] != "trained":
        return None
    try:
        from models.forecaster import SkillForecaster
        return SkillForecaster()
    except FileNotFoundError:
        return None


def get_forecaster():
    return _load_forecaster(_mtime(LSTM_FILE))


@st.cache_data
def _load_emb(version: float):
    return np.load(EMB_CACHE) if EMB_CACHE.exists() else None


def get_embeddings():
    """المصفوفة فقط إذا طابقت صفوفها ترتيب الإعلانات الحالية (يمنع الاستعلام بتمثيلات إعلانات أخرى)."""
    emb = _load_emb(_mtime(EMB_CACHE))
    jobs, _ = get_data()
    if emb is None or jobs is None or len(emb) != len(jobs):
        return None
    if EMB_IDS.exists():
        try:
            if json.loads(EMB_IDS.read_text()) != jobs["job_id"].tolist():
                return None
        except Exception:  # noqa: BLE001
            return None
    return emb


def clf_status() -> dict:
    return read_status(CLASSIFIER_METRICS)


def lstm_status() -> dict:
    return read_status(LSTM_METRICS)


def clf_metrics():
    """المقاييس فقط إن كان النموذج مدرباً على البيانات الحالية، وإلا None."""
    s = clf_status()
    return s if s["status"] == "trained" else None


def lstm_metrics():
    s = lstm_status()
    return s if s["status"] == "trained" else None


def data_report():
    return read_json(DATA_REPORT)


def require_data():
    jobs, js = get_data()
    if jobs is None or jobs.empty:
        st.error(f"لا توجد بيانات معالجة بعد في وضع {'العرض التجريبي' if IS_DEMO_MODE else 'البيانات الحقيقية'}.")
        if IS_DEMO_MODE:
            st.code("python -m scripts.train_demo", language="bash")
        else:
            st.markdown(f"اجمع الإعلانات الحقيقية ثم درّب (المجلد: `{RAW_DIR}`):")
            st.code("python -m scripts.scrape_yemen --site yemenhr\npython -m scripts.train_all", language="bash")
            st.caption("للتجربة دون بيانات حقيقية: وضع عرض تجريبي منفصل — LMI_MODE=demo streamlit run app.py")
        st.stop()
    return jobs, js


def demo_banner(jobs: pd.DataFrame):
    if IS_DEMO_MODE or (jobs is not None and "is_demo" in jobs and jobs["is_demo"].any()):
        note("🧪 <b>وضع العرض التجريبي:</b> كل البيانات والنماذج في هذه الجلسة اصطناعية (DEMO) ومنفصلة عن البيانات الحقيقية، "
             "والأرقام المعروضة <b>ليست إحصائيات أو توقعات حقيقية لسوق العمل</b>. "
             "للبيانات الحقيقية شغّل التطبيق دون <code>LMI_MODE=demo</code>.", "warn")
    elif jobs is not None and len(jobs):
        src = ", ".join(f"{k} ({v})" for k, v in jobs["source"].value_counts().items())
        dt = pd.to_datetime(jobs.get("scraped_at"), errors="coerce")
        when = f" • جُمعت {dt.min():%Y-%m-%d} → {dt.max():%Y-%m-%d}" if dt is not None and dt.notna().any() else ""
        st.caption(f"📌 بيانات حقيقية: {src}{when}")


def skills_table(js: pd.DataFrame, jobs: pd.DataFrame) -> pd.DataFrame:
    """job_skills مدمجة بمعلومات الوظيفة والمهارة."""
    skills, by_id = taxonomy()
    meta = pd.DataFrame(skills)[["id", "name_ar", "name_en", "type"]].rename(columns={"id": "skill_id"})
    cols = ["job_id", "market", "country", "domain_final", "location", "posted_date"]
    return js.merge(jobs[cols], on="job_id").merge(meta, on="skill_id")
