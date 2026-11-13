"""مشترك لصفحات التطبيق: تنسيق RTL، المحرك المخزّن مؤقتاً، بطاقات المؤشرات، حالات الغياب."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from src.common import DATA_DIR, MODELS_DIR, REPORTS_DIR  # noqa: E402

PALETTE = ["#0F766E", "#2563EB", "#D97706", "#7C3AED", "#DC2626", "#0891B2", "#65A30D", "#DB2777", "#475569", "#CA8A04"]
TYPE_AR = {"JOB": "وظيفة", "INTERNSHIP": "تدريب عملي", "TRAINING": "تدريب/دورة", "SCHOLARSHIP": "منحة",
           "TENDER": "مناقصة", "VOLUNTEER": "تطوع", "NOT_RELEVANT": "غير ذي صلة", "OTHER": "أخرى"}
SCOPE_AR = {"LOCAL": "محلي (مدينة/مديرية)", "SINGLE_GOVERNORATE": "محافظة واحدة", "MULTI_GOVERNORATE": "عدة محافظات",
            "NATIONWIDE": "على مستوى اليمن", "YEMEN_UNSPECIFIED": "داخل اليمن (غير محدد)", "INTERNATIONAL": "خارج اليمن",
            "REMOTE": "عن بُعد", "UNKNOWN": "غير معروف", "YEMEN": "داخل اليمن", "OUTSIDE_YEMEN": "خارج اليمن",
            "REMOTE_GLOBAL": "عن بُعد"}
ENT_COLORS = {"JOB_TITLE": "#0F766E", "ORGANIZATION": "#2563EB", "SKILL": "#D97706", "DEGREE": "#7C3AED",
              "MAJOR": "#DB2777", "EXPERIENCE": "#0891B2", "LOCATION": "#65A30D", "SALARY": "#CA8A04", "DEADLINE": "#DC2626"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, p, li, label, h1, h2, h3, h4, input, textarea, button, .stSelectbox,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] { font-family: 'IBM Plex Sans Arabic', sans-serif !important; }
.main .block-container, [data-testid="stSidebar"], .stMarkdown, [data-testid="stMetric"], .stAlert, .stRadio,
.stSelectbox, .stMultiSelect, .stTextInput, .stTextArea, .stSlider, .stTabs { direction: rtl; text-align: right; }
[data-testid="stDataFrame"], .js-plotly-plot, code, pre { direction: ltr; }
textarea { direction: rtl; text-align: right; }
.kpi { border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: 14px 16px; background: rgba(15,118,110,.05); }
.kpi .v { font-size: 1.75rem; font-weight: 700; color: #0F766E; line-height: 1.2; }
.kpi .l { font-size: .85rem; opacity: .75; }
.chip { display:inline-block; padding: 3px 10px; margin: 3px; border-radius: 999px; font-size: .85rem;
        border: 1px solid rgba(128,128,128,.3); }
.chip.ok { background: rgba(22,163,74,.12); border-color: rgba(22,163,74,.4); }
.chip.miss { background: rgba(220,38,38,.10); border-color: rgba(220,38,38,.35); }
mark.ent { padding: 1px 5px; border-radius: 6px; color: #fff; font-weight: 500; }
mark.ent small { opacity: .85; font-size: .7em; margin-inline-start: 4px; }
.adbox { line-height: 2.1; border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: 14px; white-space: pre-wrap; }
.note { font-size: .82rem; opacity: .7; }
</style>
"""


def setup(title: str, subtitle: str | None = None):
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


@st.cache_resource(show_spinner="تحميل المحرك…")
def _engine(data: str, models: str, reports: str):
    from src.integration.engine import Engine
    return Engine(Path(data), Path(models), Path(reports))


def engine():
    """محرك واحد لكل مجموعة مسارات (المسارات جزء من مفتاح التخزين المؤقت)."""
    return _engine(str(DATA_DIR), str(MODELS_DIR), str(REPORTS_DIR))


def need_data() -> bool:
    """يعرض رسالة واضحة إن غابت البيانات النهائية؛ لا تُستخدم بيانات تجريبية بديلة."""
    e = engine()
    if e.market.available:
        return True
    st.error("البيانات النهائية غير موجودة.")
    st.markdown(f"المسار المتوقع: `{DATA_DIR}`  \n"
                "انسخ مجلد `FINAL_TRAINING_DATASET` إلى جذر المشروع، أو حدّد مساره بمتغير البيئة `LMI_FINAL_DIR`. "
                "لا يعرض التطبيق بيانات تجريبية بديلاً عنها.")
    return False


def kpis(items: list[tuple[str, object]]):
    cols = st.columns(len(items))
    for c, (label, val) in zip(cols, items):
        v = f"{val:,}" if isinstance(val, (int, float)) and not isinstance(val, bool) else (val if val is not None else "—")
        c.markdown(f"<div class='kpi'><div class='v'>{v}</div><div class='l'>{label}</div></div>", unsafe_allow_html=True)


def bar(df: pd.DataFrame, x: str, y: str, title: str = "", height: int = 420, color=PALETTE[0], text: str | None = None):
    if df is None or df.empty:
        st.info("لا توجد بيانات كافية لهذا الرسم.")
        return
    d = df.iloc[::-1]
    fig = px.bar(d, x=x, y=y, orientation="h", title=title, text=text or x, height=height)
    fig.update_traces(marker_color=color, textposition="outside", cliponaxis=False)
    fig.update_layout(margin=dict(l=10, r=10, t=40 if title else 10, b=10), yaxis_title="", xaxis_title="",
                      font=dict(family="IBM Plex Sans Arabic"))
    st.plotly_chart(fig, width="stretch")


def pie(df: pd.DataFrame, names: str, values: str, title: str = "", height: int = 360):
    if df is None or df.empty:
        st.info("لا توجد بيانات.")
        return
    fig = px.pie(df, names=names, values=values, title=title, hole=.55, color_discrete_sequence=PALETTE, height=height)
    fig.update_layout(margin=dict(l=10, r=10, t=40 if title else 10, b=10), font=dict(family="IBM Plex Sans Arabic"))
    st.plotly_chart(fig, width="stretch")


def line(df: pd.DataFrame, title: str = "", height: int = 380, yfmt: str | None = None):
    if df is None or df.empty:
        st.info("لا توجد بيانات.")
        return
    d = df.copy()
    d.index = d.index.astype(str)
    fig = px.line(d, x=d.index, y=d.columns, title=title, markers=True, color_discrete_sequence=PALETTE, height=height)
    fig.update_layout(margin=dict(l=10, r=10, t=40 if title else 10, b=10), xaxis_title="", yaxis_title="",
                      legend_title="", font=dict(family="IBM Plex Sans Arabic"))
    if yfmt:
        fig.update_yaxes(tickformat=yfmt)
    st.plotly_chart(fig, width="stretch")


def ar_type(x):
    return TYPE_AR.get(x, x)


def ar_scope(x):
    return SCOPE_AR.get(x, x)
