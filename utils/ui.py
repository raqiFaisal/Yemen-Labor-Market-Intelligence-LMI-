"""مكونات الواجهة المشتركة: التصميم RTL، بطاقات KPI، قالب الرسوم، التنبيهات."""
from __future__ import annotations

import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

NAVY = "#0B1F3A"
BLUE = "#1F5EFF"
SKY = "#5B9BFF"
TEAL = "#14B8A6"
AMBER = "#F59E0B"
RED = "#EF4444"
GRAY = "#64748B"
PALETTE = [BLUE, TEAL, AMBER, "#8B5CF6", "#EC4899", SKY, "#22C55E", GRAY]
MARKET_COLORS = {"اليمن": BLUE, "عالمي/عربي": AMBER}

pio.templates["lmi"] = go.layout.Template(layout=dict(
    font=dict(family="Tajawal, Cairo, sans-serif", size=13, color="#1E293B"),
    colorway=PALETTE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=40, b=10), hoverlabel=dict(font_family="Tajawal"),
    xaxis=dict(gridcolor="#E2E8F0", zerolinecolor="#E2E8F0"), yaxis=dict(gridcolor="#E2E8F0", zerolinecolor="#E2E8F0"),
    legend=dict(orientation="h", y=-0.15), title=dict(font=dict(size=15, color=NAVY), x=0.99, xanchor="right"),
))
pio.templates.default = "plotly_white+lmi"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, p, li, label, .stSelectbox, .stMultiSelect, input, textarea, button {{
  font-family: 'Tajawal', sans-serif !important;
}}
.main .block-container, [data-testid="stMainBlockContainer"] {{ direction: rtl; text-align: right; padding-top: 1.6rem; max-width: 1350px; }}
[data-testid="stSidebar"] {{ direction: rtl; background: linear-gradient(180deg, {NAVY} 0%, #112B52 100%); }}
[data-testid="stSidebar"] * {{ color: #E2E8F0 !important; }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ background: rgba(91,155,255,.22); border-radius: 8px; }}
[data-testid="stSidebarNavSeparator"] {{ border-color: rgba(255,255,255,.15); }}
h1, h2, h3, h4 {{ color: {NAVY}; font-weight: 800 !important; letter-spacing: 0; }}
.stTabs [data-baseweb="tab-list"] {{ direction: rtl; gap: 4px; }}
.stDataFrame, .stPlotlyChart, code, pre {{ direction: ltr; }}
[data-testid="stMetric"] {{ direction: rtl; }}
.lmi-hero {{ background: linear-gradient(120deg, {NAVY} 0%, #17407E 60%, {BLUE} 100%); color: #fff; border-radius: 18px;
  padding: 22px 26px; margin-bottom: 18px; position: relative; overflow: hidden; }}
.lmi-hero h1 {{ color: #fff; margin: 0 0 4px 0; font-size: 1.75rem; }}
.lmi-hero p {{ color: #CBD5E1; margin: 0; font-size: .98rem; }}
.lmi-hero .tag {{ display:inline-block; background: rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.25);
  padding: 2px 10px; border-radius: 999px; font-size:.78rem; margin-bottom:8px; color:#fff; }}
.kpi-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(190px,1fr)); gap: 12px; margin: 6px 0 14px; }}
.kpi {{ background:#fff; border:1px solid #E2E8F0; border-radius:14px; padding:14px 16px; box-shadow: 0 1px 2px rgba(15,23,42,.04);
  border-top: 3px solid var(--c, {BLUE}); transition: transform .15s ease, box-shadow .15s ease; }}
.kpi:hover {{ transform: translateY(-2px); box-shadow: 0 8px 20px rgba(15,23,42,.08); }}
.kpi .lbl {{ color:{GRAY}; font-size:.82rem; font-weight:500; }}
.kpi .val {{ color:{NAVY}; font-size:1.55rem; font-weight:800; line-height:1.3; }}
.kpi .sub {{ color:{GRAY}; font-size:.76rem; }}
.badge {{ display:inline-block; padding:2px 10px; margin:3px; border-radius:999px; font-size:.82rem; font-weight:500; }}
.b-tech {{ background:#E0EAFF; color:#1E40AF; }} .b-soft {{ background:#DCFCE7; color:#166534; }}
.b-prof {{ background:#FEF3C7; color:#92400E; }} .b-sem {{ outline: 1.5px dashed #94A3B8; }}
.b-miss {{ background:#FEE2E2; color:#991B1B; }}
.note {{ background:#F1F5F9; border-right:4px solid {SKY}; padding:10px 14px; border-radius:8px; color:#334155; font-size:.9rem; margin:8px 0; }}
.warn {{ background:#FFFBEB; border-right:4px solid {AMBER}; padding:10px 14px; border-radius:8px; color:#78350F; font-size:.9rem; margin:8px 0; }}
.side-card {{ background: rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.12); border-radius:10px; padding:10px 12px; font-size:.82rem; }}
@media (max-width: 768px) {{ .lmi-hero h1 {{ font-size: 1.3rem; }} .kpi .val {{ font-size: 1.25rem; }} }}
</style>
"""


def setup_page():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, tag: str | None = None):
    t = f'<span class="tag">{html.escape(tag)}</span>' if tag else ""
    st.markdown(f'<div class="lmi-hero">{t}<h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>',
                unsafe_allow_html=True)


def kpis(items: list[tuple]):
    """items: [(label, value, sub, color?)]"""
    cards = []
    for it in items:
        lbl, val, sub = it[0], it[1], it[2] if len(it) > 2 else ""
        c = it[3] if len(it) > 3 else BLUE
        cards.append(f'<div class="kpi" style="--c:{c}"><div class="lbl">{html.escape(str(lbl))}</div>'
                     f'<div class="val">{html.escape(str(val))}</div><div class="sub">{html.escape(str(sub))}</div></div>')
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def note(text: str, kind: str = "note"):
    st.markdown(f'<div class="{kind}">{text}</div>', unsafe_allow_html=True)


TYPE_CLASS = {"technical": "b-tech", "soft": "b-soft", "professional": "b-prof"}
TYPE_AR = {"technical": "تقنية", "soft": "شخصية", "professional": "مهنية"}


def skill_badges(skills: list[dict]):
    if not skills:
        st.caption("لم تُستخرج مهارات.")
        return
    out = []
    for s in skills:
        cls = TYPE_CLASS.get(s.get("type"), "b-tech")
        if s.get("method") == "semantic":
            cls += " b-sem"
        tip = html.escape(f"{s['name_en']} • {s.get('method', '')} • {s.get('score', 1):.2f}")
        out.append(f'<span class="badge {cls}" title="{tip}">{html.escape(s["name_ar"])}</span>')
    st.markdown("".join(out), unsafe_allow_html=True)


def fmt_int(n) -> str:
    return f"{int(n):,}"
