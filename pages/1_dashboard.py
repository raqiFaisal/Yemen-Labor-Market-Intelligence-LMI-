import pandas as pd
import plotly.express as px
import streamlit as st

from utils.state import demo_banner, domain_ar, require_data, skills_table
from utils.ui import AMBER, BLUE, MARKET_COLORS, NAVY, TEAL, fmt_int, hero, kpis

hero("لوحة معلومات سوق العمل", "مؤشرات الطلب على الوظائف والمهارات — اليمن مقارنة بالأسواق العربية والعالمية",
     "Dashboard")
jobs, js = require_data()
demo_banner(jobs)

st_ = skills_table(js, jobs)
ye = jobs[jobs.market == "اليمن"]
top_dom = jobs["domain_final"].value_counts().drop("unclassified", errors="ignore")
top_skill = st_.groupby("name_ar").job_id.nunique().sort_values(ascending=False)
dated = jobs["posted_date"].dropna()
span = f"{dated.min():%Y-%m} → {dated.max():%Y-%m}" if len(dated) else "—"

kpis([
    ("إجمالي الوظائف", fmt_int(len(jobs)), f"الفترة: {span}", BLUE),
    ("وظائف اليمن", fmt_int(len(ye)), f"{len(ye) / max(len(jobs), 1) * 100:.0f}% من البيانات", NAVY),
    ("المهارات المستخرجة", fmt_int(st_.skill_id.nunique()), f"{fmt_int(len(js))} إشارة مهارة", TEAL),
    ("أكثر القطاعات طلباً", domain_ar(top_dom.index[0]) if len(top_dom) else "—",
     f"{fmt_int(top_dom.iloc[0])} وظيفة" if len(top_dom) else "", AMBER),
    ("أكثر المهارات انتشاراً", top_skill.index[0] if len(top_skill) else "—",
     f"في {top_skill.iloc[0] / len(jobs) * 100:.0f}% من الإعلانات" if len(top_skill) else "", "#8B5CF6"),
])

c1, c2 = st.columns([1.15, 1])
with c1:
    d = jobs.groupby(["domain_final", "market"]).size().reset_index(name="n")
    d["share"] = d["n"] / d.groupby("market")["n"].transform("sum") * 100
    d["المجال"] = d["domain_final"].map(domain_ar)
    fig = px.bar(d, x="share", y="المجال", color="market", barmode="group", orientation="h",
                 color_discrete_map=MARKET_COLORS, labels={"share": "% من وظائف السوق", "market": "السوق"},
                 title="توزيع الوظائف على المجالات: اليمن مقابل العالمي")
    fig.update_layout(height=420, yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, width="stretch")
with c2:
    sy = st_[st_.market == "اليمن"].groupby(["name_ar", "type"]).job_id.nunique().reset_index(name="n")
    sy = sy.sort_values("n", ascending=False).head(12)
    sy["pct"] = sy["n"] / max(len(ye), 1) * 100
    fig = px.bar(sy, x="pct", y="name_ar", color="type", orientation="h", title="أكثر 12 مهارة طلباً في اليمن",
                 labels={"pct": "% من الإعلانات اليمنية", "name_ar": "", "type": "النوع"},
                 color_discrete_map={"technical": BLUE, "professional": AMBER, "soft": TEAL})
    fig.update_layout(height=420, yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, width="stretch")

c3, c4 = st.columns([1.4, 1])
with c3:
    t = jobs.dropna(subset=["posted_date"]).assign(month=lambda x: x.posted_date.dt.to_period("M").dt.to_timestamp())
    t = t.groupby(["month", "market"]).size().reset_index(name="n")
    fig = px.line(t, x="month", y="n", color="market", color_discrete_map=MARKET_COLORS, markers=True,
                  title="اتجاه عدد الإعلانات الشهري", labels={"month": "", "n": "عدد الإعلانات", "market": "السوق"})
    fig.update_layout(height=360)
    st.plotly_chart(fig, width="stretch")
with c4:
    g = ye["location"].fillna("غير محدد").value_counts().head(10).reset_index()
    g.columns = ["المحافظة", "n"]
    fig = px.bar(g, x="n", y="المحافظة", orientation="h", title="الوظائف حسب المحافظة (اليمن)",
                 labels={"n": "عدد الوظائف"}, color_discrete_sequence=[NAVY])
    fig.update_layout(height=360, yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, width="stretch")

c5, c6 = st.columns(2)
with c5:
    ty = st_.groupby(["market", "type"]).size().reset_index(name="n")
    ty["النوع"] = ty["type"].map({"technical": "تقنية", "soft": "شخصية", "professional": "مهنية"})
    fig = px.sunburst(ty, path=["market", "النوع"], values="n", title="توزيع أنواع المهارات المطلوبة",
                      color_discrete_sequence=[BLUE, AMBER, TEAL])
    fig.update_layout(height=360)
    st.plotly_chart(fig, width="stretch")
with c6:
    e = jobs.dropna(subset=["experience_years"])
    if len(e):
        fig = px.histogram(e, x="experience_years", color="market", barmode="overlay", nbins=12, histnorm="percent",
                           color_discrete_map=MARKET_COLORS, title="متطلبات سنوات الخبرة",
                           labels={"experience_years": "سنوات الخبرة", "market": "السوق"})
        fig.update_layout(height=360, yaxis_title="%")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("لا تتوفر بيانات عن سنوات الخبرة.")

with st.expander("مصادر البيانات"):
    src = jobs.groupby(["source", "market"]).size().reset_index(name="عدد السجلات")
    st.dataframe(src, width="stretch", hide_index=True)
