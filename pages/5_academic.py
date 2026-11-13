import pandas as pd
import plotly.express as px
import streamlit as st

from utils.state import demo_banner, domain_ar, get_majors, require_data, skills_table
from utils.ui import BLUE, MARKET_COLORS, TYPE_AR, hero, kpis, note

hero("التخصصات الأكاديمية وسوق العمل", "ربط التخصص الجامعي بالمجالات الوظيفية والمهارات التي يطلبها السوق فعلياً",
     "Academic Specializations")
jobs, js = require_data()
demo_banner(jobs)
majors = get_majors()
S = skills_table(js, jobs)
note("ربط التخصص بالمجالات الوظيفية مبني على قاموس منسّق (<code>data/taxonomy/majors.json</code>) قابل للتعديل؛ "
     "أما أرقام الطلب والمهارات فمحسوبة مباشرة من الإعلانات.")

market = st.radio("السوق", ["اليمن", "عالمي/عربي", "الكل"], horizontal=True)
J = jobs if market == "الكل" else jobs[jobs.market == market]
SS = S if market == "الكل" else S[S.market == market]

# نظرة مقارنة لكل التخصصات
rows = []
for m, doms in majors.items():
    sub = J[J.domain_final.isin(doms)]
    rows.append({"التخصص": m, "الوظائف المرتبطة": len(sub), "النسبة %": len(sub) / max(len(J), 1) * 100,
                 "المجالات": "، ".join(domain_ar(d) for d in doms)})
ov = pd.DataFrame(rows).sort_values("الوظائف المرتبطة", ascending=False)
fig = px.bar(ov, x="النسبة %", y="التخصص", orientation="h", hover_data=["المجالات", "الوظائف المرتبطة"],
             title="حجم فرص العمل المرتبطة بكل تخصص (قد تتداخل المجالات بين التخصصات)", color_discrete_sequence=[BLUE])
fig.update_layout(height=560, yaxis=dict(categoryorder="total ascending"))
st.plotly_chart(fig, width="stretch")

st.divider()
major = st.selectbox("اختر تخصصاً لتحليله بالتفصيل", list(majors))
doms = majors[major]
sub = J[J.domain_final.isin(doms)]
ssub = SS[SS.domain_final.isin(doms)]
top = ssub.groupby(["name_ar", "type"]).job_id.nunique().reset_index(name="n").sort_values("n", ascending=False)
top["pct"] = top.n / max(len(sub), 1) * 100
kpis([("الوظائف المرتبطة", f"{len(sub):,}", f"{len(sub) / max(len(J), 1) * 100:.1f}% من السوق"),
      ("المجالات", len(doms), "، ".join(domain_ar(d) for d in doms)),
      ("المهارات المطلوبة", top.shape[0], "مهارة مختلفة"),
      ("متوسط الخبرة المطلوبة", f"{sub.experience_years.mean():.1f} سنة" if sub.experience_years.notna().any() else "—", "")])
c1, c2 = st.columns([1.2, 1])
with c1:
    fig = px.bar(top.head(15), x="pct", y="name_ar", color=top.head(15)["type"].map(TYPE_AR), orientation="h",
                 title=f"أهم المهارات لخريجي {major}", labels={"pct": "% من الوظائف", "name_ar": "", "color": "النوع"})
    fig.update_layout(height=470, yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, width="stretch")
with c2:
    tt = sub.title.value_counts().head(12).reset_index()
    tt.columns = ["المسمى الوظيفي", "عدد الإعلانات"]
    st.markdown(f"**أكثر المسميات الوظيفية المتاحة لتخصص {major}**")
    st.dataframe(tt, hide_index=True, width="stretch", height=430)
if market == "الكل":
    mm = sub.groupby("market").size().reset_index(name="n")
    st.plotly_chart(px.pie(mm, names="market", values="n", hole=.5, color="market", color_discrete_map=MARKET_COLORS,
                           title="توزيع الفرص بين اليمن والأسواق الأخرى"), width="stretch")
