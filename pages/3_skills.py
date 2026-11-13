import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.state import demo_banner, domain_ar, require_data, skills_table
from utils.ui import AMBER, BLUE, MARKET_COLORS, TEAL, TYPE_AR, hero, kpis, note

hero("ذكاء المهارات", "ما المهارات التي يطلبها السوق؟ وكيف تختلف بين اليمن والأسواق الأخرى؟", "Skills Intelligence")
jobs, js = require_data()
demo_banner(jobs)
S = skills_table(js, jobs)
n_market = jobs.groupby("market").size()

c = st.columns(3)
market = c[0].selectbox("السوق", ["الكل", "اليمن", "عالمي/عربي"])
stype = c[1].selectbox("نوع المهارة", ["الكل", "technical", "professional", "soft"],
                       format_func=lambda x: "الكل" if x == "الكل" else TYPE_AR[x])
topn = c[2].slider("عدد المهارات", 5, 40, 20)

f = S if market == "الكل" else S[S.market == market]
if stype != "الكل":
    f = f[f.type == stype]
base = len(jobs) if market == "الكل" else n_market.get(market, 1)
by = f.groupby(["name_ar", "name_en", "type"]).job_id.nunique().reset_index(name="n")
by["pct"] = by["n"] / base * 100
by = by.sort_values("n", ascending=False)

tech = S[S.type == "technical"].skill_id.nunique()
soft = S[S.type == "soft"].skill_id.nunique()
avg = js.groupby("job_id").size().mean()
kpis([("مهارات مختلفة", len(by), "ضمن التصفية الحالية"), ("مهارات تقنية", tech, "", BLUE),
      ("مهارات شخصية", soft, "", TEAL), ("متوسط المهارات لكل إعلان", f"{avg:.1f}", "", AMBER)])

t1, t2, t3, t4 = st.tabs(["الأكثر طلباً", "حسب التخصص المهني", "اليمن مقابل العالمي", "تقنية مقابل شخصية"])
with t1:
    d = by.head(topn)
    fig = px.bar(d, x="pct", y="name_ar", color="type", orientation="h", hover_data=["name_en", "n"],
                 color_discrete_map={"technical": BLUE, "professional": AMBER, "soft": TEAL},
                 labels={"pct": "% من الإعلانات", "name_ar": "", "type": "النوع"})
    fig.update_layout(height=max(380, topn * 24), yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, width="stretch")

with t2:
    g = (S if market == "الكل" else S[S.market == market])
    top_sk = g.groupby("name_ar").job_id.nunique().nlargest(topn).index
    dom_tot = (jobs if market == "الكل" else jobs[jobs.market == market]).groupby("domain_final").size()
    h = g[g.name_ar.isin(top_sk)].groupby(["domain_final", "name_ar"]).job_id.nunique().unstack(fill_value=0)
    h = (h.div(dom_tot.reindex(h.index), axis=0) * 100).round(1)
    h.index = [domain_ar(i) for i in h.index]
    fig = px.imshow(h, aspect="auto", color_continuous_scale="Blues", text_auto=True,
                    labels=dict(color="% من وظائف المجال"), title="نسبة ظهور المهارة داخل كل مجال")
    fig.update_layout(height=520, xaxis_tickangle=-45)
    st.plotly_chart(fig, width="stretch")

with t3:
    if n_market.get("اليمن", 0) == 0 or n_market.get("عالمي/عربي", 0) == 0:
        note("المقارنة تتطلب وجود بيانات يمنية وعالمية معاً.", "warn")
    else:
        cmp = S.groupby(["name_ar", "market"]).job_id.nunique().unstack(fill_value=0)
        cmp = cmp.div(n_market, axis=1) * 100
        cmp["الفرق"] = cmp["اليمن"] - cmp["عالمي/عربي"]
        top = cmp.reindex(cmp[["اليمن", "عالمي/عربي"]].max(axis=1).nlargest(topn).index).sort_values("الفرق")
        fig = go.Figure()
        for mk in ["اليمن", "عالمي/عربي"]:
            fig.add_bar(y=top.index, x=top[mk], name=mk, orientation="h", marker_color=MARKET_COLORS[mk])
        fig.update_layout(barmode="group", height=max(420, topn * 28), title="نسبة الإعلانات التي تطلب المهارة",
                          xaxis_title="% من إعلانات السوق")
        st.plotly_chart(fig, width="stretch")
        a, b = st.columns(2)
        a.markdown("**مهارات يتميز بها السوق اليمني**")
        a.dataframe(cmp.nlargest(8, "الفرق").round(1), width="stretch")
        b.markdown("**مهارات أقل حضوراً في اليمن**")
        b.dataframe(cmp.nsmallest(8, "الفرق").round(1), width="stretch")

with t4:
    ty = (S if market == "الكل" else S[S.market == market]).groupby(["domain_final", "type"]).size().reset_index(name="n")
    ty["المجال"] = ty.domain_final.map(domain_ar)
    ty["النوع"] = ty.type.map(TYPE_AR)
    fig = px.bar(ty, x="المجال", y="n", color="النوع", barmode="relative",
                 color_discrete_sequence=[BLUE, AMBER, TEAL], title="تركيبة المهارات المطلوبة في كل مجال")
    fig.update_layout(barnorm="percent", yaxis_title="%", height=430)
    st.plotly_chart(fig, width="stretch")
