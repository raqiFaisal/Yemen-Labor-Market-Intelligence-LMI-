import plotly.graph_objects as go
import streamlit as st

from common import engine, need_data, setup

setup("تحليل فجوة المهارات", "قارن مهاراتك بما يطلبه السوق فعلاً لتخصص أو مسمى وظيفي")
if need_data():
    e = engine()
    m = e.market
    cat = m.catalog.sort_values("name_ar")
    c1, c2 = st.columns([3, 2])
    with c1:
        chosen = st.multiselect("مهاراتك (من قاموس المهارات)", cat.skill_id.tolist(), format_func=m.skill_name)
        free = st.text_input("مهارات أخرى (افصل بفاصلة)", placeholder="مثال: Power BI، إدارة مشاريع")
    with c2:
        mode = st.radio("الهدف", ["تخصص", "مسمى وظيفي", "كل السوق"], horizontal=True)
        spec = title = None
        if mode == "تخصص":
            spec = st.selectbox("التخصص", m.specializations())
        elif mode == "مسمى وظيفي":
            title = st.text_input("المسمى", placeholder="مثال: محاسب")
    skills = chosen + [s.strip() for s in free.replace("،", ",").split(",") if s.strip()]
    if st.button("حلّل الفجوة", type="primary"):
        if not skills:
            st.warning("اختر أو اكتب مهارة واحدة على الأقل.")
            st.stop()
        try:
            g = e.skill_gap(skills, target_specialization=spec, target_title=title or None)
        except ValueError as ex:
            st.warning(str(ex))
            st.stop()
        st.caption(f"الهدف: {g.target} · عدد إعلانات الهدف: {g.n_target_jobs:,}")
        if g.unrecognized:
            st.warning("مهارات غير معروفة في القاموس (لم تُحتسب): " + "، ".join(g.unrecognized))
        if g.note:
            st.info(g.note)
            st.stop()
        c1, c2 = st.columns([1, 2])
        with c1:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=g.match_pct, number={"suffix": "%"},
                                         gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#0F766E"}}))
            fig.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("نسبة التطابق مرجّحة بنسبة طلب كل مهارة في إعلانات الهدف")
        with c2:
            st.markdown("**تملكها ومطلوبة**")
            st.markdown(" ".join(f"<span class='chip ok'>{x['skill']} · {x['demand_pct']}%</span>" for x in g.matched) or "—",
                        unsafe_allow_html=True)
            st.markdown("**مطلوبة ولا تملكها**")
            st.markdown(" ".join(f"<span class='chip miss'>{x['skill']} · {x['demand_pct']}%</span>" for x in g.missing) or "—",
                        unsafe_allow_html=True)
        if g.recommendations:
            st.markdown("#### التوصيات")
            for i, r in enumerate(g.recommendations, 1):
                st.markdown(f"{i}. **{r['skill']}** — {r['why']}")
        st.markdown("#### أقرب الإعلانات لمهاراتك")
        if g.nearest_jobs is None or g.nearest_jobs.empty:
            st.info("لا يوجد إعلان يشترك معك في مهارة واحدة على الأقل.")
        else:
            nj = g.nearest_jobs.drop(columns=["job_id"]).rename(columns={"coverage": "تغطية مهارات الإعلان %", "jaccard": "Jaccard",
                                                                     "shared": "مشتركة", "missing": "تنقصك"})
            st.dataframe(nj, hide_index=True, width="stretch",
                         column_config={"source_url": st.column_config.LinkColumn("المصدر", display_text="فتح")})
