import plotly.express as px
import streamlit as st

from common import ar_scope, bar, engine, need_data, setup

setup("تحليل السوق", "صفِّ حسب المحافظة والتخصص والفترة؛ كل الأرقام من الإعلانات الفريدة (وظائف + تدريب عملي)")
if need_data():
    m = engine().market
    months = sorted(x for x in m.jobs.month.unique() if x)
    c1, c2, c3 = st.columns([1, 1, 2])
    gov = c1.selectbox("المحافظة", ["الكل"] + m.governorates())
    spec = c2.selectbox("التخصص", ["الكل"] + m.specializations())
    rng = c3.select_slider("الفترة", options=months, value=(months[0], months[-1])) if len(months) > 1 else (None, None)
    df = m.filtered(None if gov == "الكل" else gov, None if spec == "الكل" else spec, rng[0], rng[1])
    st.markdown(f"**{len(df):,}** إعلاناً فريداً مطابقاً للتصفية")
    if df.empty:
        st.warning("لا توجد إعلانات بهذه التصفية.")
        st.stop()
    t1, t2, t3, t4 = st.tabs(["المهارات", "المسميات والتخصصات", "الجغرافيا", "التخصص ↔ المهارة"])
    with t1:
        typ = st.radio("نوع المهارة", ["الكل", "technical", "soft", "professional"], horizontal=True,
                       format_func={"الكل": "الكل", "technical": "تقنية", "soft": "شخصية", "professional": "مهنية"}.get)
        ts = m.top_skills(25, df, None if typ == "الكل" else typ)
        bar(ts, "pct_of_jobs", "skill", "نسبة الإعلانات التي تطلب المهارة (%)", height=640)
        pick = st.multiselect("اتجاه شهري لمهارات مختارة", ts.skill.tolist(), default=ts.skill.tolist()[:4])
        if pick:
            ids = ts[ts.skill.isin(pick)].skill_id.tolist()
            from common import line
            line(m.skill_trend(ids, df), "الحصة الشهرية من الإعلانات (مرصودة)", yfmt=".0%")
    with t2:
        c1, c2 = st.columns(2)
        with c1:
            bar(m.top_titles(20, df), "jobs", "job_title", "المسميات الوظيفية", height=560, color="#D97706")
        with c2:
            bar(m.top_multi("specializations", 20, df), "jobs", "specializations", "التخصصات المطلوبة", height=560,
                color="#7C3AED")
        c1, c2 = st.columns(2)
        with c1:
            bar(m.distribution("education", df).query("education != 'غير محدد'").head(10), "count", "education",
                "المؤهل المطلوب", color="#0891B2")
        with c2:
            ex = df.experience_min_years.replace("", None).dropna().astype(float)
            if len(ex):
                st.plotly_chart(px.histogram(ex, nbins=15, title="الحد الأدنى لسنوات الخبرة",
                                             color_discrete_sequence=["#0F766E"]).update_layout(showlegend=False),
                                width="stretch")
    with t3:
        c1, c2 = st.columns(2)
        with c1:
            bar(m.distribution("governorate", df).query("governorate != 'غير محدد'").head(20), "count", "governorate",
                "المحافظات", height=560, color="#2563EB")
        with c2:
            s = m.distribution("geographic_scope", df)
            s["geographic_scope"] = s.geographic_scope.map(ar_scope)
            bar(s, "count", "geographic_scope", "النطاق الجغرافي", color="#65A30D")
    with t4:
        r = m.spec_skill()
        if r["share"].empty:
            st.info("لا توجد تخصصات بعدد إعلانات كافٍ (≥ 5) لبناء المصفوفة.")
        else:
            mode = st.radio("المقياس", ["الحصة", "Lift"], horizontal=True,
                            help="الحصة = نسبة إعلانات التخصص التي تطلب المهارة. Lift > 1 = المهارة أكثر ارتباطاً بهذا التخصص من المتوسط.")
            mat = r["share"] if mode == "الحصة" else r["lift"]
            fig = px.imshow(mat.astype(float), aspect="auto", color_continuous_scale="Teal", text_auto=".0%" if mode == "الحصة" else ".1f",
                            height=520)
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), font=dict(family="IBM Plex Sans Arabic"))
            st.plotly_chart(fig, width="stretch")
