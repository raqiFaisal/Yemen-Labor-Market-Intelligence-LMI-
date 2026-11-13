import streamlit as st
from common import ar_scope, ar_type, bar, engine, kpis, line, need_data, pie, setup

setup("لوحة المعلومات", "مؤشرات عامة من الإعلانات الفريدة بعد إزالة التكرار")
if need_data():
    m = engine().market
    o = m.overview()
    kpis([("فرصة فريدة", o["unique_opportunities"]), ("وظيفة وتدريب عملي", o["unique_jobs_internships"]),
          ("جهة معلنة", o["organizations"]), ("محافظة", o["governorates_covered"]), ("مهارة مميزة", o["distinct_skills"])])
    st.caption(f"الفترة: {o['date_from']} ← {o['date_to']} · المصادر: {o['sources']} · "
               f"إعلانات بمهارة مستخرجة واحدة على الأقل: {o['jobs_with_skills_pct']}%")
    st.divider()
    c1, c2 = st.columns([3, 2])
    with c1:
        mon = m.monthly().rename(columns=ar_type)
        line(mon, "عدد الإعلانات الفريدة شهرياً")
    with c2:
        d = m.distribution("opportunity_type")
        d["opportunity_type"] = d.opportunity_type.map(ar_type)
        pie(d, "opportunity_type", "count", "توزيع أنواع الفرص (كل الأنواع)")
    c1, c2 = st.columns(2)
    with c1:
        bar(m.top_skills(15), "jobs", "skill", "أكثر المهارات طلباً (عدد الإعلانات)")
    with c2:
        g = m.distribution("governorate", m.jobs).query("governorate != 'غير محدد'").head(15)
        bar(g, "count", "governorate", "أكثر المحافظات إعلاناً", color="#2563EB")
    c1, c2 = st.columns(2)
    with c1:
        bar(m.top_titles(15), "jobs", "job_title", "أكثر المسميات الوظيفية", color="#D97706")
    with c2:
        s = m.distribution("geographic_scope", m.jobs)
        s["geographic_scope"] = s.geographic_scope.map(ar_scope)
        pie(s, "geographic_scope", "count", "النطاق الجغرافي للوظائف")
