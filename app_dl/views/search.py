import streamlit as st

from common import TYPE_AR, engine, need_data, setup

setup("البحث في الإعلانات", "بحث نصي مع تصفية بالمحافظة والنوع والتخصص والمهارة")
if need_data():
    m = engine().market
    q = st.text_input("كلمات البحث", placeholder="مثال: محاسب، مهندس مدني، Excel")
    c1, c2, c3, c4 = st.columns(4)
    gov = c1.selectbox("المحافظة", ["الكل"] + m.governorates())
    types = sorted(m.all_unique.opportunity_type.unique())
    typ = c2.selectbox("النوع", ["الكل"] + types, format_func=lambda x: TYPE_AR.get(x, x))
    spec = c3.selectbox("التخصص", ["الكل"] + m.specializations())
    cat = m.catalog.sort_values("name_ar")
    sk = c4.selectbox("المهارة", ["الكل"] + cat.skill_id.tolist(),
                      format_func=lambda x: x if x == "الكل" else m.skill_name(x))
    res = m.search(q, None if gov == "الكل" else gov, None if typ == "الكل" else typ,
                   None if spec == "الكل" else spec, None if sk == "الكل" else sk, limit=500)
    st.markdown(f"**{len(res):,}** نتيجة (يُعرض حتى 500، الأحدث أولاً)")
    if not res.empty:
        res = res.assign(opportunity_type=res.opportunity_type.map(lambda x: TYPE_AR.get(x, x)))
        st.dataframe(res, hide_index=True, width="stretch", height=560,
                     column_config={"source_url": st.column_config.LinkColumn("المصدر", display_text="فتح")})
        st.download_button("تنزيل النتائج CSV", res.to_csv(index=False).encode("utf-8-sig"), "search_results.csv", "text/csv")
