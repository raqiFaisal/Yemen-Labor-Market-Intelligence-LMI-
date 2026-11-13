import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from models.gap import analyze_gap, closest_jobs, map_user_skills, requirement_profile
from utils.state import (demo_banner, domain_ar, get_domains, get_embeddings, get_encoder, get_majors, require_data,
                         taxonomy)
from utils.ui import AMBER, BLUE, RED, TEAL, hero, kpis, note

hero("تحليل فجوة المهارات", "قارن مهاراتك بمتطلبات الوظيفة التي تستهدفها كما تظهر في الإعلانات الفعلية", "Skills Gap Analysis")
jobs, js = require_data()
demo_banner(jobs)
skills, by_id = taxonomy()
enc, emb = get_encoder(), get_embeddings()

c1, c2, c3 = st.columns(3)
major = c1.selectbox("تخصصك الجامعي", list(get_majors()) + ["آخر"])
degree = c2.selectbox("المؤهل العلمي", ["بكالوريوس", "دبلوم", "ماجستير", "دكتوراه", "ثانوية"])
exp = c3.number_input("سنوات الخبرة", 0, 40, 1)

picked = st.multiselect("مهاراتك (من القاموس)", [s["id"] for s in skills],
                        format_func=lambda i: f"{by_id[i]['name_ar']} — {by_id[i]['name_en']}",
                        default=["excel", "data_analysis", "communication"])
free = st.text_input("مهارات إضافية بصياغتك (افصل بفاصلة)", placeholder="مثال: تحليل احصائي, عرض تقديمي, pandas")

levels = {}
if picked:
    st.markdown("**مستوى إتقانك لكل مهارة** (1 = مبتدئ، 5 = خبير)")
    cols = st.columns(4)
    for i, sid in enumerate(picked):
        levels[sid] = cols[i % 4].slider(by_id[sid]["name_ar"], 1, 5, 3, key=f"lvl_{sid}")

t1, t2 = st.columns([2, 1])
target = t1.text_input("الوظيفة المستهدفة", value="محلل بيانات")
doms = get_domains()
tdom = t2.selectbox("تقييد بمجال (اختياري)", ["—"] + list(doms), format_func=lambda d: "بدون" if d == "—" else domain_ar(d))

if st.button("تحليل الفجوة", type="primary"):
    user = dict(levels)
    mapped = {}
    if free.strip():
        raw = [x.strip() for x in re.split(r"[,،]", free) if x.strip()]
        mapped = map_user_skills(raw, enc)
        for r, sid in mapped.items():
            if sid and sid not in user:
                user[sid] = 3
    if mapped:
        st.caption("مطابقة المهارات المكتوبة: " + " • ".join(
            f"{r} ← {by_id[s]['name_ar'] if s else 'غير معروفة'}" for r, s in mapped.items()))
    with st.spinner("البحث الدلالي عن الإعلانات المطابقة وبناء ملف المتطلبات ..."):
        prof, matched = requirement_profile(target, jobs, js, emb, enc, domain=None if tdom == "—" else tdom)
    if prof.empty:
        note("لم يتم العثور على إعلانات كافية لهذه الوظيفة.", "warn")
        st.stop()
    res = analyze_gap(user, prof, enc)
    tb = res["table"]
    counts = tb["الحالة"].value_counts()
    score = res["score"]
    kpis([("نسبة التوافق", f"{score:.0f}%", "موزونة بأهمية كل مهارة", TEAL if score >= 70 else AMBER if score >= 40 else RED),
          ("مهارات متوفرة", counts.get("متوفرة", 0), "", TEAL),
          ("تحتاج تطوير / جزئية", counts.get("تحتاج تطوير", 0) + counts.get("قريبة (جزئية)", 0), "", AMBER),
          ("مهارات ناقصة", counts.get("ناقصة", 0), "", RED)])
    st.caption(f"ملف المتطلبات مبني من أقرب {len(matched)} إعلاناً لـ «{target}» "
               f"(متوسط التشابه {matched['similarity'].mean():.2f})" if matched['similarity'].notna().any() else "")

    a, b = st.columns([1, 1.3])
    with a:
        g = go.Figure(go.Indicator(mode="gauge+number", value=score, number={"suffix": "%"},
                                   gauge={"axis": {"range": [0, 100]}, "bar": {"color": BLUE},
                                          "steps": [{"range": [0, 40], "color": "#FEE2E2"}, {"range": [40, 70], "color": "#FEF3C7"},
                                                    {"range": [70, 100], "color": "#DCFCE7"}]}))
        g.update_layout(height=280, title="مؤشر الجاهزية للوظيفة")
        st.plotly_chart(g, width="stretch")
    with b:
        colors = {"متوفرة": TEAL, "تحتاج تطوير": AMBER, "قريبة (جزئية)": "#A78BFA", "ناقصة": RED}
        fig = go.Figure()
        for stt, col in colors.items():
            d = tb[tb["الحالة"] == stt]
            fig.add_bar(y=d["المهارة"], x=d["الأهمية"] * 100, name=stt, orientation="h", marker_color=col)
        fig.update_layout(barmode="stack", height=max(300, len(tb) * 24), title="أهمية المهارات المطلوبة وحالتك فيها",
                          xaxis_title="% من الإعلانات المطابقة التي تطلبها", yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, width="stretch")

    st.markdown("#### 📚 توصيات التطوير (حسب الأولوية)")
    rec = tb[tb["الحالة"] != "متوفرة"].sort_values("الأهمية", ascending=False)
    if rec.empty:
        st.success("ممتاز! ملفك يغطي المهارات الأساسية لهذه الوظيفة.")
    for i, r in enumerate(rec.head(8).itertuples(), 1):
        why = {"ناقصة": "مهارة غير موجودة في ملفك", "تحتاج تطوير": "موجودة بمستوى منخفض",
               "قريبة (جزئية)": "لديك أساس قريب يسهّل تعلمها"}[r.الحالة]
        st.markdown(f"**{i}. {r.المهارة}** ({r.Skill}) — مطلوبة في {r.الأهمية * 100:.0f}% من الإعلانات • {why}"
                    + (f" • {r.ملاحظة}" if r.ملاحظة else ""))

    with st.expander("الجدول التفصيلي"):
        st.dataframe(tb.drop(columns=["credit", "skill_id"]).assign(الأهمية=lambda d: (d["الأهمية"] * 100).round(1)),
                     hide_index=True, width="stretch")
    cj = closest_jobs(list(user), jobs, emb, enc)
    if len(cj):
        st.markdown("#### 🧭 الوظائف الأقرب لمهاراتك الحالية")
        cj["المجال"] = cj["domain"].map(domain_ar)
        st.dataframe(cj[["title", "المجال", "similarity", "postings"]].rename(
            columns={"title": "المسمى", "similarity": "التشابه", "postings": "عدد الإعلانات"}).round(3),
            hide_index=True, width="stretch")
    st.session_state["last_gap"] = {"target": target, "score": score, "major": major, "degree": degree, "exp": exp,
                                    "table": tb.drop(columns=["credit", "skill_id"])}
