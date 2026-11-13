import re

import pandas as pd
import plotly.express as px
import streamlit as st

from models.encoder import cosine_matrix
from preprocessing.schema import clean_text
from utils.state import (DOMAIN_SOURCE_AR, clf_status, demo_banner, domain_ar, get_classifier, get_domains,
                         get_embeddings, get_encoder, get_extractor, require_data)
from utils.ui import BLUE, hero, note, skill_badges

hero("تحليل سوق العمل", "استعراض الإعلانات وتصفيتها، وتحليل أي وصف وظيفي بالذكاء الاصطناعي", "Job Market Analysis")
jobs, js = require_data()
demo_banner(jobs)

tab1, tab2 = st.tabs(["📋 استعراض الإعلانات", "🤖 تحليل وصف وظيفة"])

with tab1:
    c = st.columns(5)
    market = c[0].selectbox("السوق", ["الكل", "اليمن", "عالمي/عربي"])
    countries = sorted(jobs["country"].dropna().unique())
    country = c[1].selectbox("الدولة", ["الكل"] + countries)
    doms = get_domains()
    dom = c[2].selectbox("المجال", ["الكل"] + list(doms), format_func=lambda d: "الكل" if d == "الكل" else domain_ar(d))
    govs = sorted(jobs.loc[jobs.market == "اليمن", "location"].dropna().unique())
    gov = c[3].selectbox("المحافظة", ["الكل"] + govs)
    q = c[4].text_input("بحث بالكلمات", placeholder="مثال: محاسب، Python")

    f = jobs
    if market != "الكل":
        f = f[f.market == market]
    if country != "الكل":
        f = f[f.country == country]
    if dom != "الكل":
        f = f[f.domain_final == dom]
    if gov != "الكل":
        f = f[f.location == gov]
    if q:
        m = f["title"].str.contains(q, case=False, na=False, regex=False) | \
            f["description"].str.contains(q, case=False, na=False, regex=False) | \
            f["requirements"].str.contains(q, case=False, na=False, regex=False)
        f = f[m]
    st.caption(f"عدد النتائج: {len(f):,}")
    view = f.assign(المجال=f.domain_final.map(domain_ar),
                    مصدر_التصنيف=f["domain_source"].map(DOMAIN_SOURCE_AR).fillna("—"))
    view = view[["title", "company", "country", "location", "المجال", "مصدر_التصنيف", "posted_date", "experience_years", "source"]]
    view.columns = ["المسمى", "الجهة", "الدولة", "المحافظة", "المجال", "مصدر التصنيف", "تاريخ النشر", "الخبرة", "المصدر"]
    st.dataframe(view, width="stretch", hide_index=True, height=380)
    st.download_button("⬇️ تنزيل النتائج CSV", view.to_csv(index=False).encode("utf-8-sig"), "jobs_filtered.csv", "text/csv")

    if len(f):
        a, b = st.columns(2)
        dd = f.domain_final.map(domain_ar).value_counts().reset_index()
        dd.columns = ["المجال", "n"]
        a.plotly_chart(px.pie(dd, names="المجال", values="n", hole=.55, title="تصنيف النتائج حسب المجال"),
                       width="stretch")
        cc = f.country.value_counts().head(10).reset_index()
        cc.columns = ["الدولة", "n"]
        b.plotly_chart(px.bar(cc, x="الدولة", y="n", title="النتائج حسب الدولة", color_discrete_sequence=[BLUE]),
                       width="stretch")

with tab2:
    sample = jobs.sample(1, random_state=7).iloc[0]
    default = f"{sample.title}\n{sample.description}\n{sample.requirements}"
    text = st.text_area("الصق نص الإعلان الوظيفي (عربي أو إنجليزي)", value=default, height=170)
    _sem_ok = get_extractor().semantic_available
    semantic = st.toggle("تفعيل المطابقة الدلالية (Semantic Matching)", value=_sem_ok, disabled=not _sem_ok,
                         help="تكتشف المهارات المذكورة بصياغات غير موجودة في القاموس عبر تشابه الـ Embeddings")
    if not _sem_ok:
        st.caption("المطابقة الدلالية تتطلب Transformer؛ المُرمِّز الحالي بديل TF-IDF فيُستخدم الاستخراج القاموسي فقط.")
    if st.button("تحليل الإعلان", type="primary"):
        t = clean_text(text)
        if len(t) < 10:
            st.warning("النص قصير جداً للتحليل.")
            st.stop()
        enc, clf = get_encoder(), get_classifier()
        with st.spinner("جاري التحليل ..."):
            all_sk = get_extractor().extract(t, semantic=semantic, include_review=True)
            skills = [x for x in all_sk if x["decision"] == "accepted"]
            review_sk = [x for x in all_sk if x["decision"] == "review"]
            emb = enc.encode([t]) if enc else None
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("#### المهارات المستخرجة")
            skill_badges(skills)
            st.caption("🔵 تقنية  🟢 شخصية  🟡 مهنية — الإطار المتقطع = اكتُشفت دلالياً")
            if skills:
                sk = pd.DataFrame(skills)
                sk["الطريقة"] = sk["method"].map({"lexicon": "قاموسية (قاعدية)", "semantic": "دلالية (Transformer)",
                                                "lexicon+semantic": "قاموسية + دلالية"})
                sk = sk.rename(columns={"score": "الثقة (استدلالية)", "evidence": "الدليل من النص", "reason": "سبب القبول"})
                cols = ["name_ar", "name_en", "type", "الطريقة", "الثقة (استدلالية)", "الدليل من النص", "سبب القبول"]
                st.dataframe(sk[cols].round(3), hide_index=True, width="stretch")
            if review_sk:
                st.markdown("**مرشحات لم تُقبل تلقائياً (تحتاج مراجعة بشرية):**")
                st.dataframe(pd.DataFrame(review_sk)[["name_ar", "method", "score", "evidence", "reason"]]
                             .rename(columns={"score": "الثقة", "evidence": "الدليل", "reason": "السبب"}),
                             hide_index=True, width="stretch")
            exp = re.findall(r"(\d+)\s*(?:سنوات|سنة|عام|years?|yrs)", t)
            if exp:
                st.markdown(f"**الخبرة المطلوبة:** {max(map(int, exp))} سنوات")
        with c2:
            st.markdown("#### تصنيف المجال (FNN)")
            if clf is None or emb is None:
                cs = clf_status()
                why = cs.get("reason") if cs["status"] != "trained" else \
                    "النموذج المحفوظ مدرب على مُرمِّز مختلف عن الحالي — أعد تشغيل scripts.train_all."
                note(f"المصنف غير متاح: {why}", "warn")
            elif emb.shape[1] != clf.in_dim:
                note("أبعاد المُرمِّز الحالي لا تطابق المصنف المدرب — أعد التدريب.", "warn")
            else:
                p = clf.predict_proba(emb)[0]
                pr = pd.DataFrame({"المجال": [domain_ar(l) for l in clf.labels], "p": p * 100}).sort_values("p")
                top = pr.iloc[-1]
                st.metric("المجال المتوقع", top["المجال"], f"ثقة {top.p:.1f}%")
                fig = px.bar(pr, x="p", y="المجال", orientation="h", labels={"p": "الاحتمال %"},
                             color="p", color_continuous_scale="Blues")
                fig.update_layout(height=320, coloraxis_showscale=False)
                st.plotly_chart(fig, width="stretch")
        allemb = get_embeddings()
        if emb is not None and allemb is not None and allemb.shape[1] == emb.shape[1]:
            st.markdown("#### أقرب الإعلانات دلالياً في قاعدة البيانات")
            sims = cosine_matrix(emb, allemb)[0]
            top = jobs.assign(التشابه=sims).nlargest(6, "التشابه")
            st.dataframe(top[["title", "company", "country", "location", "التشابه"]].round(3),
                         hide_index=True, width="stretch")
