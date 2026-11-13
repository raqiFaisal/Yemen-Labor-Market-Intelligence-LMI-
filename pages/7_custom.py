import pandas as pd
import plotly.express as px
import streamlit as st

from core.pipeline import extract_job_skills, map_domain
from preprocessing.schema import UNIFIED_COLUMNS, job_text, map_columns, unify
from utils.state import domain_ar, get_classifier, get_encoder, taxonomy
from utils.ui import BLUE, TEAL, hero, kpis, note

hero("تحليل بيانات مخصصة", "ارفع ملف إعلانات وظائف بأي تنسيق أعمدة وحلّله بالنماذج المدربة", "Custom Data Analysis")
note("الملف المرفوع يُحلَّل داخل الجلسة فقط ولا يُضاف لقاعدة التدريب. لإضافته للتدريب ضعه في <code>data/raw/</code> وأعد التدريب.")

up = st.file_uploader("ملف CSV أو Excel", type=["csv", "xlsx"])
if up is None:
    with st.expander("التنسيق المتوقع"):
        st.markdown("يكفي وجود عمود للمسمى الوظيفي **أو** للوصف. الأعمدة الأخرى اختيارية ويتم التعرف عليها تلقائياً "
                    "(عربية أو إنجليزية). أمثلة: `title / المسمى الوظيفي`، `description / الوصف`، `company`، `location`، `date`.")
        st.download_button("⬇️ ملف نموذجي", pd.DataFrame({
            "job_title": ["محاسب", "Data Analyst"], "description": ["خبرة في Excel والتقارير المالية", "SQL, Python, Power BI"],
            "company": ["شركة أ", "Company B"], "city": ["عدن", "Riyadh"], "country": ["Yemen", "Saudi Arabia"],
            "date": ["2026-05-01", "2026-06-10"]}).to_csv(index=False).encode("utf-8-sig"), "sample_jobs.csv")
    st.stop()

try:
    raw = pd.read_excel(up) if up.name.endswith(".xlsx") else pd.read_csv(up, encoding="utf-8-sig", low_memory=False)
except Exception as e:  # noqa: BLE001
    st.error(f"تعذّرت قراءة الملف: {e}")
    st.stop()
if raw.empty:
    st.error("الملف فارغ.")
    st.stop()
st.success(f"تمت قراءة {len(raw):,} صفاً و {raw.shape[1]} عموداً.")
st.dataframe(raw.head(5), width="stretch")

st.markdown("#### مطابقة الأعمدة")
auto = map_columns(raw)
targets = ["—"] + [c for c in UNIFIED_COLUMNS if c not in ("job_id", "is_demo")]
cols = st.columns(4)
mapping = {}
for i, col in enumerate(raw.columns):
    t = cols[i % 4].selectbox(str(col), targets, index=targets.index(auto[col]) if col in auto else 0, key=f"map_{col}")
    if t != "—":
        mapping[col] = t
c1, c2 = st.columns(2)
default_country = c1.text_input("الدولة الافتراضية للسجلات بلا دولة", "Yemen")
semantic = c2.toggle("استخراج دلالي إضافي (أبطأ)", value=False)

if st.button("تشغيل التحليل", type="primary"):
    try:
        df = unify(raw, source=f"upload:{up.name}", country=default_country or None, mapping=mapping)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    df["domain"] = df["category_raw"].fillna(df["domain"]).map(map_domain)
    enc, clf = get_encoder(), get_classifier()
    prog = st.progress(0, "استخراج المهارات ...")
    jsk = extract_job_skills(df, enc, semantic=bool(semantic and enc is not None and enc.kind == "transformer"))
    prog.progress(50, "تصنيف المجالات بالنموذج ...")
    if clf is not None and enc is not None:
        e = enc.encode(job_text(df).tolist())
        if e.shape[1] == clf.in_dim:
            df["domain_pred"], df["domain_conf"] = clf.predict(e)
        else:
            st.warning("أبعاد المُرمِّز لا تطابق المصنف — تخطي التصنيف.")
    else:
        st.warning("المصنف أو المُرمِّز غير متاح — تخطي التصنيف.")
    prog.progress(100, "اكتمل")
    _, by_id = taxonomy()
    names = jsk.groupby("job_id").skill_id.apply(lambda s: "، ".join(by_id[i]["name_ar"] for i in s))
    df["skills"] = df.job_id.map(names).fillna("")
    df["skills_count"] = df.job_id.map(jsk.groupby("job_id").size()).fillna(0).astype(int)

    kpis([("سجلات صالحة", f"{len(df):,}", f"من {len(raw):,} صف"), ("مهارات مكتشفة", jsk.skill_id.nunique(), "", TEAL),
          ("متوسط المهارات/إعلان", f"{df.skills_count.mean():.1f}", "", BLUE),
          ("سجلات بلا مهارات", int((df.skills_count == 0).sum()), "قد تحتاج مراجعة")])
    a, b = st.columns(2)
    top = jsk.skill_id.value_counts().head(15).rename_axis("sid").reset_index(name="n")
    top["المهارة"] = top.sid.map(lambda i: by_id[i]["name_ar"])
    fig = px.bar(top, x="n", y="المهارة", orientation="h", title="أكثر المهارات في الملف", color_discrete_sequence=[BLUE])
    fig.update_layout(yaxis=dict(categoryorder="total ascending"), height=420)
    a.plotly_chart(fig, width="stretch")
    if "domain_pred" in df:
        d = df.domain_pred.map(domain_ar).value_counts().reset_index()
        d.columns = ["المجال", "n"]
        b.plotly_chart(px.pie(d, names="المجال", values="n", hole=.5, title="المجالات حسب نموذج التصنيف"),
                       width="stretch")
        if df["domain"].notna().sum() >= 10:
            lab = df[df["domain"].notna()]
            acc = (lab["domain"] == lab["domain_pred"]).mean()
            note(f"الملف يحوي تصنيفات أصلية (بعد التوحيد) لـ {len(lab)} سجلاً — اتفاق النموذج معها: <b>{acc * 100:.1f}%</b>. "
                 "هذه ليست دقة اختبار رسمية إن كانت هذه السجلات قد استُخدمت في التدريب.")
    out = df.assign(domain_pred_ar=df.get("domain_pred", pd.Series(dtype=str)).map(domain_ar))
    st.dataframe(out[["title", "company", "country", "location", "skills", "domain_pred_ar"] +
                     (["domain_conf"] if "domain_conf" in out else [])], width="stretch", hide_index=True)
    st.download_button("⬇️ تنزيل النتائج المُعلَّمة", out.to_csv(index=False).encode("utf-8-sig"), "analyzed_jobs.csv", "text/csv")
    st.session_state["last_custom"] = {"file": up.name, "rows": len(df), "top_skills": top[["المهارة", "n"]]}
