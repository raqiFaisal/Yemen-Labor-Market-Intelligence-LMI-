import pandas as pd
import plotly.express as px
import streamlit as st

from utils.report import _table, build_excel, build_html, kpi_html
from utils.state import clf_metrics, demo_banner, domain_ar, lstm_metrics, require_data, skills_table
from utils.ui import hero, note

hero("التقارير", "تصدير نتائج التحليل والتنبؤ في تقرير قابل للتنزيل والطباعة", "Reports")
jobs, js = require_data()
demo_banner(jobs)
S = skills_table(js, jobs)
ss = st.session_state

st.markdown("#### محتوى التقرير")
c = st.columns(3)
inc_market = c[0].checkbox("ملخص سوق العمل", True)
inc_skills = c[0].checkbox("المهارات الأكثر طلباً", True)
inc_models = c[1].checkbox("تقييم النماذج", True)
inc_fc = c[1].checkbox("آخر تنبؤ LSTM", "last_forecast" in ss, disabled="last_forecast" not in ss)
inc_rank = c[2].checkbox("ترتيب نمو المهارات", "ranking" in ss, disabled="ranking" not in ss)
inc_gap = c[2].checkbox("آخر تحليل فجوة", "last_gap" in ss, disabled="last_gap" not in ss)
market = st.radio("نطاق السوق في الملخص", ["اليمن", "عالمي/عربي", "الكل"], horizontal=True)
title = st.text_input("عنوان التقرير", "تقرير ذكاء سوق العمل")
if not {"last_forecast", "ranking", "last_gap"} & set(ss):
    note("لتضمين نتائج التنبؤ أو فجوة المهارات، شغّلها أولاً من صفحاتها ثم عد إلى هنا.")

J = jobs if market == "الكل" else jobs[jobs.market == market]
SS = S if market == "الكل" else S[S.market == market]
sections, sheets = [], {}
if inc_market:
    dom = J.domain_final.map(domain_ar).value_counts().rename_axis("المجال").reset_index(name="عدد الوظائف")
    gov = J.location.fillna("غير محدد").value_counts().head(10).rename_axis("المحافظة/الموقع").reset_index(name="عدد الوظائف")
    fig = px.bar(dom, x="عدد الوظائف", y="المجال", orientation="h", title="الوظائف حسب المجال")
    sections.append(("ملخص سوق العمل — " + market, kpi_html([
        ("إجمالي الوظائف", f"{len(J):,}"), ("المهارات المكتشفة", SS.skill_id.nunique()),
        ("أكثر المجالات طلباً", dom.iloc[0, 0] if len(dom) else "—")]) +
        fig.to_html(include_plotlyjs="cdn", full_html=False) + _table(dom) + _table(gov)))
    sheets["المجالات"] = dom
    sheets["المواقع"] = gov
if inc_skills:
    top = SS.groupby(["name_ar", "name_en", "type"]).job_id.nunique().reset_index(name="عدد الإعلانات")
    top["النسبة %"] = top["عدد الإعلانات"] / max(len(J), 1) * 100
    top = top.sort_values("عدد الإعلانات", ascending=False)
    sections.append(("المهارات الأكثر طلباً", _table(top, 30)))
    sheets["المهارات"] = top
if inc_models:
    cm, lm = clf_metrics(), lstm_metrics()
    parts = []
    if cm:
        t = cm["test"]
        et = {"gold": "تصنيف يدوي مستقل", "source": "تصنيف المصدر", "silver": "تسميات ضعيفة — اتفاق مع القواعد وليس دقة حقيقية",
              "gold_mixed": "خليط يدوي"}.get(cm.get("eval_type", "source"), "")
        parts.append(f"<p><b>مصنف المجالات (FNN):</b> مُرمِّز {cm['encoder']['name']} • {cm['n_train']} تدريب / {cm['n_test']} اختبار • نوع التقييم: {et}</p>"
                     + kpi_html([("Accuracy", f"{t['accuracy']:.3f}"), ("Precision", f"{t['precision_macro']:.3f}"),
                                 ("Recall", f"{t['recall_macro']:.3f}"), ("F1", f"{t['f1_macro']:.3f}")]))
    if lm and "test" in lm:
        ev = pd.DataFrame([{"النموذج": k, "MAE": v["mae"] * 100, "RMSE": v["rmse"] * 100} for k, v in lm["test"].items()])
        parts.append(f"<p><b>نموذج LSTM:</b> {lm['n_series']} سلسلة زمنية • آخر {lm['test_months']} أشهر للاختبار (بالنقاط المئوية)</p>" + _table(ev))
        sheets["تقييم LSTM"] = ev
    from utils.state import clf_status, lstm_status
    if not cm:
        parts.append(f"<p><b>مصنف المجالات:</b> غير مدرب — {clf_status().get('reason', '')}</p>")
    if not (lm and "test" in lm):
        parts.append(f"<p><b>LSTM:</b> غير مدرب — {lstm_status().get('reason', '')}</p>")
    sections.append(("تقييم النماذج", "".join(parts)))
if inc_fc and "last_forecast" in ss:
    f = ss["last_forecast"]
    sections.append((f"تنبؤ الطلب على «{f['skill']}»", kpi_html([("النطاق", f["scope"]), ("المجال", f["domain"]),
                     ("الاتجاه المتوقع", f"{f['trend']} ({f['change']:+.0f}%)"), ("الأفق", f"{f['horizon']} أشهر")]) +
                     _table(f["table"]) + "<p class='meta'>التوقعات تقديرية وليست ضماناً.</p>"))
    sheets["التنبؤ"] = f["table"]
if inc_rank and "ranking" in ss:
    sections.append(("ترتيب المهارات حسب النمو المتوقع", _table(ss["ranking"], 40)))
    sheets["ترتيب النمو"] = ss["ranking"]
if inc_gap and "last_gap" in ss:
    g = ss["last_gap"]
    sections.append((f"تحليل فجوة المهارات — {g['target']}", kpi_html([("نسبة التوافق", f"{g['score']:.0f}%"),
                     ("التخصص", g["major"]), ("المؤهل", g["degree"]), ("الخبرة", f"{g['exp']} سنة")]) + _table(g["table"], 40)))
    sheets["فجوة المهارات"] = g["table"]

if not sections:
    st.info("اختر قسماً واحداً على الأقل.")
    st.stop()
demo = bool(jobs.is_demo.any())
html_doc = build_html(title, sections, demo)
a, b = st.columns(2)
a.download_button("⬇️ تنزيل التقرير HTML (قابل للطباعة PDF)", html_doc.encode("utf-8"), "lmi_report.html", "text/html",
                  type="primary", width="stretch")
if sheets:
    b.download_button("⬇️ تنزيل البيانات Excel", build_excel(sheets), "lmi_report.xlsx",
                      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
st.caption("لحفظه PDF: افتح ملف HTML في المتصفح ← طباعة ← حفظ كـ PDF (يحافظ على العربية والاتجاه).")
with st.expander("معاينة التقرير", expanded=True):
    st.iframe(html_doc, height=700)
