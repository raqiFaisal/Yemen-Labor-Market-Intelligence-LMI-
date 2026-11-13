"""AI Labor Market Intelligence — نقطة الدخول.

التشغيل:  streamlit run app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="AI Labor Market Intelligence", page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")

from utils.state import clf_metrics, encoder_info, get_data, lstm_metrics  # noqa: E402
from core.config import IS_DEMO_MODE  # noqa: E402
from utils.ui import setup_page  # noqa: E402

setup_page()

P = "pages/"
nav = st.navigation({
    "نظرة عامة": [
        st.Page(P + "1_dashboard.py", title="لوحة المعلومات", icon="📊", default=True),
        st.Page(P + "2_job_market.py", title="تحليل سوق العمل", icon="🔎"),
        st.Page(P + "3_skills.py", title="ذكاء المهارات", icon="🧠"),
    ],
    "التعلم العميق": [
        st.Page(P + "4_forecast.py", title="التنبؤ بالطلب المستقبلي", icon="📈"),
        st.Page(P + "9_models.py", title="النماذج وتقييمها", icon="🧬"),
    ],
    "أدوات": [
        st.Page(P + "5_academic.py", title="التخصصات الأكاديمية", icon="🎓"),
        st.Page(P + "6_gap.py", title="تحليل فجوة المهارات", icon="🎯"),
        st.Page(P + "7_custom.py", title="تحليل بيانات مخصصة", icon="📤"),
        st.Page(P + "10_review.py", title="مراجعة التصنيفات", icon="🏷️"),
        st.Page(P + "8_reports.py", title="التقارير", icon="📄"),
    ],
})

with st.sidebar:
    st.markdown("### AI Labor Market Intelligence")
    if IS_DEMO_MODE:
        st.markdown('<div class="side-card" style="background:rgba(245,158,11,.25);border-color:#F59E0B">'
                    '🧪 <b>وضع العرض التجريبي</b><br>بيانات ونماذج اصطناعية منفصلة — ليست نتائج حقيقية</div>',
                    unsafe_allow_html=True)
    st.caption("نظام ذكي لتحليل سوق العمل والتنبؤ باحتياجاته")
    jobs, _ = get_data()
    enc = encoder_info() or {}
    cm, lm = clf_metrics(), lstm_metrics()
    n = 0 if jobs is None else len(jobs)
    demo = jobs is not None and n and bool(jobs["is_demo"].any())
    enc_txt = {"transformer": "Transformer ✅", "tfidf_svd": "بديل TF-IDF ⚠️"}.get(enc.get("kind"), "غير مهيأ ❌")
    st.markdown(
        f"""<div class="side-card">
        <b>حالة النظام</b><br>
        البيانات: {n:,} إعلان {'(تجريبية ⚠️)' if demo else '(حقيقية)' if n else ''}<br>
        المُرمِّز: {enc_txt}<br>
        مصنف المجالات: {('مدرب ✅' + (' (تسميات ضعيفة)' if cm.get('eval_type') == 'silver' else '') + (' ↺ سابق' if cm.get('reused') else '')) if cm else 'غير مدرب ❌'}<br>
        نموذج LSTM: {'مدرب ✅' if lm else 'غير مدرب ❌'}
        </div>""", unsafe_allow_html=True)
    if not cm or not lm:
        st.caption("سبب عدم التدريب وتفاصيل الجاهزية في صفحة «النماذج وتقييمها».")

nav.run()
