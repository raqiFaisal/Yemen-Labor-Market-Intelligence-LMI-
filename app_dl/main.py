"""منصة ذكاء سوق العمل اليمني بالتعلم العميق — نقطة الدخول.

    streamlit run app_dl/main.py

لا يُعيد التطبيق تدريب أي نموذج: يقرأ البيانات النهائية + النماذج المحفوظة + تقارير التقييم فقط.
المسارات: LMI_FINAL_DIR / LMI_MODELS_DIR / LMI_REPORTS_DIR (الافتراضي داخل المشروع).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="ذكاء سوق العمل اليمني", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

V = "views/"
nav = st.navigation({
    "السوق": [
        st.Page(V + "dashboard.py", title="لوحة المعلومات", icon="📊", default=True),
        st.Page(V + "market.py", title="تحليل السوق", icon="🔎"),
        st.Page(V + "forecast.py", title="التنبؤ (LSTM)", icon="📈"),
        st.Page(V + "search.py", title="البحث في الإعلانات", icon="🗂️"),
    ],
    "التعلم العميق": [
        st.Page(V + "analyze.py", title="تحليل إعلان (نماذج مدربة)", icon="🧠"),
        st.Page(V + "gap.py", title="فجوة المهارات", icon="🎯"),
        st.Page(V + "models.py", title="معلومات النماذج", icon="🧬"),
    ],
    "المشروع": [st.Page(V + "about.py", title="حول المشروع", icon="ℹ️")],
})
with st.sidebar:
    st.markdown("### ذكاء سوق العمل اليمني")
nav.run()
