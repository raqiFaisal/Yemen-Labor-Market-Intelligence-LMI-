import plotly.graph_objects as go
import streamlit as st

from common import PALETTE, engine, setup

FAM_AR = {"skills": "المهارات", "specializations": "التخصصات", "job_titles": "المسميات الوظيفية", "governorates": "المحافظات"}
setup("التنبؤ بالطلب (LSTM)", "التوقعات تُقرأ من نتائج التدريب المحفوظة — التطبيق لا يعيد التدريب")
f = engine().forecasts
summ = f.summary()
if summ is None:
    st.warning("لم يُشغَّل تدريب LSTM بعد، فلا توجد توقعات لعرضها.")
    st.code("python -m src.forecasting.train_lstm", language="bash")
    st.stop()
fam = st.radio("العائلة", list(FAM_AR), format_func=FAM_AR.get, horizontal=True)
r = f.family(fam)
if r["status"] != "ok":
    st.error(f"لم يُدرَّب نموذج لهذه العائلة: {r.get('reason') or r['status']}")
    if r.get("sufficiency") is not None:
        st.markdown("**فحص كفاية السلاسل** (شرط التدريب: عدد أشهر، تغطية، وعدد إشارات كافٍ):")
        st.dataframe(r["sufficiency"], width="stretch", hide_index=True)
    st.stop()
mt = r["metrics"]
if "target_policy" not in mt:
    st.warning("⚠ هذه أوزان LSTM محفوظة من تدريب سابق. تم اكتشاف خلل في طريقة التعامل مع أهداف الأشهر غير المرصودة وإصلاحه في كود إعادة التدريب، لكن هذه الأوزان لم يُعَد تدريبها بعد. نتائجها استرشادية ولا تُعد تقييمًا أكاديميًا نهائيًا.")
st.caption(f"نافذة البيانات {mt['window'][0]} ← {mt['window'][1]} · تدريب {mt['train_period'][0]}–{mt['train_period'][1]} · "
           f"تحقق {mt['validation_period'][0]}–{mt['validation_period'][1]} · اختبار {mt['test_period'][0]}–{mt['test_period'][1]}")
tr = r["trend"].copy()
if fam == "skills":
    tr["item"] = tr["item"].map(engine().market.skill_name)
st.markdown("**الاتجاه المتوقع للأشهر القادمة**")
st.dataframe(tr.rename(columns={"item": "البند", "recent_6m_mean": "متوسط آخر 6 أشهر", "forecast_mean": "متوسط التوقع",
                                "change": "التغير", "direction": "الاتجاه"}), hide_index=True, width="stretch")
hist, fut, tp = r["history"].copy(), r["future"].copy(), r["test_predictions"].copy()
if fam == "skills":
    nm = engine().market.skill_name
    hist, fut = hist.rename(columns=nm), fut.rename(columns=nm)
    tp["item"] = tp["item"].map(nm)
items = list(hist.iloc[-6:].mean().sort_values(ascending=False).index)      # الأعلى حصة حديثاً أولاً
pick = st.multiselect("البنود", items, default=items[:3])
fig = go.Figure()
for i, it in enumerate(pick):
    c = PALETTE[i % len(PALETTE)]
    fig.add_scatter(x=hist.index.astype(str), y=hist[it], name=f"{it} — مرصود", line=dict(color=c))
    p = tp[tp.item == it]
    fig.add_scatter(x=p.month.astype(str), y=p.lstm, name=f"{it} — LSTM (اختبار)", line=dict(color=c, dash="dot"))
    fig.add_scatter(x=fut.index.astype(str), y=fut[it], name=f"{it} — توقع", line=dict(color=c, dash="dash"))
fig.update_layout(height=460, yaxis_tickformat=".0%", yaxis_title="حصة الإعلانات الشهرية", legend_title="",
                  margin=dict(l=10, r=10, t=10, b=10), font=dict(family="IBM Plex Sans Arabic"))
st.plotly_chart(fig, width="stretch")
st.caption("الخطوط المتقطعة توقعات وليست قيماً مرصودة. الأشهر بعدد إعلانات قليل تُعامل كغير مرصودة (فراغ) لا كصفر.")
with st.expander("خطأ كل سلسلة على الاختبار"):
    st.dataframe(r["per_series"], hide_index=True, width="stretch")
