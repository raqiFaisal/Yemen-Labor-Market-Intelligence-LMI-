import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.config import COUNTRY_YEMEN, FREQ_AR, IS_DEMO_MODE, LSTM_PREDICTIONS, LSTM_TEST_MONTHS
from models.forecaster import monthly_share, series_sufficiency, trend_label
from utils.state import (demo_banner, domain_ar, get_domains, get_forecaster, lstm_metrics, lstm_status, require_data,
                         taxonomy)
from utils.ui import AMBER, BLUE, GRAY, NAVY, RED, TEAL, hero, kpis, note

hero("التنبؤ بالطلب المستقبلي على المهارات", "نموذج LSTM مدرب على السلاسل الزمنية لنسبة الإعلانات التي تطلب كل مهارة",
     "Future Demand Prediction • LSTM")
jobs, js = require_data()
demo_banner(jobs)
fc_model, M = get_forecaster(), lstm_metrics()
if fc_model is None or not M:
    S_ = lstm_status()
    note(f"❌ <b>نموذج LSTM غير مدرب على البيانات الحالية.</b> السبب: {S_.get('reason', '—')}", "warn")
    det = S_.get("details") or {}
    markets = det.get("by_market") or ({"—": det} if det.get("n_jobs") is not None else {})
    for mk, d in markets.items():
        st.markdown(f"**السوق: {'اليمن' if mk == 'Yemen' else 'عالمي/عربي' if mk == 'Global' else mk}**")
        kpis([("الإعلانات المؤرخة", f"{d.get('n_dated', 0):,} / {d.get('n_jobs', 0):,}", f"{d.get('dated_pct', 0):.0f}% لها تاريخ نشر"),
              ("المدى الزمني المتاح", f"{d.get('span_periods', 0)} {d.get('unit', '')}",
               f"{d.get('first') or '—'} → {d.get('last') or '—'}", AMBER),
              ("المطلوب", f"{d.get('min_periods')} {d.get('unit', '')}", f"ينقص {d.get('missing_periods', 0)}", RED),
              ("فترات مرصودة", d.get("observed_periods", 0), f"≥ {d.get('min_postings_per_period', 10)} إعلانات/فترة", GRAY)])
    if det.get("skip_reasons"):
        st.caption("أسباب استبعاد السلاسل: " + " • ".join(f"{k} ({v})" for k, v in det["skip_reasons"].items()))
    note("<b>كيف تصل للجاهزية؟</b><br>"
         "1) YemenHR يعرض الإعلانات النشطة فقط، لذا يُبنى التاريخ بالجمع الدوري: <code>python -m scripts.scrape_yemen</code><br>"
         "2) شغّل الجامع دورياً (يُلحق الجديد فقط) واحفظ <code>data/raw</code> في Google Drive حتى يتراكم التاريخ.<br>"
         "3) مع جمع يومي قصير المدى يمكن استخدام سلاسل أسبوعية: <code>python -m scripts.train_all --freq W</code><br>"
         "لن يعرض التطبيق أي تنبؤ قبل توفر بيانات كافية. للتوضيح فقط يوجد وضع عرض تجريبي منفصل: "
         "<code>LMI_MODE=demo streamlit run app.py</code> (نتائجه موسومة كتجريبية).")
    st.stop()
FREQ = M.get("freq", "M")
UNIT1, UNIT = FREQ_AR.get(FREQ, ("شهر", "أشهر"))

skills, by_id = taxonomy()
freq = js.skill_id.value_counts()
opts = [s for s in freq.index if s in by_id]
DEMO_TAG = " — نتائج تجريبية (DEMO)" if (IS_DEMO_MODE or M.get("trained_on_demo_data")) else ""
if DEMO_TAG:
    note("🧪 <b>وضع العرض التجريبي:</b> النموذج مدرب على بيانات اصطناعية باتجاهات مبرمجة. كل الأرقام هنا "
         "<b>توضيحية وليست توقعات فعلية لسوق العمل</b>.", "warn")

COMPARE = "مقارنة: اليمن مقابل العالمي"
c = st.columns(4)
sid = c[0].selectbox("المهارة", opts, format_func=lambda i: f"{by_id[i]['name_ar']} ({by_id[i]['name_en']})")
doms = get_domains()
dom = c[1].selectbox("القطاع / المجال", ["الكل"] + list(doms), format_func=lambda d: "الكل" if d == "الكل" else domain_ar(d))
# الأسواق منفصلة: لا يوجد خيار يخلط اليمن بالعالمي في سلسلة واحدة
scopes = ["اليمن", "عالمي/عربي (كل الدول عدا اليمن)", COMPARE] + sorted(set(jobs.country.dropna()) - {COUNTRY_YEMEN, "Unknown"})
scope = c[2].selectbox("السوق / الدولة", scopes)
horizon = c[3].slider(f"الفترة المستقبلية ({UNIT})", 3, 12, 6)


def subset(scope, dom):
    s = jobs
    if scope == "اليمن":
        s = s[s.country == COUNTRY_YEMEN]
    elif scope.startswith("عالمي"):
        s = s[s.country != COUNTRY_YEMEN]
    else:
        s = s[s.country == scope]
    if dom != "الكل":
        s = s[s.domain_final == dom]
    return s


def get_series(scope):
    sh = monthly_share(subset(scope, dom), js, sid, freq=FREQ)
    if sh.empty or sid not in sh:
        return None, 0, False, "لا توجد بيانات مؤرخة لهذه المهارة ضمن التصفية."
    ser = sh[sid]
    men = int(sh.attrs["counts"][sid].sum())
    ok_, why_ = series_sufficiency(ser, men, M.get("min_periods", 18), M.get("min_mentions", 25), FREQ)
    return ser, men, ok_, why_


if scope == COMPARE:
    fig = go.Figure()
    rows_ = []
    for name, sc_, col in (("اليمن", "اليمن", BLUE), ("عالمي/عربي", "عالمي/عربي (كل الدول عدا اليمن)", AMBER)):
        ser, men, ok_, why_ = get_series(sc_)
        if ser is None:
            rows_.append({"السوق": name, "الحالة": why_})
            continue
        x = ser.index.to_timestamp()
        fig.add_scatter(x=x, y=ser * 100, mode="markers", name=f"{name}: فعلي (مرصود)", marker=dict(color=col, size=5, opacity=.5))
        if ok_:
            o = fc_model.forecast(ser.to_numpy(), horizon=horizon)
            fut = pd.period_range(ser.index[-1] + 1, periods=horizon, freq=FREQ).to_timestamp()
            fig.add_scatter(x=x, y=o["smoothed_history"] * 100, mode="lines", name=f"{name}: تاريخي (متوسط 3)", line=dict(color=col))
            fig.add_scatter(x=fut, y=o["point"] * 100, mode="lines+markers", name=f"{name}: توقع LSTM", line=dict(color=col, dash="dash"))
            lb, ch = trend_label(o["smoothed_history"], o["point"])
            rows_.append({"السوق": name, "الحالة": "كافية", "الحالي %": round(o["smoothed_history"][-1] * 100, 2),
                          "المتوقع %": round(o["point"][-1] * 100, 2), "الاتجاه": f"{lb} ({ch:+.0f}%)"})
        else:
            rows_.append({"السوق": name, "الحالة": f"غير كافية: {why_}"})
    fig.update_layout(title=f"«{by_id[sid]['name_ar']}»: اليمن مقابل العالمي{DEMO_TAG}", yaxis_title="% من إعلانات السوق",
                      height=470, hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame(rows_), hide_index=True, width="stretch")
    note("كل سوق تُبنى سلسلته من إعلاناته فقط ويُتنبأ بها منفصلة؛ لا يُخلط السوقان في سلسلة واحدة.")
    st.stop()

series, mentions, ok, why = get_series(scope)
if series is None:
    note(why, "warn")
    st.stop()
months = series.index.to_timestamp()
if not ok:
    note(f"⚠️ البيانات غير كافية لتنبؤ موثوق: {why} — لن يُعرض أي تنبؤ لهذه السلسلة.", "warn")
    fig = go.Figure(go.Scatter(x=months, y=series * 100, mode="lines+markers", name="فعلي (الفترات المرصودة)",
                               line=dict(color=BLUE), connectgaps=False))
    fig.update_layout(title="البيانات التاريخية المتاحة (الفجوات = فترات غير مرصودة)", yaxis_title="% من الإعلانات", height=360)
    st.plotly_chart(fig, width="stretch")
    st.stop()

out = fc_model.forecast(series.to_numpy(), horizon=horizon)
bt = fc_model.backtest(series.to_numpy())
label, change = trend_label(out["smoothed_history"], out["point"])
future = pd.period_range(series.index[-1] + 1, periods=horizon, freq=FREQ).to_timestamp()
fmt = lambda v, k=100, d=2: "—" if v is None else f"{v * k:.{d}f}"

kpis([
    (f"الطلب الحالي (متوسط 3 {UNIT})", f"{out['smoothed_history'][-1] * 100:.1f}%", "من إعلانات النطاق", BLUE),
    (f"المتوقع بعد {horizon} {UNIT}", f"{out['point'][-1] * 100:.1f}%",
     f"النطاق 80%: {out['low'][-1] * 100:.1f}% – {out['high'][-1] * 100:.1f}%", NAVY),
    ("الاتجاه المتوقع", label, f"{change:+.0f}% مقارنة بآخر 6 {UNIT}",
     TEAL if label == "متزايد" else RED if label == "متراجع" else GRAY),
    ("دقة النموذج على هذه السلسلة", f"MAE {fmt(bt['mae'])}",
     f"نقطة مئوية • {bt['n']} فترة اختبار مرصودة" + (f" • MAPE {bt['mape']:.0f}%" if bt.get("mape") is not None else ""), AMBER),
])

fig = go.Figure()
fig.add_scatter(x=months, y=series * 100, mode="markers", name="فعلي (الفترات المرصودة فقط)", marker=dict(color="#94A3B8", size=6))
fig.add_scatter(x=months, y=out["smoothed_history"] * 100, mode="lines", name="التاريخي (متوسط متحرك 3)", line=dict(color=BLUE, width=2.5))
bt_x = months[-LSTM_TEST_MONTHS:]
fig.add_scatter(x=bt_x, y=bt["pred"] * 100, mode="lines+markers", name="تنبؤ LSTM على فترة الاختبار",
                line=dict(color=AMBER, dash="dot", width=2))
fx = [months[-1]] + list(future)
fy = np.r_[out["smoothed_history"][-1], out["point"]] * 100
fig.add_scatter(x=list(future) + list(future[::-1]), y=np.r_[out["high"], out["low"][::-1]] * 100, fill="toself",
                fillcolor="rgba(20,184,166,.15)", line=dict(width=0), name="نطاق عدم اليقين 80% (MC Dropout)", hoverinfo="skip")
fig.add_scatter(x=fx, y=fy, mode="lines+markers", name="التوقع المستقبلي", line=dict(color=TEAL, width=3))
fig.add_vline(x=months[-1], line_dash="dash", line_color=GRAY)
fig.update_layout(title=f"الطلب على «{by_id[sid]['name_ar']}» — {scope}" + ("" if dom == "الكل" else f" • {domain_ar(dom)}") + DEMO_TAG,
                  yaxis_title="% من الإعلانات التي تطلب المهارة", height=470, hovermode="x unified")
st.plotly_chart(fig, width="stretch")
note("📌 التوقعات <b>تقديرية</b> مبنية على أنماط البيانات التاريخية المتاحة، وليست ضماناً لما سيحدث. "
     "الفترات التي فيها إعلانات قليلة جداً تُعامل كبيانات مفقودة لا كطلب صفري.")

c1, c2 = st.columns([1.1, 1])
with c1:
    st.markdown("#### تقييم النموذج على فترة الاختبار (لم تدخل التدريب ولا الإيقاف المبكر)")
    t = M["test"]
    names = {"lstm": "LSTM (نموذجنا)", "naive": "Naive (القيمة السابقة)", "moving_avg_3": f"متوسط متحرك 3 {UNIT}"}
    rows_ = []
    for mk, mt in [("الكل", t)] + [(("اليمن" if k == "Yemen" else "عالمي"), v) for k, v in M.get("by_market", {}).items()]:
        for k, nm in names.items():
            if k in mt:
                rows_.append({"السوق": mk, "النموذج": nm, "MAE": fmt(mt[k]["mae"], 100, 3), "RMSE": fmt(mt[k]["rmse"], 100, 3),
                              "MAPE %": "غير مناسب" if mt[k].get("mape") is None else f"{mt[k]['mape']:.1f}",
                              "نقاط": mt[k].get("n", "")})
    st.dataframe(pd.DataFrame(rows_), hide_index=True, width="stretch")
    gain = (1 - t["lstm"]["mae"] / t["naive"]["mae"]) * 100 if t["naive"]["mae"] else 0
    st.caption(f"MAE/RMSE بالنقاط المئوية. {M['n_series']} سلسلة • نافذة {M.get('window')} • تحسن MAE مقابل Naive: {gain:+.1f}%. "
               f"{M.get('mape_rule', '')}")
with c2:
    st.markdown("#### البيانات التاريخية والتوقعات")
    _fmt = "%Y-%m" if FREQ == "M" else "%Y-%m-%d"
    tbl = pd.DataFrame({"الفترة": list(months[-6:].strftime(_fmt)) + list(future.strftime(_fmt)),
                        "النوع": ["فعلي"] * min(6, len(months)) + ["متوقع" + (" (تجريبي)" if DEMO_TAG else "")] * horizon,
                        "القيمة %": list((series.to_numpy()[-6:] * 100).round(2)) + list((out["point"] * 100).round(2)),
                        "الحد الأدنى %": [None] * min(6, len(months)) + list((out["low"] * 100).round(2)),
                        "الحد الأعلى %": [None] * min(6, len(months)) + list((out["high"] * 100).round(2))})
    st.dataframe(tbl, hide_index=True, width="stretch", height=300)
    st.session_state["last_forecast"] = {"skill": by_id[sid]["name_ar"], "scope": scope, "domain": domain_ar(dom) if dom != "الكل" else "الكل",
                                         "horizon": horizon, "trend": label, "change": change, "table": tbl,
                                         "mae": bt["mae"], "demo": bool(DEMO_TAG)}

if LSTM_PREDICTIONS.exists():
    with st.expander("📊 الفعلي مقابل المتوقع على فترة الاختبار لكل السلاسل (من ملف التقييم)"):
        P = pd.read_csv(LSTM_PREDICTIONS)
        if len(P):
            P["السوق"] = P["market"].map({"Yemen": "اليمن", "Global": "عالمي"})
            fig = px.scatter(P, x=P["actual_smoothed"] * 100, y=P["lstm"] * 100, color="السوق",
                             labels={"x": "الفعلي %", "y": "تنبؤ LSTM %"}, opacity=.6, title="كل نقاط الاختبار")
            mx = float(max(P["actual_smoothed"].max(), P["lstm"].max()) * 100)
            fig.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx, line=dict(dash="dash", color=GRAY))
            st.plotly_chart(fig, width="stretch")
            st.caption("الخط المتقطع = تنبؤ مثالي. المصدر: artifacts/lstm_test_predictions.csv")

st.divider()
st.markdown("### 🏆 ترتيب المهارات حسب النمو المتوقع")
st.caption("يُشغّل النموذج على كل مهارة ذات بيانات كافية ضمن السوق/المجال المختار.")
if st.button("حساب الترتيب", type="primary"):
    sub = subset(scope, dom)
    allsh = monthly_share(sub, js, freq=FREQ)
    rows = []
    if not allsh.empty:
        cnt = allsh.attrs["counts"]
        with st.spinner("تشغيل LSTM على كل المهارات ..."):
            for s in allsh.columns:
                ok, _ = series_sufficiency(allsh[s], int(cnt[s].sum()), M.get("min_periods", 18), M.get("min_mentions", 25), FREQ)
                if not ok or s not in by_id:
                    continue
                o = fc_model.forecast(allsh[s].to_numpy(), horizon=horizon, mc_samples=20)
                lb, ch = trend_label(o["smoothed_history"], o["point"])
                rows.append({"المهارة": by_id[s]["name_ar"], "Skill": by_id[s]["name_en"], "الحالي %": o["smoothed_history"][-1] * 100,
                             "المتوقع %": o["point"].mean() * 100, "التغير %": ch, "الاتجاه": lb})
    if rows:
        r = pd.DataFrame(rows).sort_values("التغير %", ascending=False).round(2)
        st.session_state["ranking"] = r
        a, b = st.columns(2)
        a.markdown("**📈 الأعلى نمواً**" + DEMO_TAG)
        a.dataframe(r.head(10), hide_index=True, width="stretch")
        b.markdown("**📉 الأكثر تراجعاً**" + DEMO_TAG)
        b.dataframe(r.tail(10).iloc[::-1], hide_index=True, width="stretch")
    else:
        note("لا توجد مهارات ذات بيانات كافية في هذا النطاق.", "warn")
