import html

import pandas as pd
import streamlit as st

from common import ENT_COLORS, ar_scope, ar_type, engine, setup
from src.integration.engine import ENTITY_AR

setup("تحليل إعلان بالتعلم العميق")
an = engine().ads
stt = an.status()
cols = st.columns(3)
for c, (k, lab) in zip(cols, [("opportunity_type", "تصنيف نوع الفرصة"), ("geographic_scope", "تصنيف النطاق"), ("ner", "استخراج الكيانات")]):
    c.markdown(f"{'🟢' if stt[k]['available'] else '🔴'} **{lab}**  \n<span class='note'>{'محمّل من' if stt[k]['available'] else 'غير موجود في'} "
               f"<code>{stt[k]['path']}</code><br><small>{stt[k]['model_family']}</small></span>", unsafe_allow_html=True)
if not any(v["available"] for v in stt.values()):
    st.error("لا يوجد أي نموذج مدرب. درّب النماذج (دفاتر Colab في notebooks/) ثم انسخ مجلد models_dl إلى المشروع.")
    st.stop()
text = st.text_area("نص الإعلان", height=220, placeholder="الصق نص إعلان وظيفة كما نُشر…")
if st.button("تحليل", type="primary"):
    if not text.strip():
        st.warning("الصق نص الإعلان أولاً.")
        st.stop()
    try:
        with st.spinner("تشغيل النماذج…"):
            r = an.analyze(text)
    except ValueError as e:
        st.warning(str(e))
        st.stop()
    comp = r["components"]
    c1, c2 = st.columns(2)
    for c, key, fmt, title in [(c1, "opportunity_type", ar_type, "نوع الفرصة"), (c2, "geographic_scope", ar_scope, "النطاق الجغرافي")]:
        with c:
            st.markdown(f"#### {title}")
            v = comp[key]
            if v["status"] != "ok":
                st.error("النموذج غير متاح")
                continue
            st.metric("التنبؤ", fmt(v["label"]))
            if v.get("confidence") is not None:
                if key == "geographic_scope" and v.get("decision_source") == "trained_neural_model":
                    import json
                    from src.common import REPORTS_DIR
                    metric_cols = st.columns(2)
                    metric_cols[0].metric("ثقة التنبؤ", f"{v['confidence']:.1%}")
                    score_file = REPORTS_DIR / "evaluation" / "geographic_scope_v2.json"
                    if score_file.is_file():
                        scores = json.loads(score_file.read_text(encoding="utf-8"))
                        metric_cols[1].metric("دقة الاختبار (Silver)", f"{scores['accuracy']:.1%}")
                else:
                    st.metric("ثقة النموذج", f"{v['confidence']:.1%}")
            for t in v["top"]:
                st.progress(float(t["probability"]), text=f"{fmt(t['label'])} — {t['probability']:.1%}")
    st.markdown("#### الكيانات المستخرجة")
    n = comp["ner"]
    if n["status"] != "ok":
        st.error("نموذج NER غير متاح")
    elif not n["entities"]:
        st.info("لم يستخرج النموذج أي كيان من هذا النص.")
    else:
        toks = [m.group(0) for m in __import__("src.inference.predictors", fromlist=["TOKEN_RX"]).TOKEN_RX.finditer(text)]
        spans = {e["start"]: e for e in n["entities"]}
        out, i = [], 0
        while i < len(toks):
            if i in spans:
                e = spans[i]
                col = ENT_COLORS.get(e["label"], "#475569")
                out.append(f"<mark class='ent' style='background:{col}'>{html.escape(e['text'])}"
                           f"<small>{ENTITY_AR.get(e['label'], e['label'])}</small></mark>")
                i = e["end"]
            else:
                out.append(html.escape(toks[i]))
                i += 1
        st.markdown(f"<div class='adbox'>{' '.join(out)}</div>", unsafe_allow_html=True)
        rows = [{"النوع": ENTITY_AR.get(k, k), "القيمة": x["text"], "الثقة": round(x["confidence"], 3)}
                for k, lst in n["grouped"].items() for x in lst]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with st.expander("المخرجات المهيكلة (JSON)"):
        st.json({k: {kk: vv for kk, vv in v.items() if kk not in ("entities",)} for k, v in comp.items()})
