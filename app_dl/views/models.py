import json
from pathlib import Path

import pandas as pd
import streamlit as st

from common import engine, setup

setup("معلومات النماذج", "الإعدادات، نتائج التحقق، ونتائج الاختبار المستقل مقارنة بخطوط الأساس")
e = engine()
stt = e.ads.status()
ev = e.evaluation_results()
st.markdown("#### حالة النماذج")
rows = []
for k, lab in [("opportunity_type", "تصنيف نوع الفرصة"), ("geographic_scope", "النطاق الجغرافي"), ("ner", "استخراج الكيانات NER")]:
    s = stt[k]
    tc = s.get("training_config", s.get("config", {}))
    rows.append({"النموذج": lab, "عائلة النموذج":s["model_family"], "متاح": "✅" if s["available"] else "❌", "النموذج الأساسي": tc.get("base_model", "—"),
                 "Epochs": tc.get("epochs_run", "—"), "أفضل Epoch": tc.get("best_epoch", "—"),
                 "تاريخ التدريب": tc.get("trained_at", "—"), "المسار": s["path"]})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.markdown("#### نتائج الاختبار (test) مقابل خطوط الأساس")
cmp = []
for task, lab in [("opportunity_type", "نوع الفرصة"), ("geographic_scope", "النطاق")]:
    d = ev.get(f"classification_{task}")
    if d:
        cmp.append({"المهمة": lab, "النموذج": "AraBERT", "Accuracy": d.get("accuracy"), "Macro-F1": d.get("macro_f1"),
                    "Weighted-F1": d.get("weighted_f1")})
    b = ev.get(f"baseline_{task}")
    for name in ("tfidf_logreg", "tfidf_linearsvc"):
        if b and name in b:
            t = b[name]["test"]
            cmp.append({"المهمة": lab, "النموذج": name, "Accuracy": t.get("accuracy"), "Macro-F1": t.get("macro_f1"),
                        "Weighted-F1": t.get("weighted_f1")})
# Real separately trained CPU neural checkpoints (NOT the AraBERT results).
for task, lab in [("opportunity_type", "نوع الفرصة"), ("geographic_scope", "النطاق")]:
    f = e.reports_dir / "evaluation" / f"offline_{task}.json"
    if f.exists():
        d = json.loads(f.read_text(encoding="utf8"))
        cmp.append({"المهمة":lab, "النموذج":"TinyClassifier CPU (Silver)", "Accuracy":d.get("accuracy"),
                    "Macro-F1":d.get("macro_f1"), "Weighted-F1":d.get("weighted_f1")})
# New geographically focused broad-scope classifier: separate from 8-way benchmarks.
g_path = e.reports_dir / "evaluation" / "geographic_scope_v2.json"
if g_path.exists():
    gd = json.loads(g_path.read_text(encoding="utf-8"))
    cmp.append({"المهمة": "النطاق (داخل/خارج/عن بُعد) — 3 فئات",
                "النموذج": "GeoScopeNetV2 PyTorch (Silver)",
                "Accuracy": gd.get("accuracy"), "Macro-F1": gd.get("macro_f1"),
                "Weighted-F1": gd.get("weighted_f1")})
f=e.reports_dir / "evaluation" / "offline_ner.json"
if f.exists():
    d=json.loads(f.read_text(encoding="utf8"))
    cmp.append({"المهمة":"NER", "النموذج":"TinyNER CPU (Silver)", "Entity-P":d.get("entity_precision"),
                "Entity-R":d.get("entity_recall"), "Entity-F1":d.get("entity_f1")})
n, bn = ev.get("ner"), ev.get("baseline_ner")
if n:
    cmp.append({"المهمة": "NER", "النموذج": "AraBERT", "Entity-P": n["entity_precision"], "Entity-R": n["entity_recall"],
                "Entity-F1": n["entity_f1"]})
if bn and "crf" in bn:
    t = bn["crf"]["test"]
    cmp.append({"المهمة": "NER", "النموذج": "CRF", "Entity-P": t["entity_precision"], "Entity-R": t["entity_recall"],
                "Entity-F1": t["entity_f1"]})
if cmp:
    st.dataframe(pd.DataFrame(cmp), hide_index=True, width="stretch")
else:
    st.warning("لا توجد نتائج تقييم محفوظة بعد في reports/evaluation. لا تُعرض أرقام قبل التقييم الفعلي.")

tabs = st.tabs(["نوع الفرصة", "النطاق", "NER", "LSTM"])
for tab, key in zip(tabs[:2], ["classification_opportunity_type", "classification_geographic_scope"]):
    with tab:
        d = ev.get(key)
        if not d:
            st.info("لم يُقيَّم بعد.")
            continue
        st.json({k: v for k, v in d.items() if k not in ("confusion_matrix", "report_text", "per_class")}, expanded=False)
        if d.get("per_class"):
            st.dataframe(pd.DataFrame(d["per_class"]), hide_index=True, width="stretch")
        img = e.reports_dir / "evaluation" / key / "confusion_matrix.png"
        if img.exists():
            st.image(str(img), caption="مصفوفة الالتباس (test)")
with tabs[2]:
    if n:
        st.dataframe(pd.DataFrame(n["per_entity"]), hide_index=True, width="stretch")
        st.json(n.get("errors", {}), expanded=False)
    else:
        st.info("لم يُقيَّم بعد.")
with tabs[3]:
    s = ev.get("lstm")
    if not s:
        st.info("لم يُشغَّل تدريب LSTM بعد.")
    else:
        st.dataframe(pd.DataFrame([{"العائلة": f, "دُرّب": v.get("trained"), "السبب": v.get("reason"),
                                    "LSTM MAE": (v.get("test") or {}).get("lstm", {}).get("mae"),
                                    "Naive MAE": (v.get("test") or {}).get("naive", {}).get("mae"),
                                    "MA3 MAE": (v.get("test") or {}).get("moving_avg_3", {}).get("mae")}
                                   for f, v in s.items() if isinstance(v, dict) and "trained" in v]),
                     hide_index=True, width="stretch")
st.caption("جودة الوسوم: بيانات التدريب والاختبار موسومة بقواعد (silver). القياس يعني الاتفاق مع هذه الوسوم؛ "
           "التقييم على Gold البشري يظهر فقط بعد توسيم ملفات gold_annotation.")
