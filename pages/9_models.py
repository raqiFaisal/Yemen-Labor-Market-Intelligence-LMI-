import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.state import (DOMAIN_SOURCE_AR, clf_metrics, clf_status, data_report, domain_ar, encoder_info,
                         lstm_metrics, lstm_status)
from utils.ui import AMBER, BLUE, GRAY, RED, TEAL, hero, kpis, note

EVAL_AR = {"gold": ("تصنيف يدوي مستقل (Gold)", "دقة حقيقية على تصنيف بشري لم يدخل التدريب."),
           "gold_mixed": ("خليط يدوي", "تصنيفات يدوية مختلطة بالتدريب لقلة البيانات — مؤشر تقريبي."),
           "source": ("تصنيف المصدر", "تقييم على تصنيفات المواقع نفسها."),
           "silver": ("تسميات ضعيفة (Silver)", "يقيس اتفاق النموذج مع قواعد التسمية الآلية، <b>وليس الدقة الحقيقية</b>. "
                      "للحصول على دقة حقيقية: <code>python -m scripts.label_jobs export</code> ثم صنّف 20 إعلاناً أو أكثر يدوياً.")}

hero("النماذج وتقييمها", "كيف يعمل كل نموذج من نماذج التعلم العميق، وما نتائجه الفعلية على بيانات الاختبار", "Deep Learning Models")
enc, cm, lm = encoder_info(), clf_metrics(), lstm_metrics()
from core.config import IS_DEMO_MODE  # noqa: E402
if IS_DEMO_MODE:
    note("🧪 <b>وضع العرض التجريبي:</b> كل النماذج والمقاييس في هذه الصفحة ناتجة عن بيانات اصطناعية منفصلة — "
         "توضح سلامة خط التدريب فقط ولا تُعد نتائج أداء حقيقية.", "warn")

t1, t2, t3, t4 = st.tabs(["① Transformer", "② مصنف FNN", "③ LSTM", "④ خط المعالجة"])

with t1:
    st.markdown("### Sentence Transformer متعدد اللغات")
    if not enc:
        note("لم يُهيَّأ المُرمِّز بعد.", "warn")
    else:
        if enc["kind"] == "transformer":
            kpis([("النموذج", "MiniLM-L12 متعدد اللغات"), ("أبعاد التمثيل", enc["dim"]), ("الجهاز", enc["device"])])
        else:
            note(f"⚠️ يعمل حالياً <b>المُرمِّز البديل</b> (TF-IDF حروف + SVD، {enc['dim']} بُعداً) لأن تحميل الـ Transformer تعذّر: "
                 f"<code>{enc.get('load_error')}</code><br>على Google Colab أو أي جهاز متصل بالإنترنت سيُحمَّل الـ Transformer تلقائياً "
                 "عند إعادة تشغيل <code>scripts.train_all</code>.", "warn")
    st.markdown("""
**النموذج:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` — مدرب مسبقاً على أزواج جمل بأكثر من 50 لغة منها العربية.

| المرحلة | ما يحدث |
|---|---|
| Tokenization | تقسيم النص إلى وحدات فرعية (SentencePiece) مشتركة بين العربية والإنجليزية |
| Embeddings | كل وحدة → متجه 384 بُعداً + ترميز الموضع |
| Self-Attention × 12 | كل كلمة "تنتبه" لكل الكلمات الأخرى لفهم السياق (مثلاً "تحليل" بجوار "بيانات" ≠ "تحليل مالي") |
| Mean Pooling | متوسط تمثيلات الكلمات → تمثيل واحد للجملة |
| Normalization | تطبيع L2 لتصبح المقارنة بـ Cosine Similarity = ضرب نقطي |

**ما الذي لا نفعله (للدقة العلمية):** الـ Transformer **لم يُدرَّب ولم يُضبط (no fine-tuning)** على بياناتنا؛ يُستخدم مجمَّداً
لإنتاج Embeddings فقط. والاستخراج القاموسي للمهارات **قاعدي وليس Deep Learning**؛ المكوّن العصبي في الاستخراج هو المطابقة الدلالية
بالـ Embeddings، ودرجتها Cosine Similarity وليست احتمالاً معايَراً. الـ Embeddings تُخزَّن تزايدياً (`artifacts/embedding_store/`)
فلا يُعاد ترميز إلا الإعلانات الجديدة.

**أين يُستخدم:** (1) مدخلات مصنف المجالات، (2) المطابقة الدلالية لاستخراج المهارات، (3) البحث الدلالي عن الإعلانات المشابهة،
(4) مطابقة مهارات الباحث بالمتطلبات في تحليل الفجوة. الـ Embeddings تُحسب على دفعات وتُخزَّن (`artifacts/job_embeddings.npy`) لتسريع الواجهة.
""")

with t2:
    st.markdown("### شبكة Feed-Forward لتصنيف مجال الوظيفة (PyTorch)")
    if not cm:
        cs, rep = clf_status(), (data_report() or {})
        note(f"❌ <b>المصنف غير مدرب على البيانات الحالية.</b> السبب: {cs.get('reason', '—')}", "warn")
        d = cs.get("details") or rep.get("classifier") or {}
        if d:
            kpis([("سجلات معنونة صالحة", d.get("n_labeled", 0), f"المطلوب {d.get('min_labeled', '—')}", RED),
                  ("مجالات مؤهلة", len(d.get("classes", {})), f"≥ {d.get('min_per_class', '—')} سجلات لكل مجال", AMBER),
                  ("مجالات مستبعدة لقلتها", len(d.get("dropped_classes", {})), "", GRAY)])
        if d.get("per_class"):
            st.markdown("**العينات المعنونة الموثوقة لكل مجال وما ينقصها**")
            st.dataframe(pd.DataFrame([{"المجال": domain_ar(k), "المتوفر": v["have"],
                                        f"ينقص للحد الأدنى ({d.get('min_per_class')})": v["missing_for_min"],
                                        f"ينقص للموصى به ({d.get('recommended_per_class', 30)})": v["missing_for_recommended"]}
                                       for k, v in d["per_class"].items()]), hide_index=True, width="stretch")
        if rep.get("labels"):
            st.markdown("**مصادر التسمية في البيانات الحالية**")
            st.dataframe(pd.DataFrame([{"المصدر": DOMAIN_SOURCE_AR.get(k, k), "عدد السجلات": v}
                                       for k, v in rep["labels"].items()]), hide_index=True)
        note("الحلول: (1) جمع إعلانات أكثر — تصنيفات YemenHR تُستخدم تلقائياً بعد توحيدها عبر "
             "<code>data/taxonomy/category_map.json</code>؛ (2) مراجعة يدوية من صفحة «مراجعة التصنيفات» أو "
             "<code>python -m scripts.label_jobs export</code>. التسميات الآلية الضعيفة لا تُستخدم كتسميات تدريب افتراضياً.")
    else:
        t = cm["test"]
        et, ed = EVAL_AR.get(cm.get("eval_type", "source"), ("—", ""))
        note(f"<b>نوع التقييم:</b> {et} — {ed}", "warn" if cm.get("eval_type") == "silver" else "note")
        if cm.get("trusted_test"):
            tt = cm["trusted_test"]
            st.caption(f"على السجلات ذات التصنيف الموثوق فقط ضمن الاختبار ({tt['n']}): Accuracy {tt['accuracy']:.3f} • F1 {tt['f1_macro']:.3f}")
        if cm.get("label_sources_train"):
            st.caption("مصادر تسميات التدريب: " + " • ".join(f"{DOMAIN_SOURCE_AR.get(k, k)}: {v}"
                                                          for k, v in cm["label_sources_train"].items()))
        if cm.get("encoder", {}).get("kind") != "transformer":
            note("⚠️ هذا المصنف درب على تمثيلات المُرمِّز البديل TF-IDF (ليس Deep Learning، ومُلاءَم على كل النصوص). "
                 "أعد التدريب على Colab مع الـ Transformer قبل اعتماد أي رقم.", "warn")
        if cm.get("reused"):
            note(f"↺ <b>نموذج من تدريب سابق:</b> {cm['reused']['note']}. لم يُعد التدريب في آخر تشغيل لأن: "
                 f"{cm['reused']['why']}", "warn")
        bm, sp = cm.get("baseline_majority"), cm.get("split", {})
        if bm:
            st.caption(f"خط الأساس (التنبؤ دائماً بالفئة الأكثر «{domain_ar(bm['class'])}»): Accuracy {bm['accuracy']:.3f} • "
                       f"F1 {bm['f1_macro']:.3f} — أي دقة يجب مقارنتها بهذا الرقم.")
        if sp:
            st.caption(f"التقسيم: {sp.get('method')} • تداخل مجموعات التكرار بين التدريب والاختبار: {sp.get('group_overlap_train_test')} "
                       f"• مجموعات مكررة في البيانات: {sp.get('duplicate_groups')} • الفئات في الاختبار: {sp.get('test_class_counts')}")
        if cm.get("n_test", 0) < 50:
            note(f"⚠️ بيانات الاختبار {cm.get('n_test')} سجلاً فقط — المقاييس ذات تباين عالٍ ولا تصلح لاستنتاجات قاطعة.", "warn")
        if cm.get("trained_on_demo_data"):
            note("مدرب على بيانات تجريبية — الدقة المرتفعة هنا متوقعة لأن البيانات اصطناعية، ولا تعبّر عن الأداء على إعلانات حقيقية.", "warn")
        kpis([("Accuracy", f"{t['accuracy']:.3f}", "بيانات اختبار مستقلة"), ("Precision (macro)", f"{t['precision_macro']:.3f}", "", BLUE),
              ("Recall (macro)", f"{t['recall_macro']:.3f}", "", TEAL), ("F1 (macro)", f"{t['f1_macro']:.3f}", "", AMBER)])
        st.caption(f"تدريب {cm['n_train']} • تحقق {cm['n_val']} • اختبار {cm['n_test']} • {cm['epochs_run']} دورة • "
                   f"{cm['params']:,} معامل • المُرمِّز: {cm['encoder']['name']} • {cm['trained_at']}")
        a, b = st.columns(2)
        h = pd.DataFrame(cm["history"])
        fig = go.Figure()
        fig.add_scatter(x=h.epoch, y=h.train_loss, name="Train Loss", line=dict(color=BLUE))
        fig.add_scatter(x=h.epoch, y=h.val_loss, name="Val Loss", line=dict(color=AMBER))
        fig.add_scatter(x=h.epoch, y=h.val_f1, name="Val F1", yaxis="y2", line=dict(color=TEAL, dash="dot"))
        fig.update_layout(title="منحنيات التدريب", yaxis2=dict(overlaying="y", side="right", range=[0, 1]), height=380)
        a.plotly_chart(fig, width="stretch")
        labs = [domain_ar(l) for l in cm["labels"]]
        fig = px.imshow(cm["confusion_matrix"], x=labs, y=labs, text_auto=True, color_continuous_scale="Blues",
                        labels=dict(x="المتوقع", y="الفعلي"), title="مصفوفة الالتباس (بيانات الاختبار)")
        fig.update_layout(height=380)
        b.plotly_chart(fig, width="stretch")
        rep = pd.DataFrame(cm["report"]).T.drop(index=["accuracy"], errors="ignore")
        rep.index = [domain_ar(i) if i in cm["labels"] else i for i in rep.index]
        st.dataframe(rep.round(3), width="stretch")
        with st.expander("المعمارية"):
            st.code(cm["architecture"])
    st.markdown("""
**الفكرة:** نجمّد الـ Transformer (Transfer Learning) ونستخدم تمثيلاته كمدخلات لشبكة صغيرة تتعلم حدود المجالات.
- `Linear → ReLU → Dropout(0.3)` × 2 ثم `Linear` إلى عدد المجالات، و **Softmax** لتحويل المخرجات لاحتمالات.
- الخسارة **Cross-Entropy** موزونة بعكس تكرار الفئة لمعالجة عدم التوازن؛ المُحسِّن **AdamW**؛ **Early Stopping** على F1 للتحقق.
- تقسيم طبقي 70/15/15 **يحترم مجموعات التكرار**: الإعلان المعاد نشره بنفس النص لا يظهر في التدريب والاختبار معاً.
- التسميات: يدوية (gold) أو تصنيف الموقع بعد توحيد صريح؛ عند توفر 20+ تسمية يدوية تُستخدم وحدها كاختبار مستقل.
- الإعلانات بلا تصنيف في المصدر تُصنَّف بالنموذج، ويظهر في الواجهة إن كان التصنيف "من المصدر" أو "تنبؤ النموذج".
""")

with t3:
    st.markdown("### LSTM للتنبؤ بالطلب على المهارات")
    if not lm or "test" not in lm:
        ls_ = lstm_status()
        note(f"❌ <b>LSTM غير مدرب على البيانات الحالية.</b> السبب: {ls_.get('reason', '—')}", "warn")
        det = ls_.get("details") or {}
        for mk, d in (det.get("by_market") or ({"—": det} if det.get("n_jobs") is not None else {})).items():
            st.markdown(f"**{'اليمن' if mk == 'Yemen' else 'عالمي/عربي' if mk == 'Global' else mk}**")
            kpis([("إعلانات مؤرخة", f"{d.get('n_dated', 0)} / {d.get('n_jobs', 0)}", f"{d.get('dated_pct', 0):.0f}%"),
                  ("المدى الزمني", f"{d.get('span_periods', 0)} {d.get('unit', '')}", f"{d.get('first')} → {d.get('last')}", AMBER),
                  ("المطلوب", f"{d.get('min_periods')} {d.get('unit', '')}", "طول السلسلة", RED),
                  ("فترات مرصودة", d.get("observed_periods", 0), f"≥{d.get('min_postings_per_period', 10)} إعلانات", GRAY)])
        note("لا يُعرض أي تنبؤ LSTM على أنه حقيقي قبل توفر سلاسل كافية. وضع العرض التجريبي المنفصل متاح للتوضيح فقط.")
    else:
        t = lm["test"]
        gain = (1 - t["lstm"]["mae"] / t["naive"]["mae"]) * 100
        kpis([("MAE — LSTM", f"{t['lstm']['mae'] * 100:.3f}", "نقطة مئوية", BLUE),
              ("MAE — Naive", f"{t['naive']['mae'] * 100:.3f}", "خط أساس", "#94A3B8"),
              ("التحسن مقابل Naive", f"{gain:+.1f}%", "", TEAL if gain > 0 else "#EF4444"),
              ("السلاسل", lm["n_series"], f"{lm['n_windows']} نافذة تدريب", AMBER)])
        if lm.get("trained_on_demo_data"):
            note("مدرب على اتجاهات اصطناعية مبرمجة في البيانات التجريبية — يثبت أن النموذج يلتقط الاتجاهات، لا أنه يتنبأ بالسوق الحقيقي.", "warn")
        st.caption(f"{lm.get('split', '')} • نافذة {lm.get('window')} • أفضل دورة {lm.get('best_epoch')} • {lm.get('scaling', '')} • "
                   f"{lm.get('target', '')}")
        if lm.get("by_market"):
            rows_ = []
            for mk, mt in lm["by_market"].items():
                for k, nm in (("lstm", "LSTM"), ("naive", "Naive"), ("moving_avg_3", "MA-3")):
                    rows_.append({"السوق": "اليمن" if mk == "Yemen" else "عالمي", "النموذج": nm,
                                  "MAE (نقطة %)": round(mt[k]["mae"] * 100, 3), "RMSE": round(mt[k]["rmse"] * 100, 3),
                                  "MAPE %": "غير مناسب" if mt[k].get("mape") is None else round(mt[k]["mape"], 1),
                                  "نقاط الاختبار": mt[k]["n"], "السلاسل": mt.get("n_series")})
            st.dataframe(pd.DataFrame(rows_), hide_index=True, width="stretch")
        a, b = st.columns(2)
        h = pd.DataFrame(lm["history"])
        a.plotly_chart(px.line(h, x="epoch", y=[c for c in ("train_mse", "val_mse") if c in h], title="خسارة التدريب والتحقق (MSE)",
                               markers=True), width="stretch")
        ev = pd.DataFrame([{"النموذج": {"lstm": "LSTM", "naive": "Naive", "moving_avg_3": "MA-3"}[k], "MAE": v["mae"] * 100,
                            "RMSE": v["rmse"] * 100} for k, v in t.items()]).melt("النموذج")
        b.plotly_chart(px.bar(ev, x="النموذج", y="value", color="variable", barmode="group",
                              title="المقارنة مع خطوط الأساس (أقل = أفضل)", labels={"value": "نقطة مئوية"}),
                       width="stretch")
        with st.expander("المعمارية"):
            st.code(lm["architecture"])
    st.markdown("""
**المتغير المتنبأ به:** نسبة إعلانات الفترة التي تطلب المهارة (وليس العدد الخام) — حتى لا يُفسَّر نمو عدد الإعلانات الكلي كنمو في المهارة.

**خلية LSTM:** تحتفظ بحالة ذاكرة (Cell State) تتحكم بها ثلاث بوابات:
- **Forget Gate** تقرر ما يُنسى من الاتجاه القديم، **Input Gate** ما يُضاف من الشهر الجديد، **Output Gate** ما يُمرر للتنبؤ.

**غياب البيانات ≠ صفر:** الفترة التي فيها أقل من 10 إعلانات تُعد غير مرصودة (NaN)، تُستوفى لمدخلات النموذج فقط ولا تكون هدفاً للتدريب أو التقييم.

**التدريب:** نموذج واحد على كل السلاسل (سلاسل اليمن والعالمي تُبنى منفصلة)، كل سلسلة تُطبَّع بمتوسط **جزء تدريبها فقط**،
نافذة تُختار تلقائياً (3–6) حسب طول السلاسل، خسارة **MSE**، **Adam**، إيقاف مبكر على فترة تحقق زمنية.
**التقسيم الزمني:** [تدريب | تحقق 3 فترات | اختبار 6 فترات] لكل سلسلة — لا تسرب من المستقبل.
**التقييم:** MAE و RMSE و MAPE (حين تكون مناسبة) مقابل Naive والمتوسط المتحرك — إن لم يتفوق LSTM عليهما يظهر ذلك هنا.
**عدم اليقين:** Monte-Carlo Dropout ينتج نطاق 10%–90% بدل رقم واحد.
**شرط الكفاية:** 18 فترة على الأقل، 80% منها مرصودة، و 25 إعلاناً يذكر المهارة؛ وإلا يرفض التطبيق التنبؤ ويوضح السبب.
""")

with t4:
    rep = data_report()
    if rep:
        st.markdown(f"#### تشخيص جاهزية البيانات — آخر تدريب {rep.get('generated_at', '')}")
        dd, sk, tsr = rep.get("data", {}), rep.get("skills", {}), rep.get("time_series", {})
        cl = rep.get("classifier", {})
        kpis([("الإعلانات", dd.get("n_jobs", 0), "تجريبية ⚠️" if dd.get("is_demo") else "حقيقية"),
              ("علاقات المهارات", sk.get("relations", 0), f"{sk.get('distinct', 0)} مهارة مميزة", TEAL),
              ("مصنف FNN", "مدرب ✅" if cl.get("trained") else ("مُعاد من تدريب سابق ↺" if cl.get("reused_previous") else "غير مدرب ❌"),
               cl.get("reason") or cl.get("eval_type", ""), TEAL if cl.get("trained") else RED),
              ("LSTM", "مدرب ✅" if tsr.get("trained") else "غير مدرب ❌",
               f"{tsr.get('valid_series', 0)} سلسلة صالحة", TEAL if tsr.get("trained") else RED),
              ("الاستخراج", "قاموسي + دلالي" if sk.get("semantic_enabled") else "قاموسي فقط",
               " • ".join(f"{k}: {v}" for k, v in (sk.get("by_method") or {}).items()), BLUE)])
        if rep.get("provenance"):
            st.markdown("**مصدر البيانات وتاريخ جمعها**")
            st.dataframe(pd.DataFrame([{"المصدر": k, "السجلات": v["rows"], "السوق": v["market"],
                                        "جُمعت من": v["collected_from"], "إلى": v["collected_to"],
                                        "تواريخ النشر": f"{v['posted_from']} → {v['posted_to']}", "مؤرخ %": v["dated_pct"],
                                        "الملفات": ", ".join(v["files"])} for k, v in rep["provenance"].items()]),
                         hide_index=True, width="stretch")
        cats = (rep.get("categories") or {}).get("by_raw_category")
        if cats:
            st.markdown("**توحيد تصنيفات المصدر** (عدّل `data/taxonomy/category_map.json` للغامض وغير المطابق)")
            st.dataframe(pd.DataFrame([{"تصنيف المصدر": k, "العدد": v["count"], "المجال": domain_ar(v["domain"]) if v["domain"] else "—",
                                        "الحالة": {"mapped": "مطابق", "ambiguous": "غامض (يحتاج مراجعة)", "unmapped": "غير معروف",
                                                   "conflict": "متعارض"}.get(v["status"], v["status"])}
                                       for k, v in cats.items()]), hide_index=True, width="stretch")
        with st.expander("التقرير الكامل (data_report.json)"):
            st.json(rep)
    st.markdown("""
```
data/raw/*.csv  (Yemen HR / ShortList / Board • ArabJobs • ملفات عالمية)
      │  preprocessing/schema.py — مطابقة الأعمدة، تنظيف HTML/الروابط، توحيد العربية، توحيد المحافظات، إزالة التكرار
      ▼
jobs_unified  ──►  استخراج المهارات القاموسي (+ الدلالي)  ──►  job_skills
      │
      ├─► Transformer Embeddings (مخزنة) ──► FNN ──► domain_pred / domain_final
      │
      └─► سلاسل شهرية (نسبة الإعلانات لكل مهارة × سوق) ──► LSTM ──► توقعات + نطاق عدم يقين
```
**مبدأ الفصل:** كل سجل يحمل `country` و `source` و `is_demo`؛ المؤشرات اليمنية تُحسب من السجلات اليمنية فقط، والبيانات التجريبية لا تختلط بالحقيقية.
""")
