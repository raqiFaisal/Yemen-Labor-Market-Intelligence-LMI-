"""مراجعة التصنيفات يدوياً — تُحفظ في data/labels/gold_labels.csv وتُستخدم في التدريب والاختبار المستقل."""
from datetime import datetime

import pandas as pd
import streamlit as st

from core.config import GOLD_LABELS_FILE, IS_DEMO_MODE
from core.pipeline import load_skill_decisions
from preprocessing.labeling import load_gold, review_queue
from preprocessing.skill_review import load_reviews, save_reviews
from utils.state import taxonomy
from utils.state import demo_banner, domain_ar, get_domains, require_data
from utils.ui import hero, kpis, note

hero("مراجعة التصنيفات", "صنّف الإعلانات غير المؤكدة وراجع المهارات منخفضة الثقة لبناء بيانات موثوقة", "Label Review")
jobs, js = require_data()
demo_banner(jobs)
if IS_DEMO_MODE:
    note("وضع العرض التجريبي: التصنيفات هنا تُحفظ في data/labels_demo ولا تؤثر على البيانات الحقيقية.", "warn")

doms = get_domains()
tab_dom, tab_sk = st.tabs(["🏷️ مجالات الإعلانات", "🧠 المهارات منخفضة الثقة"])

with tab_dom:
    gold = load_gold()
    q = review_queue(jobs)
    q = q[~q.job_id.isin(gold.job_id)]
    pri_ar = {0: "تصنيف المصدر غامض/غير مطابق", 1: "تعارض الاقتراح مع النموذج", 2: "ثقة منخفضة", 3: "بلا تسمية"}
    kpis([("تصنيفات يدوية محفوظة", len(gold), "gold"), ("بانتظار المراجعة", len(q), ""),
          ("تسميات موثوقة حالياً", int(jobs["label_source"].isin(["gold", "source"]).sum()), "gold + source")])
    note("الاقتراح الآلي (من المسمى/المهارات) <b>ليس تسمية</b>؛ اختر المجال الصحيح بنفسك، أو اترك الخانة فارغة إن لم تكن متأكداً. "
         "بعد الحفظ أعد التدريب: <code>python -m scripts.train_all</code>.")
    if q.empty:
        st.success("لا توجد إعلانات بانتظار المراجعة.")
    else:
        n = st.slider("عدد الإعلانات في هذه الدفعة", 1, min(100, len(q)), min(20, len(q)))
        batch = q.head(n)
        opts = [""] + list(doms)
        view = pd.DataFrame({
            "job_id": batch.job_id, "المسمى": batch.title, "الجهة": batch.company,
            "الوصف (مقتطف)": batch.description.fillna("").str.slice(0, 220),
            "تصنيف المصدر": batch.get("category_raw", pd.Series("", index=batch.index)).fillna(""),
            "سبب المراجعة": batch.review_priority.map(pri_ar),
            "اقتراح آلي": batch.suggested_label.map(lambda d: domain_ar(d) if isinstance(d, str) else ""),
            "تنبؤ النموذج": batch.domain_pred.map(lambda d: domain_ar(d) if isinstance(d, str) else ""),
            "المجال الصحيح": "",
        })
        ed = st.data_editor(view, hide_index=True, width="stretch", disabled=[c for c in view.columns if c != "المجال الصحيح"],
                            column_config={"المجال الصحيح": st.column_config.SelectboxColumn(
                                options=opts, help="اتركه فارغاً إن لم تكن متأكداً")}, key="review_editor")
        if st.button("💾 حفظ التصنيفات", type="primary"):
            done = ed[ed["المجال الصحيح"].isin(doms)]
            if done.empty:
                st.info("لم يُختر أي مجال.")
            else:
                new = pd.DataFrame({"job_id": done.job_id, "domain": done["المجال الصحيح"],
                                    "labeled_at": datetime.now().isoformat(timespec="seconds"), "labeled_by": "app_review"})
                old = pd.read_csv(GOLD_LABELS_FILE, dtype=str) if GOLD_LABELS_FILE.exists() else pd.DataFrame()
                pd.concat([old, new]).drop_duplicates("job_id", keep="last").to_csv(GOLD_LABELS_FILE, index=False,
                                                                                  encoding="utf-8-sig")
                st.success(f"حُفظ {len(new)} تصنيفاً في {GOLD_LABELS_FILE.name}. أعد تشغيل التدريب لاستخدامها.")
    with st.expander("التصنيفات اليدوية المحفوظة"):
        st.dataframe(gold.assign(المجال=gold.domain.map(domain_ar)), hide_index=True, width="stretch")

with tab_sk:
    dec = load_skill_decisions()
    if dec is None or dec.empty:
        note("لا توجد قرارات مهارات بعد — شغّل <code>python -m scripts.train_all</code>.", "warn")
    else:
        _, by_id = taxonomy()
        rv = load_reviews()
        done_keys = set(zip(rv.job_id, rv.skill_id))
        thr = st.slider("اعرض المقبولة التي ثقتها أقل من", 0.5, 0.95, 0.80, 0.05)
        pool = dec[(dec.decision == "review") | ((dec.decision == "accepted") & (dec.confidence < thr))]
        pool = pool[[(j, s) not in done_keys for j, s in zip(pool.job_id, pool.skill_id)]]
        kpis([("بانتظار المراجعة (review)", int((dec.decision == "review").sum()), "لم تُقبل تلقائياً"),
              (f"مقبولة بثقة < {thr}", int(((dec.decision == "accepted") & (dec.confidence < thr)).sum()), ""),
              ("مراجعات محفوظة", len(rv), "تُستخدم في التقرير والتدريب")])
        note("قرّر لكل مهارة: هل يطلب الإعلان هذه المهارة فعلاً بناءً على الدليل؟ «صحيحة» تقبلها، «خاطئة» تحذفها. "
             "المراجعات تُطبَّق في التشغيل التالي لـ <code>train_all</code> وتظهر في تقرير الجودة "
             "(<code>python -m scripts.inspect_skills eval</code>).")
        if pool.empty:
            st.success("لا توجد مهارات بانتظار المراجعة ضمن هذا الحد.")
        else:
            titles = jobs.set_index("job_id")["title"]
            b = pool.sort_values(["decision", "confidence"], ascending=[False, True]).head(60)
            view = pd.DataFrame({"job_id": b.job_id, "skill_id": b.skill_id, "الإعلان": b.job_id.map(titles),
                                 "المهارة": b.skill_id.map(lambda s: f"{by_id[s]['name_ar']} ({by_id[s]['name_en']})"),
                                 "الطريقة": b.method, "القرار الآلي": b.decision, "الثقة": b.confidence,
                                 "الدليل من النص": b.evidence, "السبب": b.reason, "حكمك": ""})
            ed2 = st.data_editor(view, hide_index=True, width="stretch",
                                 disabled=[c for c in view.columns if c != "حكمك"],
                                 column_config={"حكمك": st.column_config.SelectboxColumn(options=["", "صحيحة", "خاطئة"])},
                                 key="skill_review_editor")
            if st.button("💾 حفظ مراجعات المهارات", type="primary"):
                d = ed2[ed2["حكمك"].isin(["صحيحة", "خاطئة"])]
                if d.empty:
                    st.info("لم تُحدَّد أي مراجعة.")
                else:
                    save_reviews([{"job_id": r.job_id, "skill_id": r.skill_id,
                                   "verdict": "correct" if r.حكمك == "صحيحة" else "incorrect"} for r in d.itertuples()])
                    st.success(f"حُفظت {len(d)} مراجعة.")
