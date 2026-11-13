"""اختبارات انحدار لاستخراج المهارات — عربية وإنجليزية، إيجابية وسلبية، تشمل الأخطاء التي ظهرت في تشغيل Colab.

    python -m tests.test_skill_extraction

مهم: هذه اختبارات صحة للقواعد (Regression tests) بجمل كُتبت للاختبار، وليست قياساً لدقة النظام على بيانات حقيقية.
نسبة النجاح هنا لا تُقدَّم كـ Precision/Recall — القياس العلمي يتطلب تعليماً بشرياً لإعلانات حقيقية
(scripts/inspect_skills.py export/eval).
"""
import hashlib
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

import models.embedding_store as es
from models.skill_extractor import SkillExtractor
from preprocessing.text import clean_for_model

# (النص، مهارات يجب قبولها، مهارات يجب ألا تُقبل، وصف)
CASES = [
    # ---- أخطاء ظهرت فعلياً في تشغيل Colab (كلها يجب ألا تُقبل) ----
    ("Surgeon – Daily Worker. Perform surgical operations and ensure adequate rest periods for patients.",
     [], ["api"], "Colab: REST APIs في إعلان جراح"),
    ("The project aims to improve learning outcomes for children in rural schools.",
     [], ["machine_learning"], "Colab: Machine Learning من Learning Outcomes"),
    ("Position: Admin Officer. Admin and finance support to the field office.",
     [], ["leadership"], "Colab: Leadership من Admin"),
    ("مكان العمل: مأرب", [], ["recruitment"], "Colab: Recruitment من مكان العمل"),
    ("الالتزام بإجراءات السلامة ونظم الأمان داخل المستشفى.", [], ["cybersecurity"], "Colab: Cybersecurity من ونظم الأمان"),
    ("متابعة الإيرادات وحساب الأرباح الشهرية للشركة.", [], ["payroll"], "Colab: Payroll من الأرباح"),
    # ---- تعدد المعاني والسياق ----
    ("إعلان توظيف: مطلوب محاسب للعمل في عدن.", [], ["recruitment"], "التوظيف كإعلان وظيفة لا كمهارة"),
    ("Food security and livelihoods programme in Hodeidah.", [], ["cybersecurity"], "Food security ليست أمناً سيبرانياً"),
    ("رخصة قيادة سارية وخبرة في قيادة السيارات.", [], ["leadership"], "قيادة السيارات ليست مهارة قيادة"),
    ("Reporting to the Programme Manager.", [], ["report_writing"], "Reporting to = خط إداري"),
    ("Coordinate cash assistance distribution to beneficiaries.", [], ["cash_management"], "Cash assistance ليست إدارة نقدية"),
    ("Wash hands before every procedure.", [], ["wash"], "wash فعل لا قطاع WASH"),
    ("يفضل أن يكون خريج علوم الحاسوب.", [], ["ms_office"], "علوم الحاسوب تخصص لا مهارة أوفيس"),
    ("Administer 5 ml of medication as prescribed.", [], ["machine_learning"], "ml وحدة قياس"),
    ("Able to excel in a fast-paced environment.", [], ["excel"], "excel فعل"),
    ("The training will be provided by the organization.", [], ["training_dev"], "تدريب يُقدَّم للموظف"),
    # ---- النفي ----
    ("Experience with Excel is not required.", [], ["excel"], "نفي إنجليزي"),
    ("لا يشترط إجادة اللغة الإنجليزية.", [], ["english"], "نفي عربي"),
    # ---- أسطر التقديم ----
    ("How to apply: send your CV and mention your Excel skills in the email subject.", [], ["excel"], "سطر التقديم"),
    # ---- إيجابية ----
    ("Proficiency in Microsoft Excel and Power BI.", ["excel", "power_bi"], [], "EN صريح"),
    ("Experience with PowerBI dashboards.", ["power_bi"], [], "كتابة ملتصقة PowerBI"),
    ("خبرة في إعداد التقارير المالية وفق المعايير الدولية IFRS.", ["financial_reporting", "ifrs"], [], "AR مالي"),
    ("Strong communication skills and fluency in English.", ["communication", "english"], [], "EN مهارات شخصية"),
    ("إجادة استخدام برنامج الإكسل والحاسب الآلي.", ["excel", "ms_office"], [], "AR مرادف غامض مع سياق"),
    ("يُجيد الإكسِل", ["excel"], [], "AR مع تشكيل"),
    ("Experience in WASH programmes is essential.", ["wash"], [], "WASH بأحرف كبيرة"),
    ("Experience developing REST APIs with Node.js.", ["api", "nodejs"], [], "REST APIs صحيحة"),
    ("مهارات التواصل الفعال والعمل ضمن فريق.", ["communication", "teamwork"], [], "AR شخصية"),
    ("Knowledge of cyber security and network security.", ["cybersecurity"], [], "أمن سيبراني صحيح"),
    ("خبرة في الأمن السيبراني.", ["cybersecurity"], [], "AR أمن سيبراني"),
    ("خبرة في إعداد الرواتب وقانون العمل اليمني.", ["payroll", "labor_law"], [], "AR رواتب"),
    ("Hands-on experience in machine learning and Python.", ["machine_learning", "python"], [], "ML صحيح"),
    ("إدارة عمليات التوظيف والاستقطاب وإجراء المقابلات.", ["recruitment"], [], "AR توظيف كمهارة"),
    ("Clinical experience and patient care in emergency settings.", ["clinical", "patient_care"], [], "صحي"),
    ("Monitoring and Evaluation (MEAL) experience.", ["me"], [], "MEAL مع سياق"),
    ("Petty cash management and bank reconciliation.", ["cash_management"], [], "إدارة نقدية صحيحة"),
    ("Leadership and team management experience.", ["leadership"], [], "قيادة صحيحة"),
    ("Mixed: خبرة في Excel و SQL Server وكتابة التقارير.", ["excel", "sql", "report_writing"], [], "نص مختلط"),
]

# (مهارة، عبارة، مقطع، تشابه أول، تشابه ثانٍ، يجب أن يكون القرار من)
SEMANTIC_CASES = [
    ("machine_learning", "learning outcomes", "Improve learning outcomes", 0.80, 0.70, {"rejected", "review"}),
    ("leadership", "Admin", "Admin", 0.78, 0.70, {"rejected", "review"}),
    ("recruitment", "مكان العمل", "مكان العمل: مأرب", 0.76, 0.70, {"rejected"}),
    ("cybersecurity", "ونظم الأمان", "الالتزام بنظم الأمان", 0.79, 0.70, {"rejected", "review"}),
    ("payroll", "الأرباح", "حساب الأرباح", 0.77, 0.70, {"rejected", "review"}),
    ("api", "rest periods", "ensure rest periods", 0.76, 0.70, {"rejected", "review"}),
    ("statistics", "التحليل الإحصائي المتقدم", "خبرة في التحليل الإحصائي المتقدم", 0.80, 0.70, {"accepted"}),
    ("data_analysis", "analysis of survey results", "analysis of survey results", 0.80, 0.70, {"accepted"}),
    ("excel", "spreadsheet software", "spreadsheet software", 0.80, 0.70, {"rejected"}),
    ("excel", "spreadsheet software", "spreadsheet software", 0.90, 0.70, {"review"}),
    ("communication", "communication abilities", "communication abilities", 0.80, 0.70, {"review"}),
    ("excel", "excel reports", "excel reports", 0.80, 0.79, {"rejected"}),   # غامضة بين مهارتين
]


class FakeTransformer:
    """مُرمِّز اختبار: كيس كلمات مجزّأ — يحاكي تشابهاً مرتفعاً عند اشتراك كلمات (أسوأ حالة للكلمات العامة)."""
    kind, dim = "transformer", 256

    def info(self):
        return {"kind": "transformer", "name": "fake-test-encoder", "dim": self.dim}

    def encode(self, texts, show_progress=False):
        out = np.zeros((len(texts), self.dim), np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"\w+", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)


def main():
    fails, total = [], 0
    ex = SkillExtractor(None)
    for text, must, must_not, desc in CASES:
        dec, _ = ex.extract_all(["j"], [text], semantic=False)
        acc = set(dec[dec.decision == "accepted"].skill_id)
        total += 1
        problems = [f"ناقصة {m}" for m in must if m not in acc] + [f"خاطئة مقبولة {m}" for m in must_not if m in acc]
        if dec.duplicated(["job_id", "skill_id"]).any():
            problems.append("تكرار")
        if (dec.confidence >= 1.0).any():
            problems.append("ثقة 1.00")
        clean = clean_for_model(text)
        for ev in dec[dec.decision == "accepted"].evidence:
            if ev.strip("…") not in clean:
                problems.append(f"الدليل ليس من النص: {ev}")
        if problems:
            fails.append((desc, problems, dec[["skill_id", "decision", "reason"]].to_dict("records")))
        print(f"{'✔' if not problems else '✖'} {desc:45s} مقبولة={sorted(acc)}")

    sem = SkillExtractor.__new__(SkillExtractor)
    for sid, ph, seg, s1, s2, allowed in SEMANTIC_CASES:
        total += 1
        d, c, why = SkillExtractor.semantic_decision(sem, sid, ph, seg, s1, s2)
        ok = d in allowed and c < 1.0
        print(f"{'✔' if ok else '✖'} دلالي {sid:16s} «{ph}» s={s1} → {d} ({why})")
        if not ok:
            fails.append((f"semantic {sid} {ph}", [f"قرار {d} غير متوقع"], why))

    # خط كامل بمُرمِّز اختبار: الكلمات العامة المشتركة يجب ألا تكفي للقبول
    tmp = Path(tempfile.mkdtemp())
    old = es.EMB_STORE_DIR
    es.EMB_STORE_DIR = tmp
    try:
        fx = SkillExtractor(FakeTransformer())
        texts = ["The training improves learning outcomes.", "Admin Officer", "مكان العمل: مأرب",
                 "Experience in machine learning models and Python."]
        dec, cands = fx.extract_all(["a", "b", "c", "d"], texts, semantic=True)
        acc = dec[dec.decision == "accepted"]
        bad = acc[(acc.job_id.isin(["a", "b", "c"])) & acc.skill_id.isin(["machine_learning", "leadership", "recruitment"])]
        good = {"machine_learning", "python"} <= set(acc[acc.job_id == "d"].skill_id)
        total += 1
        ok = bad.empty and good and (acc.confidence < 1.0).all() and not dec.duplicated(["job_id", "skill_id"]).any()
        print(f"{'✔' if ok else '✖'} خط كامل بمُرمِّز اختبار: مرشحات {len(cands)} | مقبولة {len(acc)} | "
              f"طرق {sorted(acc.method.unique())}")
        if not ok:
            fails.append(("pipeline fake encoder", bad.to_dict("records"), acc.to_dict("records")))
    finally:
        es.EMB_STORE_DIR = old
        shutil.rmtree(tmp)

    print(f"\n{total - len(fails)}/{total} حالة اختبار نجحت (اختبارات قواعد — ليست مقياس دقة).")
    for f in fails:
        print("✖", f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
