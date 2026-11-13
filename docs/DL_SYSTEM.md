# نظام التعلم العميق — دليل التشغيل

## البنية
```
src/
  common.py                 المسارات (LMI_FINAL_DIR / LMI_MODELS_DIR / LMI_REPORTS_DIR)، البذور، الجهاز
  data/loaders.py           قراءة التصنيف وNER مع تحقق صارم من الأعمدة والوسوم
  data/audit.py             تدقيق: جودة، تسرب بين التقسيمات، صلاحية BIO، Gold، كفاية السلاسل
  training/train_classifier.py   AraBERT تصنيف تسلسل (نوع الفرصة / النطاق)
  training/train_ner.py          AraBERT تصنيف رموز (BIO)
  training/baselines.py          TF-IDF+LR/SVM، CRF
  evaluation/metrics.py          مقاييس موحدة (sklearn + seqeval صارم + MAE/RMSE/MAPE)
  evaluation/evaluate_*.py       تقييم test مستقل + تحليل أخطاء + Gold إن وُجد
  inference/predictors.py        TextClassifier / NERTagger (طبقة الاستدلال الوحيدة)
  forecasting/series.py          السلاسل الشهرية + فحص الكفاية
  forecasting/train_lstm.py      LSTM عام لكل عائلة + Naive + MA-3
  integration/engine.py          محرك التكامل (السوق، تحليل الإعلان، فجوة المهارات، التوقعات)
app_dl/                      تطبيق Streamlit عربي RTL (8 صفحات) — مستقل عن app.py القديم
notebooks/train_dl_colab.ipynb   تدريب كامل على Colab GPU مع نقل الملفات بلا Drive
scripts/package_results.py   تجميع reports + models (fp16) للتنزيل
tests/test_dl_system.py      36 اختبار pytest
```

## ضمانات منهجية
| الضمان | أين يُطبَّق |
|---|---|
| test لا يُقرأ أثناء التدريب أو اختيار النموذج | `train_*` تحمّل train/validation فقط؛ `training_config.json` يسجل `test_used_during_training: false` |
| اختيار المعاملات/Checkpoint على validation | Macro-F1 (تصنيف)، Entity-F1 (NER)، MSE (LSTM)، C/c1-c2 (خطوط الأساس) |
| لا تسرب | التقسيم بالمجموعات (canonical_job_id) + دمج النصوص المتطابقة؛ `audit.py` يرفض أي تداخل |
| تقييم NER صارم | seqeval IOB2 strict: الكيان صحيح فقط بتطابق النوع والحدود |
| LSTM لا يُدرَّب على بيانات غير كافية | ≥18 شهراً، تغطية ≥80%، ≥25 إشارة؛ الأشهر قليلة الإعلانات = NaN لا صفر |
| التطبيق لا يعيد التدريب ولا يستخدم قواعد بديلة | `AdAnalyzer` يعيد `model_not_available` صراحة |

## التشغيل
```bash
# 1) تدقيق
python -m src.data.audit
# 2) خطوط أساس (CPU يكفي)
python -m src.training.baselines --task all
# 3) AraBERT (GPU موصى به — notebooks/train_dl_colab.ipynb)
python -m src.training.train_classifier --task opportunity_type && python -m src.evaluation.evaluate_classifier --task opportunity_type
python -m src.training.train_classifier --task geographic_scope && python -m src.evaluation.evaluate_classifier --task geographic_scope
python -m src.training.train_ner && python -m src.evaluation.evaluate_ner
# 4) LSTM (CPU يكفي)
python -m src.forecasting.train_lstm
# 5) التطبيق
streamlit run app_dl/main.py
# 6) الاختبارات
pytest -q tests/test_dl_system.py
```

## نقل البيانات والنتائج بين Colab والكمبيوتر (بدون Google Drive)
- **Colab → الكمبيوتر**: `shutil.make_archive(...)` ثم `google.colab.files.download(...)` → مجلد Downloads.
- **الكمبيوتر → Colab**: `google.colab.files.upload()` ثم فك الضغط من الذاكرة.
- النتائج: `scripts/package_results.py --fp16` ينتج `lmi_reports.zip` (صغير) و`lmi_models.zip` (أوزان fp16).

## مطابقة أسماء الكيانات مع المطلوب
| المطلوب | في البيانات |
|---|---|
| JOB_TITLE, SKILL, EXPERIENCE, ORGANIZATION, LOCATION | بنفس الاسم |
| QUALIFICATION | DEGREE |
| SPECIALIZATION | MAJOR |
| (إضافي) | SALARY, DEADLINE |
الأنواع بلا أمثلة في test تُستبعد تلقائياً من المتوسطات (`per_entity` بـ support > 0).
