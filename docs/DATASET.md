# الـ Dataset الموحد (المرحلة 5)

**التشغيل:** `python -m scripts.build_dataset [--skills lexicon|hybrid|auto|none]`. يستغرق حوالي 10 إلى 15 دقيقة على 36 ألف منشور على المعالج.
**التقييم:** `python -m scripts.evaluate_dataset`، بعد أن يعبّئ إنسان ملفات Gold.

> **جودة التسميات:** كل تسمية هنا **Silver**، أي ناتجة عن قواعد آلية مع السبب والثقة، ولم يتحقق منها إنسان.
> الأرقام الوحيدة الصالحة للتقرير العلمي هي التي ينتجها `evaluate_dataset` على عينة Gold. الثقة درجة استدلالية وليست احتمالاً معايَراً.

## المصادر والمبادئ
- المنشورات الخام: `data/raw/telegram/<channel>.csv`. صفحات التفاصيل: `data/raw/telegram/telegraph/pages.csv`. موقع YemenHR (`data/raw/yemen_yemenhr.csv`) يُستخدم مرجعاً لكشف التكرار فقط.
- لا يُعدَّل أي ملف خام، ولا يُحذف أي سجل. المكرر يُربط بـ `canonical_job_id` ويُعلَّم، ويبقى في الملفات.
- كل حقل مستخرج نص موجود فعلاً في الإعلان. ما لم يُذكر يبقى فارغاً، ولا توجد قيم افتراضية.
- أسطر التوقيع المتكررة في كل قناة (روابط، وسوم ثابتة، دعوات متابعة) تُستبعد **من التحليل فقط**. مثال حقيقي: تذييل «#وظائف #عمل #منح #دورات» في كل منشورات @forus_taiz كان يعطي إشارة «منحة» زائفة.

## الملفات — `data/processed/dataset/`
| الملف | المحتوى |
|---|---|
| `posts_all.csv` | كل منشور مفحوص: نوعه ونطاقه مع الثقة والسبب، وسبب الاستبعاد إن وُجد، وعدد الوظائف فيه |
| `opportunities.csv` / `.pkl` | سجل لكل فرصة، بعد تقسيم المنشورات التي تحتوي عدة وظائف، بالمخطط الموحد (أدناه) |
| `duplicate_pairs.csv` | كل زوج مرتبط مع الدليل: `exact_link`، `exact_text`، `near_same_org`، `near_text`، `possible`، `blocked_conflict` |
| `skill_decisions.csv` | كل قرار مهارة (accepted/review/rejected) مع الدليل النصي والسبب |
| `review_queue.csv` | حالات غير مؤكدة: نوع الفرصة، النطاق، تكرار محتمل |
| `classification/opportunity_type_{train,val,test}.csv` | نص المنشور والتسمية (Silver، ثقة ≥ 0.8، نسخة واحدة لكل مجموعة تكرار) |
| `classification/geographic_scope_{train,val,test}.csv` | مثله للنطاق الجغرافي (الفرص ذات الصلة فقط) |
| `ner/silver_{train,val,test}.jsonl` | `text`، و`spans` (مواضع حرفية)، و`tokens` و`ner_tags` (BIO) |
| `ner/label_scheme.json` | الوسوم: JOB_TITLE، ORG، LOCATION، EDUCATION، EXPERIENCE، SKILL |
| `timeseries/*.csv` | شهرياً، للإعلانات الفريدة فقط: حسب النوع، النطاق، المصدر، المحافظة، والطلب على كل مهارة (`share` و`observed` عند ≥ 10 إعلانات) |
| `telegram_jobs_for_training.csv` | وظائف وتدريب عملي فريدة، غير مكررة مع إعلانات موقع YemenHR، يقرؤها `train_all` تلقائياً |

**التقارير:** `reports/dataset_report.md` / `.json`.
**ملفات Gold** (تُنشأ مرة واحدة ولا تُستبدل): `labels/opportunity_gold_to_annotate.csv` (300 منشور، عينة طبقية) و`labels/ner_gold_to_annotate.jsonl` (100 مثال).

## المخطط الموحد (أهم الأعمدة)
- **المعرّفات والتكرار:** `job_id` (`tg:<channel>:<msg>` أو `…#p<n>` للوظيفة رقم n داخل منشور)، و`post_id`، و`canonical_job_id`، و`is_canonical`، و`dup_status`، و`dup_reason`.
- **المصدر:** `source`، `source_channel`، `source_message_id`، `source_url`، `telegraph_url`، `raw_text` (حرفياً).
- **الحقول المستخرجة:** `job_title`، `organization`، `location`، `governorate`، `country`، `work_location_type`، `description` (النص الكامل من telegra.ph إن وُجد)، `skills`، `qualifications`، `education`/`education_min`، `experience_min_years`/`experience_max_years` (مع الدليل)، `specializations`، `employment_type`، `salary`.
- **التواريخ (منفصلة):**
  - `message_date`: وقت النشر في القناة.
  - `published_date`: التاريخ المكتوب داخل الإعلان.
  - `application_deadline`: آخر موعد للتقديم.
  - `posting_date`: المعتمد للسلاسل الزمنية. هو تاريخ الإعلان إن كان ضمن ±60 يوماً من تاريخ الرسالة، وإلا تاريخ الرسالة. `posting_date_source` يحدد أيهما استُخدم.
- **التصنيف:** `opportunity_type` و`geographic_scope`، ومعهما `*_confidence` و`*_status` (accepted/review) و`*_reason`.
- **التقسيم:** `split_random` (70/15/15 بتجزئة `canonical_job_id`، فلا تنقسم مجموعة تكرار بين الأقسام، والتقسيم ثابت عبر التشغيلات) و`split_temporal` (حسب أقدم تاريخ في المجموعة).

## قواعد التكرار
1. **رابط مطابق:** نفس معرّف إعلان YemenHR في الرابط، أو نفس صفحة telegra.ph. هذا دمج مؤكد.
2. **نص مطابق بعد التطبيع:** خلال 120 يوماً. النص نفسه بعد مدة أطول يُعامل كإعادة نشر مستقلة.
3. **نفس المسمى المطبّع + نفس الجهة:** بشرط ألا يتعارض الموقع، وخلال 45 يوماً.
4. **نفس المسمى + تشابه نص ≥ 0.6:** خلال 45 يوماً.
5. **نفس المسمى والموقع والجهة غير معروفة:** لا دمج، والحالة تذهب للمراجعة (`possible_duplicate_review`).

**القيود:**
- جهتان مختلفتان معروفتان لا تُدمجان أبداً.
- وظيفتان من المنشور نفسه لا تُدمجان، ولا عبر سلسلة متعدية.
- لا يتجاوز المدى الزمني لأي مجموعة 90 يوماً، حتى لا يصبح إعلان يتكرر شهرياً لسنوات إعلاناً واحداً.

## القيود المعروفة
- **التصنيف قاعدي:** الحالات الغامضة (مثل الإعلانات التجارية المكتوبة بلغة الوظائف) تذهب إلى `review` ولا تُفرض.
- **المنشورات الصور فقط** تُستبعد بالسبب `no_text (media only)`، لأنه لا توجد قراءة لنص الصور في هذه المرحلة.
- **روابط `go.al-yemen.com`** المختصرة في @TheYemenGate لا تُفك. تكرارها مع غيرها يُكشف بالمسمى والجهة والتاريخ.
- **NER Silver** ناتج عن مطابقة حرفية لمخرجات القواعد. يفيد للتدريب الأولي، لكن تقييمه يحتاج Gold.

## التشغيل الكامل الآلي → FINAL_TRAINING_DATASET
خلية واحدة في Colab (`COLAB_RUN_ALL_CELL.py`) تستدعي `python -m scripts.run_all`:
1. فحص الأرشيف (بلا إعادة جمع) → 2. استكمال telegra.ph (3 جولات إعادة محاولة) → 3. OCR لمنشورات الصور بلا نص (بميزانية زمنية)
→ 4. `build_dataset` → 5. `finalize_dataset` → 6. `validate_final_dataset` (فحوص فعلية على الملفات) → 7. `FINAL_REPORT.md`.

المجلد النهائي (على Drive: `MyDrive/lmi_data/FINAL_TRAINING_DATASET`):
`all_posts.csv · all_opportunities.csv · unique_opportunities.csv · duplicate_pairs.csv ·
classification_opportunity_type/ (6 فئات + relevance_filter/ + train_balanced.csv) · classification_geographic_scope/ (8 فئات تفصيلية) ·
ner/ (9 كيانات، BIO، jsonl + conll) · skills/ (job_skills, skill_catalog, extraction/) · timeseries/ (+ lstm_ready/) ·
gold_annotation/ (فارغ للتصنيف البشري + دليل) · review_queue/ · dataset_statistics/ · data_dictionary.md ·
dataset_quality_report.html · dataset_manifest.json (بصمات SHA-256 وأعمدة وأنواع) · FINAL_REPORT.md`

**OCR:** Tesseract (ara+eng)؛ ثلاث قراءات لكل صورة (مقياسان × نمطا تقسيم) واختيار معلّل، لأن Tesseract غير مستقر مع المقياس والخلفيات الملونة
(مختبر على صور مولدة). نص OCR يُعلَّم `text_source=ocr` ويُرسل للمراجعة ولا يدخل ملفات التدريب الآلية.
