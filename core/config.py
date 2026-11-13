"""إعدادات المشروع المركزية — كل المسارات والثوابت في مكان واحد.

وضعا التشغيل (متغير البيئة LMI_MODE):
    real  (الافتراضي)  بيانات حقيقية فقط: data/raw → data/processed → artifacts/
    demo               بيانات اصطناعية للعرض فقط: data/demo → data/processed_demo → artifacts_demo/
المساران منفصلان كلياً، فلا يمكن أن تختلط بيانات أو نماذج أو مقاييس DEMO بالنتائج الحقيقية.
    تشغيل وضع العرض:  LMI_MODE=demo streamlit run app.py   |   python -m scripts.train_demo
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODE = "demo" if os.environ.get("LMI_MODE", "real").strip().lower() == "demo" else "real"
IS_DEMO_MODE = MODE == "demo"

DATA_DIR = ROOT / "data"
TAXONOMY_DIR = DATA_DIR / "taxonomy"   # مشترك بين الوضعين (قاموس فقط، لا بيانات)
REPORTS_DIR = ROOT / "reports"
DEMO_INPUT_DIR = DATA_DIR / "demo"     # مكان ملف DEMO الوحيد — خارج data/raw دائماً

if IS_DEMO_MODE:
    RAW_DIR = DEMO_INPUT_DIR
    PROCESSED_DIR = DATA_DIR / "processed_demo"
    ARTIFACTS_DIR = ROOT / "artifacts_demo"
    LABELS_DIR = DATA_DIR / "labels_demo"
else:
    RAW_DIR = DATA_DIR / "raw"          # ملفات خام حقيقية كما جُمعت (YemenHR، ArabJobs، ...)
    PROCESSED_DIR = DATA_DIR / "processed"
    ARTIFACTS_DIR = ROOT / "artifacts"
    LABELS_DIR = DATA_DIR / "labels"    # تصنيفات يدوية (Gold)

LOGS_DIR = RAW_DIR / "logs"            # سجلات الجمع وأسباب الفشل
STATE_DIR = RAW_DIR / "state"          # الروابط المكتشفة (ليست إعلانات منزّلة)
EMB_STORE_DIR = ARTIFACTS_DIR / "embedding_store"

for _d in (RAW_DIR, PROCESSED_DIR, TAXONOMY_DIR, ARTIFACTS_DIR, REPORTS_DIR, LABELS_DIR, DEMO_INPUT_DIR,
           LOGS_DIR, STATE_DIR, EMB_STORE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

JOBS_FILE = PROCESSED_DIR / "jobs_unified.pkl"
JOB_SKILLS_FILE = PROCESSED_DIR / "job_skills.pkl"
JOB_SKILLS_CSV = PROCESSED_DIR / "job_skills.csv"      # نسخة مقروءة: job_id, skill, method, confidence, evidence
SKILL_DECISIONS_FILE = PROCESSED_DIR / "skill_decisions.pkl"   # كل زوج (إعلان، مهارة): accepted/review/rejected + السبب
SKILL_CANDIDATES_CSV = PROCESSED_DIR / "skill_candidates.csv"  # كل المرشحات الخام من الطريقتين (للتدقيق)
SKILLS_TAXONOMY_FILE = TAXONOMY_DIR / "skills.json"
DOMAINS_FILE = TAXONOMY_DIR / "domains.json"
MAJORS_FILE = TAXONOMY_DIR / "majors.json"
CATEGORY_MAP_FILE = TAXONOMY_DIR / "category_map.json"  # تصنيفات المواقع → المجالات الثمانية (قابل للمراجعة)

CLASSIFIER_FILE = ARTIFACTS_DIR / "job_classifier.pt"
CLASSIFIER_METRICS = ARTIFACTS_DIR / "classifier_metrics.json"
LSTM_FILE = ARTIFACTS_DIR / "skill_lstm.pt"
LSTM_METRICS = ARTIFACTS_DIR / "lstm_metrics.json"
LSTM_PREDICTIONS = ARTIFACTS_DIR / "lstm_test_predictions.csv"  # الفعلي مقابل المتوقع على فترة الاختبار
ENCODER_INFO = ARTIFACTS_DIR / "encoder_info.json"
EMB_CACHE = ARTIFACTS_DIR / "job_embeddings.npy"        # مصفوفة مرتبة بترتيب jobs_unified
EMB_IDS = ARTIFACTS_DIR / "job_embeddings_ids.json"     # job_id لكل صف في المصفوفة (للتحقق من التطابق)
TFIDF_FALLBACK = ARTIFACTS_DIR / "tfidf_svd_encoder.joblib"
DATA_REPORT = ARTIFACTS_DIR / "data_report.json"   # تشخيص جاهزية البيانات للتدريب
GOLD_LABELS_FILE = LABELS_DIR / "gold_labels.csv"  # job_id,domain — تصنيف بشري
TO_LABEL_FILE = LABELS_DIR / "to_label.csv"        # ملف التسمية اليدوية المُصدَّر
SKILL_REVIEWS_FILE = LABELS_DIR / "skill_reviews.csv"  # مراجعة بشرية للمهارات: job_id,skill_id,verdict(correct|incorrect)
SKILL_ANNOTATIONS_FILE = LABELS_DIR / "skills_to_annotate.csv"  # تعليم كامل لعينة (Ground Truth للاستدعاء)

# النموذج المعتمد في وثيقة المشروع (يُستخدم كمُرمِّز مجمَّد — لا Fine-tuning)
TRANSFORMER_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# قيمة الحقل source التي تميز البيانات التجريبية عن الحقيقية
DEMO_SOURCE = "DEMO_SYNTHETIC"

# إعدادات مصنف المجالات
CLF_MIN_LABELED = 60     # أقل عدد سجلات معنونة موثوقة لبدء التدريب
CLF_MIN_PER_CLASS = 5    # الفئات الأقل من ذلك تُستبعد من التدريب (بدل انهيار التقسيم الطبقي)
CLF_RECOMMENDED_PER_CLASS = 30  # العدد الموصى به لكل فئة لتقييم مستقر
CLF_MIN_GOLD_TEST = 20   # إن توفر هذا العدد من التصنيفات اليدوية تُستخدم كبيانات اختبار مستقلة
CLF_MIN_CONFIDENCE = 0.5 # أقل ثقة لقبول تنبؤ النموذج كمجال للإعلان، وإلا "غير مصنف"

# إعدادات LSTM (الوحدة الزمنية: M شهري افتراضياً أو W أسبوعي)
LSTM_FREQ = "M"
LSTM_WINDOW = 6        # أقصى نافذة؛ تُصغَّر تلقائياً حسب طول السلاسل الفعلي (حد أدنى 3)
LSTM_MIN_MONTHS = 18   # أقل طول سلسلة (بالفترات) لاعتبار البيانات كافية للتدريب
LSTM_TEST_MONTHS = 6   # آخر N فترة للاختبار (لا تدخل التدريب ولا الإيقاف المبكر)
LSTM_VAL_MONTHS = 3    # الفترات السابقة للاختبار: تحقق للإيقاف المبكر
LSTM_MIN_MENTIONS = 25 # أقل عدد إعلانات تذكر المهارة في السلسلة
LSTM_MIN_POSTINGS_PER_PERIOD = 10  # فترة فيها إعلانات أقل من ذلك = "بيانات غير كافية" (NaN) وليست طلباً صفرياً
LSTM_MIN_COVERAGE = 0.8  # أقل نسبة فترات مرصودة فعلياً داخل السلسلة
FREQ_AR = {"M": ("شهر", "أشهر"), "W": ("أسبوع", "أسابيع")}

COUNTRY_YEMEN = "Yemen"
