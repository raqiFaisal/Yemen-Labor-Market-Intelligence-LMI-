# 🇾🇪 Yemen Labor Market Intelligence (LMI)
### AI-Powered Yemeni Labor Market Analysis & Forecasting

**ذكاء سوق العمل اليمني**

An intelligent platform designed to analyze job market announcements, extract employment information, discover skill demand, and explore future labor market trends using Deep Learning and Natural Language Processing.

منصة ذكية لتحليل إعلانات سوق العمل اليمني، واستخراج المعلومات الوظيفية، وتحليل المهارات المطلوبة، واستكشاف اتجاهات الطلب المستقبلية باستخدام تقنيات التعلم العميق ومعالجة اللغة الطبيعية.

**Developed by:** Eng. Ruqayya Faisal Al-Taybi  
**إعداد وتطوير:** المهندسة رقية فيصل الطيبي

---

## 1. Project Overview | نظرة عامة

Yemen Labor Market Intelligence (LMI) transforms unstructured job announcements into structured information and interactive labor market insights.

The platform integrates data processing, Arabic language understanding, deep learning models, and time-series forecasting into one application.

يهدف المشروع إلى تحويل إعلانات الوظائف غير المنظمة إلى معلومات منظمة ومؤشرات تفاعلية تساعد على فهم سوق العمل اليمني واتجاهاته.

## 2. Main Features | الوظائف الرئيسية

- **Interactive Dashboard:** Overview of labor market indicators and statistics.
- **Market Analysis:** Explore job opportunities, specializations, skills, and geographic distribution.
- **Job Advertisement Analysis:** Classify announcements and extract important information.
- **Skill Demand Analysis:** Identify frequently requested professional skills.
- **Future Forecasting:** Explore expected demand trends.
- **Skill Gap Analysis:** Compare individual skills with market requirements.
- **Job Search:** Search and explore available opportunities.
- **Model Information:** View information about the integrated AI models.

## 3. Deep Learning Architecture | معمارية التعلم العميق

### AraBERT Models

The project uses fine-tuning of pretrained AraBERT for Arabic job market understanding.

Three main tasks:

1. **Opportunity Type Classification:** Classifies announcements into opportunity categories.
2. **Geographic Scope Classification:** Identifies the geographic coverage of opportunities.
3. **Named Entity Recognition (NER):** Extracts information such as job titles, organizations, skills, qualifications, experience, locations, salaries, and deadlines.

### LSTM Forecasting

LSTM models are implemented for four time-series families:

| Model Family | Forecasting Target |
|---|---|
| Skills | Demand for professional skills |
| Specializations | Demand for academic and professional specializations |
| Job Titles | Demand for job titles |
| Governorates | Geographic distribution of opportunities |

Forecasting models are evaluated against Naive and Moving Average baselines.

## 4. Technology Stack | التقنيات المستخدمة

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep Learning |
| AraBERT | Arabic NLP |
| Transformers | Pretrained language models and fine-tuning |
| LSTM | Time-series forecasting |
| Pandas & NumPy | Data processing |
| Scikit-learn | Evaluation and baseline models |
| Streamlit | Interactive application |
| Plotly | Interactive visualizations |
| Google Colab | Model training environment |

## 5. Project Structure | هيكل المشروع

```text
Yemen-Labor-Market-Intelligence-LMI/
│
├── app_dl/
│   ├── main.py
│   ├── common.py
│   └── views/
│
├── src/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   ├── forecasting/
│   ├── integration/
│   └── offline/
│
├── preprocessing/
├── core/
├── config/
├── utils/
├── pages/
├── scripts/
├── notebooks/
├── docs/
├── tests/
│
├── app.py
├── requirements.txt
├── requirements_dl.txt
├── requirements_runtime.txt
├── .gitignore
└── README.md
```

## 6. Installation | التثبيت

Clone the repository:

```bash
git clone https://github.com/raqiFaisal/Yemen-Labor-Market-Intelligence-LMI-.git
```

Navigate to the project directory:

```bash
cd Yemen-Labor-Market-Intelligence-LMI-
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements_dl.txt
```

## 7. Running the Application | تشغيل التطبيق

The primary Deep Learning application entry point is:

```bash
python -m streamlit run app_dl/main.py
```

The application provides an Arabic right-to-left interface.

**Note:** The previous application entry point `app.py` is retained as a legacy version. The recommended entry point for the Deep Learning release is `app_dl/main.py`.

## 8. Training Models | تدريب النماذج

### AraBERT Classification

```bash
python -m src.training.train_classifier --task opportunity_type

python -m src.training.train_classifier --task geographic_scope
```

### AraBERT NER

```bash
python -m src.training.train_ner
```

### LSTM Forecasting

```bash
python -m src.forecasting.train_lstm
```

The Colab training notebook is available at:

`notebooks/train_dl_colab.ipynb`

GPU is recommended for AraBERT fine-tuning. LSTM training supports CPU execution.

## 9. Evaluation | تقييم النماذج

Classification evaluation:

```bash
python -m src.evaluation.evaluate_classifier --task opportunity_type
```

NER evaluation:

```bash
python -m src.evaluation.evaluate_ner
```

The evaluation pipeline supports independent test-set evaluation and performance reporting.

## 10. Data and Model Assets | البيانات والنماذج

The repository contains source code, training scripts, notebooks, and documentation.

Large training datasets, trained model weights, and generated reports are maintained separately and are not included in this public repository.

To run the complete application, provide the required dataset and model directories.

The application supports these environment variables:

- `LMI_FINAL_DIR`: Final training dataset directory.
- `LMI_MODELS_DIR`: Trained models directory.
- `LMI_REPORTS_DIR`: Evaluation reports directory.

Dataset annotation methodology and model training procedures are documented in the `docs/` directory.

## 11. Documentation | التوثيق

- `docs/DL_SYSTEM.md` — Deep Learning architecture and training guide.
- `docs/DATASET.md` — Dataset structure and processing methodology.
- `docs/RESULTS.md` — Recorded evaluation and forecasting results.
- `FINAL_PROJECT_REPORT.md` — Project technical report.

## 12. Testing | الاختبارات

Run the Deep Learning system tests:

```bash
pytest -q tests/test_dl_system.py
```

---

### Project Vision

**From Data to Insights Supporting the Future of Yemen's Labor Market.**
---

## 📸 Application Screenshots | صور واجهة التطبيق

<p align="center">
  <img src="assets/Screenshot%202026-11-13%20140918.png" width="48%" alt="LMI Screenshot 1">
  <img src="assets/Screenshot%202026-11-13%20141005.png" width="48%" alt="LMI Screenshot 2">
</p>

<p align="center">
  <img src="assets/Screenshot%202026-11-13%20141032.png" width="48%" alt="LMI Screenshot 3">
  <img src="assets/Screenshot%202026-11-13%20141100.png" width="48%" alt="LMI Screenshot 4">
</p>

<p align="center">
  <img src="assets/Screenshot%202026-11-13%20141151.png" width="48%" alt="LMI Screenshot 5">
  <img src="assets/Screenshot%202026-11-13%20141400.png" width="48%" alt="LMI Screenshot 6">
</p>

«من البيانات إلى رؤى تدعم مستقبل سوق العمل اليمني»

**Developed by: Eng. Ruqia Faisal Al-Taibi**

**Yemen Labor Market Intelligence (LMI)**
