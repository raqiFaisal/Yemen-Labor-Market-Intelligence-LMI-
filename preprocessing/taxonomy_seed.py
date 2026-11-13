"""قاموس المهارات والمجالات والتخصصات الأساسي (Seed Taxonomy).

قاموس منسّق يدوياً بأسماء عربية/إنجليزية ومرادفات شائعة في إعلانات الوظائف اليمنية والعربية.
يمكن توسيعه آلياً من O*NET و ESCO عبر scripts/import_onet_esco.py.
كل مهارة: (id, الاسم الإنجليزي, الاسم العربي, النوع, [المجالات], [مرادفات])
النوع: technical | soft | professional
"""
import json
from core.config import SKILLS_TAXONOMY_FILE, DOMAINS_FILE, MAJORS_FILE

DOMAINS = {
    "software_engineering": {"ar": "هندسة البرمجيات", "en": "Software Engineering"},
    "data_science": {"ar": "علم البيانات", "en": "Data Science"},
    "accounting": {"ar": "المحاسبة والمالية", "en": "Accounting & Finance"},
    "marketing": {"ar": "التسويق", "en": "Marketing"},
    "human_resources": {"ar": "الموارد البشرية", "en": "Human Resources"},
    "healthcare": {"ar": "الرعاية الصحية", "en": "Healthcare"},
    "engineering": {"ar": "الهندسة", "en": "Engineering"},
    "management": {"ar": "الإدارة والمشاريع", "en": "Management"},
}

T, S, P = "technical", "soft", "professional"
SE, DS, AC, MK, HR, HC, EN, MG = DOMAINS.keys()

SKILLS = [
    # --- برمجيات ---
    ("python", "Python", "بايثون", T, [SE, DS], ["python3", "بايثون", "pandas", "numpy"]),
    ("javascript", "JavaScript", "جافاسكربت", T, [SE], ["js", "جافا سكربت", "java script"]),
    ("typescript", "TypeScript", "تايب سكربت", T, [SE], ["ts"]),
    ("java", "Java", "جافا", T, [SE], []),
    ("csharp", "C#", "سي شارب", T, [SE], ["c sharp", ".net", "dotnet", "asp.net"]),
    ("php", "PHP", "بي اتش بي", T, [SE], ["laravel", "لارافيل"]),
    ("react", "React", "رياكت", T, [SE], ["reactjs", "react.js"]),
    ("nodejs", "Node.js", "نود جي اس", T, [SE], ["node", "express"]),
    ("flutter", "Flutter", "فلاتر", T, [SE], ["dart"]),
    ("html_css", "HTML/CSS", "HTML و CSS", T, [SE, MK], ["html", "css", "html5"]),
    ("sql", "SQL", "قواعد البيانات SQL", T, [SE, DS, AC], ["mysql", "postgresql", "sql server", "قواعد البيانات", "قواعد بيانات"]),
    ("git", "Git", "جيت", T, [SE], ["github", "gitlab"]),
    ("docker", "Docker", "دوكر", T, [SE], ["kubernetes"]),
    ("cloud", "Cloud Computing", "الحوسبة السحابية", T, [SE], ["aws", "azure", "gcp", "الحوسبة السحابية"]),
    ("api", "REST APIs", "واجهات برمجة التطبيقات", T, [SE], ["restful", "rest api", "rest apis", "api", "apis", "web api"]),
    ("networking", "Networking", "الشبكات", T, [SE, EN], ["ccna", "شبكات", "الشبكات"]),
    ("cybersecurity", "Cybersecurity", "الأمن السيبراني", T, [SE], ["cyber security", "information security", "it security", "network security", "أمن المعلومات", "امن المعلومات", "الأمن السيبراني"]),
    ("linux", "Linux", "لينكس", T, [SE], ["ubuntu"]),
    ("ui_ux", "UI/UX Design", "تصميم واجهات وتجربة المستخدم", T, [SE, MK], ["ui/ux", "figma", "فيجما", "تجربة المستخدم"]),
    ("testing", "Software Testing", "اختبار البرمجيات", T, [SE], ["unit testing", "software testing", "qa testing", "test automation"]),
    # --- بيانات ---
    ("machine_learning", "Machine Learning", "تعلم الآلة", T, [DS], ["ml", "تعلم الالة"]),
    ("deep_learning", "Deep Learning", "التعلم العميق", T, [DS], ["pytorch", "tensorflow", "keras"]),
    ("data_analysis", "Data Analysis", "تحليل البيانات", T, [DS, AC, MK], ["data analytics", "تحليل بيانات", "تحليل البيانات"]),
    ("statistics", "Statistics", "الإحصاء", T, [DS, HC], ["إحصاء", "احصاء", "احصائي", "إحصائي", "تحليل احصائي", "spss", "stata"]),
    ("power_bi", "Power BI", "باور بي آي", T, [DS, AC], ["powerbi", "tableau", "تابلو"]),
    ("excel", "Microsoft Excel", "إكسل", T, [AC, DS, HR, MG], ["excel", "اكسل", "إكسل", "ms excel"]),
    ("nlp", "NLP", "معالجة اللغات الطبيعية", T, [DS], ["natural language processing"]),
    ("data_viz", "Data Visualization", "تصوير البيانات", T, [DS], ["visualization", "dashboards", "لوحات المعلومات"]),
    ("kobo", "KoboToolbox / ODK", "كوبو تولبوكس", T, [DS, MG], ["kobo", "odk", "كوبو"]),
    ("me", "Monitoring & Evaluation", "الرصد والتقييم", P, [DS, MG], ["m&e", "meal", "الرصد والتقييم", "المتابعة والتقييم"]),
    # --- محاسبة ---
    ("accounting", "Accounting", "المحاسبة", P, [AC], ["محاسبة", "accounting"]),
    ("financial_reporting", "Financial Reporting", "إعداد التقارير المالية", P, [AC], ["التقارير المالية", "financial statements", "القوائم المالية"]),
    ("ifrs", "IFRS", "المعايير الدولية IFRS", P, [AC], ["ifrs", "المعايير المحاسبية"]),
    ("auditing", "Auditing", "التدقيق والمراجعة", P, [AC], ["audit", "مراجعة", "تدقيق"]),
    ("budgeting", "Budgeting", "إعداد الموازنات", P, [AC, MG], ["budget", "الموازنة", "موازنة", "الميزانية"]),
    ("erp", "ERP Systems", "أنظمة ERP", T, [AC, MG], ["erp", "sap", "oracle", "odoo", "أودو"]),
    ("accounting_software", "Accounting Software", "برامج المحاسبة", T, [AC], ["quickbooks", "peachtree", "يمن سوفت", "الأمين"]),
    ("tax", "Taxation", "الضرائب والزكاة", P, [AC], ["ضرائب", "زكاة", "tax"]),
    ("cash_management", "Cash Management", "إدارة النقدية", P, [AC], ["الصندوق", "cash"]),
    # --- تسويق ---
    ("digital_marketing", "Digital Marketing", "التسويق الرقمي", P, [MK], ["التسويق الإلكتروني", "التسويق الالكتروني", "digital marketing"]),
    ("social_media", "Social Media Management", "إدارة وسائل التواصل", P, [MK], ["social media", "السوشيال ميديا", "التواصل الاجتماعي"]),
    ("seo", "SEO", "تحسين محركات البحث", T, [MK], ["seo", "sem"]),
    ("content_writing", "Content Writing", "كتابة المحتوى", P, [MK], ["copywriting", "كتابة المحتوى", "صناعة المحتوى"]),
    ("graphic_design", "Graphic Design", "التصميم الجرافيكي", T, [MK], ["photoshop", "illustrator", "فوتوشوب", "تصميم جرافيك"]),
    ("sales", "Sales", "المبيعات", P, [MK, MG], ["مبيعات", "البيع"]),
    ("market_research", "Market Research", "أبحاث السوق", P, [MK], ["دراسة السوق", "بحوث التسويق"]),
    ("branding", "Branding", "بناء العلامة التجارية", P, [MK], ["brand", "الهوية البصرية"]),
    ("customer_service", "Customer Service", "خدمة العملاء", P, [MK, MG], ["خدمة العملاء", "customer care"]),
    # --- موارد بشرية ---
    ("recruitment", "Recruitment", "التوظيف والاستقطاب", P, [HR], ["recruiting", "الاستقطاب", "التوظيف"]),
    ("payroll", "Payroll", "إعداد الرواتب", P, [HR, AC], ["الرواتب", "مسير الرواتب"]),
    ("labor_law", "Labor Law", "قانون العمل", P, [HR], ["قانون العمل اليمني", "labour law"]),
    ("training_dev", "Training & Development", "التدريب والتطوير", P, [HR], ["التدريب", "تطوير الموظفين"]),
    ("performance_mgmt", "Performance Management", "إدارة الأداء", P, [HR, MG], ["تقييم الأداء"]),
    ("hris", "HRIS", "أنظمة الموارد البشرية", T, [HR], ["hris", "hrms"]),
    # --- صحة ---
    ("patient_care", "Patient Care", "رعاية المرضى", P, [HC], ["رعاية المرضى", "التمريض", "nursing"]),
    ("clinical", "Clinical Skills", "المهارات السريرية", P, [HC], ["سريري", "clinical"]),
    ("public_health", "Public Health", "الصحة العامة", P, [HC], ["الصحة العامة"]),
    ("nutrition", "Nutrition", "التغذية", P, [HC], ["التغذية", "سوء التغذية", "cmam"]),
    ("pharmacy", "Pharmacy", "الصيدلة", P, [HC], ["صيدلة", "صيدلي", "الأدوية"]),
    ("laboratory", "Laboratory", "المختبرات", P, [HC], ["مختبر", "مختبرات", "lab"]),
    ("first_aid", "First Aid", "الإسعافات الأولية", P, [HC], ["الإسعافات الأولية", "الاسعافات الاولية"]),
    ("infection_control", "Infection Control", "مكافحة العدوى", P, [HC], ["ipc", "مكافحة العدوى"]),
    # --- هندسة ---
    ("autocad", "AutoCAD", "أوتوكاد", T, [EN], ["autocad", "اوتوكاد", "civil 3d"]),
    ("civil_eng", "Civil Engineering", "الهندسة المدنية", P, [EN], ["هندسة مدنية", "الإنشاءات", "construction"]),
    ("electrical_eng", "Electrical Engineering", "الهندسة الكهربائية", P, [EN], ["كهرباء", "هندسة كهربائية"]),
    ("solar", "Solar Energy", "الطاقة الشمسية", T, [EN], ["solar", "الطاقة الشمسية", "طاقة متجددة"]),
    ("wash", "WASH", "المياه والإصحاح البيئي", P, [EN, HC], ["wash", "المياه والصرف الصحي", "الإصحاح"]),
    ("site_supervision", "Site Supervision", "الإشراف الميداني", P, [EN], ["الإشراف على المواقع", "الإشراف الهندسي"]),
    ("quantity_surveying", "Quantity Surveying", "حساب الكميات", P, [EN], ["boq", "جداول الكميات"]),
    ("revit", "Revit / BIM", "ريفيت", T, [EN], ["revit", "bim"]),
    # --- إدارة ---
    ("project_management", "Project Management", "إدارة المشاريع", P, [MG], ["pmp", "إدارة المشاريع", "ادارة المشاريع"]),
    ("proposal_writing", "Proposal Writing", "كتابة المقترحات", P, [MG], ["كتابة المقترحات", "grant writing"]),
    ("procurement", "Procurement", "المشتريات", P, [MG, AC], ["المشتريات", "التوريد", "supply chain", "سلاسل الإمداد"]),
    ("logistics", "Logistics", "اللوجستيات", P, [MG], ["لوجستي", "المخازن"]),
    ("strategic_planning", "Strategic Planning", "التخطيط الاستراتيجي", P, [MG], ["التخطيط"]),
    ("report_writing", "Report Writing", "كتابة التقارير", P, [MG, DS, HC], ["كتابة التقارير", "إعداد التقارير", "reporting"]),
    ("ms_office", "MS Office", "حزمة أوفيس", T, [MG, HR, AC, MK], ["microsoft office", "ms office", "microsoft word", "ms word", "powerpoint", "الحاسوب", "الحاسب الآلي", "اوفيس", "عرض تقديمي", "العروض التقديمية"]),
    ("humanitarian", "Humanitarian Programming", "العمل الإنساني", P, [MG, HC], ["المنظمات الإنسانية", "العمل الإنساني", "ngo"]),
    # --- مهارات شخصية ---
    ("communication", "Communication", "مهارات التواصل", S, [], ["التواصل", "الاتصال", "communication skills"]),
    ("teamwork", "Teamwork", "العمل الجماعي", S, [], ["العمل ضمن فريق", "روح الفريق", "team player"]),
    ("leadership", "Leadership", "القيادة", S, [], ["القيادة", "leadership"]),
    ("problem_solving", "Problem Solving", "حل المشكلات", S, [], ["حل المشكلات", "problem-solving"]),
    ("time_management", "Time Management", "إدارة الوقت", S, [], ["إدارة الوقت", "العمل تحت الضغط"]),
    ("critical_thinking", "Critical Thinking", "التفكير النقدي", S, [], ["التفكير التحليلي", "analytical skills"]),
    ("negotiation", "Negotiation", "التفاوض", S, [], ["التفاوض"]),
    ("english", "English Language", "اللغة الإنجليزية", S, [], ["english", "الانجليزية", "الإنجليزية", "اللغة الانجليزية"]),
    ("attention_detail", "Attention to Detail", "الدقة والانتباه للتفاصيل", S, [], ["الدقة", "attention to detail"]),
    ("adaptability", "Adaptability", "المرونة والتكيف", S, [], ["المرونة", "flexibility"]),
]

# التخصصات الجامعية → المجالات المتوقعة (يُستخدم في صفحة التخصصات الأكاديمية)
MAJORS = {
    "علوم الحاسوب": [SE, DS],
    "تقنية المعلومات": [SE, DS],
    "هندسة البرمجيات": [SE],
    "نظم المعلومات": [SE, DS, MG],
    "الإحصاء": [DS],
    "المحاسبة": [AC],
    "العلوم المالية والمصرفية": [AC],
    "إدارة الأعمال": [MG, HR, MK],
    "التسويق": [MK],
    "الموارد البشرية": [HR],
    "الطب البشري": [HC],
    "التمريض": [HC],
    "الصيدلة": [HC],
    "المختبرات الطبية": [HC],
    "الهندسة المدنية": [EN],
    "الهندسة الكهربائية": [EN],
    "الهندسة المعمارية": [EN],
    "الإعلام": [MK],
}


def write_seed():
    skills = [
        {"id": i, "name_en": en, "name_ar": ar, "type": t, "domains": d, "aliases": al}
        for i, en, ar, t, d, al in SKILLS
    ]
    # الاحتفاظ بالمهارات المضافة من O*NET/ESCO عبر scripts.fetch_external
    if SKILLS_TAXONOMY_FILE.exists():
        seed_ids = {s["id"] for s in skills}
        old = json.loads(SKILLS_TAXONOMY_FILE.read_text(encoding="utf-8"))
        skills += [s for s in old if s["id"] not in seed_ids and s["id"].startswith(("onet_", "esco_"))]
    SKILLS_TAXONOMY_FILE.write_text(json.dumps(skills, ensure_ascii=False, indent=1), encoding="utf-8")
    DOMAINS_FILE.write_text(json.dumps(DOMAINS, ensure_ascii=False, indent=1), encoding="utf-8")
    MAJORS_FILE.write_text(json.dumps(MAJORS, ensure_ascii=False, indent=1), encoding="utf-8")
    return len(skills)


if __name__ == "__main__":
    print("skills:", write_seed())
