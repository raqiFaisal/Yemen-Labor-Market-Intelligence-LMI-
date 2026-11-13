"""توليد بيانات تجريبية (DEMO) لتشغيل التطبيق واختباره قبل توفر البيانات الحقيقية.

⚠️ هذه البيانات اصطناعية بالكامل. كل سجل يحمل source=DEMO_SYNTHETIC و is_demo=True،
   والتطبيق يعرض شريط تحذير واضح عند استخدامها. لا تُقدَّم أبداً كإحصائيات حقيقية.

الاستخدام:  python -m scripts.build_demo_data --n 3000
"""
import argparse
import json
import random

import numpy as np
import pandas as pd

from core.config import DEMO_INPUT_DIR, DEMO_SOURCE, SKILLS_TAXONOMY_FILE
from preprocessing.taxonomy_seed import write_seed

TITLES = {
    "software_engineering": ["مطور برمجيات", "مطور ويب", "مهندس برمجيات", "مطور تطبيقات موبايل", "Full Stack Developer",
                             "Backend Developer", "مسؤول أنظمة وشبكات", "IT Officer"],
    "data_science": ["محلل بيانات", "Data Analyst", "مسؤول إدارة المعلومات", "Information Management Officer",
                     "مهندس تعلم آلة", "Data Scientist", "مسؤول الرصد والتقييم"],
    "accounting": ["محاسب", "محاسب أول", "Finance Officer", "مدقق داخلي", "مسؤول مالي", "Accountant", "أمين صندوق"],
    "marketing": ["مسؤول تسويق", "مسؤول تسويق رقمي", "Marketing Officer", "مندوب مبيعات", "مصمم جرافيك",
                  "Social Media Specialist", "مسؤول خدمة عملاء"],
    "human_resources": ["مسؤول موارد بشرية", "HR Officer", "أخصائي توظيف", "HR Assistant", "مسؤول شؤون الموظفين"],
    "healthcare": ["ممرض/ة", "طبيب عام", "صيدلي", "فني مختبر", "Nutrition Officer", "مسؤول صحة عامة", "Health Officer"],
    "engineering": ["مهندس مدني", "مهندس كهرباء", "مهندس موقع", "WASH Engineer", "مهندس طاقة شمسية", "Site Engineer"],
    "management": ["مدير مشروع", "Project Manager", "مسؤول مشتريات", "منسق مشروع", "Project Coordinator",
                   "مسؤول لوجستيات", "Program Officer"],
}
YEMEN_ORGS = ["منظمة إنسانية محلية (تجريبي)", "بنك تجاري (تجريبي)", "شركة اتصالات (تجريبي)", "مستشفى خاص (تجريبي)",
              "شركة مقاولات (تجريبي)", "شركة برمجيات ناشئة (تجريبي)", "منظمة دولية (تجريبي)", "شركة تجارية (تجريبي)"]
YEMEN_GOVS = ["صنعاء"] * 9 + ["عدن"] * 6 + ["تعز"] * 4 + ["حضرموت"] * 3 + ["مأرب"] * 3 + ["الحديدة"] * 3 + \
             ["إب"] * 2 + ["ذمار"] + ["حجة"] + ["لحج"] + ["المهرة"] + ["شبوة"]
GLOBAL_COUNTRIES = ["Saudi Arabia"] * 4 + ["United Arab Emirates"] * 3 + ["Egypt"] * 2 + ["Jordan"] * 2 + \
                   ["United States"] * 2 + ["United Kingdom", "Germany", "Qatar", "Oman"]
DOMAIN_W_YE = {"management": 22, "healthcare": 18, "accounting": 15, "engineering": 12, "data_science": 10,
               "marketing": 9, "human_resources": 7, "software_engineering": 7}
DOMAIN_W_GL = {"software_engineering": 24, "data_science": 16, "marketing": 14, "accounting": 12,
               "management": 12, "healthcare": 10, "engineering": 8, "human_resources": 4}

# اتجاهات اصطناعية للطلب على المهارات (مضاعف سنوي) — لاختبار خط LSTM فقط
TREND = {"machine_learning": 1.55, "deep_learning": 1.6, "nlp": 1.5, "cloud": 1.35, "power_bi": 1.3,
         "data_analysis": 1.25, "python": 1.25, "solar": 1.4, "digital_marketing": 1.3, "social_media": 1.2,
         "cybersecurity": 1.3, "flutter": 1.25, "kobo": 1.1, "erp": 1.15,
         "accounting_software": 0.85, "php": 0.8, "ms_office": 0.92, "cash_management": 0.85, "logistics": 0.95}

GENERIC_TITLES = ["مسؤول", "مساعد إداري", "Officer", "Assistant", "منسق", "Coordinator", "موظف", "Specialist", "أخصائي", "Team Lead"]
EDU = ["بكالوريوس", "بكالوريوس", "بكالوريوس", "دبلوم", "ماجستير", "Bachelor's degree"]


def _mention(skill, rng):
    opts = [skill["name_ar"], skill["name_en"]] + skill["aliases"][:2]
    return rng.choice(opts)


def generate(n: int, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    skills = json.loads(SKILLS_TAXONOMY_FILE.read_text(encoding="utf-8"))
    by_domain = {d: [s for s in skills if d in s["domains"]] for d in TITLES}
    soft = [s for s in skills if s["type"] == "soft"]
    months = pd.period_range("2023-01", "2026-10", freq="M")
    # حجم الإعلانات ينمو قليلاً مع الزمن + موسمية بسيطة
    mw = np.array([(1 + 0.012 * i) * (1 + 0.12 * np.sin(i / 12 * 2 * np.pi)) for i in range(len(months))])
    mw = mw / mw.sum()

    rows = []
    for _ in range(n):
        yemen = rng.random() < 0.62
        dw = DOMAIN_W_YE if yemen else DOMAIN_W_GL
        domain = rng.choices(list(dw), weights=list(dw.values()))[0]
        mi = int(nprng.choice(len(months), p=mw))
        years = mi / 12
        pool = by_domain[domain]
        w = np.array([TREND.get(s["id"], 1.0) ** years for s in pool])
        k = min(len(pool), rng.randint(3, 6))
        idx = nprng.choice(len(pool), size=k, replace=False, p=w / w.sum())
        chosen = [pool[i] for i in idx]
        if rng.random() < 0.25:  # ضجيج: مهارة من مجال آخر
            other = rng.choice([d for d in TITLES if d != domain])
            chosen.append(rng.choice(by_domain[other]))
        chosen += rng.sample(soft, rng.randint(1, 3))
        # 30% مسميات عامة غامضة حتى يعتمد المصنف على المحتوى لا على المسمى فقط
        title = rng.choice(GENERIC_TITLES) if rng.random() < 0.30 else rng.choice(TITLES[domain])
        exp = rng.choice([0, 1, 2, 2, 3, 3, 4, 5, 7])
        req = "، ".join(_mention(s, rng) for s in chosen)
        if yemen:
            loc, country, company = rng.choice(YEMEN_GOVS), "Yemen", rng.choice(YEMEN_ORGS)
            desc = rng.choice([
                f"تعلن {company} عن حاجتها لشغل وظيفة {title} في محافظة {loc}.",
                f"مطلوب {title} للعمل في {loc} ضمن فريق العمل.",
                f"فرصة عمل: {title} — مقر العمل {loc}. يتولى شاغل الوظيفة تنفيذ المهام المتعلقة بالمجال.",
            ])
        else:
            loc, country, company = None, rng.choice(GLOBAL_COUNTRIES), "Global Company (demo)"
            desc = rng.choice([
                f"We are hiring a {title} to join our team in {country}.",
                f"{company} is looking for a {title}. You will work with cross-functional teams.",
                f"نبحث عن {title} للانضمام إلى فريقنا في {country}.",
            ])
        day = rng.randint(1, 28)
        rows.append({
            "title": title, "description": desc,
            "requirements": f"المتطلبات: {req}. خبرة لا تقل عن {exp} سنوات. المؤهل: {rng.choice(EDU)}.",
            "company": company, "location": loc, "country": country, "domain": domain,
            "posted_date": f"{months[mi].year}-{months[mi].month:02d}-{day:02d}",
            "experience_years": exp, "education": rng.choice(EDU),
            "source": DEMO_SOURCE, "url": "",
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    a = ap.parse_args()
    write_seed()
    df = generate(a.n)
    out = DEMO_INPUT_DIR / "demo_jobs.csv"  # خارج data/raw دائماً — لا يُقرأ إلا في LMI_MODE=demo
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"wrote {len(df)} DEMO rows -> {out}")
