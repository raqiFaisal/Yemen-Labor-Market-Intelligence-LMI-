"""تسمية مجالات الوظائف (Labeling) — حل مشكلة "لا توجد بيانات معنونة".

الإعلانات اليمنية المجموعة لا تحمل تصنيفاً موحداً، لذلك نجمع ثلاثة مصادر للتسمية، لكل منها وسم صريح:

| label_source   | المصدر                                          | الاستخدام                         |
|----------------|-------------------------------------------------|-----------------------------------|
| gold           | تصنيف بشري (data/labels/gold_labels.csv)       | تدريب + **اختبار مستقل موثوق**      |
| source         | تصنيف الموقع نفسه (عمود category/domain)        | تدريب + اختبار                     |
| weak_title     | قاعدة كلمات مفتاحية على المسمى الوظيفي          | تدريب فقط (Weak Supervision)       |
| weak_skills    | تصويت المهارات المستخرجة حسب مجالاتها في القاموس | تدريب فقط (Weak Supervision)       |

التسمية الضعيفة (Weak/Distant Supervision) أسلوب معروف لبناء بيانات تدريب أولية؛ الشبكة تتعلم من الـ Embeddings
أنماطاً أوسع من القواعد فتصنّف إعلانات لا تغطيها القواعد. لكن تقييم النموذج على تسميات ضعيفة يقيس
"الاتفاق مع القواعد" لا الدقة الحقيقية — لذلك يُوسَم التقييم بنوعه، والدقة الحقيقية تتطلب تصنيفات gold.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from core.config import GOLD_LABELS_FILE
from models.skill_extractor import load_taxonomy
from preprocessing.schema import normalize_arabic

# كلمات مفتاحية عالية الدقة تُطابق على المسمى الوظيفي فقط
TITLE_RULES = {
    "software_engineering": ["مطور", "مبرمج", "برمجيات", "developer", "programmer", "software", "full stack", "backend",
                             "frontend", "front-end", "back-end", "devops", "it officer", "it specialist", "network",
                             "شبكات", "تقنية المعلومات", "مسؤول انظمه", "system admin", "web", "mobile app"],
    "data_science": ["data analyst", "data scientist", "محلل بيانات", "علم البيانات", "information management", "ادارة المعلومات",
                     "monitoring and evaluation", "m&e", "meal", "الرصد والتقييم", "رصد وتقييم", "رصد", "المتابعه والتقييم", "machine learning",
                     "تعلم الاله", "statistic", "احصاء", "gis", "business intelligence"],
    "accounting": ["محاسب", "accountant", "accounting", "finance", "financial", "مالي", "ماليه", "مدقق", "auditor", "audit",
                   "treasur", "امين صندوق", "cashier", "خزينه", "payroll accountant"],
    "marketing": ["تسويق", "marketing", "مبيعات", "sales", "مندوب", "social media", "سوشيال", "مصمم جرافيك", "graphic designer",
                  "content", "محتوي", "اعلام", "media", "communication officer", "خدمه العملاء", "customer service", "call center"],
    "human_resources": ["موارد بشريه", "human resource", "hr ", "hr officer", "hr assistant", "recruit", "توظيف",
                        "شؤون الموظفين", "talent", "payroll officer"],
    "healthcare": ["طبيب", "doctor", "ممرض", "nurse", "صيدل", "pharmac", "مختبر", "laboratory", "lab technician", "قابله",
                   "midwife", "تغذيه", "nutrition", "health", "صحه", "صحي", "clinical", "medical", "طبي", "mental health",
                   "psycholog", "نفسي"],
    "engineering": ["مهندس", "engineer", "هندس", "wash", "مياه واصحاح", "construction", "انشاءات", "site supervisor",
                    "مشرف موقع", "مساح", "surveyor", "solar", "طاقه شمسيه", "electrical", "كهرباء", "architect", "معماري"],
    "management": ["مدير مشروع", "project manager", "منسق", "coordinator", "program officer", "programme", "مدير برنامج",
                   "مشتريات", "procurement", "لوجست", "logistic", "supply chain", "سلاسل الامداد", "مدير", "manager",
                   "اداري", "admin", "director", "team lead", "head of"],
}
# ترتيب الأولوية عند تطابق أكثر من مجال: الكلمات الأكثر تحديداً أولاً ("مهندس برمجيات" → برمجيات وليس هندسة)
PRIORITY = ["software_engineering", "data_science", "healthcare", "accounting", "human_resources", "marketing",
            "engineering", "management"]
_RULES = {d: [normalize_arabic(k.lower()) for k in kws] for d, kws in TITLE_RULES.items()}


def title_rule(title: str) -> str | None:
    if not isinstance(title, str) or not title.strip():
        return None
    t = " " + normalize_arabic(title.lower()) + " "
    hits = [d for d in PRIORITY if any(k in t for k in _RULES[d])]
    if not hits:
        return None
    # "مدير/manager/منسق" عامة: لا تُحسم إلا إن لم يوجد مجال آخر
    specific = [d for d in hits if d != "management"]
    return specific[0] if specific else "management"


def skill_vote(skill_ids: list[str], min_votes: float = 2.5, margin: float = 1.5) -> str | None:
    _, by_id = load_taxonomy()
    votes = defaultdict(float)
    for sid in skill_ids:
        doms = by_id.get(sid, {}).get("domains", [])
        for d in doms:
            votes[d] += 1.0 / len(doms)
    if not votes:
        return None
    ranked = sorted(votes.items(), key=lambda x: -x[1])
    top, v1 = ranked[0]
    v2 = ranked[1][1] if len(ranked) > 1 else 0.0
    return top if v1 >= min_votes and v1 - v2 >= margin else None


def load_gold() -> pd.DataFrame:
    if not GOLD_LABELS_FILE.exists():
        return pd.DataFrame(columns=["job_id", "domain"])
    g = pd.read_csv(GOLD_LABELS_FILE, dtype=str).dropna(subset=["job_id", "domain"])
    return g.drop_duplicates("job_id", keep="last")[["job_id", "domain"]]


def apply_labels(jobs: pd.DataFrame, job_skills: pd.DataFrame, use_weak: bool = False) -> pd.DataFrame:
    """يضيف الأعمدة: label, label_source, weak_title, weak_skills, suggested_label.

    الافتراضي: label = تسمية موثوقة فقط (gold يدوي ثم source من الموقع). التسميات الضعيفة تُحفظ كاقتراح
    (suggested_label) لتسريع المراجعة اليدوية، ولا تصبح label إلا إذا فُعّلت صراحة (use_weak=True، تجريبي)."""
    jobs = jobs.copy()
    sk = job_skills.groupby("job_id").skill_id.apply(list)
    jobs["weak_title"] = jobs["title"].map(title_rule)
    jobs["weak_skills"] = jobs["job_id"].map(lambda j: skill_vote(sk.get(j, [])))
    gold = load_gold().set_index("job_id")["domain"]
    jobs["label"], jobs["label_source"] = None, None

    def pick(r):
        if r.job_id in gold.index:
            return gold[r.job_id], "gold"
        if isinstance(r.domain, str) and r.domain:
            return r.domain, "source"
        if not use_weak:
            return None, None
        wt = r.weak_title if isinstance(r.weak_title, str) else None   # NaN → None
        ws = r.weak_skills if isinstance(r.weak_skills, str) else None
        if wt and (ws is None or ws == wt):
            return wt, "weak_title"
        if ws and wt is None:
            return ws, "weak_skills"
        return None, None  # تعارض بين المسمى والمهارات → غير مؤكد، لا يُستخدم

    if len(jobs):
        picked = jobs.apply(pick, axis=1, result_type="expand")
        jobs["label"], jobs["label_source"] = picked[0], picked[1]

    def suggest(r):
        wt = r.weak_title if isinstance(r.weak_title, str) else None
        ws = r.weak_skills if isinstance(r.weak_skills, str) else None
        if wt and (ws is None or ws == wt):
            return wt
        return ws if (ws and wt is None) else None
    jobs["suggested_label"] = jobs.apply(suggest, axis=1) if len(jobs) else None
    return jobs


def review_queue(jobs: pd.DataFrame) -> pd.DataFrame:
    """الإعلانات التي تحتاج مراجعة يدوية: بلا تسمية موثوقة، مرتبة بالأولوية
    (فئة مصدر غامضة/متعارضة أولاً، ثم تعارض الاقتراح مع تنبؤ النموذج، ثم الثقة المنخفضة)."""
    trusted = jobs["label_source"].isin(["gold", "source"]) if "label_source" in jobs else False
    q = jobs[~trusted & ~jobs.get("is_demo", False)].copy()
    pri = pd.Series(3, index=q.index)
    if "category_status" in q:
        pri[q["category_status"].isin(["ambiguous", "conflict", "unmapped"])] = 0
    if "domain_pred" in q and "suggested_label" in q:
        dis = q["domain_pred"].notna() & q["suggested_label"].notna() & (q["domain_pred"] != q["suggested_label"])
        pri[dis & (pri > 1)] = 1
    if "domain_conf" in q:
        pri[(q["domain_conf"].fillna(0) < 0.6) & (pri > 2)] = 2
    q["review_priority"] = pri
    return q.sort_values(["review_priority", "posted_date"], ascending=[True, False])


def label_summary(jobs: pd.DataFrame) -> dict:
    if "label_source" not in jobs:
        return {}
    return {k: int(v) for k, v in jobs["label_source"].fillna("none").value_counts().items()}

