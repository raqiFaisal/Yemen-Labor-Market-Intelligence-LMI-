"""كشف الإعلانات المكررة عبر المصادر وربطها بـ canonical_job_id — بلا حذف أي سجل أصلي.

الأدلة (من الأقوى):
  exact_link      نفس معرف إعلان YemenHR في الرابط، أو نفس صفحة telegra.ph
  exact_text      نفس النص بعد التطبيع (منشور معاد توجيهه/منسوخ حرفياً)
  near_same_org   نفس المسمى المطبّع + نفس الجهة + الموقع غير متعارض + فرق التاريخ ≤ WINDOW_DAYS
  near_text       نفس المسمى المطبّع + تشابه نص ≥ TEXT_SIM (Jaccard على 5-grams حرفية) + فرق التاريخ ≤ WINDOW_DAYS
  قيود: لا يُدمج سجلان من المنشور نفسه، ولا تتجاوز المجموعة MAX_GROUP_SPAN_DAYS (منع السلاسل المتعدية عبر الزمن).
  possible        نفس المسمى + نفس الموقع + فرق التاريخ ≤ WINDOW_DAYS لكن الجهة غير معروفة لأحدهما والنص قصير
                  → لا يُدمج؛ يُسجَّل للمراجعة اليدوية
تشابه المسمى وحده لا يكفي أبداً. جهتان مختلفتان معروفتان = ليستا مكررتين مهما تشابه الباقي.
"""
from __future__ import annotations

import hashlib
import re

import pandas as pd

from preprocessing.schema import normalize_arabic

WINDOW_DAYS = 45
MAX_GROUP_SPAN_DAYS = 90      # أقصى مدى زمني لمجموعة تكرار (عدا الروابط المطابقة)
EXACT_WINDOW_DAYS = 120
TEXT_SIM = 0.6
NOISE_ORG = {"", "confidential", "سري", "غير محدد", "anonymous", "nan"}
TITLE_NOISE = re.compile(r"\(\s*\d+\s*(?:positions?|وظائف|وظيفة)\s*\)|\bre-?advertis\w*|\bextension\b|اعادة اعلان|إعادة إعلان|"
                         r"\(?(?:sana'?a|aden|taiz|marib|ma'rib|hodeidah|صنعاء|عدن|تعز|مارب|مأرب|الحديدة)\)?", re.I)


def norm_title(t: str) -> str:
    t = TITLE_NOISE.sub(" ", str(t or "").lower())
    t = normalize_arabic(t)
    t = re.sub(r"[^\w؀-ۿ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def norm_org(o: str) -> str:
    o = normalize_arabic(str(o or "").lower())
    o = re.sub(r"\b(?:organization|organisation|foundation|international|yemen|ngo|inc|ltd|co)\b|منظمة|مؤسسة|شركة|اليمن", " ", o)
    o = re.sub(r"[^\w؀-ۿ]+", " ", o)
    return re.sub(r"\s+", " ", o).strip()


def norm_text_hash(t: str) -> str:
    t = normalize_arabic(str(t or "").lower())
    t = re.sub(r"https?://\S+|#\S+|@\S+", " ", t)
    t = re.sub(r"[^\w؀-ۿ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return hashlib.sha1(t.encode()).hexdigest()[:16] if len(t) >= 30 else ""


def shingles(t: str, k: int = 5) -> set:
    t = re.sub(r"\s+", " ", normalize_arabic(str(t or "").lower()))[:1200]
    return {t[i:i + k] for i in range(max(0, len(t) - k + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def post_of(jid: str) -> str:
    return jid.split("#")[0]


class UF:
    """Union-Find مع قيد «لا يُدمج سجلان مختلفان من المنشور نفسه» (شاغران في إعلان واحد ليسا تكراراً، ولا عبر سلسلة متعدية)."""
    def __init__(self, items, dates=None):
        self.p = {i: i for i in items}
        self.posts = {i: {post_of(i): {i}} for i in items}
        dates = dates or {}
        self.span = {i: (dates.get(i), dates.get(i)) for i in items}     # (أقدم، أحدث) تاريخ في المجموعة

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def conflict(self, a, b) -> bool:
        pa, pb = self.posts[self.find(a)], self.posts[self.find(b)]
        return any(k in pb and pa[k] != pb[k] for k in pa)

    def merged_span(self, ra, rb):
        ds = [x for x in (*self.span[ra], *self.span[rb]) if x is not None and not pd.isna(x)]
        return (min(ds), max(ds)) if ds else (None, None)

    def union(self, a, b, force: bool = False) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if not force and self.conflict(a, b):
            return False
        lo, hi = self.merged_span(ra, rb)
        if not force and lo is not None and (hi - lo).days > MAX_GROUP_SPAN_DAYS:
            return False                      # يمنع سلسلة متعدية تجمع إعلانات متكررة شهرياً عبر سنوات في إعلان واحد
        keep, gone = min(ra, rb), max(ra, rb)
        self.span[keep] = (lo, hi)
        self.span.pop(gone, None)
        self.p[gone] = keep
        for k, v in self.posts.pop(gone).items():
            self.posts[keep].setdefault(k, set()).update(v)
        return True


def find_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """df أعمدة مطلوبة: job_id, source, title, organization, governorate, date (datetime), text, link_keys (قائمة مفاتيح).
    يعيد (assign: job_id→canonical_job_id/dup_status/dup_reason, pairs: كل زوج مع الدليل)."""
    d = df.copy()
    d["nt"] = d.title.map(norm_title)
    d["no"] = d.organization.map(norm_org)
    d["nh"] = d.text.map(norm_text_hash)
    uf, pairs = UF(d.job_id, dict(zip(d.job_id, d.date))), []

    def link(a, b, reason, score=1.0):
        if reason != "possible" and not uf.union(a, b, force=reason.startswith("exact_link")):
            reason = "blocked_conflict"          # نفس المنشور، أو مدى زمني للمجموعة > MAX_GROUP_SPAN_DAYS
        pairs.append({"job_id_a": a, "job_id_b": b, "evidence": reason, "score": round(score, 3)})

    # 1) مفاتيح روابط مشتركة
    owner = {}
    for jid, keys in zip(d.job_id, d.link_keys):
        for k in keys or []:
            if k in owner and owner[k] != jid:
                link(owner[k], jid, "exact_link:" + k.split(":")[0])
            owner.setdefault(k, jid)
    # 2) نص مطابق بعد التطبيع
    first = {}
    for jid, h, dt in sorted(zip(d.job_id, d.nh, d.date), key=lambda x: (pd.Timestamp.min if pd.isna(x[2]) else x[2])):
        if not h:
            continue
        if h in first and (pd.isna(dt) or pd.isna(first[h][1]) or (dt - first[h][1]).days <= EXACT_WINDOW_DAYS):
            link(first[h][0], jid, "exact_text")
        else:
            first[h] = (jid, dt)          # نفس النص بعد أشهر = إعادة نشر لاحقة، تُعامل كإعلان مستقل
    # 3) تقريبي داخل كتل نفس المسمى المطبّع
    for t, g in d[d.nt.str.len() >= 3].groupby("nt"):
        if len(g) < 2:
            continue
        g = g.sort_values("date")
        rows = list(g.itertuples(index=False))
        sh = {}
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                if pd.notna(a.date) and pd.notna(b.date) and (b.date - a.date).days > WINDOW_DAYS:
                    break
                if uf.find(a.job_id) == uf.find(b.job_id) or post_of(a.job_id) == post_of(b.job_id):
                    continue                                   # وظيفتان بنفس المسمى داخل المنشور نفسه = شاغران مستقلان
                oa, ob = (a.no if a.no not in NOISE_ORG else ""), (b.no if b.no not in NOISE_ORG else "")
                if oa and ob and oa != ob and oa not in ob and ob not in oa:
                    continue                                   # جهتان مختلفتان معروفتان
                if a.governorate and b.governorate and a.governorate != b.governorate:
                    continue                                   # موقعان متعارضان
                if oa and ob:
                    link(a.job_id, b.job_id, "near_same_org", 0.9)
                    continue
                sa = sh.setdefault(a.job_id, shingles(a.text))
                sb = sh.setdefault(b.job_id, shingles(b.text))
                sim = jaccard(sa, sb)
                if sim >= TEXT_SIM:
                    link(a.job_id, b.job_id, "near_text", sim)
                elif a.governorate and a.governorate == b.governorate:
                    link(a.job_id, b.job_id, "possible", sim)
    d["canonical_job_id"] = d.job_id.map(lambda j: "c_" + hashlib.md5(uf.find(j).encode()).hexdigest()[:12])
    pr = pd.DataFrame(pairs, columns=["job_id_a", "job_id_b", "evidence", "score"])
    reason = {}
    for r in pr[~pr.evidence.isin(["possible", "blocked_conflict"])].itertuples():
        reason.setdefault(r.job_id_b, r.evidence)
        reason.setdefault(r.job_id_a, r.evidence)
    possible = set(pr.loc[pr.evidence == "possible", "job_id_a"]) | set(pr.loc[pr.evidence == "possible", "job_id_b"])
    size = d.canonical_job_id.map(d.canonical_job_id.value_counts())
    d["dup_group_size"] = size
    d["dup_reason"] = d.job_id.map(reason).fillna("")
    d["dup_status"] = "unique"
    d.loc[size > 1, "dup_status"] = "duplicate"
    d.loc[(size == 1) & d.job_id.isin(possible), "dup_status"] = "possible_duplicate_review"
    return d[["job_id", "canonical_job_id", "dup_status", "dup_reason", "dup_group_size"]], pr


def pick_representative(d: pd.DataFrame, priority: dict) -> pd.Series:
    """ممثل كل مجموعة: أولوية المصدر (موقع YemenHR > تفاصيل telegra.ph > ...) ثم أطول نص. يعيد Series منطقية."""
    s = d.assign(_p=d.source_kind.map(priority).fillna(9), _l=-d.text.str.len())
    rep = s.sort_values(["canonical_job_id", "_p", "_l"]).drop_duplicates("canonical_job_id").job_id
    return d.job_id.isin(set(rep))
