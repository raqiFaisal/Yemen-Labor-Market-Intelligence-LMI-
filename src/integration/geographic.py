"""High-precision Arabic location-evidence reconciliation for job-ad *inference*.

A location in an advertisement is not proof of the advertised job's worksite.
This module keeps the neural model's raw decision/probabilities and can supply
an explicitly-labelled, rule-derived scope when stronger location evidence is
present. It does not pretend to recalibrate or retrain the neural model.

No third-party dependencies; intentionally independent from model loading.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_DIAC = re.compile(r"[\u064b-\u065f\u0670\u0640]")


def _norm(text: str) -> str:
    text = _DIAC.sub("", str(text or "")).lower()
    for a, b in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ٱ", "ا"), ("ى", "ي"), ("ة", "ه")):
        text = text.replace(a, b)
    text = re.sub(r"[^\w\s:\-،,.;؟?]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# Yemen governorates + distinct frequent cities; match whole words and optional 'ال'.
_GOV_NAMES: dict[str, tuple[str, ...]] = {
    "صنعاء": ("صنعاء", "صنعا", "امانه العاصمه", "العاصمه صنعاء"),
    "عدن": ("عدن",), "تعز": ("تعز",), "إب": ("اب",),
    "الحديدة": ("الحديده",), "ذمار": ("ذمار",), "البيضاء": ("البيضاء",),
    "لحج": ("لحج",), "أبين": ("ابين",), "الضالع": ("الضالع",),
    "حضرموت": ("حضرموت", "المكلا", "سيئون", "سيون"),
    "شبوة": ("شبوه",), "مأرب": ("مارب",), "صعدة": ("صعده",),
    "الجوف": ("الجوف",), "المحويت": ("المحويت",), "حجة": ("حجه",),
    "ريمة": ("ريمه",), "عمران": ("عمران",), "المهرة": ("المهره",),
    "سقطرى": ("سقطري",),
}

_FOREIGN = {
    "السعودية": ("السعوديه", "المملكه العربيه السعوديه", "الرياض", "جده", "مكه"),
    "الإمارات": ("الامارات", "دبي", "ابوظبي", "ابو ظبي", "الشارقه"),
    "قطر": ("قطر", "الدوحه"), "الكويت": ("الكويت",),
    "البحرين": ("البحرين",), "عُمان": ("سلطنه عمان", "مسقط"),
    "مصر": ("مصر", "القاهره",), "الأردن": ("الاردن", "عمان الاردن"),
    "تركيا": ("تركيا", "اسطنبول"), "ألمانيا": ("المانيا",),
    "فرنسا": ("فرنسا",), "بريطانيا": ("بريطانيا", "لندن"),
    "الولايات المتحدة": ("الولايات المتحده", "امريكا"),
}


def _contains(text: str, name: str) -> bool:
    return bool(re.search(r"(?<!\w)" + r"(?:و)?" + re.escape(name) + r"(?!\w)", text))


def _locations(txt: str):
    local = sorted(n for n, aliases in _GOV_NAMES.items() if any(_contains(txt, alias) for alias in aliases))
    foreign = sorted(n for n, aliases in _FOREIGN.items() if any(_contains(txt, alias) for alias in aliases))
    yemen = bool(re.search(r"(?<!\w)(?:اليمن|يمني|اليمنيه|جمهوريه اليمن)(?!\w)", txt))
    # Do not mistake "خارج اليمن" for a Yemen worksite.
    outside = bool(re.search(r"(?:خارج|بره|برا)\s+(?:اليمن|البلاد)|العمل\s+في\s+الخارج", txt))
    return local, foreign, yemen, outside


_REMOTE = re.compile(r"(?<!\w)(?:عن\s+بعد|اونلاين|اون\s+لاين|remote|work\s+from\s+home)(?!\w)")
_WORK = re.compile(
    r"(?:للعمل|العمل|مكان\s+العمل|مقر\s+العمل|موقع\s+العمل|"
    r"مكان\s+الوظيفه|موقع\s+الوظيفه|الوظيفه|وظائف\s+شاغره|فرص\s+عمل|فرصه\s+عمل)\s*(?:هو\s+)?(?:في|ب|:)\s*"
)
_ALL_YEMEN = re.compile(r"(?:جميع|كافه|مختلف)\s+محافظات\s+اليمن|(?:في\s+)?(?:كافه|جميع)\s+انحاء\s+اليمن|على\s+مستوي\s+اليمن")


@dataclass(frozen=True)
class Evidence:
    label: str
    strength: str  # explicit_worksite / inferred_context
    places: tuple[str, ...]
    explanation: str


def geographic_evidence(text: str) -> Evidence | None:
    """Conservative heuristic: location evidence, **not** a trained prediction.

    Explicit worksite beats headquarters/nationality mentions. If multiple
    opposing worksites are asserted, abstain rather than force a label.
    """
    t = _norm(text)
    if not t:
        return None
    # Strong remote work cue. Negated cue does not count as an offered remote job.
    remote = bool(_REMOTE.search(t)) and not re.search(r"(?:ليس|غير|لا\s+يتوفر|ممنوع)\s+(?:العمل\s+)?عن\s+بعد", t)
    # Remote must describe THIS opportunity, not just a technology, benefit, or
    # another job mentioned deep in the ad's body.
    remote_header = bool(re.search(r"(?:عمل|وظيفه|وظائف|دوام|فرصه)\s+(?:جديده\s+)?(?:عن\s+بعد|اونلاين|اون\s+لاين|remote)", t[:170]))
    remote_worksite = bool(re.search(r"(?:موقع|مكان|نظام|طبيعه)\s+العمل\s*(?::|هو|ب)?\s*(?:عن\s+بعد|اونلاين|remote)", t))
    if remote and (remote_header or remote_worksite):
        return Evidence("REMOTE", "explicit_worksite", ("عن بُعد",), "الإعلان يحدد أن العمل عن بُعد؛ مكان إقامة المتقدم ليس موقع الوظيفة.")

    # Search only near explicit worksite phrases; other place mentions may be
    # company headquarters, application addresses, sponsors or nationalities.
    worksites: list[tuple[list[str], list[str], bool, bool]] = []
    for m in _WORK.finditer(t):
        # Restrict to a short clause; avoid dragging in a later application address.
        seg = re.split(r"[.;،,؟?]|(?:\s+)(?:بينما|لكن|والتقديم|التقديم|المقابلات|الشركه\s+مقرها)", t[m.end():m.end()+75], maxsplit=1)[0]
        loc, out, yem, outside = _locations(seg)
        if loc or out or yem or outside:
            worksites.append((loc, out, yem, outside))
    # Explicit outside-Yemen work opportunity without a "في"-phrase.
    if re.search(r"(?:وظائف|فرص|فرصه|عمل|وظيفه)\s+(?:متاحه\s+)?خارج\s+اليمن", t):
        return Evidence("INTERNATIONAL", "explicit_worksite", ("خارج اليمن",), "الإعلان يصف الفرصة بأنها خارج اليمن.")
    if worksites:
        local = sorted({x for ws in worksites for x in ws[0]})
        foreign = sorted({x for ws in worksites for x in ws[1]})
        yemen = any(ws[2] for ws in worksites)
        outside = any(ws[3] for ws in worksites)
        if (local or yemen) and (foreign or outside):
            return None  # conflicting worksites: ask for review / keep model
        if foreign or outside:
            return Evidence("INTERNATIONAL", "explicit_worksite", tuple(foreign) or ("خارج اليمن",), "ذُكر موقع عمل خارج اليمن صراحةً.")
        if _ALL_YEMEN.search(t):
            return Evidence("NATIONWIDE", "explicit_worksite", ("جميع محافظات اليمن",), "نطاق العمل مصرح به على مستوى اليمن.")
        if len(local) > 1:
            return Evidence("MULTI_GOVERNORATE", "explicit_worksite", tuple(local), "ورد أكثر من موقع عمل داخل اليمن.")
        if len(local) == 1:
            return Evidence("SINGLE_GOVERNORATE", "explicit_worksite", tuple(local), "ذُكر موقع عمل في محافظة يمنية.")
        if yemen:
            return Evidence("YEMEN_UNSPECIFIED", "explicit_worksite", ("اليمن",), "ذُكر العمل داخل اليمن دون محافظة محددة.")

    local, foreign, yem, outside = _locations(t)
    if outside:
        # A bare 'خارج اليمن' can describe an applicant's residence; don't force.
        return None
    if foreign:
        # Mention of an international office doesn't establish worksite.
        return None
    if _ALL_YEMEN.search(t):
        return Evidence("NATIONWIDE", "inferred_context", ("جميع محافظات اليمن",), "ذُكر نطاق يشمل جميع المحافظات، دون تحديد صريح لمقر العمل.")
    if len(local) == 1:
        return Evidence("SINGLE_GOVERNORATE", "inferred_context", tuple(local), "ورد ذكر محافظة يمنية في الإعلان؛ قد تكون مقر الجهة لا موقع العمل.")
    if len(local) > 1:
        return None  # several city mentions may be contact office addresses
    if yem:
        return Evidence("YEMEN_UNSPECIFIED", "inferred_context", ("اليمن",), "ورد ذكر اليمن؛ لم يُثبت أن الموقع هو مقر العمل.")
    return None


def reconcile_geographic_prediction(text: str, prediction: dict) -> dict:
    """Keep model posterior unchanged and record any evidence-based display change.

    The evidence is NOT calibrated; when using it confidence=None rather than
    misrepresenting the neural probability as probability for the new label.
    """
    out = dict(prediction)
    evidence = geographic_evidence(text)
    out["model_prediction"] = {
        "label": prediction.get("label"), "confidence": prediction.get("confidence"),
        "top": prediction.get("top", []),
    }
    out["decision_source"] = "neural_model"
    out["evidence"] = None
    out["override"] = False
    if evidence is not None:
        out["evidence"] = {
            "suggested_label": evidence.label,
            "strength": evidence.strength,
            "places": list(evidence.places),
            "explanation": evidence.explanation,
        }
        original = prediction.get("label")
        domestic = {"LOCAL", "SINGLE_GOVERNORATE", "MULTI_GOVERNORATE", "NATIONWIDE", "YEMEN_UNSPECIFIED"}
        # Never replace a domestic subcategory with another domestic subcategory.
        # LOCAL vs SINGLE_GOVERNORATE is especially hard to infer from prose,
        # and was empirically harmed by blanket overrides on the Silver test set.
        # Worksite evidence is used only for major boundary errors.
        boundary_conflict = (
            (original == "INTERNATIONAL" and evidence.label in domestic) or
            (original in domestic and evidence.label == "INTERNATIONAL" and evidence.strength == "explicit_worksite") or
            (original == "REMOTE" and evidence.label in domestic and evidence.strength == "explicit_worksite") or
            (original in domestic and evidence.label == "REMOTE" and evidence.strength == "explicit_worksite") or
            (original == "INTERNATIONAL" and evidence.label == "REMOTE" and evidence.strength == "explicit_worksite") or
            (original == "REMOTE" and evidence.label == "INTERNATIONAL" and evidence.strength == "explicit_worksite")
        )
        if boundary_conflict:
            # No fake confidence: UI explicitly states that this is a rule-based
            # evidence reconciliation, not the model's predicted posterior.
            out["label"] = evidence.label
            out["confidence"] = None
            out["decision_source"] = "location_evidence"
            out["override"] = True
    return out
