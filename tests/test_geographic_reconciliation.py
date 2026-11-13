"""Geographic text-evidence correction; no invented confidence or NER retraining."""
from src.integration.geographic import geographic_evidence, reconcile_geographic_prediction


EXAMPLES = [
    ("تعلن إحدى الشركات في صنعاء عن حاجتها إلى مهندس برمجيات يجيد Python وSQL", "SINGLE_GOVERNORATE"),
    ("مطلوب مهندس للعمل في صنعاء يجيد بايثون", "SINGLE_GOVERNORATE"),
    ("فرص عمل في اليمن لمهندسي البرمجيات", "YEMEN_UNSPECIFIED"),
    ("وظائف للعمل في عدن وتعز ضمن مشروع إنساني", "MULTI_GOVERNORATE"),
    ("مطلوب مدير للعمل في صنعاء وعدن", "MULTI_GOVERNORATE"),
    ("مطلوب منسق للعمل في جميع محافظات اليمن", "NATIONWIDE"),
    ("إعلان من صنعاء عن وظائف للعمل في الرياض", "INTERNATIONAL"),
    ("الوظائف متاحة خارج اليمن، للاستفسار يرجى مراجعة المكتب في صنعاء", "INTERNATIONAL"),
    ("وظيفة في صنعاء: العمل عن بعد من المنزل", "REMOTE"),
    ("مطلوب مطور للعمل في صنعاء، التقديم في السعودية", "SINGLE_GOVERNORATE"),
    ("التقديم في صنعاء لوظيفة مدير للعمل في دبي", "INTERNATIONAL"),
]


def test_location_evidence_examples():
    for sample, expected in EXAMPLES:
        e = geographic_evidence(sample)
        assert e is not None and e.label == expected, (sample, e, expected)


def test_confidence_not_transferred_to_changed_label():
    raw = {"label": "INTERNATIONAL", "confidence": 0.95,
           "top": [{"label": "INTERNATIONAL", "probability": 0.95}]}
    result = reconcile_geographic_prediction(EXAMPLES[0][0], raw)
    assert result["label"] == "SINGLE_GOVERNORATE"
    assert result["confidence"] is None
    assert result["decision_source"] == "location_evidence"
    assert result["model_prediction"]["label"] == "INTERNATIONAL"
    assert result["model_prediction"]["confidence"] == 0.95
    assert result["top"] == raw["top"]
    assert result["override"] is True


def test_model_unchanged_for_uninformative_ad():
    raw = {"label": "INTERNATIONAL", "confidence": 0.62, "top": []}
    result = reconcile_geographic_prediction("مطلوب مهندس شبكات لديه خبرة", raw)
    assert result["label"] == "INTERNATIONAL"
    assert result["confidence"] == 0.62
    assert not result["override"]


def test_negative_and_ambiguous_cases_abstain():
    for sample in [
        "وظيفة لمترجم يجيد العربية والإنجليزية بدون ذكر موقع العمل",
        "تعلن جهة عن عمل في صنعاء ومطلوب أيضاً للعمل في قطر",
        "شركة في الرياض ولها مقر في صنعاء تطلب موظفين دون تحديد مكان العمل",
    ]:
        # Contradictory/ambiguous locations should not force a false decision.
        result = reconcile_geographic_prediction(sample, {"label": "INTERNATIONAL", "confidence": .75})
        assert not result["override"], (sample, result)
