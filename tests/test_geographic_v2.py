"""Meaningful integration tests for a real trained checkpoint (not a rule override)."""
from pathlib import Path
import json

import numpy as np
import pytest

from src.offline.geographic_v2 import GeoScopeClassifier, LABELS, geo_features
from src.integration.engine import AdAnalyzer
from src.common import MODELS_DIR, REPORTS_DIR

MODEL_DIR=MODELS_DIR/'offline'/'classification'/'geographic_scope_v2'

@pytest.mark.parametrize('ad,expected', [
 ('تعلن إحدى الشركات في صنعاء عن حاجتها إلى مهندس برمجيات يمتلك خبرة ثلاث سنوات في Python وSQL.','YEMEN'),
 ('فرصة عمل في اليمن لمهندس مدني','YEMEN'),
 ('مطلوب ممرضة للعمل في عدن في مستشفى خاص','YEMEN'),
 ('مطلوب محاسب للعمل في الرياض لدى شركة','OUTSIDE_YEMEN'),
 ('مطلوب مهندس للعمل في دبي في الإمارات','OUTSIDE_YEMEN'),
 ('مقر الشركة في صنعاء لكن مكان العمل في الرياض','OUTSIDE_YEMEN'),
 ('مقر الشركة في الرياض لكن مكان العمل في صنعاء','YEMEN'),
 ('تعلن شركة في صنعاء عن فرصة عمل عن بعد لمطور تطبيقات','REMOTE'),
 ('وظيفة مطور نظم عن بعد','REMOTE'),
])
def test_real_neural_inference(ad,expected):
    r=GeoScopeClassifier(MODEL_DIR).predict(ad)
    assert r['decision_source']=='trained_neural_model'
    assert r['label']==expected
    assert 0<=r['confidence']<=1
    assert abs(sum(x['probability'] for x in r['top'])-1)<1e-5


def test_end_to_end_ad_analyzer():
    ad=AdAnalyzer(models_dir=MODELS_DIR)
    status=ad.status()['geographic_scope']
    assert status['available'] and 'GeoScopeNetV2' in status['model_family']
    r=ad.analyze('فرصة عمل لمحاسب في الرياض بخبرة ثلاث سنوات')['components']['geographic_scope']
    assert r['label']=='OUTSIDE_YEMEN'
    assert r['decision_source']=='trained_neural_model'
    assert 'override' not in r and 'model_prediction' not in r


def test_location_feature_not_deterministic_override():
    a=geo_features('فرصة عمل في عدن')
    b=geo_features('فرصة عمل في دبي')
    assert a.shape==b.shape and not np.array_equal(a,b)


def test_metrics_real_silver_not_gold():
    metrics=json.loads((REPORTS_DIR/'evaluation'/'geographic_scope_v2.json').read_text(encoding='utf8'))
    assert metrics['n']==2575
    assert metrics['label_quality'].startswith('SILVER')
    assert metrics['scenario_count']>=9
    assert metrics['scenario_passed'] == metrics['scenario_count']


def test_banners_removed():
    p=Path(__file__).resolve().parents[1]/'app_dl'/'views'
    a=(p/'analyze.py').read_text(encoding='utf8')
    f=(p/'forecast.py').read_text(encoding='utf8')
    assert 'هذه النتائج ناتجة عن نماذج PyTorch صغيرة' not in a
    assert 'نموذج LSTM لم يتفوق على خط الأساس Naive' not in f
    assert 'st.error("نموذج LSTM' not in f
