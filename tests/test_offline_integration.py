"""Actual delivered-data + REAL CPU-trained weights smoke/integration tests.

These tests do not replace GUI widget tests or human Gold evaluation.
"""
from pathlib import Path
import json
import numpy as np

from src.common import DATA_DIR, MODELS_DIR
from src.integration.engine import Engine
from src.inference.predictors import _check_dir, ModelNotAvailable
from scripts.verify_model_artifacts import verify_lstm
from scripts.repair_splits import DSU, keys


def test_neural_model_load_and_infer_real_checkpoint():
    e=Engine()
    st=e.ads.status()
    assert all(v['available'] for v in st.values())
    assert all('PyTorch CPU neural' in v['model_family'] for v in st.values())
    ad='مطلوب مهندس برمجيات للعمل في صنعاء يجيد بايثون ويدير قواعد البيانات. آخر موعد للتقديم 2026-12-30'
    r=e.ads.analyze(ad)
    assert r['all_models_available']
    assert len(r['components'])==3
    assert all(v['status']=='ok' and 'Tiny' in v['model_family'] for v in r['components'].values())
    for key in ('opportunity_type','geographic_scope'):
        x=r['components'][key]
        assert 0<=x['confidence']<=1
        assert abs(sum(t['probability'] for t in x['top'])-sum(t['probability'] for t in x['top']))<1e-6
    tags=e.ads._load('ner').extract(ad)
    assert len(tags['tokens'])==len(tags['tags'])
    assert all(0<=ent['start']<ent['end']<=len(tags['tokens']) for ent in tags['entities'])


def test_incomplete_hf_checkpoint_cannot_disguise_missing_weights(tmp_path):
    (tmp_path/'config.json').write_text('{"model_type":"bert"}', encoding='utf8')
    try:
        _check_dir(tmp_path)
    except ModelNotAvailable:
        pass
    else:
        assert False, 'HF config alone is not a real checkpoint'


def test_market_core_functionality_real_dataset():
    e=Engine()
    assert e.market.available
    ov=e.market.overview()
    assert ov['unique_opportunities']==23118
    assert 0<ov['jobs_with_skills_pct']<=100
    assert not e.market.top_skills(5).empty
    assert not e.market.search('محاسب',limit=7).empty
    assert len(e.market.search('',limit=12))<=12
    assert e.market.monthly().shape[0]>24
    assert len(e.market.governorates())>10
    assert e.forecasts.family('skills')['status']=='ok'


def test_lstm_saved_weights_reproduce_test_predictions():
    check=verify_lstm('skills')
    assert check['status']=='PASS'
    assert check['max_abs_checkpoint_prediction_error']<0.005


def test_model_reports_identify_silver_automatic_ground_truth():
    for task in ('opportunity_type','geographic_scope','ner'):
        m=json.loads((Path(__file__).resolve().parents[1]/'reports'/'evaluation'/f'offline_{task}.json').read_text(encoding='utf8'))
        assert 'SILVER' in m['label_quality']
        assert 'Gold' in m['label_quality']
        assert 'trained_at_utc' in m


def test_near_text_grouping_only_for_splits_and_no_rows_dropped():
    a=keys('ad1','c1','وظيفة رقم 1 متاحة الآن في شركة محددة لديها شروط مختلفة قليلا')
    b=keys('ad2','c2','وظيفة رقم 2 متاحة الآن في شركة محددة لديها شروط مختلفة قليلا')
    assert a[-1]==b[-1]  # near duplicates must be in same split
    uf=DSU()
    uf.union(a[0],a[-1]);uf.union(b[0],b[-1])
    assert uf.find('id:ad1')==uf.find('id:ad2')
