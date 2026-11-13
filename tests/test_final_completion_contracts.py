"""Reproducibility/critical holdout guards; no external libraries beyond project CPU deps."""
import json
from datetime import date
import numpy as np
import pandas as pd
import pytest
from src.common import ROOT, DATA_DIR, MODELS_DIR, REPORTS_DIR
from src.forecasting.series import build_family, load_unique
from src.evaluation.metrics import forecast_metrics
from scripts.gold_holdout import check_gold
from scripts.evaluate_gold_strict import exact_span_metrics

FAMS=('skills','specializations','job_titles','governorates')

def test_gold_not_automatically_approved():
    s=check_gold()
    assert s['errors']==[]
    # Current release has no human labels; do not hardcode a future invariant
    # that would fail as soon as real annotators finish the work.
    j=json.loads((REPORTS_DIR/'evaluation'/'human_gold_status.json').read_text())
    if s['classification_verified']==0 and s['ner_verified']==0:
        assert s['ready_for_independent_gold_eval'] is False
        assert j['status']=='BLOCKED_PENDING_HUMAN_ANNOTATION'

def test_strict_gold_span_metric_exact_boundaries():
    gold=[{'start':0,'end':10,'label':'SKILL'}, {'start':20,'end':25,'label':'LOCATION'}]
    pred=[{'start':0,'end':9,'label':'SKILL'},{'start':20,'end':25,'label':'LOCATION'}]
    m=exact_span_metrics(gold,pred)
    assert (m['tp'],m['fp'],m['fn'])==(1,1,1)
    assert m['f1']==pytest.approx(0.5)

def test_current_month_excluded_from_temporal_data():
    u=load_unique(as_of=date(2026,11,5))
    assert u.month.max()==pd.Period('2026-10')
    assert not (u.month==pd.Period('2026-11')).any()
    assert not u.empty

@pytest.mark.parametrize('family',FAMS)
def test_train_only_feature_selection_and_temporal_splits(family):
    fam=build_family(family,as_of=date(2026,11,5))
    assert fam['status']=='ok'
    S=fam['series']
    cutoff=pd.Period(fam['train_feature_cutoff'],freq='M')
    assert S.index[-1]==pd.Period('2026-10')
    assert cutoff < pd.Period('2026-05')
    trainidx=S.index<=cutoff
    assert trainidx.sum()+9==len(S)
    assert len(fam['ready'])>0
    assert set(fam['ready'])==set(S.columns)
    suff=fam['sufficiency'].set_index('item')
    for it in fam['ready']:
        assert suff.loc[it,'train_mentions']>=25
        assert suff.loc[it,'train_coverage']>=0.8
        assert np.isfinite(S[it].loc[:cutoff]).sum()>0

@pytest.mark.parametrize('family',FAMS)
def test_saved_temporal_predictions_reproduce_metrics_and_dates(family):
    folder=REPORTS_DIR/'evaluation'/f'lstm_{family}'
    met=json.loads((folder/'metrics.json').read_text())
    cfg=json.loads((MODELS_DIR/'lstm'/family/'config.json').read_text())
    df=pd.read_csv(folder/'test_predictions.csv')
    assert met['trained'] is True and met['as_of']=='2026-11-05'
    assert cfg['feature_selection_cutoff'] == cfg['train_period'][1]
    assert cfg['excluded_incomplete_month']=='2026-11'
    assert met['test_period']==['2026-05','2026-10']
    assert str(df.month.min())=='2026-05' and str(df.month.max())=='2026-10'
    assert met['test_used_during_training'] is False
    assert not df[['lstm','naive','moving_avg_3']].isna().any().any()
    for name in ('lstm','naive','moving_avg_3'):
        actual=forecast_metrics(df.actual,df[name])
        assert actual['mae']==met['test'][name]['mae']
        assert actual['rmse']==met['test'][name]['rmse']
        assert actual['n']==met['test'][name]['n']
    assert cfg['test_evaluation_mode'].startswith('One-step')

def test_ui_page_logic_all_scenarios():
    from scripts.ui_logic_smoke import run
    report=run()
    assert report['total']==19
    assert report['all_passed']
