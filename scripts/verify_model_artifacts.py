"""Numerical verification of shipped neural weight files against saved output.

This checks weights genuinely produce recorded predictions and test metrics;
it CANNOT prove the originally reported training history or human Gold quality.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

from src.common import MODELS_DIR, REPORTS_DIR, write_json
from src.evaluation.metrics import forecast_metrics
from src.forecasting.train_lstm import make_model


def verify_lstm(family: str) -> dict:
    base = MODELS_DIR / 'lstm' / family
    rep = REPORTS_DIR / 'evaluation' / f'lstm_{family}'
    config = json.loads((base/'config.json').read_text(encoding='utf8'))
    scaler = json.loads((base/'scaler.json').read_text(encoding='utf8'))
    historic = json.loads((rep/'metrics.json').read_text(encoding='utf8'))
    frame = pd.read_csv(rep/'series_used.csv', index_col=0)
    predictions = pd.read_csv(rep/'test_predictions.csv')
    items = scaler['items']; mu = np.asarray(scaler['mean'],dtype=float)
    sd = np.asarray(scaler['std'],dtype=float)
    if list(frame.columns) != items:
        raise ValueError(f'{family} series and scaler items not in same order')
    model = make_model(config['hidden_size'],1,config['dropout'])
    model.load_state_dict(torch.load(base/'lstm.pt',weights_only=True,map_location='cpu'))
    model.eval()
    x = frame.ffill().fillna(0).to_numpy(dtype=float)
    z = (x-mu)/sd
    lb = int(config['lookback'])
    lookup = {str(k):i for i,k in enumerate(frame.index)}
    item_pos = {x:i for i,x in enumerate(items)}
    input_x=[]
    for row in predictions.itertuples():
        t=lookup[str(row.month)]; s=item_pos[str(row.item)]
        if t < lb:
            raise ValueError('test window reaches before lookback')
        input_x.append(z[t-lb:t,s])
    xx=torch.tensor(np.array(input_x,dtype=np.float32)[...,None])
    with torch.inference_mode():
        out=[]
        for part in xx.split(200):
            out.extend(model(part).numpy())
    expected=np.asarray(out,dtype=float)*np.array([sd[item_pos[i]] for i in predictions.item])+np.array([mu[item_pos[i]] for i in predictions.item])
    expected=np.clip(expected,0,None)
    error=np.abs(expected-predictions.lstm.to_numpy(dtype=float))
    recorded={k:forecast_metrics(predictions.actual,predictions[k]) for k in ('lstm','naive','moving_avg_3')}
    comparisons={k: {field: (recorded[k].get(field), historic['test'][k].get(field)) for field in ('n','mae','rmse','mape')}
                 for k in recorded}
    metric_mismatch=[f'{k}:{field}' for k, pairs in comparisons.items() for field,(a,b) in pairs.items() if a!=b]
    if error.max()>0.005:
        raise ValueError(f'{family}: checkpoint does not reproduce test predictions (max discrepancy {error.max():.6g})')
    if metric_mismatch:
        raise ValueError(f'{family}: saved test metrics not reproduced by saved predictions: {metric_mismatch}')
    return {'family': family,'status':'PASS', 'n_predictions':len(predictions),
            'max_abs_checkpoint_prediction_error':round(float(error.max()),6),
            'mean_abs_checkpoint_prediction_error':round(float(error.mean()),6),
            'test_metrics_recomputed_from_saved_predictions':recorded,
            'caveat':'Not proof of proper training, Gold evaluation, or absence of all time-series bias.'}


def verify_offline() -> list[dict]:
    from src.offline.neural import OfflineClassifier, OfflineNERTagger
    out=[]
    for task in ('opportunity_type','geographic_scope'):
        path=MODELS_DIR/'offline'/'classification'/task
        model=OfflineClassifier(path)
        probs=model.predict_proba(['مطلوب مهندس برمجيات','ندوة تدريبية في البرمجة'])
        assert probs.shape==(2,len(model.labels)) and np.isfinite(probs).all()
        assert np.allclose(probs.sum(axis=1),1,atol=1e-6)
        out.append({'model':'offline_'+task,'status':'PASS','weights_loaded':True,'test_results':str(REPORTS_DIR/'evaluation'/f'offline_{task}.json')})
    ner=OfflineNERTagger(MODELS_DIR/'offline'/'ner')
    sample=ner.extract('مطلوب محاسب في عدن يجيد Excel ويملك خبرة ثلاث سنوات')
    assert len(sample['tokens'])==len(sample['tags'])
    out.append({'model':'offline_ner','status':'PASS','weights_loaded':True,'n_entities_in_smoke_sample':len(sample['entities'])})
    return out


def main():
    data = {'lstm':{},'offline':[],'status':'PASS','meaning':'numerical/functional artifact verification, NOT human Gold'}
    try:
        for fam in ('skills','specializations','job_titles','governorates'):
            data['lstm'][fam]=verify_lstm(fam)
            print('LSTM',fam,'PASS',data['lstm'][fam]['max_abs_checkpoint_prediction_error'])
        data['offline']=verify_offline()
    except Exception as exc:
        data['status']='FAIL'
        data['error']=type(exc).__name__+': '+str(exc)
        print(data['error'])
    write_json(REPORTS_DIR/'repair'/'model_verification.json',data)
    return 1 if data['status']=='FAIL' else 0

if __name__=='__main__':
    raise SystemExit(main())
