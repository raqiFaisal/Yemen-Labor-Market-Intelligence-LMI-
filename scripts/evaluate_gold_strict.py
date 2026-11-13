"""Evaluate only completed, human-verified, independent Gold records.

NO SILVER labels are ever substituted. A blocked JSON status is written and a
nonzero exit returned if no Gold exists. Test set is never used for training.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import Counter
import pandas as pd
from src.common import DATA_DIR, MODELS_DIR, REPORTS_DIR, write_json
from scripts.gold_holdout import check_gold, _jsonl
from src.evaluation.metrics import classification_metrics

SCOPE_COARSE={'LOCAL':'YEMEN', 'SINGLE_GOVERNORATE':'YEMEN','MULTI_GOVERNORATE':'YEMEN',
              'NATIONWIDE':'YEMEN','YEMEN_UNSPECIFIED':'YEMEN', 'INTERNATIONAL':'OUTSIDE_YEMEN',
              'REMOTE':'REMOTE_GLOBAL','UNKNOWN':'UNKNOWN'}
LABEL_COARSE={'ORGANIZATION':'ORGANIZATION','DEADLINE':'DATE'}

def exact_span_metrics(gold, pred):
    """Micro exact character span/type matches (start,end,label)."""
    G={ (int(s['start']),int(s['end']),s['label']) for s in gold}
    P={ (int(s['start']),int(s['end']),s['label']) for s in pred}
    tp=len(G & P);fp=len(P-G);fn=len(G-P)
    precision=tp/(tp+fp) if tp+fp else 0.0
    recall=tp/(tp+fn) if tp+fn else 0.0
    return {'tp':tp,'fp':fp,'fn':fn,'precision':precision,'recall':recall,
            'f1':2*precision*recall/(precision+recall) if precision+recall else 0.0}

def tokens_to_spans(text, entities):
    """Convert model word indexes to original text character offsets."""
    from src.inference.predictors import TOKEN_RX
    offsets=[m.span() for m in TOKEN_RX.finditer(text)]
    out=[]
    for e in entities:
        a,b=e['start'],e['end']
        if 0<=a<b<=len(offsets):
            k=LABEL_COARSE.get(e['label'],e['label'])
            out.append({'start':offsets[a][0],'end':offsets[b-1][1],'label':k})
    return out

def evaluate(data=DATA_DIR, models=MODELS_DIR, reports=REPORTS_DIR):
    data=Path(data); reports=Path(reports)
    check=check_gold(data)
    status={'label_quality':'HUMAN_GOLD_ONLY', 'annotation_integrity':check,
            'independent_gold':'Pre-existing split TEST, checked non-overlap against train/validation IDs and canonical groups',
            'status':'BLOCKED_PENDING_HUMAN_ANNOTATION'}
    dest=reports/'evaluation'/'human_gold_status.json'
    if check['errors']:
        status['status']='BLOCKED_INVALID_GOLD';write_json(dest,status);return status
    if not check['ready_for_independent_gold_eval']:
        write_json(dest,status);return status
    from src.integration.engine import Engine
    eng=Engine(data,models,reports)
    c=pd.read_csv(data/'gold_annotation'/'opportunity_gold_holdout_v3.csv',dtype=str,keep_default_na=False)
    c=c[c.review_status.str.lower()=='completed']
    yn={'opportunity_type':([],[]), 'geographic_scope':([],[])}
    for r in c.itertuples():
        pred=eng.analyze_ad(r.text)['components']
        for task,truth in [('opportunity_type',r.gold_opportunity_type),('geographic_scope',SCOPE_COARSE[r.gold_geographic_scope])]:
            if pred[task].get('status')!='ok': raise RuntimeError(f'{task} model unavailable for Gold eval')
            yn[task][0].append(truth);yn[task][1].append(pred[task]['label'])
    score={}
    for task,(gold,pred) in yn.items():
        classes=sorted(set(gold)|set(pred))
        score[task]=classification_metrics(gold,pred,classes)
        score[task]['ontology']='coarse geographic mapping; no fine-grained geographic claim' if task=='geographic_scope' else 'original type taxonomy'
    n=[x for x in _jsonl(data/'gold_annotation'/'ner_gold_holdout_v3.jsonl') if x.get('review_status')=='completed']
    summary=Counter(); kind={k:Counter() for k in ('JOB_TITLE','SKILL','EDUCATION','DEGREE','MAJOR','EXPERIENCE','LOCATION','ORGANIZATION','SALARY','DATE')}
    for r in n:
        pred=eng.analyze_ad(r['text'])['components']['ner']
        if pred.get('status')!='ok':raise RuntimeError('NER model unavailable')
        predicted=tokens_to_spans(r['text'],pred['entities'])
        e=exact_span_metrics(r['gold_spans'],predicted)
        summary.update({k:e[k] for k in ('tp','fp','fn')})
        for k in kind:
            sk=[s for s in r['gold_spans'] if s['label']==k]
            sp=[s for s in predicted if s['label']==k]
            ek=exact_span_metrics(sk,sp)
            kind[k].update({v:ek[v] for v in ('tp','fp','fn')})
    def from_counts(x):
        tp,fp,fn=(x.get(k,0) for k in ('tp','fp','fn'))
        pp=tp/(tp+fp) if tp+fp else 0; rr=tp/(tp+fn) if tp+fn else 0
        return {'tp':tp,'fp':fp,'fn':fn,'entity_precision':round(pp,4),'entity_recall':round(rr,4),
                'entity_f1':round(2*pp*rr/(pp+rr),4) if pp+rr else 0.0}
    score['ner']={'n_documents':len(n),'span_policy':'exact character offsets and exact label',**from_counts(summary),
                  'by_entity':{k:from_counts(v) for k,v in kind.items()},
                  'limitations':['DATE maps old DEADLINE subtype to DATE; broader DATE and EDUCATION not trained']}
    status.update(status='EVALUATED_ON_HUMAN_GOLD',results=score)
    write_json(dest,status)
    return status

def main():
    args=argparse.ArgumentParser(); args.add_argument('--data-dir',type=Path,default=DATA_DIR);a=args.parse_args()
    status=evaluate(data=a.data_dir)
    print(json.dumps({'status':status['status'],'review':status['annotation_integrity']},ensure_ascii=False,indent=2))
    if status['status']!='EVALUATED_ON_HUMAN_GOLD':raise SystemExit(2)
if __name__=='__main__':main()
