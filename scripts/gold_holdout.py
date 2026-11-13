"""Build independent human annotation queues, never invent gold labels.

Samples ONLY pre-existing held-out test partitions, avoids IDs and canonical
opportunity groups in training/validation across all matching task splits.
Original queues are never changed. Use --check to validate human-filled files.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from src.common import DATA_DIR

ENTITIES = {"JOB_TITLE", "SKILL", "EDUCATION", "DEGREE", "MAJOR", "EXPERIENCE", "LOCATION", "ORGANIZATION", "SALARY", "DATE"}
TYPE = {"JOB", "INTERNSHIP", "TRAINING", "SCHOLARSHIP", "VOLUNTEERING", "OTHER_OPPORTUNITY", "NOT_RELEVANT"}
SCOPE = {"LOCAL", "SINGLE_GOVERNORATE", "MULTI_GOVERNORATE", "NATIONWIDE", "YEMEN_UNSPECIFIED", "INTERNATIONAL", "REMOTE", "UNKNOWN"}

def _jsonl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]

def make(data=DATA_DIR, n_class=246, n_ner=100):
    data = Path(data)
    gold = data / 'gold_annotation'; gold.mkdir(exist_ok=True)
    cpath=gold/'opportunity_gold_holdout_v3.csv'
    npath=gold/'ner_gold_holdout_v3.jsonl'
    # Never overwrite a human annotation queue, even if some rows remain pending.
    if cpath.exists() or npath.exists():
        raise FileExistsError('Gold v3 files exist: refusing to overwrite any human work. Use --check.')
    ct = data / 'classification_opportunity_type'
    train_ids=set()
    train_groups=set()
    for task in ('classification_opportunity_type','classification_geographic_scope'):
        for split in ('train', 'validation'):
            df=pd.read_csv(data / task / f'{split}.csv', dtype=str, keep_default_na=False)
            train_ids.update(df.id)
            train_groups.update(x for x in df.get('group',[]) if x)
    for split in ('train','validation'):
        for row in _jsonl(data/'ner'/f'{split}.jsonl'):
            train_ids.add(row['id'])
            if row.get('canonical_job_id'):train_groups.add(row['canonical_job_id'])
    d=pd.read_csv(ct/'test.csv',dtype=str,keep_default_na=False)
    d=d[~d.id.isin(train_ids) & ~d.group.isin(train_groups)].drop_duplicates('id').copy()
    # deterministic stratified review sample on minority classes; no cherry-picked scores
    chunks=[]
    for lab,g in d.groupby('label'):
        chunks.append(g.sample(n=min(len(g),max(15,n_class//max(1,d.label.nunique()))),random_state=42))
    s=pd.concat(chunks).drop_duplicates('id')
    rest=d[~d.id.isin(s.id)]
    if len(s)<n_class:s=pd.concat([s,rest.sample(n=min(len(rest), n_class-len(s)), random_state=42)])
    s=s.head(n_class)
    unique=pd.read_csv(data/'unique_opportunities.csv',dtype=str,keep_default_na=False)
    lookup=unique.drop_duplicates('post_id').set_index('post_id')
    rec=[]
    for r in s.itertuples():
        meta=lookup.loc[r.id] if r.id in lookup.index else None
        rec.append({'post_id':r.id,'source_channel':r.source_channel,'source_url':str(meta.source_url) if meta is not None else '',
                    'text':r.text,'canonical_group':r.group,'gold_opportunity_type':'', 'gold_geographic_scope':'',
                    'annotator':'','reviewed_at':'','review_status':'pending','notes':''})
    pd.DataFrame(rec).to_csv(cpath,index=False,encoding='utf-8-sig')
    ner_train_ids=set(train_ids)
    nr=_jsonl(data/'ner'/'test.jsonl')
    nr=[r for r in nr if r['id'] not in ner_train_ids and r.get('canonical_job_id') not in train_groups]
    # Deterministic systematic sampling from independent held-out NER examples
    import random
    random.Random(42).shuffle(nr)
    with npath.open('w',encoding='utf-8') as fp:
        for r in nr[:n_ner]:
            row={'id':r['id'],'canonical_job_id':r.get('canonical_job_id'), 'text':r['text'],
                 'gold_spans':None,'review_status':'pending',
                 'annotator':'','reviewed_at':'','second_annotator':'','disagreement_note':''}
            fp.write(json.dumps(row,ensure_ascii=False)+'\n')
    status={'classification_review_count':len(rec),'ner_review_count':min(n_ner,len(nr)),
            'source':'Existing heldout test only; IDs and canonical groups filtered against training and validation; blind annotator queues WITHOUT silver hints',
            'classification_gold_completed':0,'ner_gold_completed':0,
            'human_review':'PENDING','valid_for_gold_evaluation':False}
    (gold/'holdout_status_v3.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    return status


def check_gold(data=DATA_DIR):
    data=Path(data); folder=data/'gold_annotation'
    c=pd.read_csv(folder/'opportunity_gold_holdout_v3.csv',dtype=str,keep_default_na=False)
    n=_jsonl(folder/'ner_gold_holdout_v3.jsonl')
    errors=[]; class_verified=[]; ner_verified=[]
    if c.post_id.duplicated().any():errors.append('duplicate classification review post_id')
    if len({r['id'] for r in n})!=len(n):errors.append('duplicate NER review id')
    trids=set(); trgroups=set()
    for task in ('classification_opportunity_type','classification_geographic_scope'):
        for split in ('train','validation'):
            f=pd.read_csv(data/task/f'{split}.csv',dtype=str,keep_default_na=False)
            trids.update(f.id); trgroups.update(x for x in f.group if x)
    for split in ('train','validation'):
        for r in _jsonl(data/'ner'/f'{split}.jsonl'):
            trids.add(r['id'])
            if r.get('canonical_job_id'):trgroups.add(r['canonical_job_id'])
    for r in c.to_dict(orient='records'):
        if r['post_id'] in trids or r.get('canonical_group') in trgroups:errors.append(f"classification train/group overlap: {r['post_id']}")
        if r['review_status'].strip().lower()!='completed':continue
        if not r['annotator'].strip() or not r['reviewed_at'].strip():errors.append(f"classification missing reviewer/date: {r['post_id']}");continue
        from datetime import datetime
        try: datetime.fromisoformat(r['reviewed_at'])
        except ValueError: errors.append(f"classification invalid reviewed_at: {r['post_id']}");continue
        if r['gold_opportunity_type'] not in TYPE or r['gold_geographic_scope'] not in SCOPE:
            errors.append(f"invalid classification gold class: {r['post_id']}");continue
        class_verified.append(r)
    for r in n:
        if r['id'] in trids or r.get('canonical_job_id') in trgroups:errors.append(f"NER train overlap: {r['id']}")
        if r.get('review_status','').strip().lower()!='completed':continue
        if not r.get('annotator') or not r.get('reviewed_at'): errors.append(f"NER missing reviewer/date: {r['id']}");continue
        from datetime import datetime
        try:datetime.fromisoformat(r['reviewed_at'])
        except ValueError:errors.append(f"NER invalid reviewed_at: {r['id']}");continue
        spans=r.get('gold_spans')
        if not isinstance(spans,list): errors.append(f"NER missing Gold spans list: {r['id']}");continue
        prevend=0;valid=True
        for span in sorted(spans,key=lambda x:x.get('start',-1)):
            a,b,k=span.get('start'),span.get('end'),span.get('label')
            if not isinstance(a,int) or not isinstance(b,int) or a<prevend or a<0 or b<=a or b>len(r['text']) or k not in ENTITIES:
                errors.append(f"invalid NER span {r['id']} {span}");valid=False;break
            prevend=b
        if valid: ner_verified.append(r)
    result={'classification_pending':len(c)-len(class_verified),'classification_verified':len(class_verified),
            'ner_pending':len(n)-len(ner_verified),'ner_verified':len(ner_verified),
            'errors':errors[:100],'ready_for_independent_gold_eval':not errors and len(c)>0 and len(n)>0 and len(class_verified)==len(c) and len(ner_verified)==len(n)}
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');args=ap.parse_args()
    print(json.dumps(check_gold() if args.check else make(),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
