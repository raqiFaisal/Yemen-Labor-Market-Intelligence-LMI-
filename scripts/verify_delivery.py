"""Independent preflight for the supplied derived dataset, not a substitute for human Gold."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from src.data.audit import _norm

SPLITS = ('train', 'validation', 'test')

def verify(data_dir: Path) -> dict:
    result = {'tasks': {}, 'ner': {}, 'source_provenance': {}, 'status': 'PASS', 'warnings': [], 'failures': []}
    canonical = data_dir / 'unique_opportunities.csv'
    u = pd.read_csv(canonical, dtype=str, keep_default_na=False)
    result['source_provenance'] = {
        'unique_opportunities': len(u),
        'unique_canonical_ids': int(u.canonical_job_id.nunique()),
        'original_raw_posts_present': (data_dir / 'all_posts.csv').is_file(),
        'all_opportunity_records_present': (data_dir / 'all_opportunities.csv').is_file(),
        'duplicate_pairs_present': (data_dir / 'duplicate_pairs.csv').is_file(),
        'source_urls_present': int(u.source_url.astype(bool).sum())
    }
    if not result['source_provenance']['original_raw_posts_present']:
        result['warnings'].append('Original all_posts.csv absent: cannot fully re-audit 36,598 source posts or reconstruct merging')
    for task in ('classification_opportunity_type', 'classification_geographic_scope'):
        base = data_dir / task
        frames = {sp: pd.read_csv(base / f'{sp}.csv', dtype=str, keep_default_na=False) for sp in SPLITS}
        lm = json.loads((base / 'label_map.json').read_text(encoding='utf8'))['label2id']
        report = {'rows': {sp: len(x) for sp,x in frames.items()}, 'label_counts': {}, 'exact_overlap': {}, 'digit_stripped_overlap': {}}
        for sp, df in frames.items():
            report['label_counts'][sp] = df.label.value_counts().to_dict()
            required = {'id','text','label','label_id','group'}
            if not required.issubset(df.columns):
                result['failures'].append(f'{task}/{sp}: missing {required-set(df.columns)}')
            if not (df.label.map(lm).fillna(-1).astype(int).astype(str) == df.label_id).all():
                result['failures'].append(f'{task}/{sp}: labels and numeric labels disagree')
        for a,b in (('train','validation'),('train','test'),('validation','test')):
            x,y=frames[a],frames[b]
            exact = set(x.text.map(_norm)) & set(y.text.map(_norm)) - {''}
            group = set(x.group) & set(y.group) - {''}
            near = set(x.text.map(lambda t:_norm(t,False))) & set(y.text.map(lambda t:_norm(t,False))) - {''}
            report['exact_overlap'][f'{a}_{b}'] = {'texts':len(exact),'groups':len(group)}
            report['digit_stripped_overlap'][f'{a}_{b}'] = len(near)
            if exact or group:
                result['failures'].append(f'{task}/{a}/{b}: exact text/group leakage')
            if near:
                result['warnings'].append(f'{task}/{a}/{b}: {len(near)} similar normalized strings (numeric variants): review before Gold evaluation')
        result['tasks'][task]=report
    base=data_dir/'ner'
    labels=set(json.loads((base/'label_list.json').read_text(encoding='utf8'))['bio_tags'])
    ner={sp:[json.loads(l) for l in (base/f'{sp}.jsonl').read_text(encoding='utf8').splitlines() if l.strip()] for sp in SPLITS}
    result['ner']['rows']={sp:len(rs) for sp,rs in ner.items()}
    for sp,rs in ner.items():
        for r in rs:
            if len(r['tokens'])!=len(r['ner_tags']) or not set(r['ner_tags']).issubset(labels):
                result['failures'].append(f'NER invalid tags/length: {r["id"]}')
                break
    for a,b in (('train','validation'),('train','test'),('validation','test')):
        xa,xb=ner[a],ner[b]
        text_a={_norm(' '.join(r['tokens'])) for r in xa}-{''}
        text_b={_norm(' '.join(r['tokens'])) for r in xb}-{''}
        ids_a={r['id'] for r in xa};ids_b={r['id'] for r in xb}
        overlap=len(text_a&text_b)+len(ids_a&ids_b)
        result['ner'][f'exact_overlap_{a}_{b}']=overlap
        if overlap: result['failures'].append(f'NER exact leakage: {a}/{b}')
    # A single original ad MUST NOT be training in one task and held out in
    # another task, because the same source text could leak when shared.
    assignments = {}
    for task in ('classification_opportunity_type','classification_geographic_scope'):
        mp = {}
        base = data_dir/task
        for sp in SPLITS:
            x=pd.read_csv(base/f'{sp}.csv',dtype=str,keep_default_na=False,usecols=['id'])
            for ident in x.id:
                if ident in mp and mp[ident]!=sp:
                    result['failures'].append(f'{task}: ID {ident} occurs in two splits')
                mp[ident]=sp
        assignments[task]=mp
    assignments['ner']={r['id']:sp for sp in SPLITS for r in ner[sp]}
    for a,b in (('classification_opportunity_type','classification_geographic_scope'),
                ('classification_opportunity_type','ner'),('classification_geographic_scope','ner')):
        shared=set(assignments[a]) & set(assignments[b])
        disagreements=[ident for ident in shared if assignments[a][ident]!=assignments[b][ident]]
        result.setdefault('cross_task_split_alignment',{})[a+'__'+b]={
            'shared_records':len(shared),'disagreements':len(disagreements)}
        if disagreements:
            result['failures'].append(f'{a} vs {b}: {len(disagreements)} shared IDs assigned to different splits')
    for a,b in (('train','validation'),('train','test'),('validation','test')):
        canon_a={r.get('canonical_job_id','') for r in ner[a]}-{''}
        canon_b={r.get('canonical_job_id','') for r in ner[b]}-{''}
        nd={_norm(' '.join(r['tokens']),False) for r in ner[a]} & {_norm(' '.join(r['tokens']),False) for r in ner[b]}
        nd=nd-{''}
        result['ner'][f'near_overlap_{a}_{b}']=len(nd)
        if canon_a & canon_b:
            result['failures'].append(f'NER canonical group leakage {a}/{b}')
        if nd:
            result['warnings'].append(f'NER numeric-normalized text overlap {a}/{b}: {len(nd)}')
    train_ids=set(pd.read_csv(data_dir/'classification_opportunity_type'/'train.csv',dtype=str,usecols=['id']).id)
    balanced=data_dir/'classification_opportunity_type'/'train_balanced.csv'
    if balanced.exists():
        bal_ids=set(pd.read_csv(balanced,dtype=str,usecols=['id']).id)
        if not bal_ids.issubset(train_ids):
            result['failures'].append(f'train_balanced contains {len(bal_ids-train_ids)} heldout examples')

    expected={'JOB_TITLE','SKILL','EDUCATION','DEGREE','MAJOR','EXPERIENCE','LOCATION','ORGANIZATION','SALARY','DATE'}
    existing={t[2:] for t in labels if t!='O'}
    result['ner']['actual_entities']=sorted(existing)
    result['ner']['requested_but_absent']=sorted(expected-existing)
    result['ner']['extra_entities']=sorted(existing-expected)
    result['warnings'].append('All training/NER labels are SILVER. Human Gold validation not present.')
    if result['ner']['requested_but_absent']:
        result['warnings'].append('NER labels missing vs requested: '+', '.join(result['ner']['requested_but_absent']))
    result['status']='FAIL' if result['failures'] else ('WARN' if result['warnings'] else 'PASS')
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir', type=Path, default=Path('FINAL_TRAINING_DATASET'));ap.add_argument('--output',type=Path,default=Path('reports/repair/verification.json'))
    a=ap.parse_args()
    r=verify(a.data_dir)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2,ensure_ascii=False),encoding='utf8')
    print('Status:',r['status'],'failures:',len(r['failures']),'warnings:',len(r['warnings']))
    return 1 if r['failures'] else 0
if __name__=='__main__':
    raise SystemExit(main())
