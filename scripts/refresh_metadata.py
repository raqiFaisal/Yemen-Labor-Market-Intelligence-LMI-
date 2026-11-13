"""Refresh true split-dependent statistics after conservative split repair.

Keeps the original manifest under reports/repair, never silently represents
historical counts as independently reverified source records.
"""
from __future__ import annotations
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import pandas as pd
from src.common import DATA_DIR, REPORTS_DIR

TASKS = ("classification_opportunity_type", "classification_geographic_scope")
SPLITS = ("train", "validation", "test")


def digest(f: Path):
    h = sha256()
    with f.open('rb') as inp:
        for chunk in iter(lambda: inp.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main(data_dir=DATA_DIR, report_dir=REPORTS_DIR/'repair'):
    data_dir, report_dir = Path(data_dir), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = data_dir / 'dataset_manifest.json'
    original = report_dir / 'original_dataset_manifest.json'
    if manifest.is_file() and not original.is_file():
        shutil.copyfile(manifest, original)
    historical = json.loads(original.read_text(encoding='utf8')) if original.exists() else {}
    sizes, counts = [], {}
    for task in TASKS:
        folder = data_dir/task
        d = {}
        lm = json.loads((folder/'label_map.json').read_text(encoding='utf8'))['label2id']
        for sp in SPLITS:
            frame = pd.read_csv(folder/f'{sp}.csv', dtype=str, keep_default_na=False)
            d[sp] = frame
            stats = frame.label.value_counts().to_dict()
            counts[f'{task}/{sp}'] = len(frame)
            sizes.append({'task':task,'split':sp,'rows':len(frame),'labels':json.dumps(stats,ensure_ascii=False)})
            if set(frame.label) - set(lm):
                raise ValueError(f'{task}/{sp}: labels outside label map')
        labels = list(lm)
        distribution = pd.DataFrame({'label':labels,**{sp:[int((d[sp].label==l).sum()) for l in labels] for sp in SPLITS}})
        distribution.to_csv(folder/'class_distribution.csv', index=False, encoding='utf-8-sig')
        train = d['train'].label.value_counts().to_dict()
        denom = max(len(train),1)
        weights = {l:round(len(d['train'])/(denom*train[l]),4) if train.get(l) else 0.0 for l in labels}
        (folder/'class_weights.json').write_text(json.dumps(weights,indent=2,ensure_ascii=False),encoding='utf8')
    for sp in SPLITS:
        n = sum(1 for line in (data_dir/'ner'/f'{sp}.jsonl').open(encoding='utf8') if line.strip())
        counts[f'ner/{sp}'] = n
        sizes.append({'task':'ner','split':sp,'rows':n,'labels':''})
    (report_dir/'verified_split_sizes.csv').parent.mkdir(exist_ok=True,parents=True)
    pd.DataFrame(sizes).to_csv(report_dir/'verified_split_sizes.csv',index=False,encoding='utf-8-sig')
    # Dataset's built-in statistics might include original source files that were
    # not distributed in this archive. Do not fabricate missing statistics.
    actual_files = []
    for f in sorted(data_dir.rglob('*')):
        if not f.is_file() or f == manifest:
            continue
        entry = {'path':str(f.relative_to(data_dir)), 'bytes':f.stat().st_size, 'sha256':digest(f)}
        if f.suffix == '.csv':
            entry['columns'] = pd.read_csv(f,nrows=0).columns.tolist()
        actual_files.append(entry)
    previous = {r['path'] for r in historical.get('files',[]) if 'path' in r}
    present = {r['path'] for r in actual_files}
    missing = sorted(previous-present)
    summary = {
        'name':'AI Labor Market Intelligence / verified delivered dataset',
        'generated_utc':datetime.now(timezone.utc).isoformat(),
        'label_quality':'SILVER, human annotation not verified',
        'verified_counts':{'unique_opportunities':int(len(pd.read_csv(data_dir/'unique_opportunities.csv',usecols=['job_id']))),**counts},
        'previous_reported_source_counts_not_reverified': historical.get('summary',{}),
        'missing_from_delivered_archive':missing,
        'split_method':'Canonical ID + original ID + text normalized without digits, unified across tasks, no records dropped',
        'split_backup':'reports/repair/original_splits.zip',
        'original_manifest':'reports/repair/original_dataset_manifest.json',
        'files':actual_files,
    }
    manifest.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    (report_dir/'missing_source_files.json').write_text(json.dumps(missing,ensure_ascii=False,indent=2),encoding='utf8')
    print('Updated metadata for',len(actual_files),'files; missing from original manifest:',len(missing))
    return summary

if __name__ == '__main__':
    main()
