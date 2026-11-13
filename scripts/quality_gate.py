"""Strict release gate: never claims full acceptance when essential checks blocked."""
from __future__ import annotations
import importlib.util
import json
import py_compile
from pathlib import Path
import subprocess
import sys
from src.common import ROOT, REPORTS_DIR, MODELS_DIR, write_json
from scripts.verify_delivery import verify
from scripts.verify_model_artifacts import verify_lstm, verify_offline


def main():
    result={'syntax':{},'dataset':{},'models':{},'integration':{},'full_ui_and_hf_tests':{},
            'human_gold':{'status':'BLOCKED','reason':'No human-reviewed Gold in uploaded material'},
            'final_status':'NOT_FULLY_ACCEPTED'}
    bad=[];n=0
    for p in ROOT.rglob('*.py'):
        if '__pycache__' in p.parts or any(k.startswith('.') for k in p.relative_to(ROOT).parts):continue
        try:py_compile.compile(str(p),doraise=True)
        except Exception as exc:bad.append(f'{p.relative_to(ROOT)}: {exc}')
        n+=1
    result['syntax']={'total':n,'errors':bad,'status':'PASS' if not bad else 'FAIL'}
    verification_file=REPORTS_DIR/'repair'/'verification.json'
    splits=list((ROOT/'FINAL_TRAINING_DATASET').glob('classification_*/train.csv'))
    splits += list((ROOT/'FINAL_TRAINING_DATASET'/'ner').glob('*.jsonl'))
    if verification_file.is_file() and all(verification_file.stat().st_mtime>=x.stat().st_mtime for x in splits):
        ds=json.loads(verification_file.read_text(encoding='utf8'))
        result['dataset']['report_reused']='Verified report newer than all split files; run scripts.verify_delivery to regenerate'
    else:
        ds=verify(ROOT/'FINAL_TRAINING_DATASET')
    result['dataset']={**result['dataset'], 'status':ds['status'],'failures':ds['failures'],'warnings':ds['warnings'],
                       'cross_task_split_alignment':ds.get('cross_task_split_alignment')}
    previous_model_report=REPORTS_DIR/'repair'/'model_verification.json'
    models={}
    weight_paths=list((MODELS_DIR/'lstm').glob('*/lstm.pt'))+list((MODELS_DIR/'offline').glob('**/model.pt'))
    if previous_model_report.is_file() and all(previous_model_report.stat().st_mtime >= x.stat().st_mtime for x in weight_paths):
        old=json.loads(previous_model_report.read_text(encoding='utf8'))
        models={**old.get('lstm',{}),'offline_neural':old.get('offline',[])}
        result['models']['report_reused']='Checkpoint verification newer than all weights'
    else:
        for fam in ('skills','specializations','job_titles','governorates'):
            try:models[fam]=verify_lstm(fam)
            except Exception as exc:models[fam]={'status':'FAIL','error':str(exc)}
        try:models['offline_neural']=verify_offline()
        except Exception as exc:models['offline_neural']={'status':'FAIL','error':str(exc)}
    result['models']={**result['models'], 'status':'PASS' if all(v.get('status')=='PASS' for k,v in models.items() if k!='offline_neural')
        and isinstance(models.get('offline_neural'),list) else 'FAIL',
        'details':models}
    command=[sys.executable,'-m','pytest','-q','tests/test_repair_preflight.py','tests/test_offline_integration.py']
    proc=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    result['integration']={'status':'PASS' if proc.returncode==0 else 'FAIL',
                           'command':' '.join(command),'stdout_tail':proc.stdout[-3000:],'stderr_tail':proc.stderr[-1000:]}
    required={'streamlit','transformers','seqeval','tokenizers','accelerate'}
    missing=sorted(x for x in required if importlib.util.find_spec(x) is None)
    result['full_ui_and_hf_tests']={
        'status':'BLOCKED' if missing else 'NOT_EXECUTED',
        'missing_dependencies':missing,
        'reason':'Cannot certify Streamlit widget execution and HF model fine-tuning in this environment',
    }
    result['araBERT_finetuning']={'status':'BLOCKED','reason':'No pretrained AraBERT checkpoint and no network/GPU in this environment'}
    if any(x['status']=='FAIL' for x in [result['syntax'],result['dataset'],result['models'],result['integration']]):
        result['final_status']='FAIL'
    elif missing or result['human_gold']['status']=='BLOCKED':
        result['final_status']='PARTIALLY_VERIFIED_WITH_EXPLICIT_BLOCKERS'
    else:
        result['final_status']='PARTIALLY_VERIFIED_AWAITING_HUMAN_GOLD_AND_ARABERT'
    write_json(REPORTS_DIR/'repair'/'quality_gate.json',result)
    print('QUALITY GATE:',result['final_status'])
    print('Syntax:',result['syntax']['status'],result['syntax']['total'])
    print('Data:',result['dataset']['status'],len(result['dataset']['failures']),'failed')
    print('Weight checks:',result['models']['status'],'integrations:',result['integration']['status'])
    print('HF/Streamlit dependencies missing:',missing)
    # exit code 2 = unresolved acceptance blockers, 1 = actual test failures
    return 1 if result['final_status']=='FAIL' else 2

if __name__=='__main__':
    raise SystemExit(main())
