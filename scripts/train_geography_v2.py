"""Train a real location-aware PyTorch classifier; never post-process its class output.

Run: python -m scripts.train_geography_v2
Reads only train for fitting, validation for epoch choice/temperature, test for
final reporting. Generated counterfactual examples are training-only and are
marked synthetic, NEVER included in held-out metrics or called human Gold.
"""
from __future__ import annotations
import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.common import DATA_DIR, MODELS_DIR, REPORTS_DIR
from src.offline.geographic_v2 import GeoScopeNet, GeoScopeClassifier, LABELS, packed_batch

_DOMESTIC={'LOCAL','SINGLE_GOVERNORATE','MULTI_GOVERNORATE','NATIONWIDE','YEMEN_UNSPECIFIED'}

def collapse(lab):
    if lab in _DOMESTIC:return 'YEMEN'
    if lab=='INTERNATIONAL':return 'OUTSIDE_YEMEN'
    if lab=='REMOTE':return 'REMOTE'
    return None  # unknown: not a supervised label


def synthetic_training(seed=317, each=850):
    """Training augmentation only: clear-cut examples, and location reversals."""
    rng=random.Random(seed)
    loc_y=['صنعاء','عدن','تعز','المكلا','الحديدة','ذمار','إب','مأرب','اليمن','sanaa','Aden','Yemen','Marib']
    loc_x=['الرياض','جدة','دبي','الدوحة','الكويت','مصر','القاهرة','ألمانيا','السعودية','الإمارات','Riyadh','Saudi Arabia','Dubai','Qatar']
    jobs=['مهندس برمجيات','محاسب','موظف موارد بشرية','مدير مشاريع','ممرض','مهندس مدني','مصمم جرافيك','مطور ويب','أخصائي تسويق','باحث','مدرس','مدير مبيعات']
    skills=['Python وSQL','إدارة المشاريع','المحاسبة','التواصل','Excel','اللغة الإنجليزية','Power BI']
    local_formats=[
       'تعلن إحدى الشركات في {loc} عن حاجتها إلى {job} يمتلك خبرة ثلاث سنوات في {skill}.',
       'مطلوب {job} للعمل في {loc}. المتطلبات: خبرة في {skill}.',
       'فرصة عمل: {job}، الموقع: {loc}، المهارات: {skill}.',
       'وظيفة شاغرة في {loc}: {job} مع خبرة في {skill}.',
       'تبحث شركة عن {job} في {loc} ويشترط إجادة {skill}.',
       'Location: {loc}. Job title: {job}. Required skills: {skill}.',
       'الجهة في {other} ولكن موقع العمل في {loc}، مطلوب {job} بخبرة {skill}.',
       'الشركة مقرها {other} وتعلن عن شاغر {job} للعمل في {loc}.',
       'تعلن مؤسسة عن وظيفة {job}. موقع الوظيفة: {loc}.',
       'وظيفة: {job}، مكان العمل في {loc}، يشترط {skill}.',
    ]
    remote_formats=[
       'تعلن شركة عن وظيفة {job} عن بعد بدوام كامل. المهارات {skill}.',
       'مطلوب {job} للعمل عن بُعد من المنزل. خبرة في {skill}.',
       'فرصة عمل عن بعد: {job}، يشترط معرفة {skill}.',
       'Location: Remote. Job title: {job}. Skills: {skill}.',
       'شركة في {loc} تطلب {job} للعمل عن بعد. المهارات {skill}.',
       'الشركة مقرها {other} وتطلب {job} بنظام العمل عن بعد.',
       'تعلن شركة عن {job}. مكان العمل: عن بعد، والخبرة في {skill}.',
    ]
    examples=[]
    for label,locations,otherlocations in [('YEMEN',loc_y,loc_x),('OUTSIDE_YEMEN',loc_x,loc_y)]:
        for i in range(each):
            loc=rng.choice(locations);other=rng.choice(otherlocations)
            s=rng.choice(local_formats).format(loc=loc,other=other,job=rng.choice(jobs),skill=rng.choice(skills))
            examples.append((s,label,'synthetic'))
    for i in range(each):
        loc=rng.choice(loc_y);other=rng.choice(loc_x)
        s=rng.choice(remote_formats).format(loc=loc,other=other,job=rng.choice(jobs),skill=rng.choice(skills))
        examples.append((s,'REMOTE','synthetic'))
    rng.shuffle(examples)
    return examples


def _load(split):
    src=DATA_DIR/'classification_geographic_scope'/f'{split}.csv'
    data=pd.read_csv(src,dtype=str,keep_default_na=False)
    data['broad_label']=data['label'].map(collapse)
    data=data[data.broad_label.notna()].reset_index(drop=True)
    return data


def _logits(net,texts,batch_size):
    net.eval(); out=[]
    with torch.inference_mode():
        for i in range(0,len(texts),batch_size):
            out.append(net(packed_batch(texts[i:i+batch_size])).numpy())
    return np.concatenate(out)


def _metrics(y,p):
    yp=p.argmax(axis=1)
    result={'n':int(len(y)),'accuracy':float(accuracy_score(y,yp)),
            'macro_f1':float(f1_score(y,yp,labels=[0,1,2],average='macro',zero_division=0)),
            'weighted_f1':float(f1_score(y,yp,labels=[0,1,2],average='weighted',zero_division=0)),
            'per_class':classification_report(y,yp,labels=[0,1,2],target_names=LABELS,output_dict=True,zero_division=0),
            'confusion_matrix':confusion_matrix(y,yp,labels=[0,1,2]).tolist()}
    return result


def _temperature(logits, y):
    """Single scalar chosen ONLY on Silver validation by cross entropy."""
    with torch.no_grad():
        x=torch.tensor(logits,dtype=torch.float32)
        target=torch.tensor(y,dtype=torch.long)
        ts=np.geomspace(0.35,4,80)
        losses=[float(torch.nn.functional.cross_entropy(x/float(t),target)) for t in ts]
    return float(ts[int(np.argmin(losses))])


def _softmax(logits,temp):
    x=logits/temp
    x=x-x.max(axis=1,keepdims=True)
    exp=np.exp(x)
    return exp/exp.sum(axis=1,keepdims=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--synthetic-per-class',type=int,default=850)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--output',type=Path,default=MODELS_DIR/'offline'/'classification'/'geographic_scope_v2')
    args=p.parse_args()
    start=time.monotonic();torch.set_num_threads(min(4,torch.get_num_threads()))
    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    tr,va,te=(_load(s) for s in ['train','validation','test'])
    print('Source:',len(tr),len(va),len(te),'Silver labels')
    for a,b in ((tr,va),(tr,te),(va,te)):
        if set(a['group']) & set(b['group']):raise ValueError('Group leakage in supplied geographic splits')
    synthetic=synthetic_training(each=args.synthetic_per_class)
    trtext=list(tr.text)+[x[0] for x in synthetic]
    try_=[LABELS.index(x) for x in list(tr.broad_label)+[x[1] for x in synthetic]]
    vatext=list(va.text); vay=np.array([LABELS.index(x) for x in va.broad_label])
    tetext=list(te.text); tey=np.array([LABELS.index(x) for x in te.broad_label])
    # Extra samples are training-only. Validation/test remain original posts.
    loader=DataLoader(list(range(len(trtext))),batch_size=args.batch_size,shuffle=True,generator=torch.Generator().manual_seed(42))
    counts=np.bincount(try_,minlength=3).astype(float)
    weights=np.sqrt(counts.sum()/np.maximum(counts,1)/3)
    weights=np.minimum(weights,3)
    print('Train class counts:',dict(zip(LABELS,counts.astype(int))))
    net=GeoScopeNet();opt=torch.optim.AdamW(net.parameters(),lr=0.003,weight_decay=0.01)
    weights=torch.tensor(weights,dtype=torch.float32)
    best_f1=-1;best_epoch=0;best_state=None;history=[];bad=0
    for epoch in range(args.epochs):
        net.train();total=0.0
        for indexes in loader:
            indx=indexes.tolist();batch=[trtext[i] for i in indx]
            labels=torch.tensor([try_[i] for i in indx],dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            logits=net(packed_batch(batch))
            loss=torch.nn.functional.cross_entropy(logits,labels,weight=weights)
            loss.backward();opt.step()
            total+=float(loss.item())*len(indx)
        val_logits=_logits(net,vatext,128)
        met=_metrics(vay,val_logits)
        history.append({'epoch':epoch+1,'loss':round(total/len(trtext),4),
                        'validation_macro_f1':round(met['macro_f1'],4),
                        'validation_accuracy':round(met['accuracy'],4)})
        print(history[-1],flush=True)
        if met['macro_f1']>best_f1+0.0002:
            best_f1=met['macro_f1'];best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};best_epoch=epoch+1;bad=0
        else:
            bad+=1
            if bad>=4:break
    net.load_state_dict(best_state)
    val_logits=_logits(net,vatext,args.batch_size)
    temp=_temperature(val_logits,vay)
    te_logits=_logits(net,tetext,args.batch_size)
    measured=_metrics(tey,_softmax(te_logits,temp))
    measured['ece_silver_test'] = float(np.mean(np.abs(np.max(_softmax(te_logits,temp),axis=1)-(tey==np.argmax(te_logits,axis=1)).astype(float))))
    # This metric is NOT standard binned ECE; do not call it ECE.
    measured['average_absolute_confidence_error'] = measured.pop('ece_silver_test')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.save(best_state,args.output/'model.pt')
    config={'model_family':'GeoScopeNetV2','labels':LABELS,'label_quality':'SILVER',
            'feature_engineering':'location cues fed as numeric inputs to trained PyTorch network; no hard output override',
            'base_model':'trained from scratch, not AraBERT','temperature':temp,
            'n_original_train':len(tr),'n_synthetic_train':len(synthetic),
            'n_validation':len(va),'n_test':len(te),'best_epoch':best_epoch,
            'train_history':history,'selected_with':'macro F1 on validation (SILVER)',
            'temperature_selected_with':'cross entropy on validation (SILVER)',
            'timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'train_seconds':round(time.monotonic()-start,2)}
    (args.output/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    predictor=GeoScopeClassifier(args.output)
    samples={
      'تعلن إحدى الشركات في صنعاء عن حاجتها إلى مهندس برمجيات يمتلك خبرة لا تقل عن ثلاث سنوات في Python وMachine Learning وSQL. يشترط الحصول على درجة البكالوريوس في علوم الحاسوب.':'YEMEN',
      'تعلن مؤسسة عن وظيفة في اليمن لمهندس شبكات.':'YEMEN',
      'تعلن شركة عن حاجتها إلى موظف للعمل في عدن.':'YEMEN',
      'مطلوب محاسب للعمل في الرياض بالمملكة العربية السعودية.':'OUTSIDE_YEMEN',
      'مطلوب مهندس شبكات للعمل في دبي بالإمارات.':'OUTSIDE_YEMEN',
      'شركة مقرها صنعاء تعلن عن وظيفة محاسب في الرياض.':'OUTSIDE_YEMEN',
      'شركة مقرها الرياض تعلن عن وظيفة محاسب للعمل في صنعاء.':'YEMEN',
      'تعلن مؤسسة في صنعاء عن وظيفة مطور ويب عن بعد.':'REMOTE',
      'وظيفة مطور برمجيات بنظام العمل عن بعد.':'REMOTE',
      'Job opening: Python Developer, Location: Saudi Arabia.':'OUTSIDE_YEMEN',
      'Job opening: Python Developer, Location: Yemen.':'YEMEN',
    }
    scenario=[]
    for txt,expected in samples.items():
        actual=predictor.predict(txt)
        scenario.append({'text':txt,'expected':expected,'predicted':actual['label'],
                         'confidence':round(actual['confidence'],4),'passed':actual['label']==expected})
    measured.update({'model':'GeoScopeNetV2','task':'geographic_scope_broad',
                     'label_quality':'SILVER (automatic labels; not Gold)',
                     'split':'test','validation_n':len(va),'train_n_original':len(tr),
                     'train_n_synthetic':len(synthetic),
                     'scenario_tests':scenario,
                     'scenario_passed':sum(s['passed'] for s in scenario),'scenario_count':len(scenario),
                     'interpretation':'classification with trainable PyTorch network, no post-hoc geo label replacement'})
    out=REPORTS_DIR/'evaluation'/'geographic_scope_v2.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(measured,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Final held-out SILVER test:',{k:measured[k] for k in ['n','accuracy','macro_f1','weighted_f1','scenario_passed','scenario_count']})
    for s in scenario:
        print('OK' if s['passed'] else 'FAIL',s['predicted'],s['confidence'],s['text'][:72])
    print('Output:',args.output)
    return measured

if __name__=='__main__':main()
