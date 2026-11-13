"""Small actual trained PyTorch neural baselines for offline CPU demo.

These models are trained from scratch on SILVER labels and are NOT AraBERT,
MARBERT, or human-validated models. Used only when pretrained checkpoints are
unavailable; inference response explicitly identifies the architecture.

  python -m src.offline.neural --tasks opportunity_type geographic_scope ner
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import time
import zlib

import numpy as np

from src.common import DATA_DIR, MODELS_DIR, REPORTS_DIR, set_seed, write_json
from src.data.loaders import load_classification, load_ner
from src.inference.predictors import TOKEN_RX, fix_bio, ModelNotAvailable

VOCAB = 16384
MAX_CLASS_TOKENS = 180
MAX_NER_WORDS = 96
WORD_RX = re.compile(r"(?u)\b\w+\b")


def token_id(s: str) -> int:
    return (zlib.crc32(s.lower().encode('utf8')) % VOCAB) + 1


def class_token_ids(text: str) -> list[int]:
    return [token_id(t) for t in WORD_RX.findall((text or '').lower())[:MAX_CLASS_TOKENS]] or [0]


def pack_class(items):
    import torch
    flat, offsets = [], []
    for x in items:
        offsets.append(len(flat))
        flat.extend(x)
    return torch.tensor(flat,dtype=torch.long),torch.tensor(offsets,dtype=torch.long)


class TinyClassifier(__import__('torch').nn.Module):
    def __init__(self, n_labels):
        import torch.nn as nn
        super().__init__()
        self.embedding = nn.EmbeddingBag(VOCAB+1,64,mode='mean',padding_idx=0)
        self.classifier = nn.Sequential(nn.Linear(64,128),nn.ReLU(),nn.Dropout(0.18),nn.Linear(128,n_labels))

    def forward(self, packed):
        ids, offsets = packed
        return self.classifier(self.embedding(ids,offsets))


class TinyNER(__import__('torch').nn.Module):
    def __init__(self, n_labels):
        import torch.nn as nn
        super().__init__()
        self.embedding = nn.Embedding(VOCAB+1,48,padding_idx=0)
        self.features = nn.Sequential(nn.Conv1d(48,96,5,padding=2),nn.GELU(),nn.Dropout(0.12),
                                      nn.Conv1d(96,96,3,padding=1),nn.GELU(),nn.Dropout(0.12))
        self.head = nn.Linear(96,n_labels)

    def forward(self, ids):
        x = self.features(self.embedding(ids).transpose(1,2)).transpose(1,2)
        return self.head(x)


def _require(f: Path):
    f=Path(f)
    if not (f/'model.pt').is_file() or not (f/'config.json').is_file():
        raise ModelNotAvailable(f'CPU offline neural model absent at {f}: train with python -m src.offline.neural')
    return json.loads((f/'config.json').read_text(encoding='utf8'))


class OfflineClassifier:
    def __init__(self, path):
        import torch
        self.path=Path(path)
        cfg=_require(self.path)
        if cfg.get('model_family')!='TinyClassifier':
            raise ValueError('Incompatible CPU classifier architecture')
        self.labels=cfg['labels']
        self.model=TinyClassifier(len(self.labels))
        self.model.load_state_dict(torch.load(self.path/'model.pt',map_location='cpu',weights_only=True))
        self.model.eval()

    def predict_proba(self,texts:list[str],batch_size=64):
        import torch
        chunks=[]
        with torch.inference_mode():
            for i in range(0,len(texts),batch_size):
                packed=pack_class([class_token_ids(t) for t in texts[i:i+batch_size]])
                chunks.append(self.model(packed).softmax(-1).numpy())
        return np.concatenate(chunks) if chunks else np.zeros((0,len(self.labels)))

    def predict(self,text:str,top_k:int=3):
        p=self.predict_proba([text])[0]
        order=np.argsort(-p)
        return {'label':self.labels[order[0]],'confidence':float(p[order[0]]),
                'top':[{'label':self.labels[i],'probability':float(p[i])} for i in order[:top_k]],
                'model_family':'TinyClassifier (PyTorch trained from scratch, SILVER; NOT AraBERT)'}


def ner_chunks(tokens:list[str],tags:list[str]|None=None):
    for j in range(0,len(tokens),MAX_NER_WORDS):
        sub=tokens[j:j+MAX_NER_WORDS]
        if tags is None:
            yield (sub,None)
        else:
            lab=tags[j:j+MAX_NER_WORDS].copy()
            if lab and lab[0].startswith('I-'):
                lab[0]='B-'+lab[0][2:]
            yield (sub,lab)


def pack_ner(items,labels=None):
    import torch
    maxlen=max((len(x) for x in items),default=1)
    x=np.zeros((len(items),maxlen),dtype=np.int64)
    y=np.full((len(items),maxlen),-100,dtype=np.int64)
    for j,t in enumerate(items):
        x[j,:len(t)]=[token_id(w) for w in t]
        if labels is not None:
            y[j,:len(t)]=labels[j]
    return torch.from_numpy(x),torch.from_numpy(y)


class OfflineNERTagger:
    def __init__(self,path):
        import torch
        self.path=Path(path)
        cfg=_require(self.path)
        if cfg.get('model_family')!='TinyNER':
            raise ValueError('Incompatible CPU NER architecture')
        self.labels=cfg['labels']
        self.model=TinyNER(len(self.labels))
        self.model.load_state_dict(torch.load(self.path/'model.pt',map_location='cpu',weights_only=True))
        self.model.eval()

    def tag_tokens(self,tokens_list:list[list[str]],batch_size:int=64):
        import torch
        owner=[];chunks=[]
        for d,tokens in enumerate(tokens_list):
            for j in range(0,len(tokens),MAX_NER_WORDS):
                owner.append((d,j))
                chunks.append(tokens[j:j+MAX_NER_WORDS])
        labels=[['O']*len(t) for t in tokens_list]
        confs=[[0.]*len(t) for t in tokens_list]
        with torch.inference_mode():
            for i in range(0,len(chunks),batch_size):
                subset=chunks[i:i+batch_size]
                x,_=pack_ner(subset)
                probs=self.model(x).softmax(-1)
                val, idx=probs.max(-1)
                val=val.numpy();idx=idx.numpy()
                for k,t in enumerate(subset):
                    d,j=owner[i+k]
                    labels[d][j:j+len(t)]=[self.labels[x] for x in idx[k,:len(t)]]
                    confs[d][j:j+len(t)]=val[k,:len(t)].tolist()
        return [fix_bio(x) for x in labels],confs

    def extract(self,text:str):
        from src.evaluation.metrics import extract_entities
        toks=[m.group(0) for m in TOKEN_RX.finditer(text or '')]
        if not toks:
            return {'tokens':[],'tags':[],'entities':[]}
        tags,confs=self.tag_tokens([toks])
        ents=extract_entities(toks,tags[0])
        for e in ents:
            e['confidence']=round(float(np.mean(confs[0][e['start']:e['end']])),4)
        return {'tokens':toks,'tags':tags[0],'entities':ents,
                'model_family':'TinyNER (PyTorch trained from scratch, SILVER; NOT AraBERT)'}


def _train_classifier(task:str,epochs:int=8,batch_size:int=128):
    import torch
    from sklearn.metrics import f1_score
    from torch.utils.data import DataLoader
    t0=time.time()
    data,lm=load_classification(task,DATA_DIR,splits=('train','validation','test'))
    labels=[lm['id2label'][str(i)] for i in range(len(lm['id2label']))]
    tr=data['train']; va=data['validation']; te=data['test']
    xtr=[class_token_ids(t) for t in tr.text]; xva=[class_token_ids(t) for t in va.text]
    ytr=tr.label_id.to_numpy(); yva=va.label_id.to_numpy();yte=te.label_id.to_numpy()
    counts=np.bincount(ytr,minlength=len(labels)).astype(float)
    weights=np.sqrt(counts.sum()/np.maximum(counts,1)/len(labels))
    weights=np.clip(weights,0.25,8)
    net=TinyClassifier(len(labels))
    opt=torch.optim.AdamW(net.parameters(),lr=0.004)
    g=torch.Generator().manual_seed(42)
    loader=DataLoader(list(range(len(tr))),batch_size=batch_size,shuffle=True,generator=g)
    wt=torch.tensor(weights,dtype=torch.float)
    best,best_state,bad=-1.,None,0
    history=[]
    for ep in range(epochs):
        net.train()
        total=0
        for inds in loader:
            inds=inds.tolist()
            opt.zero_grad(set_to_none=True)
            logits=net(pack_class([xtr[i] for i in inds]))
            loss=torch.nn.functional.cross_entropy(logits,torch.tensor(ytr[inds]),weight=wt)
            loss.backward()
            opt.step()
            total+=loss.item()*len(inds)
        net.eval()
        with torch.inference_mode():
            out=[]
            for i in range(0,len(xva),batch_size):
                out.extend(net(pack_class(xva[i:i+batch_size])).argmax(-1).tolist())
        score=f1_score(yva,out,labels=list(range(len(labels))),average='macro',zero_division=0)
        history.append({'epoch':ep+1,'train_loss':round(total/len(tr),5),'val_macro_f1':round(float(score),4)})
        print('Classification',task,history[-1],flush=True)
        if score>best+0.0001:
            best,best_state,bad=score,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},0
        else:
            bad+=1
            if bad>=3:
                break
    assert best_state is not None
    net.load_state_dict(best_state)
    path=MODELS_DIR/'offline'/'classification'/task
    path.mkdir(parents=True,exist_ok=True)
    torch.save(net.state_dict(),path/'model.pt')
    write_json(path/'config.json',{'model_family':'TinyClassifier','task':task,'labels':labels,'label_quality':'SILVER',
        'data_split':'verified group split','base_model':'NONE: trained from scratch','n_train':len(tr),'n_validation':len(va),
        'best_validation_macro_f1':round(float(best),4),'training_history':history,'epochs_run':len(history),
        'trained_at_utc':datetime.now(timezone.utc).isoformat(),'seconds':round(time.time()-t0,1)})
    classifier=OfflineClassifier(path)
    pred=classifier.predict_proba(list(te.text)).argmax(-1)
    from src.evaluation.metrics import classification_metrics
    metrics=classification_metrics(list(te.label),[labels[i] for i in pred],labels)
    metrics={k:v for k,v in metrics.items() if k!='report_text'}
    metrics.update(model_family='TinyClassifier',task=task,label_quality='SILVER (automatic labels, not human Gold)',
                   split='test',n_test=len(te),trained_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(REPORTS_DIR/'evaluation'/f'offline_{task}.json',metrics)
    print('Actual offline CPU classifier test macro F1',task,metrics['macro_f1'],flush=True)
    return metrics


def _entities_flat(rows):
    from src.evaluation.metrics import extract_entities
    return [set((e['start'],e['end'],e['label']) for e in extract_entities(r['tokens'],r['ner_tags'])) for r in rows]


def _exact_entity_scores(golds:list[list[str]],preds:list[list[str]]):
    from src.evaluation.metrics import extract_entities
    tp=fp=fn=0
    for gold,pred in zip(golds,preds):
        n=len(gold)
        if len(pred)!=n: raise ValueError('NER prediction length mismatch')
        ref={(e['start'],e['end'],e['label']) for e in extract_entities(['']*n,gold)}
        out={(e['start'],e['end'],e['label']) for e in extract_entities(['']*n,pred)}
        tp+=len(ref&out); fp+=len(out-ref); fn+=len(ref-out)
    p=tp/max(tp+fp,1);r=tp/max(tp+fn,1);f=2*p*r/max(p+r,1e-8)
    return {'entity_precision':round(p,4),'entity_recall':round(r,4),'entity_f1':round(f,4),
            'tp':tp,'fp':fp,'fn':fn,'n_sentences':len(golds)}


def _train_ner(epochs:int=5,max_train:int=4000,batch_size:int=48):
    import torch
    from torch.utils.data import DataLoader
    t0=time.time()
    data,tags=load_ner(DATA_DIR,splits=('train','validation','test'))
    rng=random.Random(42)
    train=data['train'].copy()
    rng.shuffle(train)
    train=train[:max_train] if max_train else train
    trainchunks=list(y for r in train for y in ner_chunks(r['tokens'],r['ner_tags']))
    lab={t:i for i,t in enumerate(tags)}
    tok=[x for x,y in trainchunks]
    labelids=[[lab[t] for t in y] for x,y in trainchunks]
    # only gold span positions? Silver labels are incomplete; still train as Silver
    counts=np.bincount([i for row in labelids for i in row],minlength=len(tags)).astype(float)
    weights=np.sqrt(counts.sum()/(np.maximum(counts,1)*len(tags)))
    weights=np.clip(weights,0.2,5.0)
    wt=torch.tensor(weights,dtype=torch.float)
    net=TinyNER(len(tags));opt=torch.optim.AdamW(net.parameters(),lr=0.002)
    g=torch.Generator().manual_seed(42)
    loader=DataLoader(list(range(len(tok))),batch_size=batch_size,shuffle=True,generator=g)
    val=data['validation'][:350]
    best,bad,best_state=-1.,0,None
    history=[]
    for ep in range(epochs):
        net.train()
        running=0;seen=0
        for indexes in loader:
            indexes=indexes.tolist()
            x,y=pack_ner([tok[j] for j in indexes],[labelids[j] for j in indexes])
            opt.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(net(x).reshape(-1,len(tags)),y.flatten(),weight=wt,ignore_index=-100)
            loss.backward()
            opt.step()
            running+=float(loss.item())*len(indexes);seen+=len(indexes)
        net.eval()
        # Validation never influences learning; used only for early stopping.
        gold=[];preds=[]
        with torch.inference_mode():
            for r in val:
                ids=r['tokens'];p=[]
                for x,_ in ner_chunks(ids):
                    xx,_=pack_ner([x]); ii=net(xx).argmax(-1)[0,:len(x)].tolist()
                    p.extend(tags[i] for i in ii)
                gold.append(r['ner_tags']);preds.append(fix_bio(p))
        met=_exact_entity_scores(gold,preds)
        history.append({'epoch':ep+1,'train_loss':round(running/max(seen,1),5),'val_entity_f1':met['entity_f1']})
        print('NER CPU',history[-1],flush=True)
        if met['entity_f1']>best+0.0001:
            best,bad,best_state=met['entity_f1'],0,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        else:
            bad+=1
            if bad>=2:break
    if best_state is None:raise ValueError('NER failed to produce a checkpoint')
    net.load_state_dict(best_state)
    path=MODELS_DIR/'offline'/'ner'
    path.mkdir(parents=True,exist_ok=True)
    torch.save(net.state_dict(),path/'model.pt')
    write_json(path/'config.json',{'model_family':'TinyNER','labels':tags,'label_quality':'SILVER',
        'base_model':'NONE: trained from scratch','trained_sentences':len(train),'train_chunks':len(trainchunks),
        'validation_early_stopping_sentences':len(val),'epochs_run':len(history),
        'validation_best_micro_entity_f1':best,'history':history,'trained_at_utc':datetime.now(timezone.utc).isoformat(),
        'seconds':round(time.time()-t0,1)})
    pred=OfflineNERTagger(path)
    test=data['test']
    gt=[r['ner_tags'] for r in test]
    pt=pred.tag_tokens([r['tokens'] for r in test],batch_size=64)[0]
    metrics=_exact_entity_scores(gt,pt)
    metrics.update({'model_family':'TinyNER','split':'test','label_quality':'SILVER (automatic, not human Gold)',
                    'train_sentences':len(train),'n_test_sentences':len(test),
                    'trained_at_utc':datetime.now(timezone.utc).isoformat()})
    write_json(REPORTS_DIR/'evaluation'/'offline_ner.json',metrics)
    print('Actual offline NER entity F1',metrics['entity_f1'],flush=True)
    return metrics


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--tasks',nargs='+',choices=('opportunity_type','geographic_scope','ner'),default=['opportunity_type','geographic_scope','ner'])
    ap.add_argument('--classification-epochs',type=int,default=8)
    ap.add_argument('--ner-epochs',type=int,default=5)
    ap.add_argument('--max-ner-train',type=int,default=4000)
    a=ap.parse_args()
    set_seed(42)
    for t in a.tasks:
        if t=='ner':_train_ner(epochs=a.ner_epochs,max_train=a.max_ner_train)
        else:_train_classifier(t,epochs=a.classification_epochs)

if __name__=='__main__':
    main()
