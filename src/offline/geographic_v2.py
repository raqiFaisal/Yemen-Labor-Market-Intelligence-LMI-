"""Trainable CPU geographic-scope model for LMI, not a rule-based output override.

The model combines hashed lexical embeddings with *location/context input features*.
All three class logits are produced by a trained PyTorch network. The features
only describe the input; they never set the output label or confidence.

Labels: YEMEN (work inside Yemen), OUTSIDE_YEMEN (abroad), REMOTE.
Source annotations are SILVER. Probabilities are temperature-scaled against
held-out SILVER validation data and are not certified Gold accuracy.
"""
from __future__ import annotations
import json
from functools import lru_cache
import re
import zlib
from pathlib import Path

import numpy as np

LABELS = ['YEMEN', 'OUTSIDE_YEMEN', 'REMOTE']
V_SIZE = 16384
MAX_WORDS = 200
FEATURE_NAMES = [
    'mentions_yemen','mentions_yemen_city','mentions_foreign','mentions_remote',
    'mentions_explicit_outside','near_location_yemen','near_location_foreign',
    'near_work_yemen','near_work_foreign','near_work_remote',
    'near_work_outside','mentions_all_yemen','yemen_city_first_part',
    'foreign_first_part','yemen_city_after_in','foreign_after_in',
    'explicit_yemen_in','explicit_foreign_location_en',
    'explicit_yemen_location_en','remote_in_first_part',
    'local_mentions_count','foreign_mentions_count',
    'is_company_based_in_yemen','is_company_based_abroad',
    'contrast_but_work_local','contrast_but_work_abroad',
]

# These lists supply *input evidence features*, not hard-coded labels.
YEMEN_LOCATIONS = (
 'صنعاء','صنعا','عدن','تعز','اب','إب','ذمار','الحديدة','الحديده','حضرموت',
 'مأرب','مارب','المكلا','سيئون','سيون','صعدة','صعده','عمران','لحج','الضالع',
 'البيضاء','أبين','ابين','الجوف','حجة','حجه','شبوة','شبوه','ريمة','ريمه',
 'المحويت','المهرة','المهره','سقطرى','سقطري','المخا','الحوبان',
 'sanaa','sana a','sana\'a','aden','taiz','hodeidah','marib','ma\'rib',
 'mukalla','hadramout','ibb','lahj','dhamar','yemen'
)
FOREIGN_LOCATIONS = (
 'السعودية','السعوديه','الرياض','جدة','جده','مكة','مكه','الدمام',
 'الإمارات','الامارات','دبي','ابوظبي','أبو ظبي','الشارقة','الشارقه',
 'الكويت','قطر','الدوحة','الدوحه','البحرين','سلطنة عمان','سلطنه عمان','مسقط',
 'مصر','القاهرة','القاهره','الأردن','الاردن','تركيا','اسطنبول','ألمانيا','المانيا',
 'فرنسا','لندن','بريطانيا','أمريكا','امريكا','الولايات المتحدة','الولايات المتحده',
 'saudi arabia','riyadh','jeddah','dubai','uae','qatar','doha',
 'kuwait','cairo','egypt','amman','amman jordan','germany',
 'turkey','istanbul','london','united kingdom','united states'
)

_NORM_RE = re.compile(r'[\u064b-\u065f\u0670\u0640]')
_TOKEN_RE = re.compile(r'(?u)\b\w+\b')

def normalize(s: str) -> str:
    s = _NORM_RE.sub('', str(s or '')).lower()
    for a,b in (('أ','ا'),('إ','ا'),('آ','ا'),('ٱ','ا'),('ى','ي'),('ة','ه')):
        s = s.replace(a,b)
    return re.sub(r'\s+', ' ', s).strip()


def _phrase_rx(words):
    variants = {normalize(w) for w in words}
    return re.compile(r'(?<!\w)(?:'+'|'.join(re.escape(w) for w in sorted(variants,key=len,reverse=True))+r')(?!\w)')

YEMEN_RX = _phrase_rx(YEMEN_LOCATIONS)
ABROAD_RX = _phrase_rx(FOREIGN_LOCATIONS)
YEMEN_COUNTRY_RX = re.compile(r'(?<!\w)(?:اليمن|يمنيه|يمني|yemen)(?!\w)')
REMOTE_RX = re.compile(r'(?<!\w)(?:عن بعد|اون ?لاين|عمل من المنزل|remote|work from home|online work)(?!\w)')
OUTSIDE_RX = re.compile(r'(?:خارج اليمن|خارج البلاد|outside yemen|abroad)')
MULTI_RX = re.compile(r'(?:جميع|كافه|مختلف)\s+محافظات|(?:كافه|جميع)\s+انحاء اليمن')
WORK_RX = re.compile(r'(?:للعمل|مكان العمل|موقع العمل|مقر العمل|العمل|موقع الوظيفة|مكان الوظيفة|فرصة عمل|فرصه عمل|وظيفة|وظيفه|workplace|duty station|job location|work location)\s*(?::|-|هو|في|ب|at|in)?\s*', re.I)
LOCATION_RX = re.compile(r'(?:الموقع|المدينة|المحافظة|مقر الوظيفة|location|duty station|city)\s*(?::|-|في|in|at)?\s*', re.I)
IN_RX = re.compile(r'(?:في|ب|داخل|in|at)\s+', re.I)
COMPANY_RX = re.compile(r'(?:مقر(?:ها|ه)?|الشركه|الموسسه|company based|headquarters|based in)\s*(?:في|ب|:|in)?\s*', re.I)

def _in_context(rx, t, place):
    for m in rx.finditer(t):
        # Exclude next sentence/phone/address from contextual evidence.
        chunk = re.split(r'[\.،;,؛\n]', t[m.end():m.end()+80],maxsplit=1)[0]
        if place.search(chunk):
            return True
    return False

@lru_cache(maxsize=40000)
def geo_features(text: str) -> np.ndarray:
    t = normalize(text[:2400])
    prefix=t[:260]
    local=list(YEMEN_RX.finditer(t))
    foreign=list(ABROAD_RX.finditer(t))
    work_local=_in_context(WORK_RX,t,YEMEN_RX)
    work_foreign=_in_context(WORK_RX,t,ABROAD_RX)
    kwloc_local=_in_context(LOCATION_RX,t,YEMEN_RX)
    kwloc_foreign=_in_context(LOCATION_RX,t,ABROAD_RX)
    arr = [
        bool(YEMEN_COUNTRY_RX.search(t)), bool(local),bool(foreign),bool(REMOTE_RX.search(t)),
        bool(OUTSIDE_RX.search(t)),kwloc_local,kwloc_foreign,
        work_local,work_foreign,_in_context(WORK_RX,t,REMOTE_RX),
        _in_context(WORK_RX,t,OUTSIDE_RX),bool(MULTI_RX.search(t)),
        bool(YEMEN_RX.search(prefix)),bool(ABROAD_RX.search(prefix)),
        _in_context(IN_RX,t,YEMEN_RX),_in_context(IN_RX,t,ABROAD_RX),
        bool(re.search(r'في اليمن|داخل اليمن|in yemen',t)),
        bool(re.search(r'(?:location|duty station|city)\s*[:\-=]?\s*(?:saudi|riyadh|dubai|qatar|kuwait|egypt|jeddah)',t)),
        bool(re.search(r'(?:location|duty station|city)\s*[:\-=]?\s*(?:yemen|sanaa|aden|taiz|marib)',t)),
        bool(REMOTE_RX.search(prefix)),min(len(local),4)/4, min(len(foreign),4)/4,
        _in_context(COMPANY_RX,t,YEMEN_RX),_in_context(COMPANY_RX,t,ABROAD_RX),
        bool(foreign) and work_local, bool(local) and work_foreign,
    ]
    assert len(arr)==len(FEATURE_NAMES)
    return np.asarray(arr, dtype=np.float32)


@lru_cache(maxsize=40000)
def word_ids(text: str) -> list[int]:
    # Include bigrams to preserve local word order, unlike the older EmbeddingBag.
    toks = _TOKEN_RE.findall(normalize(text))[:MAX_WORDS]
    units = toks + ['_'.join(toks[i:i+2]) for i in range(len(toks)-1)]
    return [(zlib.crc32(s.encode('utf-8')) % V_SIZE)+1 for s in units] or [0]


def packed_batch(texts: list[str]):
    import torch
    ids=[];offsets=[]
    for t in texts:
        offsets.append(len(ids))
        ids.extend(word_ids(t))
    feats=np.vstack([geo_features(t) for t in texts]) if texts else np.zeros((0,len(FEATURE_NAMES)),dtype=np.float32)
    return (torch.tensor(ids,dtype=torch.long),torch.tensor(offsets,dtype=torch.long),torch.from_numpy(feats))


class GeoScopeNet(__import__('torch').nn.Module):
    def __init__(self):
        from torch import nn
        super().__init__()
        self.embedding=nn.EmbeddingBag(V_SIZE+1,64,mode='mean',padding_idx=0)
        self.locations=nn.Sequential(nn.Linear(len(FEATURE_NAMES),64),nn.ReLU(),nn.Dropout(0.10))
        self.head=nn.Sequential(nn.Linear(128,96),nn.ReLU(),nn.Dropout(0.18),nn.Linear(96,len(LABELS)))

    def forward(self, batch):
        import torch
        ids, offsets, feats=batch
        return self.head(torch.cat([self.embedding(ids,offsets),self.locations(feats)],dim=-1))


class GeoScopeClassifier:
    def __init__(self, path: str|Path):
        import torch
        p=Path(path)
        with (p/'config.json').open(encoding='utf-8') as f:
            cfg=json.load(f)
        if cfg.get('model_family')!='GeoScopeNetV2' or cfg.get('labels')!=LABELS:
            raise ValueError('Geographic checkpoint config differs from GeoScopeNetV2')
        self.labels=LABELS
        self.temperature=float(cfg.get('temperature',1.0))
        if not (0.05<=self.temperature<=30):raise ValueError('Bad calibration temperature')
        self.net=GeoScopeNet()
        self.net.load_state_dict(torch.load(p/'model.pt',map_location='cpu',weights_only=True))
        self.net.eval()
        self.config=cfg

    def predict_proba(self,texts:list[str],batch_size:int=128):
        import torch
        all_p=[]
        with torch.inference_mode():
            for i in range(0,len(texts),batch_size):
                data=packed_batch(texts[i:i+batch_size])
                all_p.append((self.net(data)/self.temperature).softmax(dim=-1).numpy())
        return np.concatenate(all_p) if all_p else np.zeros((0,len(LABELS)))

    def predict(self,text:str,top_k:int=3):
        probabilities=self.predict_proba([text])[0]
        order=np.argsort(-probabilities)
        return {'label':LABELS[int(order[0])],
                'confidence':float(probabilities[order[0]]),
                'top':[{'label':LABELS[int(i)],'probability':float(probabilities[i])} for i in order[:top_k]],
                'model_family':'GeoScopeNetV2 (PyTorch, SILVER + synthetic training examples)',
                'confidence_kind':'temperature_scaled_silver_validation',
                'decision_source':'trained_neural_model'}
