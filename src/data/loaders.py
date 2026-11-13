"""قراءة ملفات FINAL_TRAINING_DATASET للمهام الثلاث مع تحقق صارم (لا افتراض لأسماء الأعمدة: تُفحص وتُرفض إن غابت)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.common import DATA_DIR

SPLITS = ("train", "validation", "test")
CLS_TASKS = {"opportunity_type": "classification_opportunity_type",
             "geographic_scope": "classification_geographic_scope",
             "relevance": "classification_opportunity_type/relevance_filter"}


def load_classification(task: str, data_dir: Path = DATA_DIR, splits=SPLITS) -> tuple[dict, dict]:
    """({split: DataFrame[id,text,label,label_id]}, label_map). يرفض ملفاً بلا الأعمدة المطلوبة أو بتسميات خارج label_map."""
    folder = Path(data_dir) / CLS_TASKS[task]
    lm = json.loads((folder / "label_map.json").read_text(encoding="utf-8"))
    out = {}
    for sp in splits:
        d = pd.read_csv(folder / f"{sp}.csv", dtype=str, keep_default_na=False)
        miss = {"id", "text", "label", "label_id"} - set(d.columns)
        if miss:
            raise ValueError(f"{folder}/{sp}.csv: أعمدة مفقودة {miss}")
        bad = set(d.label) - set(lm["label2id"])
        if bad:
            raise ValueError(f"{folder}/{sp}.csv: تسميات خارج label_map: {bad}")
        d["label_id"] = d.label.map(lm["label2id"]).astype(int)
        out[sp] = d.reset_index(drop=True)
    return out, lm


def load_ner(data_dir: Path = DATA_DIR, splits=SPLITS) -> tuple[dict, list[str]]:
    """({split: [ {id, tokens, ner_tags} ]}, bio_tags). يتحقق من تساوي الأطوال وصلاحية الوسوم."""
    folder = Path(data_dir) / "ner"
    meta = json.loads((folder / "label_list.json").read_text(encoding="utf-8"))
    tags = meta["bio_tags"]
    out = {}
    for sp in splits:
        recs = []
        with open(folder / f"{sp}.jsonl", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if len(r["tokens"]) != len(r["ner_tags"]):
                    raise ValueError(f"{sp}: طول غير متساوٍ في {r['id']}")
                bad = set(r["ner_tags"]) - set(tags)
                if bad:
                    raise ValueError(f"{sp}: وسوم غير معروفة {bad}")
                recs.append({"id": r["id"], "tokens": r["tokens"], "ner_tags": r["ner_tags"], "text": r.get("text", "")})
        out[sp] = recs
    return out, tags


_PRE = {}


def arabert_preprocess(texts, model_name: str):
    """معالجة AraBERT الرسمية (فصل الترقيم/التطبيع حسب إصدار النموذج) إن توفرت مكتبة arabert، وإلا النص كما هو.
    لا تُستخدم لـ NER (الرموز مُقطّعة مسبقاً والوسوم مرتبطة بها)."""
    try:
        from arabert.preprocess import ArabertPreprocessor
        if model_name not in _PRE:
            _PRE[model_name] = ArabertPreprocessor(model_name=model_name)
        p = _PRE[model_name]
        return [p.preprocess(t) for t in texts]
    except ImportError as exc:
        if "arabert" in model_name.lower():
            raise RuntimeError("AraBERT preprocessing requires the arabert package; install requirements_dl.txt") from exc
        return list(texts)
    except Exception as exc:
        if "arabert" in model_name.lower():
            raise RuntimeError(f"AraBERT preprocessing failed for {model_name}") from exc
        return list(texts)
