"""تقييم NER على test بمقاييس الكيان (seqeval صارم) + تحليل الأخطاء (كيانات فائتة / خاطئة / بحدود خاطئة) + Gold إن وُجد.

    python -m src.evaluation.evaluate_ner [--model_dir models_dl/ner]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from src.common import DATA_DIR, MODELS_DIR, REPORTS_DIR, now, write_json
from src.data.loaders import load_ner
from src.evaluation.metrics import entity_metrics, extract_entities
from src.inference.predictors import NERTagger, TOKEN_RX


def error_analysis(recs, pred_tags, limit=80):
    missed, spurious, boundary = [], [], []
    for r, pt in zip(recs, pred_tags):
        gt = {(e["start"], e["end"], e["label"]): e["text"] for e in extract_entities(r["tokens"], r["ner_tags"])}
        pr = {(e["start"], e["end"], e["label"]): e["text"] for e in extract_entities(r["tokens"], pt)}
        for k, t in gt.items():
            if k not in pr:
                over = [p for p in pr if p[2] == k[2] and p[0] < k[1] and k[0] < p[1]]
                (boundary if over else missed).append({"id": r["id"], "label": k[2], "gold_text": t,
                                                       "pred_text": pr[over[0]] if over else None})
        for k, t in pr.items():
            if k not in gt and not any(g[2] == k[2] and g[0] < k[1] and k[0] < g[1] for g in gt):
                spurious.append({"id": r["id"], "label": k[2], "pred_text": t})
    return missed, spurious, boundary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=Path, default=MODELS_DIR / "ner")
    ap.add_argument("--data_dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "evaluation" / "ner")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    data, tags = load_ner(a.data_dir, splits=["test"])
    te = data["test"]
    tagger = NERTagger(a.model_dir)
    pred, _ = tagger.tag_tokens([r["tokens"] for r in te])
    m = entity_metrics([r["ner_tags"] for r in te], pred)
    missed, spurious, boundary = error_analysis(te, pred)
    res = {"split": "test", "model_dir": str(a.model_dir), "evaluated_at": now(), **m,
           "errors": {"missed": len(missed), "spurious": len(spurious), "boundary": len(boundary),
                      "missed_by_label": dict(Counter(x["label"] for x in missed)),
                      "spurious_by_label": dict(Counter(x["label"] for x in spurious))},
           "label_quality": "silver — اتفاق مع وسوم القواعد؛ الكيانات التي فاتت القواعد موسومة O في الاختبار أيضاً"}
    g = gold_ner(tagger, a.data_dir)
    if g:
        res["gold"] = g
    write_json(a.out / "metrics.json", res)
    pd.DataFrame(m["per_entity"]).to_csv(a.out / "per_entity.csv", index=False, encoding="utf-8-sig")
    for name, rows in [("missed_entities", missed), ("spurious_entities", spurious), ("boundary_errors", boundary)]:
        pd.DataFrame(rows[:300]).to_csv(a.out / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(m["per_entity"]).to_string(index=False))
    print(f"Entity P={m['entity_precision']} R={m['entity_recall']} F1={m['entity_f1']} → {a.out}")


def gold_ner(tagger, data_dir):
    f = Path(data_dir) / "gold_annotation" / "ner_gold_to_annotate.jsonl"
    if not f.exists():
        return None
    ex = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    # Human reviewer identity and explicit completed status required. A
    # programmatically populated list of spans must NOT be treated as Gold.
    ex = [e for e in ex if isinstance(e.get("gold_spans"), list)
          and str(e.get("annotator", "")).strip()
          and e.get("review_status") == "completed"]
    if len(ex) < 10:
        return {"status": f"insufficient human-annotated docs ({len(ex)})"}
    true, pred = [], []
    valid_labels = set(tagger.labels)
    valid_entities = {x[2:] for x in valid_labels if x.startswith('B-')}
    for e in ex:
        for s in e["gold_spans"]:
            if s["label"] not in valid_entities:
                raise ValueError(f"Gold NER entity {s['label']} absent from model tag set; retrain with target label schema")
            a, b = int(s["start"]), int(s["end"])
            if not (0 <= a < b <= len(e['text'])):
                raise ValueError(f"Invalid gold NER character span for {e['id']}")
        toks = [(m.group(0), m.start(), m.end()) for m in TOKEN_RX.finditer(e["text"])]
        tt = ["O"] * len(toks)
        for s in e["gold_spans"]:
            inside = [i for i, (_, a, b) in enumerate(toks) if a >= s["start"] and b <= s["end"]]
            if not inside or toks[inside[0]][1] != a or toks[inside[-1]][2] != b:
                raise ValueError(f"NER Gold span for {e['id']} not aligned to word tokenizer")
            for j, i in enumerate(inside):
                if tt[i] != 'O':
                    raise ValueError(f"NER Gold has overlapping entities for {e['id']}")
                tt[i] = ("B-" if j == 0 else "I-") + s["label"]
        true.append(tt)
        pred.append(tagger.tag_tokens([[t for t, _, _ in toks]])[0][0])
    return entity_metrics(true, pred) | {"label_quality": "human-annotated with reviewer identifier", "n_docs": len(ex)}


if __name__ == "__main__":
    main()
