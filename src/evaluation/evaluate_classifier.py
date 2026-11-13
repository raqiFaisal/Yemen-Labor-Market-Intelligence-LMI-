"""تقييم مستقل لنموذج تصنيف مدرّب على test (مرة واحدة بعد التدريب) + Gold إن وُجد مُعبأً.

    python -m src.evaluation.evaluate_classifier --task opportunity_type [--model_dir models_dl/classification/opportunity_type]

المخرجات (reports/evaluation/classification_<task>/): metrics.json · classification_report.txt · confusion_matrix.png/.csv ·
predictions.csv · errors_sample.csv (عينات التصنيف الخاطئ مع الثقة)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import CONFIG, DATA_DIR, MODELS_DIR, REPORTS_DIR, device, now, write_json
from src.data.loaders import load_classification
from src.evaluation.metrics import classification_metrics, plot_confusion
from src.inference.predictors import TextClassifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["opportunity_type", "geographic_scope", "relevance"])
    ap.add_argument("--model_dir", type=Path)
    ap.add_argument("--data_dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    md = a.model_dir or MODELS_DIR / "classification" / a.task
    out = a.out or REPORTS_DIR / "evaluation" / f"classification_{a.task}"
    out.mkdir(parents=True, exist_ok=True)
    data, lm = load_classification(a.task, a.data_dir, splits=["test"])
    te = data["test"]
    clf = TextClassifier(md)
    probs = clf.predict_proba(list(te.text))
    labels = clf.labels
    pred = [labels[i] for i in probs.argmax(1)]
    m = classification_metrics(list(te.label), pred, labels)
    res = {"task": a.task, "model_dir": str(md), "split": "test", "evaluated_at": now(), "device": str(device()),
           "label_quality": "silver — المقاييس تقيس الاتفاق مع تسميات القواعد الآلية، لا مع تصنيف بشري", **m}
    gold = gold_eval(a.task, clf, a.data_dir)
    if gold:
        res["gold"] = gold
    write_json(out / "metrics.json", {k: v for k, v in res.items() if k != "report_text"})
    (out / "classification_report.txt").write_text(m["report_text"], encoding="utf-8")
    pd.DataFrame(m["confusion_matrix"]["matrix"], index=m["confusion_matrix"]["labels"],
                 columns=m["confusion_matrix"]["labels"]).to_csv(out / "confusion_matrix.csv", encoding="utf-8-sig")
    plot_confusion(m["confusion_matrix"], out / "confusion_matrix.png", f"AraBERT — {a.task} (test, n={m['n']})")
    p = te[["id", "text", "label"]].assign(pred=pred, confidence=probs.max(1).round(4))
    p.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    err = p[p.label != p.pred].sort_values("confidence", ascending=False)
    err.assign(text=err.text.str[:400]).head(60).to_csv(out / "errors_sample.csv", index=False, encoding="utf-8-sig")
    print(m["report_text"])
    print(f"Accuracy={m['accuracy']} Macro-F1={m['macro_f1']} Weighted-F1={m['weighted_f1']} → {out}")


def gold_eval(task, clf, data_dir) -> dict | None:
    """إن كانت عينة Gold مُعبأة بشرياً (≥20 صفاً) تُقيَّم أيضاً — هذه وحدها أرقام «صحة» حقيقية."""
    col = {"opportunity_type": "gold_opportunity_type", "geographic_scope": "gold_scope_detail"}.get(task)
    f = Path(data_dir) / "gold_annotation" / "opportunity_gold_to_annotate.csv"
    if not col or not f.exists():
        return None
    g = pd.read_csv(f, dtype=str, keep_default_na=False)
    if col not in g:
        return None
    # Gold must have a documented human annotator; filled labels without
    # provenance are not evidence of human review.
    if 'annotator' not in g.columns:
        return {"status": "no human-annotator provenance column"}
    g = g[g[col].str.strip().isin(clf.labels) & g['annotator'].str.strip().ne('') & g['text'].str.strip().ne('')]
    if len(g) < 20:
        return {"status": f"insufficient human labels ({len(g)})"}
    pred = [clf.labels[i] for i in clf.predict_proba(list(g.text)).argmax(1)]
    m = classification_metrics(list(g[col].str.strip()), pred, clf.labels)
    return {k: v for k, v in m.items() if k != "report_text"} | {"label_quality": "human-annotated according to user-supplied annotator identifiers"}


if __name__ == "__main__":
    main()
