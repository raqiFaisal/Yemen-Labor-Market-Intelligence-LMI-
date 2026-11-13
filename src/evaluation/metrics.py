"""مقاييس موحدة لكل النماذج — تُستخدم في التحقق والاختبار والمقارنة بخطوط الأساس (نفس الكود للجميع)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def classification_metrics(y_true, y_pred, labels: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
    present = [l for l in labels if l in set(y_true) | set(y_pred)]
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=present, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, labels=[l for l in present if l in set(y_true)],
                                                    average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, labels=present, average="weighted", zero_division=0)
    return {
        "n": int(len(y_true)), "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(float(pm), 4), "macro_recall": round(float(rm), 4), "macro_f1": round(float(fm), 4),
        "weighted_precision": round(float(pw), 4), "weighted_recall": round(float(rw), 4), "weighted_f1": round(float(fw), 4),
        "per_class": [{"label": l, "precision": round(float(a), 4), "recall": round(float(b), 4), "f1": round(float(c), 4),
                       "support": int(d)} for l, a, b, c, d in zip(present, p, r, f, s)],
        "confusion_matrix": {"labels": present, "matrix": confusion_matrix(y_true, y_pred, labels=present).tolist()},
        "report_text": classification_report(y_true, y_pred, labels=present, zero_division=0, digits=4),
        "note_macro": "macro_f1 محسوب على الفئات الموجودة فعلاً في y_true فقط",
    }


def plot_confusion(cm: dict, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels, m = cm["labels"], np.array(cm["matrix"])
    norm = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 3, 1.0 * len(labels) + 2.5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{m[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def entity_metrics(true_tags: list[list[str]], pred_tags: list[list[str]]) -> dict:
    """تقييم على مستوى الكيان (seqeval, IOB2 صارم): الكيان صحيح فقط إن تطابق نوعه وحدوده كاملة."""
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
    from seqeval.scheme import IOB2
    rep = classification_report(true_tags, pred_tags, mode="strict", scheme=IOB2, output_dict=True, zero_division=0)
    per = [{"entity": k, "precision": round(float(v["precision"]), 4), "recall": round(float(v["recall"]), 4), "f1": round(float(v["f1-score"]), 4),
            "support": int(v["support"])} for k, v in rep.items() if k not in ("micro avg", "macro avg", "weighted avg")]
    per = [x for x in per if x["support"] > 0]          # فقط الأنواع الموجودة فعلاً في مجموعة الاختبار
    tok_true = [t for s in true_tags for t in s]
    tok_pred = [t for s in pred_tags for t in s]
    return {
        "entity_precision": round(float(precision_score(true_tags, pred_tags, mode="strict", scheme=IOB2)), 4),
        "entity_recall": round(float(recall_score(true_tags, pred_tags, mode="strict", scheme=IOB2)), 4),
        "entity_f1": round(float(f1_score(true_tags, pred_tags, mode="strict", scheme=IOB2)), 4),
        "macro_entity_f1": round(float(np.mean([x["f1"] for x in per])) if per else 0.0, 4),
        "per_entity": per,
        "token_accuracy_for_reference": round(float(np.mean([a == b for a, b in zip(tok_true, tok_pred)])), 4) if tok_true else None,
        "n_sentences": len(true_tags),
    }


def extract_entities(tokens: list[str], tags: list[str]) -> list[dict]:
    """إعادة بناء الكيانات (نوع + نص + موضع الرموز) من وسوم BIO."""
    ents, cur = [], None
    for i, (tok, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith("B-") or (tag.startswith("I-") and (cur is None or cur["label"] != tag[2:])):
            if cur:
                ents.append(cur)
            cur = {"label": tag[2:], "start": i, "end": i + 1, "tokens": [tok]}
        elif tag.startswith("I-") and cur:
            cur["end"] = i + 1
            cur["tokens"].append(tok)
        else:
            if cur:
                ents.append(cur)
            cur = None
    if cur:
        ents.append(cur)
    for e in ents:
        e["text"] = " ".join(e.pop("tokens"))
    return ents


def forecast_metrics(actual, pred, mape_floor: float = 0.01) -> dict:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    m = ~np.isnan(a) & ~np.isnan(p)
    a, p = a[m], p[m]
    if not len(a):
        return {"n": 0, "mae": None, "rmse": None, "mape": None}
    mm = a >= mape_floor          # MAPE فقط حيث القيمة الفعلية ≥ 1% (القسمة على قيم صغيرة تضخم الخطأ بلا معنى)
    return {"n": int(len(a)), "mae": round(float(np.mean(np.abs(a - p))), 6), "rmse": round(float(np.sqrt(np.mean((a - p) ** 2))), 6),
            "mape": round(float(np.mean(np.abs((a[mm] - p[mm]) / a[mm])) * 100), 2) if mm.any() else None,
            "mape_n": int(mm.sum()), "mape_note": f"MAPE محسوب فقط حيث القيمة الفعلية ≥ {mape_floor}"}


def save_table(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
