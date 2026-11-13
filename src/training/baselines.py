"""خطوط أساس للمقارنة بالتعلم العميق — نفس التقسيمات ونفس المقاييس.

    python -m src.training.baselines --task opportunity_type|geographic_scope|ner|all

التصنيف: TF-IDF (حروف 2–5 + كلمات 1–2) + LogisticRegression / LinearSVC، اختيار C على validation فقط، ثم تقييم test مرة واحدة.
NER: CRF (sklearn-crfsuite) بخصائص الكلمة وسياقها، اختيار c1/c2 على validation، تقييم كياني على test.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import pandas as pd

from src.common import DATA_DIR, MODELS_DIR, REPORTS_DIR, now, set_seed, write_json
from src.data.loaders import load_classification, load_ner
from src.evaluation.metrics import classification_metrics, entity_metrics, plot_confusion


def cls_baseline(task, data_dir, out_models, out_rep):
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    data, lm = load_classification(task, data_dir)
    tr, va, te = data["train"], data["validation"], data["test"]
    labels = [lm["id2label"][str(i)] for i in range(len(lm["id2label"]))]
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=200000, sublinear_tf=True)
    wv = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True,
                         token_pattern=r"(?u)\b\w+\b")
    Xtr = hstack([cv.fit_transform(tr.text), wv.fit_transform(tr.text)]).tocsr()     # المفردات من train فقط
    Xva = hstack([cv.transform(va.text), wv.transform(va.text)]).tocsr()
    Xte = hstack([cv.transform(te.text), wv.transform(te.text)]).tocsr()
    results, best_models = {}, {}
    for name, make in [("tfidf_logreg", lambda C: LogisticRegression(C=C, max_iter=3000, class_weight="balanced")),
                       ("tfidf_linearsvc", lambda C: LinearSVC(C=C, class_weight="balanced"))]:
        grid = []
        for C in (0.1, 0.5, 1, 3, 10):
            mdl = make(C).fit(Xtr, tr.label)
            grid.append((classification_metrics(list(va.label), list(mdl.predict(Xva)), labels)["macro_f1"], C, mdl))
        vbest, Cb, mdl = max(grid, key=lambda x: x[0])
        pred = list(mdl.predict(Xte))
        m = classification_metrics(list(te.label), pred, labels)
        results[name] = {"best_C_on_validation": Cb, "validation_macro_f1": vbest,
                         "validation_grid": [{"C": c, "macro_f1": f} for f, c, _ in grid],
                         "test": {k: v for k, v in m.items() if k != "report_text"}}
        best_models[name] = mdl
        plot_confusion(m["confusion_matrix"], out_rep / f"{name}_confusion.png", f"{name} — {task} (test)")
        (out_rep / f"{name}_report.txt").write_text(m["report_text"], encoding="utf-8")
        print(f"[{task}] {name}: C={Cb} val-F1={vbest} | test Acc={m['accuracy']} Macro-F1={m['macro_f1']}")
    out_models.mkdir(parents=True, exist_ok=True)
    joblib.dump({"char": cv, "word": wv, "models": best_models, "labels": labels}, out_models / f"tfidf_{task}.joblib")
    return results


def word2features(s, i):
    w = s[i]
    f = {"bias": 1.0, "w.lower": w.lower(), "w[-3:]": w[-3:], "w[-2:]": w[-2:], "w[:3]": w[:3], "w.isdigit": w.isdigit(),
         "w.isupper": w.isupper(), "w.istitle": w.istitle(), "w.len": min(len(w), 15),
         "w.arabic": any("؀" <= c <= "ۿ" for c in w), "w.has_al": w.startswith("ال")}
    for k, off in (("-1", -1), ("+1", 1), ("-2", -2), ("+2", 2)):
        j = i + off
        if 0 <= j < len(s):
            f[f"{k}:w.lower"] = s[j].lower()
            f[f"{k}:w[-3:]"] = s[j][-3:]
        else:
            f[f"{k}:pad"] = True
    return f


def ner_baseline(data_dir, out_models, out_rep, max_train=None):
    import sklearn_crfsuite
    data, _ = load_ner(data_dir)
    tr, va, te = (data[s] for s in ("train", "validation", "test"))
    if max_train:
        tr = tr[:max_train]

    def feats(recs):
        return [[word2features(r["tokens"], i) for i in range(len(r["tokens"]))] for r in recs]
    Xtr, ytr = feats(tr), [r["ner_tags"] for r in tr]
    Xva, Xte = feats(va), feats(te)
    grid = []
    for c1, c2 in ((0.05, 0.05), (0.1, 0.01), (0.3, 0.05)):
        crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=150, all_possible_transitions=True)
        crf.fit(Xtr, ytr)
        grid.append((entity_metrics([r["ner_tags"] for r in va], crf.predict(Xva))["entity_f1"], (c1, c2), crf))
    vbest, cc, crf = max(grid, key=lambda x: x[0])
    m = entity_metrics([r["ner_tags"] for r in te], crf.predict(Xte))
    out_models.mkdir(parents=True, exist_ok=True)
    joblib.dump(crf, out_models / "crf_ner.joblib")
    pd.DataFrame(m["per_entity"]).to_csv(out_rep / "crf_per_entity.csv", index=False, encoding="utf-8-sig")
    print(f"[ner] CRF c1,c2={cc} val-F1={vbest} | test Entity-F1={m['entity_f1']}")
    return {"crf": {"best_c1_c2_on_validation": cc, "validation_entity_f1": vbest,
                    "validation_grid": [{"c1_c2": c, "entity_f1": f} for f, c, _ in grid], "test": m}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", choices=["opportunity_type", "geographic_scope", "ner", "all"])
    ap.add_argument("--data_dir", type=Path, default=DATA_DIR)
    ap.add_argument("--max_train_ner", type=int)
    a = ap.parse_args()
    set_seed()
    tasks = ["opportunity_type", "geographic_scope", "ner"] if a.task == "all" else [a.task]
    for t in tasks:
        t0 = time.time()
        rep = REPORTS_DIR / "evaluation" / f"baseline_{t}"
        rep.mkdir(parents=True, exist_ok=True)
        res = ner_baseline(a.data_dir, MODELS_DIR / "baselines", rep, a.max_train_ner) if t == "ner" else \
            cls_baseline(t, a.data_dir, MODELS_DIR / "baselines", rep)
        write_json(rep / "metrics.json", {"task": t, "evaluated_at": now(), "seconds": round(time.time() - t0),
                                          "selection": "hyperparameters chosen on validation only; test evaluated once", **res})


if __name__ == "__main__":
    main()
