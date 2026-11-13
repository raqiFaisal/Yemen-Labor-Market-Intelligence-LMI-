"""تقييم تسميات الـ Dataset الآلية (Silver) مقابل تصنيف بشري (Gold).

    python -m scripts.evaluate_dataset           → reports/dataset_evaluation.md + .json

المدخلات (يعبئها إنسان):
  labels/opportunity_gold_to_annotate.csv  عمودا gold_opportunity_type و gold_geographic_scope
  labels/ner_gold_to_annotate.jsonl        حقل gold_spans: [{"start","end","label"}] (أو [] إن لا كيانات)
لا تقييم بدون Gold — يطبع عدد الصفوف المصنفة ويتوقف. الأرقام الناتجة هي الأرقام الوحيدة الصالحة للتقرير العلمي.
"""
import json

import pandas as pd

from core.config import LABELS_DIR, PROCESSED_DIR, REPORTS_DIR
from preprocessing.opportunity import SCOPES, TYPES


def prf(y_true, y_pred, labels):
    rows = []
    for l in labels:
        tp = sum((t == l) & (p == l) for t, p in zip(y_true, y_pred))
        fp = sum((t != l) & (p == l) for t, p in zip(y_true, y_pred))
        fn = sum((t == l) & (p != l) for t, p in zip(y_true, y_pred))
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        rows.append({"label": l, "support": tp + fn, "precision": round(P, 3), "recall": round(R, 3),
                     "f1": round(2 * P * R / (P + R), 3) if P + R else 0.0})
    df = pd.DataFrame(rows)
    sup = df[df.support > 0]
    return df, {"accuracy": round(sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true), 3) if len(y_true) else None,
                "macro_f1": round(float(sup.f1.mean()), 3) if len(sup) else None, "n": len(y_true)}


def main():
    gold_f = LABELS_DIR / "opportunity_gold_to_annotate.csv"
    posts_f = PROCESSED_DIR / "dataset" / "posts_all.csv"
    if not gold_f.exists() or not posts_f.exists():
        raise SystemExit("✖ شغّل build_dataset أولاً")
    g = pd.read_csv(gold_f, dtype=str, keep_default_na=False)
    p = pd.read_csv(posts_f, dtype=str, keep_default_na=False).set_index("post_id")
    rep, L = {}, ["# تقييم تسميات الـ Dataset مقابل Gold بشري", ""]
    for task, gcol, pcol, labels in [("opportunity_type", "gold_opportunity_type", "opportunity_type", TYPES),
                                     ("geographic_scope", "gold_geographic_scope", "geographic_scope", SCOPES)]:
        lab = g[g[gcol].str.strip() != ""]
        lab = lab[lab.post_id.isin(p.index)]
        if task == "geographic_scope":
            lab = lab[lab[gcol] != "NA"]
        if len(lab) < 20:
            L.append(f"## {task}\nلا يكفي: {len(lab)} صف مصنف بشرياً (الحد الأدنى للتقرير 20).")
            rep[task] = {"n_gold": len(lab), "status": "insufficient"}
            continue
        yt, yp = list(lab[gcol].str.strip().str.upper()), list(p.loc[lab.post_id, pcol].fillna(""))
        st = list(p.loc[lab.post_id, "type_status" if task == "opportunity_type" else "scope_status"])
        df, summ = prf(yt, yp, labels)
        acc_only = [(t, q) for t, q, s in zip(yt, yp, st) if s == "accepted"]
        _, summ_acc = prf([t for t, _ in acc_only], [q for _, q in acc_only], labels)
        cm = pd.crosstab(pd.Series(yt, name="gold"), pd.Series(yp, name="silver"))
        rep[task] = {"overall": summ, "accepted_only": summ_acc, "per_class": df.to_dict("records"),
                     "confusion": cm.to_dict()}
        L += [f"## {task}", f"- كل الحالات: n={summ['n']} | Accuracy={summ['accuracy']} | Macro-F1={summ['macro_f1']}",
              f"- المقبولة آلياً فقط: n={summ_acc['n']} | Accuracy={summ_acc['accuracy']} | Macro-F1={summ_acc['macro_f1']}",
              "", df[df.support > 0].to_markdown(index=False), "", "مصفوفة الالتباس:", "", cm.to_markdown(), ""]
    nf = LABELS_DIR / "ner_gold_to_annotate.jsonl"
    if nf.exists():
        ex = [json.loads(l) for l in open(nf, encoding="utf-8") if l.strip()]
        ex = [e for e in ex if e.get("gold_spans") is not None]
        if len(ex) >= 10:
            per = {}
            for e in ex:
                gs = {(s["start"], s["end"], s["label"]) for s in e["gold_spans"]}
                ss = {(s["start"], s["end"], s["label"]) for s in e["silver_spans"]}
                for lab in {x[2] for x in gs | ss}:
                    d = per.setdefault(lab, [0, 0, 0])
                    gl, sl = {x for x in gs if x[2] == lab}, {x for x in ss if x[2] == lab}
                    d[0] += len(gl & sl); d[1] += len(sl - gl); d[2] += len(gl - sl)
            rows = [{"label": k, "tp": v[0], "fp": v[1], "fn": v[2],
                     "precision": round(v[0] / (v[0] + v[1]), 3) if v[0] + v[1] else 0,
                     "recall": round(v[0] / (v[0] + v[2]), 3) if v[0] + v[2] else 0} for k, v in sorted(per.items())]
            rep["ner_silver_vs_gold"] = {"n_docs": len(ex), "per_label": rows}
            L += ["## NER (مطابقة حرفية للمواضع)", f"وثائق مصححة: {len(ex)}", "", pd.DataFrame(rows).to_markdown(index=False)]
        else:
            L.append(f"## NER\nوثائق مصححة بشرياً: {len(ex)} (الحد الأدنى 10).")
    (REPORTS_DIR / "dataset_evaluation.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (REPORTS_DIR / "dataset_evaluation.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
