"""تدقيق الـ Dataset قبل التدريب: الجودة، التقسيمات، التسرب بين التقسيمات، صلاحية NER، وحالة Gold.

    python -m src.data.audit [--data_dir FINAL_TRAINING_DATASET]

المخرجات: reports/audit/audit_report.json + audit_report.md. لا يعدّل أي ملف بيانات.
الحكم النهائي: PASS / WARN / FAIL لكل فحص — FAIL (مثل تسرب نصوص متطابقة بين train و test) يجب إصلاحه قبل التدريب.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import DATA_DIR, REPORTS_DIR, now, write_json
from src.data.loaders import CLS_TASKS, SPLITS

MIN_TEST_PER_CLASS = 10


def _norm(t: str, digits: bool = True) -> str:
    """تطبيع للمقارنة. digits=False يحذف الأرقام أيضاً (كشف شبه التكرار: نفس الإعلان بتاريخ/رقم مختلف)."""
    t = re.sub(r"https?://\S+|@\w+" + ("" if digits else r"|\d"), " ", str(t).lower())
    t = re.sub("[إأآا]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t)).strip()


def check(name, status, detail, **extra):
    return {"check": name, "status": status, "detail": detail, **extra}


def audit_classification(task: str, data_dir: Path) -> dict:
    folder = data_dir / CLS_TASKS[task]
    if not (folder / "label_map.json").exists():
        return {"task": task, "status": "missing", "checks": [check("files", "FAIL", f"غير موجود: {folder}")]}
    lm = json.loads((folder / "label_map.json").read_text(encoding="utf-8"))
    d = {sp: pd.read_csv(folder / f"{sp}.csv", dtype=str, keep_default_na=False) for sp in SPLITS}
    checks, sizes = [], {sp: len(x) for sp, x in d.items()}
    dist = pd.DataFrame({sp: x.label.value_counts() for sp, x in d.items()}).fillna(0).astype(int)
    dist["total"] = dist.sum(axis=1)
    for sp, x in d.items():
        miss = {"id", "text", "label"} - set(x.columns)
        bad = set(x.label) - set(lm["label2id"])
        checks.append(check(f"schema_{sp}", "FAIL" if miss or bad else "PASS",
                            f"أعمدة مفقودة {sorted(miss)} / تسميات خارج الخريطة {sorted(bad)}" if miss or bad else "سليم"))
        empty = int((x.text.str.strip().str.len() < 20).sum())
        checks.append(check(f"short_texts_{sp}", "WARN" if empty else "PASS", f"{empty} نص أقل من 20 حرفاً"))
        dup = int(x.text.map(_norm).duplicated().sum())
        checks.append(check(f"dup_within_{sp}", "WARN" if dup > 0.02 * len(x) else "PASS", f"{dup} نص مكرر داخل {sp}"))
    ids = {sp: set(x.id) for sp, x in d.items()}
    txt = {sp: set(x.text.map(_norm)) - {""} for sp, x in d.items()}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        io, to = len(ids[a] & ids[b]), len(txt[a] & txt[b])
        near = len({_norm(t, False) for t in d[a].text} & {_norm(t, False) for t in d[b].text} - {""})
        checks.append(check(f"leak_id_{a}_{b}", "FAIL" if io else "PASS", f"{io} معرّف مشترك"))
        checks.append(check(f"leak_text_{a}_{b}", "FAIL" if to else "PASS", f"{to} نص متطابق (بعد التطبيع) مشترك"))
        checks.append(check(f"near_dup_{a}_{b}", "WARN" if near > to else "PASS", f"{near} نص متطابق بعد حذف الأرقام"))
    if "group" in d["train"].columns:
        g = {sp: set(x.group) - {""} for sp, x in d.items()}
        gl = len(g["train"] & g["test"]) + len(g["train"] & g["validation"]) + len(g["validation"] & g["test"])
        checks.append(check("leak_group", "FAIL" if gl else "PASS", f"{gl} مجموعة (canonical_job_id) في أكثر من تقسيم"))
    for lab in lm["label2id"]:
        n_te = int(dist.loc[lab, "test"]) if lab in dist.index else 0
        n_tr = int(dist.loc[lab, "train"]) if lab in dist.index else 0
        if n_tr == 0:
            checks.append(check(f"class_{lab}", "WARN", "غائبة عن train — لن يتعلمها النموذج"))
        elif n_te < MIN_TEST_PER_CLASS:
            checks.append(check(f"class_{lab}", "WARN", f"{n_te} مثال فقط في test — مقاييس هذه الفئة غير مستقرة"))
    imb = dist.total[dist.total > 0]
    ratio = float(imb.max() / imb.min()) if len(imb) else 0
    checks.append(check("imbalance", "WARN" if ratio > 20 else "PASS", f"نسبة أكبر فئة لأصغرها = {ratio:.1f}"))
    lens = d["train"].text.str.len()
    lsrc = Counter(d["train"].get("label_source", pd.Series(["unknown"] * len(d["train"]))))
    return {"task": task, "sizes": sizes, "distribution": dist.reset_index(names="label").to_dict("records"),
            "train_text_chars": {"median": int(lens.median()), "p95": int(lens.quantile(.95)), "max": int(lens.max())},
            "label_sources_train": dict(lsrc), "checks": checks}


def audit_ner(data_dir: Path) -> dict:
    folder = data_dir / "ner"
    if not (folder / "label_list.json").exists():
        return {"task": "ner", "status": "missing", "checks": [check("files", "FAIL", f"غير موجود: {folder}")]}
    tags = json.loads((folder / "label_list.json").read_text(encoding="utf-8"))["bio_tags"]
    checks, stats, recs = [], {}, {}
    for sp in SPLITS:
        rs = [json.loads(l) for l in open(folder / f"{sp}.jsonl", encoding="utf-8") if l.strip()]
        recs[sp] = rs
        bad_len = sum(len(r["tokens"]) != len(r["ner_tags"]) for r in rs)
        bad_tag = sum(bool(set(r["ner_tags"]) - set(tags)) for r in rs)
        invalid_i = 0
        ents = Counter()
        for r in rs:
            prev = "O"
            for t in r["ner_tags"]:
                if t.startswith("I-") and prev[2:] != t[2:]:
                    invalid_i += 1
                if t.startswith("B-"):
                    ents[t[2:]] += 1
                prev = t
        no_ent = sum(all(t == "O" for t in r["ner_tags"]) for r in rs)
        ntok = np.array([len(r["tokens"]) for r in rs]) if rs else np.array([0])
        stats[sp] = {"docs": len(rs), "entities": dict(ents), "docs_without_entities": no_ent,
                     "tokens_median": int(np.median(ntok)), "tokens_p95": int(np.percentile(ntok, 95)), "tokens_max": int(ntok.max())}
        checks.append(check(f"lengths_{sp}", "FAIL" if bad_len else "PASS", f"{bad_len} وثيقة بطول رموز ≠ وسوم"))
        checks.append(check(f"tagset_{sp}", "FAIL" if bad_tag else "PASS", f"{bad_tag} وثيقة بوسوم خارج القائمة"))
        checks.append(check(f"bio_{sp}", "WARN" if invalid_i else "PASS", f"{invalid_i} وسم I- بلا B- قبله"))
        checks.append(check(f"long_docs_{sp}", "WARN" if ntok.max() > 2000 else "PASS",
                            f"أطول وثيقة {int(ntok.max())} كلمة (تُقطّع إلى مقاطع 180 كلمة أثناء التدريب)"))
    ids = {sp: {r["id"] for r in recs[sp]} for sp in SPLITS}
    tx = {sp: {_norm(" ".join(r["tokens"])) for r in recs[sp]} for sp in SPLITS}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        checks.append(check(f"leak_id_{a}_{b}", "FAIL" if ids[a] & ids[b] else "PASS", f"{len(ids[a] & ids[b])} معرّف مشترك"))
        checks.append(check(f"leak_text_{a}_{b}", "FAIL" if tx[a] & tx[b] else "PASS", f"{len(tx[a] & tx[b])} نص مشترك"))
        nd = {_norm(" ".join(r["tokens"]), False) for r in recs[a]} & {_norm(" ".join(r["tokens"]), False) for r in recs[b]}
        checks.append(check(f"near_dup_{a}_{b}", "WARN" if len(nd) > len(tx[a] & tx[b]) else "PASS", f"{len(nd)} نص متطابق بعد حذف الأرقام"))
    present = set().union(*[set(s["entities"]) for s in stats.values()])
    declared = {t[2:] for t in tags if t != "O"}
    absent = sorted(declared - present)
    if absent:
        checks.append(check("absent_entities", "WARN", f"أنواع معرّفة بلا أي مثال: {absent} — تُستبعد من التقييم تلقائياً"))
    for e in present:
        if stats["test"]["entities"].get(e, 0) < MIN_TEST_PER_CLASS:
            checks.append(check(f"entity_{e}", "WARN", f"{stats['test']['entities'].get(e, 0)} كيان فقط في test"))
    return {"task": "ner", "tags": tags, "entity_types_present": sorted(present), "stats": stats, "checks": checks}


def audit_gold(data_dir: Path) -> dict:
    g = data_dir / "gold_annotation"
    out = {}
    f = g / "opportunity_gold_to_annotate.csv"
    if f.exists():
        d = pd.read_csv(f, dtype=str, keep_default_na=False)
        cols = [c for c in d.columns if c.startswith("gold_")]
        out["classification"] = {"rows": len(d), **{c: int((d[c].str.strip() != "").sum()) for c in cols}}
    f = g / "ner_gold_to_annotate.jsonl"
    if f.exists():
        ex = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        out["ner"] = {"docs": len(ex), "annotated": sum(e.get("gold_spans") is not None for e in ex)}
    lab = sum(v for k, v in out.get("classification", {}).items() if k.startswith("gold_"))
    status = "PASS" if lab >= 100 else "WARN"
    return {**out, "checks": [check("gold_labelled", status,
                                    "لا يوجد توسيم بشري كافٍ — النتائج قياس اتفاق مع وسوم silver فقط" if status == "WARN" else "متوفر")]}


def audit_timeseries(data_dir: Path) -> dict:
    from src.forecasting.series import build_family
    out = {}
    for fam in ("skills", "specializations", "job_titles", "governorates"):
        try:
            r = build_family(fam, data_dir)
            out[fam] = {"status": r["status"], "months": r.get("months"), "window": r.get("window"),
                        "ready_series": len(r.get("ready", []))}
        except Exception as e:  # noqa: BLE001
            out[fam] = {"status": f"error: {e}"}
    ok = any(v["status"] == "ok" for v in out.values())
    return {**out, "checks": [check("lstm_sufficiency", "PASS" if ok else "WARN",
                                    "سلاسل كافية لعائلة واحدة على الأقل" if ok else "لا توجد سلاسل كافية لأي عائلة")]}


def to_md(rep: dict) -> str:
    L = [f"# تقرير تدقيق الـ Dataset\n\nالبيانات: `{rep['data_dir']}` · التاريخ: {rep['audited_at']}\n",
         f"**الحكم العام: {rep['verdict']}** — FAIL: {rep['counts']['FAIL']} · WARN: {rep['counts']['WARN']} · PASS: {rep['counts']['PASS']}\n"]
    for sec, v in rep["sections"].items():
        L.append(f"\n## {sec}\n")
        if "sizes" in v:
            L.append("التقسيمات: " + " · ".join(f"{k}={n:,}" for k, n in v["sizes"].items()) + "\n")
        if "stats" in v:
            for sp, s in v["stats"].items():
                L.append(f"- {sp}: {s['docs']:,} وثيقة · كيانات {s['entities']}\n")
        L.append("\n| الفحص | الحالة | التفاصيل |\n|---|---|---|\n")
        for c in v["checks"]:
            if c["status"] != "PASS" or sec != "ner":
                L.append(f"| {c['check']} | {c['status']} | {c['detail']} |\n")
    return "".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "audit")
    a = ap.parse_args()
    if not (a.data_dir / "unique_opportunities.csv").exists():
        raise SystemExit(f"✖ البيانات غير موجودة: {a.data_dir}")
    sec = {"opportunity_type": audit_classification("opportunity_type", a.data_dir),
           "geographic_scope": audit_classification("geographic_scope", a.data_dir),
           "ner": audit_ner(a.data_dir), "gold": audit_gold(a.data_dir), "timeseries": audit_timeseries(a.data_dir)}
    cnt = Counter(c["status"] for s in sec.values() for c in s["checks"])
    rep = {"data_dir": str(a.data_dir), "audited_at": now(), "verdict": "FAIL" if cnt["FAIL"] else ("WARN" if cnt["WARN"] else "PASS"),
           "counts": {k: cnt.get(k, 0) for k in ("FAIL", "WARN", "PASS")}, "sections": sec}
    write_json(a.out / "audit_report.json", rep)
    (a.out / "audit_report.md").write_text(to_md(rep), encoding="utf-8")
    for s, v in sec.items():
        for c in v["checks"]:
            if c["status"] != "PASS":
                print(f"[{c['status']}] {s}.{c['check']}: {c['detail']}")
    print(f"الحكم: {rep['verdict']} {rep['counts']} → {a.out}")


if __name__ == "__main__":
    main()
