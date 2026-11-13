"""يولّد docs/RESULTS.md من ملفات التقييم المحفوظة فقط — كل رقم في التقرير مصدره ملف في reports/.

    python scripts/build_results_section.py

إن غاب ملف يُكتب «لم يُنفَّذ بعد» بدل أي رقم.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import REPORTS_DIR, now  # noqa: E402

EV = REPORTS_DIR / "evaluation"
NA = "_لم يُنفَّذ بعد — لا توجد نتائج محفوظة._\n"


def J(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def f(x):
    return "—" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))


def cls_section(task, title):
    m, b = J(EV / f"classification_{task}" / "metrics.json"), J(EV / f"baseline_{task}" / "metrics.json")
    out = [f"\n### {title}\n"]
    if not m and not b:
        return out + [NA]
    out.append("| النموذج | Accuracy | Macro-P | Macro-R | Macro-F1 | Weighted-F1 |\n|---|---|---|---|---|---|\n")
    if m:
        out.append(f"| **AraBERT** | {f(m['accuracy'])} | {f(m['macro_precision'])} | {f(m['macro_recall'])} | **{f(m['macro_f1'])}** | {f(m['weighted_f1'])} |\n")
    for k in ("tfidf_logreg", "tfidf_linearsvc"):
        if b and k in b:
            t = b[k]["test"]
            out.append(f"| {k} (C={b[k]['best_C_on_validation']}) | {f(t['accuracy'])} | {f(t['macro_precision'])} | "
                       f"{f(t['macro_recall'])} | {f(t['macro_f1'])} | {f(t['weighted_f1'])} |\n")
    if m:
        out.append(f"\nعدد أمثلة test = {m['n']}. ")
        if m.get("per_class"):
            out.append("\n\n| الفئة | P | R | F1 | Support |\n|---|---|---|---|---|\n")
            out += [f"| {c['label']} | {f(c['precision'])} | {f(c['recall'])} | {f(c['f1'])} | {c['support']} |\n" for c in m["per_class"]]
        if m.get("gold"):
            out.append(f"\n**على Gold البشري:** {json.dumps({k: v for k, v in m['gold'].items() if k in ('n', 'accuracy', 'macro_f1', 'status')}, ensure_ascii=False)}\n")
        img = EV / f"classification_{task}" / "confusion_matrix.png"
        if img.exists():
            out.append(f"\n![مصفوفة الالتباس]({img.relative_to(ROOT) if img.is_relative_to(ROOT) else img})\n")
    return out


def ner_section():
    m, b = J(EV / "ner" / "metrics.json"), J(EV / "baseline_ner" / "metrics.json")
    out = ["\n### استخراج الكيانات (NER)\n"]
    if not m and not b:
        return out + [NA]
    out.append("| النموذج | Entity-P | Entity-R | Entity-F1 | Macro Entity-F1 |\n|---|---|---|---|---|\n")
    if m:
        out.append(f"| **AraBERT** | {f(m['entity_precision'])} | {f(m['entity_recall'])} | **{f(m['entity_f1'])}** | {f(m['macro_entity_f1'])} |\n")
    if b:
        t = b["crf"]["test"]
        out.append(f"| CRF | {f(t['entity_precision'])} | {f(t['entity_recall'])} | {f(t['entity_f1'])} | {f(t['macro_entity_f1'])} |\n")
    if b and not m:
        out.append("\nCRF لكل كيان:\n\n| الكيان | P | R | F1 | Support |\n|---|---|---|---|---|\n")
        out += [f"| {e['entity']} | {f(e['precision'])} | {f(e['recall'])} | {f(e['f1'])} | {e['support']} |\n" for e in b["crf"]["test"]["per_entity"]]
    if m:
        out.append("\n| الكيان | P | R | F1 | Support |\n|---|---|---|---|---|\n")
        out += [f"| {e['entity']} | {f(e['precision'])} | {f(e['recall'])} | {f(e['f1'])} | {e['support']} |\n" for e in m["per_entity"]]
        er = m.get("errors", {})
        out.append(f"\nتحليل الأخطاء: فائتة {er.get('missed')} · زائدة {er.get('spurious')} · حدود خاطئة {er.get('boundary')}\n")
    return out


def best(t):
    c = {k: (t.get(k) or {}).get("mae") for k in ("lstm", "naive", "moving_avg_3")}
    c = {k: v for k, v in c.items() if v is not None}
    return {"lstm": "LSTM", "naive": "Naive", "moving_avg_3": "MA-3"}[min(c, key=c.get)] if c else "—"


def lstm_section():
    s = J(EV / "lstm_summary.json")
    out = ["\n### التنبؤ الزمني (LSTM)\n"]
    if not s:
        return out + [NA]
    out.append("| العائلة | دُرّب | السلاسل | LSTM MAE | Naive MAE | MA-3 MAE | LSTM RMSE | الأفضل (MAE) | السبب إن لم يُدرَّب |\n|---|---|---|---|---|---|---|---|---|\n")
    for fam, v in s.items():
        if not isinstance(v, dict) or "trained" not in v:
            continue
        t = v.get("test") or {}
        g = lambda k, x: (t.get(k) or {}).get(x)  # noqa: E731
        out.append(f"| {fam} | {'نعم' if v['trained'] else 'لا'} | {v.get('series', '—')} | {f(g('lstm', 'mae'))} | "
                   f"{f(g('naive', 'mae'))} | {f(g('moving_avg_3', 'mae'))} | {f(g('lstm', 'rmse'))} | {best(t)} | {v.get('reason') or ''} |\n")
    return out


def main():
    a = J(REPORTS_DIR / "audit" / "audit_report.json")
    L = [f"# النتائج\n\n_مولَّد تلقائياً من `reports/` في {now()} — لا أرقام مكتوبة يدوياً._\n",
         "\nكل النتائج على تقسيم **test** المستقل (لم يُستخدم في التدريب أو اختيار النموذج). "
         "الوسوم silver (قواعد)، لذا المقاييس تقيس الاتفاق مع هذه الوسوم ما لم يُذكر Gold.\n"]
    L.append("\n### تدقيق البيانات\n")
    L.append(f"الحكم: **{a['verdict']}** — {a['counts']}\n" if a else NA)
    L += cls_section("opportunity_type", "تصنيف نوع الفرصة")
    L += cls_section("geographic_scope", "تصنيف النطاق الجغرافي")
    L += ner_section()
    L += lstm_section()
    out = ROOT / "docs" / "RESULTS.md"
    out.write_text("".join(L), encoding="utf-8")
    print(f"✔ {out}")


if __name__ == "__main__":
    main()
