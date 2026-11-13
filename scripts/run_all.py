"""تشغيل كامل وآلي لتجهيز الـ Dataset — خلية Colab واحدة تستدعي هذا الملف.

    python -m scripts.run_all [--ocr-minutes 180] [--skills auto] [--skip-telegraph] [--skip-ocr]

المراحل (كل مرحلة قابلة للاستئناف؛ فشل مرحلة جمع لا يوقف المراحل التالية):
  0) فحص البيئة والأرشيف (لا جمع للمنشورات — يُستخدم المجمّع سابقاً)
  1) استكمال صفحات telegra.ph (دفعات + إعادة محاولة للفشل المؤقت حتى 3 جولات)
  2) OCR لمنشورات الصور بلا نص (بميزانية زمنية؛ يكمل لاحقاً من حيث توقف)
  3) build_dataset   4) finalize_dataset → FINAL_TRAINING_DATASET   5) validate_final_dataset
  6) evaluate_dataset (فقط إن كانت عينة Gold مُعبأة)   7) FINAL_REPORT.md
كل المخرجات والسجل في $LMI_FINAL_DIR (افتراضياً <ROOT>/FINAL_TRAINING_DATASET).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

from core.config import LABELS_DIR, RAW_DIR, ROOT

FINAL = Path(os.environ.get("LMI_FINAL_DIR", ROOT / "FINAL_TRAINING_DATASET"))
LOG = ROOT / "reports" / "run_all.log"
CHANNELS = ["hr_yemen", "job_yemen", "TheYemenGate", "jobsforyemeni", "forus_taiz"]
stage_results = []


def say(msg):
    line = f"{datetime.now():%H:%M:%S} | {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def stage(name, fn, critical=False):
    say(f"══ {name}")
    t = time.time()
    try:
        info = fn() or {}
        stage_results.append({"stage": name, "status": "ok", "seconds": round(time.time() - t), **info})
        say(f"   ✔ {name} ({time.time() - t:.0f} ث) {json.dumps(info, ensure_ascii=False, default=str)[:400]}")
        return True
    except SystemExit as e:
        ok = e.code in (0, None)
        stage_results.append({"stage": name, "status": "ok" if ok else f"exit {e.code}", "seconds": round(time.time() - t)})
        say(("   ✔ " if ok else "   ✖ ") + f"{name}: exit {e.code}")
        if not ok and critical:
            raise
        return ok
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        stage_results.append({"stage": name, "status": "error", "error": f"{type(e).__name__}: {e}", "seconds": round(time.time() - t)})
        say(f"   ✖ {name}: {type(e).__name__}: {e}\n{tb[-1500:]}")
        if critical:
            raise
        return False


def sub(module, *args):
    """تشغيل وحدة كعملية مستقلة مع بث مخرجاتها للسجل."""
    p = subprocess.Popen([sys.executable, "-u", "-m", module, *args], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    tail = []
    for line in p.stdout:
        line = line.rstrip()
        tail = (tail + [line])[-40:]
        if line:
            say("   " + line)
    p.wait()
    if p.returncode:
        raise RuntimeError(f"{module} exit {p.returncode}: " + " | ".join(tail[-5:]))
    return {"returncode": 0}


def check_env():
    tg = RAW_DIR / "telegram"
    counts = {}
    for ch in CHANNELS:
        f = tg / f"{ch}.csv"
        if not f.exists():
            raise SystemExit(f"✖ أرشيف القناة مفقود: {f} — تأكد من ربط Drive (لا يُعاد الجمع تلقائياً)")
        counts[ch] = len(pd.read_csv(f, usecols=["source_message_id"], dtype=str))
    if os.environ.get("LMI_MODE", "real") == "demo":
        raise SystemExit("✖ LMI_MODE=demo")
    return {"posts": counts, "total": sum(counts.values()), "raw_is_link": os.path.islink(ROOT / "data" / "raw")}


def telegraph_stage(delay):
    from scripts.collectors import telegraph
    last = None
    for rnd in range(1, 4):
        st = telegraph.run(["hr_yemen"], None, max(delay, 2.0), retry_failed=False)
        say(f"   جولة {rnd}: حاولت {st.attempted} | نجحت {st.saved_ok} | فشلت {st.failed} | المتبقي {st.remaining}")
        if st.remaining == 0 or st.attempted == 0 or (last is not None and st.saved_ok == 0):
            break
        last = st
        time.sleep(30)
    pages = TG = RAW_DIR / "telegram" / "telegraph" / "pages.csv"
    n = len(pd.read_csv(TG, usecols=["telegraph_path"])) if pages.exists() else 0
    return {"pages_saved": n, "remaining": st.remaining, "permanent_failures": st.skipped_permanent_failures}


def ocr_stage(minutes, delay):
    import shutil
    if not shutil.which("tesseract"):
        raise RuntimeError("tesseract غير مثبت (الخلية تثبته: apt-get install tesseract-ocr tesseract-ocr-ara)")
    from scripts.collectors import telegram_media_ocr
    return telegram_media_ocr.run(CHANNELS, budget_minutes=minutes, delay=delay)


def final_report():
    st = FINAL / "dataset_statistics"
    S = json.loads((st / "summary.json").read_text(encoding="utf-8"))
    Q = json.loads((st / "quality_checks.json").read_text(encoding="utf-8"))
    ss = pd.read_csv(st / "split_sizes.csv")
    fr = pd.read_csv(st / "file_row_counts.csv")
    piv = ss.pivot_table(index="task", columns="split", values="rows", aggfunc="sum").reindex(columns=["train", "validation", "test"])
    unresolved = [f"{c['check']} — {c['detail']}" for c in Q["checks"] if not c["passed"]]
    if not S.get("gold_annotated_rows"):
        unresolved.append("عينة Gold (300) غير مصنفة بشرياً: لا رقم دقة صالح للتقرير العلمي قبل تعبئتها (تدخل بشري لا يمكن أتمتته).")
    for r in stage_results:
        if r["status"] != "ok":
            unresolved.append(f"مرحلة {r['stage']}: {r.get('error', r['status'])}")
    tg = next((r for r in stage_results if r["stage"].startswith("1)")), {})
    if tg.get("remaining"):
        unresolved.append(f"صفحات telegra.ph متبقية: {tg['remaining']} (أعد تشغيل الخلية لاستكمالها).")
    oc = next((r for r in stage_results if r["stage"].startswith("2)")), {})
    if oc.get("remaining"):
        unresolved.append(f"منشورات صور لم تُعالج بـ OCR بعد: {oc['remaining']} (الميزانية الزمنية انتهت؛ إعادة التشغيل تكمل).")
    ts_rows = int(fr[fr.file == "timeseries/monthly_skill_counts.csv"].rows.sum()) if len(fr) else 0
    L = ["# التقرير النهائي — تجهيز Dataset سوق العمل اليمني", f"_{datetime.now():%Y-%m-%d %H:%M}_ — المسار: `{FINAL}`", "",
         "> كل التسميات **Silver** (قواعد آلية بسبب وثقة). لم يُصنَّف أي جزء بشرياً بعد.", "",
         "## الأعداد",
         f"- المنشورات النهائية: **{S['posts_total']:,}** (حسب مصدر النص: {S['posts_by_text_source']}) — مستبعدة: {S['posts_excluded']:,}",
         f"- الفرص المستخرجة (بعد تقسيم المنشورات متعددة الفرص): **{S['opportunity_records']:,}**",
         f"- الإعلانات الفريدة (6 أنواع): **{S['unique_opportunities_6types']:,}** — وظائف/تدريب عملي فريدة: {S['unique_jobs_internships']:,}",
         f"- السجلات المكررة (نسخ زائدة مرتبطة بأصلها): **{S['duplicate_records']:,}** — تكرار محتمل للمراجعة: {S['possible_duplicate_review']:,}",
         f"- توزيع الأنواع (فريدة): {S['type_distribution_unique']}",
         f"- النطاق التفصيلي (فريدة): {S['scope_detail_unique']}", "",
         "## تغطية الحقول (وظائف فريدة)", *(f"- {k}: {v:.1%}" for k, v in S["field_coverage_unique_jobs"].items()), "",
         f"## NER\n- أمثلة: **{S['ner_examples']:,}** — كيانات: {S['ner_entity_counts']}", "",
         f"## المهارات\n- أزواج (إعلان، مهارة) مقبولة: **{S['skills_accepted_pairs']:,}** — مهارات مختلفة: {S['skills_distinct']} — للمراجعة: {S['skills_review_pairs']:,}", "",
         f"## السلاسل الزمنية\n- أشهر: {S['timeseries_months']} — صفوف (شهر×مهارة): **{ts_rows:,}**",
         f"- LSTM: نافذة {S['lstm'].get('window')} ({S['lstm'].get('window_months')} شهراً، تغطية {S['lstm'].get('window_coverage')}) — "
         f"مهارات جاهزة: **{S['lstm'].get('skills_ready')}** من {S['lstm'].get('skills_total')} — تقسيم: {S['lstm'].get('lstm_split')}", "",
         "## أحجام Train / Validation / Test", piv.fillna(0).astype(int).to_markdown(), "",
         f"## فحوص الجودة: {Q['passed']}/{Q['total']} — إخفاقات حرجة: {Q['critical_failures']}",
         *(f"- {'✔' if c['passed'] else ('✖' if c['critical'] else '⚠')} {c['check']}" for c in Q["checks"]), "",
         "## مراحل التشغيل", *(f"- {r['stage']}: {r['status']} ({r['seconds']} ث)" for r in stage_results), "",
         "## مشكلات لم تُحل", *(f"- {u}" for u in unresolved or ["لا يوجد"]), "",
         "## الملفات الرئيسية", "`dataset_quality_report.html` · `dataset_manifest.json` · `data_dictionary.md` · `gold_annotation/ANNOTATION_GUIDE.md`"]
    (FINAL / "FINAL_REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))
    return {"critical_failures": Q["critical_failures"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-minutes", type=float, default=180)
    ap.add_argument("--skills", default="auto", choices=["auto", "hybrid", "lexicon", "none"])
    ap.add_argument("--delay", type=float, default=3.0)
    ap.add_argument("--skip-telegraph", action="store_true")
    ap.add_argument("--skip-ocr", action="store_true")
    a = ap.parse_args()
    os.environ["LMI_FINAL_DIR"] = str(FINAL)
    say(f"بدء التشغيل الكامل → {FINAL}")
    stage("0) فحص البيئة والأرشيف", check_env, critical=True)
    if not a.skip_telegraph:
        stage("1) صفحات telegra.ph", lambda: telegraph_stage(a.delay))
    if not a.skip_ocr:
        stage("2) OCR لمنشورات الصور", lambda: ocr_stage(a.ocr_minutes, a.delay))
    stage("3) build_dataset", lambda: sub("scripts.build_dataset", "--skills", a.skills), critical=True)
    stage("4) finalize_dataset", lambda: sub("scripts.finalize_dataset", "--out", str(FINAL)), critical=True)
    stage("5) validate_final_dataset", lambda: sub("scripts.validate_final_dataset", "--dir", str(FINAL)))
    g = LABELS_DIR / "opportunity_gold_to_annotate.csv"
    if g.exists() and (pd.read_csv(g, dtype=str, keep_default_na=False).gold_opportunity_type.str.strip() != "").sum() >= 20:
        stage("6) evaluate_dataset (Gold)", lambda: sub("scripts.evaluate_dataset"))
    stage("7) FINAL_REPORT.md", final_report)
    try:
        import shutil
        shutil.copy(LOG, FINAL / "run_all.log")
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
