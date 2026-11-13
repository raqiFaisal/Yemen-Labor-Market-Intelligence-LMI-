"""اختبار كل صفحات Streamlit دون متصفح (AppTest) + الأزرار الرئيسية + منطق رفع CSV/Excel.

    python -m tests.test_pages                 # الوضع الحقيقي
    LMI_MODE=demo python -m tests.test_pages   # وضع العرض التجريبي
"""
import io
import os
import sys
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
PAGES = ["1_dashboard", "2_job_market", "3_skills", "4_forecast", "9_models", "5_academic", "6_gap", "7_custom",
         "8_reports", "10_review"]
BUTTONS = {"تحليل الإعلان", "حساب الترتيب", "تحليل الفجوة"}


def run_pages():
    bad = 0
    for p in PAGES:
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)
        at.run()
        if p != PAGES[0]:
            at.switch_page(f"pages/{p}.py")
            at.run()
        errs = [e.value for e in at.exception]
        for b in list(at.button):
            if b.label in BUTTONS:
                b.click()
                at.run()
                errs += [e.value for e in at.exception]
        stopped = [m.value for m in at.error][:1]
        status = "OK" if not errs else "ERROR"
        bad += bool(errs)
        print(f"{p:14s} {status} {'| رسالة: ' + stopped[0][:70] if stopped else ''} {[str(e)[:300] for e in errs][:1]}")
    return bad


def run_upload():
    sys.path.insert(0, str(ROOT))
    from core.pipeline import extract_job_skills, map_domain
    from preprocessing.schema import map_columns, unify
    raw = pd.DataFrame({"المسمى الوظيفي": ["محاسب أول", "Data Analyst"], "الوصف": ["خبرة في IFRS وإكسل", "SQL, Python, Power BI"],
                        "التصنيف": ["Finance/Accounting", "Computers/IT"], "المحافظة": ["عدن", "صنعاء"]})
    ok = True
    for kind in ("csv", "xlsx"):
        buf = io.BytesIO()
        raw.to_csv(buf, index=False, encoding="utf-8-sig") if kind == "csv" else raw.to_excel(buf, index=False)
        buf.seek(0)
        df_in = pd.read_csv(buf, encoding="utf-8-sig") if kind == "csv" else pd.read_excel(buf)
        df = unify(df_in, source="upload", country="Yemen", mapping=map_columns(df_in))
        df["domain"] = df["category_raw"].map(map_domain)
        js = extract_job_skills(df)
        good = len(df) == 2 and set(df.domain) == {"accounting", "software_engineering"} and len(js) >= 4
        ok &= good
        print(f"رفع {kind}: {'OK' if good else 'ERROR'} | سجلات {len(df)} | مهارات {sorted(js.skill_id)}")
    return 0 if ok else 1


if __name__ == "__main__":
    print("الوضع:", os.environ.get("LMI_MODE", "real"))
    n = run_pages() + run_upload()
    print("✔ كل الاختبارات نجحت" if n == 0 else f"✖ {n} فشل")
    sys.exit(1 if n else 0)
