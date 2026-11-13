"""وضع العرض التجريبي المنفصل (DEMO) — بيانات اصطناعية بالكامل لتوضيح عمل النظام فقط.

    python -m scripts.train_demo
    LMI_MODE=demo streamlit run app.py

يقرأ data/demo فقط ويكتب في data/processed_demo و artifacts_demo — لا يلمس البيانات أو النماذج الحقيقية.
كل نتائجه موسومة "تجريبية" في الواجهة والتقارير ولا تُعد توقعات أو مقاييس حقيقية.
"""
import os
import sys

os.environ["LMI_MODE"] = "demo"

from core.config import DEMO_INPUT_DIR, IS_DEMO_MODE  # noqa: E402

assert IS_DEMO_MODE

if __name__ == "__main__":
    if not (DEMO_INPUT_DIR / "demo_jobs.csv").exists():
        from preprocessing.taxonomy_seed import write_seed
        from scripts.build_demo_data import generate
        write_seed()
        generate(3500).to_csv(DEMO_INPUT_DIR / "demo_jobs.csv", index=False, encoding="utf-8-sig")
        print("[i] تم توليد data/demo/demo_jobs.csv (اصطناعي)")
    from scripts import train_all
    sys.argv = [sys.argv[0], *sys.argv[1:]]
    train_all.main()
