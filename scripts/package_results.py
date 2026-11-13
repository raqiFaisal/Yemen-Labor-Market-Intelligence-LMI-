"""تجميع نتائج التدريب للنقل من Colab إلى الكمبيوتر (بدون Google Drive).

    python scripts/package_results.py [--out /content] [--fp16]

ينتج ملفين منفصلين حتى لا يضيع أحدهما إن فشل تنزيل الآخر:
  lmi_reports.zip : reports/ كاملة + ملفات إعداد وسجل تدريب كل نموذج (بدون أوزان) — صغير
  lmi_models.zip  : models_dl/ بالأوزان (مع --fp16 تُحفظ الأوزان بنصف الدقة: حجم أقل ≈ النصف، والتطبيق يحولها إلى fp32 عند التحميل)
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import MODELS_DIR, REPORTS_DIR  # noqa: E402

WEIGHT_EXT = {".safetensors", ".bin", ".pt", ".joblib"}


def zip_dir(src: Path, zf: zipfile.ZipFile, arc_root: str, skip_weights: bool = False):
    for f in sorted(src.rglob("*")):
        if f.is_file() and not (skip_weights and f.suffix in WEIGHT_EXT):
            zf.write(f, f"{arc_root}/{f.relative_to(src)}")


def fp16_copy(src: Path, dst: Path):
    """نسخة من models_dl بأوزان AraBERT بنصف الدقة؛ بقية الملفات (LSTM/baselines/tokenizer/json) تُنسخ كما هي."""
    import torch
    from safetensors.torch import load_file, save_file
    shutil.copytree(src, dst)
    for st in dst.rglob("model.safetensors"):
        w = load_file(st)
        save_file({k: (v.half() if v.is_floating_point() else v).contiguous() for k, v in w.items()}, st, metadata={"format": "pt"})
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT)
    ap.add_argument("--fp16", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rep = a.out / "lmi_reports.zip"
    with zipfile.ZipFile(rep, "w", zipfile.ZIP_DEFLATED) as zf:
        if REPORTS_DIR.exists():
            zip_dir(REPORTS_DIR, zf, "reports")
        if MODELS_DIR.exists():
            zip_dir(MODELS_DIR, zf, "models_dl", skip_weights=True)
    print(f"✔ {rep}  {rep.stat().st_size / 1e6:.1f} MB")
    if not MODELS_DIR.exists():
        print("✖ لا يوجد models_dl — لم يُدرَّب أي نموذج")
        return
    src = MODELS_DIR
    tmp = None
    if a.fp16:
        tmp = Path(tempfile.mkdtemp())
        src = fp16_copy(MODELS_DIR, tmp / "models_dl")
    mod = a.out / "lmi_models.zip"
    with zipfile.ZipFile(mod, "w", zipfile.ZIP_STORED) as zf:      # الأوزان لا تنضغط تقريباً — STORED أسرع
        zip_dir(src, zf, "models_dl")
    print(f"✔ {mod}  {mod.stat().st_size / 1e6:.1f} MB{' (fp16)' if a.fp16 else ''}")
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
