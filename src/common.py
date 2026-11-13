"""أدوات مشتركة: المسارات، البذور، الجهاز، القراءة/الكتابة. لا منطق نموذج هنا."""
from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config" / "training.json").read_text(encoding="utf-8"))
# مجلد الـ Dataset النهائي (FINAL_TRAINING_DATASET) — يُحدَّد بمتغير بيئة أو المسار الافتراضي داخل المشروع
DATA_DIR = Path(os.environ.get("LMI_FINAL_DIR", ROOT / "FINAL_TRAINING_DATASET"))
MODELS_DIR = Path(os.environ.get("LMI_MODELS_DIR", ROOT / "models_dl"))
REPORTS_DIR = Path(os.environ.get("LMI_REPORTS_DIR", ROOT / "reports"))


def set_seed(seed: int = CONFIG["seed"]):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def env_info() -> dict:
    import platform
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch
        info.update(torch=torch.__version__, cuda=torch.cuda.is_available(),
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except ImportError:
        pass
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    return info
