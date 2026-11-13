"""تحميل النماذج المدربة المحفوظة والتنبؤ بها — طبقة واحدة يستخدمها التقييم والمحرك والتطبيق (بلا تكرار للكود).

TextClassifier : AraBERT مُدرَّب على تصنيف التسلسل
NERTagger      : AraBERT مُدرَّب على تصنيف الرموز (BIO) مع محاذاة الكلمات ↔ الرموز الفرعية وتقطيع النصوص الطويلة
لا قواعد بديلة هنا: إن غاب النموذج يُرفع ModelNotAvailable صريحاً.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from src.data.loaders import arabert_preprocess

TOKEN_RX = re.compile(r"[^\s.,:;!?؟،؛()\[\]{}\"«»|]+|[.,:;!?؟،؛()\[\]{}\"«»|]")   # نفس تقطيع بيانات NER


def word_windows(tokens, tokenizer, max_words=180, max_length=512):
    """Split by words AND model subtoken budget; preserve every input word.

    Uses fast-tokenizer word_ids to count exact subtokens. Raises on individual
    words that cannot fit; never silently drop gold labels or inference tokens.
    """
    if not tokens:
        return []
    if max_words < 1:
        raise ValueError("max_words must be >= 1")
    if tokenizer is None or max_length is None:
        return [(i, min(i + max_words, len(tokens))) for i in range(0, len(tokens), max_words)]
    if not tokenizer.is_fast:
        raise ValueError("NER requires a fast tokenizer")
    budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if budget < 1:
        raise ValueError("max_length too small for tokenizer special tokens")
    encoded = tokenizer(tokens, is_split_into_words=True, add_special_tokens=False, truncation=False)
    counts = [0] * len(tokens)
    for wi in encoded.word_ids():
        if wi is not None:
            counts[wi] += 1
    if any(c == 0 for c in counts):
        raise ValueError("Tokenizer produced zero subtokens for at least one word; cannot align NER reliably")
    if any(c > budget for c in counts):
        raise ValueError("A word exceeds the configured model token budget")
    windows = []
    start, size, words = 0, 0, 0
    for idx, count in enumerate(counts):
        if idx > start and (size + count > budget or words >= max_words):
            windows.append((start, idx))
            start, size, words = idx, 0, 0
        size += count
        words += 1
    if start < len(tokens):
        windows.append((start, len(tokens)))
    return windows


class ModelNotAvailable(RuntimeError):
    pass


def _check_dir(d: Path):
    d = Path(d)
    if not (d / "config.json").exists():
        raise ModelNotAvailable(f"لا يوجد نموذج مدرب في {d} (config.json مفقود). درّب النموذج أولاً أو انسخ مجلده إلى هذا المسار.")
    weights = ["model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"]
    if not any((d / w).is_file() for w in weights):
        raise ModelNotAvailable(f"Checkpoint incomplete at {d}: model weights are missing")
    if not any((d / t).exists() for t in ["tokenizer.json", "vocab.txt", "spiece.model", "tokenizer.model"]):
        raise ModelNotAvailable(f"Checkpoint incomplete at {d}: tokenizer files are missing")
    return d


def _base_model(d: Path) -> str:
    f = d / "training_config.json"
    return json.loads(f.read_text(encoding="utf-8")).get("base_model", "") if f.exists() else ""


class TextClassifier:
    def __init__(self, model_dir, max_length: int | None = None):
        self.dir = _check_dir(model_dir)
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(self.dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.dir).float().eval()   # أوزان fp16 المصدّرة تُحوَّل إلى fp32 للاستدلال
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.dev)
        cfg = self.model.config
        self.labels = [cfg.id2label[i] for i in range(cfg.num_labels)]
        tc = self.dir / "training_config.json"
        self.train_cfg = json.loads(tc.read_text(encoding="utf-8")) if tc.exists() else {}
        self.max_length = max_length or self.train_cfg.get("max_length", 256)
        self.base = _base_model(self.dir)

    def predict_proba(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        import torch
        texts = arabert_preprocess([t or "" for t in texts], self.base)
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tok(texts[i:i + batch_size], truncation=True, max_length=self.max_length, padding=True,
                               return_tensors="pt").to(self.dev)
                out.append(torch.softmax(self.model(**enc).logits.float(), -1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)))

    def predict(self, text: str, top_k: int = 3) -> dict:
        p = self.predict_proba([text])[0]
        order = p.argsort()[::-1]
        return {"label": self.labels[order[0]], "confidence": float(p[order[0]]),
                "top": [{"label": self.labels[i], "probability": float(p[i])} for i in order[:top_k]]}


class NERTagger:
    def __init__(self, model_dir, max_words: int = 180, max_length: int = 512):
        self.dir = _check_dir(model_dir)
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(self.dir)
        if not getattr(self.tok, "is_fast", False):
            raise ModelNotAvailable("NER يتطلب Fast Tokenizer لمحاذاة الكلمات (word_ids)")
        self.model = AutoModelForTokenClassification.from_pretrained(self.dir).float().eval()   # أوزان fp16 المصدّرة تُحوَّل إلى fp32 للاستدلال
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.dev)
        cfg = self.model.config
        self.labels = [cfg.id2label[i] for i in range(cfg.num_labels)]
        self.max_words, self.max_length = max_words, max_length

    @classmethod
    def from_objects(cls, model, tok, max_words: int = 180, max_length: int = 512):
        """للتقييم أثناء التدريب (نفس منطق المحاذاة والتقطيع المستخدم في الاستدلال)."""
        obj = cls.__new__(cls)
        obj.model, obj.tok, obj.dev = model, tok, next(model.parameters()).device
        obj.labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
        obj.max_words, obj.max_length = max_words, max_length
        return obj

    def tag_tokens(self, tokens_list: list[list[str]], batch_size: int = 16) -> tuple[list[list[str]], list[list[float]]]:
        """وسم قائمة جمل مُقطّعة إلى كلمات. كل جملة تُقسم إلى مقاطع ≤ max_words كلمة، ووسم الكلمة = وسم أول رمز فرعي لها."""
        import torch
        chunks, owner = [], []
        for si, toks in enumerate(tokens_list):
            for s, e in word_windows(toks, self.tok, self.max_words, self.max_length):
                chunks.append(toks[s:e])
                owner.append((si, s))
        tags = [["O"] * len(t) for t in tokens_list]
        confs = [[0.0] * len(t) for t in tokens_list]
        with torch.no_grad():
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                enc = self.tok(batch, is_split_into_words=True, truncation=True, max_length=self.max_length, padding=True,
                               return_tensors="pt")
                probs = torch.softmax(self.model(**{k: v.to(self.dev) for k, v in enc.items()}).logits.float(), -1).cpu().numpy()
                for bi in range(len(batch)):
                    si, start = owner[i + bi]
                    seen = set()
                    for ti, wid in enumerate(enc.word_ids(bi)):
                        if wid is None or wid in seen:
                            continue                      # رموز خاصة/رموز فرعية لاحقة
                        seen.add(wid)
                        k = int(probs[bi, ti].argmax())
                        tags[si][start + wid] = self.labels[k]
                        confs[si][start + wid] = float(probs[bi, ti, k])
        return [fix_bio(t) for t in tags], confs

    def extract(self, text: str) -> dict:
        from src.evaluation.metrics import extract_entities
        toks = [m.group(0) for m in TOKEN_RX.finditer(text or "")]
        if not toks:
            return {"tokens": [], "tags": [], "entities": []}
        tags, confs = self.tag_tokens([toks])
        ents = extract_entities(toks, tags[0])
        for e in ents:
            e["confidence"] = round(float(np.mean(confs[0][e["start"]:e["end"]])), 4)
        return {"tokens": toks, "tags": tags[0], "entities": ents}


def fix_bio(tags: list[str]) -> list[str]:
    """تصحيح تسلسل غير صالح من النموذج: I-X بعد O أو بعد نوع آخر يصبح B-X (معالجة قياسية قبل تقييم seqeval الصارم)."""
    out, prev = [], "O"
    for t in tags:
        if t.startswith("I-") and prev[2:] != t[2:]:
            t = "B-" + t[2:]
        out.append(t)
        prev = t
    return out
