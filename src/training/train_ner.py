"""Fine-tuning لـ AraBERT على Token Classification (NER بوسوم BIO) لإعلانات سوق العمل.

    python -m src.training.train_ner [--model aubmindlab/bert-base-arabertv02] [--epochs 6]

التنفيذ:
  • الوسوم والكيانات تُقرأ من ner/label_list.json (لا افتراض). النصوص مقطّعة إلى كلمات مسبقاً.
  • المحاذاة: is_split_into_words → word_ids؛ أول رمز فرعي لكل كلمة يأخذ وسمها، الرموز الفرعية اللاحقة والرموز الخاصة
    والحشو = -100 (لا تدخل حساب Loss).
  • الجمل الطويلة تُقسم إلى مقاطع ≤ max_words كلمة (حتى لا تُقص كيانات عند 512 رمزاً).
  • أفضل Checkpoint بـ Entity-F1 (seqeval صارم) على validation فقط. test لا يُقرأ هنا.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import CONFIG, DATA_DIR, MODELS_DIR, device, env_info, now, set_seed, write_json
from src.data.loaders import load_ner
from src.evaluation.metrics import entity_metrics
from src.inference.predictors import NERTagger


def chunk(recs, max_words, tok=None, max_length=None):
    """Produce token-budget-safe windows; never silently truncate supervised entities."""
    from src.inference.predictors import word_windows
    out = []
    for r in recs:
        for start, end in word_windows(r["tokens"], tok, max_words, max_length):
            out.append((r["tokens"][start:end], r["ner_tags"][start:end]))
    return out


def align(tok, chunks, label2id, max_length):
    enc = tok([c[0] for c in chunks], is_split_into_words=True, truncation=True, max_length=max_length)
    labels, lost = [], 0
    for i, (_, tags) in enumerate(chunks):
        wids, prev, lab = enc.word_ids(i), None, []
        covered = set()
        for w in wids:
            if w is None:
                lab.append(-100)
            elif w != prev:
                lab.append(label2id[tags[w]])
                covered.add(w)
            else:
                lab.append(-100)
            prev = w
        lost += sum(1 for w, t in enumerate(tags) if w not in covered and t != "O")   # كيانات قُصّت بالحد الأقصى
        labels.append(lab)
    enc["labels"] = labels
    return enc, lost


class _DS:
    def __init__(self, enc):
        self.enc = enc

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        return {k: self.enc[k][i] for k in self.enc.keys()}


def main():
    import torch
    from torch.utils.data import DataLoader
    from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                              get_linear_schedule_with_warmup)
    C = CONFIG["ner"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=CONFIG["arabert_model"])
    ap.add_argument("--data_dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=MODELS_DIR / "ner")
    ap.add_argument("--epochs", type=int, default=C["epochs"])
    ap.add_argument("--lr", type=float, default=C["learning_rate"])
    ap.add_argument("--batch_size", type=int, default=C["batch_size"])
    ap.add_argument("--max_words", type=int, default=C["max_words_per_chunk"])
    ap.add_argument("--patience", type=int, default=C["early_stopping_patience"])
    ap.add_argument("--max_train", type=int)
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])
    a = ap.parse_args()
    set_seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    dev = device()
    amp = dev.type == "cuda"
    data, tags = load_ner(a.data_dir, splits=["train", "validation"])          # ← test غير مُحمَّل
    tr, va = data["train"], data["validation"]
    if a.max_train:
        tr = tr[:a.max_train]
    l2i = {t: i for i, t in enumerate(tags)}
    tok = AutoTokenizer.from_pretrained(a.model)
    if not tok.is_fast:
        raise SystemExit("✖ يلزم Fast Tokenizer")
    model = AutoModelForTokenClassification.from_pretrained(a.model, num_labels=len(tags), id2label=dict(enumerate(tags)),
                                                            label2id=l2i).to(dev)
    enc, lost = align(tok, chunk(tr, a.max_words, tok, C["max_length"]), l2i, C["max_length"])
    if lost:
        raise ValueError(f"NER: {lost} annotated entity words were truncated; refusing invalid training")
    print(f"[NER] train={len(tr)} جملة → {len(enc['input_ids'])} مقطع | val={len(va)} | كلمات كيان مقصوصة بالحد: {lost} | device={dev}")
    coll = DataCollatorForTokenClassification(tok)
    g = torch.Generator().manual_seed(a.seed)
    dl = DataLoader(_DS(enc), batch_size=a.batch_size, shuffle=True, collate_fn=coll, generator=g)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=C["weight_decay"])
    accumulation = max(1, int(C.get("gradient_accumulation_steps", 1)))
    steps = math.ceil(len(dl) / accumulation) * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(C["warmup_ratio"] * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    hist, best, bad, best_ep, t0 = [], -1.0, 0, None, time.time()
    va_true = [r["ner_tags"] for r in va]
    for ep in range(1, a.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, b in enumerate(dl, 1):
            b = {k: v.to(dev) for k, v in b.items()}
            with torch.autocast(device_type=dev.type, enabled=amp):
                loss = model(**b).loss                   # CrossEntropy يتجاهل -100 (حشو/رموز فرعية/خاصة)
            window_size = min(accumulation, len(dl) - ((step - 1) // accumulation) * accumulation)
            scaler.scale(loss / window_size).backward()
            if step % accumulation == 0 or step == len(dl):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sch.step()
            tot += loss.item()
            n += 1
        model.eval()
        pred, _ = NERTagger.from_objects(model, tok, a.max_words, C["max_length"]).tag_tokens([r["tokens"] for r in va])
        m = entity_metrics(va_true, pred)
        row = {"epoch": ep, "train_loss": round(tot / max(n, 1), 5), "val_entity_precision": m["entity_precision"],
               "val_entity_recall": m["entity_recall"], "val_entity_f1": m["entity_f1"], "seconds": round(time.time() - t0)}
        hist.append(row)
        print(row, flush=True)
        if m["entity_f1"] > best:
            best, bad, best_ep = m["entity_f1"], 0, ep
            model.save_pretrained(a.out)
            tok.save_pretrained(a.out)
            write_json(a.out / "best_validation_metrics.json", m | {"epoch": ep})
        else:
            bad += 1
            if bad >= a.patience:
                print(f"إيقاف مبكر بعد {ep} (أفضل Epoch {best_ep})")
                break
    pd.DataFrame(hist).to_csv(a.out / "training_history.csv", index=False)
    write_json(a.out / "training_config.json", {
        "task": "ner", "base_model": a.model, "architecture": "AutoModelForTokenClassification (BERT encoder + token linear head)",
        "tags": tags, "n_train_sentences": len(tr), "n_train_chunks": len(enc["input_ids"]), "n_validation": len(va),
        "epochs_run": len(hist), "best_epoch": best_ep, "best_validation_entity_f1": best, "learning_rate": a.lr,
        "batch_size": a.batch_size, "max_words_per_chunk": a.max_words, "max_length": C["max_length"],
        "label_alignment": "first subword; others/special/pad = -100", "entity_words_truncated": lost, "mixed_precision": amp,
        "gradient_accumulation_steps": accumulation, "seed": a.seed, "trained_at": now(), "train_seconds": round(time.time() - t0), "env": env_info(),
        "test_used_during_training": False, "label_quality": "silver (rule/extractor spans)"})
    print(f"✔ أفضل Entity-F1 على validation = {best} (Epoch {best_ep}) → {a.out}")


if __name__ == "__main__":
    main()
