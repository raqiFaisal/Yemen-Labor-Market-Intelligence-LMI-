"""Fine-tuning لنموذج AraBERT لتصنيف النصوص (نوع الفرصة / النطاق الجغرافي / مرشح الصلة).

    python -m src.training.train_classifier --task opportunity_type [--model aubmindlab/bert-base-arabertv02] [--epochs 5]

الضمانات العلمية:
  • يُستخدم train للتدريب و validation لاختيار أفضل Epoch والإيقاف المبكر فقط. **test لا يُقرأ هنا إطلاقاً**
    (التقييم عليه في src.evaluation.evaluate_classifier بعد انتهاء التدريب).
  • أوزان الفئات من train فقط (عكس التكرار)، بذور ثابتة، Mixed Precision عند توفر GPU.
  • يُحفظ: أفضل نموذج + Tokenizer + label_map + إعدادات التدريب + سجل كل Epoch.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import CONFIG, MODELS_DIR, DATA_DIR, device, env_info, now, set_seed, write_json
from src.data.loaders import arabert_preprocess, load_classification
from src.evaluation.metrics import classification_metrics


def encode(tok, texts, max_len):
    return tok(list(texts), truncation=True, max_length=max_len, padding=False)


class _DS:
    def __init__(self, enc, labels):
        self.enc, self.labels = enc, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = int(self.labels[i])
        return item


def predict(model, loader, dev, amp):
    import torch
    model.eval()
    logits_all = []
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(dev) for k, v in b.items() if k != "labels"}
            with torch.autocast(device_type=dev.type, enabled=amp):
                out = model(**b)
            logits_all.append(out.logits.float().cpu().numpy())
    return np.concatenate(logits_all) if logits_all else np.zeros((0, model.config.num_labels))


def main():
    import torch
    from torch.utils.data import DataLoader
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              get_linear_schedule_with_warmup)
    C = CONFIG["classification"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["opportunity_type", "geographic_scope", "relevance"])
    ap.add_argument("--model", default=CONFIG["arabert_model"])
    ap.add_argument("--data_dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--epochs", type=int, default=C["epochs"])
    ap.add_argument("--lr", type=float, default=C["learning_rate"])
    ap.add_argument("--batch_size", type=int, default=C["batch_size"])
    ap.add_argument("--max_length", type=int, default=C["max_length"])
    ap.add_argument("--patience", type=int, default=C["early_stopping_patience"])
    ap.add_argument("--train_file", default="train", help="train أو train_balanced (لنوع الفرصة)")
    ap.add_argument("--no_class_weights", action="store_true")
    ap.add_argument("--max_train", type=int, help="للاختبار السريع فقط")
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])
    a = ap.parse_args()
    set_seed(a.seed)
    out = a.out or MODELS_DIR / "classification" / a.task
    out.mkdir(parents=True, exist_ok=True)
    dev = device()
    amp = dev.type == "cuda"
    splits = ["train", "validation"] if a.train_file == "train" else [a.train_file, "validation"]
    #FINE_TUNING
    data, lm = load_classification(a.task, a.data_dir, splits=splits)        # ← test غير مُحمَّل
    tr, va = data[splits[0]], data["validation"]
    if a.max_train:
        tr = tr.sample(min(a.max_train, len(tr)), random_state=a.seed)
    labels = [lm["id2label"][str(i)] for i in range(len(lm["id2label"]))]
    present = sorted(set(tr.label_id))
    print(f"[{a.task}] train={len(tr)} validation={len(va)} | فئات train: {tr.label.value_counts().to_dict()} | device={dev}")
    #تحميل النموذج المدرب مسبقا
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=len(labels), id2label={i: l for i, l in enumerate(labels)}, label2id={l: i for i, l in enumerate(labels)})
    model.to(dev)
    # تحقق توافق Tokenizer/النموذج
    assert tok.vocab_size <= model.config.vocab_size, "Tokenizer لا يطابق مفردات النموذج"
    Xtr = arabert_preprocess(tr.text, a.model)
    Xva = arabert_preprocess(va.text, a.model)
    lens = [len(x) for x in tok(Xtr[:500], truncation=False)["input_ids"]]
    trunc_rate = float(np.mean([l > a.max_length for l in lens])) if lens else 0.0
    coll = DataCollatorWithPadding(tok)
    g = torch.Generator().manual_seed(a.seed)
    dl_tr = DataLoader(_DS(encode(tok, Xtr, a.max_length), tr.label_id.values), batch_size=a.batch_size, shuffle=True,
                       collate_fn=coll, generator=g)
    dl_va = DataLoader(_DS(encode(tok, Xva, a.max_length), va.label_id.values), batch_size=C["eval_batch_size"],
                       collate_fn=coll)
    if a.no_class_weights or not C["class_weights"]:
        weights = None
    else:
        cnt = np.bincount(tr.label_id.values, minlength=len(labels)).astype(float)
        w = np.where(cnt > 0, cnt.sum() / (len(present) * np.maximum(cnt, 1)), 0.0)   # فئة غائبة عن train = وزن 0
        weights = torch.tensor(w, dtype=torch.float, device=dev)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=C["weight_decay"])
    if C["gradient_accumulation_steps"] < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    steps = math.ceil(len(dl_tr) / C["gradient_accumulation_steps"]) * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(C["warmup_ratio"] * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    hist, best, bad, best_ep = [], -1.0, 0, None
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        tot, n = 0.0, 0
        for step, b in enumerate(dl_tr, 1):
            b = {k: v.to(dev) for k, v in b.items()}
            y = b.pop("labels")
            # Final accumulation window can contain fewer batches than the
            # configured step count; normalize by its actual size.
            accumulation = C["gradient_accumulation_steps"]
            window_size = min(accumulation, len(dl_tr) - ((step - 1) // accumulation) * accumulation)
            with torch.autocast(device_type=dev.type, enabled=amp):
                unscaled_loss = loss_fn(model(**b).logits.float(), y)
                loss = unscaled_loss / window_size
            scaler.scale(loss).backward()
            if step % C["gradient_accumulation_steps"] == 0 or step == len(dl_tr):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sch.step()
            tot += unscaled_loss.item() * len(y)
            n += len(y)
        lg = predict(model, dl_va, dev, amp)
        vloss = float(torch.nn.functional.cross_entropy(torch.tensor(lg), torch.tensor(va.label_id.values)).item()) if len(va) else None
        pred = [labels[i] for i in lg.argmax(1)]
        m = classification_metrics(list(va.label), pred, labels)
        row = {"epoch": ep, "train_loss": round(tot / max(n, 1), 5), "val_loss": round(vloss, 5) if vloss is not None else None,
               "val_accuracy": m["accuracy"], "val_macro_f1": m["macro_f1"], "val_weighted_f1": m["weighted_f1"],
               "seconds": round(time.time() - t0)}
        hist.append(row)
        print(row, flush=True)
        score = m[C["metric_for_best"]]
        if score > best:
            best, bad, best_ep = score, 0, ep
            model.save_pretrained(out)
            tok.save_pretrained(out)
            write_json(out / "best_validation_metrics.json", {k: v for k, v in m.items() if k != "report_text"} | {"epoch": ep})
        else:
            bad += 1
            if bad >= a.patience:
                print(f"إيقاف مبكر بعد {ep} (أفضل Epoch {best_ep})")
                break
    pd.DataFrame(hist).to_csv(out / "training_history.csv", index=False)
    write_json(out / "label_map.json", lm)
    write_json(out / "training_config.json", {
        "task": a.task, "base_model": a.model, "architecture": "AutoModelForSequenceClassification (BERT encoder + linear head)",
        "train_file": splits[0], "n_train": len(tr), "n_validation": len(va), "epochs_run": len(hist), "best_epoch": best_ep,
        f"best_validation_{C['metric_for_best']}": best, "learning_rate": a.lr, "batch_size": a.batch_size, "max_length": a.max_length,
        "truncation_rate_train_sample": round(trunc_rate, 4), "weight_decay": C["weight_decay"], "warmup_ratio": C["warmup_ratio"],
        "class_weights": None if weights is None else dict(zip(labels, [round(float(x), 4) for x in weights.cpu().numpy()])),
        "early_stopping_patience": a.patience, "mixed_precision": amp, "seed": a.seed,
        "gradient_accumulation_steps": C["gradient_accumulation_steps"], "preprocessing": "ArabertPreprocessor(model)" if "arabert" in a.model.lower() else "identity",
        "trained_at": now(), "train_seconds": round(time.time() - t0), "env": env_info(),
        "test_used_during_training": False, "label_quality": "silver (rule-based)"})
    print(f"✔ أفضل {C['metric_for_best']} على validation = {best} (Epoch {best_ep}) → {out}")


if __name__ == "__main__":
    main()
