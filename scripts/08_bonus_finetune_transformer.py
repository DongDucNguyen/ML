"""Bước 8 (bonus, cần GPU) – Fine-tune transformer (DeBERTa-v3 / ModernBERT) phát hiện injection.

Vòng lặp huấn luyện PyTorch thuần (không phụ thuộc phiên bản Trainer API):
AdamW + warmup tuyến tính, bf16 autocast, gradient accumulation, đánh giá
trên val sau mỗi epoch (giữ checkpoint tốt nhất theo PR-AUC).
Cắt chuỗi kiểu head+tail: giữ phần đầu và phần cuối prompt → chống tấn công
"pha loãng" (giấu lệnh độc ở cuối một đoạn văn dài vô hại).

Ví dụ (RTX 5070 Ti 16GB):
  python scripts/08_bonus_finetune_transformer.py --model microsoft/deberta-v3-base --epochs 2 --batch 16 --max-len 256
  python scripts/08_bonus_finetune_transformer.py --model answerdotai/ModernBERT-base --epochs 2 --batch 32 --max-len 512
  python scripts/08_bonus_finetune_transformer.py --max-train 50000      # mặc định: lấy toàn bộ (~25k mẫu)
Chạy thử trên CPU:
  python scripts/08_bonus_finetune_transformer.py --model prajjwal1/bert-tiny --max-train 400 --max-eval 100 --epochs 1
"""
import argparse
import math
import random
import time

import _common  # noqa: F401
import numpy as np
import torch
from torch.utils.data import DataLoader

from pidetect import bonus as B
from pidetect import config as C
from pidetect import data as D
from pidetect import evaluation as E
from pidetect.robustness import PERTURBATIONS, defensive_normalize

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="microsoft/deberta-v3-base")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--grad-accum", type=int, default=1)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--weight-decay", type=float, default=0.01)
ap.add_argument("--warmup", type=float, default=0.06)
ap.add_argument("--max-len", type=int, default=256)
ap.add_argument("--max-train", type=int, default=None)
ap.add_argument("--max-eval", type=int, default=None)
ap.add_argument("--workers", type=int, default=0, help="DataLoader workers (Windows: để 0)")
ap.add_argument("--no-bf16", action="store_true")
ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (tiết kiệm VRAM cho model large)")
ap.add_argument("--skip-robustness", action="store_true")
args = ap.parse_args()

from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

torch.manual_seed(C.SEED)
random.seed(C.SEED)
np.random.seed(C.SEED)
device = B.pick_device()
use_bf16 = (not args.no_bf16) and device.type == "cuda" and torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
print(f"[cfg] model={args.model} bf16={use_bf16} max_len={args.max_len} batch={args.batch}x{args.grad_accum}")

tr, va, evals = B.combined_frames()
if args.max_train:
    tr = D.maybe_subsample(tr, args.max_train)
    va = D.maybe_subsample(va, max(200, args.max_train // 4))
evals = B.subsample_evals(evals, args.max_eval)
short = args.model.split("/")[-1]

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).float().to(device)
# .float(): Transformers 5 giữ dtype của checkpoint (vd. fp16) → huấn luyện fp16 + AdamW dễ ra NaN
if args.grad_ckpt:
    model.gradient_checkpointing_enable()


_EMPTY = tok("", add_special_tokens=True)["input_ids"]  # ví dụ [CLS, SEP]
_N_PREFIX = 1 if _EMPTY and _EMPTY[0] in {tok.cls_token_id, tok.bos_token_id} else 0


def encode_head_tail(texts, max_len=args.max_len):
    """Token hoá; nếu quá dài thì giữ 70% đầu + 30% cuối (thay vì cắt bỏ toàn bộ phần cuối)."""
    budget = max_len - len(_EMPTY)
    ids_list = []
    for t in texts:
        ids = tok(t[: 4 * C.MAX_CHARS], add_special_tokens=False, truncation=False)["input_ids"]
        if len(ids) > budget:
            head = int(budget * 0.7)
            ids = ids[:head] + ids[-(budget - head):]
        ids_list.append(_EMPTY[:_N_PREFIX] + ids + _EMPTY[_N_PREFIX:])
    return ids_list


def collate(batch):
    ids = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    enc = tok.pad({"input_ids": ids}, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"], labels


class DS(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.ids = encode_head_tail(list(texts))
        self.labels = list(labels)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], self.labels[i]


@torch.no_grad()
def predict(texts, bs=None):
    model.eval()
    bs = bs or args.batch * 2
    ds = DS(texts, [0] * len(texts))
    # sắp theo độ dài để giảm padding → nhanh hơn
    order = np.argsort([len(x) for x in ds.ids])
    out = np.zeros(len(ds), dtype=np.float32)
    for s in range(0, len(order), bs):
        idx = order[s: s + bs]
        ii, am, _ = collate([ds[i] for i in idx])
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_bf16):
            logits = model(input_ids=ii.to(device), attention_mask=am.to(device)).logits.float()
        out[idx] = torch.softmax(logits, -1)[:, 1].cpu().numpy()
    return out


t0 = time.time()
train_ds = DS(tr["text"], tr["label"])
print(f"[data] train={len(train_ds)} val={len(va)}  token hoá {time.time() - t0:.0f}s")
loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=args.workers)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
total = math.ceil(len(loader) / args.grad_accum) * args.epochs
sched = get_linear_schedule_with_warmup(opt, int(args.warmup * total), total)

out_dir = C.MODELS_DIR / f"transformer_{short}"
best_ap, history = -1.0, []
for ep in range(1, args.epochs + 1):
    model.train()
    t_ep, run_loss = time.time(), 0.0
    for step, (ii, am, y) in enumerate(loader, 1):
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_bf16):
            logits = model(input_ids=ii.to(device), attention_mask=am.to(device)).logits
        loss = torch.nn.functional.cross_entropy(logits.float(), y.to(device)) / args.grad_accum
        loss.backward()
        run_loss += loss.item() * args.grad_accum
        if step % args.grad_accum == 0 or step == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        if step % 200 == 0:
            print(f"  ep{ep} step {step}/{len(loader)} loss={run_loss / step:.4f} ({time.time() - t_ep:.0f}s)")
    p_val = predict(va["text"].tolist())
    m = E.binary_metrics(va["label"], p_val, 0.5)
    history.append({"epoch": ep, "train_loss": run_loss / len(loader), "val_pr_auc": m["pr_auc"], "val_f1@0.5": m["f1"],
                    "seconds": time.time() - t_ep})
    print(f"[epoch {ep}] loss={run_loss / len(loader):.4f} val PR-AUC={m['pr_auc']:.4f} F1@0.5={m['f1']:.4f} "
          f"({time.time() - t_ep:.0f}s)")
    if m["pr_auc"] > best_ap:
        best_ap = m["pr_auc"]
        model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)

model = AutoModelForSequenceClassification.from_pretrained(out_dir).float().to(device)
p_val = predict(va["text"].tolist())
eval_scores = {ev: predict(fr["text"].tolist()) for ev, fr in evals.items()}
extra = {"model": args.model, "epochs": args.epochs, "max_len": args.max_len, "history": history,
         "n_train": len(tr), "bf16": use_bf16,
         "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)}
res = B.evaluate_scores(f"ft_{short}", va["label"].values, p_val, eval_scores, evals, extra=extra, fixed_threshold=0.5)

if not args.skip_robustness:
    thr = E.choose_threshold(va["label"], p_val, "cost")
    atk = D.maybe_subsample(
        __import__("pandas").concat([evals["safeguard"], evals["wild"], evals["gandalf"], evals["deepset"]]).query("label == 1"),
        args.max_eval or 1500)
    rows = []
    for pname, fn in PERTURBATIONS.items():
        rng = random.Random(C.SEED)
        pert = [fn(t, rng) for t in atk["text"]]
        for defense in (False, True):
            texts = [defensive_normalize(t) for t in pert] if defense else pert
            rows.append({"model": f"ft_{short}", "perturbation": pname, "defense": defense,
                         "recall": float((predict(texts) >= thr).mean())})
    import pandas as pd

    pd.DataFrame(rows).to_csv(C.TABLE_DIR / f"bonus_ft_{short}_robustness.csv", index=False)
    print(pd.DataFrame(rows).pivot(index="perturbation", columns="defense", values="recall").round(3))
print("Xong. Checkpoint:", out_dir)
