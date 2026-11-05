"""Tiện ích dùng chung cho các thí nghiệm bonus (embedding, fine-tune, model có sẵn, LLM).

Mọi mô hình bonus được đánh giá *giống hệt* mô hình cổ điển ở bước 5 (nguồn
"combined"): cùng tập val để chọn ngưỡng, cùng 7 tập đánh giá, cùng việc loại
câu trùng với dữ liệu huấn luyện → so sánh công bằng.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import evaluation as E
from .training import save_json

EVAL_SETS = C.MAIN_DATASETS + C.OOD_DATASETS


def combined_frames(df: pd.DataFrame | None = None):
    df = D.prepare_all() if df is None else df
    tr = D.get_split(df, C.MAIN_DATASETS, "train")
    va = D.get_split(df, C.MAIN_DATASETS, "val")
    seen = pd.concat([tr, va])
    evals = {}
    for ev in EVAL_SETS:
        ed = D.get_split(df, ev, "test") if ev in C.MAIN_DATASETS else D.get_split(df, ev, "ood")
        evals[ev] = D.remove_overlap(ed, seen)[0]
    return tr, va, evals


def subsample_evals(evals: dict, n: int | None) -> dict:
    if not n:
        return evals
    return {k: D.maybe_subsample(v, n) for k, v in evals.items()}


def evaluate_scores(name: str, y_val, s_val, eval_scores: dict, evals: dict, extra: dict | None = None,
                    fixed_threshold: float | None = None) -> pd.DataFrame:
    """Chọn τ trên val (theo chi phí) rồi tính chỉ số trên từng tập; lưu JSON + CSV."""
    thr = E.choose_threshold(y_val, s_val, "cost")
    rows = []
    for ev, s in eval_scores.items():
        y = evals[ev]["label"].values
        for tag, t in (("cost", thr),) + ((("fixed", fixed_threshold),) if fixed_threshold is not None else ()):
            m = E.binary_metrics(y, s, t)
            rows.append({"method": name, "eval": ev, "threshold_type": tag,
                         **{k: m[k] for k in ("n", "threshold", "pr_auc", "roc_auc", "f1", "precision", "recall", "fpr")}})
    res = pd.DataFrame(rows)
    res.to_csv(C.TABLE_DIR / f"bonus_{name}.csv", index=False)
    save_json({"method": name, "threshold": thr, **(extra or {}), "results": rows}, C.RESULTS_DIR / f"bonus_{name}.json")
    for ev, s in eval_scores.items():
        np.savez_compressed(C.DATA_DIR / "scores" / f"bonus__{name}__{ev}.npz", y=evals[ev]["label"].values, s=s, thr=thr)
    print(res[res.threshold_type == "cost"][["eval", "n", "pr_auc", "f1", "recall", "fpr"]].round(3).to_string(index=False))
    return res


def pick_device():
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        print(f"[device] CUDA: {name} (sm_{major}{minor}), torch {torch.__version__}, CUDA {torch.version.cuda}")
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("[device] Apple MPS")
        return torch.device("mps")
    print("[device] CPU (sẽ chậm – nên dùng GPU cho phần này)")
    return torch.device("cpu")
