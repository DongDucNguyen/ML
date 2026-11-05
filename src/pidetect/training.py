"""Tinh chỉnh siêu tham số bằng cross-validation và đánh giá trên val/test.

Quy trình (đúng tinh thần slide Model Assessment):
  1. GridSearchCV trên **train** (chỉ số tối ưu: Average Precision / PR-AUC –
     phù hợp dữ liệu mất cân bằng). Toàn bộ pipeline (TF-IDF, SVD, scaler)
     được fit lại trong từng fold.
  2. Kiểu CV phụ thuộc cấu trúc dữ liệu: StratifiedKFold (safeguard),
     TimeSeriesSplit (wild – có thời gian), StratifiedGroupKFold (spml – nhóm
     system prompt).
  3. Chọn ngưỡng τ trên **validation**.
  4. Báo cáo một lần trên **test** (kèm khoảng tin cậy bootstrap).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.base import clone
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)

from . import config as C
from . import evaluation as E
from .data import maybe_subsample
from .models import MODEL_SPECS, is_probabilistic, scores

CACHE = Memory(location=str(C.DATA_DIR / "cache" / "sklearn"), verbose=0)


def make_cv(dataset: str, train_df: pd.DataFrame, n_splits: int = 3):
    """Trả về (cv, groups) phù hợp cấu trúc dữ liệu."""
    if dataset == "wild" and "date" in train_df:
        return TimeSeriesSplit(n_splits=n_splits), None
    if dataset == "spml" and "group" in train_df:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=C.SEED), train_df["group"].values
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=C.SEED), None


def tune(model_key: str, dataset: str, train_df: pd.DataFrame, n_splits: int = 3, verbose: int = 0):
    spec = MODEL_SPECS[model_key]
    tr = train_df
    if dataset == "wild" and "date" in tr:
        tr = tr.sort_values("date", kind="stable").reset_index(drop=True)
    limit = spec.max_train
    if C.FAST:
        limit = min(limit or 10**9, 3000)
    if limit and len(tr) > limit:
        tr = maybe_subsample(tr, limit)
        if dataset == "wild" and "date" in tr:
            tr = tr.sort_values("date", kind="stable").reset_index(drop=True)

    pipe = spec.build()
    pipe.set_params(memory=CACHE)
    cv, groups = make_cv(dataset, tr, n_splits)
    heavy = model_key in {"rf", "knn", "lgbm"}  # đã song song bên trong
    gs = GridSearchCV(
        pipe, spec.grid, scoring="average_precision", cv=cv, refit=True,
        n_jobs=1 if heavy else min(4, C.N_JOBS if C.N_JOBS > 0 else 4), verbose=verbose,
        error_score="raise",
    )
    t0 = time.time()
    gs.fit(tr["text"].tolist(), tr["label"].values, groups=groups)
    fit_time = time.time() - t0
    model = gs.best_estimator_
    model.set_params(memory=None)
    cv_table = pd.DataFrame(gs.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "mean_fit_time", "rank_test_score"]
    ].sort_values("rank_test_score")
    return model, {
        "best_params": {k: str(v) for k, v in gs.best_params_.items()},
        "cv_pr_auc": float(gs.best_score_),
        "cv_pr_auc_std": float(cv_table.iloc[0]["std_test_score"]),
        "tune_seconds": fit_time,
        "n_train_used": len(tr),
    }, cv_table


def evaluate(model, val_df, test_df, with_ci: bool = True) -> tuple[dict, dict]:
    """Chọn ngưỡng trên val, đánh giá trên test. Trả về (kết quả, điểm số thô)."""
    prob_model = is_probabilistic(model)
    t0 = time.time()
    s_val = scores(model, val_df["text"].tolist())
    s_test = scores(model, test_df["text"].tolist())
    infer_ms = 1000 * (time.time() - t0) / (len(val_df) + len(test_df))

    from .models import LinearProbabilityModel

    is_lpm = isinstance(model.steps[-1][1], LinearProbabilityModel)
    thr_default = 0.5 if (prob_model or is_lpm) else 0.0
    thr_cost = E.choose_threshold(val_df["label"], s_val, "cost")
    thr_f1 = E.choose_threshold(val_df["label"], s_val, "f1")

    # Xác suất đã hiệu chỉnh (Platt trên val) – dùng cho Brier / conformal
    cal = E.PlattCalibrator().fit(s_val, val_df["label"])
    p_test = cal.predict_proba(s_test)
    p_val = cal.predict_proba(s_val)

    res = {"probabilistic": prob_model, "infer_ms_per_text": infer_ms}
    for tag, thr in (("default", thr_default), ("cost", thr_cost), ("f1", thr_f1)):
        m = E.binary_metrics(test_df["label"], s_test, thr, prob=p_test)
        res.update({f"test_{tag}_{k}": v for k, v in m.items()})
    mv = E.binary_metrics(val_df["label"], s_val, thr_cost)
    res.update({"val_pr_auc": mv["pr_auc"], "val_roc_auc": mv["roc_auc"]})
    if prob_model:
        raw = np.clip(s_test, 0, 1)
        res["test_brier_raw"] = float(np.mean((raw - test_df["label"].values) ** 2))
        res["test_ece_raw"] = E.expected_calibration_error(test_df["label"], raw)
    res["test_ece_platt"] = E.expected_calibration_error(test_df["label"], p_test)
    if with_ci:
        res.update({f"test_cost_{k}": v for k, v in E.bootstrap_ci(test_df["label"], s_test, thr_cost).items()})
    raw_scores = {"s_val": s_val, "s_test": s_test, "p_val": p_val, "p_test": p_test,
                  "thr_cost": thr_cost, "thr_f1": thr_f1, "thr_default": thr_default}
    return res, raw_scores


def save_model(model, dataset: str, key: str) -> Path:
    path = C.MODELS_DIR / f"{dataset}__{key}.joblib"
    joblib.dump(model, path, compress=3)
    return path


def load_model(dataset: str, key: str):
    return joblib.load(C.MODELS_DIR / f"{dataset}__{key}.joblib")


def save_json(obj, path: Path):
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_default), encoding="utf-8")


def fit_fixed(model_key: str, params: dict, train_df: pd.DataFrame):
    """Huấn luyện lại một mô hình với tham số đã chọn (dùng cho thí nghiệm chéo)."""
    spec = MODEL_SPECS[model_key]
    pipe = spec.build()
    grid_values = {k: v for k, v in spec.grid.items()}
    # map tham số dạng chuỗi về giá trị gốc trong grid
    real = {}
    for k, v in params.items():
        cands = grid_values.get(k, [])
        match = [c for c in cands if str(c) == str(v)]
        real[k] = clone(match[0]) if match and hasattr(match[0], "get_params") else (match[0] if match else v)
    pipe.set_params(**real)
    tr = train_df
    if spec.max_train and len(tr) > spec.max_train:
        tr = maybe_subsample(tr, spec.max_train)
    pipe.fit(tr["text"].tolist(), tr["label"].values)
    return pipe
