"""Đánh giá mô hình (slide Model Assessment + Logistic Regression).

* Chỉ số: Accuracy, Precision, Recall, F1, Specificity, FPR, ROC-AUC, PR-AUC
  (Average Precision), Brier, Expected Cost.
* Chọn ngưỡng τ trên **validation** (không bao giờ dùng test): theo chi phí
  (C_FN·FN + C_FP·FP) hoặc theo F1.
* Khoảng tin cậy bootstrap 95% cho chỉ số trên test.
* Hiệu chỉnh xác suất (Platt scaling) cho mô hình chỉ có decision_function.
* Conformal prediction (bonus) – tập dự đoán với bảo đảm độ phủ 1−α.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

from . import config as C


# ---------------------------------------------------------------------------
# Chỉ số
# ---------------------------------------------------------------------------
def binary_metrics(y, score, thr, prob=None, cost_fn=C.COST_FN, cost_fp=C.COST_FP) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(score) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    n = len(y)
    out = {
        "n": n,
        "pos_rate": y.mean() if n else np.nan,
        "threshold": float(thr),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "accuracy": (tp + tn) / n if n else np.nan,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "fpr": fp / (fp + tn) if (fp + tn) else np.nan,
        "f1": f1_score(y, pred, zero_division=0) if y.sum() else np.nan,
        "cost_per_1k": 1000 * (cost_fn * fn + cost_fp * fp) / n if n else np.nan,
    }
    both = 0 < y.sum() < n
    out["roc_auc"] = roc_auc_score(y, score) if both else np.nan
    out["pr_auc"] = average_precision_score(y, score) if both else np.nan
    if prob is not None:
        out["brier"] = brier_score_loss(y, np.clip(prob, 0, 1))
    return out


def choose_threshold(y_val, s_val, method: str = "cost", cost_fn=C.COST_FN, cost_fp=C.COST_FP) -> float:
    """Chọn ngưỡng trên tập validation.

    method="cost": cực tiểu C_FN·FN + C_FP·FP.  method="f1": cực đại F1.
    """
    y = np.asarray(y_val).astype(int)
    s = np.asarray(s_val, dtype=float)
    cand = np.unique(s)
    if len(cand) > 2000:
        cand = np.unique(np.quantile(s, np.linspace(0, 1, 2001)))
    # Ngưỡng giữa hai điểm liền kề để ổn định hơn
    best_thr, best_val = cand[0], np.inf
    P = y.sum()
    for t in cand:
        pred = s >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = P - tp
        if method == "cost":
            val = cost_fn * fn + cost_fp * fp
        else:
            denom = 2 * tp + fp + fn
            val = -(2 * tp / denom) if denom else 0.0
        if val < best_val:
            best_val, best_thr = val, t
    return float(best_thr)


def default_threshold(probabilistic: bool) -> float:
    return 0.5 if probabilistic else 0.0


def bootstrap_ci(y, score, thr, metrics=("f1", "recall", "precision", "pr_auc"), n_boot=1000, seed=C.SEED) -> dict:
    """Khoảng tin cậy 95% bằng bootstrap (lấy mẫu lại có hoàn lại tập test)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    s = np.asarray(score)
    vals = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0 or y[idx].sum() == len(idx):
            continue
        m = binary_metrics(y[idx], s[idx], thr)
        for k in metrics:
            vals[k].append(m[k])
    return {f"{k}_ci": (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in vals.items()}


# ---------------------------------------------------------------------------
# Hiệu chỉnh xác suất
# ---------------------------------------------------------------------------
class PlattCalibrator:
    """Hồi quy logistic 1 chiều trên điểm số (fit trên validation)."""

    def fit(self, s, y):
        self.lr_ = LogisticRegression(C=1e4).fit(np.asarray(s).reshape(-1, 1), np.asarray(y).astype(int))
        return self

    def predict_proba(self, s):
        return self.lr_.predict_proba(np.asarray(s).reshape(-1, 1))[:, 1]


def reliability_table(y, prob, n_bins=10) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    prob = np.clip(np.asarray(prob), 0, 1)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(prob, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            rows.append({"bin": b, "mean_pred": prob[m].mean(), "frac_pos": y[m].mean(), "count": int(m.sum())})
    return pd.DataFrame(rows)


def expected_calibration_error(y, prob, n_bins=10) -> float:
    t = reliability_table(y, prob, n_bins)
    return float((t["count"] * (t["mean_pred"] - t["frac_pos"]).abs()).sum() / t["count"].sum())


# ---------------------------------------------------------------------------
# Conformal prediction (bonus – ngoài chương trình)
# ---------------------------------------------------------------------------
class SplitConformalClassifier:
    """Split conformal cho bài toán 2 lớp với điểm không phù hợp s = 1 − p̂(y|x).

    ``mondrian=True``: hiệu chỉnh riêng cho từng lớp (class-conditional), bảo đảm
    độ phủ cho *cả* lớp tấn công hiếm – thứ mà conformal thông thường không bảo đảm.
    Tập dự đoán: {0}, {1}, {0,1} (không chắc → chuyển người kiểm duyệt), ∅.
    """

    def __init__(self, alpha: float = 0.05, mondrian: bool = True):
        self.alpha = alpha
        self.mondrian = mondrian

    @staticmethod
    def _q(scores, alpha):
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        return float(np.quantile(scores, level, method="higher"))

    def fit(self, p_cal, y_cal):
        p_cal = np.asarray(p_cal)
        y_cal = np.asarray(y_cal).astype(int)
        nc = np.where(y_cal == 1, 1 - p_cal, p_cal)
        if self.mondrian:
            self.q_ = {c: self._q(nc[y_cal == c], self.alpha) for c in (0, 1)}
        else:
            q = self._q(nc, self.alpha)
            self.q_ = {0: q, 1: q}
        return self

    def predict_sets(self, p):
        p = np.asarray(p)
        inc1 = (1 - p) <= self.q_[1]
        inc0 = p <= self.q_[0]
        return np.column_stack([inc0, inc1])

    def evaluate(self, p, y) -> dict:
        y = np.asarray(y).astype(int)
        S = self.predict_sets(p)
        covered = S[np.arange(len(y)), y]
        size = S.sum(1)
        out = {
            "alpha": self.alpha,
            "mondrian": self.mondrian,
            "coverage": covered.mean(),
            "coverage_benign": covered[y == 0].mean() if (y == 0).any() else np.nan,
            "coverage_attack": covered[y == 1].mean() if (y == 1).any() else np.nan,
            "avg_set_size": size.mean(),
            "frac_uncertain": (size == 2).mean(),
            "frac_empty": (size == 0).mean(),
            "frac_singleton": (size == 1).mean(),
        }
        single = size == 1
        if single.any():
            pred = S[single][:, 1].astype(int)
            out["singleton_accuracy"] = (pred == y[single]).mean()
        return out


# ---------------------------------------------------------------------------
# Tiện ích đường cong
# ---------------------------------------------------------------------------
def cost_curve(y, score, cost_fn=C.COST_FN, cost_fp=C.COST_FP, n=200) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    s = np.asarray(score)
    thrs = np.unique(np.quantile(s, np.linspace(0, 1, n)))
    rows = []
    for t in thrs:
        m = binary_metrics(y, s, t, cost_fn=cost_fn, cost_fp=cost_fp)
        rows.append({k: m[k] for k in ("threshold", "precision", "recall", "fpr", "f1", "cost_per_1k")})
    return pd.DataFrame(rows)


def pr_points(y, score):
    p, r, t = precision_recall_curve(y, score)
    return p, r, t
