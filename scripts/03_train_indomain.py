"""Bước 3 – Huấn luyện & so sánh các mô hình trên từng bộ dữ liệu chính.

Với mỗi (bộ dữ liệu × mô hình):
  GridSearchCV trên train → chọn ngưỡng trên val → đánh giá 1 lần trên test.

Chạy:
  python scripts/03_train_indomain.py                       # tất cả
  python scripts/03_train_indomain.py --datasets wild --models logreg nb
  PIDETECT_FAST=1 python scripts/03_train_indomain.py       # chạy thử nhanh
Kết quả: reports/tables/indomain_results.csv, models/*.joblib,
         data/scores/*.npz, reports/figures/indomain_*.png
"""
import argparse
import time

import _common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from pidetect import config as C
from pidetect import data as D
from pidetect import plots as P
from pidetect import training as T
from pidetect.models import ALL_MODELS, MODEL_SPECS

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=C.MAIN_DATASETS)
ap.add_argument("--models", nargs="+", default=ALL_MODELS)
ap.add_argument("--force", action="store_true", help="huấn luyện lại dù đã có kết quả")
args = ap.parse_args()

df = D.prepare_all()
RES_DIR = C.RESULTS_DIR / "indomain"
SCORE_DIR = C.DATA_DIR / "scores"
SCORE_DIR.mkdir(parents=True, exist_ok=True)

for ds in args.datasets:
    tr, va, te = (D.get_split(df, ds, s) for s in ("train", "val", "test"))
    print(f"\n=== {ds}: train={len(tr)} val={len(va)} test={len(te)} (pos test={te.label.mean():.3f})")
    for key in args.models:
        out_json = RES_DIR / f"{ds}__{key}.json"
        if out_json.exists() and not args.force:
            print(f"  [skip] {key} (đã có kết quả)")
            continue
        t0 = time.time()
        model, tune_info, cv_table = T.tune(key, ds, tr)
        res, raw = T.evaluate(model, va, te)
        res.update(tune_info)
        res.update({"dataset": ds, "model": key, "model_name": MODEL_SPECS[key].name_vi,
                    "slide": MODEL_SPECS[key].slide, "total_seconds": time.time() - t0})
        T.save_json(res, out_json)
        cv_table.to_csv(RES_DIR / f"{ds}__{key}__cv.csv", index=False)
        np.savez_compressed(SCORE_DIR / f"indomain__{ds}__{key}.npz", y_val=va.label.values,
                            y_test=te.label.values, **raw)
        T.save_model(model, ds, key)
        print(f"  {key:<11} CV-PR-AUC={res['cv_pr_auc']:.3f}  test PR-AUC={res['test_cost_pr_auc']:.3f}  "
              f"F1@τcost={res['test_cost_f1']:.3f}  Recall={res['test_cost_recall']:.3f}  "
              f"FPR={res['test_cost_fpr']:.3f}  ({res['total_seconds']:.0f}s)  {res['best_params']}")

# ---------------------------------------------------------------------------
# Tổng hợp bảng & hình
# ---------------------------------------------------------------------------
rows = [pd.read_json(p, typ="series") for p in sorted(RES_DIR.glob("*.json"))]
if not rows:
    raise SystemExit("Chưa có kết quả nào.")
res = pd.DataFrame(rows)
cols = ["dataset", "model", "model_name", "slide", "cv_pr_auc", "cv_pr_auc_std", "val_pr_auc",
        "test_cost_pr_auc", "test_cost_roc_auc", "test_cost_f1", "test_cost_precision", "test_cost_recall",
        "test_cost_fpr", "test_cost_cost_per_1k", "test_default_f1", "test_default_recall", "test_default_fpr",
        "test_f1_f1", "test_cost_threshold", "test_ece_platt", "test_cost_f1_ci", "test_cost_recall_ci",
        "test_cost_pr_auc_ci", "infer_ms_per_text", "tune_seconds", "n_train_used", "best_params"]
res = res[[c for c in cols if c in res]].sort_values(["dataset", "test_cost_pr_auc"], ascending=[True, False])
res.to_csv(C.TABLE_DIR / "indomain_results.csv", index=False)
print("\n", res[["dataset", "model", "test_cost_pr_auc", "test_cost_f1", "test_cost_recall", "test_cost_fpr"]]
      .round(3).to_string(index=False))

order = [m for m in ALL_MODELS if m in set(res.model)]
dsets = [d for d in C.MAIN_DATASETS if d in set(res.dataset)]
for metric, title in (("test_cost_pr_auc", "PR-AUC trên test"), ("test_cost_f1", "F1 trên test (τ chọn theo chi phí)"),
                      ("test_cost_recall", "Recall trên test (τ theo chi phí)"), ("test_cost_fpr", "FPR trên test (τ theo chi phí)")):
    piv = res.pivot(index="model", columns="dataset", values=metric).reindex(index=order, columns=dsets)
    fig, ax = plt.subplots(figsize=(1.8 + 1.6 * len(dsets), 0.45 * len(order) + 1.5))
    vmax = 1 if metric != "test_cost_fpr" else max(0.2, float(np.nanmax(piv.values)))
    P.heatmap(ax, piv.values, [MODEL_SPECS[m].name_vi for m in order], dsets, fmt="{:.3f}", vmin=0, vmax=vmax)
    ax.set_title(title)
    P.save(fig, f"indomain_{metric}")

# Đường PR & ROC cho mỗi bộ dữ liệu
for ds in dsets:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for i, key in enumerate(order):
        f = SCORE_DIR / f"indomain__{ds}__{key}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        p, r, _ = precision_recall_curve(z["y_test"], z["s_test"])
        fpr, tpr, _ = roc_curve(z["y_test"], z["s_test"])
        color = P.SERIES[i % len(P.SERIES)] if i < 8 else P.TEXT_SECONDARY
        ls = "-" if i < 8 else "--"
        axes[0].plot(r, p, color=color, ls=ls, lw=1.6, label=MODEL_SPECS[key].name_vi)
        axes[1].plot(fpr, tpr, color=color, ls=ls, lw=1.6)
    base = D.get_split(df, ds, "test").label.mean()
    axes[0].axhline(base, color=P.TEXT_SECONDARY, lw=1, ls=":")
    axes[0].set(xlabel="Recall", ylabel="Precision", title=f"{ds}: Precision–Recall (test)")
    axes[1].plot([0, 1], [0, 1], color=P.TEXT_SECONDARY, lw=1, ls=":")
    axes[1].set(xlabel="FPR", ylabel="TPR", title=f"{ds}: ROC (test)", xscale="log", xlim=(1e-3, 1))
    axes[0].legend(fontsize=8, loc="upper right" if ds == "wild" else "lower left")
    P.save(fig, f"indomain_curves_{ds}")
print("Xong. Bảng:", C.TABLE_DIR / "indomain_results.csv")
