"""Bước 10 – Tổng hợp mọi kết quả thành reports/RESULTS.md (+ hình so sánh cuối cùng).

Chạy lại bất cứ lúc nào (ví dụ sau khi chạy các script bonus trên GPU) để cập nhật.
"""
import json

import _common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pidetect import config as C
from pidetect import plots as P
from pidetect.models import MODEL_SPECS

T = C.TABLE_DIR
EVAL_SETS = C.MAIN_DATASETS + C.OOD_DATASETS


def md(df: pd.DataFrame, floatfmt=3) -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "–" if pd.isna(v) else f"{v:.{floatfmt}f}")
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.astype(str).values)
    return "\n".join([head, sep, body])


def read(name):
    p = T / name
    return pd.read_csv(p) if p.exists() else None


out = ["# Kết quả thực nghiệm (tự động sinh bởi `scripts/10_make_report.py`)\n",
       f"Chi phí dùng để chọn ngưỡng: bỏ lọt tấn công (FN) = {C.COST_FN:g}, chặn nhầm (FP) = {C.COST_FP:g}.\n"]

ov = read("dataset_overview.csv")
if ov is not None:
    out += ["## 1. Dữ liệu\n", md(ov[["dataset", "split", "n", "n_pos", "pos_rate", "median_chars", "role", "license"]]),
            "\n![](figures/eda_01_dataset_size_posrate.png)\n"]
cl = read("data_cleaning_stats.csv")
if cl is not None:
    out += ["### Làm sạch & khử trùng lặp\n", md(cl), ""]

ind = read("indomain_results.csv")
if ind is not None:
    out.append("## 2. So sánh mô hình trong từng bộ dữ liệu (train/val/test cùng nguồn)\n")
    for ds in C.MAIN_DATASETS:
        sub = ind[ind.dataset == ds]
        if sub.empty:
            continue
        t = sub[["model_name", "slide", "cv_pr_auc", "test_cost_pr_auc", "test_cost_roc_auc", "test_cost_f1",
                 "test_cost_precision", "test_cost_recall", "test_cost_fpr", "test_default_f1", "test_ece_platt",
                 "test_cost_f1_ci"]].rename(columns={
            "model_name": "Mô hình", "cv_pr_auc": "CV PR-AUC", "test_cost_pr_auc": "PR-AUC", "test_cost_roc_auc": "ROC-AUC",
            "test_cost_f1": "F1 (τ chi phí)", "test_cost_precision": "Precision", "test_cost_recall": "Recall",
            "test_cost_fpr": "FPR", "test_default_f1": "F1 (τ mặc định)", "test_ece_platt": "ECE", "test_cost_f1_ci": "KTC 95% F1"})
        out += [f"### {ds}\n", md(t), f"\n![](figures/indomain_curves_{ds}.png)\n"]
    out.append("![](figures/indomain_test_cost_pr_auc.png)\n")

ctx = read("spml_context_classification.csv")
reg = read("spml_degree_regression.csv")
if ctx is not None:
    out += ["## 3. SPML – vai trò của ngữ cảnh system prompt\n", md(ctx, 4), "\n![](figures/spml_context_effect.png)\n"]
if reg is not None:
    out += ["## 4. SPML – hồi quy mức độ tấn công (Degree)\n",
            md(reg[["target", "model_name", "cv_mae", "mae", "rmse", "r2", "spearman", "within_1"]]),
            "\n![](figures/spml_regression_mae.png)\n![](figures/spml_regression_residuals_attack_only.png)\n"]

cs = read("cross_dataset_summary.csv")
cr = read("cross_dataset_results.csv")
if cr is not None:
    out.append("## 5. Tổng quát hoá giữa các bộ dữ liệu\n")
    piv = cr[cr.model == "logreg"].pivot(index="source", columns="eval", values="pr_auc").reindex(columns=EVAL_SETS)
    if not piv.empty:
        out += ["PR-AUC của Logistic Regression (hàng = nguồn huấn luyện, cột = tập đánh giá):\n", md(piv.reset_index()), ""]
    if cs is not None:
        out += ["Trung bình trên các tập **không** dùng để huấn luyện (trừ gandalf):\n", md(cs), ""]
    out += ["![](figures/cross_dataset_pr_auc.png)\n![](figures/cross_dataset_recall.png)\n"]
    orm = read("cross_dataset_overlap_removed.csv")
    if orm is not None:
        out += ["Số câu trùng đã loại khỏi tập đánh giá (chống rò rỉ):\n", md(orm), ""]

# Bảng so sánh cuối: mô hình combined cổ điển vs bonus
rows = []
if cr is not None:
    for _, r in cr[cr.source == "combined"].iterrows():
        rows.append({"method": MODEL_SPECS[r.model].name_vi, "group": "Cổ điển (TF-IDF)", "eval": r["eval"],
                     "pr_auc": r.pr_auc, "recall": r.recall, "fpr": r.fpr, "f1": r.f1})
for p in sorted(T.glob("bonus_*.csv")):
    if p.stem.endswith("robustness"):
        continue
    b = pd.read_csv(p)
    b = b[b.threshold_type == "cost"]
    grp = ("Embedding + ML" if p.stem.startswith("bonus_emb") else "Fine-tune transformer" if p.stem.startswith("bonus_ft")
           else "LLM zero-shot" if p.stem.startswith("bonus_llm") else "Bộ lọc có sẵn")
    for _, r in b.iterrows():
        rows.append({"method": r.method, "group": grp, "eval": r["eval"], "pr_auc": r.pr_auc, "recall": r.recall,
                     "fpr": r.fpr, "f1": r.f1})
if rows:
    comp = pd.DataFrame(rows)
    comp.to_csv(T / "final_comparison_long.csv", index=False)
    wide = comp.pivot_table(index=["group", "method"], columns="eval", values="pr_auc").reindex(
        columns=[e for e in EVAL_SETS if e != "gandalf"])  # gandalf chỉ có lớp dương → PR-AUC không xác định
    ood_cols = [c for c in C.OOD_DATASETS if c != "gandalf" and c in wide]
    wide["TB ngoài miền"] = wide[ood_cols].mean(axis=1)
    gd = comp[comp["eval"] == "gandalf"].set_index(["group", "method"])["recall"]
    wide["Recall gandalf"] = gd
    wide = wide.sort_values("TB ngoài miền", ascending=False)
    wide.to_csv(T / "final_comparison_pr_auc.csv")
    out += ["## 6. So sánh tổng thể (huấn luyện trên combined) – PR-AUC\n",
            "Lưu ý: mô hình *bonus* có thể được đánh giá trên mẫu con nếu chạy với `--max-eval`.\n",
            md(wide.reset_index()), "\n![](figures/final_comparison.png)\n"]
    fig, ax = plt.subplots(figsize=(1.6 + 1.05 * len(wide.columns), 0.42 * len(wide) + 1.6))
    labels = [f"{m} [{g}]" for g, m in wide.index]
    P.heatmap(ax, wide.values, labels, list(wide.columns), fmt="{:.2f}")
    ax.set_title("PR-AUC theo tập đánh giá (cột cuối: Recall trên gandalf)")
    P.save(fig, "final_comparison")

for name, title, fig in (
    ("error_counts.csv", "## 7. Phân tích lỗi – số lỗi của bộ phát hiện cuối", "error_breakdown"),
    ("error_robustness.csv", "## 8. Độ bền trước biến đổi đối kháng (Recall)", "error_robustness_raw"),
    ("conformal_results.csv", "## 9. Conformal prediction", "conformal_coverage"),
):
    t = read(name)
    if t is None:
        continue
    if name == "error_robustness.csv":
        t = t.pivot_table(index="perturbation", columns=["model", "defense"], values="recall")
        t.columns = [f"{m}{' +phòng thủ' if d else ''}" for m, d in t.columns]
        t = t.reset_index()
    if name == "conformal_results.csv":
        t = t[t.alpha == 0.05][["eval", "mondrian", "coverage", "coverage_benign", "coverage_attack", "avg_set_size",
                                 "frac_uncertain", "frac_empty"]]
    out += [title + "\n", md(t), f"\n![](figures/{fig}.png)\n"]
    if name == "error_robustness.csv":
        out.append("![](figures/error_robustness_defense.png)\n")
for extra in ("error_learning_curve", "error_calibration", "error_cost_threshold", "error_logreg_coefficients"):
    if (C.FIG_DIR / f"{extra}.png").exists():
        out.append(f"![](figures/{extra}.png)\n")

(C.REPORTS_DIR / "RESULTS.md").write_text("\n".join(out), encoding="utf-8")
print("Đã ghi", C.REPORTS_DIR / "RESULTS.md")
