"""Bước 4 – SPML: (A) giá trị của ngữ cảnh system prompt; (B) hồi quy mức độ tấn công.

(A) Phân loại injection: chỉ dùng user prompt  vs  user prompt + đặc trưng ngữ cảnh.
(B) Hồi quy Degree (0–10 = số luật của system prompt bị phá) – Slide 2 Linear
    Regression (OLS, Ridge, Lasso) so với KNN, Cây, RF, SVR, LightGBM.
    CV theo nhóm system prompt (GroupKFold) để tránh rò rỉ.

Chạy:  python scripts/04_spml_context_regression.py
"""
import time

import _common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Memory
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeRegressor

from pidetect import config as C
from pidetect import data as D
from pidetect import evaluation as E
from pidetect import plots as P
from pidetect.context import user_only_features, user_plus_context_features
from pidetect.features import feature_names
from pidetect.training import save_json

CACHE = Memory(location=str(C.DATA_DIR / "cache" / "sklearn_spml"), verbose=0)
df = D.prepare_all()
tr, va, te = (D.get_split(df, "spml", s) for s in ("train", "val", "test"))
trva = pd.concat([tr, va], ignore_index=True)  # hồi quy: CV trên train+val, test giữ riêng
COLS = ["text", "system_prompt"]
if C.FAST:
    tr = D.maybe_subsample(tr, 3000)
    trva = D.maybe_subsample(trva, 4000)

# ---------------------------------------------------------------------------
# (A) Ngữ cảnh có giúp phát hiện injection?
# ---------------------------------------------------------------------------
print("=== (A) Phân loại: user-only vs user+context ===")
rows_a = []
for feat_name, feat_fn in (("Chỉ user prompt", user_only_features), ("User + ngữ cảnh system", user_plus_context_features)):
    for clf_name, clf, grid in (
        ("Logistic Regression", LogisticRegression(max_iter=3000), {"clf__C": [1.0, 4.0, 16.0]}),
        ("Linear SVM", LinearSVC(max_iter=5000), {"clf__C": [0.1, 0.3, 1.0]}),
    ):
        pipe = Pipeline([("feats", feat_fn()), ("clf", clf)], memory=CACHE)
        gs = GridSearchCV(pipe, grid, scoring="average_precision", n_jobs=3,
                          cv=StratifiedGroupKFold(3, shuffle=True, random_state=C.SEED))
        gs.fit(tr[COLS], tr["label"], groups=tr["group"])
        m = gs.best_estimator_
        s_val = m.decision_function(va[COLS])
        s_te = m.decision_function(te[COLS])
        thr = E.choose_threshold(va["label"], s_val, "cost")
        met = E.binary_metrics(te["label"], s_te, thr)
        rows_a.append({"features": feat_name, "model": clf_name, "cv_pr_auc": gs.best_score_, **{
            k: met[k] for k in ("pr_auc", "roc_auc", "f1", "precision", "recall", "fpr")}})
        print(f"  {feat_name:<24} {clf_name:<20} PR-AUC={met['pr_auc']:.4f} F1={met['f1']:.4f} "
              f"Recall={met['recall']:.4f} FPR={met['fpr']:.4f}")
ctx_df = pd.DataFrame(rows_a)
ctx_df.to_csv(C.TABLE_DIR / "spml_context_classification.csv", index=False)

# ---------------------------------------------------------------------------
# (B) Hồi quy Degree
# ---------------------------------------------------------------------------
def F(kind="word+char+hand", **kw):
    return user_plus_context_features(kind, **kw)


REG_MODELS = {
    "mean": ("Dự đoán trung bình (baseline)", lambda: Pipeline([("feats", F("hand")), ("reg", DummyRegressor())]), {}),
    "ols": ("Linear Regression (OLS, SVD-300)", lambda: Pipeline([
        ("feats", F()), ("svd", TruncatedSVD(300, random_state=C.SEED)), ("reg", LinearRegression())]), {}),
    "ridge": ("Ridge (L2)", lambda: Pipeline([("feats", F()), ("reg", Ridge())]),
              {"reg__alpha": [0.3, 1.0, 3.0, 10.0]}),
    "lasso": ("Lasso (L1)", lambda: Pipeline([("feats", F("word+hand", word_max=20_000)),
                                              ("reg", Lasso(max_iter=5000))]),
              {"reg__alpha": [1e-4, 3e-4, 1e-3]}),
    "knn": ("KNN Regressor", lambda: Pipeline([("feats", F("word+char")), ("svd", TruncatedSVD(200, random_state=C.SEED)),
                                               ("norm", Normalizer()), ("reg", KNeighborsRegressor())]),
            {"reg__n_neighbors": [5, 15, 31], "reg__weights": ["uniform", "distance"]}),
    "tree": ("Decision Tree Regressor", lambda: Pipeline([("feats", F("word+hand", word_max=20_000)),
                                                          ("reg", DecisionTreeRegressor(random_state=C.SEED))]),
             {"reg__max_depth": [8, 16, None], "reg__min_samples_leaf": [2, 10]}),
    "rf": ("Random Forest Regressor", lambda: Pipeline([
        ("feats", F()), ("svd", TruncatedSVD(300, random_state=C.SEED)),
        ("reg", RandomForestRegressor(n_estimators=300, max_features=0.3, n_jobs=-1, random_state=C.SEED))]),
           {"reg__min_samples_leaf": [1, 5]}),
    "svr": ("Linear SVR", lambda: Pipeline([("feats", F()), ("reg", LinearSVR(max_iter=10000, random_state=C.SEED))]),
            {"reg__C": [0.03, 0.1, 0.3], "reg__epsilon": [0.0, 0.5]}),
    "lgbm": ("LightGBM Regressor (bonus)", None, {"reg__num_leaves": [15, 63]}),
}


def _lgbm_reg():
    from lightgbm import LGBMRegressor

    return Pipeline([("feats", F("word+char+hand", word_max=20_000, char_max=30_000)),
                     ("reg", LGBMRegressor(n_estimators=500, learning_rate=0.05, colsample_bytree=0.3,
                                           subsample=0.8, subsample_freq=1, random_state=C.SEED, verbose=-1,
                                           force_row_wise=True, max_bin=63))])


REG_MODELS["lgbm"] = (REG_MODELS["lgbm"][0], _lgbm_reg, REG_MODELS["lgbm"][2])


def reg_metrics(y, p):
    p = np.clip(p, 0, 10)
    out = {"mae": mean_absolute_error(y, p), "rmse": mean_squared_error(y, p) ** 0.5, "r2": r2_score(y, p),
           "spearman": spearmanr(y, p).statistic, "within_1": float(np.mean(np.abs(y - p) <= 1))}
    return out


rows_b, preds = [], {}
for target_name, sub_tr, sub_te in (
    ("all", trva, te),
    ("attack_only", trva[trva.label == 1], te[te.label == 1]),
):
    print(f"\n=== (B) Hồi quy Degree – {target_name}: train+val={len(sub_tr)} test={len(sub_te)} ===")
    for key, (name, build, grid) in REG_MODELS.items():
        t0 = time.time()
        pipe = build()
        pipe.set_params(memory=CACHE)
        gs = GridSearchCV(pipe, grid or {"feats__remainder": ["drop"]}, scoring="neg_mean_absolute_error",
                          cv=GroupKFold(3), n_jobs=1 if key in {"rf", "lgbm", "knn"} else 3)
        gs.fit(sub_tr[COLS], sub_tr["degree"].astype(float), groups=sub_tr["group"])
        p = gs.best_estimator_.predict(sub_te[COLS])
        met = reg_metrics(sub_te["degree"].values, p)
        rows_b.append({"target": target_name, "model": key, "model_name": name, "cv_mae": -gs.best_score_,
                       **met, "best_params": str(gs.best_params_), "seconds": time.time() - t0})
        preds[(target_name, key)] = p
        print(f"  {name:<34} CV-MAE={-gs.best_score_:.3f}  test MAE={met['mae']:.3f} RMSE={met['rmse']:.3f} "
              f"R²={met['r2']:.3f} ρ={met['spearman']:.3f}  ({time.time() - t0:.0f}s)")
        if key == "lasso":  # Lasso chọn đặc trưng: hệ số khác 0 dễ giải thích
            est = gs.best_estimator_
            ct = est.named_steps["feats"]
            names = np.concatenate([feature_names(ct.named_transformers_["user"]),
                                    ct.named_transformers_["ctx"].named_steps["cf"].get_feature_names_out()])
            coef = est.named_steps["reg"].coef_
            nz = np.flatnonzero(coef)
            top = nz[np.argsort(-np.abs(coef[nz]))][:30]
            pd.DataFrame({"feature": names[top], "coef": coef[top]}).to_csv(
                C.TABLE_DIR / f"spml_lasso_top_coef_{target_name}.csv", index=False)
            print(f"    Lasso giữ {len(nz)}/{len(coef)} đặc trưng khác 0")
reg_df = pd.DataFrame(rows_b)
reg_df.to_csv(C.TABLE_DIR / "spml_degree_regression.csv", index=False)
save_json({"context": rows_a, "regression": rows_b}, C.RESULTS_DIR / "spml_context_regression.json")

# ---------------------------------------------------------------------------
# Hình
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for ax, tgt, title in zip(axes, ("all", "attack_only"), ("Toàn bộ (Degree 0–10)", "Chỉ prompt tấn công (Degree 1–10)")):
    sub = reg_df[reg_df.target == tgt].sort_values("mae")
    P.barh(ax, sub["model_name"], sub["mae"], fmt="{:.2f}")
    ax.set_title(f"MAE trên test – {title}")
    ax.set_xlabel("MAE (càng thấp càng tốt)")
P.save(fig, "spml_regression_mae")

best_key = reg_df[reg_df.target == "all"].sort_values("mae").iloc[0]["model"]
for tgt, sub_te in (("all", te), ("attack_only", te[te.label == 1])):
    k = reg_df[reg_df.target == tgt].sort_values("mae").iloc[0]["model"]
    p = np.clip(preds[(tgt, k)], 0, 10)
    y = sub_te["degree"].values
    levels = sorted(np.unique(y))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    axes[0].boxplot([p[y == lv] for lv in levels], positions=levels, widths=0.6, showfliers=False,
                    patch_artist=True, boxprops={"facecolor": "#cde2fb", "edgecolor": P.SERIES[0]},
                    medianprops={"color": P.ATTACK})
    axes[0].plot([0, 10], [0, 10], ls=":", color=P.TEXT_SECONDARY, lw=1)
    axes[0].set(xlabel="Degree thật", ylabel="Degree dự đoán", title=f"{REG_MODELS[k][0]} – dự đoán theo Degree thật")
    res_by = pd.DataFrame({"y": y, "err": p - y}).groupby("y")["err"].agg(["mean", lambda e: np.mean(np.abs(e))])
    res_by.columns = ["bias", "mae"]
    axes[1].bar(res_by.index - 0.18, res_by["bias"], width=0.36, color=P.SERIES[0], label="Sai số trung bình (bias)")
    axes[1].bar(res_by.index + 0.18, res_by["mae"], width=0.36, color=P.ATTACK, label="MAE")
    axes[1].axhline(0, color=P.TEXT_SECONDARY, lw=1)
    axes[1].set(xlabel="Degree thật", ylabel="Sai số", title="Sai số theo mức độ – hồi quy về trung bình")
    axes[1].legend()
    P.save(fig, f"spml_regression_residuals_{tgt}")
    res_by.round(3).to_csv(C.TABLE_DIR / f"spml_regression_residuals_{tgt}.csv")

fig, ax = plt.subplots(figsize=(8, 3.2))
piv = ctx_df.pivot(index="model", columns="features", values="pr_auc")
x = np.arange(len(piv.index))
for i, col in enumerate(piv.columns):
    ax.bar(x + (i - 0.5) * 0.35, piv[col], width=0.35, color=P.SERIES[i], label=col)
    for xi, v in zip(x, piv[col]):
        ax.text(xi + (i - 0.5) * 0.35, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8, color=P.TEXT_SECONDARY)
ax.set_xticks(x, piv.index)
ax.set_ylim(max(0, piv.values.min() - 0.05), 1.0)
ax.set_ylabel("PR-AUC (test)")
ax.set_title("SPML: thêm ngữ cảnh system prompt có giúp không?")
ax.legend(loc="lower right")
ax.grid(axis="x", visible=False)
P.save(fig, "spml_context_effect")
print("\nXong.")
