"""Bước 6 – Phân tích kết quả & lỗi của bộ phát hiện cuối cùng (huấn luyện trên combined).

Nội dung:
  6.1 Learning curve (bias–variance) – Slide Model Assessment
  6.2 Hiệu chỉnh xác suất (reliability diagram, ECE) & đường chi phí theo ngưỡng
  6.3 Giải thích mô hình: hệ số Logistic lớn nhất → phát hiện đặc trưng "giả" (shortcut)
  6.4 Phân rã lỗi theo nhóm: độ dài, cộng đồng/nền tảng/tháng (wild), Degree (spml), độc hại (toxicchat)
  6.5 Ví dụ FN / FP tự tin nhất (xuất CSV để đọc thủ công)
  6.6 Độ bền trước biến đổi đối kháng + phòng thủ bằng chuẩn hoá văn bản
  6.7 Conformal prediction (bonus): bảo đảm độ phủ & điều gì xảy ra khi dữ liệu dịch chuyển

Chạy (sau bước 5):  python scripts/06_error_analysis.py [--model logreg]
"""
import argparse
import random

import _common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone

from pidetect import config as C
from pidetect import data as D
from pidetect import evaluation as E
from pidetect import plots as P
from pidetect import training as T
from pidetect.features import feature_names
from pidetect.models import MODEL_SPECS, scores
from pidetect.robustness import PERTURBATIONS, defensive_normalize

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="logreg", help="mô hình combined dùng để phân tích chi tiết")
ap.add_argument("--compare", nargs="+", default=["lpm", "logreg", "nb", "svm_linear", "rf", "lgbm"])
args = ap.parse_args()

df = D.prepare_all()
SCORE_DIR = C.DATA_DIR / "scores"
EVAL_SETS = C.MAIN_DATASETS + C.OOD_DATASETS
avail = [k for k in args.compare if (C.MODELS_DIR / f"combined__{k}.joblib").exists()]
if args.model not in avail:
    raise SystemExit(f"Chưa có models/combined__{args.model}.joblib – hãy chạy scripts/05_cross_dataset.py trước.")
model = T.load_model("combined", args.model)
tr = D.get_split(df, C.MAIN_DATASETS, "train")
va = D.get_split(df, C.MAIN_DATASETS, "val")
seen = pd.concat([tr, va])


def load_scores(key, ev):
    z = np.load(SCORE_DIR / f"cross__combined__{key}__{ev}.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def eval_frame(name):
    ed = D.get_split(df, name, "test") if name in C.MAIN_DATASETS else D.get_split(df, name, "ood")
    return D.remove_overlap(ed, seen)[0]


summary = {}

# ---------------------------------------------------------------------------
# 6.1 Learning curve
# ---------------------------------------------------------------------------
print("6.1 Learning curve …")
rng = np.random.default_rng(C.SEED)
fracs = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
ood_lc = pd.concat([eval_frame("jackhhao"), eval_frame("toxicchat"), eval_frame("deepset")], ignore_index=True)
lc_rows = []
for key in [k for k in ("logreg", "nb") if k in avail]:
    base = T.load_model("combined", key)
    for f in fracs:
        sub = D.maybe_subsample(tr, int(len(tr) * f), seed=C.SEED) if f < 1 else tr
        m = clone(base).fit(sub["text"].tolist(), sub["label"].values)
        for name, frame in (("train (phần đã dùng)", sub), ("validation (cùng miền)", va), ("ngoài miền (OOD)", ood_lc)):
            s = scores(m, frame["text"].tolist())
            lc_rows.append({"model": key, "frac": f, "n_train": len(sub), "set": name,
                            "pr_auc": E.binary_metrics(frame["label"], s, 0.5)["pr_auc"]})
lc = pd.DataFrame(lc_rows)
lc.to_csv(C.TABLE_DIR / "error_learning_curve.csv", index=False)
fig, axes = plt.subplots(1, lc.model.nunique(), figsize=(6 * lc.model.nunique(), 3.8), squeeze=False)
for ax, (key, g) in zip(axes.ravel(), lc.groupby("model")):
    for i, (name, gg) in enumerate(g.groupby("set", sort=False)):
        ax.plot(gg["n_train"], gg["pr_auc"], marker="o", color=P.SERIES[i], label=name)
    ax.set(xscale="log", xlabel="Số mẫu huấn luyện", ylabel="PR-AUC", title=f"Learning curve – {MODEL_SPECS[key].name_vi}")
    ax.legend(fontsize=8)
P.save(fig, "error_learning_curve")

# ---------------------------------------------------------------------------
# 6.2 Hiệu chỉnh & chi phí
# ---------------------------------------------------------------------------
print("6.2 Calibration & cost …")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
cal_rows = []
for i, key in enumerate(avail):
    ys, raw, pl = [], [], []
    for ev in C.MAIN_DATASETS:
        z = load_scores(key, ev)
        ys.append(z["y"]); raw.append(z["s"]); pl.append(z["p"])
    y, s, p = np.concatenate(ys), np.concatenate(raw), np.concatenate(pl)
    is_prob = s.min() >= 0 and s.max() <= 1
    row = {"model": key, "ece_platt": E.expected_calibration_error(y, p)}
    if is_prob:
        row["ece_raw"] = E.expected_calibration_error(y, s)
        t = E.reliability_table(y, s)
        axes[0].plot(t["mean_pred"], t["frac_pos"], marker="o", color=P.SERIES[i % 8], label=MODEL_SPECS[key].name_vi)
    t = E.reliability_table(y, p)
    axes[1].plot(t["mean_pred"], t["frac_pos"], marker="o", color=P.SERIES[i % 8], label=MODEL_SPECS[key].name_vi)
    cal_rows.append(row)
for ax, title in zip(axes, ("Xác suất gốc của mô hình", "Sau Platt scaling (fit trên val)")):
    ax.plot([0, 1], [0, 1], ls=":", color=P.TEXT_SECONDARY)
    ax.set(xlabel="Xác suất dự đoán trung bình", ylabel="Tỉ lệ tấn công thực tế", title=title)
axes[1].legend(fontsize=8)
P.save(fig, "error_calibration")
pd.DataFrame(cal_rows).to_csv(C.TABLE_DIR / "error_calibration_ece.csv", index=False)

z_val = load_scores(args.model, "VAL")
cc_val = E.cost_curve(z_val["y"], z_val["s"])
z_w = load_scores(args.model, "wild")
cc_w = E.cost_curve(z_w["y"], z_w["s"])
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(cc_val["threshold"], cc_val["cost_per_1k"], color=P.SERIES[0], label="validation (combined)")
axes[0].plot(cc_w["threshold"], cc_w["cost_per_1k"], color=P.SERIES[1], label="test wild (tương lai)")
axes[0].axvline(float(z_val["thr"]), ls="--", color=P.TEXT_SECONDARY, lw=1)
axes[0].set(xlabel="Ngưỡng τ", ylabel=f"Chi phí /1000 prompt (FN×{C.COST_FN:g} + FP×{C.COST_FP:g})",
            title="Chi phí theo ngưỡng")
axes[0].legend()
axes[1].plot(cc_val["threshold"], cc_val["recall"], color=P.SERIES[0], label="Recall")
axes[1].plot(cc_val["threshold"], cc_val["precision"], color=P.SERIES[2], label="Precision")
axes[1].axvline(float(z_val["thr"]), ls="--", color=P.TEXT_SECONDARY, lw=1)
axes[1].set(xlabel="Ngưỡng τ", title="Đánh đổi Precision–Recall (validation)")
axes[1].legend()
P.save(fig, "error_cost_threshold")

# ---------------------------------------------------------------------------
# 6.3 Giải thích: hệ số Logistic
# ---------------------------------------------------------------------------
print("6.3 Hệ số Logistic …")
if "logreg" in avail:
    lr = T.load_model("combined", "logreg")
    names = feature_names(lr.named_steps["feats"])
    coef = lr.named_steps["clf"].coef_.ravel()
    order = np.argsort(coef)
    top_pos, top_neg = order[-20:][::-1], order[:20]
    coef_df = pd.DataFrame({"feature": np.r_[names[top_pos], names[top_neg]],
                            "coef": np.r_[coef[top_pos], coef[top_neg]],
                            "odds_ratio": np.exp(np.r_[coef[top_pos], coef[top_neg]])})
    coef_df.to_csv(C.TABLE_DIR / "error_logreg_top_coef.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    P.barh(axes[0], names[top_pos], coef[top_pos], color=P.ATTACK, fmt="{:.2f}")
    axes[0].set_title("Đẩy về TẤN CÔNG (hệ số dương lớn nhất)")
    P.barh(axes[1], names[top_neg], -coef[top_neg], color=P.BENIGN, fmt="{:.2f}")
    axes[1].set_title("Đẩy về BENIGN (|hệ số âm| lớn nhất)")
    P.save(fig, "error_logreg_coefficients")

# ---------------------------------------------------------------------------
# 6.4 Phân rã lỗi theo nhóm
# ---------------------------------------------------------------------------
print("6.4 Phân rã lỗi …")
thr = float(z_val["thr"])
frames = {}
for ev in EVAL_SETS:
    ed = eval_frame(ev)
    z = load_scores(args.model, ev)
    assert len(ed) == len(z["y"]), ev
    ed = ed.assign(score=z["s"], prob=z["p"], pred=(z["s"] >= thr).astype(int))
    ed["error"] = np.select([(ed.pred == 1) & (ed.label == 0), (ed.pred == 0) & (ed.label == 1)], ["FP", "FN"], "")
    frames[ev] = ed
allf = pd.concat(frames.values(), ignore_index=True)
allf["len_bucket"] = pd.cut(allf["text"].str.len(), [0, 50, 150, 500, 1500, 5000, 10**7],
                            labels=["≤50", "51–150", "151–500", "501–1.5k", "1.5k–5k", ">5k"])
grp_rows = []


def rates(g):
    pos, neg = g[g.label == 1], g[g.label == 0]
    return pd.Series({"n": len(g), "n_attack": len(pos), "recall": (pos.pred == 1).mean() if len(pos) else np.nan,
                      "fpr": (neg.pred == 1).mean() if len(neg) else np.nan})


by_len = allf.groupby(["dataset", "len_bucket"], observed=True).apply(rates, include_groups=False).reset_index()
by_len.to_csv(C.TABLE_DIR / "error_by_length.csv", index=False)
w = frames["wild"]
w_comm = w[w.label == 1].assign(community=w["community"].fillna("(không gán)")).groupby("community").apply(
    rates, include_groups=False).sort_values("n", ascending=False)
w_plat = w.groupby("platform").apply(rates, include_groups=False)
w_month = w.assign(month=w["date"].dt.to_period("M").astype(str)).groupby("month").apply(rates, include_groups=False)
s_deg = frames["spml"].groupby("degree").apply(rates, include_groups=False)
tc = frames["toxicchat"].groupby("toxicity").apply(rates, include_groups=False)
for name, t in (("wild_community", w_comm), ("wild_platform", w_plat), ("wild_month", w_month),
                ("spml_degree", s_deg), ("toxicchat_toxicity", tc)):
    t.round(4).to_csv(C.TABLE_DIR / f"error_by_{name}.csv")

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
bl = by_len[by_len.dataset.isin(["safeguard", "wild", "jackhhao", "toxicchat"])]
for i, (ds, g) in enumerate(bl.groupby("dataset", sort=False)):
    axes[0, 0].plot(g["len_bucket"].astype(str), g["recall"], marker="o", color=P.SERIES[i], label=ds)
axes[0, 0].set(title="Recall theo độ dài prompt", ylabel="Recall", ylim=(0, 1.05))
axes[0, 0].legend(fontsize=8)
P.barh(axes[0, 1], w_comm.index, w_comm["recall"].values, color=P.ATTACK, fmt="{:.2f}")
axes[0, 1].set_title("wild: Recall theo cộng đồng jailbreak")
axes[1, 0].plot(w_month.index, w_month["recall"], marker="o", color=P.SERIES[1], label="Recall")
axes[1, 0].plot(w_month.index, w_month["fpr"], marker="s", color=P.SERIES[0], label="FPR")
axes[1, 0].set(title="wild (test 10–12/2023): theo tháng", ylim=(0, 1.05))
axes[1, 0].legend()
axes[1, 1].bar(s_deg.index.astype(int), s_deg["recall"].fillna(1 - s_deg["fpr"]), color=P.SERIES[0], width=0.7)
axes[1, 1].set(title="spml: Recall theo Degree (cột 0 = 1 − FPR)", xlabel="Degree", ylim=(0, 1.05))
axes[1, 1].grid(axis="x", visible=False)
P.save(fig, "error_breakdown")

# ---------------------------------------------------------------------------
# 6.5 Ví dụ lỗi
# ---------------------------------------------------------------------------
print("6.5 Ví dụ lỗi …")
ex = []
for ev, ed in frames.items():
    fn = ed[ed.error == "FN"].nsmallest(15, "score")
    fp = ed[ed.error == "FP"].nlargest(15, "score")
    ex.append(pd.concat([fn, fp]))
ex = pd.concat(ex)
ex["text_short"] = ex["text"].str.replace(r"\s+", " ", regex=True).str[:400]
ex[["dataset", "error", "score", "prob", "label", "text_short"]].to_csv(C.TABLE_DIR / "error_examples.csv", index=False)
err_counts = allf.groupby("dataset").apply(lambda g: pd.Series({
    "n": len(g), "FN": int((g.error == "FN").sum()), "FP": int((g.error == "FP").sum())}), include_groups=False)
err_counts.to_csv(C.TABLE_DIR / "error_counts.csv")
print(err_counts)

# ---------------------------------------------------------------------------
# 6.6 Độ bền trước biến đổi đối kháng
# ---------------------------------------------------------------------------
print("6.6 Robustness …")
atk = pd.concat([frames["safeguard"], frames["wild"], frames["gandalf"], frames["deepset"]])
atk = atk[atk.label == 1].sample(min(1500, int((atk.label == 1).sum())), random_state=C.SEED)
ben = pd.concat([frames["safeguard"], frames["wild"]])
ben = ben[ben.label == 0].sample(1500, random_state=C.SEED)
rob_rows = []
for key in avail:
    m = T.load_model("combined", key)
    thr_k = float(load_scores(key, "VAL")["thr"])
    for pname, fn in PERTURBATIONS.items():
        rng = random.Random(C.SEED)
        pert = [fn(t, rng) for t in atk["text"]]
        for defense in (False, True):
            texts = [defensive_normalize(t) for t in pert] if defense else pert
            rec = float((scores(m, texts) >= thr_k).mean())
            rob_rows.append({"model": key, "perturbation": pname, "defense": defense, "recall": rec})
    btexts = ben["text"].tolist()
    for defense in (False, True):
        texts = [defensive_normalize(t) for t in btexts] if defense else btexts
        rob_rows.append({"model": key, "perturbation": "FPR trên prompt benign", "defense": defense,
                         "recall": float((scores(m, texts) >= thr_k).mean())})
rob = pd.DataFrame(rob_rows)
rob.to_csv(C.TABLE_DIR / "error_robustness.csv", index=False)
pnames = list(PERTURBATIONS)
for defense in (False, True):
    piv = rob[(rob.defense == defense) & (rob.perturbation.isin(pnames))].pivot(
        index="perturbation", columns="model", values="recall").reindex(index=pnames, columns=avail)
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(avail), 4.6))
    P.heatmap(ax, piv.values, pnames, [MODEL_SPECS[k].name_vi for k in avail], fmt="{:.2f}")
    ax.set_title("Recall dưới biến đổi đối kháng" + (" – CÓ chuẩn hoá phòng thủ" if defense else " – không phòng thủ"))
    P.save(fig, f"error_robustness_{'defense' if defense else 'raw'}")

# ---------------------------------------------------------------------------
# 6.7 Conformal prediction (bonus)
# ---------------------------------------------------------------------------
print("6.7 Conformal …")
conf_rows = []
pv, yv = z_val["p"], z_val["y"]
for alpha in (0.05, 0.10):
    for mondrian in (False, True):
        cp = E.SplitConformalClassifier(alpha=alpha, mondrian=mondrian).fit(pv, yv)
        for ev in EVAL_SETS:
            z = load_scores(args.model, ev)
            r = cp.evaluate(z["p"], z["y"])
            conf_rows.append({"eval": ev, "in_domain": ev in C.MAIN_DATASETS, **r})
conf = pd.DataFrame(conf_rows)
conf.to_csv(C.TABLE_DIR / "conformal_results.csv", index=False)
sub = conf[(conf.alpha == 0.05)]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
x = np.arange(len(EVAL_SETS))
for i, mond in enumerate((False, True)):
    s = sub[sub.mondrian == mond].set_index("eval").reindex(EVAL_SETS)
    lbl = "Mondrian (theo lớp)" if mond else "Chuẩn (marginal)"
    axes[0].bar(x + (i - 0.5) * 0.38, s["coverage_attack"], width=0.38, color=P.SERIES[i], label=lbl)
    axes[1].bar(x + (i - 0.5) * 0.38, s["frac_uncertain"], width=0.38, color=P.SERIES[i], label=lbl)
axes[0].axhline(0.95, ls="--", color=P.TEXT_SECONDARY, lw=1)
axes[0].set(title="Độ phủ trên lớp TẤN CÔNG (mục tiêu 95%)", ylim=(0, 1.05))
axes[1].set(title="Tỉ lệ 'không chắc' {0,1} → chuyển người duyệt")
for ax in axes:
    ax.set_xticks(x, EVAL_SETS, rotation=30, ha="right")
    ax.grid(axis="x", visible=False)
axes[0].legend(fontsize=8, loc="lower left")
P.save(fig, "conformal_coverage")
print(sub[["eval", "mondrian", "coverage", "coverage_attack", "coverage_benign", "frac_uncertain"]].round(3).to_string(index=False))

T.save_json({"model": args.model, "threshold": thr, "error_counts": err_counts.to_dict()},
            C.RESULTS_DIR / "error_analysis.json")
print("Xong.")
