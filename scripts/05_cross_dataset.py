"""Bước 5 – Khả năng tổng quát hoá: huấn luyện trên một nguồn, kiểm tra trên nguồn khác.

Nguồn huấn luyện: safeguard | wild | spml | combined (gộp cả 3).
Tập đánh giá: test của 3 bộ chính + 4 bộ ngoài miền (deepset, jackhhao,
toxicchat, gandalf). Câu trùng với tập huấn luyện bị loại trước khi đánh giá.
Ngưỡng τ chọn trên validation của *nguồn huấn luyện* (đúng như khi triển khai
thật: ta không có nhãn của dữ liệu mới).

Chạy:  python scripts/05_cross_dataset.py [--models logreg nb svm_linear rf lgbm]
"""
import argparse
import json

import _common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pidetect import config as C
from pidetect import data as D
from pidetect import evaluation as E
from pidetect import plots as P
from pidetect import training as T
from pidetect.models import MODEL_SPECS

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["lpm", "logreg", "nb", "svm_linear", "rf", "lgbm"])
ap.add_argument("--sources", nargs="+", default=["safeguard", "wild", "spml", "combined"])
ap.add_argument("--force", action="store_true")
args = ap.parse_args()

df = D.prepare_all()
EVAL_SETS = C.MAIN_DATASETS + C.OOD_DATASETS
RES_DIR = C.RESULTS_DIR / "cross"
RES_DIR.mkdir(parents=True, exist_ok=True)
SCORE_DIR = C.DATA_DIR / "scores"
SCORE_DIR.mkdir(parents=True, exist_ok=True)


def source_splits(src):
    if src == "combined":
        return D.get_split(df, C.MAIN_DATASETS, "train"), D.get_split(df, C.MAIN_DATASETS, "val")
    return D.get_split(df, src, "train"), D.get_split(df, src, "val")


def eval_frame(name):
    return D.get_split(df, name, "test") if name in C.MAIN_DATASETS else D.get_split(df, name, "ood")


rows, overlap_rows = [], []
for src in args.sources:
    tr, va = source_splits(src)
    seen = pd.concat([tr, va])
    print(f"\n=== Nguồn huấn luyện: {src} (train={len(tr)}, val={len(va)}, pos={tr.label.mean():.3f})")
    for key in args.models:
        out = RES_DIR / f"{src}__{key}.json"
        if out.exists() and not args.force:
            rows.extend(json.loads(out.read_text(encoding="utf-8")))
            print(f"  [skip] {key}")
            continue
        indomain = C.RESULTS_DIR / "indomain" / f"{src}__{key}.json"
        if indomain.exists():  # tái sử dụng siêu tham số đã chọn ở bước 3
            params = json.loads(indomain.read_text(encoding="utf-8"))["best_params"]
            model = T.fit_fixed(key, params, tr)
        else:
            model, info, _ = T.tune(key, "combined" if src == "combined" else src, tr)
            params = info["best_params"]
        if src == "combined":
            T.save_model(model, "combined", key)
        from pidetect.models import scores

        s_val = scores(model, va["text"].tolist())
        thr = E.choose_threshold(va["label"], s_val, "cost")
        cal = E.PlattCalibrator().fit(s_val, va["label"])
        res_list = []
        for ev in EVAL_SETS:
            ed = eval_frame(ev)
            ed, n_removed = D.remove_overlap(ed, seen)
            s = scores(model, ed["text"].tolist())
            m = E.binary_metrics(ed["label"], s, thr)
            r = {"source": src, "model": key, "eval": ev, "in_domain": ev == src or (src == "combined" and ev in C.MAIN_DATASETS),
                 "n_overlap_removed": n_removed, "params": params,
                 **{k: m[k] for k in ("n", "pos_rate", "pr_auc", "roc_auc", "f1", "precision", "recall", "fpr", "cost_per_1k")}}
            res_list.append(r)
            np.savez_compressed(SCORE_DIR / f"cross__{src}__{key}__{ev}.npz", y=ed["label"].values, s=s,
                                p=cal.predict_proba(s), thr=thr, hash=ed["text_hash"].values)
            if key == args.models[0]:
                overlap_rows.append({"source": src, "eval": ev, "n_overlap_removed": n_removed})
        np.savez_compressed(SCORE_DIR / f"cross__{src}__{key}__VAL.npz", y=va["label"].values, s=s_val,
                            p=cal.predict_proba(s_val), thr=thr)
        T.save_json(res_list, out)
        rows.extend(res_list)
        brief = " ".join(f"{r['eval'][:6]}={r['pr_auc']:.2f}" if not np.isnan(r['pr_auc']) else
                         f"{r['eval'][:6]}(R)={r['recall']:.2f}" for r in res_list)
        print(f"  {key:<11} PR-AUC: {brief}")

res = pd.DataFrame(rows)
res.to_csv(C.TABLE_DIR / "cross_dataset_results.csv", index=False)
if overlap_rows:
    pd.DataFrame(overlap_rows).pivot(index="source", columns="eval", values="n_overlap_removed").to_csv(
        C.TABLE_DIR / "cross_dataset_overlap_removed.csv")

# Tóm tắt: trung bình trên các tập ngoài miền (không tính gandalf vì chỉ có lớp dương)
ood = res[(~res.in_domain) & (res["eval"] != "gandalf")]
summary = ood.groupby(["source", "model"]).agg(ood_mean_pr_auc=("pr_auc", "mean"), ood_mean_recall=("recall", "mean"),
                                               ood_mean_fpr=("fpr", "mean")).reset_index()
gd = res[res["eval"] == "gandalf"][["source", "model", "recall"]].rename(columns={"recall": "gandalf_recall"})
summary = summary.merge(gd, on=["source", "model"], how="left").sort_values("ood_mean_pr_auc", ascending=False)
summary.to_csv(C.TABLE_DIR / "cross_dataset_summary.csv", index=False)
print("\nTóm tắt ngoài miền (trung bình PR-AUC trên các tập không dùng để huấn luyện):")
print(summary.round(3).to_string(index=False))

# So sánh CÔNG BẰNG giữa các nguồn: mỗi nguồn loại câu trùng với *chính nó*, nên tập đánh giá
# khác nhau (vd. jackhhao còn 1.457 câu khi train trên safeguard nhưng chỉ 1.355 khi train trên
# combined). Ở đây đánh giá mọi nguồn trên cùng tập con chung: các câu ngoài miền không trùng với
# BẤT KỲ bộ huấn luyện nào (= tập đánh giá của nguồn combined).
EXTERNAL = [e for e in C.OOD_DATASETS if e != "gandalf"]
common_rows = []
for key in models if (models := [m for m in args.models if m in set(res.model)]) else []:
    for ev in EXTERNAL + ["gandalf"]:
        ref = SCORE_DIR / f"cross__combined__{key}__{ev}.npz"
        if not ref.exists():
            continue
        keep = set(np.load(ref, allow_pickle=True)["hash"])
        for src in args.sources:
            f = SCORE_DIR / f"cross__{src}__{key}__{ev}.npz"
            if not f.exists():
                continue
            z = np.load(f, allow_pickle=True)
            mask = np.isin(z["hash"], list(keep))
            m = E.binary_metrics(z["y"][mask], z["s"][mask], float(z["thr"]))
            common_rows.append({"source": src, "model": key, "eval": ev, "n": int(mask.sum()),
                                **{k: m[k] for k in ("pr_auc", "recall", "fpr")}})
if common_rows:
    cm = pd.DataFrame(common_rows)
    cm.to_csv(C.TABLE_DIR / "cross_dataset_common_subset.csv", index=False)
    ext = cm[cm["eval"].isin(EXTERNAL)]
    fair = ext.groupby(["source", "model"]).agg(ext_mean_pr_auc=("pr_auc", "mean"), ext_mean_recall=("recall", "mean"),
                                                ext_mean_fpr=("fpr", "mean")).reset_index()
    fair = fair.merge(cm[cm["eval"] == "gandalf"][["source", "model", "recall"]].rename(
        columns={"recall": "gandalf_recall"}), on=["source", "model"], how="left")
    fair = fair.sort_values("ext_mean_pr_auc", ascending=False)
    fair.to_csv(C.TABLE_DIR / "cross_dataset_fair_summary.csv", index=False)
    print("\nSo sánh công bằng trên tập con chung (deepset, jackhhao, toxicchat):")
    print(fair.round(3).to_string(index=False))

# Hình: ma trận PR-AUC (nguồn × tập đánh giá) cho từng mô hình
models = [m for m in args.models if m in set(res.model)]
sources = [s for s in args.sources if s in set(res.source)]
ncol = min(3, len(models))
nrow = int(np.ceil(len(models) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.1 * nrow), squeeze=False, gridspec_kw={"hspace": 0.55})
for ax, key in zip(axes.ravel(), models):
    sub = res[res.model == key]
    piv = sub.pivot(index="source", columns="eval", values="pr_auc").reindex(index=sources, columns=EVAL_SETS)
    P.heatmap(ax, piv.values, sources, EVAL_SETS, fmt="{:.2f}")
    ax.set_title(f"{MODEL_SPECS[key].name_vi}: PR-AUC")
for ax in axes.ravel()[len(models):]:
    ax.axis("off")
fig.suptitle("Huấn luyện trên nguồn (hàng) → kiểm tra trên tập (cột). gandalf chỉ có lớp dương → xem Recall", y=1.02)
P.save(fig, "cross_dataset_pr_auc")

for metric, title in (("recall", "Recall tại τ (chọn trên val của nguồn)"), ("fpr", "FPR tại τ (chọn trên val của nguồn)")):
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.1 * nrow), squeeze=False, gridspec_kw={"hspace": 0.55})
    for ax, key in zip(axes.ravel(), models):
        piv = res[res.model == key].pivot(index="source", columns="eval", values=metric).reindex(index=sources, columns=EVAL_SETS)
        P.heatmap(ax, piv.values, sources, EVAL_SETS, fmt="{:.2f}", vmin=0, vmax=1)
        ax.set_title(f"{MODEL_SPECS[key].name_vi}: {metric}")
    for ax in axes.ravel()[len(models):]:
        ax.axis("off")
    fig.suptitle(title, y=1.02)
    P.save(fig, f"cross_dataset_{metric}")
print("Xong.")
