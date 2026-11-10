"""Bước 3b – In-the-Wild: chia ngẫu nhiên vs chia theo thời gian (drift & gần-trùng-lặp).

Câu hỏi: vì sao PR-AUC cross-validation ~0.8 nhưng test (10–12/2023) chỉ ~0.2?
  1. So sánh cùng mô hình khi chia NGẪU NHIÊN (cách làm phổ biến nhưng sai cho dữ
     liệu có thời gian) với chia THEO THỜI GIAN.
  2. Đo "gần trùng lặp": với mỗi prompt test, độ tương đồng cosine (TF-IDF ký tự)
     tới prompt train gần nhất. Jailbreak thường là biến thể của nhau (DAN v6, v7…)
     → chia ngẫu nhiên làm test "nhìn thấy" anh em của nó ở train → điểm ảo.
  3. PR-AUC theo nguồn (source) và theo tháng trên giai đoạn test.

Chạy:  python scripts/03b_wild_drift.py
"""
import _common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from pidetect import config as C
from pidetect import data as D
from pidetect import evaluation as E
from pidetect import plots as P
from pidetect import training as T
from pidetect.models import scores

df = D.prepare_all()
w = df[df.dataset == "wild"].reset_index(drop=True)
temporal = {s: w[w.split == s].reset_index(drop=True) for s in ("train", "val", "test")}

# Chia ngẫu nhiên với cùng kích thước & phân tầng theo nhãn
idx_trval, idx_te = train_test_split(np.arange(len(w)), test_size=len(temporal["test"]), stratify=w.label,
                                     random_state=C.SEED)
idx_tr, idx_va = train_test_split(idx_trval, test_size=len(temporal["val"]), stratify=w.label.iloc[idx_trval],
                                  random_state=C.SEED)
random_split = {"train": w.iloc[idx_tr].reset_index(drop=True), "val": w.iloc[idx_va].reset_index(drop=True),
                "test": w.iloc[idx_te].reset_index(drop=True)}

rows, sims = [], {}
for split_name, sp in (("Ngẫu nhiên", random_split), ("Theo thời gian", temporal)):
    for key in ("logreg", "nb", "svm_linear"):
        params = pd.read_json(C.RESULTS_DIR / "indomain" / f"wild__{key}.json", typ="series")["best_params"]
        m = T.fit_fixed(key, params, sp["train"])
        s_val, s_te = scores(m, sp["val"]["text"].tolist()), scores(m, sp["test"]["text"].tolist())
        thr = E.choose_threshold(sp["val"]["label"], s_val, "cost")
        mt = E.binary_metrics(sp["test"]["label"], s_te, thr)
        mv = E.binary_metrics(sp["val"]["label"], s_val, thr)
        rows.append({"split": split_name, "model": key, "val_pr_auc": mv["pr_auc"], "test_pr_auc": mt["pr_auc"],
                     "test_roc_auc": mt["roc_auc"], "test_recall": mt["recall"], "test_fpr": mt["fpr"],
                     "test_f1": mt["f1"]})
        print(f"{split_name:<15} {key:<11} val PR-AUC={mv['pr_auc']:.3f}  test PR-AUC={mt['pr_auc']:.3f}  "
              f"F1={mt['f1']:.3f}")
        if key == "logreg" and split_name == "Theo thời gian":
            te = sp["test"].assign(score=s_te)
            src = te.groupby("source").apply(lambda g: pd.Series({
                "n": len(g), "n_jb": int(g.label.sum()),
                "pr_auc": E.binary_metrics(g.label, g.score, thr)["pr_auc"] if 0 < g.label.sum() < len(g) else np.nan,
                "recall": ((g.score >= thr) & (g.label == 1)).sum() / max(g.label.sum(), 1),
                "fpr": ((g.score >= thr) & (g.label == 0)).sum() / max((g.label == 0).sum(), 1)}), include_groups=False)
            src.sort_values("n", ascending=False).round(3).to_csv(C.TABLE_DIR / "wild_drift_by_source.csv")
            print(src.sort_values("n", ascending=False).round(3))
    # Gần trùng lặp: cosine tới hàng xóm gần nhất trong train (chỉ xét prompt jailbreak ở test)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=200_000)
    Xtr = vec.fit_transform(sp["train"]["text"].str[: C.MAX_CHARS])
    te_jb = sp["test"][sp["test"].label == 1]
    Xte = vec.transform(te_jb["text"].str[: C.MAX_CHARS])
    sim = np.asarray((Xte @ Xtr.T).max(1).todense()).ravel()
    sims[split_name] = sim

res = pd.DataFrame(rows)
res.to_csv(C.TABLE_DIR / "wild_random_vs_temporal.csv", index=False)
near = pd.DataFrame({k: {"median_max_cosine": float(np.median(v)), "frac_cos>0.8": float((v > 0.8).mean()),
                         "frac_cos>0.9": float((v > 0.9).mean())} for k, v in sims.items()}).T
near.to_csv(C.TABLE_DIR / "wild_near_duplicates.csv")
print(near.round(3))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
piv = res.pivot(index="model", columns="split", values="test_pr_auc")[["Ngẫu nhiên", "Theo thời gian"]]
x = np.arange(len(piv))
for i, col in enumerate(piv.columns):
    axes[0].bar(x + (i - 0.5) * 0.36, piv[col], width=0.36, color=P.SERIES[i], label=col)
    for xi, v in zip(x, piv[col]):
        axes[0].text(xi + (i - 0.5) * 0.36, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=P.TEXT_SECONDARY)
axes[0].set_xticks(x, piv.index)
axes[0].set(title="PR-AUC trên test: ngẫu nhiên vs theo thời gian", ylim=(0, 1))
axes[0].legend()
axes[0].grid(axis="x", visible=False)
bins = np.linspace(0, 1, 26)
for i, (k, v) in enumerate(sims.items()):
    axes[1].hist(v, bins=bins, alpha=0.6, color=P.SERIES[i], label=k, density=True)
axes[1].set(title="Jailbreak test: cosine tới câu train gần nhất", xlabel="cosine (TF-IDF ký tự)", ylabel="mật độ")
axes[1].legend()
P.save(fig, "wild_random_vs_temporal")
T.save_json({"results": rows, "near_duplicates": near.to_dict()}, C.RESULTS_DIR / "wild_drift.json")
print("Xong.")
