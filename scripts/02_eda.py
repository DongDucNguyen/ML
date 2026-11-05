"""Bước 2 – Phân tích khám phá dữ liệu (EDA).

Chạy:  python scripts/02_eda.py
Hình:  reports/figures/eda_*.png     Bảng: reports/tables/eda_*.csv
"""
import _common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from pidetect import config as C
from pidetect import data as D
from pidetect import plots as P
from pidetect.features import ATTACK_PATTERNS, HandcraftedFeatures

df = D.prepare_all()
df["n_chars"] = df["text"].str.len()
ORDER = ["safeguard", "wild", "spml", "deepset", "jackhhao", "toxicchat", "gandalf"]

# 1. Quy mô & tỉ lệ tấn công ------------------------------------------------
agg = df.groupby("dataset").agg(n=("label", "size"), pos_rate=("label", "mean")).reindex(ORDER)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
P.barh(axes[0], agg.index, agg["n"], fmt="{:,.0f}")
axes[0].set_title("Số mẫu sau khi làm sạch")
P.barh(axes[1], agg.index, agg["pos_rate"] * 100, color=P.ATTACK, fmt="{:.1f}%")
axes[1].set_title("Tỉ lệ prompt tấn công (%)")
P.save(fig, "eda_01_dataset_size_posrate")

# 2. Độ dài văn bản theo lớp --------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=False)
for ax, name in zip(axes, C.MAIN_DATASETS):
    sub = df[df.dataset == name]
    bins = np.logspace(0.5, np.log10(max(sub.n_chars.max(), 10)), 40)
    for lab, color, txt in ((0, P.BENIGN, "Benign"), (1, P.ATTACK, "Tấn công")):
        ax.hist(sub.loc[sub.label == lab, "n_chars"], bins=bins, alpha=0.6, color=color, label=txt, density=True)
    ax.set_xscale("log")
    ax.set_title(name)
    ax.set_xlabel("Số ký tự (log)")
axes[0].legend()
axes[0].set_ylabel("Mật độ")
P.save(fig, "eda_02_length_by_class")
df.groupby(["dataset", "label"])["n_chars"].describe().round(0).to_csv(C.TABLE_DIR / "eda_length_stats.csv")

# 3. N-gram đặc trưng nhất của lớp tấn công (log-odds có làm trơn) ----------
rows = []
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, name in zip(axes, C.MAIN_DATASETS):
    sub = D.get_split(df, name, "train")
    cv = CountVectorizer(ngram_range=(1, 2), min_df=5, stop_words="english", binary=True, max_features=30000)
    X = cv.fit_transform(sub.text.str[: C.MAX_CHARS])
    y = sub.label.values
    pos = np.asarray(X[y == 1].sum(0)).ravel() + 1
    neg = np.asarray(X[y == 0].sum(0)).ravel() + 1
    lo = np.log(pos / (y == 1).sum()) - np.log(neg / (y == 0).sum())
    vocab = cv.get_feature_names_out()
    top = np.argsort(-lo)[:15]
    for i in top:
        rows.append({"dataset": name, "ngram": vocab[i], "log_odds": lo[i], "df_attack": int(pos[i] - 1),
                     "df_benign": int(neg[i] - 1)})
    P.barh(ax, vocab[top], lo[top], color=P.ATTACK, fmt="{:.1f}")
    ax.set_title(f"{name}: n-gram đặc trưng của tấn công")
    ax.set_xlabel("log-odds (tấn công / benign)")
P.save(fig, "eda_03_top_attack_ngrams")
pd.DataFrame(rows).to_csv(C.TABLE_DIR / "eda_top_attack_ngrams.csv", index=False)

# 4. In-the-Wild theo thời gian (drift) --------------------------------------
w = df[df.dataset == "wild"].copy()
w["month"] = w["date"].dt.to_period("M").dt.to_timestamp()
m = w.groupby("month").agg(n=("label", "size"), rate=("label", "mean"))
fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
axes[0].bar(m.index, m["n"], width=20, color=P.BENIGN)
axes[0].set_title("In-the-Wild: số prompt mỗi tháng")
axes[1].plot(m.index, m["rate"] * 100, marker="o", color=P.ATTACK)
axes[1].set_title("Tỉ lệ jailbreak mỗi tháng (%)")
for ax in axes:
    ax.axvline(pd.Timestamp(C.WILD_TRAIN_END), color=P.TEXT_SECONDARY, ls="--", lw=1)
    ax.axvline(pd.Timestamp(C.WILD_VAL_END), color=P.TEXT_SECONDARY, ls="--", lw=1)
axes[1].text(pd.Timestamp("2023-03-01"), axes[1].get_ylim()[1] * 0.85, "TRAIN", color=P.TEXT_SECONDARY)
axes[1].text(pd.Timestamp("2023-08-10"), axes[1].get_ylim()[1] * 0.85, "VAL", color=P.TEXT_SECONDARY)
axes[1].text(pd.Timestamp("2023-10-20"), axes[1].get_ylim()[1] * 0.85, "TEST", color=P.TEXT_SECONDARY)
P.save(fig, "eda_04_wild_over_time")

comm = w[w.label == 1]["community"].fillna("(không gán)").value_counts()
plat = pd.crosstab(w["platform"], w["label"], normalize="index")
comm.to_csv(C.TABLE_DIR / "eda_wild_communities.csv")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
P.barh(axes[0], comm.index, comm.values, color=P.ATTACK, fmt="{:,.0f}")
axes[0].set_title("Jailbreak theo cộng đồng (community)")
P.barh(axes[1], plat.index, plat[1].values * 100, color=P.ATTACK, fmt="{:.1f}%")
axes[1].set_title("Tỉ lệ jailbreak theo nền tảng (%)")
P.save(fig, "eda_05_wild_community_platform")

# 5. SPML – phân bố mức độ tấn công Degree ------------------------------------
s = df[df.dataset == "spml"]
deg = s["degree"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.bar(deg.index.astype(int), deg.values, color=[P.BENIGN] + [P.ATTACK] * (len(deg) - 1), width=0.7)
ax.set_xticks(deg.index.astype(int))
ax.set_xlabel("Degree (0 = benign, càng cao càng nhiều luật bị phá)")
ax.set_ylabel("Số mẫu")
ax.set_title("SPML: phân bố mức độ tấn công")
ax.grid(axis="x", visible=False)
P.save(fig, "eda_06_spml_degree")
print("SPML – số system prompt (nhóm):", s["group"].nunique())

# 6. Trùng lặp giữa các bộ dữ liệu -------------------------------------------
ov = D.overlap_matrix(df).loc[ORDER, ORDER]
off = ov.values.astype(float).copy()
np.fill_diagonal(off, np.nan)
fig, ax = plt.subplots(figsize=(7, 5.5))
P.heatmap(ax, off, ORDER, ORDER, fmt="{:.0f}", vmin=0, vmax=np.nanmax(off))
ax.set_title("Số câu trùng nhau giữa các bộ dữ liệu\n(nguy cơ rò rỉ khi đánh giá chéo)")
P.save(fig, "eda_07_overlap")

# 7. Tần suất mẫu từ khoá tấn công theo lớp -----------------------------------
hc = HandcraftedFeatures()
names = list(ATTACK_PATTERNS)
mat, labels = [], []
for name in ORDER:
    sub = df[df.dataset == name]
    sub = sub.sample(min(len(sub), 4000), random_state=C.SEED)
    F = hc.transform(sub.text.tolist())[:, -len(names):] > 0
    for lab in (0, 1):
        if (sub.label == lab).sum() >= 20:
            mat.append(F[sub.label.values == lab].mean(0))
            labels.append(f"{name} · {'tấn công' if lab else 'benign'}")
mat = np.array(mat)
pd.DataFrame(mat, index=labels, columns=names).round(3).to_csv(C.TABLE_DIR / "eda_keyword_rates.csv")
fig, ax = plt.subplots(figsize=(12, 6))
P.heatmap(ax, mat, labels, [n.replace("kw_", "") for n in names], fmt="{:.2f}", vmin=0, vmax=1)
ax.set_title("Tỉ lệ prompt chứa mẫu từ khoá tấn công (theo bộ dữ liệu & lớp)")
P.save(fig, "eda_08_keyword_rates")

print("Đã lưu hình EDA vào", C.FIG_DIR)
