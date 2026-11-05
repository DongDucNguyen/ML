"""Tải, chuẩn hoá, khử trùng lặp và chia train/val/test cho các bộ dữ liệu.

Mỗi bộ dữ liệu được đưa về cùng một lược đồ (schema):

    text, label (0 = benign, 1 = injection/jailbreak), dataset, split,
    text_hash  + các cột meta riêng (date, platform, community, degree, ...)

Các quyết định quan trọng để tránh rò rỉ dữ liệu (data leakage – slide Model
Assessment):
  * Khử trùng lặp *trước* khi chia, nên một câu không thể nằm ở cả train và test.
  * Câu bị gán nhãn mâu thuẫn (cùng nội dung, khác nhãn) bị loại hẳn.
  * In-the-Wild: chia theo **thời gian** (train quá khứ → test tương lai).
  * SPML: chia theo **nhóm system prompt** (GroupShuffleSplit) để mô hình không
    "học thuộc" chatbot cụ thể.
  * Khi đánh giá ngoài miền (OOD) luôn loại các câu đã có trong tập train.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from . import config as C

# ---------------------------------------------------------------------------
# Tiện ích văn bản
# ---------------------------------------------------------------------------
_WS = re.compile(r"\s+")


def normalize_for_hash(text: str) -> str:
    """Chuẩn hoá nhẹ để phát hiện trùng lặp: NFKC, chữ thường, gộp khoảng trắng."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return _WS.sub(" ", text).strip()


def text_hash(text: str) -> str:
    return hashlib.md5(normalize_for_hash(text).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Tải dữ liệu thô từ HuggingFace Hub
# ---------------------------------------------------------------------------
def download_all(names: list[str] | None = None) -> dict[str, list[Path]]:
    from huggingface_hub import hf_hub_download

    names = names or list(C.DATASETS)
    out: dict[str, list[Path]] = {}
    for name in names:
        spec = C.DATASETS[name]
        paths = []
        for f in spec["files"]:
            p = hf_hub_download(spec["repo"], f, repo_type="dataset", local_dir=C.RAW_DIR / name)
            paths.append(Path(p))
        out[name] = paths
        print(f"[download] {name:<10} ← {spec['repo']} ({len(paths)} file)")
    return out


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _raw(name: str) -> list[pd.DataFrame]:
    spec = C.DATASETS[name]
    paths = [C.RAW_DIR / name / f for f in spec["files"]]
    if not all(p.exists() for p in paths):
        download_all([name])
    return [_read(p) for p in paths]


# ---------------------------------------------------------------------------
# Loader cho từng bộ dữ liệu → lược đồ chung
# ---------------------------------------------------------------------------
def load_safeguard() -> pd.DataFrame:
    tr, te = _raw("safeguard")
    tr = tr.assign(orig_split="train")
    te = te.assign(orig_split="test")
    df = pd.concat([tr, te], ignore_index=True)
    return pd.DataFrame({"text": df["text"], "label": df["label"].astype(int), "orig_split": df["orig_split"]})


def load_wild() -> pd.DataFrame:
    jb, reg = _raw("wild")
    df = pd.concat([jb, reg], ignore_index=True)
    return pd.DataFrame(
        {
            "text": df["prompt"],
            "label": df["jailbreak"].astype(int),
            "date": pd.to_datetime(df["date"]),
            "platform": df["platform"],
            "source": df["source"],
            "community": df.get("community"),
        }
    )


def load_spml() -> pd.DataFrame:
    (df,) = _raw("spml")
    return pd.DataFrame(
        {
            "text": df["User Prompt"],
            "label": df["Prompt injection"].astype(int),
            "degree": df["Degree"].astype(int),
            "system_prompt": df["System Prompt"],
            "source": df["Source"].fillna("GPT-4 generated"),
        }
    )


def load_deepset() -> pd.DataFrame:
    df = pd.concat(_raw("deepset"), ignore_index=True)
    return pd.DataFrame({"text": df["text"], "label": df["label"].astype(int)})


def load_jackhhao() -> pd.DataFrame:
    (df,) = _raw("jackhhao")
    return pd.DataFrame({"text": df["prompt"], "label": (df["type"] == "jailbreak").astype(int)})


def load_toxicchat() -> pd.DataFrame:
    (df,) = _raw("toxicchat")
    return pd.DataFrame(
        {"text": df["user_input"], "label": df["jailbreaking"].astype(int), "toxicity": df["toxicity"].astype(int)}
    )


def load_gandalf() -> pd.DataFrame:
    (df,) = _raw("gandalf")
    return pd.DataFrame({"text": df["text"], "label": 1})


LOADERS = {
    "safeguard": load_safeguard,
    "wild": load_wild,
    "spml": load_spml,
    "deepset": load_deepset,
    "jackhhao": load_jackhhao,
    "toxicchat": load_toxicchat,
    "gandalf": load_gandalf,
}


# ---------------------------------------------------------------------------
# Làm sạch + khử trùng lặp
# ---------------------------------------------------------------------------
def clean_and_dedupe(df: pd.DataFrame, name: str, sort_by: str | None = None) -> tuple[pd.DataFrame, dict]:
    stats = {"dataset": name, "n_raw": len(df)}
    df = df.copy()
    df["text"] = df["text"].astype("string").fillna("").str.strip()
    df = df[df["text"].str.len() > 0]
    stats["n_empty_removed"] = stats["n_raw"] - len(df)

    df["text_hash"] = df["text"].map(text_hash)
    n_labels = df.groupby("text_hash")["label"].nunique()
    conflict = n_labels[n_labels > 1].index
    stats["n_conflict_hashes"] = int(len(conflict))
    stats["n_conflict_rows"] = int(df["text_hash"].isin(conflict).sum())
    df = df[~df["text_hash"].isin(conflict)]

    if sort_by is not None:
        df = df.sort_values(sort_by, kind="stable")
    before = len(df)
    df = df.drop_duplicates("text_hash", keep="first")
    stats["n_duplicates_removed"] = before - len(df)
    stats["n_clean"] = len(df)
    stats["pos_rate"] = float(df["label"].mean())
    df.insert(0, "dataset", name)
    return df.reset_index(drop=True), stats


# ---------------------------------------------------------------------------
# Chia tập
# ---------------------------------------------------------------------------
def split_safeguard(df: pd.DataFrame) -> pd.DataFrame:
    """Giữ nguyên test chính thức; tách 20% train làm validation (phân tầng theo nhãn)."""
    df = df.copy()
    df["split"] = np.where(df["orig_split"] == "test", "test", "train")
    tr_idx = df.index[df["split"] == "train"]
    _, val_idx = train_test_split(tr_idx, test_size=0.2, stratify=df.loc[tr_idx, "label"], random_state=C.SEED)
    df.loc[val_idx, "split"] = "val"
    return df.drop(columns="orig_split")


def split_wild(df: pd.DataFrame) -> pd.DataFrame:
    """Chia theo thời gian: train ≤ 07/2023 < val ≤ 09/2023 < test."""
    df = df.copy()
    d = df["date"]
    df["split"] = np.select(
        [d <= C.WILD_TRAIN_END, d <= C.WILD_VAL_END], ["train", "val"], default="test"
    )
    return df


def split_spml(df: pd.DataFrame) -> pd.DataFrame:
    """Chia theo nhóm system prompt: 60/20/20 (không chatbot nào nằm ở hai tập)."""
    df = df.copy()
    groups = df["system_prompt"].map(text_hash)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=C.SEED)
    trval, test = next(gss.split(df, df["label"], groups))
    df["split"] = "train"
    df.loc[df.index[test], "split"] = "test"
    sub = df.iloc[trval]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=C.SEED)
    _, val = next(gss2.split(sub, sub["label"], groups.iloc[trval]))
    df.loc[sub.index[val], "split"] = "val"
    df["group"] = groups
    return df


SPLITTERS = {"safeguard": split_safeguard, "wild": split_wild, "spml": split_spml}


# ---------------------------------------------------------------------------
# API chính
# ---------------------------------------------------------------------------
def prepare_all(force: bool = False) -> pd.DataFrame:
    """Tạo (hoặc đọc từ cache) bảng dữ liệu hợp nhất ``data/processed/all.parquet``."""
    out_path = C.PROCESSED_DIR / "all.parquet"
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)

    frames, stats = [], []
    for name in C.DATASETS:
        raw = LOADERS[name]()
        sort_by = "date" if name == "wild" else None  # giữ bản xuất hiện sớm nhất
        df, st = clean_and_dedupe(raw, name, sort_by=sort_by)
        df = SPLITTERS[name](df) if name in SPLITTERS else df.assign(split="ood")
        frames.append(df)
        stats.append(st)
        print(f"[prepare] {name:<10} raw={st['n_raw']:>6}  clean={st['n_clean']:>6}  "
              f"dup={st['n_duplicates_removed']:>4}  conflict={st['n_conflict_rows']:>3}  pos={st['pos_rate']:.3f}")

    all_df = pd.concat(frames, ignore_index=True)
    for col in ("community", "platform", "source", "system_prompt", "group"):
        if col in all_df:
            all_df[col] = all_df[col].astype("string")
    all_df.to_parquet(out_path, index=False)
    pd.DataFrame(stats).to_csv(C.TABLE_DIR / "data_cleaning_stats.csv", index=False)
    return all_df


def get_split(df: pd.DataFrame, dataset: str | list[str], split: str | list[str]) -> pd.DataFrame:
    ds = [dataset] if isinstance(dataset, str) else dataset
    sp = [split] if isinstance(split, str) else split
    return df[df["dataset"].isin(ds) & df["split"].isin(sp)].reset_index(drop=True)


def remove_overlap(eval_df: pd.DataFrame, train_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Loại khỏi tập đánh giá những câu đã xuất hiện trong tập huấn luyện."""
    mask = eval_df["text_hash"].isin(set(train_df["text_hash"]))
    return eval_df[~mask].reset_index(drop=True), int(mask.sum())


def overlap_matrix(df: pd.DataFrame) -> pd.DataFrame:
    names = list(C.DATASETS)
    sets = {n: set(df.loc[df["dataset"] == n, "text_hash"]) for n in names}
    return pd.DataFrame([[len(sets[a] & sets[b]) for b in names] for a in names], index=names, columns=names)


def maybe_subsample(df: pd.DataFrame, n: int, seed: int = C.SEED) -> pd.DataFrame:
    """Lấy mẫu con phân tầng (dùng cho chế độ FAST hoặc mô hình chậm như SVM-RBF)."""
    if len(df) <= n:
        return df
    frac = n / len(df)
    idx = []
    for _, g in df.groupby("label"):
        idx.extend(g.sample(max(1, int(round(len(g) * frac))), random_state=seed).index)
    return df.loc[sorted(idx)].reset_index(drop=True)
