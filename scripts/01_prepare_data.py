"""Bước 1 – Tải 7 bộ dữ liệu từ HuggingFace, làm sạch, khử trùng lặp, chia tập.

Chạy:  python scripts/01_prepare_data.py [--force]
Kết quả: data/processed/all.parquet, reports/tables/data_cleaning_stats.csv,
         reports/tables/dataset_overview.csv, reports/tables/overlap_matrix.csv
"""
import argparse

import _common  # noqa: F401
import pandas as pd

from pidetect import config as C
from pidetect import data as D

ap = argparse.ArgumentParser()
ap.add_argument("--force", action="store_true", help="tạo lại dù đã có cache")
args = ap.parse_args()

D.download_all()
df = D.prepare_all(force=True if args.force else False)

ov = (
    df.groupby(["dataset", "split"])
    .agg(n=("label", "size"), n_pos=("label", "sum"), pos_rate=("label", "mean"),
         median_chars=("text", lambda s: int(s.str.len().median())))
    .reset_index()
)
ov["role"] = ov["dataset"].map(lambda d: C.DATASETS[d]["role"])
ov["license"] = ov["dataset"].map(lambda d: C.DATASETS[d]["license"])
ov.to_csv(C.TABLE_DIR / "dataset_overview.csv", index=False)
D.overlap_matrix(df).to_csv(C.TABLE_DIR / "overlap_matrix.csv")
pd.set_option("display.width", 160)
print(ov.to_string(index=False))
print("\nTrùng lặp giữa các bộ dữ liệu (số câu chung):")
print(D.overlap_matrix(df))
