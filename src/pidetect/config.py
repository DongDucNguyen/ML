"""Cấu hình chung cho toàn bộ dự án.

Mọi đường dẫn đều tính từ thư mục gốc của repo để các script chạy được
từ bất kỳ thư mục nào.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
TABLE_DIR = REPORTS_DIR / "tables"
RESULTS_DIR = REPORTS_DIR / "results"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, FIG_DIR, TABLE_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = int(os.environ.get("PIDETECT_SEED", 42))
N_JOBS = int(os.environ.get("PIDETECT_N_JOBS", -1))

# Chế độ chạy nhanh (dùng khi thử code): lấy mẫu con dữ liệu huấn luyện.
FAST = os.environ.get("PIDETECT_FAST", "0") == "1"

# ---------------------------------------------------------------------------
# Chi phí sai lầm dùng để chọn ngưỡng (slide Logistic Regression: "threshold
# 0.5 là mặc định, không phải tối ưu"). Bỏ lọt một cuộc tấn công (FN) nguy
# hiểm hơn chặn nhầm một câu hỏi bình thường (FP).
# ---------------------------------------------------------------------------
COST_FN = float(os.environ.get("PIDETECT_COST_FN", 5.0))
COST_FP = float(os.environ.get("PIDETECT_COST_FP", 1.0))

# ---------------------------------------------------------------------------
# Danh mục bộ dữ liệu (HuggingFace Hub). Tất cả đều công khai, không cần token.
# ---------------------------------------------------------------------------
DATASETS = {
    # ----- 3 bộ dữ liệu chính (train / val / test) -----
    "safeguard": dict(
        repo="xTRam1/safe-guard-prompt-injection",
        files=["data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"],
        role="main",
        desc="Prompt injection tổng hợp bằng GPT-3.5 theo cây danh mục tấn công (UC Berkeley)",
        license="không ghi (dùng cho mục đích học thuật)",
    ),
    "wild": dict(
        repo="TrustAIRLab/in-the-wild-jailbreak-prompts",
        files=[
            "jailbreak_2023_12_25/train-00000-of-00001.parquet",
            "regular_2023_12_25/train-00000-of-00001.parquet",
        ],
        role="main",
        desc="Jailbreak thực tế thu thập từ Reddit, Discord, website (CCS'24, 12/2022–12/2023)",
        license="MIT",
    ),
    "spml": dict(
        repo="reshabhs/SPML_Chatbot_Prompt_Injection",
        files=["spml_prompt_injection.csv"],
        role="main",
        desc="Injection có ngữ cảnh system prompt + mức độ tấn công Degree 0–10",
        license="MIT",
    ),
    # ----- Bộ kiểm tra ngoài miền (out-of-distribution) -----
    "deepset": dict(
        repo="deepset/prompt-injections",
        files=[
            "data/train-00000-of-00001-9564e8b05b4757ab.parquet",
            "data/test-00000-of-00001-701d16158af87368.parquet",
        ],
        role="ood",
        desc="Prompt injection ngắn, có cả tiếng Đức",
        license="Apache-2.0",
    ),
    "jackhhao": dict(
        repo="jackhhao/jailbreak-classification",
        files=["default/jailbreak_dataset_full.csv"],
        role="ood",
        desc="Jailbreak vs benign (jailbreakchat.com + prompt thông thường)",
        license="Apache-2.0",
    ),
    "toxicchat": dict(
        repo="lmsys/toxic-chat",
        files=["data/0124/toxic-chat_annotation_all.csv"],
        role="ood",
        desc="Câu hỏi thật của người dùng Vicuna demo, nhãn jailbreaking (~2%)",
        license="CC-BY-NC-4.0",
    ),
    "gandalf": dict(
        repo="Lakera/gandalf_ignore_instructions",
        files=["data/train-00000-of-00001-ded53be747ff55cd.parquet"],
        role="ood",
        desc="Tấn công thật từ trò chơi Gandalf (Lakera) – chỉ có nhãn dương",
        license="MIT",
    ),
}

MAIN_DATASETS = [k for k, v in DATASETS.items() if v["role"] == "main"]
OOD_DATASETS = [k for k, v in DATASETS.items() if v["role"] == "ood"]

# Mốc thời gian chia dữ liệu In-the-Wild (chia theo thời gian để mô phỏng drift)
WILD_TRAIN_END = "2023-07-31"  # train: 12/2022 – 07/2023
WILD_VAL_END = "2023-09-30"    # val:   08/2023 – 09/2023; test: 10/2023 – 12/2023

# Giới hạn độ dài văn bản (ký tự) khi trích đặc trưng, tránh prompt cực dài
MAX_CHARS = 6000
