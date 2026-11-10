#!/usr/bin/env bash
# Chạy toàn bộ pipeline. Phần 1 (CPU) bắt buộc; Phần 2 (bonus) nên chạy trên GPU.
#   bash run_all.sh            # Phần 1
#   bash run_all.sh --gpu      # Phần 1 + Phần 2
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}

$PY scripts/01_prepare_data.py
$PY scripts/02_eda.py
$PY scripts/03_train_indomain.py
$PY scripts/03b_wild_drift.py
$PY scripts/04_spml_context_regression.py
$PY scripts/05_cross_dataset.py
$PY scripts/06_error_analysis.py

if [[ "${1:-}" == "--gpu" ]]; then
  $PY scripts/07_bonus_embeddings.py --encoder intfloat/multilingual-e5-base --batch 128
  $PY scripts/08_bonus_finetune_transformer.py --model microsoft/deberta-v3-base --epochs 2 --batch 16 --max-len 256
  $PY scripts/08_bonus_finetune_transformer.py --model answerdotai/ModernBERT-base --epochs 2 --batch 32 --max-len 512
  $PY scripts/09_bonus_pretrained_guard.py --llm Qwen/Qwen2.5-1.5B-Instruct --llm-max-eval 500
  $PY scripts/09_bonus_pretrained_guard.py --skip-guard --llm Qwen/Qwen2.5-3B-Instruct --llm-max-eval 500
fi

$PY scripts/10_make_report.py
$PY scripts/make_notebook.py
(cd notebooks && $PY -m jupyter nbconvert --to notebook --execute --inplace prompt_injection_detection.ipynb)
