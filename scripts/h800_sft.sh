#!/bin/bash
# Run inside the H800 Pod; output and resume directories must be supplied by the caller.
set -euo pipefail
root=/ainf-cfs/yangtianze
runtime="$root/alvarez-acceleration-20261118"
export PYTHONPATH="$root/alvarez-runtime/deps:$runtime/src"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$root/alvarez-runtime/cache/huggingface"
export TORCH_HOME="$root/alvarez-runtime/cache/torch"
export TRITON_CACHE_DIR="$root/alvarez-runtime/cache/triton"
export TMPDIR="$runtime/tmp"
mkdir -p "$TMPDIR"
exec /opt/nemo_rl_venv/bin/python -m alvarezgui.minicpm_training \
    --algorithm sft --finetuning-method lora --train-merger-mlp --multiview \
    --base-model "$root/alvarez-models/MiniCPM-V-4.7-1B" \
    --dataset "$root/alvarez-data/sokoban-sft-v2/action-v1/train.jsonl" \
    --learning-rate 1e-5 --seed 20261117 --epochs 10 \
    --batch-size 4 --gradient-accumulation 1 \
    --cache-preprocessed --no-gradient-checkpointing "$@"
