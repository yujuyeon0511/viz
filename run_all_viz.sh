#!/bin/bash
# Multi-dataset visualization pipeline
# Runs sequentially to avoid GPU memory conflicts

PYTHON="/home/juyeon/miniconda3/envs/chartinternvl/bin/python"
SCRIPT="/NetDisk/juyeon/vlm_viz/redundancy_viz/visualize_multi_dataset.py"
V4_PATH="/NetDisk/j_son/internvl_35_v4_20260303/finetuning_20260303"
DOCSP_PATH="/NetDisk/juyeon/DocSP/outputs/stage2_multinode"

export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Phase 1: InternVL v4 FT"
echo "=========================================="

# v4 FT - DocVQA
echo "[1/6] v4 FT - DocVQA (20 samples)"
$PYTHON $SCRIPT \
    --model_path "$V4_PATH" \
    --model_name internvl_v4_ft \
    --dataset docvqa \
    --n_samples 20

# v4 FT - KDTCBench
echo "[2/6] v4 FT - KDTCBench (20 samples)"
$PYTHON $SCRIPT \
    --model_path "$V4_PATH" \
    --model_name internvl_v4_ft \
    --dataset kdtcbench \
    --n_samples 20

# v4 FT - KOFFVQA
echo "[3/6] v4 FT - KOFFVQA (20 samples)"
$PYTHON $SCRIPT \
    --model_path "$V4_PATH" \
    --model_name internvl_v4_ft \
    --dataset koffvqa \
    --n_samples 20

echo "=========================================="
echo "Phase 2: DocSP Stage2"
echo "=========================================="

# DocSP - DocVQA
echo "[4/6] DocSP - DocVQA (20 samples)"
$PYTHON $SCRIPT \
    --model_path "$DOCSP_PATH" \
    --model_name docsp_stage2 \
    --dataset docvqa \
    --n_samples 20

# DocSP - KDTCBench
echo "[5/6] DocSP - KDTCBench (20 samples)"
$PYTHON $SCRIPT \
    --model_path "$DOCSP_PATH" \
    --model_name docsp_stage2 \
    --dataset kdtcbench \
    --n_samples 20

# DocSP - KOFFVQA
echo "[6/6] DocSP - KOFFVQA (20 samples)"
$PYTHON $SCRIPT \
    --model_path "$DOCSP_PATH" \
    --model_name docsp_stage2 \
    --dataset koffvqa \
    --n_samples 20

echo "=========================================="
echo "All done!"
echo "=========================================="
