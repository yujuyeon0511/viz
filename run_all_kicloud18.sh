#!/bin/bash
# Full visualization pipeline on Kicloud18
# Runs all datasets x all models sequentially on GPU 1

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1

PYTHON="/home/juyeon/miniconda3/envs/chartinternvl/bin/python"
FINETUNED="/NetDisk/juyeon/vlm_viz/redundancy_viz/visualize_internvl_finetuned.py"
MULTI="/NetDisk/juyeon/vlm_viz/redundancy_viz/visualize_multi_dataset.py"
V4="/NetDisk/j_son/internvl_35_v4_20260303/finetuning_20260303"
DOCSP="/NetDisk/juyeon/DocSP/outputs/stage2_multinode"
OUTBASE="/NetDisk/juyeon/vlm_viz/redundancy_viz/outputs"

N=20

echo "===== [1/10] v4 FT - ChartQA ====="
$PYTHON $FINETUNED --model_path "$V4" --model_name internvl_v4_ft --n_samples $N --output_dir outputs/chartqa_internvl_v4_ft_20samples

echo "===== [2/10] v4 FT - DocVQA ====="
$PYTHON $MULTI --model_path "$V4" --model_name internvl_v4_ft --dataset docvqa --n_samples $N

echo "===== [3/10] v4 FT - InfoVQA ====="
$PYTHON $MULTI --model_path "$V4" --model_name internvl_v4_ft --dataset infovqa --n_samples $N

echo "===== [4/10] v4 FT - KDTCBench ====="
$PYTHON $MULTI --model_path "$V4" --model_name internvl_v4_ft --dataset kdtcbench --n_samples $N

echo "===== [5/10] v4 FT - KOFFVQA ====="
$PYTHON $MULTI --model_path "$V4" --model_name internvl_v4_ft --dataset koffvqa --n_samples $N

echo "===== [6/10] DocSP - ChartQA ====="
$PYTHON $FINETUNED --model_path "$DOCSP" --model_name docsp_stage2 --n_samples $N --output_dir outputs/chartqa_docsp_stage2_20samples

echo "===== [7/10] DocSP - DocVQA ====="
$PYTHON $MULTI --model_path "$DOCSP" --model_name docsp_stage2 --dataset docvqa --n_samples $N

echo "===== [8/10] DocSP - InfoVQA ====="
$PYTHON $MULTI --model_path "$DOCSP" --model_name docsp_stage2 --dataset infovqa --n_samples $N

echo "===== [9/10] DocSP - KDTCBench ====="
$PYTHON $MULTI --model_path "$DOCSP" --model_name docsp_stage2 --dataset kdtcbench --n_samples $N

echo "===== [10/10] DocSP - KOFFVQA ====="
$PYTHON $MULTI --model_path "$DOCSP" --model_name docsp_stage2 --dataset koffvqa --n_samples $N

echo "===== ALL DONE ====="
date
