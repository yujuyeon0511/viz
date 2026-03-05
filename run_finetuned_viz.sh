#!/bin/bash
# Run redundancy visualization for two finetuned InternVL models
# Uses chartinternvl conda env (transformers 4.57.6)

set -e

PYTHON="/home/juyeon/miniconda3/envs/chartinternvl/bin/python"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${SCRIPT_DIR}/visualize_internvl_finetuned.py"

N_SAMPLES="${1:-5}"

echo "============================================================"
echo "  [1/2] InternVL v4 Finetuned (internvl_v4_ft)"
echo "============================================================"
$PYTHON "$SCRIPT" \
    --model_path /NetDisk/j_son/internvl_35_v4_20260303/finetuning_20260303 \
    --model_name internvl_v4_ft \
    --n_samples "$N_SAMPLES"

echo ""
echo "============================================================"
echo "  [2/2] DocSP Stage2 Multinode (docsp_stage2)"
echo "============================================================"
$PYTHON "$SCRIPT" \
    --model_path /NetDisk/juyeon/DocSP/outputs/stage2_multinode \
    --model_name docsp_stage2 \
    --n_samples "$N_SAMPLES"

echo ""
echo "============================================================"
echo "  All done!"
echo "  Results:"
echo "    ${SCRIPT_DIR}/outputs/chartqa_internvl_v4_ft_viz/"
echo "    ${SCRIPT_DIR}/outputs/chartqa_docsp_stage2_viz/"
echo "============================================================"
