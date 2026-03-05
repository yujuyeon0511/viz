"""
Path configuration for visualization scripts.
Modify these paths according to your environment.
"""
import os

# ─── Model Paths ─────────────────────────────────────────────
INTERNVL_MODEL_PATH = "/NetDisk/juyeon/DocSP/outputs/stage2_multinode"
QWEN25VL_MODEL_PATH = "/NetDisk/juyeon/models/Qwen2.5-VL-7B-Instruct"

VICUNA_PATH = "/NetDisk/juyeon/models/vicuna-7b-v1.5"
LORA_POOLING = "/NetDisk/juyeon/models/llava-sp-pooling-lora"
LORA_CROPPING = "/NetDisk/juyeon/models/llava-sp-cropping-lora"

# ─── Dataset Paths ───────────────────────────────────────────
CHARTQA_DIR = "/NetDisk/juyeon/train/chartQA/ChartQA Dataset/test"
CHARTQA_IMAGES = os.path.join(CHARTQA_DIR, "png")
CHARTQA_HUMAN = os.path.join(CHARTQA_DIR, "test_human.json")
CHARTQA_AUG = os.path.join(CHARTQA_DIR, "test_augmented.json")

# ─── Conda Environments ─────────────────────────────────────
# InternVL:    /home/juyeon/miniconda3/envs/chartinternvl/bin/python (transformers 4.57.6)
# Qwen2.5-VL:  /home/juyeon/miniconda3/envs/chartinternvl/bin/python (transformers 4.57.6)
# LLaVA-SP:    /home/juyeon/miniconda3/envs/vlm_exp/bin/python (transformers 4.44)
