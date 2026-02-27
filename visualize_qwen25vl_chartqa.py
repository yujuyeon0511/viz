#!/usr/bin/env python3
"""
Qwen2.5-VL-7B-Instruct × ChartQA Visualization
Based on "From Redundancy to Relevance" (NAACL 2025)

Visualizations:
1. Layer-wise ViT Attention Heatmap
2. Information Flow Analysis (token similarity across LLM layers)
3. Token Redundancy Analysis (image token cosine similarity matrix)
4. Comprehensive Dashboard per sample

Usage:
    python visualize_qwen25vl_chartqa.py --n_samples 3
    python visualize_qwen25vl_chartqa.py --n_samples 5 --skip_llm
"""

import argparse
import json
import os
import sys
import gc
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import (
    QWEN25VL_MODEL_PATH,
    CHARTQA_DIR, CHARTQA_IMAGES, CHARTQA_HUMAN, CHARTQA_AUG,
)

QWEN_MODEL_PATH = QWEN25VL_MODEL_PATH

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "custom_heat", ["#000033", "#0000ff", "#00ffff", "#ffff00", "#ff0000"], N=256
)


def load_chartqa_samples(n_samples=5, split="human"):
    json_path = CHARTQA_HUMAN if split == "human" else CHARTQA_AUG
    with open(json_path, "r") as f:
        data = json.load(f)

    samples = []
    seen_images = set()
    for item in data:
        imgname = item["imgname"]
        if imgname in seen_images:
            continue
        img_path = os.path.join(CHARTQA_IMAGES, imgname)
        if os.path.exists(img_path):
            samples.append({
                "image_path": img_path,
                "question": item["query"],
                "answer": item["label"],
                "imgname": imgname,
            })
            seen_images.add(imgname)
        if len(samples) >= n_samples:
            break
    return samples


def load_model():
    """Load Qwen2.5-VL-7B-Instruct."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print("Loading Qwen2.5-VL-7B-Instruct ...")
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_PATH)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="eager",  # Need eager for attention capture
    ).eval()

    print(f"  Model loaded.")
    print(f"  Vision: {len(model.visual.blocks)} blocks, spatial_merge={model.visual.spatial_merge_size}")
    print(f"  LLM: {len(model.model.language_model.layers)} layers, hidden={model.config.hidden_size}")
    return model, processor


def prepare_input(processor, image_path, question):
    """Prepare model inputs."""
    image = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    return inputs, image


# ═══════════════════════════════════════════════════════════════
# 1. ViT Attention Capture
# ═══════════════════════════════════════════════════════════════

class QwenViTAttentionCapture:
    """Capture attention from Qwen2.5-VL ViT blocks.

    Qwen2.5-VL ViT attention input is [seq_len, dim] (2D, no batch).
    We use forward_pre_hook to compute simplified attention (without RoPE)
    from Q*K^T before the actual forward runs.
    """

    def __init__(self, model):
        self.model = model
        self.attention_maps = {}
        self._hooks = []

    def __enter__(self):
        for idx, block in enumerate(self.model.visual.blocks):
            attn_module = block.attn

            def make_hook(layer_idx, storage):
                def hook_fn(module, args):
                    hidden_states = args[0]  # [N, C] (2D)
                    if hidden_states.dim() == 2:
                        N, C = hidden_states.shape
                    else:
                        # Fallback for unexpected shapes
                        return

                    num_heads = module.num_heads  # 16
                    head_dim = C // num_heads      # 80

                    qkv = module.qkv(hidden_states)  # [N, 3*C]
                    qkv = qkv.reshape(N, 3, num_heads, head_dim)
                    qkv = qkv.permute(1, 2, 0, 3)  # [3, H, N, D]
                    q, k, _ = qkv.unbind(0)          # each [H, N, D]

                    scale = head_dim ** -0.5
                    attn = (q * scale) @ k.transpose(-2, -1)  # [H, N, N]
                    attn = attn.softmax(dim=-1)
                    # Store as [1, H, N, N]
                    storage[layer_idx] = attn.unsqueeze(0).detach().cpu().float()
                return hook_fn

            h = attn_module.register_forward_pre_hook(make_hook(idx, self.attention_maps))
            self._hooks.append(h)

        return self

    def __exit__(self, *args):
        for h in self._hooks:
            h.remove()


# ═══════════════════════════════════════════════════════════════
# 2. Visualization Functions
# ═══════════════════════════════════════════════════════════════

def plot_vit_attention_heatmaps(attention_maps, original_image, output_path,
                                 grid_h, grid_w, num_layers_to_show=6):
    """
    Plot ViT attention heatmaps.
    Qwen2.5-VL ViT has no CLS token, so we use mean attention received per patch.
    """
    n_layers = len(attention_maps)
    if n_layers == 0:
        print("  Warning: No attention maps captured, skipping.")
        return

    # Use fullatt blocks preferentially + first/last
    fullatt = [7, 15, 23, 31]
    available = sorted(attention_maps.keys())
    preferred = [i for i in fullatt if i in attention_maps]
    if len(preferred) < num_layers_to_show:
        others = [i for i in available if i not in preferred]
        step = max(1, len(others) // (num_layers_to_show - len(preferred)))
        preferred = sorted(preferred + others[::step])[:num_layers_to_show]
    layer_indices = preferred[:num_layers_to_show]

    img_size = max(original_image.size)
    fig, axes = plt.subplots(2, len(layer_indices), figsize=(4 * len(layer_indices), 8))
    if len(layer_indices) == 1:
        axes = axes.reshape(2, 1)
    img_np = np.array(original_image.resize((img_size, img_size)))

    for col, layer_idx in enumerate(layer_indices):
        attn = attention_maps[layer_idx]  # [B, H, N, N]
        # Mean attention received per patch (column-wise mean, averaged over heads)
        mean_attn = attn[0].mean(dim=0).mean(dim=0)  # [N] - mean attention received
        n_patches = mean_attn.shape[0]

        # Try to reshape to 2D grid
        if n_patches == grid_h * grid_w:
            attn_2d = mean_attn.reshape(grid_h, grid_w).numpy()
        else:
            # Windowed attention - use what we have
            side = int(np.ceil(np.sqrt(n_patches)))
            padded = np.zeros(side * side)
            padded[:n_patches] = mean_attn.numpy()
            attn_2d = padded.reshape(side, side)

        attn_2d = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-8)
        attn_resized = np.array(Image.fromarray(attn_2d).resize(
            (img_size, img_size), Image.BILINEAR))

        is_full = layer_idx in [7, 15, 23, 31]
        label = f"Layer {layer_idx}" + (" (full)" if is_full else " (win)")

        axes[0, col].imshow(attn_2d, cmap=HEATMAP_CMAP, interpolation='bilinear')
        axes[0, col].set_title(label, fontsize=12, fontweight='bold')
        axes[0, col].axis('off')

        axes[1, col].imshow(img_np)
        axes[1, col].imshow(attn_resized, cmap=HEATMAP_CMAP, alpha=0.5, interpolation='bilinear')
        axes[1, col].axis('off')

    axes[0, 0].set_ylabel("Attention\n(Mean Recv.)", fontsize=11, rotation=0, labelpad=80, va='center')
    axes[1, 0].set_ylabel("Overlay", fontsize=11, rotation=0, labelpad=80, va='center')

    fig.suptitle("Qwen2.5-VL ViT Attention Heatmaps (Mean Attention Received)",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_information_flow(hidden_states, img_positions, output_path):
    """Plot information flow analysis across LLM layers."""
    n_layers = len(hidden_states)
    img_idx = np.where(img_positions)[0]

    if len(img_idx) == 0:
        print("  Warning: No image tokens found, skipping information flow plot.")
        return [], []

    img_hidden = [hs[0, img_idx, :] for hs in hidden_states]

    cosine_sims = []
    l2_norms = []
    variances = []
    mean_token_sims = []

    for i in range(n_layers):
        h = img_hidden[i]
        l2_norms.append(h.norm(dim=-1).mean().item())
        variances.append(h.var(dim=0).mean().item())

        h_norm = F.normalize(h, dim=-1)
        sim_matrix = h_norm @ h_norm.T
        mask = ~torch.eye(sim_matrix.shape[0], dtype=torch.bool)
        if mask.any():
            mean_token_sims.append(sim_matrix[mask].mean().item())
        else:
            mean_token_sims.append(1.0)

        if i > 0:
            cos = F.cosine_similarity(h, img_hidden[i - 1], dim=-1).mean().item()
            cosine_sims.append(cos)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    layers = list(range(n_layers))

    ax = axes[0, 0]
    ax.plot(range(1, n_layers), cosine_sims, 'b-o', markersize=3, linewidth=1.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Inter-layer Similarity (Image Tokens)", fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    if len(cosine_sims) > 1:
        diffs = np.diff(cosine_sims)
        cliff_idx = np.argmin(diffs) + 1
        ax.axvline(x=cliff_idx + 1, color='red', linestyle='--', alpha=0.7,
                   label=f'Cliff @ Layer {cliff_idx + 1}')
        ax.legend()

    ax = axes[0, 1]
    ax.plot(layers, l2_norms, 'g-o', markersize=3, linewidth=1.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean L2 Norm")
    ax.set_title("Image Token L2 Norm per Layer", fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(layers, variances, 'r-o', markersize=3, linewidth=1.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Variance")
    ax.set_title("Image Token Variance (Info Convergence)", fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(layers, mean_token_sims, 'm-o', markersize=3, linewidth=1.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Pairwise Cosine Sim")
    ax.set_title("Token Redundancy (Higher = More Redundant)", fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.suptitle("Information Flow Analysis — Qwen2.5-VL-7B (28 LLM Layers)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")

    return cosine_sims, mean_token_sims


def plot_token_redundancy_matrices(hidden_states, img_positions, output_path, layers_to_show=None):
    """Plot image token cosine similarity matrices at selected LLM layers."""
    n_layers = len(hidden_states)
    img_idx = np.where(img_positions)[0]
    if len(img_idx) == 0:
        return

    if layers_to_show is None:
        layers_to_show = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]

    max_tokens = 64
    if len(img_idx) > max_tokens:
        step = len(img_idx) // max_tokens
        img_idx_sub = img_idx[::step][:max_tokens]
    else:
        img_idx_sub = img_idx

    n_show = len(layers_to_show)
    fig, axes = plt.subplots(1, n_show, figsize=(4.5 * n_show, 4))
    if n_show == 1:
        axes = [axes]

    for col, layer_idx in enumerate(layers_to_show):
        if layer_idx >= n_layers:
            continue
        h = hidden_states[layer_idx][0, img_idx_sub, :]
        h_norm = F.normalize(h, dim=-1)
        sim_matrix = (h_norm @ h_norm.T).numpy()

        im = axes[col].imshow(sim_matrix, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='equal')
        axes[col].set_title(f"Layer {layer_idx}", fontsize=12, fontweight='bold')
        axes[col].set_xlabel("Image Token")
        if col == 0:
            axes[col].set_ylabel("Image Token")

    plt.colorbar(im, ax=axes, shrink=0.8, label="Cosine Similarity")
    fig.suptitle("Image Token Redundancy (Cosine Similarity Matrices) — Qwen2.5-VL",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_comprehensive_dashboard(
    original_image, question, answer, model_answer,
    attention_maps, hidden_states, img_positions,
    output_path, grid_h=0, grid_w=0,
):
    """Comprehensive dashboard for one sample."""
    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    img_size = max(original_image.size)
    img_np = np.array(original_image.resize((img_size, img_size)))
    img_idx = np.where(img_positions)[0]
    n_vit = len(attention_maps)
    n_layers = len(hidden_states)

    # Row 0, Col 0: Image + Q&A
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img_np)
    ax.set_title("Chart Image", fontsize=11, fontweight='bold')
    ax.axis('off')
    qa_text = (f"Q: {question[:80]}{'...' if len(question) > 80 else ''}\n"
               f"GT: {answer}\nPred: {model_answer[:80]}")
    ax.text(0.5, -0.15, qa_text, transform=ax.transAxes, fontsize=8,
            ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8), wrap=True)

    # Row 0, Col 1-3: ViT attention (full-attention blocks: 7, 15, 31)
    vit_show = [b for b in [7, 15, 31] if b in attention_maps]
    if not vit_show and attention_maps:
        keys = sorted(attention_maps.keys())
        vit_show = [keys[0], keys[len(keys)//2], keys[-1]]

    for col, layer_idx in enumerate(vit_show[:3]):
        ax = fig.add_subplot(gs[0, col + 1])
        attn = attention_maps[layer_idx]
        mean_attn = attn[0].mean(dim=0).mean(dim=0)
        n_patches = mean_attn.shape[0]
        if n_patches == grid_h * grid_w and grid_h > 0:
            attn_2d = mean_attn.reshape(grid_h, grid_w).numpy()
        else:
            side = int(np.ceil(np.sqrt(n_patches)))
            padded = np.zeros(side * side)
            padded[:n_patches] = mean_attn.numpy()
            attn_2d = padded.reshape(side, side)
        attn_2d = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-8)
        attn_resized = np.array(Image.fromarray(attn_2d).resize(
            (img_size, img_size), Image.BILINEAR))
        ax.imshow(img_np)
        ax.imshow(attn_resized, cmap=HEATMAP_CMAP, alpha=0.5, interpolation='bilinear')
        is_full = layer_idx in [7, 15, 23, 31]
        ax.set_title(f"ViT L{layer_idx} ({'full' if is_full else 'win'})",
                     fontsize=11, fontweight='bold')
        ax.axis('off')

    # Row 1: Information Flow
    if len(img_idx) > 0:
        img_hidden = [hs[0, img_idx, :] for hs in hidden_states]
        cosine_sims = []
        mean_token_sims = []
        for i in range(n_layers):
            h = img_hidden[i]
            h_norm = F.normalize(h, dim=-1)
            sim_matrix = h_norm @ h_norm.T
            mask = ~torch.eye(sim_matrix.shape[0], dtype=torch.bool)
            mean_token_sims.append(sim_matrix[mask].mean().item() if mask.any() else 1.0)
            if i > 0:
                cos = F.cosine_similarity(h, img_hidden[i - 1], dim=-1).mean().item()
                cosine_sims.append(cos)

        ax = fig.add_subplot(gs[1, 0:2])
        ax.plot(range(1, n_layers), cosine_sims, 'b-o', markersize=2, linewidth=1.5,
                label='Inter-layer sim')
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Information Flow (Inter-layer)", fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = fig.add_subplot(gs[1, 2:4])
        ax.plot(range(n_layers), mean_token_sims, 'm-o', markersize=2, linewidth=1.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean Pairwise Cosine Sim")
        ax.set_title("Token Redundancy per Layer", fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    # Row 2: Redundancy Matrices
    if len(img_idx) > 0:
        max_tokens = 64
        if len(img_idx) > max_tokens:
            step = len(img_idx) // max_tokens
            img_idx_sub = img_idx[::step][:max_tokens]
        else:
            img_idx_sub = img_idx

        for col, layer_idx in enumerate([0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]):
            ax = fig.add_subplot(gs[2, col])
            h = hidden_states[layer_idx][0, img_idx_sub, :]
            h_norm = F.normalize(h, dim=-1)
            sim_matrix = (h_norm @ h_norm.T).numpy()
            ax.imshow(sim_matrix, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='equal')
            ax.set_title(f"Redundancy @ Layer {layer_idx}", fontsize=10, fontweight='bold')

    fig.suptitle("Qwen2.5-VL-7B × ChartQA — Information Flow & Redundancy Analysis",
                 fontsize=16, fontweight='bold', y=1.01)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_aggregate_analysis(all_results, output_path):
    """Aggregate analysis across all samples."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    all_cosine = [r['cosine_sims'] for r in all_results if r.get('cosine_sims')]
    all_redundancy = [r['redundancy'] for r in all_results if r.get('redundancy')]

    if all_cosine:
        min_len = min(len(c) for c in all_cosine)
        cosine_array = np.array([c[:min_len] for c in all_cosine])
        mean_cosine = cosine_array.mean(axis=0)
        std_cosine = cosine_array.std(axis=0)

        ax = axes[0]
        layers = np.arange(1, min_len + 1)
        ax.plot(layers, mean_cosine, 'b-', linewidth=2, label='Mean')
        ax.fill_between(layers, mean_cosine - std_cosine, mean_cosine + std_cosine,
                        alpha=0.2, color='blue')
        for c in all_cosine:
            ax.plot(range(1, len(c[:min_len]) + 1), c[:min_len], alpha=0.15, color='gray')
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Inter-layer Similarity (All Samples)", fontweight='bold')
        ax.grid(True, alpha=0.3)
        diffs = np.diff(mean_cosine)
        cliff_idx = np.argmin(diffs) + 1
        ax.axvline(x=cliff_idx + 1, color='red', linestyle='--', alpha=0.7,
                   label=f'Mean Cliff @ Layer {cliff_idx + 1}')
        ax.legend()

    if all_redundancy:
        min_len = min(len(r) for r in all_redundancy)
        red_array = np.array([r[:min_len] for r in all_redundancy])
        mean_red = red_array.mean(axis=0)
        std_red = red_array.std(axis=0)

        ax = axes[1]
        layers = np.arange(min_len)
        ax.plot(layers, mean_red, 'm-', linewidth=2, label='Mean')
        ax.fill_between(layers, mean_red - std_red, mean_red + std_red, alpha=0.2, color='magenta')
        for r in all_redundancy:
            ax.plot(range(len(r[:min_len])), r[:min_len], alpha=0.15, color='gray')
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean Pairwise Cosine Sim")
        ax.set_title("Token Redundancy (All Samples)", fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.legend()

    if all_cosine:
        cliff_layers = []
        for c in all_cosine:
            if len(c) > 1:
                cliff_layers.append(np.argmin(np.diff(c)) + 2)
        ax = axes[2]
        if cliff_layers:
            ax.hist(cliff_layers, bins=range(0, max(cliff_layers) + 2),
                    color='coral', edgecolor='black', alpha=0.8)
            ax.axvline(np.mean(cliff_layers), color='red', linestyle='--', linewidth=2,
                       label=f'Mean={np.mean(cliff_layers):.1f}')
            ax.set_xlabel("Layer")
            ax.set_ylabel("Count")
            ax.set_title("Information Flow Cliff Layer Distribution", fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

    fig.suptitle("Aggregate Analysis: Qwen2.5-VL-7B on ChartQA", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL ChartQA Visualization")
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--split", type=str, default="human", choices=["human", "augmented"])
    parser.add_argument("--output_dir", type=str, default="outputs/chartqa_qwen25vl_viz")
    parser.add_argument("--skip_llm", action="store_true")
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    samples = load_chartqa_samples(args.n_samples, args.split)
    print(f"Loaded {len(samples)} ChartQA samples ({args.split} split)")

    model, processor = load_model()

    all_results = []

    for idx, sample in enumerate(samples):
        print(f"\n{'='*60}")
        print(f"Sample {idx+1}/{len(samples)}: {sample['imgname']}")
        print(f"  Q: {sample['question']}")
        print(f"  A: {sample['answer']}")
        print(f"{'='*60}")

        inputs, original_image = prepare_input(processor, sample["image_path"], sample["question"])
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        input_ids = inputs["input_ids"]

        # Compute grid shape for ViT attention visualization
        image_grid_thw = inputs.get("image_grid_thw", None)
        if image_grid_thw is not None:
            t, grid_h, grid_w = image_grid_thw[0].tolist()
            grid_h = int(grid_h)
            grid_w = int(grid_w)
        else:
            grid_h = grid_w = 0

        # 1. ViT Attention
        print("  [1/4] Extracting ViT attention maps...")
        with QwenViTAttentionCapture(model) as cap:
            with torch.no_grad():
                # Run vision encoder only
                pixel_values = inputs["pixel_values"]
                grid_thw = inputs["image_grid_thw"]
                _ = model.visual(pixel_values, grid_thw=grid_thw)
            attention_maps = cap.attention_maps

        if attention_maps:
            plot_vit_attention_heatmaps(
                attention_maps, original_image,
                os.path.join(output_dir, f"sample_{idx:03d}_vit_attention.png"),
                grid_h=grid_h, grid_w=grid_w,
            )
        print(f"    ViT attention captured: {len(attention_maps)} layers")

        # 2. Generate answer
        print("  [2/4] Getting model answer...")
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                )
            # Decode only generated tokens
            gen_ids = output_ids[0, input_ids.shape[1]:]
            model_answer = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        except Exception as e:
            import traceback
            print(f"  Warning: Generation failed ({e})")
            traceback.print_exc()
            model_answer = "[generation failed]"
        print(f"  Model answer: {model_answer}")

        # 3. LLM Hidden States
        result = {}
        if not args.skip_llm:
            print("  [3/4] Extracting LLM hidden states...")
            try:
                with torch.no_grad():
                    outputs = model(
                        **inputs,
                        output_hidden_states=True,
                        output_attentions=False,
                        return_dict=True,
                    )

                hidden_states = [h.detach().cpu().float() for h in outputs.hidden_states]
                seq_len = hidden_states[0].shape[1]

                # Find image token positions
                # Qwen2.5-VL uses image_token_id (151655), not vision_token_id
                image_token_id = model.config.image_token_id  # 151655
                img_positions = (input_ids[0].cpu() == image_token_id).numpy()
                n_img_tokens = img_positions.sum()

                print(f"    Seq len: {seq_len}, Image tokens: {n_img_tokens}, "
                      f"LLM layers: {len(hidden_states)}")

                del outputs
                torch.cuda.empty_cache()

                # Compute metrics
                img_idx = np.where(img_positions)[0]
                if len(img_idx) > 0:
                    img_hidden = [hs[0, img_idx, :] for hs in hidden_states]
                    cosine_sims = []
                    redundancy = []
                    for i in range(len(hidden_states)):
                        h = img_hidden[i]
                        h_norm = F.normalize(h, dim=-1)
                        sim_matrix = h_norm @ h_norm.T
                        mask = ~torch.eye(sim_matrix.shape[0], dtype=torch.bool)
                        redundancy.append(sim_matrix[mask].mean().item() if mask.any() else 1.0)
                        if i > 0:
                            cos = F.cosine_similarity(h, img_hidden[i - 1], dim=-1).mean().item()
                            cosine_sims.append(cos)
                    result = {'cosine_sims': cosine_sims, 'redundancy': redundancy}

                plot_information_flow(
                    hidden_states, img_positions,
                    os.path.join(output_dir, f"sample_{idx:03d}_info_flow.png")
                )
                plot_token_redundancy_matrices(
                    hidden_states, img_positions,
                    os.path.join(output_dir, f"sample_{idx:03d}_redundancy.png")
                )
                plot_comprehensive_dashboard(
                    original_image, sample["question"], sample["answer"], model_answer,
                    attention_maps, hidden_states, img_positions,
                    os.path.join(output_dir, f"sample_{idx:03d}_dashboard.png"),
                    grid_h=grid_h, grid_w=grid_w,
                )

                del hidden_states
                torch.cuda.empty_cache()

            except Exception as e:
                import traceback
                print(f"  Warning: LLM analysis failed: {e}")
                traceback.print_exc()
        else:
            print("  [3/4] Skipping LLM hidden states (--skip_llm)")

        print("  [4/4] Done with this sample.")
        all_results.append(result)
        gc.collect()
        torch.cuda.empty_cache()

    if any(r for r in all_results):
        print(f"\n{'='*60}")
        print("Generating aggregate analysis...")
        plot_aggregate_analysis(all_results, os.path.join(output_dir, "aggregate_analysis.png"))

    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
