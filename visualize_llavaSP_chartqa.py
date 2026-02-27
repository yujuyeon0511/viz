#!/usr/bin/env python3
"""
LLaVA-SP × ChartQA Visualization
Based on "From Redundancy to Relevance" (NAACL 2025)

Applies information flow & token redundancy analysis to LLaVA-SP (pooling variant)
on ChartQA test data, for comparison with InternVL3.5-8B results.

Usage:
    python visualize_llavaSP_chartqa.py --n_samples 5
    python visualize_llavaSP_chartqa.py --variant cropping --n_samples 3
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
    VICUNA_PATH, LORA_POOLING, LORA_CROPPING,
    CHARTQA_DIR, CHARTQA_IMAGES, CHARTQA_HUMAN, CHARTQA_AUG,
)

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "custom_heat", ["#000033", "#0000ff", "#00ffff", "#ffff00", "#ff0000"], N=256
)

# ─── Add local llava to path ────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


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


def load_llava_sp_model(variant="pooling"):
    """Load LLaVA-SP model (pooling or cropping variant)."""
    from llava.model.builder import load_pretrained_model

    lora_path = LORA_POOLING if variant == "pooling" else LORA_CROPPING
    model_name = f"llava-sp-{variant}-lora"

    print(f"Loading LLaVA-SP ({variant}) ...")
    print(f"  Base: {VICUNA_PATH}")
    print(f"  LoRA: {lora_path}")

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=lora_path,
        model_base=VICUNA_PATH,
        model_name=model_name,
        device_map="cuda:0",
        device="cuda:0",
    )
    model.eval()

    print(f"  Model loaded successfully.")
    print(f"  Vision tower: {model.config.mm_vision_tower}")
    print(f"  Projector: {model.config.mm_projector_type}")
    return tokenizer, model, image_processor, context_len


def prepare_input(model, tokenizer, image_processor, image_path, question):
    """Prepare model inputs for a single image + question."""
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.mm_utils import tokenizer_image_token, process_images

    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    image_sizes = [image.size]

    conv = conv_templates["v1"].copy()
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)

    return input_ids, image_tensor, image_sizes, image


# ═══════════════════════════════════════════════════════════════
# 1. ViT Attention Capture
# ═══════════════════════════════════════════════════════════════

class CLIPAttentionCapture:
    """Capture attention maps from CLIP ViT encoder."""

    def __init__(self, vision_tower):
        self.vision_tower = vision_tower
        self.attention_maps = {}
        self._hooks = []

    def __enter__(self):
        encoder = self.vision_tower.vision_tower.vision_model.encoder

        for idx, layer in enumerate(encoder.layers):
            attn_module = layer.self_attn

            def make_hook(layer_idx):
                def hook_fn(module, args, output):
                    if isinstance(output, tuple) and len(output) >= 2:
                        attn_weights = output[1]
                        if attn_weights is not None:
                            self.attention_maps[layer_idx] = attn_weights.detach().cpu().float()
                return hook_fn

            h = attn_module.register_forward_hook(make_hook(idx))
            self._hooks.append(h)

        return self

    def __exit__(self, *args):
        for h in self._hooks:
            h.remove()


# ═══════════════════════════════════════════════════════════════
# 2. Visualization Functions
# ═══════════════════════════════════════════════════════════════

def plot_vit_attention_heatmaps(attention_maps, original_image, output_path, num_layers_to_show=6):
    """Plot CLIP ViT attention heatmaps."""
    n_layers = len(attention_maps)
    if n_layers == 0:
        print("  Warning: No attention maps captured, skipping ViT attention plot.")
        return
    layer_indices = np.linspace(0, n_layers - 1, num_layers_to_show, dtype=int)

    fig, axes = plt.subplots(2, num_layers_to_show, figsize=(4 * num_layers_to_show, 8))
    img_np = np.array(original_image.resize((336, 336)))

    for col, layer_idx in enumerate(layer_indices):
        if layer_idx not in attention_maps:
            continue
        attn = attention_maps[layer_idx]
        cls_attn = attn[0, :, 0, 1:].mean(dim=0)
        h = w = int(cls_attn.shape[0] ** 0.5)
        cls_attn_2d = cls_attn.reshape(h, w).numpy()
        cls_attn_2d = (cls_attn_2d - cls_attn_2d.min()) / (cls_attn_2d.max() - cls_attn_2d.min() + 1e-8)
        attn_resized = np.array(Image.fromarray(cls_attn_2d).resize((336, 336), Image.BILINEAR))

        axes[0, col].imshow(cls_attn_2d, cmap=HEATMAP_CMAP, interpolation='bilinear')
        axes[0, col].set_title(f"Layer {layer_idx}", fontsize=12, fontweight='bold')
        axes[0, col].axis('off')

        axes[1, col].imshow(img_np)
        axes[1, col].imshow(attn_resized, cmap=HEATMAP_CMAP, alpha=0.5, interpolation='bilinear')
        axes[1, col].axis('off')

    axes[0, 0].set_ylabel("Attention\n(CLS→Patch)", fontsize=11, rotation=0, labelpad=80, va='center')
    axes[1, 0].set_ylabel("Overlay", fontsize=11, rotation=0, labelpad=80, va='center')

    fig.suptitle("CLIP ViT Layer-wise Attention Heatmaps (CLS → Patches)", fontsize=14, fontweight='bold', y=1.02)
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
        return {}, {}

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

    fig.suptitle("Information Flow Analysis — LLaVA-SP (Vicuna-7B, 32 Layers)", fontsize=14, fontweight='bold')
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
    fig.suptitle("Image Token Redundancy (Cosine Similarity Matrices) — LLaVA-SP",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_comprehensive_dashboard(
    original_image, question, answer, model_answer,
    attention_maps, hidden_states, img_positions,
    output_path, variant="pooling"
):
    """Comprehensive dashboard for one sample."""
    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    img_np = np.array(original_image.resize((336, 336)))
    img_idx = np.where(img_positions)[0]
    n_vit = len(attention_maps)
    n_layers = len(hidden_states)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img_np)
    ax.set_title("Chart Image", fontsize=11, fontweight='bold')
    ax.axis('off')
    qa_text = f"Q: {question[:80]}{'...' if len(question) > 80 else ''}\nGT: {answer}\nPred: {model_answer[:80]}"
    ax.text(0.5, -0.15, qa_text, transform=ax.transAxes, fontsize=8,
            ha='center', va='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            wrap=True)

    if n_vit > 0:
        for col, layer_idx in enumerate([0, n_vit // 2, n_vit - 1]):
            ax = fig.add_subplot(gs[0, col + 1])
            if layer_idx in attention_maps:
                attn = attention_maps[layer_idx]
                cls_attn = attn[0, :, 0, 1:].mean(dim=0)
                h = w = int(cls_attn.shape[0] ** 0.5)
                cls_attn_2d = cls_attn.reshape(h, w).numpy()
                cls_attn_2d = (cls_attn_2d - cls_attn_2d.min()) / (cls_attn_2d.max() - cls_attn_2d.min() + 1e-8)
                attn_resized = np.array(Image.fromarray(cls_attn_2d).resize((336, 336), Image.BILINEAR))
                ax.imshow(img_np)
                ax.imshow(attn_resized, cmap=HEATMAP_CMAP, alpha=0.5, interpolation='bilinear')
            ax.set_title(f"ViT Layer {layer_idx}", fontsize=11, fontweight='bold')
            ax.axis('off')

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
        ax.plot(range(1, n_layers), cosine_sims, 'b-o', markersize=2, linewidth=1.5, label='Inter-layer sim')
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

    fig.suptitle(f"LLaVA-SP ({variant}) x ChartQA — Information Flow & Redundancy Analysis",
                 fontsize=16, fontweight='bold', y=1.01)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_aggregate_analysis(all_results, output_path, variant="pooling"):
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
        ax.fill_between(layers, mean_cosine - std_cosine, mean_cosine + std_cosine, alpha=0.2, color='blue')
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

    fig.suptitle(f"Aggregate Analysis: LLaVA-SP ({variant}) on ChartQA",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LLaVA-SP ChartQA Visualization")
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--split", type=str, default="human", choices=["human", "augmented"])
    parser.add_argument("--variant", type=str, default="pooling", choices=["pooling", "cropping"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_llm", action="store_true")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"outputs/chartqa_llavaSP_{args.variant}_viz"

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    samples = load_chartqa_samples(args.n_samples, args.split)
    print(f"Loaded {len(samples)} ChartQA samples ({args.split} split)")

    tokenizer, model, image_processor, context_len = load_llava_sp_model(args.variant)

    all_results = []

    for idx, sample in enumerate(samples):
        print(f"\n{'='*60}")
        print(f"Sample {idx+1}/{len(samples)}: {sample['imgname']}")
        print(f"  Q: {sample['question']}")
        print(f"  A: {sample['answer']}")
        print(f"{'='*60}")

        input_ids, image_tensor, image_sizes, original_image = prepare_input(
            model, tokenizer, image_processor, sample["image_path"], sample["question"]
        )

        # 1. ViT Attention
        print("  [1/4] Extracting ViT attention maps...")
        vision_tower = model.get_vision_tower()
        with CLIPAttentionCapture(vision_tower) as cap:
            with torch.no_grad():
                from llava.mm_utils import process_images
                img_tensor = process_images([original_image], image_processor, model.config)
                img_tensor = img_tensor.to(model.device, dtype=torch.float16)
                vision_tower.vision_tower.vision_model(
                    pixel_values=img_tensor,
                    output_attentions=True,
                    return_dict=True,
                )
            attention_maps = cap.attention_maps

        if attention_maps:
            plot_vit_attention_heatmaps(
                attention_maps, original_image,
                os.path.join(output_dir, f"sample_{idx:03d}_vit_attention.png")
            )

        # 2. Generate answer
        print("  [2/4] Getting model answer...")
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    inputs=input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    max_new_tokens=128,
                    use_cache=True,
                )
            model_answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
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
                from llava.constants import IMAGE_TOKEN_INDEX

                with torch.no_grad():
                    prepared = model.prepare_inputs_labels_for_multimodal(
                        input_ids, None, None, None, None, images=image_tensor, image_sizes=image_sizes
                    )
                    _, position_ids, attention_mask, past_kv, inputs_embeds, _ = prepared

                    outputs = model.model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        output_hidden_states=True,
                        return_dict=True,
                    )

                hidden_states = [h.detach().cpu().float() for h in outputs.hidden_states]

                seq_len = hidden_states[0].shape[1]
                text_len = input_ids.shape[1] - 1
                n_inserted = seq_len - text_len

                img_token_pos = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
                img_positions = np.zeros(seq_len, dtype=bool)
                if len(img_token_pos) > 0:
                    start = img_token_pos[0].item()
                    img_positions[start:start + n_inserted] = True

                print(f"    Seq len: {seq_len}, Image tokens: {n_inserted}, LLM layers: {len(hidden_states)}")

                del outputs
                torch.cuda.empty_cache()

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
                    variant=args.variant
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
        plot_aggregate_analysis(
            all_results,
            os.path.join(output_dir, "aggregate_analysis.png"),
            variant=args.variant
        )

    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
