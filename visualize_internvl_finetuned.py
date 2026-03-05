#!/usr/bin/env python3
"""
Finetuned InternVL × ChartQA Visualization
Based on "From Redundancy to Relevance" (NAACL 2025)

Runs the same redundancy/information-flow analysis as visualize_internvl_chartqa.py
but accepts --model_path and --model_name so any InternVL-family checkpoint
(including DocSP variants) can be visualized.

Usage:
    # InternVL v4 finetuned
    python visualize_internvl_finetuned.py \
        --model_path /NetDisk/j_son/internvl_35_v4_20260303/finetuning_20260303 \
        --model_name internvl_v4_ft \
        --n_samples 5

    # DocSP stage2
    python visualize_internvl_finetuned.py \
        --model_path /NetDisk/juyeon/DocSP/outputs/stage2_multinode \
        --model_name docsp_stage2 \
        --n_samples 5
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
from torchvision import transforms

from config import CHARTQA_DIR, CHARTQA_IMAGES, CHARTQA_HUMAN, CHARTQA_AUG

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


def load_model(model_path):
    """Load an InternVL-family model (standard or DocSP) with eager attention."""
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_flash_attn=False,
        device_map="auto",
    ).eval()

    use_docsp = getattr(model.config, 'use_docsp', False)
    print(f"  Model loaded. num_image_token={model.num_image_token}, use_docsp={use_docsp}")
    print(f"  Device map: {model.hf_device_map}")
    return model, tokenizer


def load_and_preprocess_image(image_path, image_size=448):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    pixel_values = transform(image).unsqueeze(0).cuda().to(torch.bfloat16)
    return pixel_values, image


# ═══════════════════════════════════════════════════════════════
# ViT Attention Capture
# ═══════════════════════════════════════════════════════════════

class AttentionCapture:
    """Context manager to capture attention maps from InternViT."""

    def __init__(self, model):
        self.model = model
        self.attention_maps = {}
        self._original_forwards = {}

    def __enter__(self):
        for idx, layer in enumerate(self.model.vision_model.encoder.layers):
            attn_module = layer.attn
            original_forward = attn_module._naive_attn

            def make_patched(orig, layer_idx, storage):
                def patched_naive_attn(x):
                    B, N, C = x.shape
                    module = self.model.vision_model.encoder.layers[layer_idx].attn
                    qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads)
                    qkv = qkv.permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)

                    if module.qk_normalization:
                        B_, H_, N_, D_ = q.shape
                        q = module.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
                        k = module.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

                    attn = (q * module.scale) @ k.transpose(-2, -1)
                    attn = attn.softmax(dim=-1)
                    storage[layer_idx] = attn.detach().cpu().float()
                    attn = module.attn_drop(attn)

                    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                    x = module.proj(x)
                    x = module.proj_drop(x)
                    return x
                return patched_naive_attn

            self._original_forwards[idx] = original_forward
            attn_module._naive_attn = make_patched(original_forward, idx, self.attention_maps)

        return self

    def __exit__(self, *args):
        for idx, layer in enumerate(self.model.vision_model.encoder.layers):
            if idx in self._original_forwards:
                layer.attn._naive_attn = self._original_forwards[idx]


# ═══════════════════════════════════════════════════════════════
# Visualization Functions
# ═══════════════════════════════════════════════════════════════

def plot_vit_attention_heatmaps(attention_maps, original_image, output_path, model_name, num_layers_to_show=6):
    n_layers = len(attention_maps)
    layer_indices = np.linspace(0, n_layers - 1, num_layers_to_show, dtype=int)

    fig, axes = plt.subplots(2, num_layers_to_show, figsize=(4 * num_layers_to_show, 8))
    img_np = np.array(original_image.resize((448, 448)))

    for col, layer_idx in enumerate(layer_indices):
        attn = attention_maps[layer_idx]
        cls_attn = attn[0, :, 0, 1:].mean(dim=0)
        h = w = int(cls_attn.shape[0] ** 0.5)
        cls_attn_2d = cls_attn.reshape(h, w).numpy()
        cls_attn_2d = (cls_attn_2d - cls_attn_2d.min()) / (cls_attn_2d.max() - cls_attn_2d.min() + 1e-8)
        attn_resized = np.array(Image.fromarray(cls_attn_2d).resize((448, 448), Image.BILINEAR))

        axes[0, col].imshow(cls_attn_2d, cmap=HEATMAP_CMAP, interpolation='bilinear')
        axes[0, col].set_title(f"Layer {layer_idx}", fontsize=12, fontweight='bold')
        axes[0, col].axis('off')

        axes[1, col].imshow(img_np)
        axes[1, col].imshow(attn_resized, cmap=HEATMAP_CMAP, alpha=0.5, interpolation='bilinear')
        axes[1, col].axis('off')

    axes[0, 0].set_ylabel("Attention\n(CLS→Patch)", fontsize=11, rotation=0, labelpad=80, va='center')
    axes[1, 0].set_ylabel("Overlay", fontsize=11, rotation=0, labelpad=80, va='center')

    fig.suptitle(f"[{model_name}] InternViT Layer-wise Attention Heatmaps (CLS → Patches)",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_information_flow(hidden_states, img_positions, output_path, model_name):
    n_layers = len(hidden_states)
    img_idx = np.where(img_positions)[0]

    if len(img_idx) == 0:
        print("  Warning: No image tokens found, skipping information flow plot.")
        return

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
        mean_token_sims.append(sim_matrix[mask].mean().item())

        if i > 0:
            prev_h = img_hidden[i - 1]
            cos = F.cosine_similarity(h, prev_h, dim=-1).mean().item()
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
        ax.axvline(x=cliff_idx + 1, color='red', linestyle='--', alpha=0.7, label=f'Cliff @ Layer {cliff_idx + 1}')
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

    fig.suptitle(f"[{model_name}] Information Flow Analysis (LLM Layers)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_token_redundancy_matrices(hidden_states, img_positions, output_path, model_name, layers_to_show=None):
    n_layers = len(hidden_states)
    img_idx = np.where(img_positions)[0]
    if len(img_idx) == 0:
        return

    if layers_to_show is None:
        layers_to_show = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]

    n_show = len(layers_to_show)
    max_tokens = 64
    if len(img_idx) > max_tokens:
        step = len(img_idx) // max_tokens
        img_idx_sub = img_idx[::step][:max_tokens]
    else:
        img_idx_sub = img_idx

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
    fig.suptitle(f"[{model_name}] Image Token Redundancy (Cosine Similarity Matrices)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_comprehensive_dashboard(
    original_image, question, answer, model_answer,
    attention_maps, hidden_states, img_positions,
    output_path, model_name
):
    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    img_np = np.array(original_image.resize((448, 448)))
    img_idx = np.where(img_positions)[0]

    # Row 0, Col 0: Original Image + Q&A
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img_np)
    ax.set_title("Chart Image", fontsize=11, fontweight='bold')
    ax.axis('off')
    qa_text = f"Q: {question[:80]}{'...' if len(question) > 80 else ''}\nGT: {answer}\nPred: {model_answer}"
    ax.text(0.5, -0.15, qa_text, transform=ax.transAxes, fontsize=8,
            ha='center', va='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            wrap=True)

    # Row 0, Col 1-3: ViT attention at layers 0, mid, last
    n_vit = len(attention_maps)
    for col, layer_idx in enumerate([0, n_vit // 2, n_vit - 1]):
        ax = fig.add_subplot(gs[0, col + 1])
        if layer_idx in attention_maps:
            attn = attention_maps[layer_idx]
            cls_attn = attn[0, :, 0, 1:].mean(dim=0)
            h = w = int(cls_attn.shape[0] ** 0.5)
            cls_attn_2d = cls_attn.reshape(h, w).numpy()
            cls_attn_2d = (cls_attn_2d - cls_attn_2d.min()) / (cls_attn_2d.max() - cls_attn_2d.min() + 1e-8)
            attn_resized = np.array(Image.fromarray(cls_attn_2d).resize((448, 448), Image.BILINEAR))
            ax.imshow(img_np)
            ax.imshow(attn_resized, cmap=HEATMAP_CMAP, alpha=0.5, interpolation='bilinear')
        ax.set_title(f"ViT Layer {layer_idx}", fontsize=11, fontweight='bold')
        ax.axis('off')

    # Row 1: Information Flow
    n_layers = len(hidden_states)
    if len(img_idx) > 0:
        img_hidden = [hs[0, img_idx, :] for hs in hidden_states]
        cosine_sims = []
        mean_token_sims = []
        for i in range(n_layers):
            h = img_hidden[i]
            h_norm = F.normalize(h, dim=-1)
            sim_matrix = h_norm @ h_norm.T
            mask = ~torch.eye(sim_matrix.shape[0], dtype=torch.bool)
            mean_token_sims.append(sim_matrix[mask].mean().item())
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

    # Row 2: Token Redundancy Matrices
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

    fig.suptitle(f"[{model_name}] ChartQA — Information Flow & Redundancy Analysis",
                 fontsize=16, fontweight='bold', y=1.01)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_aggregate_analysis(all_results, output_path, model_name):
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
        ax.axvline(x=cliff_idx + 1, color='red', linestyle='--', alpha=0.7, label=f'Mean Cliff @ Layer {cliff_idx + 1}')
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

    fig.suptitle(f"[{model_name}] Aggregate Analysis on ChartQA", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Finetuned InternVL ChartQA Visualization")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the finetuned InternVL checkpoint")
    parser.add_argument("--model_name", type=str, required=True, help="Short name for output directory and plot titles")
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--split", type=str, default="human", choices=["human", "augmented"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_llm", action="store_true", help="Skip LLM hidden state analysis (faster)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"outputs/chartqa_{args.model_name}_viz"

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    samples = load_chartqa_samples(args.n_samples, args.split)
    print(f"Loaded {len(samples)} ChartQA samples ({args.split} split)")

    model, tokenizer = load_model(args.model_path)

    all_results = []

    for idx, sample in enumerate(samples):
        print(f"\n{'='*60}")
        print(f"Sample {idx+1}/{len(samples)}: {sample['imgname']}")
        print(f"  Q: {sample['question']}")
        print(f"  A: {sample['answer']}")
        print(f"{'='*60}")

        pixel_values, original_image = load_and_preprocess_image(sample["image_path"])

        # 1. ViT Attention
        print("  [1/4] Extracting ViT attention maps...")
        with AttentionCapture(model) as cap:
            with torch.no_grad():
                _ = model.vision_model(pixel_values, output_hidden_states=False, return_dict=True)
            attention_maps = cap.attention_maps

        if attention_maps:
            plot_vit_attention_heatmaps(
                attention_maps, original_image,
                os.path.join(output_dir, f"sample_{idx:03d}_vit_attention.png"),
                args.model_name
            )

        # 2. Model answer
        print("  [2/4] Getting model answer...")
        try:
            generation_config = dict(max_new_tokens=128, do_sample=False)
            model_answer = model.chat(
                tokenizer, pixel_values, sample["question"],
                generation_config=generation_config
            )
        except Exception as e:
            print(f"  Warning: Generation failed ({e}), using placeholder.")
            model_answer = "[generation failed]"
        print(f"  Model answer: {model_answer}")

        # 3. LLM Hidden States
        result = {}
        if not args.skip_llm:
            print("  [3/4] Extracting LLM hidden states...")
            try:
                if args.model_path not in sys.path:
                    sys.path.insert(0, args.model_path)
                from conversation import get_conv_template

                with torch.no_grad():
                    vit_embeds = model.extract_feature(pixel_values)

                IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
                img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
                model.img_context_token_id = img_context_token_id
                num_patches = pixel_values.shape[0]

                question_text = sample["question"]
                if '<image>' not in question_text:
                    question_text = '<image>\n' + question_text

                template = get_conv_template(model.template)
                template.system_message = model.system_message
                template.append_message(template.roles[0], question_text)
                template.append_message(template.roles[1], None)
                query = template.get_prompt()

                image_tokens = '<img>' + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + '</img>'
                query = query.replace('<image>', image_tokens, 1)

                model_inputs = tokenizer(query, return_tensors='pt')
                input_ids = model_inputs['input_ids'].cuda()
                attention_mask = model_inputs['attention_mask'].cuda()

                input_embeds = model.language_model.get_input_embeddings()(input_ids)
                B, N, C = input_embeds.shape
                input_embeds_flat = input_embeds.reshape(B * N, C)
                input_ids_flat = input_ids.reshape(B * N)
                selected = (input_ids_flat == img_context_token_id)
                input_embeds_flat[selected] = vit_embeds.reshape(-1, C).to(input_embeds_flat.device)
                input_embeds = input_embeds_flat.reshape(B, N, C)

                img_positions = (input_ids[0] == img_context_token_id).cpu().numpy()

                outputs = model.language_model(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    output_attentions=False,
                    return_dict=True,
                )

                hidden_states = [h.detach().cpu().float() for h in outputs.hidden_states]
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
                        redundancy.append(sim_matrix[mask].mean().item())
                        if i > 0:
                            cos = F.cosine_similarity(h, img_hidden[i - 1], dim=-1).mean().item()
                            cosine_sims.append(cos)
                    result = {'cosine_sims': cosine_sims, 'redundancy': redundancy}

                plot_information_flow(
                    hidden_states, img_positions,
                    os.path.join(output_dir, f"sample_{idx:03d}_info_flow.png"),
                    args.model_name
                )
                plot_token_redundancy_matrices(
                    hidden_states, img_positions,
                    os.path.join(output_dir, f"sample_{idx:03d}_redundancy.png"),
                    args.model_name
                )
                plot_comprehensive_dashboard(
                    original_image, sample["question"], sample["answer"], model_answer,
                    attention_maps, hidden_states, img_positions,
                    os.path.join(output_dir, f"sample_{idx:03d}_dashboard.png"),
                    args.model_name
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
        plot_aggregate_analysis(all_results, os.path.join(output_dir, "aggregate_analysis.png"), args.model_name)

    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
