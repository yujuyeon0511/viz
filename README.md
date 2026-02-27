# Redundancy Visualization

**"From Redundancy to Relevance" (NAACL 2025)** 논문의 분석 방법론을 InternVL3.5-8B, LLaVA-SP, Qwen2.5-VL 모델에 적용한 시각화 도구.

## 분석 항목

1. **ViT Attention Heatmap** — Vision Encoder의 레이어별 CLS→Patch attention 시각화
2. **Information Flow** — LLM 레이어 간 image token의 cosine similarity, L2 norm, variance 변화
3. **Token Redundancy** — LLM 레이어별 image token 간 pairwise cosine similarity matrix
4. **Dashboard** — 위 분석을 종합한 per-sample 대시보드
5. **Aggregate Analysis** — 전체 샘플에 대한 평균 곡선 및 cliff layer 분포

## 환경 설정

### 필수 패키지

```bash
pip install -r requirements.txt
```

### 경로 설정

`config.py`에서 모델 및 데이터 경로를 환경에 맞게 수정:

```python
# 모델 경로
INTERNVL_MODEL_PATH  = "/NetDisk/juyeon/models/InternVL3_5-8B"
QWEN25VL_MODEL_PATH  = "/NetDisk/juyeon/models/Qwen2.5-VL-7B-Instruct"
VICUNA_PATH          = "/NetDisk/juyeon/models/vicuna-7b-v1.5"
LORA_POOLING         = "/NetDisk/juyeon/models/llava-sp-pooling-lora"
LORA_CROPPING        = "/NetDisk/juyeon/models/llava-sp-cropping-lora"

# 데이터 경로
CHARTQA_DIR = "/NetDisk/juyeon/train/chartQA/ChartQA Dataset/test"
```

### Conda 환경

| 스크립트 | Conda 환경 | transformers 버전 |
|---------|-----------|------------------|
| InternVL | `chartinternvl` | 4.57.6 |
| Qwen2.5-VL | `chartinternvl` | 4.57.6 |
| LLaVA-SP | `vlm_exp` | 4.44 |

## 실행 방법

### InternVL3.5-8B

```bash
CUDA_VISIBLE_DEVICES=0 /home/juyeon/miniconda3/envs/chartinternvl/bin/python \
    visualize_internvl_chartqa.py --n_samples 3
```

### LLaVA-SP (Pooling)

```bash
CUDA_VISIBLE_DEVICES=0 /home/juyeon/miniconda3/envs/vlm_exp/bin/python \
    visualize_llavaSP_chartqa.py --n_samples 3 --variant pooling
```

### Qwen2.5-VL-7B

```bash
CUDA_VISIBLE_DEVICES=0 /home/juyeon/miniconda3/envs/chartinternvl/bin/python \
    visualize_qwen25vl_chartqa.py --n_samples 3
```

### LLaVA-SP (Cropping)

```bash
CUDA_VISIBLE_DEVICES=0 /home/juyeon/miniconda3/envs/vlm_exp/bin/python \
    visualize_llavaSP_chartqa.py --n_samples 3 --variant cropping
```

### 공통 옵션

| 옵션 | 설명 | 기본값 |
|------|------|-------|
| `--n_samples` | 시각화할 ChartQA 샘플 수 | 5 |
| `--split` | `human` 또는 `augmented` | `human` |
| `--output_dir` | 출력 디렉토리 | 자동 설정 |
| `--skip_llm` | LLM hidden state 분석 생략 (ViT attention만) | False |

## 출력 파일

```
outputs/chartqa_{model}_viz/
├── sample_000_vit_attention.png   # ViT 레이어별 attention heatmap
├── sample_000_info_flow.png       # Information flow 4개 지표
├── sample_000_redundancy.png      # Token redundancy matrix
├── sample_000_dashboard.png       # 종합 대시보드
├── ...                            # 샘플별 반복
└── aggregate_analysis.png         # 전체 샘플 통계
```

## 참고

- **분석 방법론**: Zhang et al., "From Redundancy to Relevance: Information Flow in LVLMs Across Reasoning Tasks", NAACL 2025 ([GitHub](https://github.com/zhangbaijin/From-Redundancy-to-Relevance))
- **LLaVA-SP 웨이트**: [Levideus/llava-sp-pooling-lora](https://huggingface.co/Levideus/llava-sp-pooling-lora)
- **GPU 요구사항**: NVIDIA A100 80GB x1 권장
