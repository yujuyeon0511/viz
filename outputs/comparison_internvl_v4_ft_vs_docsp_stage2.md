# Redundancy Visualization 비교: InternVL v4 FT vs DocSP Stage2

> 분석 날짜: 2026-03-05
> 기반 논문: "From Redundancy to Relevance" (NAACL 2025)
> 데이터셋: ChartQA (human split, 5 samples)

---

## 1. 모델 개요

| 항목 | InternVL v4 FT | DocSP Stage2 |
|------|---------------|-------------|
| 경로 | `/NetDisk/j_son/internvl_35_v4_20260303/finetuning_20260303` | `/NetDisk/juyeon/DocSP/outputs/stage2_multinode` |
| 아키텍처 | InternVLChatModel | InternVLChatModel |
| Vision Encoder | InternViT-6B (24 layers) | InternViT-6B (24 layers) |
| LLM | Qwen3-8B (36 layers) | Qwen3-8B (36 layers) |
| Template | internvl2_5 | internvl2_5 |
| DocSP Projector | No (`use_docsp=False`) | Yes (`use_docsp=True`) |
| num_image_token | 268 | 264 (256 patch + 8 DocSP spatial) |
| 추가 모듈 | SFE pool/crop convs, DFI | DocSP (FreqDecomp + MultiScale SFE + DAG-FI) |

---

## 2. ChartQA 답변 비교

| # | Question | GT | InternVL v4 FT | DocSP Stage2 |
|---|----------|----|--------------------|----------------|
| 0 | How many food item is shown in the bar graph? | 14 | 12 food items are shown in the bar graph. | 15. |
| 1 | How many bars are shown in the chart? | 3 | There are three bars shown in the chart. | 3. |
| 2 | What's the value of the lowest bar? | 23 | The value of the lowest bar is 23%. | 23. |
| 3 | What percent who think of President Donald Trump as Dangerous? | 62 | 62% think of President Donald Trump as dangerous. | 62. |
| 4 | What was the 4th most popular emotion? | Inspired | The 4th most popular emotion was inspired. | Inspired. |

### 답변 정확도 요약

| 모델 | 정답 수 (Relaxed Match) | 비고 |
|------|----------------------|------|
| InternVL v4 FT | 4/5 (80%) | Sample 0에서 14 -> 12로 오답. 자연어 문장 형태로 답변 |
| DocSP Stage2 | 4/5 (80%) | Sample 0에서 14 -> 15로 오답. 간결한 답변 (값만 출력) |

- 두 모델 모두 Sample 0 (bar graph counting)에서 오답이지만, 오답 방향이 다름 (v4: under-count, DocSP: over-count)
- DocSP는 간결한 단답형, v4 FT는 자연어 문장 형태로 답변하는 경향

---

## 3. Information Flow 분석

### 3.1 Inter-layer Similarity (레이어 간 코사인 유사도)

| 메트릭 | InternVL v4 FT | DocSP Stage2 |
|--------|---------------|-------------|
| 초기 레이어 (0-5) 유사도 | ~0.88-0.93 | ~0.95-1.00 |
| 중간 레이어 (10-25) 유사도 | ~0.93-0.96 | ~0.93-0.95 |
| Cliff 위치 (Mean) | Layer 36 | Layer 36 |
| Cliff 분포 | 5/5 samples @ Layer 36 | 5/5 samples @ Layer 36 |
| Cliff 이후 유사도 하락 | ~0.96 -> ~0.30 (급격한 하락) | ~0.93 -> ~0.45 (급격한 하락) |

**관찰:**
- 두 모델 모두 Layer 36 (마지막 레이어)에서 정보 흐름 cliff가 발생
- DocSP가 초기 레이어에서 더 높은 inter-layer similarity를 보임 (~1.0에 근접) -> 초기 레이어에서 이미지 토큰 표현이 더 안정적으로 유지됨
- v4 FT의 cliff 하락폭이 더 큼 (~0.66 drop vs ~0.48 drop)

### 3.2 Image Token L2 Norm

| 메트릭 | InternVL v4 FT | DocSP Stage2 |
|--------|---------------|-------------|
| 초기 레이어 Norm | ~800 | ~150 |
| 중간 레이어 Norm | ~4000-5000 | ~150-200 |
| 최종 레이어 Norm | ~5500 | ~1200 |
| Norm 증가 패턴 | Layer 0-5에서 급격 증가 후 완만한 상승 | Layer 0-25 완만, Layer 25+ 급격 증가 |

**관찰:**
- **DocSP의 L2 norm이 v4 FT보다 현저히 낮음** (약 4-5배 차이)
- DocSP는 후반부 레이어에서만 norm이 증가하는 반면, v4 FT는 초기부터 높은 norm을 보임
- DocSP의 낮은 norm은 FrequencyDecomposition을 통한 feature 정규화 효과로 해석 가능

### 3.3 Image Token Variance (정보 수렴/발산)

| 메트릭 | InternVL v4 FT | DocSP Stage2 |
|--------|---------------|-------------|
| 초기 레이어 Variance | ~200 | ~10 |
| 중간 레이어 Variance | ~200-1000 | ~10-15 |
| 최종 레이어 Variance | ~200,000 (폭발적 증가) | ~60-70 |
| Variance 폭발 시점 | Layer 32+ | Layer 30+ (상대적으로 완만) |

**관찰:**
- **v4 FT는 마지막 레이어에서 variance가 폭발적으로 증가** (~200,000), DocSP는 ~60-70으로 매우 안정적
- DocSP의 극적으로 낮은 variance는 DocSP projector가 이미지 토큰 표현을 보다 균일하게 유지함을 시사
- 이는 DocSP의 Frequency Decomposition이 구조적/의미적 정보를 분리하여 정보가 더 효율적으로 흐르게 한 결과로 보임

### 3.4 Token Redundancy (이미지 토큰 간 코사인 유사도)

| 메트릭 | InternVL v4 FT | DocSP Stage2 |
|--------|---------------|-------------|
| 초기 레이어 Redundancy | ~0.20 | ~0.30-0.35 |
| 중간 레이어 Redundancy | ~0.25-0.40 | ~0.20-0.25 |
| 최종 레이어 Redundancy | ~0.60-0.80 | ~0.55-0.75 |
| 전체 추세 | 점진적 증가 | U-shape (초기 높음 -> 중간 낮음 -> 후반 증가) |

**관찰:**
- **DocSP는 중간 레이어에서 redundancy가 v4 FT보다 낮음** -> 중간 처리 과정에서 토큰 다양성이 더 잘 유지됨
- DocSP의 U-shape 패턴은 초기 DocSP spatial 토큰의 유사성이 중간 레이어에서 분화된 후 최종적으로 다시 수렴하는 것을 보여줌
- 두 모델 모두 최종 레이어에서 높은 redundancy (0.6-0.8)를 보임 -> 토큰 압축/제거의 여지가 있음

---

## 4. ViT Attention 분석

두 모델은 동일한 InternViT-6B를 사용하지만, 파인튜닝 과정에서 ViT 가중치도 업데이트되었을 수 있어 attention 패턴에 차이가 나타남.

- **초기 레이어 (Layer 0-4):** 두 모델 모두 균일한 attention 분포
- **중간 레이어 (Layer 12):** 차트의 주요 구조 영역에 attention이 집중되기 시작
- **최종 레이어 (Layer 23):** 두 모델 모두 차트의 핵심 정보 영역(축, 레이블, 데이터 포인트)에 높은 attention

---

## 5. 핵심 비교 정리

| 분석 항목 | InternVL v4 FT | DocSP Stage2 | 우위 |
|-----------|---------------|-------------|------|
| ChartQA 정확도 (5 samples) | 4/5 | 4/5 | 동등 |
| 답변 스타일 | 자연어 문장 | 간결한 단답 | - |
| Inter-layer Stability | 보통 (0.88-0.96) | 높음 (0.93-1.00) | DocSP |
| L2 Norm 안정성 | 높은 norm, 초기 급증 | 낮은 norm, 안정적 | DocSP |
| Variance 안정성 | 최종 레이어 폭발 (~200K) | 전 구간 안정 (~70) | DocSP |
| 중간 레이어 Redundancy | 보통 (~0.30) | 낮음 (~0.22) | DocSP |
| Information Cliff | Layer 36 | Layer 36 | 동등 |

---

## 6. 결론

1. **DocSP Stage2가 전반적으로 더 안정적인 정보 흐름을 보임**: L2 norm, variance, inter-layer similarity 모두에서 DocSP가 더 안정적인 패턴을 나타냄. 이는 DocSP projector의 Frequency Decomposition과 Multi-Scale SFE가 이미지 토큰의 정보를 더 효율적으로 인코딩하고 있음을 시사.

2. **중간 레이어 redundancy 감소**: DocSP는 중간 레이어에서 토큰 간 다양성이 더 높게 유지되어, LLM이 이미지 정보를 처리할 때 더 풍부한 정보를 활용할 수 있음.

3. **Variance 폭발 억제**: v4 FT에서 관찰되는 극단적인 variance 폭발(~200,000)이 DocSP에서는 크게 억제됨(~70). 이는 학습 안정성과 추론 안정성 모두에 기여할 수 있음.

4. **Information Cliff 위치 동일**: 두 모델 모두 Layer 36에서 cliff가 발생하며, 이는 Qwen3-8B LLM의 구조적 특성으로 보임.

5. **답변 정확도는 동등**: 5개 샘플 기준으로는 유의미한 차이 없음. 더 많은 샘플에서의 정량 평가가 필요.

---

## 7. 시각화 파일 경로

| 모델 | 디렉토리 |
|------|---------|
| InternVL v4 FT | `outputs/chartqa_internvl_v4_ft_viz/` |
| DocSP Stage2 | `outputs/chartqa_docsp_stage2_viz/` |

각 디렉토리 내 파일 구성:
- `sample_XXX_vit_attention.png` - ViT 레이어별 CLS->Patch attention 히트맵
- `sample_XXX_info_flow.png` - LLM 레이어별 정보 흐름 4개 메트릭
- `sample_XXX_redundancy.png` - 이미지 토큰 코사인 유사도 매트릭스
- `sample_XXX_dashboard.png` - 종합 대시보드
- `aggregate_analysis.png` - 전체 샘플 집계 분석
