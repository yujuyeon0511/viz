# ChartQA × 5-Model Comparison: Information Flow & Token Redundancy 종합 분석

**분석 기간**: 2026-02-27 ~ 2026-03-05
**분석 방법론**: "From Redundancy to Relevance" (NAACL 2025, Zhang et al.)
**데이터**: ChartQA test_human split
**환경**: NVIDIA A100 80GB x2

---

## 1. 비교 대상 모델 (5종)

### 1.1 Pretrained 모델 (3종, 20 samples)

| 항목 | InternVL3.5-8B | Qwen2.5-VL-7B | LLaVA-SP (Pooling) |
|------|---------------|---------------|-------------------|
| Vision Encoder | InternViT (24L, 1024d) | Qwen2.5-ViT (32L, 1280d) | CLIP-ViT-L/14 (24L, 1024d) |
| Image Size | 448×448 | 동적 해상도 | 336×336 |
| Patch Size | 14 | 14 | 14 |
| Raw Patches | 1024 (32×32) | 동적 | 576 (24×24) |
| Downsampling | Pixel Shuffle 0.5× | Spatial Merge 2×2 | 없음 |
| Spatial Tokens | 없음 | 없음 | 6개 (multi-scale conv) |
| **Total Image Tokens** | **256** | **동적 (84~840)** | **582** (576 + 6) |
| Projector | MLP 2× GELU | PatchMerger (MLP) | MLP 2× GELU |
| LLM | Qwen3-8B (36L, 4096d) | Qwen2.5-7B (28L, 3584d) | Vicuna-7B (32L, 4096d) |
| Attention Type | Full | Window + Full (L7,15,23,31) | Full |
| Weight Type | Full weight | Full weight | LoRA (r=128, α=256) |

### 1.2 Finetuned 모델 (2종, 5 samples)

| 항목 | InternVL v4 FT | DocSP Stage2 |
|------|---------------|-------------|
| 체크포인트 | `j_son/internvl_35_v4_20260303/finetuning_20260303` | `juyeon/DocSP/outputs/stage2_multinode` |
| Base | InternVL3.5-8B | InternVL3.5-8B |
| Vision Encoder | InternViT-6B (24L, 1024d) | InternViT-6B (24L, 1024d) |
| LLM | Qwen3-8B (36L, 4096d) | Qwen3-8B (36L, 4096d) |
| DocSP Projector | No (`use_docsp=False`) | Yes (`use_docsp=True`) |
| num_image_token | 268 (256 patch + 12 SFE/DFI) | 264 (256 patch + 8 DocSP spatial) |
| Projector 구조 | Pixel Shuffle + SFE pool/crop convs + DFI | Pixel Shuffle + mlp1 + **DocSP** (FreqDecomp + MultiScale SFE + DAG-FI) |

### 1.3 DocSP vs SFE/DFI 구조 차이

```
[InternVL v4 FT]  ViT → pixel_shuffle → SFE/DFI convs → mlp1 → 268 tokens
                                          ↑ downsampled feature에서 추출

[DocSP Stage2]    ViT ─┬→ pixel_shuffle → mlp1 → 256 tokens
                       └→ DocSPProjector (원본 32×32 해상도)
                            ├─ Frequency Decomposition (high/low 분리)
                            ├─ Multi-Scale SFE (구조/의미 별도 추출)
                            ├─ DAG-FI (cross-attention 융합)
                            └─ → +8 spatial tokens
                          = 264 tokens
```

**핵심 차이**: DocSP는 pixel_shuffle **전** 원본 해상도에서 주파수 분해를 수행하여, 고주파(텍스트/경계선)와 저주파(레이아웃/영역) 정보를 분리 인코딩

---

## 2. ChartQA 정확도 비교

### 2.1 전체 요약

| 모델 | 샘플 수 | 정답 수 | 정확도 |
|------|---------|---------|--------|
| **InternVL3.5-8B** (pretrained) | 20 | 15 | **75.0%** |
| **Qwen2.5-VL-7B** (pretrained) | 20 | 14 | **70.0%** |
| **InternVL v4 FT** (finetuned) | 5 | 4 | **80.0%** |
| **DocSP Stage2** (finetuned) | 5 | 4 | **80.0%** |
| **LLaVA-SP Pooling** (pretrained) | 20 | 4 | **20.0%** |

### 2.2 Pretrained 모델 상세 결과 (20 samples)

| # | 이미지 | 질문 | 정답 | InternVL | Qwen2.5-VL | LLaVA-SP |
|---|--------|------|------|----------|------------|----------|
| 1 | 41699051005347.png | How many food item is shown in the bar graph? | 14 | 14 **O** | 15 **X** | 10 **X** |
| 2 | 41810321001157.png | How many bars are shown in the chart? | 3 | three **O** | three **O** | two **X** |
| 3 | 8127.png | What's the value of the lowest bar? | 23 | 23% **O** | 23% **O** | (설명만) **X** |
| 4 | 166.png | What percent who think of Trump as Dangerous? | 62 | 62% **O** | 62% **O** | 67% **X** |
| 5 | 3960.png | What was the 4th most popular emotion? | Inspired | Connected **X** | Connected **X** | amused **X** |
| 6 | 01499440003158.png | What is the value of Slovenia in the graph? | 1 | 1.00 **O** | 1 GPI **O** | 8.7 **X** |
| 7 | 1366.png | What's the lefmost value of bar in China? | 17 | 17% **O** | 17 **O** | 17 **O** |
| 8 | 13750.png | What's the percentage of U.S adults who refused? | 2 | 2% **O** | 2% **O** | 24% **X** |
| 9 | 08524901006324.png | When does the line reach the peak? | 2014 | 2014 **O** | 2014 **O** | 2011 **X** |
| 10 | 20374873014871.png | How many colors are used in the graph? | 1 | two **X** | one **O** | two **X** |
| 11 | 77342851005157.png | Which line represents data about boys? | green line | blue **X** | teal **X** | orange **X** |
| 12 | 1392.png | Find missing data: 24, _, 32, 33, 42? | 29 | 29 **O** | 29 **O** | 24 **X** |
| 13 | 5831.png | Is the percentage value of "STEM" segment 52? | Yes | Yes **O** | Yes **O** | Yes **O** |
| 14 | 15948.png | What's the percentage of biggest segment? | 80 | 80% **O** | 80% **O** | 80% **O** |
| 15 | 5967.png | How many percent are fewer refugees in Jordan? | 0.6 | 23% **X** | 60% **X** | (설명만) **X** |
| 16 | OECD_FDI...042.png | How many years are represented on this graph? | 13 | 13 **O** | 14 **X** | 2005-2015 **X** |
| 17 | 5417.png | How many are Somewhat confident Trump can...? | 24 | 21% **X** | 21% **X** | 43% **X** |
| 18 | OECD_SECONDARY...019.png | Which country has highest rate in 2018? | Italy | Italy **O** | Italy **O** | Spain **X** |
| 19 | 4178.png | How many waited in Total for 10mins? | 33 | 33% **O** | 33% **O** | 10 **X** |
| 20 | 8597.png | How many categories are there in the chart? | 4 | four **O** | four **O** | four **O** |

### 2.3 Finetuned 모델 상세 결과 (5 samples, 동일 첫 5개 이미지)

| # | 이미지 | 정답 | InternVL v4 FT | DocSP Stage2 |
|---|--------|------|----------------|-------------|
| 1 | 41699051005347.png | 14 | 12 **X** (under-count) | 15 **X** (over-count) |
| 2 | 41810321001157.png | 3 | three **O** | 3 **O** |
| 3 | 8127.png | 23 | 23% **O** | 23 **O** |
| 4 | 166.png | 62 | 62% **O** | 62 **O** |
| 5 | 3960.png | Inspired | inspired **O** | Inspired **O** |

**주목할 점**: Sample 5 (4th most popular emotion)에서 pretrained InternVL/Qwen 모두 "Connected"로 오답 → 두 finetuned 모델은 모두 "Inspired"로 **정답**. Finetuning이 순서 해석 능력을 개선.

### 2.4 오답 패턴 분석

**InternVL3.5-8B** (5/20 오답): 순서 해석(S5), 색상 인식(S10,S11), 모호한 질문(S15), 수치 읽기(S17)
**Qwen2.5-VL-7B** (6/20 오답): 세기(S1), 순서(S5), 색상(S11), 모호한 질문(S15), 연도 세기(S16), 수치(S17)
**LLaVA-SP** (16/20 오답): 수치 읽기 실패 다수, 질문에 설명만 제공, 시각적 요소 식별 실패

**답변 스타일 차이**:
- Pretrained 모델: 자연어 문장 ("The 4th most popular emotion was inspired.")
- DocSP Stage2: 간결한 단답 ("Inspired.")

### 2.5 모델 간 일치/불일치 (20 samples 기준)

- **3모델 모두 정답 (3개)**: S7, S13, S14 → 단순 수치 읽기 + 확인형
- **3모델 모두 오답 (3개)**: S5, S11, S15 → 순서/색상/모호한 질문
- **InternVL+Qwen 정답, LLaVA 오답 (11개)**: 대부분의 수치/비교 질문
- **InternVL만 정답 (2개)**: S1(음식 14개), S16(13년)
- **Qwen만 정답 (1개)**: S10(1색)

---

## 3. 이미지 토큰 효율성 분석

| 모델 | Image Tokens (평균) | 정확도 | 토큰당 정확도 | LLM Layers |
|------|-------------------|--------|-------------|------------|
| InternVL3.5-8B | 256 | 75.0% | 0.293%/tok | 36 |
| InternVL v4 FT | 268 | 80.0%* | 0.299%/tok | 36 |
| DocSP Stage2 | 264 | 80.0%* | 0.303%/tok | 36 |
| Qwen2.5-VL-7B | ~395 (동적) | 70.0% | 0.177%/tok | 28 |
| LLaVA-SP | 582 | 20.0% | 0.034%/tok | 32 |

*5 samples 기준 (통계적 유의성 한계)

**토큰 압축 전략 비교**:

| 전략 | 모델 | 압축률 | 효과 |
|------|------|--------|------|
| Pixel Shuffle 0.5× | InternVL (all) | 4:1 (1024→256) | 최소 토큰, 최고 효율 |
| Pixel Shuffle + DocSP | DocSP Stage2 | 4:1 + 8 spatial | 안정적 정보 흐름 |
| Pixel Shuffle + SFE/DFI | InternVL v4 FT | 4:1 + 12 extra | 비교 대상 |
| Spatial Merge 2×2 | Qwen2.5-VL | 4:1 (동적) | 이미지 적응형 |
| 압축 없음 + Conv | LLaVA-SP | 1:1 (576+6) | 비효율적 |

---

## 4. Information Flow 분석 (LLM Hidden States)

### 4.1 Inter-layer Cosine Similarity

| 모델 | 초기 레이어 | 중간 레이어 | 후기 레이어 | Cliff 위치 | Cliff 상대위치 |
|------|------------|------------|------------|-----------|-------------|
| InternVL3.5-8B | ~0.85 | ~0.90-0.95 | 급락 ~0.60 | L30-33 | ~86% |
| **InternVL v4 FT** | ~0.88-0.93 | ~0.93-0.96 | 급락 ~0.30 | **L36** | **100%** |
| **DocSP Stage2** | **~0.95-1.00** | ~0.93-0.95 | 급락 ~0.45 | **L36** | **100%** |
| Qwen2.5-VL-7B | ~0.90 | ~0.93-0.96 | 급락 ~0.65 | L23-26 | ~82% |
| LLaVA-SP | ~0.80 | ~0.85-0.90 | 완만한 하강 | L25-28 | ~78% |

**핵심 발견**:
- 모든 모델에서 LLM 후기 레이어(~80-100%)에서 information cliff 관찰
- **DocSP가 전 모델 중 초기 레이어 유사도 가장 높음** (~1.0) → 이미지 토큰 표현이 가장 안정적
- Finetuned 모델(v4 FT, DocSP)은 cliff가 마지막 레이어(L36)로 밀림 → 정보가 더 오래 유지
- LLaVA-SP는 cliff가 가장 완만 → 정보 전달이 덜 효율적

### 4.2 L2 Norm 변화

| 모델 | 초기 Norm | 최종 Norm | 증가 배율 | 패턴 |
|------|----------|----------|----------|------|
| InternVL3.5-8B | 낮음 | 급증 | - | 후기 급증 |
| **InternVL v4 FT** | **~800** | **~5500** | **~7×** | 초기 급증 후 완만 상승 |
| **DocSP Stage2** | **~150** | **~1200** | **~8×** | 후반부에서만 증가 |
| Qwen2.5-VL-7B | 중간 | 급증 | - | 후기 급증 |
| LLaVA-SP | 전반적 낮음 | 낮음 | - | 텍스트 대비 consistently 낮음 |

**핵심 발견**:
- **DocSP의 L2 norm이 v4 FT보다 약 4-5배 낮음** → FrequencyDecomposition의 feature 정규화 효과
- DocSP는 후반부 레이어에서만 norm 증가, v4 FT는 초기부터 높은 norm

### 4.3 Image Token Variance (정보 수렴/발산)

| 모델 | 초기 Variance | 최종 Variance | 패턴 |
|------|-------------|-------------|------|
| **InternVL v4 FT** | ~200 | **~200,000** | **후기 폭발** |
| **DocSP Stage2** | ~10 | **~60-70** | **전 구간 안정** |

**핵심 발견**:
- v4 FT는 마지막 레이어에서 variance **폭발** (~200,000) → 일부 토큰의 표현이 극단적으로 변화
- DocSP는 ~60-70으로 **약 3000배 낮은 variance** → 학습/추론 안정성에 기여
- DocSP의 Frequency Decomposition이 구조적/의미적 정보를 분리하여 정보가 균일하게 흐르게 함

### 4.4 Token Redundancy (이미지 토큰 간 Pairwise Cosine Similarity)

| 모델 | Image Tokens | 초기 Redundancy | 중간 Redundancy | 최종 Redundancy | 증가 폭 |
|------|-------------|----------------|----------------|----------------|---------|
| InternVL3.5-8B | 256 | ~0.50 | - | ~0.85-0.95 | +0.35~0.45 |
| **InternVL v4 FT** | 268 | ~0.20 | **~0.25-0.40** | ~0.60-0.80 | +0.40~0.60 |
| **DocSP Stage2** | 264 | ~0.30-0.35 | **~0.20-0.25** | ~0.55-0.75 | +0.25~0.45 |
| Qwen2.5-VL-7B | ~395 | ~0.55 | - | ~0.88-0.96 | +0.33~0.41 |
| LLaVA-SP | 582 | ~0.45 | - | ~0.80-0.88 | +0.35~0.43 |

**핵심 발견**:
- 모든 모델에서 후기 레이어로 갈수록 redundancy 급증 → "From Redundancy to Relevance" 논문과 일치
- **DocSP가 중간 레이어에서 가장 낮은 redundancy** (~0.20-0.25) → 토큰 다양성 최대 유지
- DocSP의 **U-shape 패턴** (초기 높음 → 중간 낮음 → 후반 증가): spatial 토큰의 초기 유사성이 중간 레이어에서 분화 후 다시 수렴
- InternVL의 Pixel Shuffle (256 tokens)이 LLaVA-SP (582 tokens)보다 높은 초기 redundancy → 압축으로 인한 정보 집약

---

## 5. ViT Attention 분석

| 모델 | ViT Attention 방식 | CLS Token | 특성 |
|------|-------------------|-----------|------|
| InternVL (all) | CLS→Patch | 있음 | 초기 균일 → 후기 특정 영역 집중 |
| Qwen2.5-VL-7B | Mean Attention Received | 없음 | Window+Full 혼합, L7,15,23,31에서 전역 통합 |
| LLaVA-SP | CLS→Patch | 있음 | 범용 특징 집중, 차트 특화 약함 |

- InternVL v4 FT와 DocSP Stage2는 동일 InternViT-6B 사용, finetuning으로 attention 패턴에 미세 차이
- 두 finetuned 모델 모두 최종 레이어에서 차트 핵심 영역(축, 레이블, 데이터 포인트)에 높은 attention

---

## 6. 종합 비교 정리

### 6.1 5모델 핵심 메트릭 비교

| 분석 항목 | InternVL 3.5 (pre) | InternVL v4 FT | DocSP Stage2 | Qwen2.5-VL | LLaVA-SP |
|-----------|:------------------:|:--------------:|:------------:|:----------:|:--------:|
| 정확도 | 75% (20s) | 80% (5s) | 80% (5s) | 70% (20s) | 20% (20s) |
| Image Tokens | 256 | 268 | 264 | ~395 | 582 |
| 토큰 효율 | ★★★★ | ★★★★ | ★★★★★ | ★★★ | ★ |
| Inter-layer Stability | ★★★ | ★★★ | ★★★★★ | ★★★★ | ★★ |
| L2 Norm 안정성 | ★★★ | ★★ | ★★★★★ | ★★★ | ★★★★ |
| Variance 안정성 | ★★★ | ★ | ★★★★★ | ★★★ | ★★★★ |
| 중간 Redundancy | - | ★★★ | ★★★★★ | - | - |
| Cliff 선명도 | ★★★★ | ★★★★★ | ★★★★★ | ★★★★ | ★★ |

### 6.2 모델 그룹별 특성

**고성능 그룹 (InternVL family + Qwen)**:
- 정확도 70-80%, 명확한 information cliff, 효율적인 토큰 압축
- 후기 레이어에서 이미지→텍스트 정보 전이가 명확하게 관찰됨

**DocSP (최고 안정성)**:
- 정확도 동등하면서 정보 흐름이 가장 안정적
- Variance 폭발 억제 (~200,000 → ~70), 중간 레이어 redundancy 최소화
- Frequency Decomposition으로 구조/의미 정보를 분리 인코딩한 효과

**LLaVA-SP (비효율)**:
- 가장 많은 토큰(582)으로 가장 낮은 정확도(20%)
- 정보 흐름 cliff가 완만 → 정보 전달 비효율
- LoRA finetuning의 한계 가능성

---

## 7. 결론

### 7.1 토큰 압축 관점
1. **Pixel Shuffle 4:1 압축이 가장 효율적**: InternVL family가 256 토큰으로 최고 토큰 효율 달성
2. **토큰 수 ≠ 성능**: LLaVA-SP (582 tokens) < InternVL (256 tokens). 압축 품질이 핵심
3. **DocSP의 8개 spatial 토큰**: 적은 추가 토큰으로 정보 흐름 안정성을 크게 개선

### 7.2 Information Flow 관점
1. **Information Cliff는 보편적**: 5모델 모두에서 LLM 후기 레이어(~80-100%)에서 관찰
2. **Finetuning이 cliff를 후방으로 이동**: Pretrained L30-33 → Finetuned L36 (정보 보존 기간 연장)
3. **DocSP의 Frequency Decomposition 효과**: L2 norm, variance, redundancy 모두에서 가장 안정적
4. **Redundancy 급증은 보편적**: 모든 모델에서 후기 레이어 redundancy 급증 → 토큰 제거/압축 여지 존재

### 7.3 DocSP의 차별화 포인트
| 메트릭 | v4 FT → DocSP | 개선 |
|--------|--------------|------|
| Variance (최종) | 200,000 → 70 | **~3000× 억제** |
| L2 Norm (초기) | 800 → 150 | **~5× 감소** |
| 중간 Redundancy | 0.30 → 0.22 | **27% 감소** |
| Inter-layer Sim (초기) | 0.90 → 0.98 | **더 안정적** |
| Cliff 하락폭 | 0.66 → 0.48 | **27% 완화** |

→ DocSP projector는 정확도를 유지하면서 정보 흐름의 안정성을 대폭 개선

---

## 8. 출력 파일 구조

```
outputs/
├── chartqa_internvl_20samples/              (Pretrained InternVL3.5-8B, 81 files)
├── chartqa_qwen25vl_20samples/              (Pretrained Qwen2.5-VL-7B, 81 files)
├── chartqa_llavaSP_20samples/               (Pretrained LLaVA-SP Pooling, 81 files)
├── chartqa_internvl_v4_ft_viz/              (Finetuned InternVL v4 FT, 21 files)
├── chartqa_docsp_stage2_viz/                (Finetuned DocSP Stage2, 21 files)
├── comparison_internvl_v4_ft_vs_docsp_stage2.md  (FT 모델 비교 상세)
├── 실험결과_3모델비교_20samples_20260227.md       (Pretrained 3모델 비교 상세)
└── 종합_5모델비교_information_flow_redundancy.md  (본 문서)
```

각 모델 디렉토리 내 파일:
- `sample_XXX_vit_attention.png` — ViT CLS→Patch attention 히트맵
- `sample_XXX_info_flow.png` — LLM 레이어별 4개 메트릭 (cosine sim, L2 norm, variance, redundancy)
- `sample_XXX_redundancy.png` — 이미지 토큰 코사인 유사도 매트릭스
- `sample_XXX_dashboard.png` — 종합 대시보드
- `aggregate_analysis.png` — 전체 샘플 집계 분석

---

## 9. 한계 및 향후 과제

1. **샘플 수 불균형**: Pretrained 20 samples vs Finetuned 5 samples → 동일 조건 비교 필요
2. **평가 기준**: 수동 판정 → ChartQA 공식 relaxed accuracy (5% tolerance) 적용 필요
3. **Cliff-layer token truncation**: Information cliff 이후 이미지 토큰 제거 시 성능 변화 실험
4. **DocSP spatial token 분석**: 8개 spatial 토큰이 어떤 정보를 담고 있는지 (struct 4개 vs sem 4개) 개별 분석
5. **전체 벤치마크**: ChartQA 전체 1250개 + DocVQA, InfoVQA 등에 대한 정량 평가
6. **Variance 폭발 원인**: v4 FT의 극단적 variance(~200K)가 성능에 미치는 영향 추가 조사
