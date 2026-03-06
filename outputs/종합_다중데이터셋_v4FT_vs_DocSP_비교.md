# InternVL v4 FT vs DocSP Stage2: 다중 데이터셋 Information Flow & Redundancy 비교

> 분석 날짜: 2026-03-06
> 기반 논문: "From Redundancy to Relevance" (NAACL 2025)
> 샘플 수: 각 데이터셋 20 samples
> 실행 환경: Kicloud18 (A100-PCIE-40GB x2)

---

## 1. 모델 개요

| 항목 | InternVL v4 FT | DocSP Stage2 |
|------|---------------|-------------|
| 경로 | `/NetDisk/j_son/internvl_35_v4_20260303/finetuning_20260303` | `/NetDisk/juyeon/DocSP/outputs/stage2_multinode` |
| Vision Encoder | InternViT-6B (24 layers) | InternViT-6B (24 layers) |
| LLM | Qwen3-8B (36 layers) | Qwen3-8B (36 layers) |
| DocSP Projector | No (`use_docsp=False`) | Yes (`use_docsp=True`) |
| num_image_token | 268 | 264 (256 patch + 8 DocSP spatial) |
| 추가 모듈 | SFE pool/crop convs, DFI | DocSP (FreqDecomp + MultiScale SFE + DAG-FI) |

---

## 2. 데이터셋 구성

| Dataset | 유형 | 언어 | 샘플 수 | 데이터 출처 |
|---------|------|------|---------|------------|
| **ChartQA** | Chart Understanding | English | 20 (human split) | 공식 test set |
| **DocVQA** | Document VQA | English | 20 (validation) | 공식 HuggingFace validation |
| **InfoVQA** | Infographic VQA | English | 20 (validation) | 공식 LMUData validation |
| **KDTCBench** | Document/Table/Chart | Korean | 20/240 | 공식 K-DTCBench |
| **KOFFVQA** | Open-ended VQA | Korean | 20/275 | 공식 maum-ai/KOFFVQA |

---

## 3. 결과 파일 구조

각 `{dataset}_{model}_20samples/` 디렉토리 내:
- `sample_XXX_vit_attention.png` - ViT 레이어별 CLS->Patch attention 히트맵
- `sample_XXX_info_flow.png` - LLM 레이어별 정보 흐름 4개 메트릭
- `sample_XXX_redundancy.png` - 이미지 토큰 코사인 유사도 매트릭스
- `sample_XXX_dashboard.png` - 종합 대시보드
- `aggregate_analysis.png` - 전체 샘플 집계 분석
- `answers_summary.json` - 질문/답변 요약

### 전체 디렉토리 목록

| # | 디렉토리 | 모델 | 데이터셋 | 파일 수 |
|---|---------|------|---------|---------|
| 1 | `chartqa_internvl_v4_ft_20samples/` | v4 FT | ChartQA | 81 PNGs |
| 2 | `chartqa_docsp_stage2_20samples/` | DocSP | ChartQA | 81 PNGs |
| 3 | `docvqa_internvl_v4_ft_20samples/` | v4 FT | DocVQA | 81 PNGs |
| 4 | `docvqa_docsp_stage2_20samples/` | DocSP | DocVQA | 81 PNGs |
| 5 | `infovqa_internvl_v4_ft_20samples/` | v4 FT | InfoVQA | 81 PNGs |
| 6 | `infovqa_docsp_stage2_20samples/` | DocSP | InfoVQA | 81 PNGs |
| 7 | `kdtcbench_internvl_v4_ft_20samples/` | v4 FT | KDTCBench | 81 PNGs |
| 8 | `kdtcbench_docsp_stage2_20samples/` | DocSP | KDTCBench | 81 PNGs |
| 9 | `koffvqa_internvl_v4_ft_20samples/` | v4 FT | KOFFVQA | 81 PNGs |
| 10 | `koffvqa_docsp_stage2_20samples/` | DocSP | KOFFVQA | 81 PNGs |

### 기존 데이터 (pretrained models, 이전 실험)

| # | 디렉토리 | 모델 | 데이터셋 |
|---|---------|------|---------|
| 11 | `chartqa_internvl_20samples/` | InternVL3.5-8B (pretrained) | ChartQA |
| 12 | `chartqa_qwen25vl_20samples/` | Qwen2.5-VL-7B (pretrained) | ChartQA |
| 13 | `chartqa_llavaSP_20samples/` | LLaVA-SP (pretrained) | ChartQA |

---

## 4. 분석 관점

### 4.1 Inter-layer Similarity
- LLM 레이어 간 이미지 토큰의 코사인 유사도 변화
- Cliff 위치: 정보 흐름이 급격히 변하는 레이어

### 4.2 L2 Norm
- 이미지 토큰의 L2 norm 변화 패턴
- DocSP의 Frequency Decomposition이 norm에 미치는 영향

### 4.3 Variance
- 이미지 토큰 표현의 분산 변화
- Variance 폭발 억제 여부

### 4.4 Token Redundancy
- 이미지 토큰 간 pairwise 코사인 유사도
- 높은 redundancy = 토큰 압축 여지

### 4.5 ViT Attention
- InternViT-6B의 레이어별 CLS->Patch attention 분포
- 데이터셋별 attention 패턴 차이 (chart vs document vs infographic vs Korean)

---

## 5. 데이터셋별 핵심 관찰 포인트

### ChartQA (English Charts)
- 두 모델의 기본 비교 대상 (이전 5-sample 분석에서 확장)
- DocSP의 variance 3000x 억제, L2 norm 5x 감소 패턴이 20 samples에서도 유지되는지

### DocVQA (English Documents)
- 문서 이미지에서의 정보 흐름 패턴
- OCR/텍스트 중심 이미지에서 DocSP spatial token의 역할

### InfoVQA (English Infographics)
- 복합적 정보(텍스트+그래프+아이콘)에서의 attention 분포
- DocSP의 구조적/의미적 분리가 infographic에서 효과적인지

### KDTCBench (Korean Document/Table/Chart)
- 한국어 문서에서의 모델 행동 차이
- 문서/테이블/차트 카테고리별 분석 가능

### KOFFVQA (Korean Open-ended VQA)
- 자연 이미지 + 한국어 질의에서의 정보 흐름
- DocSP가 비문서 이미지에서 어떤 패턴을 보이는지 (catastrophic forgetting 관련)

---

## 6. 스크립트 정보

| 스크립트 | 용도 |
|---------|------|
| `visualize_internvl_finetuned.py` | ChartQA 전용 시각화 |
| `visualize_multi_dataset.py` | 다중 데이터셋 시각화 (ChartQA/DocVQA/InfoVQA/KDTCBench/KOFFVQA) |
| `run_all_kicloud18.sh` | Kicloud18에서 전체 파이프라인 실행 |
| `config.py` | 경로 설정 |

---

*Generated: 2026-03-06*
