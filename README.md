# mood-music-ai
## 이미지 분위기(mood) 분석 기반 음악 추천 시스템

> 숭실대학교 컴퓨터비전응용(CVA) 수업 프로젝트 | 2025.06 | 유승주

Instagram URL을 입력하면 게시물 이미지의 분위기를 AI로 분석하고, 어울리는 음악 10곡을 자동 추천하는 엔드투엔드 서비스입니다.

---

## Motivation

SNS에 게시물을 올릴 때 배경음악을 설정하는 기능이 있지만, 대부분 유행곡 위주로 반복 추천됩니다.
사람은 이미지를 볼 때 **인물의 표정뿐 아니라 배경의 맥락까지 함께 고려**한다는 점에 착안해,
이를 AI로 구현하는 분위기 기반 음악 추천 시스템을 설계했습니다.

---

## Key Results

| 방식 | 정답률 |
|---|---|
| 전체 이미지 단일 분석 (baseline) | 0.33 |
| 인물/배경 분리 분석 (본 시스템) | **0.67** |

> **인물/배경 분리 분석이 baseline 대비 약 2배 높은 정확도**를 보였습니다.
> 최적 가중치: 인물(기본 CLIP) 0.8 + 배경(파인튜닝 CLIP) 0.2 → 최고 정답률 **0.6774**

---

## Research Questions

| RQ | 질문 | 결과 |
|---|---|---|
| RQ1 | 인물/배경 분리 vs 전체 이미지 — 어느 쪽이 분위기를 더 잘 추출하는가? | 분리 방식 우세 (0.67 vs 0.33) |
| RQ2 | 전신(YOLOv8) vs 얼굴(MTCNN) — 어느 방식이 감정을 더 잘 포착하는가? | YOLOv8 전신 방식 우세 (0.67 vs 0.33) |
| RQ3 | Zero-shot CLIP vs Fine-tuned CLIP — 배경 분위기 추출 성능 차이는? | 파인튜닝 CLIP이 배경 분위기 인식 개선 |
| RQ4 | 인물:배경 가중치 최적 비율은? | 8:2 (weight_person=0.8) |
| RQ5 | 최적 조합 파이프라인 구성 | 인물(기본 CLIP) + 배경(파인튜닝 CLIP) 8:2 |

---

## System Architecture

```
Instagram URL 입력
        ↓
instaloder로 이미지 다운로드
        ↓
YOLOv8 — 인물 영역 탐지 및 crop
        ↓
┌────────────────┬─────────────────────┐
│  인물 이미지   │    배경 이미지      │
│  기본 CLIP     │  파인튜닝 CLIP      │
│  mood 유사도   │  mood 유사도        │
└────────────────┴─────────────────────┘
        ↓ 가중 평균 (8:2)
    best_mood 추출
        ↓
Last.fm API — mood 태그 기반 음악 검색
        ↓
음악 10곡 + 링크 출력
```

---

## Methods

### 데이터셋
- **파인튜닝용**: AI HUB 감정 이미지 데이터셋, FER2013, Emotion6(배경 이미지 감정) — 총 111장 직접 라벨링
- **실험용**: Unsplash 수집 이미지 (31장, Labeled)
- **음악 매핑**: Last.fm Mood & Tag Dataset

### 모델
- **YOLOv8**: 인물 전신 탐지 및 배경 분리
- **CLIP (ViT-B/32)**: zero-shot 분위기 분석 (10개 mood 태그와 cosine similarity)
- **Fine-tuned CLIP**: 배경 이미지 111장으로 이미지 인코더 파인튜닝 (text 고정)

### mood 태그 (10개)
`happy`, `sad`, `peaceful`, `energetic`, `romantic`, `nostalgic`, `dark`, `bright`, `tense`, `playful`

---

## Tech Stack

| 분류 | 기술 |
|---|---|
| AI/ML | Python, PyTorch, CLIP (OpenAI), YOLOv8, MTCNN |
| 데이터 | Jupyter Notebook, pandas, matplotlib |
| API | Last.fm API, Instagram (instaloder) |
| 개발 환경 | Google Colab, Git |

---

## Project Structure

```
mood-music-ai/
├── RQ1_combined_vs_whole.ipynb      # 인물/배경 분리 vs 전체 이미지
├── RQ2_yolo_vs_mtcnn.ipynb          # YOLOv8 vs MTCNN 비교
├── RQ3_finetuned_clip.ipynb         # CLIP 파인튜닝
├── RQ4_weight_optimization.ipynb    # 가중치 최적화 실험
├── RQ5_best_pipeline.ipynb          # 최적 조합 파이프라인
├── service_demo.ipynb               # Instagram URL → 음악 추천 데모
├── data/
│   ├── train/                       # 파인튜닝용 라벨링 데이터 (111장)
│   └── test/                        # 실험용 이미지 (31장)
├── models/
│   └── clip_finetuned.pt            # 파인튜닝된 CLIP 모델 가중치
└── README.md
```

---

## How to Run

```bash
# 1. 환경 설치
pip install torch torchvision
pip install git+https://github.com/openai/CLIP.git
pip install ultralytics
pip install pylast  # Last.fm API

# 2. 데모 실행
jupyter notebook service_demo.ipynb
# Instagram URL 입력 → 음악 10곡 + 링크 자동 출력
```

---

## Results & Presentation

발표 자료(포스터): [`presentation.pdf`](./presentation.pdf)

실험 결과 요약:
- RQ1/RQ2: 인물/배경 분리 + YOLOv8 전신 탐지 방식이 일관되게 우수
- RQ3: 파인튜닝 CLIP이 배경의 미묘한 분위기 변화 포착 개선 (sensitive → gloomy 등)
- RQ4: 인물 가중치 0.8일 때 최고 정답률 0.6774 달성
- RQ5: 인물(기본 CLIP 0.8) + 배경(파인튜닝 CLIP 0.2) 조합이 최적

---

## Key Takeaways

- 인물 중심 분석(CLIP baseline)과 배경 중심 분석(CLIP fine-tuned)을 분리한 후, **가중 조합을 통해 분위기 예측 정확도를 크게 향상**시킬 수 있음
- 인물이 포함된 이미지에서는 분리 분석이 효과적이며, 인물이 없는 경우 배경 분석만으로도 충분한 감정 인식이 가능
- 최적 가중치는 상황에 따라 달라질 수 있으므로 실험적 도출이 필요

---

*숭실대학교 IT대학 컴퓨터학부 | 컴퓨터비전응용 수업 프로젝트*
