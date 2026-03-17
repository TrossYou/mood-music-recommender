# Mood-based Music Recommendation via CLIP Analysis

> 숭실대학교 컴퓨터비전응용(CVA) 수업 프로젝트 | 2025.06 | 유승주 (TrossYou)

본 프로젝트는 이미지를 분석하여 **인물**과 **배경**의 분위기(mood)를 분리 추출한 뒤, 이를 기반으로 적절한 음악을 추천하는 시스템입니다.  
OpenAI의 CLIP 모델을 활용하여 Instagram 게시글의 이미지의 분위기를 예측하고, Last.fm의 태그 기반 데이터를 통해 음악과 분위기를 매핑합니다.

---

## Key Results

| 방식                                | 정답률   |
| ----------------------------------- | -------- |
| 전체 이미지 단일 분석 (baseline)    | 0.33     |
| **인물/배경 분리 분석 (본 시스템)** | **0.67** |

> 인물/배경 분리 방식이 baseline 대비 **약 2배** 높은 정확도를 달성했습니다.  
> 최적 가중치: 인물(기본 CLIP) 0.8 + 배경(파인튜닝 CLIP) 0.2 → 최고 정답률 **0.6774**

---

## Research Questions

| RQ  | 질문                              | 결과                                          |
| --- | --------------------------------- | --------------------------------------------- |
| RQ1 | 인물/배경 분리 vs 전체 이미지     | 분리 방식 우세 (0.67 vs 0.33)                 |
| RQ2 | 전신(YOLOv8) vs 얼굴(MTCNN)       | YOLOv8 전신 우세 (0.67 vs 0.33)               |
| RQ3 | Zero-shot CLIP vs Fine-tuned CLIP | 파인튜닝 CLIP이 배경 분위기 인식 개선         |
| RQ4 | 인물:배경 최적 가중치 비율        | 8:2 (weight_person=0.8)                       |
| RQ5 | 최적 조합 파이프라인              | 인물(기본 CLIP 0.8) + 배경(파인튜닝 CLIP 0.2) |

---

## System Flow

```
Instagram URL 입력
        ↓
Instaloader로 이미지 다운로드
        ↓
YOLOv8 — 인물 영역 탐지 및 crop / 배경 마스킹 분리
        ↓
┌─────────────────────┬─────────────────────┐
│   인물 이미지       │    배경 이미지      │
│   기본 CLIP         │  파인튜닝 CLIP      │
│   mood 유사도 계산  │  mood 유사도 계산   │
└─────────────────────┴─────────────────────┘
        ↓ 가중 평균 (8:2)
    best_mood 추출
        ↓
Last.fm API — mood 태그 기반 음악 검색
        ↓
음악 10곡 + 링크 출력
```

---

## 🔧 실행 방법

### 1. 가상환경 생성 및 패키지 설치

```bash
# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
# macOS / Linux:
source venv/bin/activate
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
venv\Scripts\activate.bat

# 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt
```

### 1-1. 환경 변수 설정 (Last.fm API Key)

`run_final.py` 실행 전, `code/.env.example`를 복사해 `code/.env`를 만든 뒤 키를 입력하세요.

```bash
cd code
cp .env.example .env
# .env 파일에서 LASTFM_API_KEY 값을 본인 키로 수정
```

### 2. 인터프리터 선택 (VSCode 사용자용)

> Python 가상환경을 만든 후, VSCode에서 인터프리터를 직접 선택해줘야 할 수 있습니다.

1. `Ctrl + Shift + P` (또는 `Cmd + Shift + P`) → "Python: Select Interpreter" 입력
2. 목록에서 `./venv/bin/python` (또는 Windows의 경우 `.\venv\Scripts\python.exe`) 선택 (Recommended)

---

## 🚀 파일 설명 및 실행 방법

| 파일명        | 설명                                                                            |
| ------------- | ------------------------------------------------------------------------------- |
| run_rq1.py    | RQ1 실험: 인물+배경 vs 전체 이미지 CLIP 분석                                    |
| run_rq2.py    | RQ2 실험: YOLO vs MTCNN 인물 감지 성능 비교                                     |
| run_rq4.py    | RQ4 실험: 인물 vs 배경 가중 평균 조절 (인물:YOLO, 기본CLIP, 배경:finetunedCLIP) |
| run_rq5.py    | RQ5 실험: 다양한 모델 조합 종합 비교                                            |
| run_final.py  | Instagram 게시물로부터 mood 분석 후 음악 추천 실행                              |
| run_rq3.ipynb | Colab에서 fine-tuning 후 RQ3 실험 수행 (로컬 실행 불가)                         |

```bash
python3 run_rq1.py
python3 run_rq2.py
python3 run_rq4.py
python3 run_rq5.py
```

```bash
python3 run_final.py
```

📌 run_final.py instagram 예시:

- https://www.instagram.com/p/DKqZL78T-_m/
- https://www.instagram.com/p/DKpBWWphOzY/

(인스타그램 정책 상, 일부 링크는 다운이 안될 수 있음)

---

## ⚠️ 특이사항

- python 대신 python3 사용 권장 (특히 macOS).
- run_rq3.ipynb는 Google Drive 마운트가 필요하며, 다른 계정에서는 실행이 제한될 수 있습니다.

---

## 📁 디렉토리

### 1. 디렉토리 구조

```bash
.
├── README.md                        # 포트폴리오/프로젝트 개요
├── report.pdf                       # 프로젝트 보고서
└── code/
        ├── README.md                    # 실행 및 상세 구조 설명
        ├── requirements.txt             # 필수 패키지 명세
        ├── .env.example                 # 환경변수 예시
        ├── run_rq1.py                   # RQ1 실험
        ├── run_rq2.py                   # RQ2 실험
        ├── run_rq3.ipynb                # RQ3 실험 (Colab)
        ├── run_rq4.py                   # RQ4 실험
        ├── run_rq5.py                   # RQ5 실험
        ├── run_final.py                 # Instagram 기반 최종 추천 실행
        ├── data/                        # 실험 입력 이미지 및 전처리 결과
        ├── results/                     # RQ별 CSV/시각화 결과
        ├── src/                         # 분석/전처리/시각화 코드
        ├── instagram_samples/           # 인스타 샘플(포트폴리오 공개 시 원본 제외 권장)
        └── clip_finetuned.pth           # 파인튜닝 모델 가중치(외부 링크 제공 권장)
```

### 2. 데이터 및 결과 디렉토리

| 디렉토리 이름             | 설명                                                       |
| ------------------------- | ---------------------------------------------------------- |
| `code/data/rq*/`          | RQ별 입력 이미지 및 YOLO, MTCNN 처리 결과 저장             |
| `code/results/rq*/`       | RQ별 분석 결과 CSV 및 시각화 이미지 저장                   |
| `code/results/final/`     | 최종 음악 추천 결과 및 mood 분포 시각화 이미지 저장        |
| `code/clip_finetuned.pth` | 파인튜닝된 모델 가중치 파일(Colab에서 학습 후 로컬 저장됨) |

### 3. 실행 디렉토리

| 디렉토리 이름          | 설명                                   |
| ---------------------- | -------------------------------------- |
| `code/run_*.py`        | 주요 실행 스크립트                     |
| `code/src/models/`     | CLIP 모델 로딩, 파인튜닝 모델 불러오기 |
| `code/src/preprocess/` | YOLO, MTCNN 기반 전처리                |
| `code/src/analysis/`   | 유사도 분석, 결합 가중치 분석 등       |
| `code/src/visualize/`  | match rate 비교 시각화                 |

---

## 📁 데이터 다운로드

일부 실험에 필요한 이미지 데이터는 Google Drive에 업로드되어 있으며, 자동으로 다운로드되도록 구성되어 있습니다.

- 📂 모델(pth) + 파인튜닝 train/test 데이터(통합): [구글드라이브0](https://drive.google.com/drive/folders/1Zkn7TOXLStHEmJk9USBV6Z3jJ88LNajh?usp=drive_link)

- 📂 RQ1 전용 이미지: [구글드라이브1](https://drive.google.com/drive/folders/14ZmYrfB8Uu9IiZzGZeLZ_QEHFTVdJd9y?usp=drive_link)
- 📂 RQ2, RQ4, RQ5 공통 이미지: [구글드라이브2](https://drive.google.com/drive/folders/106CLtANP_dGKeCsyu9GlsX8kgjxFsSsX?usp=drive_link)
- 📂 RQ3 훈련용 이미지: [구글드라이브3](https://drive.google.com/drive/folders/1obKzdMPidXG5O2ojlinLugzd0ckfn37B?usp=sharing)
- 📂 RQ3 실험용 이미지: [구글드라이브4](https://drive.google.com/drive/folders/19dhOVR8c2aNWi2qMRCkhV6R49kDDMQWF?usp=drive_link)

---

## 💡 사용된 기술 및 모델 파일

- OpenAI CLIP: 이미지와 텍스트 간의 의미적 유사도 분석
- Ultralytics YOLOv8: 인물 검출 (Object Detection 기반)
- Last.fm API: 분위기(mood) 키워드 기반 음악 트랙 검색
- Instaloader: Instagram 게시글에서 이미지 수집
- YOLO 모델 파일: yolov8n.pt
- CLIP 파인튜닝 모델: clip_finetuned.pth (Colab에서 학습한 후 포함됨)
- Pandas, Matplotlib

---

_숭실대학교 IT대학 컴퓨터학부 | 컴퓨터비전응용 수업 프로젝트 | 2025_
