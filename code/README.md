# Mood-based Music Recommendation via CLIP Analysis

본 프로젝트는 이미지를 분석하여 **인물**과 **배경**의 분위기(mood)를 분리 추출한 뒤, 이를 기반으로 적절한 음악을 추천하는 시스템입니다.  
OpenAI의 CLIP 모델을 활용하여 Instagram 게시글의 이미지의 분위기를 예측하고, Last.fm의 태그 기반 데이터를 통해 음악과 분위기를 매핑합니다.

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

`run_final.py` 실행 전, `.env.example`를 복사해 `.env`를 만들고 키를 입력하세요.

```bash
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
├── data/                  # 실험용 이미지 저장 폴더 (공통 이미지들은 rq245_ 접두어로 명시)
├── results/               # CSV 및 시각화 결과 저장 (rq245 관련 결과 다수 포함)
├── src/                   # 분석/전처리/시각화 코드
│   ├── analysis/          # CLIP 분석, 결합, 모델 로딩 등
│   ├── preprocess/        # YOLO, MTCNN 기반 전처리
│   ├── visualize/         # match rate 및 mood 분포 시각화
│   └── data/              # gdown을 통한 Google Drive 다운로드
├── run_rq1.py             # RQ1: 인물+배경 vs 전체
├── run_rq2.py             # RQ2: YOLO vs MTCNN
├── run_rq4.py             # RQ4: 가중치 변화 분석
├── run_rq5.py             # RQ5: 모든 조합 실험
├── run_final.py           # 실제 Instagram 게시물 기반 음악 추천 실행
├── run_rq3.ipynb          # RQ3: CLIP fine-tuning 성능 비교 (Colab 전용)
├── requirements.txt       # 필수 패키지 명세
├── README.md              # 실행 및 구조 설명
└── clip_finetuned.pth     # Colab에서 학습한 fine-tuned CLIP 모델 (RQ3,4,5에서 사용)
```

### 2. 데이터 및 결과 디렉토리

| 디렉토리 이름        | 설명                                                       |
| -------------------- | ---------------------------------------------------------- |
| `data/rq*/`          | RQ별 입력 이미지 및 YOLO,MTCNN 처리 결과 저장              |
| `results/rq*/`       | RQ별 분석 결과 CSV 및 시각화 이미지 저장                   |
| `results/final/`     | 최종 음악 추천 결과 및 mood 분포 시각화 이미지 저장        |
| `clip_finetuned.pth` | 파인튜닝된 모델 가중치 파일(Colab에서 학습 후 로컬 저장됨) |

### 3. 실행 디렉토리

| 디렉토리 이름     | 설명                                   |
| ----------------- | -------------------------------------- |
| `run_*.py`        | 주요 실행 스크립트                     |
| `src/models/`     | CLIP 모델 로딩, 파인튜닝 모델 불러오기 |
| `src/preprocess/` | YOLO, MTCNN기반 전처리                 |
| `src/analysis/`   | 유사도 분석, 결합 가중치 분석 등       |
| `src/visualize/`  | match rate 비교 시각화                 |

---

## 📁 데이터 다운로드

일부 실험에 필요한 이미지 데이터는 Google Drive에 업로드되어 있으며, 자동으로 다운로드되도록 구성되어 있습니다.

- 📂 모델(pth) + 파인튜닝 train/test 데이터(통합): [구글드라이브0](https://drive.google.com/drive/folders/1Zkn7TOXLStHEmJk9USBV6Z3jJ88LNajh?usp=drive_link)

- 📂 RQ1 전용 이미지 : [구글드라이브1](https://drive.google.com/drive/folders/14ZmYrfB8Uu9IiZzGZeLZ_QEHFTVdJd9y?usp=drive_link)

- 📂 RQ2, RQ4, RQ5 공통 이미지 : [구글드라이브2](https://drive.google.com/drive/folders/106CLtANP_dGKeCsyu9GlsX8kgjxFsSsX?usp=drive_link)

- 📂 RQ3 훈련용, 실험용 이미지 : [구글드라이브3](https://drive.google.com/drive/folders/1obKzdMPidXG5O2ojlinLugzd0ckfn37B?usp=sharing),
  [구글드라이브3](https://drive.google.com/drive/folders/19dhOVR8c2aNWi2qMRCkhV6R49kDDMQWF?usp=drive_link)

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

## 🔐 포트폴리오 공개 가이드

- `instagram_samples/` 원본 이미지/텍스트는 공개 저장소에서 제외하는 것을 권장합니다.
- `clip_finetuned.pth`와 파인튜닝 train/test 이미지는 저장소에 직접 포함하지 않고, 아래 Drive 링크로 제공하는 방식을 권장합니다.
  - https://drive.google.com/drive/folders/1Zkn7TOXLStHEmJk9USBV6Z3jJ88LNajh?usp=drive_link
- 저장소 클론 후 실행 시에는 필요한 파일을 직접 준비해 아래 위치에 두면 됩니다.
  - 샘플 이미지: `instagram_samples/`
  - 파인튜닝 모델: `clip_finetuned.pth`
