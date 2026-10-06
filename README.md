# Korean Heritage API Server | Jeju AI ARS

![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-inference-ee4c2c?logo=pytorch&logoColor=white) ![Google Cloud](https://img.shields.io/badge/Google%20Cloud-Vertex%20AI%20%7C%20GCS-4285F4?logo=googlecloud&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-ready-2496ed?logo=docker&logoColor=white)

> 제주어 음성을 인식하고 민원 답변을 생성해 제주어 음성으로 반환하는 FastAPI 서버입니다.

## 프로젝트 목적

Web의 음성 요청을 Whisper 제주어 LoRA, Vertex AI Gemini와 Jeju VITS로 처리합니다. 요청 음성과 라벨은 GCS, 대시보드 샘플은 Firestore에 기록합니다.

## 핵심 기능

- POST /translate: 제주어 STT, 표준어 번역, 제주어 ARS 답변, WAV 합성.
- POST /tts: 텍스트를 WAV로 합성.
- RAG_CORPUS가 설정되면 RAG Engine에서 안내 자료를 검색합니다. 검색을 못 쓰면 Few-Shot으로 처리합니다.
- 요청마다 브라우저의 최근 history 최대 5턴을 사용하며 서버 세션은 없습니다.
- GCS에 발화와 라벨 저장, Firestore에 대시보드 메타데이터 저장.
- 대시보드 통계·샘플·오디오 조회 및 승인·거부·라벨 수정 API.
- STT LoRA·TTS checkpoint GCS 로딩, CUDA 자동 사용 및 CPU fallback.

## 아키텍처

~~~mermaid
sequenceDiagram
  participant Web
  participant API as FastAPI
  participant STT as Whisper + Jeju LoRA
  participant RAG as Vertex AI RAG
  participant Gemini as Tuned Gemini
  participant GCS
  participant FS as Firestore
  participant TTS as Jeju VITS
  Web->>API: POST /translate (file, history)
  API->>STT: 음성 전사
  API->>RAG: 설정된 경우 검색
  alt 검색 결과 사용 가능
    RAG-->>API: 안내 예시
  else 미설정·검색 실패·유효 결과 없음
    API->>API: Few-Shot prompt
  end
  API->>Gemini: 질문·history
  Gemini-->>API: 번역·답변
  API->>GCS: 음성·라벨 (best effort)
  API->>FS: 샘플 metadata
  API->>TTS: 답변 합성
  TTS-->>API: WAV
  API-->>Web: JSON + Base64 WAV
~~~

요청 추론은 인스턴스에서 직렬화됩니다. 저장 실패는 음성 응답을 중단하지 않습니다.

## 기술 스택

FastAPI, Uvicorn, Pydantic, Whisper, PEFT, PyTorch, Transformers, librosa, Google Gen AI SDK, Vertex AI RAG Engine, Jeju VITS, Cloud Storage, Firestore, Docker, Cloud Run.

## 설치 및 실행

Python 3.11과 Google Cloud ADC가 필요합니다.

~~~bash
python -m pip install -r requirements.txt
python -m uvicorn api_server:app --host 0.0.0.0 --port 8080
~~~

~~~bash
docker build -t korean-heritage-api .
docker run --rm -d --name korean-heritage-api -p 8080:8080 -e PORT=8080 korean-heritage-api
~~~

컨테이너 실행 후:

~~~bash
curl http://localhost:8080/health
~~~

서버 시작 때 STT를 적재하며 기본 LoRA·TTS checkpoint는 GCS에서 읽습니다. Dockerfile은 고정한 VITS runtime을 빌드합니다.

## 환경 변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| PORT | 8080 (Docker) | 서버 포트 |
| GCP_PROJECT_ID, GCP_LOCATION | 385248657749, us-central1 | Vertex AI |
| GEMINI_TUNED_ENDPOINT | 코드 기본 endpoint | Gemini endpoint |
| CORS_ORIGINS | * | 허용 origin 목록 |
| LORA_MODEL_PATH | 운영 GCS prefix | STT adapter 경로 |
| LORA_MODEL_CACHE_PATH | /tmp/whisper-jeju-lora-final | adapter cache |
| TTS_CONFIG_PATH | ./tts_config/jeju_vits.json | VITS config |
| TTS_CHECKPOINT_PATH | GCS의 tts/jeju_vits.pth | TTS 가중치 |
| TTS_CHECKPOINT_CACHE_PATH | /tmp/jeju_vits.pth | TTS cache |
| VITS_ROOT | /opt/vits (Docker) | VITS runtime |
| TTS_MAX_CHARS, TTS_PAUSE_MS, TTS_TAIL_SILENCE_MS | 45, 220, 350 | 분할·무음 |
| TTS_LENGTH_SCALE, TTS_NOISE_SCALE, TTS_NOISE_SCALE_W, TTS_SEED | 1.10, 0.667, 0.35, 1234 | 합성값 |
| RAG_CORPUS, RAG_TOP_K, RAG_DISTANCE_THRESHOLD | 빈 값, 3, 0.25 | 선택 검색 |
| DATASET_BUCKET | malmoi-jeju-dataset-2026 | GCS bucket |
| DATASET_AUDIO_PREFIX, DATASET_TEXT_PREFIX | dataset/extracted/Audio, dataset/extracted/Text | 음성·라벨 경로 |
| DATASET_FIRESTORE_COLLECTION | dataset_samples | Firestore collection |

## API 및 데이터 흐름

| Method | 경로 | 동작 |
|---|---|---|
| GET | /health | 장치와 모델 상태 |
| POST | /translate | multipart file/history, JSON 결과 |
| POST | /tts | JSON text, raw audio/wav |
| GET | /dataset/stats | Firestore 전체·상태별 수 |
| GET | /dataset/samples?limit=20&offset=0 | 샘플 페이지, limit 최대 100 |
| GET | /dataset/audio/{sample_id} | GCS WAV |
| PATCH | /dataset/samples/{sample_id} | approved/rejected 및 선택 라벨 수정 |

응답은 전사, 번역, 답변, Base64 WAV, sample rate, 처리 시간을 포함합니다. 오디오·라벨은 GCS에 저장하고 metadata는 Firestore에 기록합니다. STT와 번역 confidence가 모두 0.8 이상이면 상태는 approved, 아니면 pending입니다.

## 디렉터리 구조

~~~text
.
├── api_server.py
├── ars_prompt.py
├── dataset_logger.py
├── dataset_dashboard.py
├── gcs_model_loader.py
├── tts_engine.py
├── tts_config/jeju_vits.json
├── models/tts/README.md
├── whisper-jeju-lora-final/README.md
├── tests/
├── requirements.txt
└── Dockerfile
~~~

## 학습 및 평가 지표

Confidence는 데이터 초기 상태 분류 기준이지 모델 성능 점수가 아닙니다. 발표 자료는 카카오브레인 제주어 200개 샘플 기준 평가를 기재했지만 수치 점수와 재현 코드가 없습니다. WER·BLEU 승격 조건은 [학습 저장소](https://github.com/sunghopp/Korean-Heritage-FLYWHEEL-TRAIN)에서 관리합니다.

## 주의사항

- translate, tts, 데이터셋 API에는 인증·권한 검사가 없습니다. 공개 배포 전 데이터 조회와 수정 권한을 제한해야 합니다.
- CORS 기본값은 전체 origin 허용입니다. 운영 시 제한하세요.
- GCS·Firestore 권한이 필요하며 RAG를 켤 경우 corpus 접근 권한도 필요합니다.
- TTS 적재가 실패하면 서버는 진단용으로 시작할 수 있지만 TTS 응답은 실패합니다.
- 추론 직렬화와 모델 메모리를 고려해 Cloud Run 동시성을 설정하세요.
