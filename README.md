# Korean-Heritage-API-SRV — Jeju AI ARS

Ajou Univ. 26' Google AI Capstone Project

웹에서 받은 음성을 **제주어 STT → 사전 조정 Gemini → 제주어 VITS TTS** 순서로 처리하는 AI ARS API 서버입니다.

## 처리 흐름

```text
Web 음성 파일
   ↓
Whisper Small + Jeju LoRA
   ↓
제주어 STT Text
   ↓
Tuned Gemini + Few-Shot AI ARS Demo Prompt
   ├─ ① 표준어 번역
   └─ ② 제주어 AI ARS 답변
                ↓
        Jeju single-speaker VITS
                ↓
              WAV
                ↓
JSON Response
  - jeju_text
  - standard_text
  - ars_reply_text
  - audio_base64 (audio/wav)
```

## 이번 변경사항

### 1. Gemini Few-Shot AI ARS

기존에는 Gemini가 제주어를 표준어로 번역한 문자열 하나만 반환했습니다.

이제 `ars_prompt.py`의 Demo 시나리오와 Few-Shot 예시를 함께 전달하고 Gemini가 아래 두 값을 생성합니다.

```json
{
  "standard_text": "표준어 번역",
  "ars_reply_jeju": "제주어 AI ARS 답변"
}
```

Few-Shot 시나리오는 `ars_prompt.py`의 다음 두 값만 수정하면 됩니다.

```python
DEMO_SCENARIO
FEW_SHOT_EXAMPLES
```

Gemini 출력은 Pydantic response schema로 구조화합니다.

> 기존 `vertexai.generative_models`는 2026-06-24 이후 제거 대상이므로 현재 구현은 `google-genai` SDK를 사용합니다.

## 2. Jeju VITS TTS 연결

SageMaker에서 학습한 **Generator 가중치 `G_*.pth` 하나만** 아래 폴더에 추가합니다.

```text
models/tts/jeju_vits.pth
```

`D_*.pth`는 추론에 필요하지 않습니다.

선택한 checkpoint는 배포 시 파일명을 `jeju_vits.pth`로 통일하는 것을 권장합니다.

```bash
cp /path/to/G_XXXX.pth models/tts/jeju_vits.pth
```

VITS checkpoint가 GitHub 일반 파일 제한을 넘을 수 있으므로 `.gitattributes`에 Git LFS 설정을 추가했습니다.

```bash
git lfs install
git lfs track "models/tts/*.pth"
git add .gitattributes models/tts/jeju_vits.pth
```

GitHub Actions checkout도 `lfs: true`로 설정되어 있습니다.

TTS config는 학습 당시 사용한 값을 그대로 `tts_config/jeju_vits.json`에 포함했습니다.

Docker 빌드 시 원본 `jaywalnut310/vits` runtime을 `/opt/vits`에 가져오고 commit `2e561ba`로 고정합니다.

### TTS 끝 음절 보호

기존 테스트에서 짧은 문장도 마지막 발음이 잘리는 경우가 있었기 때문에 API용 TTS에는 다음 설정을 기본 적용합니다.

```text
max chunk chars    45
length_scale       1.10
noise_scale_w      0.35
chunk pause        220 ms
final tail silence 350 ms
end guard          enabled
```

긴 ARS 답변은 자동 분할 후 WAV 하나로 연결합니다.

## API

### `POST /translate`

기존과 동일하게 `multipart/form-data`의 `file`로 음성을 전송합니다.

예:

```bash
curl -X POST http://localhost:8080/translate \
  -F "file=@sample.wav"
```

응답 예:

```json
{
  "status": "success",
  "jeju_text": "상담원 연결해줍서",
  "standard_text": "상담원 연결해 주세요.",
  "ars_reply_text": "예, 상담원 연결을 도와드리쿠다. 잠시만 기다려줍서.",
  "audio_mime_type": "audio/wav",
  "audio_filename": "ars_reply.wav",
  "audio_sample_rate": 22050,
  "audio_base64": "UklGR...",
  "processing_time": 2.31
}
```

하나의 HTTP body에서 JSON과 raw WAV를 동시에 일반 JSON으로 보낼 수 없기 때문에 `/translate`에서는 WAV bytes를 Base64로 넣습니다.

프론트에서는 Base64를 `Blob([bytes], {type: 'audio/wav'})`로 변환해 바로 재생하면 됩니다.

### `POST /tts`

TTS만 테스트할 때 raw `audio/wav`를 바로 반환합니다.

```bash
curl -X POST http://localhost:8080/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"혼저옵서예. 무슨 일로 전화합신가?"}' \
  --output test.wav
```

### `GET /health`

STT/Gemini/TTS 로드 상태를 확인합니다.

```bash
curl http://localhost:8080/health
```

## 환경 변수

`.env.example` 참고.

핵심 값:

```bash
GCP_PROJECT_ID=385248657749
GCP_LOCATION=us-central1
GEMINI_TUNED_ENDPOINT=projects/385248657749/locations/us-central1/endpoints/7571681821318971392
TTS_CHECKPOINT_PATH=gs://malmoi-jeju-dataset-2026/tts/jeju_vits.pth
```

실제 Gemini tuned endpoint가 바뀌면 환경변수만 수정하면 됩니다.

## 로컬 Docker 실행

먼저 가중치를 넣습니다.

```text
models/tts/jeju_vits.pth
```

빌드:

```bash
docker build -t korean-heritage-api .
```

Google Cloud ADC가 설정된 개발 환경이라면 필요한 credential을 전달해 실행합니다. Cloud Run에서는 배포 서비스 계정의 Vertex AI 권한을 사용합니다.

```bash
docker run --rm -p 8080:8080 \
  -e GCP_PROJECT_ID=385248657749 \
  -e GCP_LOCATION=us-central1 \
  -e GEMINI_TUNED_ENDPOINT=projects/385248657749/locations/us-central1/endpoints/7571681821318971392 \
  -e TTS_CHECKPOINT_PATH=gs://malmoi-jeju-dataset-2026/tts/jeju_vits.pth \
  korean-heritage-api
```

## Cloud Run 배포 시 참고

현재 기존 서버와 동일하게 CPU PyTorch 설치를 유지했습니다. STT + VITS를 같은 요청에서 순차 실행하므로 데모에서는 **instance concurrency=1**을 권장합니다.

Cloud Run GPU를 사용할 경우 Docker/PyTorch를 CUDA 지원 이미지로 바꿔야 하며, 현재 CPU wheel 그대로는 `torch.cuda.is_available()`이 `False`입니다.

## 프로젝트 구조

```text
.
├── api_server.py
├── ars_prompt.py
├── tts_engine.py
├── tts_config/
│   └── jeju_vits.json
├── models/
│   └── tts/
│       ├── README.md
│       └── jeju_vits.pth       # 사용자가 추가
├── whisper-jeju-lora-final/
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── .env.example
```

## Gemini SDK

이 서버는 `google-genai` SDK를 사용하여 Vertex AI의 tuned Gemini endpoint를 호출합니다. 시스템 지시문과 Few-Shot turn을 함께 보내고 structured JSON response를 요청합니다.

## VITS runtime

TTS 구조는 원본 `jaywalnut310/vits`를 사용합니다. Dockerfile에서 upstream commit `2e561ba`로 고정하며, 학습된 `G_*.pth`는 별도 모델 artifact입니다.

## TTS 가중치: Google Cloud Storage 로딩

TTS Generator 가중치는 GitHub/Docker 이미지에 포함하지 않고 아래 GCS 객체를 사용합니다.

```text
gs://malmoi-jeju-dataset-2026/tts/jeju_vits.pth
```

API 프로세스가 시작되면 `tts_engine.py`가 해당 객체를 아래 임시 경로로 한 번 다운로드한 뒤 VITS 모델을 적재합니다.

```text
/tmp/jeju_vits.pth
```

기본 환경변수:

```bash
TTS_CHECKPOINT_PATH=gs://malmoi-jeju-dataset-2026/tts/jeju_vits.pth
TTS_CHECKPOINT_CACHE_PATH=/tmp/jeju_vits.pth
```

Cloud Run에서 사용하는 런타임 서비스 계정에는 버킷의 해당 객체를 읽을 수 있는 권한(`storage.objects.get`, 일반적으로 Storage Object Viewer 역할)이 필요합니다.

STT LoRA 로딩 방식은 이번 변경에서 수정하지 않았습니다.

## 학습 데이터셋 자동 저장: Google Cloud Storage 업로드

`POST /translate` 요청마다 사용자 발화 음성과 STT/번역 결과를 이후 모델 재학습용 데이터셋으로 GCS에 적재합니다 (`dataset_logger.py`).

```text
gs://malmoi-jeju-dataset-2026/dataset/extracted/Audio/{tier}/{id}.wav
gs://malmoi-jeju-dataset-2026/dataset/extracted/Text/{tier}/{id}.json
```

`{id}`는 요청마다 새로 생성되는 타임스탬프+랜덤 hex 키이며, 오디오/라벨 파일이 1:1로 짝지어집니다. 라벨 파일은 JSON 객체 하나로, 참조 데이터셋(`extracted/extracted/Text/Text/Label/*.json`)의 `utterance` 레벨 필드명(`form`, `standard_form`, `dialect_form`, `speaker_id`, `note`)을 재사용하고, 어절(공백 기준 단어) 단위로 쪼갠 `eojeolList`도 함께 담습니다.

`eojeolList`는 `jeju_text`(방언 원문)와 `standard_text`(Gemini가 생성한 문장 전체 표준어 번역)를 각각 공백 기준으로 어절 분리한 뒤 같은 순서로 위치 매칭한 것입니다. Gemini로부터 실제 어절 단위 정렬을 받지 않기 때문에 나온 휴리스틱으로, 두 문장의 어절 수가 다르면(조사 추가/삭제 등) 부정확할 수 있습니다 — 표준어 쪽 어절이 모자라면 `standard`/`isDialect`는 `null`로 남습니다.

### Confidence 기반 3단계 티어 분류

`{tier}`는 Whisper STT 결과의 confidence(토큰별 평균 확률, `api_server.py`의 `transcribe_jeju()`가 `compute_transition_scores`로 계산)에 따라 세 값 중 하나로 정해집니다 (`dataset_logger._confidence_tier`):

| confidence | tier | 의미 |
|---|---|---|
| `>= 0.9` | `auto_approved` | 사람 리뷰 없이 바로 학습에 쓸 수 있는 데이터 |
| `0.7 <= confidence < 0.9` | `needs_monitoring` | 즉시 리뷰하지는 않지만 관찰 대상 |
| `< 0.7` | `needs_review` | 사람이 직접 리뷰/라벨링해야 하는 데이터 |

라벨 JSON에는 `confidence`(0~1 소수, 소수점 4자리 반올림)와 `review_status`(위 tier 문자열) 필드가 함께 저장됩니다.

기본 환경변수:

```bash
DATASET_BUCKET=malmoi-jeju-dataset-2026
DATASET_AUDIO_PREFIX=dataset/extracted/Audio
DATASET_TEXT_PREFIX=dataset/extracted/Text
```

이 저장은 부가 기능(best-effort)입니다 — 업로드가 실패해도 `/translate` 응답(STT/Gemini/TTS 결과)에는 영향을 주지 않고 서버 로그에 warning만 남습니다.
