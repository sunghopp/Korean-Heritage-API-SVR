# AI ARS + TTS 변경 요약

## 기존

```text
Web Audio -> Whisper Jeju STT -> Tuned Gemini -> standard_text
```

## 변경

```text
Web Audio
  -> Whisper Jeju STT
  -> Tuned Gemini + Few-Shot Demo Scenario
       -> standard_text
       -> ars_reply_jeju
  -> Jeju VITS (ars_reply_jeju -> WAV)
  -> JSON + WAV Base64
```

## 변경 파일

- `api_server.py`: 전체 STT → Gemini → TTS 파이프라인, `/health`, `/tts`
- `ars_prompt.py`: Demo 시나리오 / Few-Shot 예시
- `tts_engine.py`: 학습한 single-speaker VITS Generator 추론
- `tts_config/jeju_vits.json`: 학습 당시 VITS config
- `models/tts/`: 선택한 `G_*.pth`를 `jeju_vits.pth`로 배치
- `Dockerfile`: pinned VITS runtime 설치 및 monotonic_align build
- `requirements.txt`: `google-genai`, TTS 의존성 추가
- `.github/workflows/deploy.yml`: Git LFS checkout + Cloud Run concurrency=1
- `.gitattributes`: `models/tts/*.pth` Git LFS

## 모델 파일

API 서버에 필요한 TTS weight는 Generator 하나입니다.

```text
G_XXXX.pth -> models/tts/jeju_vits.pth
```

Discriminator `D_*.pth`는 배포하지 않습니다.
