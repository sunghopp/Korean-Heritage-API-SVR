# Jeju VITS checkpoint

학습이 끝난 Generator checkpoint 하나만 이 폴더에 둡니다.

예:

```text
models/tts/jeju_vits.pth
```

Cloud Run/로컬 실행 시 환경변수로 지정합니다.

```bash
TTS_CHECKPOINT_PATH=/app/models/tts/jeju_vits.pth
```

`D_*.pth`는 추론에 필요하지 않습니다.
