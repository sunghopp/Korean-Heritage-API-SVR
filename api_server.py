from __future__ import annotations

import asyncio
import base64
import logging
import os
import tempfile
import time
from pathlib import Path

import librosa
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from google import genai
from google.genai import types
from peft import PeftConfig, PeftModel
from pydantic import BaseModel, Field
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from ars_prompt import SYSTEM_INSTRUCTION, build_few_shot_contents
from dataset_logger import save_training_sample
from tts_engine import JejuVITSEngine

logger = logging.getLogger(__name__)


# ==========================================
# 1. FastAPI / environment
# ==========================================
app = FastAPI(title="Jeju AI ARS API", version="2.0.0")

cors_origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials="*" not in cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

LORA_MODEL_PATH = os.getenv("LORA_MODEL_PATH", "./whisper-jeju-lora-final")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "385248657749")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
GEMINI_TUNED_ENDPOINT = os.getenv(
    "GEMINI_TUNED_ENDPOINT",
    "projects/385248657749/locations/us-central1/endpoints/7571681821318971392",
)

TTS_CONFIG_PATH = os.getenv("TTS_CONFIG_PATH", "./tts_config/jeju_vits.json")
TTS_CHECKPOINT_PATH = os.getenv(
    "TTS_CHECKPOINT_PATH",
    "gs://malmoi-jeju-dataset-2026/tts/jeju_vits.pth",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INFERENCE_LOCK = asyncio.Lock()


class GeminiARSResult(BaseModel):
    standard_text: str = Field(description="입력 제주어를 자연스러운 표준어로 번역한 결과")
    ars_reply_jeju: str = Field(description="Demo 시나리오를 참고해 생성한 제주어 AI ARS 답변")


class TTSRequest(BaseModel):
    text: str


# ==========================================
# 2. Models: load once at process startup
# ==========================================
print(f"서버 구동 준비: STT 모델 적재 중... device={DEVICE}")
stt_config = PeftConfig.from_pretrained(LORA_MODEL_PATH)
base_model_name = stt_config.base_model_name_or_path
processor = WhisperProcessor.from_pretrained(base_model_name)
base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
stt_model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH).to(DEVICE).eval()

print("Gemini client 준비 중...")
gemini_client = genai.Client(
    vertexai=True,
    project=GCP_PROJECT_ID,
    location=GCP_LOCATION,
    http_options=types.HttpOptions(api_version="v1"),
)

print("Jeju VITS TTS 모델 적재 중...")
tts_model = None
_tts_error = None
try:
    tts_model = JejuVITSEngine(
        config_path=TTS_CONFIG_PATH,
        checkpoint_path=TTS_CHECKPOINT_PATH,
        device=DEVICE,
    )
except Exception as exc:  # Keep server diagnosable before checkpoint is copied.
    _tts_error = str(exc)
    print(f"⚠️ TTS 비활성화: {_tts_error}")

print("✅ 모델 적재 단계 완료")


# ==========================================
# 3. Pipeline helpers
# ==========================================
def transcribe_jeju(audio_path: str) -> str:
    speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
    inputs = processor(
        speech_array,
        sampling_rate=sampling_rate,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.inference_mode():
        predicted_ids = stt_model.generate(
            **inputs,
            language="ko",
            task="transcribe",
        )

    return processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def call_gemini_ars(jeju_text: str) -> GeminiARSResult:
    response = gemini_client.models.generate_content(
        model=GEMINI_TUNED_ENDPOINT,
        contents=build_few_shot_contents(jeju_text),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.15,
            top_p=0.8,
            max_output_tokens=512,
            response_mime_type="application/json",
            response_schema=GeminiARSResult,
        ),
    )

    if response.parsed is not None:
        if isinstance(response.parsed, GeminiARSResult):
            return response.parsed
        return GeminiARSResult.model_validate(response.parsed)

    if not response.text:
        raise RuntimeError("Gemini가 빈 응답을 반환했습니다.")
    return GeminiARSResult.model_validate_json(response.text)


def synthesize_ars_reply(text: str) -> bytes:
    if tts_model is None:
        raise RuntimeError(_tts_error or "TTS 모델이 초기화되지 않았습니다.")

    return tts_model.synthesize_wav(
        text,
        max_chars=int(os.getenv("TTS_MAX_CHARS", "45")),
        pause_ms=int(os.getenv("TTS_PAUSE_MS", "220")),
        tail_silence_ms=int(os.getenv("TTS_TAIL_SILENCE_MS", "350")),
        length_scale=float(os.getenv("TTS_LENGTH_SCALE", "1.10")),
        noise_scale=float(os.getenv("TTS_NOISE_SCALE", "0.667")),
        noise_scale_w=float(os.getenv("TTS_NOISE_SCALE_W", "0.35")),
    )


# ==========================================
# 4. Endpoints
# ==========================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "stt_loaded": True,
        "gemini_model": GEMINI_TUNED_ENDPOINT,
        "tts_loaded": tts_model is not None,
        "tts_checkpoint": TTS_CHECKPOINT_PATH,
        "tts_error": _tts_error,
    }


@app.post("/translate")
async def translate_audio(file: UploadFile = File(...)):
    """Single-call AI ARS pipeline.

    Web audio -> Whisper Jeju STT -> tuned Gemini -> Jeju VITS -> JSON.

    WAV bytes are base64-encoded because a single HTTP body cannot normally be
    both a JSON document and a raw WAV file at the same time. Frontend can turn
    audio_base64 back into a Blob(audio/wav) and play it immediately.
    """
    start_time = time.time()
    suffix = Path(file.filename or "input.wav").suffix or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_file_path = tmp.name

    try:
        # Model objects share CPU/GPU memory. Serialize demo requests to avoid
        # concurrent inference spikes on a single Cloud Run instance.
        async with INFERENCE_LOCK:
            jeju_text = await asyncio.to_thread(transcribe_jeju, temp_file_path)
            if not jeju_text:
                raise HTTPException(status_code=422, detail="STT 결과가 비어 있습니다.")

            gemini_result = await asyncio.to_thread(call_gemini_ars, jeju_text)

            try:
                await asyncio.to_thread(
                    save_training_sample,
                    audio_path=temp_file_path,
                    jeju_text=jeju_text,
                    standard_text=gemini_result.standard_text,
                )
            except Exception:
                logger.warning("데이터셋 저장 호출 실패", exc_info=True)

            if tts_model is None:
                raise HTTPException(
                    status_code=503,
                    detail=f"TTS 모델이 준비되지 않았습니다: {_tts_error}",
                )

            wav_bytes = await asyncio.to_thread(
                synthesize_ars_reply,
                gemini_result.ars_reply_jeju,
            )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"AI ARS 처리 실패: {exc}") from exc
    finally:
        try:
            os.remove(temp_file_path)
        except FileNotFoundError:
            pass

    end_time = time.time()
    return {
        "status": "success",
        "jeju_text": jeju_text,
        "standard_text": gemini_result.standard_text,
        "ars_reply_text": gemini_result.ars_reply_jeju,
        "audio_mime_type": "audio/wav",
        "audio_filename": "ars_reply.wav",
        "audio_sample_rate": tts_model.sample_rate if tts_model else 22050,
        "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
        "processing_time": round(end_time - start_time, 2),
    }


@app.post("/tts")
async def tts_only(request: TTSRequest):
    """Optional raw-WAV endpoint for debugging/frontend reuse."""
    if tts_model is None:
        raise HTTPException(status_code=503, detail=f"TTS 모델 미준비: {_tts_error}")
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="text가 비어 있습니다.")

    async with INFERENCE_LOCK:
        try:
            wav_bytes = await asyncio.to_thread(synthesize_ars_reply, request.text.strip())
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"TTS 생성 실패: {exc}") from exc

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": 'inline; filename="jeju_tts.wav"'},
    )
