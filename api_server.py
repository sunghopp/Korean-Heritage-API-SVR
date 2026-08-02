import torch
import librosa
import os
import time
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel, PeftConfig
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# ==========================================
# 1. FastAPI 앱 및 설정
# ==========================================
app = FastAPI(title="Jeju Dialect Translator API")

# 프론트엔드(웹)에서 API를 호출할 수 있도록 CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 실제 서비스 시에는 웹 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LORA_MODEL_PATH = "./whisper-jeju-lora-final"
GCP_PROJECT_ID = "385248657749"
GCP_LOCATION = "asia-northeast3"
ENDPOINT_PATH = "projects/385248657749/locations/us-central1/endpoints/7571681821318971392"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. 모델 로드 (서버 가동 시 1회만 실행)
# ==========================================
print("서버 구동 준비: 모델을 메모리에 적재합니다...")
config = PeftConfig.from_pretrained(LORA_MODEL_PATH)
base_model_name = config.base_model_name_or_path
processor = WhisperProcessor.from_pretrained(base_model_name)
base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
stt_model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH).to(DEVICE).eval()

vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
system_prompt = (
    "당신은 제주 방언을 표준어로 정확하게 번역하는 전문 번역가입니다. "
    "사용자가 제주어 텍스트를 입력하면, 부연 설명 없이 오직 '표준어로 번역된 결과'만 텍스트로 출력하세요."
)
translator_model = GenerativeModel(model_name=ENDPOINT_PATH, system_instruction=[system_prompt])
generation_config = GenerationConfig(temperature=0.1, top_p=0.8, max_output_tokens=512)
print("✅ 모델 적재 완료!")

# ==========================================
# 3. API 엔드포인트 구현
# ==========================================
@app.post("/translate")
async def translate_audio(file: UploadFile = File(...)):
    """웹에서 전달받은 음성 파일을 처리하여 JSON으로 반환"""
    start_time = time.time()
    
    # 1. 업로드된 파일을 임시 저장
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        # 2. STT (음성 -> 제주어)
        speech_array, sampling_rate = librosa.load(temp_file_path, sr=16000)
        inputs = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            predicted_ids = stt_model.generate(
                **inputs,
                language="ko",
                task="transcribe"
            )
            
        jeju_text = processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()
        
        # 3. LLM (제주어 -> 표준어)
        response = translator_model.generate_content(jeju_text, generation_config=generation_config)
        standard_text = response.text.strip()
        
    finally:
        # 처리가 끝난 후 임시 파일 삭제
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
    end_time = time.time()
    
    # 4. JSON 형태로 프론트엔드에 응답
    return {
        "status": "success",
        "jeju_text": jeju_text,
        "standard_text": standard_text,
        "processing_time": round(end_time - start_time, 2)
    }
