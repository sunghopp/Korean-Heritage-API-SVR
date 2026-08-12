"""Jeju single-speaker VITS inference used by the API server.

The trained Generator checkpoint (G_*.pth) is the only model artifact that
needs to be copied into this repository. VITS runtime source is installed at
/opt/vits by Dockerfile from the pinned upstream jaywalnut310/vits commit.
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import unicodedata
import wave
from pathlib import Path
from typing import List

import numpy as np
import torch


VITS_ROOT = Path(os.getenv("VITS_ROOT", "/opt/vits"))
if str(VITS_ROOT) not in sys.path:
    sys.path.insert(0, str(VITS_ROOT))

try:
    import commons  # type: ignore
    from models import SynthesizerTrn  # type: ignore
except ModuleNotFoundError as exc:
    raise RuntimeError(
        f"VITS runtime를 찾지 못했습니다: {VITS_ROOT}. "
        "Dockerfile로 빌드했는지 확인하세요."
    ) from exc


# Must match text/symbols.py generated during this Jeju VITS training.
_PAD = "_"
_PUNCTUATION = ';:,.!?—…"\'() '
_CHOSEONG = "".join(chr(i) for i in range(0x1100, 0x1113))
_JUNGSEONG = "".join(chr(i) for i in range(0x1161, 0x1176))
_JONGSEONG = "".join(chr(i) for i in range(0x11A8, 0x11C3))
_CHARS = list(dict.fromkeys(list(_PUNCTUATION + _CHOSEONG + _JUNGSEONG + _JONGSEONG)))
SYMBOLS = [_PAD] + _CHARS
SYMBOL_TO_ID = {symbol: i for i, symbol in enumerate(SYMBOLS)}


# Same modernization rules used in training preprocessing.
EXACT_REPLACEMENTS = [
    ("ᄁᆯᅩ앙근에", "꼴앙근에"),
    ("ᄁᆯᅩ아", "꼴아"),
    ("까ᅌᅨ", "까예"),
    ("ᄋᆢ라", "여러"),
    ("ᄋᆢ 섯", "여섯"),
    ("ᄋᆢ섯", "여섯"),
    ("ᄋᆢ솟", "여섯"),
    ("ᄋᆢ덥", "여덟"),
    ("ᄋᆢ답", "여덟"),
    ("ᄋᆢ돕", "여덟"),
    ("ᄋᆢ름", "여름"),
    ("ᄋᆢ물", "여물"),
    ("ᄋᆢᆯ", "열"),
    ("ᄋᆢ든", "여든"),
    ("ᄋᆢ끄", "엮"),
]


def modernize_jeju_text(text: str) -> str:
    text = str(text)
    for old, new in EXACT_REPLACEMENTS:
        text = text.replace(old, new)
    text = text.replace("ᄋᆢ", "여")
    text = text.replace("ᅌ", "ㅇ")
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def to_vits_text(text: str) -> str:
    return unicodedata.normalize("NFD", modernize_jeju_text(text))


def _sequence(text: str, add_blank: bool) -> torch.LongTensor:
    ids = [SYMBOL_TO_ID[c] for c in to_vits_text(text) if c in SYMBOL_TO_ID]
    if not ids:
        raise ValueError("TTS로 변환할 수 있는 한글/문장부호가 없습니다.")
    if add_blank:
        ids = commons.intersperse(ids, 0)
    return torch.LongTensor(ids)


def _load_state_dict(checkpoint_path: Path):
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")

    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise RuntimeError("지원하지 않는 VITS checkpoint 형식입니다.")

    # Defensive compatibility with checkpoints saved from DDP wrappers.
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    return state


def _split_long_text(text: str, max_chars: int) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []

    sentences = re.findall(r"[^.!?。！？]+[.!?。！？]?", text)
    chunks: List[str] = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            chunks.append(sentence)
            continue

        # Prefer clause and whitespace boundaries.
        remaining = sentence
        while len(remaining) > max_chars:
            window = remaining[: max_chars + 1]
            cut = max(
                window.rfind(","),
                window.rfind("，"),
                window.rfind(";"),
                window.rfind(":"),
                window.rfind(" "),
            )
            if cut < max_chars // 2:
                cut = max_chars
            else:
                cut += 1
            chunks.append(remaining[:cut].strip())
            remaining = remaining[cut:].strip()
        if remaining:
            chunks.append(remaining)

    return chunks


def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


class JejuVITSEngine:
    """Load one Generator checkpoint and synthesize ARS replies."""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str | None = None,
    ):
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"TTS config가 없습니다: {self.config_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"TTS checkpoint가 없습니다: {self.checkpoint_path}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        with self.config_path.open("r", encoding="utf-8") as f:
            self.hps = json.load(f)

        data = self.hps["data"]
        train = self.hps["train"]
        model_cfg = self.hps["model"]

        self.sample_rate = int(data["sampling_rate"])
        self.hop_length = int(data["hop_length"])
        self.add_blank = bool(data.get("add_blank", True))

        self.model = SynthesizerTrn(
            len(SYMBOLS),
            int(data["filter_length"]) // 2 + 1,
            int(train["segment_size"]) // int(data["hop_length"]),
            **model_cfg,
        ).to(self.device).eval()

        state = _load_state_dict(self.checkpoint_path)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "TTS checkpoint와 config/symbol 구성이 일치하지 않습니다. "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )

        # Deterministic-ish demo behavior. SDP still uses noise_scale_w below.
        torch.manual_seed(int(os.getenv("TTS_SEED", "1234")))

    def _synthesize_chunk(
        self,
        text: str,
        length_scale: float,
        noise_scale: float,
        noise_scale_w: float,
        guard_extra_frames: int,
    ) -> np.ndarray:
        # End guard helps protect the last real syllable. We trim guard frames
        # using attention after synthesis, retaining a few release frames.
        main = _sequence(text, self.add_blank)
        guarded = _sequence(text + " .", self.add_blank)
        main_seq_len = len(main)

        x = guarded.unsqueeze(0).to(self.device)
        x_lengths = torch.LongTensor([x.size(1)]).to(self.device)

        with torch.inference_mode():
            output, attn, _, _ = self.model.infer(
                x,
                x_lengths,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                max_len=None,
            )

        audio = output[0, 0].float().cpu().numpy()
        attention = attn[0, 0].float().cpu()

        main_frames = int(round(float(attention[:, :main_seq_len].sum().item())))
        keep_frames = min(
            int(attention.shape[0]),
            main_frames + max(0, guard_extra_frames),
        )
        keep_samples = keep_frames * self.hop_length
        if keep_samples > 0:
            audio = audio[: min(len(audio), keep_samples)]

        return np.clip(audio, -1.0, 1.0).astype(np.float32)

    def synthesize_wav(
        self,
        text: str,
        *,
        max_chars: int = 45,
        pause_ms: int = 220,
        tail_silence_ms: int = 350,
        length_scale: float = 1.10,
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.35,
        guard_extra_frames: int = 2,
    ) -> bytes:
        chunks = _split_long_text(text, max_chars=max_chars)
        if not chunks:
            raise ValueError("TTS 입력 문장이 비어 있습니다.")

        pause = np.zeros(int(self.sample_rate * pause_ms / 1000.0), dtype=np.float32)
        tail = np.zeros(int(self.sample_rate * tail_silence_ms / 1000.0), dtype=np.float32)

        pieces: List[np.ndarray] = []
        for idx, chunk in enumerate(chunks):
            pieces.append(
                self._synthesize_chunk(
                    chunk,
                    length_scale=length_scale,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    guard_extra_frames=guard_extra_frames,
                )
            )
            if idx < len(chunks) - 1:
                pieces.append(pause)
        pieces.append(tail)

        return _wav_bytes(np.concatenate(pieces), self.sample_rate)
