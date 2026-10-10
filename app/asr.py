"""Thai speech recognition: Typhoon Whisper on faster-whisper, CUDA only.

There is deliberately no CPU path. If CTranslate2 cannot see the GPU the app
refuses to start instead of quietly answering ten times slower.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# Phrases Whisper-family models invent on noise or near-silence (subtitle credits).
_HALLUCINATIONS = re.compile(
    r"^(?:ขอบคุณ(?:ที่|สำหรับการ)รับชม.*|ซับไตเติ้ล.*|subtitles? by.*|thank(?:s| you) for watching.*|\.+)$",
    re.IGNORECASE,
)


class CudaUnavailable(RuntimeError):
    pass


def require_cuda() -> None:
    import ctranslate2

    if ctranslate2.get_cuda_device_count() < 1:
        raise CudaUnavailable("CTranslate2 does not see a CUDA device; ASR runs on CUDA only")
    if "float16" not in ctranslate2.get_supported_compute_types("cuda"):
        raise CudaUnavailable("CTranslate2 has no float16 CUDA support")


def clean_transcript(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if _HALLUCINATIONS.match(text):
        return ""
    return text


class ThaiASR:
    def __init__(self, model_dir: Path, compute_type: str = "float16", language: str = "th") -> None:
        require_cuda()
        from faster_whisper import WhisperModel

        self.language = language
        self.model = WhisperModel(str(model_dir), device="cuda", compute_type=compute_type)
        # One worker: calls are serialized on the GPU anyway, and a single
        # thread keeps CTranslate2's CUDA context warm.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size < 1600:
            return ""
        segments, _ = self.model.transcribe(
            audio.astype(np.float32, copy=False),
            language=self.language,
            task="transcribe",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,  # the endpointer already cut the utterance
            no_speech_threshold=0.6,
        )
        return clean_transcript("".join(seg.text for seg in segments))

    def warmup(self) -> None:
        rng = np.random.default_rng(0)
        self.transcribe((rng.standard_normal(16000) * 0.01).astype(np.float32))
