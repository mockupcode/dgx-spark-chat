"""Runtime settings.

Every path is derived from the project directory and every tunable can be
overridden with an environment variable, so nothing here is tied to one machine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT / "models"))
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    host: str = os.environ.get("APP_HOST", "0.0.0.0")
    port: int = _env_int("APP_PORT", 8443)
    tls_cert: str = os.environ.get("TLS_CERT", "")
    tls_key: str = os.environ.get("TLS_KEY", "")

    llm_url: str = os.environ.get("LLM_URL", "http://127.0.0.1:8080")
    llm_chat_max_tokens: int = _env_int("LLM_CHAT_MAX_TOKENS", 384)
    llm_reason_max_tokens: int = _env_int("LLM_REASON_MAX_TOKENS", 4096)
    llm_prefill: bool = _env_bool("LLM_PREFILL", True)
    history_max_messages: int = _env_int("HISTORY_MAX_MESSAGES", 24)

    asr_model_dir: Path = Path(os.environ.get("ASR_MODEL_DIR", MODELS_DIR / "typhoon-whisper-turbo-ct2"))
    asr_compute_type: str = os.environ.get("ASR_COMPUTE_TYPE", "float16")
    asr_language: str = os.environ.get("ASR_LANGUAGE", "th")

    tts_model_dir: Path = Path(os.environ.get("TTS_MODEL_DIR", MODELS_DIR / "wayu-paxa-tts-edge"))
    tts_voice: str = os.environ.get("TTS_VOICE", "f_teen_bright")
    tts_speed: float = _env_float("TTS_SPEED", 1.0)

    face_dir: Path = Path(os.environ.get("FACE_MODEL_DIR", MODELS_DIR / "face"))
    face_db: Path = Path(os.environ.get("FACE_DB", DATA_DIR / "faces.json"))
    face_match_threshold: float = _env_float("FACE_MATCH_THRESHOLD", 0.40)

    vad_threshold: float = _env_float("VAD_THRESHOLD", 0.5)
    vad_end_silence_ms: int = _env_int("VAD_END_SILENCE_MS", 450)
    vad_early_silence_ms: int = _env_int("VAD_EARLY_SILENCE_MS", 200)
    vad_min_speech_ms: int = _env_int("VAD_MIN_SPEECH_MS", 250)

    frame_max_age_s: float = _env_float("FRAME_MAX_AGE_S", 3.0)
    timings_log: bool = _env_bool("TIMINGS_LOG", True)


SETTINGS = Settings()
