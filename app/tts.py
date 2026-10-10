"""Thai speech synthesis with the Wayu fixed-voice model."""

from __future__ import annotations

import contextlib
import io
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Upstream notices printed while the voice model loads and runs; harmless here.
for _message in (r"dropout option adds dropout", r".*weight_norm.* is deprecated", r".*torch\.jit\.script.* is deprecated"):
    warnings.filterwarnings("ignore", message=_message)

# espeak-ng keeps its data directory in a fixed 160-byte buffer and aborts the
# whole process when the path does not fit, so the English fallback is only
# used when the virtualenv lives at a short enough path.
_ESPEAK_PATH_LIMIT = 150


def english_g2p():
    """None to keep the default English G2P, or a replacement without espeak."""
    try:
        import espeakng_loader

        if len(espeakng_loader.get_data_path()) <= _ESPEAK_PATH_LIMIT:
            return None
    except Exception:  # loader missing: misaki already runs without espeak
        return None
    from wayu_tts import MisakiEnglish

    class NoEspeakEnglish(MisakiEnglish):
        def _load(self):
            from misaki import en

            return en.G2P(trf=False, british=self._british, fallback=None)

    log.warning("virtualenv path too long for espeak-ng; English words missing from the dictionary will be skipped")
    return NoEspeakEnglish()


class ThaiTTS:
    def __init__(self, model_dir: Path, voice: str, speed: float = 1.0) -> None:
        import torch
        from wayu_tts import ThaiTTS as WayuTTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            log.warning("torch does not see CUDA; TTS falls back to CPU")
        english = english_g2p()
        g2p_options = {} if english is None else {"english": english}
        # The model class prints a repo_id notice; the weights come from model_dir and
        # nothing is fetched.
        with contextlib.redirect_stdout(io.StringIO()):
            self.engine = WayuTTS.from_pretrained(model_dir, device=device, **g2p_options)
        self.voices = self.engine.voices
        if voice not in self.voices:
            raise ValueError(f"unknown voice {voice!r}; available: {', '.join(self.voices)}")
        self.default_voice = voice
        self.speed = speed
        self.sample_rate = self.engine.sample_rate
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")

    def synthesize(self, text: str, voice: str | None = None) -> bytes:
        """int16 little-endian PCM at :attr:`sample_rate`."""
        audio = self.engine(text, voice=voice or self.default_voice, speed=self.speed)
        if audio.size == 0:
            return b""
        return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    def warmup(self) -> None:
        self.synthesize("สวัสดีค่ะ ทดสอบเสียง Hello")
