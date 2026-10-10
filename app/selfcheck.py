"""Checks run by setup.sh so a finished setup is known to work.

    python -m app.selfcheck ct2     CTranslate2 sees CUDA with float16
    python -m app.selfcheck asr     the Thai ASR model runs on CUDA
    python -m app.selfcheck tts     the Thai voice synthesizes
    python -m app.selfcheck faces   the face models load
"""

from __future__ import annotations

import sys
import time

import numpy as np

from .config import SETTINGS


def check_ct2() -> str:
    from .asr import require_cuda

    require_cuda()
    import ctranslate2

    return f"ctranslate2 {ctranslate2.__version__}, {ctranslate2.get_cuda_device_count()} CUDA device(s)"


def check_asr() -> str:
    from .asr import ThaiASR

    asr = ThaiASR(SETTINGS.asr_model_dir, SETTINGS.asr_compute_type, SETTINGS.asr_language)
    asr.warmup()
    audio = (np.random.default_rng(1).standard_normal(16000 * 3) * 0.01).astype(np.float32)
    t = time.perf_counter()
    asr.transcribe(audio)
    return f"3 s of audio in {(time.perf_counter() - t) * 1000:.0f} ms on CUDA"


def check_tts() -> str:
    import torch

    from .tts import ThaiTTS

    tts = ThaiTTS(SETTINGS.tts_model_dir, SETTINGS.tts_voice, SETTINGS.tts_speed)
    tts.warmup()
    t = time.perf_counter()
    pcm = tts.synthesize("วันนี้อากาศดีมาก เหมาะกับการออกไปเดินเล่นค่ะ")
    seconds = len(pcm) / 2 / tts.sample_rate
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    return f"{seconds:.1f} s of speech in {(time.perf_counter() - t) * 1000:.0f} ms on {device}"


def check_faces() -> str:
    import cv2

    from .faces import FaceBook

    book = FaceBook(SETTINGS.face_dir, SETTINGS.face_db, SETTINGS.face_match_threshold)
    ok, jpeg = cv2.imencode(".jpg", np.zeros((240, 320, 3), dtype=np.uint8))
    if not ok:
        raise RuntimeError("OpenCV cannot encode JPEG")
    book.recognize(jpeg.tobytes())
    return f"models loaded, {len(book.names)} enrolled"


CHECKS = {"ct2": check_ct2, "asr": check_asr, "tts": check_tts, "faces": check_faces}


def main(argv: list[str]) -> int:
    names = argv or list(CHECKS)
    failed = False
    for name in names:
        try:
            print(f"  {name}: {CHECKS[name]()}")
        except Exception as exc:  # report every check, then fail
            print(f"  {name}: FAILED - {exc}", file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
