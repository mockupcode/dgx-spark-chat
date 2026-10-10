"""Server-side end-of-speech detection.

The browser streams raw 16 kHz PCM in 32 ms frames. Each frame gets a speech
probability from Silero VAD (the ONNX file that ships inside faster-whisper,
so there is nothing extra to download), and :class:`Endpointer` turns that
stream of probabilities into utterance events.

Besides ``start`` and ``end`` the endpointer emits ``pause`` as soon as a short
silence appears. The session starts transcribing at that moment, so by the
time the silence is long enough to count as the end of the turn the transcript
is usually ready.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512  # what Silero expects at 16 kHz
FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE  # 32
_CONTEXT_SAMPLES = 64


class SileroStream:
    """Silero VAD with its recurrent state carried across frames."""

    def __init__(self) -> None:
        import onnxruntime
        from faster_whisper.utils import get_assets_path

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        path = f"{get_assets_path()}/silero_vad_v6.onnx"
        self._session = onnxruntime.InferenceSession(
            path, providers=["CPUExecutionProvider"], sess_options=opts
        )
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(_CONTEXT_SAMPLES, dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        window = np.concatenate([self._context, frame])[None, :]
        self._context = frame[-_CONTEXT_SAMPLES:]
        out, self._h, self._c = self._session.run(
            None, {"input": window, "h": self._h, "c": self._c}
        )
        return float(np.asarray(out).reshape(-1)[0])


@dataclass
class VadEvent:
    kind: str  # "start" | "pause" | "resume" | "end" | "discard"
    audio: np.ndarray | None = None


class Endpointer:
    """Turns per-frame speech probabilities into utterance events.

    Pure logic with no model inside, so it is easy to test with synthetic
    probabilities.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        end_silence_ms: int = 450,
        early_silence_ms: int = 200,
        min_speech_ms: int = 250,
        start_frames: int = 3,
        preroll_ms: int = 320,
        max_utterance_s: float = 25.0,
    ) -> None:
        self.on_threshold = threshold
        self.off_threshold = max(0.0, threshold - 0.15)
        self.end_frames = max(1, end_silence_ms // FRAME_MS)
        self.early_frames = max(1, min(early_silence_ms // FRAME_MS, self.end_frames - 1))
        self.min_speech_frames = max(1, min_speech_ms // FRAME_MS)
        self.start_frames = start_frames
        self.max_frames = int(max_utterance_s * 1000 / FRAME_MS)
        self._preroll: deque[np.ndarray] = deque(maxlen=max(1, preroll_ms // FRAME_MS))
        self.reset()

    def reset(self) -> None:
        self.in_speech = False
        self._candidate = 0
        self._silence = 0
        self._speech_frames = 0
        self._paused = False
        self._frames: list[np.ndarray] = []
        self._preroll.clear()

    def snapshot(self) -> np.ndarray:
        """Audio of the utterance so far."""
        if not self._frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._frames)

    def push(self, frame: np.ndarray, prob: float) -> list[VadEvent]:
        events: list[VadEvent] = []
        if not self.in_speech:
            self._preroll.append(frame)
            self._candidate = self._candidate + 1 if prob >= self.on_threshold else 0
            if self._candidate >= self.start_frames:
                self.in_speech = True
                self._frames = list(self._preroll)
                self._speech_frames = self._candidate
                self._silence = 0
                self._paused = False
                self._preroll.clear()
                events.append(VadEvent("start"))
            return events

        self._frames.append(frame)
        if prob >= self.off_threshold:
            self._speech_frames += 1
            self._silence = 0
            if self._paused:
                self._paused = False
                events.append(VadEvent("resume"))
        else:
            self._silence += 1
            if not self._paused and self._silence >= self.early_frames:
                self._paused = True
                events.append(VadEvent("pause", self._trimmed()))

        if self._silence >= self.end_frames or len(self._frames) >= self.max_frames:
            audio = self._trimmed()
            enough = self._speech_frames >= self.min_speech_frames
            self.reset()
            events.append(VadEvent("end" if enough else "discard", audio if enough else None))
        return events

    def _trimmed(self) -> np.ndarray:
        """Utterance audio minus most of the trailing silence.

        Trimming at a fixed point after the last speech frame means the audio
        handed out at ``pause`` and at ``end`` is identical when nothing was
        said in between, which lets the session reuse the early transcript.
        """
        keep = len(self._frames) - self._silence + 3
        return np.concatenate(self._frames[: max(1, keep)])


def pcm16_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


class FrameAssembler:
    """Re-slices arbitrary-length PCM chunks into exact 512-sample frames."""

    def __init__(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)

    def feed(self, samples: np.ndarray) -> list[np.ndarray]:
        self._buf = np.concatenate([self._buf, samples])
        n = len(self._buf) // FRAME_SAMPLES
        frames = [self._buf[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES] for i in range(n)]
        self._buf = self._buf[n * FRAME_SAMPLES:]
        return frames
