import numpy as np

from app.vad import FRAME_SAMPLES, Endpointer, FrameAssembler, pcm16_to_float


def frame(value: float = 0.0) -> np.ndarray:
    return np.full(FRAME_SAMPLES, value, dtype=np.float32)


def run(ep: Endpointer, probs: list[float]) -> list:
    events = []
    for i, p in enumerate(probs):
        for ev in ep.push(frame(i), p):
            events.append(ev)
    return events


def kinds(events) -> list[str]:
    return [e.kind for e in events]


def test_short_speech_then_silence_ends_turn():
    ep = Endpointer(end_silence_ms=448, early_silence_ms=192, min_speech_ms=192)
    events = run(ep, [0.0] * 5 + [0.9] * 20 + [0.0] * 20)
    assert kinds(events) == ["start", "pause", "end"]
    assert events[-1].audio is not None and events[-1].audio.size > 0


def test_end_audio_equals_pause_audio_when_nothing_was_said_in_between():
    ep = Endpointer(end_silence_ms=448, early_silence_ms=192)
    events = run(ep, [0.0] * 3 + [0.9] * 30 + [0.0] * 20)
    pause = next(e for e in events if e.kind == "pause")
    end = next(e for e in events if e.kind == "end")
    assert np.array_equal(pause.audio, end.audio)


def test_speech_resuming_after_pause_does_not_end_turn():
    ep = Endpointer(end_silence_ms=448, early_silence_ms=192)
    events = run(ep, [0.9] * 10 + [0.0] * 8 + [0.9] * 10 + [0.0] * 20)
    assert kinds(events) == ["start", "pause", "resume", "pause", "end"]


def test_blip_is_discarded():
    ep = Endpointer(end_silence_ms=448, early_silence_ms=192, min_speech_ms=320, start_frames=2)
    events = run(ep, [0.9] * 3 + [0.0] * 20)
    assert kinds(events)[-1] == "discard"


def test_isolated_frames_do_not_start_speech():
    ep = Endpointer(start_frames=3)
    assert kinds(run(ep, [0.9, 0.0, 0.9, 0.0, 0.9, 0.0])) == []


def test_preroll_is_included():
    ep = Endpointer(end_silence_ms=448, early_silence_ms=192, preroll_ms=160)
    events = run(ep, [0.0] * 10 + [0.9] * 20 + [0.0] * 20)
    audio = events[-1].audio
    # The first frame of the utterance carries an index from before speech started.
    assert audio[0] < 10


def test_long_monologue_is_cut():
    ep = Endpointer(max_utterance_s=1.0)
    events = run(ep, [0.9] * 100)
    assert "end" in kinds(events)


def test_frame_assembler_reslices():
    fa = FrameAssembler()
    out = fa.feed(np.zeros(700, dtype=np.float32)) + fa.feed(np.zeros(400, dtype=np.float32))
    assert [len(f) for f in out] == [512, 512]


def test_pcm16_to_float_range():
    data = np.array([-32768, 0, 32767], dtype="<i2").tobytes()
    out = pcm16_to_float(data)
    assert out[0] == -1.0 and out[1] == 0.0 and 0.99 < out[2] < 1.0
