"""One browser connection: microphone in, camera frames in, speech out.

Wire protocol (WebSocket):

client -> server
  binary  0x01 + int16 PCM, 16 kHz mono   microphone, streamed continuously
  binary  0x02 + JPEG                     camera frame, sent when asked
  json    {"type": "hello"|"settings", "voice", "camera", "barge_in"}
  json    {"type": "text", "text"}        typed message
  json    {"type": "playback", "playing"} whether the browser is still playing audio
  json    {"type": "interrupt"} | {"type": "reset"}
  json    {"type": "enroll", "name"} | {"type": "forget", "name"}

server -> client
  binary  uint32 turn id + int16 PCM      speech for that turn
  json    state, need_frame, user, assistant_delta, reasoning_delta, assistant_done,
          audio_start, stop_audio, seen, faces, error
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import struct
import time
from dataclasses import dataclass

import numpy as np
from fastapi import WebSocket

from . import prompts
from .chunker import SpeechChunker, thai_word_segmenter
from .commands import parse_face_command
from .config import Settings
from .faces import SeenFace, describe
from .llm import CHAT, REASON, LLMClient
from .router import needs_reasoning
from .vad import Endpointer, FrameAssembler, SileroStream, pcm16_to_float

log = logging.getLogger("session")
SEGMENT_THAI = thai_word_segmenter()

MIC = 0x01
FRAME = 0x02


@dataclass
class Engines:
    settings: Settings
    asr: object
    tts: object
    faces: object
    llm: LLMClient


def trim_history(history: list[dict], max_messages: int) -> list[dict]:
    """Drop old turns in large blocks.

    Removing one message per turn would change the start of the prompt every
    time and defeat the prompt cache; halving only occasionally keeps the
    prefix stable for many turns in a row.
    """
    if len(history) <= max_messages:
        return history
    kept = history[-(max_messages // 2):]
    while kept and kept[0]["role"] != "user":
        kept = kept[1:]
    return kept


def build_user_content(text: str, note: str, image_jpeg: bytes | None, reasoning: bool) -> tuple[object, str]:
    """(content for this request, text-only version to keep in history)."""
    words = f"{note}\n{text}" if note else text
    if reasoning:
        words += "\n" + prompts.REASONING_ADDENDUM.strip()
    if image_jpeg is None:
        return words, words
    image_url = "data:image/jpeg;base64," + base64.b64encode(image_jpeg).decode("ascii")
    content = [
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": words},
    ]
    return content, words


class Session:
    def __init__(self, ws: WebSocket, engines: Engines) -> None:
        self.ws = ws
        self.e = engines
        self.s = engines.settings
        self.voice = engines.tts.default_voice
        self.camera_on = False
        self.barge_in = False
        self.history: list[dict] = []
        self.turn = 0
        self.turn_task: asyncio.Task | None = None
        self.client_playing = False

        self.silero = SileroStream()
        self.endpointer = Endpointer(
            threshold=self.s.vad_threshold,
            end_silence_ms=self.s.vad_end_silence_ms,
            early_silence_ms=self.s.vad_early_silence_ms,
            min_speech_ms=self.s.vad_min_speech_ms,
        )
        self.assembler = FrameAssembler()
        self.ignoring = False
        self.early: tuple[int, asyncio.Future] | None = None
        self.speech_end_at = 0.0

        self.frame: bytes | None = None
        self.frame_at = 0.0
        self.frame_arrived = asyncio.Event()
        self.wants_faces = False
        self.faces_task: asyncio.Task | None = None
        self.background: set[asyncio.Task] = set()
        self._send_lock = asyncio.Lock()

    # ---------------------------------------------------------------- sending

    async def send_json(self, payload: dict) -> None:
        async with self._send_lock:
            await self.ws.send_json(payload)

    async def send_audio(self, turn: int, pcm: bytes) -> None:
        async with self._send_lock:
            await self.ws.send_bytes(struct.pack("<I", turn) + pcm)

    async def state(self, name: str) -> None:
        await self.send_json({"type": "state", "state": name})

    def spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self.background.add(task)
        task.add_done_callback(self._background_done)
        return task

    def _background_done(self, task: asyncio.Task) -> None:
        self.background.discard(task)
        if not task.cancelled() and task.exception() is not None:
            log.debug("background task failed: %s", task.exception())

    # ---------------------------------------------------------------- main loop

    async def run(self) -> None:
        await self.send_json({
            "type": "ready",
            "voices": self.e.tts.voices,
            "voice": self.voice,
            "faces": self.e.faces.names,
        })
        try:
            while True:
                message = await self.ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes"):
                    await self.on_binary(message["bytes"])
                elif message.get("text"):
                    await self.on_json(message["text"])
        finally:
            await self.interrupt(notify=False)
            for task in list(self.background):
                task.cancel()

    async def on_binary(self, data: bytes) -> None:
        kind, body = data[0], data[1:]
        if kind == MIC:
            for frame in self.assembler.feed(pcm16_to_float(body)):
                prob = self.silero(frame)
                for event in self.endpointer.push(frame, prob):
                    await self.on_vad(event.kind, event.audio)
        elif kind == FRAME and body:
            self.frame, self.frame_at = body, time.monotonic()
            self.frame_arrived.set()
            if self.wants_faces:
                self.wants_faces = False
                self.faces_task = self.spawn(self.recognize(body))

    async def on_json(self, raw: str) -> None:
        msg = json.loads(raw)
        kind = msg.get("type")
        if kind in ("hello", "settings"):
            if msg.get("voice") in self.e.tts.voices:
                self.voice = msg["voice"]
            self.camera_on = bool(msg.get("camera", self.camera_on))
            self.barge_in = bool(msg.get("barge_in", self.barge_in))
            if not self.camera_on:
                self.frame = None
        elif kind == "playback":
            self.client_playing = bool(msg.get("playing"))
        elif kind == "interrupt":
            await self.interrupt()
        elif kind == "reset":
            await self.interrupt()
            self.history = []
        elif kind == "text":
            text = str(msg.get("text", "")).strip()[:2000]
            if text:
                await self.start_turn(self.text_turn(text))
        elif kind == "enroll":
            await self.ui_enroll(str(msg.get("name", "")).strip()[:40])
        elif kind == "forget":
            ok = self.e.faces.forget(str(msg.get("name", "")))
            await self.send_json({"type": "faces", "names": self.e.faces.names, "ok": ok})

    # ---------------------------------------------------------------- speech events

    @property
    def busy(self) -> bool:
        return (self.turn_task is not None and not self.turn_task.done()) or self.client_playing

    async def on_vad(self, kind: str, audio: np.ndarray | None) -> None:
        if kind == "start":
            if self.busy:
                if not self.barge_in:
                    self.ignoring = True
                    return
                await self.interrupt()
            self.ignoring = False
            await self.state("listening")
            if self.camera_on:
                self.wants_faces = True
                await self.send_json({"type": "need_frame"})
            if self.s.llm_prefill:
                self.spawn(self.prefill())
        elif self.ignoring:
            if kind in ("end", "discard"):
                self.ignoring = False
        elif kind == "pause":
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(self.e.asr.executor, self.e.asr.transcribe, audio)
            self.early = (len(audio), future)
            if self.camera_on:
                await self.send_json({"type": "need_frame"})  # freshest view for the answer
        elif kind == "resume":
            self.early = None
        elif kind == "end":
            self.speech_end_at = time.monotonic()
            early, self.early = self.early, None
            await self.start_turn(self.voice_turn(audio, early))
        elif kind == "discard":
            self.early = None
            await self.state("idle")

    async def start_turn(self, coro) -> None:
        if self.turn_task is not None and not self.turn_task.done():
            await self.interrupt()
        self.turn += 1
        self.turn_task = asyncio.create_task(self._guarded(coro))

    async def _guarded(self, coro) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # one failed turn must not end the session
            log.exception("turn failed")
            await self.send_json({"type": "error", "message": f"ตอบไม่สำเร็จ: {exc}"})
            await self.state("idle")

    async def interrupt(self, notify: bool = True) -> None:
        task, self.turn_task = self.turn_task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if notify:
            self.client_playing = False
            await self.send_json({"type": "stop_audio", "turn": self.turn})

    # ---------------------------------------------------------------- turns

    async def voice_turn(self, audio: np.ndarray, early: tuple[int, asyncio.Future] | None) -> None:
        t0 = self.speech_end_at
        await self.state("thinking")
        if early is not None and early[0] == len(audio):
            text = await early[1]  # transcribed during the pause, before the turn ended
        else:
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(self.e.asr.executor, self.e.asr.transcribe, audio)
        t_asr = time.monotonic()
        if not text:
            await self.state("idle")
            return
        await self.send_json({"type": "user", "text": text, "turn": self.turn})
        await self.respond(text, t0, {"asr_ms": round((t_asr - t0) * 1000)})

    async def text_turn(self, text: str) -> None:
        await self.send_json({"type": "user", "text": text, "turn": self.turn})
        await self.state("thinking")
        if self.camera_on:  # typed while the camera is on: answer about what is in view now
            self.frame_arrived.clear()
            await self.send_json({"type": "need_frame"})
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.frame_arrived.wait(), timeout=0.5)
        await self.respond(text, time.monotonic(), {})

    def fresh_frame(self) -> bytes | None:
        if self.camera_on and self.frame and time.monotonic() - self.frame_at <= self.s.frame_max_age_s:
            return self.frame
        return None

    def messages(self, user_content: object) -> list[dict]:
        return [
            {"role": "system", "content": prompts.system_prompt(self.voice)},
            *self.history,
            {"role": "user", "content": user_content},
        ]

    async def prefill(self) -> None:
        with contextlib.suppress(Exception):
            await self.e.llm.prefill(self.messages("…"))

    async def respond(self, text: str, t0: float, timings: dict) -> None:
        turn = self.turn
        # Recognition started when this utterance began; it belongs to this turn only.
        faces_task, self.faces_task = self.faces_task, None
        command = parse_face_command(text)
        if command is not None:
            await self.face_command(command.action, command.name, text, turn)
            return

        mode = REASON if needs_reasoning(text) else CHAT
        note = await self.faces_note(faces_task)
        content, history_text = build_user_content(text, note, self.fresh_frame(), mode == REASON)
        messages = self.messages(content)
        self.history.append({"role": "user", "content": history_text})

        speech: asyncio.Queue[str | None] = asyncio.Queue()
        marks: dict[str, float] = {}
        speaker = asyncio.create_task(self.speak(turn, speech, marks))
        chunker = SpeechChunker(segment=SEGMENT_THAI)
        answer = ""
        try:
            if mode == REASON:
                await self.send_json({"type": "reasoning_start", "turn": turn})
                speech.put_nowait(prompts.thinking_filler(self.voice))
            async for delta in self.e.llm.stream(
                messages, mode, self.s.llm_reason_max_tokens if mode == REASON else self.s.llm_chat_max_tokens
            ):
                if delta.reasoning:
                    await self.send_json({"type": "reasoning_delta", "turn": turn, "text": delta.reasoning})
                if delta.content:
                    marks.setdefault("first_token", time.monotonic())
                    answer += delta.content
                    await self.send_json({"type": "assistant_delta", "turn": turn, "text": delta.content})
                    for piece in chunker.feed(delta.content):
                        speech.put_nowait(piece)
            for piece in chunker.flush():
                speech.put_nowait(piece)
            speech.put_nowait(None)
            await speaker
        except asyncio.CancelledError:
            speaker.cancel()
            raise
        except Exception as exc:  # keep the session alive, tell the user
            speaker.cancel()
            log.exception("turn failed")
            await self.send_json({"type": "error", "message": f"ตอบไม่สำเร็จ: {exc}"})
        finally:
            if answer.strip():
                self.history.append({"role": "assistant", "content": answer.strip()})
            else:
                self.history.pop()
            self.history = trim_history(self.history, self.s.history_max_messages)

        for key in ("first_token", "first_audio"):
            if key in marks:
                timings[f"{key}_ms"] = round((marks[key] - t0) * 1000)
        timings["mode"] = mode
        if self.s.timings_log:
            log.info("turn %d %s", turn, timings)
        await self.send_json({"type": "assistant_done", "turn": turn, "text": answer.strip(), "timings": timings})

    async def speak(self, turn: int, queue: asyncio.Queue, marks: dict[str, float]) -> None:
        """Synthesize queued text in order, sending each piece as soon as it is ready."""
        loop = asyncio.get_running_loop()
        tts = self.e.tts
        started = False
        while (text := await queue.get()) is not None:
            pcm = await loop.run_in_executor(tts.executor, tts.synthesize, text, self.voice)
            if not pcm:
                continue
            if not started:
                started = True
                marks["first_audio"] = time.monotonic()
                await self.send_json({"type": "audio_start", "turn": turn, "sample_rate": tts.sample_rate})
                await self.state("speaking")
            self.client_playing = True
            await self.send_audio(turn, pcm)
        if not started:
            await self.state("idle")

    async def say(self, turn: int, text: str) -> None:
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        queue.put_nowait(text)
        queue.put_nowait(None)
        await self.send_json({"type": "assistant_done", "turn": turn, "text": text, "timings": {}})
        await self.speak(turn, queue, {})

    # ---------------------------------------------------------------- faces

    async def recognize(self, jpeg: bytes) -> list[SeenFace]:
        loop = asyncio.get_running_loop()
        seen = await loop.run_in_executor(None, self.e.faces.recognize, jpeg)
        await self.send_json({"type": "seen", "names": [f.name for f in seen]})
        return seen

    async def faces_note(self, task: asyncio.Task | None) -> str:
        frame = self.fresh_frame()
        if frame is None:
            return ""
        if task is None:  # typed message: look at the current frame now (a few ms)
            task = self.spawn(self.recognize(frame))
        try:
            seen = await asyncio.wait_for(asyncio.shield(task), timeout=0.3)
        except Exception:  # slow or failed recognition never holds up the answer
            return ""
        return describe(seen)

    async def face_command(self, action: str, name: str, text: str, turn: int) -> None:
        loop = asyncio.get_running_loop()
        if action == "enroll":
            frame = self.fresh_frame()
            if frame is None:
                reply = prompts.camera_needed_reply(self.voice)
            else:
                ok = await loop.run_in_executor(None, self.e.faces.enroll, name, frame)
                reply = prompts.enroll_reply(name, ok, self.voice)
        else:
            ok = self.e.faces.forget(name)
            reply = prompts.forget_reply(name, ok, self.voice)
        self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        await self.send_json({"type": "faces", "names": self.e.faces.names})
        await self.say(turn, reply)

    async def ui_enroll(self, name: str) -> None:
        frame = self.fresh_frame()
        ok = False
        if name and frame is not None:
            loop = asyncio.get_running_loop()
            ok = await loop.run_in_executor(None, self.e.faces.enroll, name, frame)
        await self.send_json({"type": "faces", "names": self.e.faces.names, "ok": ok, "enrolled": name if ok else None})
