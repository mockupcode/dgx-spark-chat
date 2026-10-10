"""Web server: static page, WebSocket session, model start-up.

Run with ``python -m app.main``. HTTPS is used when TLS_CERT and TLS_KEY point
to existing files (serve.sh creates them); otherwise it falls back to plain
HTTP, where the page still works for typed chat but the browser will not
allow the microphone.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import prompts
from .config import SETTINGS, STATIC_DIR
from .llm import CHAT, LLMClient
from .session import Engines, Session

log = logging.getLogger("app")


async def wait_for_llm(llm: LLMClient, timeout_s: float = 900.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not await llm.healthy():
        if time.monotonic() > deadline:
            raise RuntimeError(f"llama-server at {SETTINGS.llm_url} did not become ready")
        await asyncio.sleep(1.0)


def load_engines() -> Engines:
    from .asr import ThaiASR
    from .faces import FaceBook
    from .tts import ThaiTTS

    s = SETTINGS
    t = time.monotonic()
    asr = ThaiASR(s.asr_model_dir, s.asr_compute_type, s.asr_language)
    asr.warmup()
    log.info("ASR ready on CUDA (%.1fs)", time.monotonic() - t)

    t = time.monotonic()
    tts = ThaiTTS(s.tts_model_dir, s.tts_voice, s.tts_speed)
    tts.warmup()
    log.info("TTS ready, voice %s (%.1fs)", s.tts_voice, time.monotonic() - t)

    faces = FaceBook(s.face_dir, s.face_db, s.face_match_threshold)
    log.info("faces ready, %d enrolled", len(faces.names))
    return Engines(settings=s, asr=asr, tts=tts, faces=faces, llm=LLMClient(s.llm_url))


@asynccontextmanager
async def lifespan(app: FastAPI):
    engines = await asyncio.to_thread(load_engines)
    await wait_for_llm(engines.llm)
    # Put the system prompt in llama-server's cache before the first visitor.
    await engines.llm.prefill(
        [{"role": "system", "content": prompts.system_prompt(engines.tts.default_voice)},
         {"role": "user", "content": "สวัสดี"}],
        CHAT,
    )
    app.state.engines = engines
    log.info("ready")
    yield
    await engines.llm.close()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"ok": True})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    session = Session(ws, app.state.engines)
    try:
        await session.run()
    except Exception:
        log.exception("session ended with an error")


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per LLM request is noise
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    s = SETTINGS
    tls = bool(s.tls_cert and s.tls_key and Path(s.tls_cert).is_file() and Path(s.tls_key).is_file())
    if not tls:
        log.warning("no TLS certificate given: serving plain HTTP, browsers will block the microphone off localhost")
    uvicorn.run(
        app,
        host=s.host,
        port=s.port,
        ssl_certfile=s.tls_cert if tls else None,
        ssl_keyfile=s.tls_key if tls else None,
        log_level=os.environ.get("LOG_LEVEL", "info"),
        ws_max_size=8 * 1024 * 1024,
        ws_ping_interval=20,
    )


if __name__ == "__main__":
    main()
