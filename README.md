# dgx-spark-chat

A real-time Thai voice and vision assistant that runs entirely on one NVIDIA DGX Spark (GB10).
Open it on a phone or laptop on your LAN and start talking. There is no push-to-talk. Turn on the
camera and the assistant can see what you show it and recognise people it has been introduced to.

ผู้ช่วย AI คุยภาษาไทยด้วยเสียงแบบเรียลไทม์ มองเห็นผ่านกล้องได้ และจำหน้าคนได้ ทุกอย่างรันบน DGX Spark
เครื่องเดียว ไม่ต้องพึ่ง cloud ([ภาษาไทยด้านล่าง](#ภาษาไทย))

Repository: https://github.com/mockupcode/dgx-spark-chat

## What it does

- **Hands-free Thai conversation.** Voice activity detection runs on the server, so you just talk
  and pause.
- **Camera.** The current frame is sent with each question, so you can ask "what is this?" or
  "read this sign".
- **Face memory.** Say "จำหน้าฉันไว้ ชื่อ …" (remember my face as …) and the assistant greets you by
  name next time. Say "ลืมหน้า …" to forget someone. You can also do both from the Faces panel.
- **Fast replies, with real thinking when needed.** Normal chat runs with thinking off. Questions
  that involve a calculation, a plan or a comparison switch to the model's reasoning mode
  automatically.
- **Barge-in (optional).** You can interrupt the assistant while it speaks. Use headphones for this.

| Part | Model | Runs on |
|---|---|---|
| Speech recognition | Typhoon Whisper Turbo (Thai) via faster-whisper / CTranslate2 | CUDA only |
| Language + vision + reasoning | Qwen3.8 27B (one model, GGUF) via llama.cpp `llama-server` | CUDA |
| Speech synthesis | Wayu-Paxa-TTS-Edge, default voice `f_teen_bright` | CUDA |
| Voice activity | Silero VAD (bundled with faster-whisper) | CPU, < 1 ms per frame |
| Faces | OpenCV YuNet + SFace | CPU, a few ms per frame |

## Requirements

- An NVIDIA DGX Spark, or another Linux machine with a recent NVIDIA GPU, the CUDA 13 toolkit
  (`nvcc`) and about 40 GB of free disk space
- `git`, `cmake`, a C++ compiler, `curl` and `openssl`
- [uv](https://docs.astral.sh/uv/). If it is missing, `setup.sh` downloads it into `.build/`

The system Python is never used or modified. `setup.sh` gets its own Python 3.12 through uv.

## Setup (once)

```bash
./setup.sh
```

This puts everything inside the project directory:

| Path | Contents |
|---|---|
| `.venv/` | Python 3.12 environment with PyTorch (CUDA), faster-whisper, Wayu TTS, OpenCV |
| `bin/llama-server` | built from llama.cpp with CUDA |
| `models/` | the GGUF model with its vision projector and MTP head, Typhoon Whisper converted to CTranslate2, the Wayu voice, and the face models |
| `.build/` | sources, build trees and logs. You can delete it once setup has finished |

The PyPI wheel of CTranslate2 may not support CUDA on the GPU (aarch64 wheels are CPU-only). In
that case `setup.sh` builds CTranslate2 from source automatically. Compiler output goes to
`.build/logs/` and is only printed if the build fails. At the end, setup checks that ASR runs on
CUDA, that the voice synthesises and that the face models load.

Re-running `setup.sh` skips anything that is already done.

## Run

```bash
./serve.sh
```

`serve.sh` does three things:

1. Stops with a message if port 8080 (llama-server) or the app port is already in use.
2. Creates a fresh self-signed certificate for the machine's current IP addresses.
3. Starts `bin/llama-server` and then `python -m app.main`.

It installs nothing.

It prints the URLs to open, for example `https://<spark-ip>:8443`. Browsers allow the microphone
only over HTTPS (or on `localhost`), which is why the app is served over HTTPS. Accept the
certificate warning once on each device.

If the page is opened over plain HTTP, it still loads and typed chat still works. Instead of
failing, it shows a notice explaining that the microphone needs HTTPS.

Stop everything with `Ctrl+C`.

## Why it is fast

The goal is the shortest time from when you stop talking to when you hear the first word.

- **Short end-of-speech wait (450 ms).** Speech end is detected on the server from the raw 16 kHz
  stream, which arrives in 32 ms frames as binary data (no base64, no WebM decoding).
- **Transcription starts during the pause.** ASR begins after 200 ms of silence. When the turn
  ends 250 ms later, the transcript is usually already done. ASR uses greedy decoding in fp16 on
  CUDA and is warmed up at start.
- **Prompt cache stays warm.**
  - The system prompt never changes during a conversation.
  - Information that changes, such as who is in view, goes into the user message.
  - When you start speaking, the conversation so far is pre-processed so the real request only
    has to read your new words.
  - History is trimmed in large blocks rather than one message per turn.
- **Speculative decoding** with the model's own multi-token-prediction head (MTP).
- **Speech starts before the answer is finished.** The answer is split at the first Thai phrase
  boundary. A long phrase with no space is split at a word boundary instead. Each piece is
  synthesised and sent while the model is still writing the rest.
- **Small camera frames.** Frames are 512 px on the long side, about 210 image tokens, which
  saves 100–200 ms per turn compared with 640 px.
- **Face recognition runs while you talk.** It uses the frame captured when speech starts, so it
  adds no time to the answer.
- **Reasoning turns speak a short filler right away** ("ขอคิดแป๊บนึงนะคะ") while the model thinks.

Each answer shows how long it took from the end of your speech to the first audio. The server log
prints the full timing breakdown for each turn (ASR, first token, first audio).

Measured on a DGX Spark with Thai speech, from the end of speech to the first audio in the browser:

| Turn | Time |
|---|---|
| Voice chat | about 0.85–0.95 s |
| Voice + camera frame | about 1.1–1.4 s |
| Reasoning question | a short spoken filler at about 0.5 s, then the answer once thinking is done |

| Stage | Time |
|---|---|
| Transcription (3 s of speech, fp16 on CUDA) | about 180 ms, usually finished before the turn ends |
| First token (prompt cache warm) | about 170 ms |
| Generation with MTP | 17–23 tokens/s (about 13 without MTP) |
| Wayu TTS, first phrase | about 100 ms on CUDA |
| Face recognition | about 10 ms per frame on CPU |

## Configuration

Set these as environment variables when running `./serve.sh`:

| Variable | Default | Meaning |
|---|---|---|
| `APP_PORT` | `8443` | HTTPS port of the web app |
| `LLM_CTX` | `16384` | llama-server context size |
| `LLM_MTP` | `1` | `0` turns off MTP speculative decoding |
| `LLM_PREFILL` | `1` | `0` turns off prompt pre-processing while the user speaks |
| `LLM_CHAT_MAX_TOKENS` | `384` | reply limit for normal chat |
| `LLM_REASON_MAX_TOKENS` | `4096` | limit for reasoning turns (thinking + answer) |
| `TTS_VOICE` | `f_teen_bright` | default voice; the page also has a voice picker |
| `TTS_SPEED` | `1.0` | speaking rate |
| `VAD_END_SILENCE_MS` | `450` | silence that ends your turn |
| `VAD_EARLY_SILENCE_MS` | `200` | silence after which transcription starts early |
| `VAD_THRESHOLD` | `0.5` | speech probability threshold |
| `FACE_MATCH_THRESHOLD` | `0.40` | cosine similarity needed to recognise a face |

Chat turns call the model with `chat_template_kwargs.enable_thinking=false` and
`reasoning_effort: "low"`. Reasoning turns use `enable_thinking=true` and
`reasoning_effort: "medium"`.

## Privacy

Everything runs locally. Face enrolment stores only face embeddings, never images, in
`data/faces.json`. That file, the generated certificates and all models are excluded by
`.gitignore`. Delete `data/faces.json` to forget everyone.

## Development

```bash
.venv/bin/python -m pytest tests
```

The tests cover end-of-speech detection, Thai phrase chunking, routing to reasoning mode,
face-command parsing and the request payloads. They also check that the repository contains no
machine-specific paths or addresses.

```
app/
  main.py       web server, model start-up
  session.py    one conversation: audio in, turns, speech out
  vad.py        streaming end-of-speech detection
  asr.py        Typhoon Whisper on CUDA
  llm.py        llama-server client (chat / reasoning modes)
  router.py     which questions need reasoning
  chunker.py    splitting streamed text for speech
  tts.py        Wayu Thai TTS
  faces.py      face enrolment and recognition
  commands.py   spoken face commands
  prompts.py    system prompt and fixed replies
  selfcheck.py  checks run by setup.sh
  static/       the web page
tests/
```

## Licenses

The code in this repository is MIT licensed (see `LICENSE`). The models are downloaded by
`setup.sh` and are not part of this repository. Each model has its own license:

| Model | License |
|---|---|
| Qwen3.8 27B | Apache-2.0 |
| Typhoon Whisper Turbo | MIT |
| Wayu-Paxa-TTS-Edge | **CC-BY-NC-4.0 (non-commercial use only)** |
| YuNet face detector | MIT |
| SFace face recognizer | Apache-2.0 |
| Silero VAD | MIT |

The Wayu voice model may not be used commercially. Swap the TTS before using this project in a
commercial setting.

---

## ภาษาไทย

### ทำอะไรได้

- **คุยภาษาไทยด้วยเสียงแบบปล่อยพูด** ไม่ต้องกดปุ่มค้าง ระบบตรวจจับเองว่าพูดจบแล้ว
- **มองเห็นผ่านกล้อง** ถามได้ เช่น "นี่คืออะไร" หรือ "ช่วยอ่านป้ายนี้หน่อย"
- **จำหน้า**
  - พูดว่า "จำหน้าฉันไว้ ชื่อ …" ครั้งต่อไปผู้ช่วยจะทักด้วยชื่อ
  - พูดว่า "ลืมหน้า …" เพื่อลบ
  - หรือใช้แผง "ใบหน้า" บนหน้าเว็บ
- **คิดจริงเมื่อจำเป็น** คุยทั่วไปตอบเร็วแบบปิด thinking ส่วนโจทย์คำนวณ วางแผน หรือเปรียบเทียบ
  จะสลับไปใช้ reasoning ให้อัตโนมัติ
- **พูดแทรกได้ (เลือกเปิดเอง)** แนะนำให้ใช้หูฟัง

### ติดตั้งและใช้งาน

```bash
./setup.sh   # ครั้งเดียว ติดตั้งทุกอย่างไว้ในโฟลเดอร์นี้ ไม่แตะ Python ของเครื่อง
./serve.sh   # เปิดใช้งาน
```

`serve.sh` จะแสดงลิงก์ เช่น `https://<ip-ของเครื่อง>:8443` ให้เปิดจากมือถือหรือโน้ตบุ๊กในวง LAN
เดียวกัน ครั้งแรกต้องกดยอมรับใบรับรองที่สร้างขึ้นเอง ใบรับรองนี้สร้างใหม่ทุกครั้งที่รัน
`serve.sh` ตาม IP ของเครื่อง ณ ตอนนั้น

- ถ้าพอร์ต 8080 ถูกใช้อยู่ `serve.sh` จะหยุดพร้อมแจ้งข้อความ
- ถ้าเปิดผ่าน HTTP ธรรมดา หน้าเว็บยังใช้ได้ (พิมพ์คุยได้) แค่จะแจ้งว่าการใช้ไมค์ต้องเปิดผ่าน HTTPS

### หมายเหตุเรื่อง license

โมเดลเสียง Wayu ใช้ license CC-BY-NC-4.0 ห้ามนำไปใช้เชิงพาณิชย์ ถ้าจะใช้งานเชิงพาณิชย์ให้เปลี่ยน TTS
ก่อน
