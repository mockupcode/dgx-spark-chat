"use strict";

// ---------------------------------------------------------------- elements

const $ = (id) => document.getElementById(id);
const el = {
  log: $("log"), empty: $("empty"), notice: $("notice"),
  statusDot: $("statusDot"), statusText: $("statusText"), latency: $("latency"),
  callBtn: $("callBtn"), stopBtn: $("stopBtn"), resetBtn: $("resetBtn"),
  camToggle: $("camToggle"), flipBtn: $("flipBtn"), bargeToggle: $("bargeToggle"),
  voiceSelect: $("voiceSelect"), textForm: $("textForm"), textInput: $("textInput"),
  facesBtn: $("facesBtn"), facesPanel: $("facesPanel"), enrollForm: $("enrollForm"),
  enrollName: $("enrollName"), faceList: $("faceList"), seen: $("seen"),
  cameraBox: $("cameraBox"), video: $("video"),
};

const STATE_TEXT = {
  offline: "ไม่ได้เชื่อมต่อ", idle: "พร้อม", listening: "กำลังฟัง…",
  thinking: "กำลังคิด…", speaking: "กำลังพูด…",
};

const MIC = 1;
const FRAME = 2;

const store = {
  get(key, fallback) {
    try { const v = localStorage.getItem(key); return v === null ? fallback : JSON.parse(v); }
    catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode */ }
  },
};

const app = {
  ws: null,
  live: false,
  micStream: null, micCtx: null, micNode: null,
  camStream: null, facing: "user",
  playCtx: null, playAt: 0, sampleRate: 24000, sources: new Set(), stoppedTurn: 0, playing: false,
  bubbles: new Map(),
  settings: {
    voice: store.get("voice", null),
    camera: store.get("camera", false),
    barge_in: store.get("barge_in", false),
  },
};

// ---------------------------------------------------------------- capability checks

// Browsers only expose the microphone on secure origins (HTTPS or localhost).
// Over plain HTTP navigator.mediaDevices is undefined, so every call is guarded.
function micSupport() {
  if (!window.isSecureContext) return "insecure";
  if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== "function") return "no-media";
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx || !("audioWorklet" in Ctx.prototype)) return "no-worklet";
  return "ok";
}

function showNotice(text) {
  el.notice.textContent = text;
  el.notice.hidden = !text;
}

function explainMicSupport(reason) {
  if (reason === "insecure") {
    showNotice("หน้านี้เปิดผ่าน HTTP เบราว์เซอร์จึงไม่อนุญาตให้ใช้ไมค์และกล้อง ให้เปิดลิงก์ https:// ที่ ./serve.sh แสดงแทน (ยังพิมพ์คุยได้ตามปกติ)");
  } else if (reason === "no-media") {
    showNotice("เบราว์เซอร์นี้ไม่รองรับการใช้ไมค์ (ยังพิมพ์คุยได้ตามปกติ)");
  } else if (reason === "no-worklet") {
    showNotice("เบราว์เซอร์นี้เก่าเกินไปสำหรับการคุยด้วยเสียง ลองใช้ Chrome, Edge, Firefox หรือ Safari รุ่นใหม่");
  }
}

// ---------------------------------------------------------------- connection

function connect() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${scheme}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  app.ws = ws;
  ws.onopen = () => sendSettings("hello");
  ws.onmessage = (ev) => (typeof ev.data === "string" ? onJson(JSON.parse(ev.data)) : onAudio(ev.data));
  ws.onclose = () => {
    setState("offline");
    el.callBtn.disabled = true;
    setTimeout(connect, 1500);
  };
}

function send(obj) {
  if (app.ws && app.ws.readyState === WebSocket.OPEN) app.ws.send(JSON.stringify(obj));
}

function sendBinary(kind, payload) {
  if (!app.ws || app.ws.readyState !== WebSocket.OPEN) return;
  const bytes = new Uint8Array(payload.byteLength + 1);
  bytes[0] = kind;
  bytes.set(new Uint8Array(payload), 1);
  app.ws.send(bytes.buffer);
}

function sendSettings(type = "settings") {
  send({ type, ...app.settings, camera: app.settings.camera && !!app.camStream });
}

function onJson(msg) {
  switch (msg.type) {
    case "ready":
      fillVoices(msg.voices, msg.voice);
      renderFaces(msg.faces);
      el.callBtn.disabled = micSupport() !== "ok";
      setState("idle");
      break;
    case "state": setState(msg.state); break;
    case "need_frame": captureFrame(); break;
    case "user": addUser(msg.text); break;
    case "reasoning_start": assistantBubble(msg.turn).startReasoning(); break;
    case "reasoning_delta": assistantBubble(msg.turn).reasoning(msg.text); break;
    case "assistant_delta": assistantBubble(msg.turn).append(msg.text); break;
    case "assistant_done": assistantBubble(msg.turn).done(msg.text, msg.timings); break;
    case "audio_start": app.sampleRate = msg.sample_rate; break;
    case "stop_audio": stopPlayback(msg.turn); break;
    case "seen": el.seen.textContent = describeSeen(msg.names); break;
    case "faces": renderFaces(msg.names, msg); break;
    case "error": addError(msg.message); break;
    default: break;
  }
}

// ---------------------------------------------------------------- conversation log

function scrollDown() { el.log.scrollTop = el.log.scrollHeight; }

function addMessage(cls, text) {
  el.empty.hidden = true;
  const div = document.createElement("div");
  div.className = `msg ${cls}`;
  div.textContent = text;
  el.log.appendChild(div);
  scrollDown();
  return div;
}

function addUser(text) { addMessage("user", text); }
function addError(text) { addMessage("assistant error", text); }

function assistantBubble(turn) {
  let bubble = app.bubbles.get(turn);
  if (bubble) return bubble;
  const div = addMessage("assistant", "");
  const body = document.createElement("span");
  div.appendChild(body);
  let details = null;
  let pre = null;
  bubble = {
    startReasoning() {
      if (details) return;
      details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = "กำลังคิดอย่างละเอียด…";
      pre = document.createElement("pre");
      details.append(summary, pre);
      div.insertBefore(details, body);
    },
    reasoning(text) { this.startReasoning(); pre.textContent += text; scrollDown(); },
    append(text) {
      if (details) details.querySelector("summary").textContent = "ดูขั้นตอนการคิด";
      body.textContent += text;
      scrollDown();
    },
    done(text, timings) {
      if (text && !body.textContent) body.textContent = text;
      if (!body.textContent && !details) { div.remove(); return; }
      const t = timings || {};
      const parts = [];
      if (t.mode === "reason") parts.push("reasoning");
      if (t.first_audio_ms != null) parts.push(`เสียงแรก ${(t.first_audio_ms / 1000).toFixed(2)} วิ`);
      if (parts.length) {
        const meta = document.createElement("span");
        meta.className = "meta";
        meta.textContent = parts.join(" · ");
        div.appendChild(meta);
      }
      if (t.first_audio_ms != null) {
        el.latency.hidden = false;
        el.latency.textContent = `${(t.first_audio_ms / 1000).toFixed(2)} วิ`;
      }
      scrollDown();
    },
  };
  app.bubbles.set(turn, bubble);
  return bubble;
}

function setState(state) {
  el.statusDot.dataset.state = state;
  el.statusText.textContent = STATE_TEXT[state] || state;
  el.stopBtn.hidden = !(state === "speaking" || state === "thinking");
}

// ---------------------------------------------------------------- playback

function ensurePlayer() {
  if (!app.playCtx) {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    app.playCtx = new Ctx();
  }
  if (app.playCtx.state === "suspended") app.playCtx.resume();
  return app.playCtx;
}

function setPlaying(playing) {
  if (app.playing === playing) return;
  app.playing = playing;
  send({ type: "playback", playing });
  if (!playing && el.statusDot.dataset.state === "speaking") setState("idle");
}

function onAudio(buffer) {
  const turn = new DataView(buffer).getUint32(0, true);
  if (turn <= app.stoppedTurn) return;
  const pcm = new Int16Array(buffer, 4);
  if (!pcm.length) return;
  const ctx = ensurePlayer();
  const audio = ctx.createBuffer(1, pcm.length, app.sampleRate);
  const ch = audio.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
  const src = ctx.createBufferSource();
  src.buffer = audio;
  src.connect(ctx.destination);
  const at = Math.max(ctx.currentTime + 0.02, app.playAt);
  src.start(at);
  app.playAt = at + audio.duration;
  app.sources.add(src);
  setPlaying(true);
  if (el.statusDot.dataset.state !== "speaking") setState("speaking");
  src.onended = () => {
    app.sources.delete(src);
    if (app.sources.size === 0) setPlaying(false);
  };
}

function stopPlayback(turn) {
  app.stoppedTurn = Math.max(app.stoppedTurn, turn);
  for (const src of app.sources) {
    try { src.stop(); } catch { /* already stopped */ }
  }
  app.sources.clear();
  app.playAt = 0;
  setPlaying(false);
}

// ---------------------------------------------------------------- microphone

async function startCall() {
  const support = micSupport();
  if (support !== "ok") { explainMicSupport(support); return; }
  ensurePlayer();
  try {
    app.micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
    });
  } catch (err) {
    showNotice(micErrorText(err));
    return;
  }
  const Ctx = window.AudioContext || window.webkitAudioContext;
  app.micCtx = new Ctx();
  await app.micCtx.audioWorklet.addModule("/static/mic-worklet.js");
  const source = app.micCtx.createMediaStreamSource(app.micStream);
  app.micNode = new AudioWorkletNode(app.micCtx, "mic-capture");
  app.micNode.port.onmessage = (ev) => sendBinary(MIC, ev.data);
  source.connect(app.micNode);
  // Keep the graph pulled without making the microphone audible.
  const mute = app.micCtx.createGain();
  mute.gain.value = 0;
  app.micNode.connect(mute).connect(app.micCtx.destination);

  app.live = true;
  el.callBtn.textContent = "วางสาย";
  el.callBtn.classList.add("live");
  showNotice("");
  if (app.settings.camera) await startCamera();
  sendSettings();
}

function stopCall() {
  app.live = false;
  if (app.micStream) app.micStream.getTracks().forEach((t) => t.stop());
  if (app.micCtx) app.micCtx.close();
  app.micStream = app.micCtx = app.micNode = null;
  stopCamera();
  stopPlayback(app.stoppedTurn);
  send({ type: "interrupt" });
  el.callBtn.textContent = "เริ่มคุย";
  el.callBtn.classList.remove("live");
  sendSettings();
}

function micErrorText(err) {
  const name = err && err.name;
  if (name === "NotAllowedError") return "ไม่ได้รับอนุญาตให้ใช้ไมค์ กดอนุญาตที่แถบที่อยู่ของเบราว์เซอร์แล้วลองใหม่";
  if (name === "NotFoundError") return "ไม่พบไมค์ในเครื่องนี้";
  if (name === "NotReadableError") return "ไมค์ถูกโปรแกรมอื่นใช้อยู่";
  return `เปิดไมค์ไม่ได้: ${(err && err.message) || err}`;
}

// ---------------------------------------------------------------- camera

async function startCamera() {
  if (micSupport() !== "ok") return;
  stopCamera();
  try {
    app.camStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: app.facing, width: { ideal: 1280 }, height: { ideal: 720 } },
    });
  } catch (err) {
    showNotice(`เปิดกล้องไม่ได้: ${(err && err.message) || err}`);
    el.camToggle.checked = false;
    app.settings.camera = false;
    store.set("camera", false);
    return;
  }
  el.video.srcObject = app.camStream;
  await el.video.play().catch(() => {});
  el.cameraBox.hidden = false;
  el.cameraBox.classList.toggle("rear", app.facing !== "user");
  el.flipBtn.hidden = false;
  sendSettings();
}

function stopCamera() {
  if (app.camStream) app.camStream.getTracks().forEach((t) => t.stop());
  app.camStream = null;
  el.video.srcObject = null;
  el.cameraBox.hidden = true;
  el.flipBtn.hidden = true;
}

const canvas = document.createElement("canvas");

// Frames go out at most 512 px on the long side: about 210 image tokens, which
// keeps prefill short (640 px costs ~320 tokens and 100-200 ms more per turn)
// while text on a whiteboard stays readable.
function captureFrame() {
  return new Promise((resolve) => {
    const v = el.video;
    if (!app.camStream || !v.videoWidth) { resolve(false); return; }
    const scale = Math.min(1, 512 / Math.max(v.videoWidth, v.videoHeight));
    canvas.width = Math.round(v.videoWidth * scale);
    canvas.height = Math.round(v.videoHeight * scale);
    canvas.getContext("2d").drawImage(v, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(async (blob) => {
      if (!blob) { resolve(false); return; }
      sendBinary(FRAME, await blob.arrayBuffer());
      resolve(true);
    }, "image/jpeg", 0.75);
  });
}

// ---------------------------------------------------------------- faces

function renderFaces(names, info) {
  el.faceList.replaceChildren();
  for (const name of names || []) {
    const li = document.createElement("li");
    const label = document.createElement("span");
    label.textContent = name;
    const del = document.createElement("button");
    del.type = "button";
    del.className = "ghost small";
    del.textContent = "ลบ";
    del.onclick = () => send({ type: "forget", name });
    li.append(label, del);
    el.faceList.appendChild(li);
  }
  if (info && info.enrolled) el.seen.textContent = `จำหน้า ${info.enrolled} แล้ว`;
  else if (info && info.ok === false) el.seen.textContent = "ทำไม่สำเร็จ ตรวจว่ากล้องเปิดอยู่และเห็นหน้าชัด";
}

function describeSeen(names) {
  if (!names || !names.length) return "ไม่เห็นใบหน้าในภาพ";
  const known = names.filter(Boolean);
  const unknown = names.length - known.length;
  const parts = [];
  if (known.length) parts.push(`เห็น: ${known.join(", ")}`);
  if (unknown) parts.push(`ไม่รู้จัก ${unknown} คน`);
  return parts.join(" · ");
}

// ---------------------------------------------------------------- settings ui

function fillVoices(voices, serverDefault) {
  el.voiceSelect.replaceChildren();
  for (const v of voices) {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    el.voiceSelect.appendChild(opt);
  }
  const chosen = voices.includes(app.settings.voice) ? app.settings.voice : serverDefault;
  el.voiceSelect.value = chosen;
  app.settings.voice = chosen;
  sendSettings();
}

function bindUi() {
  el.camToggle.checked = !!app.settings.camera;
  el.bargeToggle.checked = !!app.settings.barge_in;

  el.callBtn.onclick = () => (app.live ? stopCall() : startCall());
  el.stopBtn.onclick = () => send({ type: "interrupt" });
  el.resetBtn.onclick = () => {
    send({ type: "reset" });
    app.bubbles.clear();
    el.log.querySelectorAll(".msg").forEach((n) => n.remove());
    el.empty.hidden = false;
  };
  el.camToggle.onchange = async () => {
    app.settings.camera = el.camToggle.checked;
    store.set("camera", app.settings.camera);
    if (app.settings.camera && app.live) await startCamera();
    if (!app.settings.camera) stopCamera();
    sendSettings();
  };
  el.flipBtn.onclick = async () => {
    app.facing = app.facing === "user" ? "environment" : "user";
    await startCamera();
  };
  el.bargeToggle.onchange = () => {
    app.settings.barge_in = el.bargeToggle.checked;
    store.set("barge_in", app.settings.barge_in);
    sendSettings();
  };
  el.voiceSelect.onchange = () => {
    app.settings.voice = el.voiceSelect.value;
    store.set("voice", app.settings.voice);
    sendSettings();
  };
  el.textForm.onsubmit = (ev) => {
    ev.preventDefault();
    const text = el.textInput.value.trim();
    if (!text) return;
    ensurePlayer();
    send({ type: "text", text });
    el.textInput.value = "";
  };
  el.facesBtn.onclick = () => {
    const open = el.facesPanel.hidden;
    el.facesPanel.hidden = !open;
    el.facesBtn.setAttribute("aria-expanded", String(open));
  };
  el.enrollForm.onsubmit = async (ev) => {
    ev.preventDefault();
    const name = el.enrollName.value.trim();
    if (!name) return;
    if (!app.camStream) { el.seen.textContent = "เปิดกล้องก่อนแล้วลองใหม่"; return; }
    await captureFrame();
    send({ type: "enroll", name });
    el.enrollName.value = "";
  };
}

bindUi();
const support = micSupport();
if (support !== "ok") explainMicSupport(support);
connect();
