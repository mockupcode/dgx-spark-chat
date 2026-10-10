#!/usr/bin/env bash
# Start the assistant: bin/llama-server plus the web app (python -m app.main).
# Nothing is installed or downloaded here.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

LLM_PORT=8080
APP_PORT="${APP_PORT:-8443}"
LLM_DIR="$ROOT/models/qwen3.8-27b"
LLM_CTX="${LLM_CTX:-16384}"
LLM_MTP="${LLM_MTP:-1}"
RUN="$ROOT/run"
mkdir -p "$RUN"

port_busy() { ss -Hltn "sport = :$1" 2>/dev/null | grep -q .; }

if port_busy "$LLM_PORT"; then
  echo "Port $LLM_PORT is already in use (พอร์ต $LLM_PORT ถูกใช้อยู่). Stop whatever is using it and try again." >&2
  exit 1
fi
if port_busy "$APP_PORT"; then
  echo "Port $APP_PORT is already in use (พอร์ต $APP_PORT ถูกใช้อยู่). Set APP_PORT to another port." >&2
  exit 1
fi

# ----------------------------------------------------------------- certificate
# A fresh self-signed certificate for whatever addresses this machine has right now.
mapfile -t ADDRS < <(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+(\.[0-9]+){3}$' || true)
SAN="DNS:localhost,IP:127.0.0.1"
HOST_NAME="$(hostname 2>/dev/null || true)"
[[ -n "$HOST_NAME" ]] && SAN="$SAN,DNS:$HOST_NAME"
for ip in "${ADDRS[@]}"; do SAN="$SAN,IP:$ip"; done

TLS_DIR="$RUN/tls"
mkdir -p "$TLS_DIR"
chmod 700 "$TLS_DIR"
openssl req -x509 -nodes -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -sha256 -days 30 \
  -keyout "$TLS_DIR/key.pem" -out "$TLS_DIR/cert.pem" \
  -subj "/CN=dgx-spark-chat" -addext "subjectAltName=$SAN" >/dev/null 2>&1
chmod 600 "$TLS_DIR/key.pem"

# ----------------------------------------------------------------- llama-server
LLM_ARGS=(
  -m "$LLM_DIR/Qwen3.8-27B-UD-Q4_K_M.gguf"
  --mmproj "$LLM_DIR/mmproj-F16.gguf"
  --host 127.0.0.1 --port "$LLM_PORT"
  -ngl all -fa on -c "$LLM_CTX" -np 1
  -b 2048 -ub 1024
  --jinja
  --no-webui
)
if [[ "$LLM_MTP" == "1" ]]; then
  # Multi-token prediction head: speculative decoding with the model's own draft layers.
  LLM_ARGS+=(--spec-type draft-mtp -md "$LLM_DIR/MTP/mtp-Qwen3.8-27B-Q4_0.gguf" --spec-draft-ngl all)
fi

LLM_LOG="$RUN/llama-server.log"
"$ROOT/bin/llama-server" "${LLM_ARGS[@]}" >"$LLM_LOG" 2>&1 &
LLM_PID=$!

cleanup() {
  trap - EXIT INT TERM
  kill "$LLM_PID" 2>/dev/null || true
  wait "$LLM_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM

echo "Loading the LLM (log: run/llama-server.log)…"
until curl -sf "http://127.0.0.1:$LLM_PORT/health" >/dev/null 2>&1; do
  if ! kill -0 "$LLM_PID" 2>/dev/null; then
    echo "llama-server exited. Last lines of run/llama-server.log:" >&2
    tail -n 30 "$LLM_LOG" >&2
    exit 1
  fi
  sleep 1
done

# ----------------------------------------------------------------- web app
echo
echo "Open in a browser (accept the self-signed certificate once):"
echo "  https://localhost:$APP_PORT"
for ip in "${ADDRS[@]}"; do echo "  https://$ip:$APP_PORT"; done
echo

export APP_PORT TLS_CERT="$TLS_DIR/cert.pem" TLS_KEY="$TLS_DIR/key.pem"
export LLM_URL="http://127.0.0.1:$LLM_PORT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$ROOT/models/.hf"
export PYTHAINLP_DATA="$ROOT/models/pythainlp"
unset PYTHAINLP_DATA_DIR  # PyThaiNLP refuses to start when both are set

"$ROOT/.venv/bin/python" -m app.main
