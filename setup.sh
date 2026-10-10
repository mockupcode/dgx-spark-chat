#!/usr/bin/env bash
# One-time install. Everything lands inside this directory:
#   .venv/     Python 3.12 managed by uv (the system Python is never used)
#   bin/       llama-server built from llama.cpp with CUDA
#   models/    LLM, ASR, TTS and face models
#   .build/    sources, build trees and logs (safe to delete afterwards)
# Re-running skips whatever is already done.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ----------------------------------------------------------------- versions
PYTHON_VERSION="3.12"
TORCH_VERSION="2.14.1"
TORCH_INDEX="https://download.pytorch.org/whl/cu130"
LLAMA_CPP_REF="b11542"
CT2_REF="v4.8.2"
WAYU_TTS_REF="07ab898d516649313926f0b843b809a962f993ad"

LLM_REPO="unsloth/Qwen3.8-27B-GGUF"
LLM_REV="4ca720788d1e01f1bff70c033e0d0028fd02e502"
LLM_FILE="Qwen3.8-27B-UD-Q4_K_M.gguf"
LLM_MMPROJ="mmproj-F16.gguf"
LLM_MTP="MTP/mtp-Qwen3.8-27B-Q4_0.gguf"

ASR_REPO="typhoon-ai/typhoon-whisper-turbo"
ASR_REV="3c03fa84c26f172944422ceb8a4e88a2dbc08b10"
TTS_REPO="wayu-ai/wayu-paxa-tts-edge"
TTS_REV="8196688df56a8d08ccfa186505343adff412ab6b"

FACE_REV="47534e27c9851bb1128ccc0102f1145e27f23f98"
FACE_DET="face_detection_yunet_2023mar.onnx"
FACE_DET_SHA256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
FACE_REC="face_recognition_sface_2021dec.onnx"
FACE_REC_SHA256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"

# ----------------------------------------------------------------- helpers
BUILD="$ROOT/.build"
LOGS="$BUILD/logs"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
MODELS="$ROOT/models"
mkdir -p "$BUILD" "$LOGS" "$MODELS" "$ROOT/bin"

export UV_PYTHON_INSTALL_DIR="$BUILD/python"
export HF_HOME="$MODELS/.hf"
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHAINLP_DATA="$MODELS/pythainlp"
unset PYTHAINLP_DATA_DIR  # PyThaiNLP refuses to start when both are set

JOBS="$(nproc)"
CURRENT_LOG=""

step() { printf '\n==> %s\n' "$*"; }
ok()   { printf '    %s\n' "$*"; }
die()  { printf '\nsetup: %s\n' "$*" >&2; exit 1; }

on_error() {
  local code=$?
  printf '\nsetup failed (exit %s).\n' "$code" >&2
  if [[ -n "$CURRENT_LOG" && -f "$CURRENT_LOG" ]]; then
    printf 'last lines of %s:\n' "${CURRENT_LOG#"$ROOT"/}" >&2
    tail -n 40 "$CURRENT_LOG" >&2
  fi
}
trap on_error ERR

# Run a noisy command with its output in a log file; the log is shown only on failure.
quiet() {
  local name="$1" rc=0; shift
  CURRENT_LOG="$LOGS/$name.log"
  "$@" >"$CURRENT_LOG" 2>&1 || rc=$?
  [[ $rc -eq 0 ]] || return "$rc"
  CURRENT_LOG=""
}

# ----------------------------------------------------------------- prerequisites
step "Checking build tools"
if [[ -z "${CUDA_HOME:-}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  else
    CUDA_HOME="/usr/local/cuda"
  fi
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
for tool in git cmake c++ curl nvcc nvidia-smi sha256sum; do
  command -v "$tool" >/dev/null 2>&1 || die "'$tool' is required but was not found"
done
CUDA_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d ' ')"
[[ -n "$CUDA_CC" ]] || die "nvidia-smi did not report a GPU"
ok "CUDA $(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p'), GPU compute capability $CUDA_CC"

if command -v uv >/dev/null 2>&1; then
  UV="$(command -v uv)"
else
  step "Installing uv into .build/uv"
  quiet uv-install env UV_INSTALL_DIR="$BUILD/uv" UV_NO_MODIFY_PATH=1 \
    sh -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
  UV="$BUILD/uv/uv"
fi
ok "uv $("$UV" --version | awk '{print $2}')"

# ----------------------------------------------------------------- python
step "Python $PYTHON_VERSION environment"
if [[ ! -x "$PY" ]]; then
  quiet venv "$UV" venv --python "$PYTHON_VERSION" --python-preference only-managed "$VENV"
fi
ok "$("$PY" --version) in .venv"

uvpip() { "$UV" pip install --python "$PY" "$@"; }

step "PyTorch $TORCH_VERSION (CUDA)"
quiet torch uvpip "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || die "the installed torch does not see CUDA"
ok "torch $("$PY" -c 'import torch; print(torch.__version__)') sees the GPU"

step "Python packages"
# tltk (Thai G2P used by the TTS) pins scikit-learn ~=1.2, which has no Python
# 3.12 wheel and is binary-incompatible with numpy 2 (required by spaCy and
# OpenCV). tltk works with current scikit-learn, so the pin is overridden.
# It also imports pandas without declaring it.
printf 'scikit-learn>=1.5\n' >"$BUILD/overrides.txt"
quiet pip uvpip --override "$BUILD/overrides.txt" \
  "fastapi>=0.115" "uvicorn[standard]>=0.30" "httpx>=0.27" "numpy>=2" "pandas>=2" "scikit-learn>=1.5" \
  "faster-whisper==1.2.1" "opencv-python-headless>=4.10" \
  "transformers>=4.45,<5" "huggingface_hub>=0.34" \
  "wayu-tts @ git+https://github.com/wayu-research/wayu-tts-inference@$WAYU_TTS_REF" \
  "pytest>=8" "pybind11>=2.11" "setuptools>=70" "wheel"
"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || die "a dependency replaced torch with a build that cannot see CUDA"

# misaki (English words inside Thai text) would otherwise download its spaCy
# model the first time it runs, i.e. while serving.
SPACY_SERIES="$("$PY" -c 'import spacy; print(".".join(spacy.about.__version__.split(".")[:2]))')"
if ! "$PY" -c 'import en_core_web_sm' >/dev/null 2>&1; then
  quiet spacy-model uvpip \
    "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-$SPACY_SERIES.0/en_core_web_sm-$SPACY_SERIES.0-py3-none-any.whl"
fi
ok "packages installed"

# ----------------------------------------------------------------- llama.cpp
step "llama-server $LLAMA_CPP_REF with CUDA"
LLAMA_SRC="$BUILD/src/llama.cpp"
LLAMA_STAMP="$BUILD/llama-server.$LLAMA_CPP_REF"
if [[ ! -x "$ROOT/bin/llama-server" || ! -f "$LLAMA_STAMP" ]]; then
  rm -rf "$LLAMA_SRC"
  quiet llama-clone git clone --depth 1 --branch "$LLAMA_CPP_REF" https://github.com/ggml-org/llama.cpp "$LLAMA_SRC"
  ok "configuring (log: .build/logs/llama-build.log)"
  quiet llama-configure cmake -S "$LLAMA_SRC" -B "$BUILD/llama-build" \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc"
  ok "compiling, this takes a while"
  quiet llama-build cmake --build "$BUILD/llama-build" --config Release --target llama-server -j "$JOBS"
  install -m 0755 "$BUILD/llama-build/bin/llama-server" "$ROOT/bin/llama-server"
  touch "$LLAMA_STAMP"
fi
"$ROOT/bin/llama-server" --list-devices >"$LOGS/llama-devices.log" 2>&1 || die "bin/llama-server does not run"
grep -q "CUDA" "$LOGS/llama-devices.log" || die "bin/llama-server does not see a CUDA device"
ok "bin/llama-server ready"

# ----------------------------------------------------------------- models
hf_get() { # repo revision local_dir [files...]
  local repo="$1" rev="$2" dir="$3"; shift 3
  quiet "download-${repo//\//_}" "$VENV/bin/hf" download "$repo" "$@" --revision "$rev" --local-dir "$dir"
}

step "Qwen3.8 27B (GGUF + vision projector + MTP head)"
LLM_DIR="$MODELS/qwen3.8-27b"
hf_get "$LLM_REPO" "$LLM_REV" "$LLM_DIR" "$LLM_FILE" "$LLM_MMPROJ" "$LLM_MTP"
ok "$(du -sh "$LLM_DIR" | cut -f1) in models/qwen3.8-27b"

step "Typhoon Whisper (converted for CTranslate2)"
ASR_DIR="$MODELS/typhoon-whisper-turbo-ct2"
if [[ ! -f "$ASR_DIR/model.bin" ]]; then
  hf_get "$ASR_REPO" "$ASR_REV" "$BUILD/typhoon-whisper-turbo"
  rm -rf "$ASR_DIR"
  quiet asr-convert "$VENV/bin/ct2-transformers-converter" \
    --model "$BUILD/typhoon-whisper-turbo" --output_dir "$ASR_DIR" \
    --quantization float16 --copy_files tokenizer.json preprocessor_config.json
fi
ok "models/typhoon-whisper-turbo-ct2"

step "Wayu Thai TTS"
hf_get "$TTS_REPO" "$TTS_REV" "$MODELS/wayu-paxa-tts-edge"
ok "models/wayu-paxa-tts-edge"

step "Face detection and recognition"
FACE_DIR="$MODELS/face"
mkdir -p "$FACE_DIR"
fetch_checked() { # url file sha256
  local url="$1" file="$2" sum="$3"
  if [[ -f "$file" ]] && echo "$sum  $file" | sha256sum -c --status; then return; fi
  curl -fsSL -o "$file.part" "$url"
  echo "$sum  $file.part" | sha256sum -c --status || die "checksum mismatch for $(basename "$file")"
  mv "$file.part" "$file"
}
ZOO="https://github.com/opencv/opencv_zoo/raw/$FACE_REV/models"
fetch_checked "$ZOO/face_detection_yunet/$FACE_DET" "$FACE_DIR/$FACE_DET" "$FACE_DET_SHA256"
fetch_checked "$ZOO/face_recognition_sface/$FACE_REC" "$FACE_DIR/$FACE_REC" "$FACE_REC_SHA256"
ok "models/face"

# ----------------------------------------------------------------- CTranslate2 on CUDA
build_ctranslate2() {
  local src="$BUILD/src/CTranslate2" prefix="$BUILD/ct2-install"
  rm -rf "$src" "$BUILD/ct2-build" "$prefix"
  quiet ct2-clone git clone --depth 1 --branch "$CT2_REF" --recurse-submodules --shallow-submodules \
    https://github.com/OpenNMT/CTranslate2 "$src"
  # CMake's FindCUDA only accepts single-digit major versions, so it rejects
  # Blackwell (12.x). Hand nvcc the -gencode for this GPU directly instead.
  local sm="${CUDA_CC/./}"
  sed -i "s|^\([[:space:]]*\)cuda_select_nvcc_arch_flags(ARCH_FLAGS .*)|\1set(ARCH_FLAGS \"-gencode=arch=compute_${sm},code=sm_${sm}\")|" \
    "$src/CMakeLists.txt"
  grep -q "compute_${sm},code=sm_${sm}" "$src/CMakeLists.txt" || die "could not set the CUDA architecture for CTranslate2"
  ok "configuring for sm_${sm} (log: .build/logs/ct2-build.log)"
  # -w and -Wno-dev silence upstream warnings; they are not problems in this app.
  quiet ct2-configure cmake -Wno-dev -S "$src" -B "$BUILD/ct2-build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" \
    -DWITH_CUDA=ON -DWITH_CUDNN=OFF -DWITH_MKL=OFF -DOPENMP_RUNTIME=NONE \
    -DBUILD_CLI=OFF \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
    -DCMAKE_CXX_FLAGS="-w" -DCMAKE_C_FLAGS="-w" \
    -DCUDA_NVCC_FLAGS="-w;-Wno-deprecated-gpu-targets"
  ok "compiling, this takes a while"
  quiet ct2-build cmake --build "$BUILD/ct2-build" -j "$JOBS"
  quiet ct2-install cmake --install "$BUILD/ct2-build"

  local libdir="$prefix/lib"
  [[ -d "$libdir" ]] || libdir="$prefix/lib64"
  # The extension finds libctranslate2 next to itself ($ORIGIN), so nothing
  # depends on this build directory or on LD_LIBRARY_PATH at run time.
  quiet ct2-python env CTRANSLATE2_ROOT="$prefix" LDFLAGS="-L$libdir -Wl,-rpath,\$ORIGIN" \
    CFLAGS="-w" CXXFLAGS="-w" \
    "$UV" pip install --python "$PY" --no-build-isolation --no-cache --reinstall --no-deps "$src/python"
  local pkg
  pkg="$("$PY" -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("ctranslate2").origin))')"
  cp -a "$libdir"/libctranslate2.so* "$pkg/"
}

step "CTranslate2 on CUDA"
if quiet ct2-probe "$PY" -W ignore -m app.selfcheck ct2 asr; then
  ok "the prebuilt wheel runs on this GPU"
else
  ok "the prebuilt wheel cannot run Whisper on this GPU; building $CT2_REF from source"
  build_ctranslate2
  quiet ct2-probe "$PY" -W ignore -m app.selfcheck ct2 asr || die "CTranslate2 still cannot use CUDA"
  ok "built from source"
fi
grep -E '^  [a-z0-9]+: ' "$LOGS/ct2-probe.log" | sed 's/^/    /' || true

# ----------------------------------------------------------------- final check
step "Self-check"
quiet selfcheck "$PY" -W ignore -m app.selfcheck tts faces
grep -E '^  [a-z0-9]+: ' "$LOGS/selfcheck.log" | sed 's/^/    /' || true

chmod +x "$ROOT/setup.sh" "$ROOT/serve.sh"
trap - ERR
printf '\nSetup complete. Start with ./serve.sh\n'
