#!/usr/bin/env bash
# serve/serve_model.sh <model-key> [--gpu N] [--port P] [--host H] [--mem-util F]
#                      [--allow-shared-gpu] [--foreground] [--dry-run]
#                      [-- <extra vllm serve args>]
#
# Starts one vLLM OpenAI-compatible server for one model on one GPU.
# Everything model-specific (HF path, reasoning parser, port, extra flags) comes
# from configs/models.json - never hard-coded here, never edited per run.
#
# Works on a box with any number of GPUs, N >= 1:
#   N >= 2 : one model per GPU, several up at once
#            ./serve/serve_model.sh qwen3.5-9b --gpu 0
#            ./serve/serve_model.sh gpt-oss-20b --gpu 1
#   N == 1 : one model at a time; to swap, stop the running server first
#            ./serve/stop_model.sh qwen3.5-9b --yes && ./serve/serve_model.sh gpt-oss-20b
#            or, if two models fit on the card, share it explicitly:
#            ./serve/serve_model.sh qwen3.5-9b  --mem-util 0.40 --allow-shared-gpu
#            ./serve/serve_model.sh gpt-oss-20b --mem-util 0.40 --allow-shared-gpu
#
# The script REFUSES to start if the port is already answering, or if the target
# GPU already holds a process, and tells you to stop it explicitly. It never
# stops anything itself: a server you did not start may have a generation running
# against it, and killing it loses that work.
#
# Background by default: log runs/serve/<key>.log, pid runs/serve/<key>.pid.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$REPO_ROOT/configs/models.json"

usage() { sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-1}"; }

[ $# -ge 1 ] || usage 1
case "$1" in -h|--help) usage 0 ;; esac
KEY="$1"; shift

GPU=""; PORT=""; HOST=""; MEM_UTIL=""; FOREGROUND=0; DRY_RUN=0; SHARE=0; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu)         GPU="$2"; shift 2 ;;
    --port)        PORT="$2"; shift 2 ;;
    --host)        HOST="$2"; shift 2 ;;
    --mem-util)    MEM_UTIL="$2"; shift 2 ;;
    --allow-shared-gpu) SHARE=1; shift ;;
    --foreground)  FOREGROUND=1; shift ;;
    --dry-run)     DRY_RUN=1; shift ;;
    --)            shift; EXTRA=("$@"); break ;;
    -h|--help)     usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage 1 ;;
  esac
done

[ -f "$CONFIG" ] || { echo "missing $CONFIG" >&2; exit 1; }

# --- resolve the model entry -------------------------------------------------
read -r HF_PATH PARSER CFG_PORT CFG_HOST CFG_GPU CFG_MEM < <(
python3 - "$CONFIG" "$KEY" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1])); key = sys.argv[2]
models = cfg["models"]
if key not in models:
    sys.exit("unknown model key %r; known: %s" % (key, ", ".join(sorted(models))))
m = models[key]
print(m["hf_path"], m["reasoning_parser"], m["port"],
      cfg.get("host", "127.0.0.1"), cfg.get("default_gpu", 0),
      cfg.get("default_gpu_memory_utilization", 0.90))
PY
)
mapfile -t CFG_FLAGS < <(
python3 - "$CONFIG" "$KEY" <<'PY'
import json, sys
for f in json.load(open(sys.argv[1]))["models"][sys.argv[2]].get("vllm_flags", []):
    print(f)
PY
)

PORT="${PORT:-$CFG_PORT}"
HOST="${HOST:-$CFG_HOST}"
MEM_UTIL="${MEM_UTIL:-$CFG_MEM}"
# GPU choice, in precedence order: --gpu wins; else $CUDA_VISIBLE_DEVICES if the calling
# shell sets it; else default_gpu in configs/models.json. Physical nvidia-smi indices.
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-$CFG_GPU}}"

# --- check the GPU exists ----------------------------------------------------
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 1; }
N_GPU="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
# One index, or a comma list for tensor parallel (`--gpu 0,2 -- --tensor-parallel-size 2`; 2026-09-09).
[[ "$GPU" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "GPU '$GPU': give an index or a comma list (this box has $N_GPU GPU(s))." >&2; exit 1; }
for g in ${GPU//,/ }; do
  [ "$g" -lt "$N_GPU" ] || { echo "GPU '$g' not available: this box has $N_GPU GPU(s), indices 0..$((N_GPU-1))." >&2; exit 1; }
done

# --- refuse to double-serve --------------------------------------------------
if curl -s -m 2 "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
  SERVED="$(curl -s -m 2 "http://$HOST:$PORT/v1/models" | python3 -c 'import json,sys; print(",".join(m["id"] for m in json.load(sys.stdin).get("data",[])))' 2>/dev/null || echo "?")"
  echo "port $PORT already answering (serving: $SERVED). Not starting a second server." >&2
  echo "If this is a stale server you want replaced, ask the author, then: ./serve/stop_model.sh $KEY --yes" >&2
  exit 1
fi

GPU_USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | sort -n | tail -1)"   # the fullest of the listed GPUs
if [ "$GPU_USED_MIB" -gt 2048 ] && [ "$SHARE" -eq 1 ]; then
  GPU_TOTAL_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU" | head -1)"
  echo "WARNING: --allow-shared-gpu: GPU $GPU already holds ${GPU_USED_MIB} of ${GPU_TOTAL_MIB} MiB;" >&2
  echo "         starting a second model there with --gpu-memory-utilization ${MEM_UTIL}." >&2
elif [ "$GPU_USED_MIB" -gt 2048 ]; then
  echo "GPU $GPU already holds ${GPU_USED_MIB} MiB (something is running on it):" >&2
  nvidia-smi -i "$GPU" --query-compute-apps=pid,used_memory,process_name --format=csv >&2 || true
  echo "This box has $N_GPU GPU(s). With one GPU, serve one model at a time:" >&2
  echo "  ask the author, then  ./serve/stop_model.sh <running-key> --yes  before starting $KEY." >&2
  echo "(Or --gpu <other id> if a free GPU exists, or --mem-util F --allow-shared-gpu to co-serve.)" >&2
  exit 1
fi

# --- build the command -------------------------------------------------------
# FlashInfer's JIT sampler cannot build for this box's sm120 card with the
# system CUDA 12.8 toolkit (needs >= 12.9) and kills EngineCore at warmup with
# a misleading "requires sm75 or higher". Use vLLM's native top-k/top-p sampler.
# 2026-09-08: "FlashInfer ... sm120". Remove when the box
# has a >= 12.9 toolkit or FlashInfer ships prebuilt sm120 kernels.
export VLLM_USE_FLASHINFER_SAMPLER=0
# Weights are cached under HF_HOME; never ask the hub (it rate-limited us, 429, 2026-09-09).
export HF_HUB_OFFLINE=1

VLLM_BIN="$REPO_ROOT/.venv/bin/vllm"
[ -x "$VLLM_BIN" ] || VLLM_BIN="$(command -v vllm || true)"
[ -n "$VLLM_BIN" ] || { echo "vllm not found (expected $REPO_ROOT/.venv/bin/vllm; run 'uv sync')" >&2; exit 1; }

CMD=("$VLLM_BIN" serve "$HF_PATH"
     --served-model-name "$KEY"
     --host "$HOST" --port "$PORT"
     --reasoning-parser "$PARSER"
     --gpu-memory-utilization "$MEM_UTIL")
[ "${#CFG_FLAGS[@]}" -gt 0 ] && CMD+=("${CFG_FLAGS[@]}")
[ "${#EXTRA[@]}" -gt 0 ] && CMD+=("${EXTRA[@]}")

RUN_DIR="$REPO_ROOT/runs/serve"; mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/$KEY.log"; PIDFILE="$RUN_DIR/$KEY.pid"

echo "model key : $KEY  ($HF_PATH)"
echo "gpu       : $GPU of $N_GPU"
echo "url       : http://$HOST:$PORT/v1   (served-model-name = $KEY)"
echo "command   : CUDA_VISIBLE_DEVICES=$GPU ${CMD[*]}"

if [ "$DRY_RUN" -eq 1 ]; then echo "(dry run: nothing started)"; exit 0; fi

if [ "$FOREGROUND" -eq 1 ]; then
  exec env CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}"
fi

{ echo; echo "=== $(date -u +'%Y-%m-%dT%H:%M:%SZ') start $KEY on gpu $GPU: ${CMD[*]}"; } >> "$LOG"
CUDA_VISIBLE_DEVICES="$GPU" nohup "${CMD[@]}" >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "pid       : $(cat "$PIDFILE")  (pidfile $PIDFILE)"
echo "log       : $LOG"
echo "poll      : until curl -s http://$HOST:$PORT/v1/models; do sleep 10; done"
