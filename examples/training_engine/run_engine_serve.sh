#!/bin/bash
# Serve the continuous MultiLoRA training engine: start Ray + the coordinator +
# Megatron workers, then expose the thin HTTP API (serve_engine.py). Clients
# connect over HTTP and never touch Ray.
#
# GPU layout (TP=1/PP=1; training scales via data parallel):
#   ENGINE_TRAIN_GPUS              # data-parallel Megatron trainers   (default 1)
#   ENGINE_INFER_GPUS             # GPUs for sglang rollout (RL only)  (default 1)
#   ENGINE_ROLLOUT_GPUS_PER_ENGINE  # GPUs per sglang engine           (default 1)
# With ENGINE_TRAIN_GPUS=4 the global batch is sharded across 4 DP ranks (grads
# all-reduced in Megatron), so you can run ~4x the tokens/step. With
# ENGINE_INFER_GPUS=4 / per-engine=1 you get 4 sglang engines behind the router.

set -ex
export PS4='+[$(date +%H:%M:%S)] '

# Data-parallel Megatron training workers (each is one GPU; TP=1, PP=1).
export ENGINE_TRAIN_GPUS="${ENGINE_TRAIN_GPUS:-1}"
export GPUS_PER_NODE="${ENGINE_TRAIN_GPUS}"
# sglang rollout GPUs (online-RL only) and how many GPUs each engine spans.
export ENGINE_INFER_GPUS="${ENGINE_INFER_GPUS:-1}"
export ENGINE_ROLLOUT_GPUS_PER_ENGINE="${ENGINE_ROLLOUT_GPUS_PER_ENGINE:-1}"
export ENGINE_API_HOST="${ENGINE_API_HOST:-0.0.0.0}"
export ENGINE_API_PORT="${ENGINE_API_PORT:-8000}"
export ENGINE_N_ADAPTERS="${ENGINE_N_ADAPTERS:-100}"
export ENGINE_ENABLE_GENERATION="${ENGINE_ENABLE_GENERATION:-0}"
export ENGINE_SGLANG_ROUTER_PORT="${ENGINE_SGLANG_ROUTER_PORT:-30000}"
# Batching window: how long the coordinator waits for more jobs/tokens before
# dispatching a step (higher = wider multi-adapter batches, more per-step latency).
export ENGINE_MAX_BATCH_WAIT_S="${ENGINE_MAX_BATCH_WAIT_S:-2.0}"

# W&B: per-adapter loss/reward/response-len + (optional) sample-rollout table.
export ENGINE_USE_WANDB=1
export ENGINE_WANDB_PROJECT="${ENGINE_WANDB_PROJECT:-miles-training-engine}"
export ENGINE_WANDB_GROUP="${ENGINE_WANDB_GROUP:-qwen3-4B-engine}"
# Decode a few sample rollouts into a W&B table every N committed steps (0 = off).
export ENGINE_LOG_SAMPLES_EVERY="${ENGINE_LOG_SAMPLES_EVERY:-0}"

# Online-RL generation mode runs DISAGGREGATED: ENGINE_TRAIN_GPUS for the
# Megatron trainers (data-parallel) and ENGINE_INFER_GPUS for the sglang
# inference engines (no --colocate). Both stay resident (offload defaults off
# for non-colocate), weights sync over NCCL broadcast from the trainers to every
# sglang engine, and sglang is always available for /sample. Train-only mode
# uses only the training GPUs.
# NOTE: do NOT pass --sglang-router-ip; setting it makes miles assume a router
# already exists and skip launching one. Pin only the port (deterministic) and
# let miles start the router on the node IP; serve_engine computes that IP.
if [ "${ENGINE_ENABLE_GENERATION}" = "1" ]; then
  TOTAL_GPUS=$((ENGINE_TRAIN_GPUS + ENGINE_INFER_GPUS))
  MODE_ARGS=(--rollout-num-gpus "${ENGINE_INFER_GPUS}"
             --sglang-router-port "${ENGINE_SGLANG_ROUTER_PORT}")
else
  TOTAL_GPUS=${ENGINE_TRAIN_GPUS}
  MODE_ARGS=(--debug-train-only)
fi

# W&B flags (mirrors examples/multi_lora/run_dynamic.sh). Off unless ENGINE_USE_WANDB=1.
if [ "${ENGINE_USE_WANDB}" = "1" ]; then
  WANDB_ARGS=(--use-wandb
              --wandb-project "${ENGINE_WANDB_PROJECT}"
              --wandb-group "${ENGINE_WANDB_GROUP}")
  # Pass an API key through if present so the Ray job authenticates non-interactively.
  if [ -n "${WANDB_API_KEY:-}" ]; then
    WANDB_ARGS+=(--wandb-key "${WANDB_API_KEY}")
  fi
else
  WANDB_ARGS=()
fi

pkill sglang || true
ray stop --force || true
sleep 3

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source scripts/models/qwen3-4B.sh

# Give the raylet a generous registration timeout on cold start.
export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-120}"


# The raylet fails to register ("GCS cannot find the node") when the head is
# pinned to 127.0.0.1 but Ray launches the GCS on the container's real IP -- the
# raylet then can't reconcile the two addresses. Don't pass --node-ip-address at
# all: Ray auto-detects the real node IP (get_node_ip_address) and uses it
# consistently for the GCS, raylet, and dashboard. Also clear any stale/foreign
# RAY_ADDRESS so we bring up a clean local head instead of joining another
# cluster. The dashboard is pinned to 127.0.0.1:8265 below for `ray job submit`.
unset RAY_ADDRESS

# ROOT CAUSE of the ray-start crash: these containers (gVisor/Modal) present
# /dev/shm with a bogus ~unlimited size ("32Z" in df). That value overflows Ray's
# int64 object-store sizing, so it computes a NEGATIVE "available" (-3686 bytes)
# and the raylet's plasma allocator aborts:
#   plasma_allocator.cc: Check failed: kFootprintLimit > kDlMallocReserved
# It crashes with OR without --object-store-memory (the overflow is in reading
# /dev/shm's size, not in the request). The sandbox blocks remounting /dev/shm,
# so instead point Ray's object store at a normal disk directory whose size
# reports sanely. The store only holds small in-flight payload ObjectRefs here,
# so disk-backed plasma is fine (Ray otherwise auto-falls-back to /tmp anyway).
RAY_PLASMA_DIRECTORY="${RAY_PLASMA_DIRECTORY:-/tmp}"
echo "[engine] using Ray plasma_directory=${RAY_PLASMA_DIRECTORY} (avoids /dev/shm 32Z overflow)" >&2
df -h /dev/shm >&2 || true

# Dump the raylet/GCS session logs so a startup failure shows its real cause
# (these live under /tmp inside the ephemeral container and are otherwise lost).
dump_ray_logs() {
  local sess="/tmp/ray/session_latest"
  echo "===== ray startup logs ($sess) =====" >&2
  for f in raylet.err raylet.out gcs_server.err gcs_server.out; do
    if [ -f "${sess}/logs/${f}" ]; then
      echo "----- ${f} (tail) -----" >&2
      tail -n 60 "${sess}/logs/${f}" >&2 || true
    fi
  done
  echo "===== /dev/shm size =====" >&2
  df -h /dev/shm >&2 || true
  echo "====================================" >&2
}

RAY_START_RETRIES="${RAY_START_RETRIES:-3}"
for attempt in $(seq 1 "${RAY_START_RETRIES}"); do
  if ray start --head --num-gpus "$TOTAL_GPUS" \
       --dashboard-host 127.0.0.1 --dashboard-port 8265 \
       --plasma-directory "$RAY_PLASMA_DIRECTORY" \
       --disable-usage-stats; then
    break
  fi
  dump_ray_logs
  if [ "${attempt}" -eq "${RAY_START_RETRIES}" ]; then
    echo "[engine] ray start --head failed after ${RAY_START_RETRIES} attempts" >&2
    exit 1
  fi
  echo "[engine] ray start --head failed (attempt ${attempt}/${RAY_START_RETRIES}); cleaning up and retrying..." >&2
  ray stop --force || true
  sleep 10
done

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{
     \"env_vars\": {
        \"PYTHONPATH\": \"/root/Megatron-LM\",
        \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
        \"ENGINE_API_HOST\": \"${ENGINE_API_HOST}\",
        \"ENGINE_API_PORT\": \"${ENGINE_API_PORT}\",
        \"ENGINE_N_ADAPTERS\": \"${ENGINE_N_ADAPTERS}\",
        \"ENGINE_ENABLE_GENERATION\": \"${ENGINE_ENABLE_GENERATION}\",
        \"ENGINE_MAX_BATCH_WAIT_S\": \"${ENGINE_MAX_BATCH_WAIT_S}\",
        \"ENGINE_MAX_TRAIN_TOKENS_PER_STEP\": \"${ENGINE_MAX_TRAIN_TOKENS_PER_STEP:-}\",
        \"ENGINE_SYNC_EVERY\": \"${ENGINE_SYNC_EVERY:-1}\",
        \"ENGINE_ADAPTER_STORE\": \"${ENGINE_ADAPTER_STORE:-}\",
        \"ENGINE_LOG_SAMPLES_EVERY\": \"${ENGINE_LOG_SAMPLES_EVERY}\",
        \"ENGINE_LOG_SAMPLES_MAX_CHARS\": \"${ENGINE_LOG_SAMPLES_MAX_CHARS:-4000}\",
        \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\"
     }
   }" \
   -- python3 examples/training_engine/serve_engine.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node $GPUS_PER_NODE \
   "${MODE_ARGS[@]}" \
   --calculate-per-token-loss \
   ${MODEL_ARGS[@]} \
   \
   --hf-checkpoint /root/Qwen3-4B/ \
   --megatron-to-hf-mode bridge \
   \
   --lora-rank 32 \
   --lora-alpha 32 \
   --lora-dropout 0.0 \
   --target-modules "all-linear" \
   --multi-lora-dir "${SCRIPT_DIR}/empty_adapters" \
   --multi-lora-n-adapters ${ENGINE_N_ADAPTERS} \
   --sglang-lora-backend triton \
   \
   --prompt-data /root/gsm8k/train.parquet \
   --input-key messages \
   --label-key label \
   --apply-chat-template \
   --num-rollout 1 \
   --rollout-batch-size 8 \
   --n-samples-per-prompt 1 \
   --rollout-max-response-len 4096 \
   --global-batch-size 8 \
   \
   --optimizer adam \
   --lr 1e-4 \
   --lr-decay-style constant \
   --weight-decay 0.0 \
   --adam-beta1 0.9 \
   --adam-beta2 0.98 \
   \
   --tensor-model-parallel-size 1 \
   --sequence-parallel \
   --pipeline-model-parallel-size 1 \
   --context-parallel-size 1 \
   --expert-model-parallel-size 1 \
   --expert-tensor-parallel-size 1 \
   --use-dynamic-batch-size \
   --max-tokens-per-gpu 9216 \
   \
   --rollout-num-gpus-per-engine ${ENGINE_ROLLOUT_GPUS_PER_ENGINE} \
   --sglang-mem-fraction-static 0.4 \
   \
   --attention-dropout 0.0 \
   --hidden-dropout 0.0 \
   --accumulate-allreduce-grads-in-fp32 \
   --attention-softmax-in-fp32 \
   --attention-backend flash \
   "${WANDB_ARGS[@]}"
