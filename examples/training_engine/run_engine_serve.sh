#!/bin/bash
# Serve the continuous MultiLoRA training engine: start Ray + the coordinator +
# Megatron workers, then expose the thin HTTP API (serve_engine.py). Clients
# connect over HTTP and never touch Ray. Single GPU, TP=1/PP=1.

set -ex
export PS4='+[$(date +%H:%M:%S)] '

export GPUS_PER_NODE=1
export ENGINE_API_HOST="${ENGINE_API_HOST:-0.0.0.0}"
export ENGINE_API_PORT="${ENGINE_API_PORT:-8000}"
export ENGINE_N_ADAPTERS="${ENGINE_N_ADAPTERS:-16}"
export ENGINE_ENABLE_GENERATION="${ENGINE_ENABLE_GENERATION:-0}"
export ENGINE_SGLANG_ROUTER_PORT="${ENGINE_SGLANG_ROUTER_PORT:-30000}"

# Online-RL generation mode runs DISAGGREGATED: 1 GPU for the Megatron trainer
# and 1 GPU for the sglang inference engine (no --colocate). Both stay resident
# (offload defaults off for non-colocate), weights sync over NCCL broadcast, and
# sglang is always available for /sample. Train-only mode is single-GPU.
# NOTE: do NOT pass --sglang-router-ip; setting it makes miles assume a router
# already exists and skip launching one. Pin only the port (deterministic) and
# let miles start the router on the node IP; serve_engine computes that IP.
if [ "${ENGINE_ENABLE_GENERATION}" = "1" ]; then
  TOTAL_GPUS=2
  MODE_ARGS=(--rollout-num-gpus 1
             --sglang-router-port "${ENGINE_SGLANG_ROUTER_PORT}")
else
  TOTAL_GPUS=1
  MODE_ARGS=(--debug-train-only)
fi

pkill sglang || true
ray stop --force || true
sleep 3

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source scripts/models/qwen3-4B.sh

ray start --head --node-ip-address 127.0.0.1 --num-gpus $TOTAL_GPUS --disable-usage-stats

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{
     \"env_vars\": {
        \"PYTHONPATH\": \"/root/Megatron-LM\",
        \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
        \"ENGINE_API_HOST\": \"${ENGINE_API_HOST}\",
        \"ENGINE_API_PORT\": \"${ENGINE_API_PORT}\",
        \"ENGINE_N_ADAPTERS\": \"${ENGINE_N_ADAPTERS}\",
        \"ENGINE_ENABLE_GENERATION\": \"${ENGINE_ENABLE_GENERATION}\",
        \"ENGINE_SYNC_EVERY\": \"${ENGINE_SYNC_EVERY:-1}\",
        \"ENGINE_ADAPTER_STORE\": \"${ENGINE_ADAPTER_STORE:-}\"
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
   --rollout-max-response-len 512 \
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
   --rollout-num-gpus-per-engine 1 \
   --sglang-mem-fraction-static 0.4 \
   \
   --attention-dropout 0.0 \
   --hidden-dropout 0.0 \
   --accumulate-allreduce-grads-in-fp32 \
   --attention-softmax-in-fp32 \
   --attention-backend flash
