#!/bin/bash
# Oversubscription stress run: submit ENGINE_NUM_JOBS (default 50000) LoRA jobs
# onto ENGINE_N_ADAPTERS (default 16) GPU slots and let the coordinator page them
# in/out. Single GPU, TP=1/PP=1. Bounded by ENGINE_MAX_STEPS.

set -ex

export GPUS_PER_NODE=1
export ENGINE_OUTPUT_ROOT="${ENGINE_OUTPUT_ROOT:-/root/stress_artifacts}"
export ENGINE_NUM_JOBS="${ENGINE_NUM_JOBS:-50000}"
export ENGINE_MAX_STEPS="${ENGINE_MAX_STEPS:-300}"
export ENGINE_EXAMPLES_PER_JOB="${ENGINE_EXAMPLES_PER_JOB:-1}"
export ENGINE_LOG_EVERY="${ENGINE_LOG_EVERY:-20}"
export ENGINE_N_ADAPTERS="${ENGINE_N_ADAPTERS:-16}"

pkill sglang || true
ray stop --force || true
sleep 3

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source scripts/models/qwen3-4B.sh

ray start --head --node-ip-address 127.0.0.1 --num-gpus $GPUS_PER_NODE --disable-usage-stats

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{
     \"env_vars\": {
        \"PYTHONPATH\": \"/root/Megatron-LM\",
        \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
        \"ENGINE_OUTPUT_ROOT\": \"${ENGINE_OUTPUT_ROOT}\",
        \"ENGINE_NUM_JOBS\": \"${ENGINE_NUM_JOBS}\",
        \"ENGINE_MAX_STEPS\": \"${ENGINE_MAX_STEPS}\",
        \"ENGINE_EXAMPLES_PER_JOB\": \"${ENGINE_EXAMPLES_PER_JOB}\",
        \"ENGINE_LOG_EVERY\": \"${ENGINE_LOG_EVERY}\"
     }
   }" \
   -- python3 examples/training_engine/engine_stress_driver.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node $GPUS_PER_NODE \
   --debug-train-only \
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
