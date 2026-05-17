#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
DATASET="${DATASET:-gsm8k}"
OUTPUT_DIR="${OUTPUT_DIR:-./interleaved_grpo_output}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

ARGS=(
  --model-id "${MODEL_ID}"
  --dataset "${DATASET}"
  --output-dir "${OUTPUT_DIR}"
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-4}"
  --num-generations "${NUM_GENERATIONS:-8}"
  --max-completion-length "${MAX_COMPLETION_LENGTH:-1024}"
  --learning-rate "${LEARNING_RATE:-5e-6}"
)

if [[ -n "${MAX_SAMPLES:-}" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

if [[ "${USE_VLLM:-0}" == "1" ]]; then
  ARGS+=(--use-vllm)
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  srun accelerate launch --num_processes "${NUM_PROCESSES}" -m interleaved_grpo.train "${ARGS[@]}"
else
  accelerate launch --num_processes "${NUM_PROCESSES}" -m interleaved_grpo.train "${ARGS[@]}"
fi

