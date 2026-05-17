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
  --num-generations "${NUM_GENERATIONS:-4}"
  --max-prompt-length "${MAX_PROMPT_LENGTH:-512}"
  --max-completion-length "${MAX_COMPLETION_LENGTH:-1024}"
  --learning-rate "${LEARNING_RATE:-5e-6}"
)

if [[ -n "${MAX_SAMPLES:-}" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

if [[ -n "${REASONING_LANG:-}" ]]; then
  ARGS+=(--reasoning-lang "${REASONING_LANG}")
fi

if [[ "${FILTER_OVERLONG_PROMPTS:-1}" != "1" ]]; then
  ARGS+=(--no-filter-overlong-prompts)
fi

if [[ "${USE_VLLM:-0}" == "1" ]]; then
  ARGS+=(--use-vllm)
fi

if [[ "${USE_LORA:-0}" == "1" ]]; then
  ARGS+=(--use-lora)
  ARGS+=(--lora-rank "${LORA_RANK:-16}")
  ARGS+=(--lora-alpha "${LORA_ALPHA:-32}")
  ARGS+=(--lora-dropout "${LORA_DROPOUT:-0.05}")
  ARGS+=(--lora-target-modules "${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}")
  ARGS+=(--lora-bias "${LORA_BIAS:-none}")
  if [[ -n "${LORA_MODULES_TO_SAVE:-}" ]]; then
    ARGS+=(--lora-modules-to-save "${LORA_MODULES_TO_SAVE}")
  fi
fi

if [[ "${WANDB:-0}" == "1" ]]; then
  ARGS+=(--wandb)
fi

if [[ -n "${GENERATION_LOG_FILE:-}" ]]; then
  ARGS+=(--generation-log-file "${GENERATION_LOG_FILE}")
fi

if [[ "${DISABLE_GENERATION_LOGGING:-0}" == "1" ]]; then
  ARGS+=(--disable-generation-logging)
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  srun accelerate launch --num_processes "${NUM_PROCESSES}" -m interleaved_grpo.train "${ARGS[@]}"
else
  accelerate launch --num_processes "${NUM_PROCESSES}" -m interleaved_grpo.train "${ARGS[@]}"
fi
