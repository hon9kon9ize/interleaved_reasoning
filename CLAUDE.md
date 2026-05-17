# CLAUDE.md

## Project Goal

Build a production-grade GRPO training pipeline for interleaved reasoning generation.
The model should learn to alternate between private planning in `<think>...</think>` and observable actions such as math answers in `<answer>...</answer>` or tool calls in `<tool_call>...</tool_call>`, rather than emitting one monolithic thought block.

## Target Package

All implementation code lives under `interleaved_grpo/`:

- `datasets_loader.py`: dataset loading and normalization for GSM8K, ToolMind `open_datasets`, and `reasoning-lang`, plus the interleaved chat system prompt.
- `parser.py`: regex-based parsing, answer extraction, and structural validation for `<think>`, `<tool_call>`, `<tool_output>`, and `<answer>` streams.
- `environment.py`: deterministic math/tool simulator for reward-time verification and smoke tests.
- `rewards.py`: GRPO-compatible reward functions for outcome correctness, conditional intermediate steps, TTFT, efficiency, and optional reasoning-language consistency.
- `train.py`: Hugging Face TRL `GRPOTrainer` entry point.
- `run_training.sh`: shell launcher for local, Accelerate, or Slurm-backed runs.

## Training Objective

Use Group Relative Policy Optimization (GRPO) to reward:

1. Correct final answers parsed from `<answer>...</answer>` for GSM8K, exact gold `<tool_call>...</tool_call>` targets for ToolMind action turns, or final public text for ToolMind response turns.
2. Intermediate Plan-Action-Reflection structure only when the final answer, target tool call, or target response is correct.
3. Short time-to-first-thought close, with a non-empty planning floor before observable action.
4. Efficient reasoning length, penalizing bloated thinking blocks relative to public output.
5. Private `<think>` text matching `reasoning_lang` when a dataset provides that column or `--reasoning-lang` is set.

## Engineering Notes

- Keep reward functions compatible with TRL custom reward signatures:
  `reward_fn(prompts, completions, **dataset_columns) -> list[float]`.
- The loader should normalize data into `question` and `answer`; `train.py` must apply `tokenizer.apply_chat_template(..., add_generation_prompt=True)` to create the final `prompt` column.
- Current primary data mix is GSM8K plus ToolMind `open_datasets`.
  - GSM8K rows use `task_type="math"` and reward strict `<answer>...</answer>` extraction with SymPy equivalence.
  - ToolMind rows use `task_type="tool"`, preserve per-row `tool_definitions`, carry JSON-string `mock_outputs` extracted from reference conversation tool outputs, and expose `target_has_tool_call` so rewards can distinguish action targets from final text targets.
  - Rows with `reasoning_lang` metadata enable language consistency checks for private `<think>` blocks while keeping the primary task reward active.
  - ToolMind `train` should resolve to `open_datasets`; avoid `graph_syn_datasets` for initial training because its tool surface is extremely long-tail.
- Prefer robust parsing over brittle string checks. Tag validation should detect unclosed, unopened, crossed, or nested `<think>` tags.
- Preserve the parser contract used by rewards: `segments`, `think_blocks`, `tool_calls`, `answer_blocks`, `public_text`, `is_valid_xml`, `is_interleaved`, `has_plan_action_reflection`, and transition/action counts.
- For ToolMind simulator-output validation, exact `mock_outputs` override built-in simulator behavior and the redundant-call guard. This is intentional because some gold trajectories repeat stochastic tools such as dice/profile generators.
- For ToolMind final action targets, the serialized gold `<tool_call>...</tool_call>` is the oracle; do not replace it with a placeholder target string.
- Synthetic completion formatting must escape literal XML-like text in assistant prose before appending generated `<tool_call>...</tool_call>` tags.
- Training defaults should be conservative and configurable through CLI flags so smoke tests can run on tiny subsets before large jobs.
- `train.py` defaults to `--num-generations 4`; increase it explicitly for larger GRPO groups.
- Any dataset with a `reasoning_lang` column, or any run with `--reasoning-lang <code>`, automatically appends `language_consistency_reward_fn` with default weight `0.2`.
- Training writes generated completions and reward components to `OUTPUT_DIR/generations.jsonl` unless `--disable-generation-logging` is set. Use `--wandb` to add Weights & Biases metrics reporting.
- Do not start a full 7B GRPO run as part of routine validation. Validate parser and reward behavior with lightweight tests first.

## Dataset Status

- `gsm8k`: math baseline; expected model completion should end with an explicit `<answer>...</answer>`.
- `toolmind` / `open_datasets`: mixed tool-action and final-response baseline.
  - Full split size checked: `205431` rows.
  - Final tool-call target rows checked: `147395` rows.
  - Target text rows: `58036` rows.
  - Tool-call target ratio: `0.7175`.
  - 5,000 mixed ToolMind self-target examples validated with `5000 / 5000` outcome correctness after preserving literal targets.
  - Negative controls over 5,000 replay rows confirmed corrupted tool names, wrong mock outputs, and missing required arguments score `0.0`.
- `reasoning-lang`: Cantonese reasoning-language math baseline using the translated GSM8K CSV from `/Users/josephcheng/Projects/rl-data-geneator/data/gsm8k_yue_translated.csv` when available.

## Validation Checklist

Before launching training:

1. Run parser and reward tests against monolithic, interleaved, malformed, and no-tag examples.
2. Run a small dataset loading smoke test for both GSM8K and ToolMind `open_datasets`.
3. Confirm `transformers`, `datasets`, `trl`, `accelerate`, and optionally `vllm` are installed in the target environment.
4. For multi-GPU runs, ensure the effective batch size is divisible by `num_generations`.
5. For ToolMind changes, rerun at least the 5,000-row mixed self-target reward check before training.
