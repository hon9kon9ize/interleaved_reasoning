# Interleaved GRPO

Training utilities for interleaved chain-of-thought GRPO experiments with GSM8K and ToolMind-style agentic tool targets.

The pipeline trains a model to alternate between private planning and observable actions:

```text
<think>plan the next step</think>
<tool_call>{"name": "calculate", "arguments": {"expression": "2 + 2"}}</tool_call>
<think>reflect on the result</think>
<answer>4</answer>
```

## What Is Included

- `interleaved_grpo/datasets_loader.py`: GSM8K, MATH, ToolMind, and hybrid dataset normalization.
- `interleaved_grpo/parser.py`: exact XML-like parser for `<think>`, `<tool_call>`, `<tool_output>`, and `<answer>` streams.
- `interleaved_grpo/environment.py`: deterministic reward-time simulator for math and tool execution checks.
- `interleaved_grpo/rewards.py`: GRPO reward functions for outcome correctness, conditional steps, TTFT, and efficiency.
- `interleaved_grpo/train.py`: TRL `GRPOTrainer` entry point.
- `interleaved_grpo/run_training.sh`: Accelerate/Slurm-friendly shell launcher.

## Prerequisites

- Python 3.11 or newer.
- A CUDA GPU is strongly recommended for GRPO training. CPU is only practical for tests and dataset inspection.
- Hugging Face access to the selected base model and datasets.
- Optional but recommended: set `HF_TOKEN` for higher Hugging Face Hub rate limits.
- Optional: `vllm` for faster generation rollouts.
- Optional: a Weights & Biases account if using `--wandb`.

Create and install the environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional vLLM install:

```bash
python -m pip install -r requirements-vllm.txt
```

For restricted Hugging Face assets:

```bash
export HF_TOKEN=your_huggingface_token
```

## Validation

Run the unit tests before training:

```bash
python -m pytest tests -q
python -m compileall interleaved_grpo tests
```

Inspect ToolMind tool schemas:

```bash
python -m interleaved_grpo.toolmind_inspect --list-splits
python -m interleaved_grpo.toolmind_inspect --split open_datasets --max-samples 1000
```

Smoke-test dataset loading:

```bash
python - <<'PY'
from interleaved_grpo.datasets_loader import load_training_dataset

for name in ("gsm8k", "toolmind", "hybrid"):
    ds = load_training_dataset(name, max_samples=8)
    print(name, len(ds), ds.column_names)
    print(ds[0]["task_type"], ds[0]["answer"][:120])
PY
```

## Datasets

Supported `--dataset` values:

- `gsm8k`: math-only training. Targets are numeric answers and model completions must emit `<answer>...</answer>`.
- `math`: Hendrycks competition math. Targets are extracted from boxed or final answers.
- `toolmind`: ToolMind `open_datasets` by default when `--dataset-split train` is used.
- `hybrid`: alternating GSM8K and ToolMind rows for a 1:1 math/tool mix.
- `reasoning-lang`: translated GSM8K-style math rows that require `<think>` reasoning in a target language such as Cantonese (`yue`), while still rewarding final `<answer>...</answer>` correctness.

ToolMind rows preserve:

- `tool_definitions`: JSON-string tool schemas from the source row.
- `mock_outputs`: replay outputs from observed previous tool calls.
- `target_has_tool_call`: whether the final target is a literal `<tool_call>...</tool_call>` action or final public text.

Reasoning-language metadata:

- `reasoning_lang`: natural-language target for private reasoning, for example `yue`, `zh`, or `en`.
- If a dataset provides a `reasoning_lang` column, the trainer automatically enables the language consistency reward.
- If a dataset does not provide `reasoning_lang`, pass `--reasoning-lang yue` to use one target language for all rows.
- The parser also accepts `--reasoning_lang` and the typo-compatible `--reansoning_lang` alias.
- The built-in translated GSM8K loader still uses `task_type="math"`; `reasoning_lang` is only auxiliary reward metadata.
- The default CSV is resolved from the referenced `rl-data-geneator` project when present. Pass a custom CSV path with `--dataset-subset path/to/file.csv`.

## Training

Start with a tiny smoke run:

```bash
source .venv/bin/activate
python -m interleaved_grpo.train \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --dataset gsm8k \
  --max-samples 16 \
  --max-steps 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --num-generations 2 \
  --output-dir ./runs/smoke_gsm8k
```

Train on GSM8K:

```bash
python -m interleaved_grpo.train \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --dataset gsm8k \
  --output-dir ./runs/gsm8k_interleaved \
  --learning-rate 5e-6 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --num-generations 4 \
  --max-completion-length 1024
```

Train on ToolMind `open_datasets`:

```bash
python -m interleaved_grpo.train \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --dataset toolmind \
  --dataset-split train \
  --max-samples 5000 \
  --output-dir ./runs/toolmind_interleaved \
  --num-generations 4
```

Use a language reward with an existing dataset that does not have a `reasoning_lang` column:

```bash
python -m interleaved_grpo.train \
  --dataset gsm8k \
  --reasoning-lang yue \
  --max-samples 1000 \
  --output-dir ./runs/gsm8k_yue_reasoning
```

Train on the hybrid GSM8K + ToolMind mix:

```bash
python -m interleaved_grpo.train \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --dataset hybrid \
  --max-samples 10000 \
  --output-dir ./runs/hybrid_interleaved \
  --num-generations 4
```

Train on Cantonese reasoning-language math:

```bash
python -m interleaved_grpo.train \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --dataset reasoning-lang \
  --dataset-subset /Users/josephcheng/Projects/rl-data-geneator/data/gsm8k_yue_translated.csv \
  --max-samples 1000 \
  --output-dir ./runs/reasoning_lang_yue \
  --num-generations 4
```

Using the launcher:

```bash
MODEL_ID=Qwen/Qwen2.5-7B-Instruct \
DATASET=hybrid \
MAX_SAMPLES=10000 \
OUTPUT_DIR=./runs/hybrid_interleaved \
NUM_GENERATIONS=4 \
bash interleaved_grpo/run_training.sh
```

With vLLM enabled:

```bash
USE_VLLM=1 DATASET=hybrid MAX_SAMPLES=10000 bash interleaved_grpo/run_training.sh
```

The launcher uses `accelerate launch` locally and switches to `srun accelerate launch` when `SLURM_JOB_ID` is present.

Enable Weights & Biases metrics:

```bash
python -m interleaved_grpo.train \
  --dataset hybrid \
  --max-samples 10000 \
  --wandb
```

or with the launcher:

```bash
WANDB=1 DATASET=hybrid MAX_SAMPLES=10000 bash interleaved_grpo/run_training.sh
```

## Generation Logs

Training writes every generated completion to a JSONL file by default:

```text
OUTPUT_DIR/generations.jsonl
```

Each line includes the prompt, completion, target metadata, reward components, weighted reward, generation index, and process rank.

Use a custom path:

```bash
python -m interleaved_grpo.train \
  --dataset gsm8k \
  --generation-log-file ./runs/gsm8k_generations.jsonl
```

Disable generation logging:

```bash
python -m interleaved_grpo.train --disable-generation-logging
```

## Reward Configuration

Default reward functions:

1. `outcome_correctness_reward_fn`: exact math answer, exact ToolMind action target, or simulator-verified tool output.
2. `conditional_step_reward_fn`: rewards Plan-Action-Reflection structure only when the target is correct.
3. `ttft_reward_fn`: rewards a non-empty but short first `<think>` block.
4. `efficiency_penalty_fn`: penalizes bloated private reasoning relative to public output.

When `reasoning_lang` metadata is present, or when `--reasoning-lang` is set, training appends:

5. `language_consistency_reward_fn`: rewards `<think>` reasoning that matches `reasoning_lang`.

Default weights:

```text
OUTCOME=1.0 STEP=0.5 TTFT=0.3 EFFICIENCY=1.0
```

For language consistency, the default language reward weight is `0.2`.

Override them with:

```bash
python -m interleaved_grpo.train --reward-weights 1.0 0.5 0.3 1.0
```

When language consistency is enabled, pass five weights:

```bash
python -m interleaved_grpo.train --dataset reasoning-lang --reward-weights 1.0 0.5 0.3 1.0 0.2
```

The language reward uses `cantofilter` when installed for Cantonese detection, with a lightweight CJK/Cantonese-marker fallback otherwise.

## Operational Notes

- `num_generations` is the GRPO group size. The default is `4`; override it with `--num-generations`.
- `--beta 0.01` is the default KL coefficient.
- `--sequential-hybrid-sampler` is enabled by default so hybrid rows preserve the alternating math/tool order.
- Do not launch a full 7B run until parser, reward, and dataset smoke tests pass.
- TensorBoard logs are written when `--report-to tensorboard` is active. W&B is enabled with `--wandb`.

```bash
tensorboard --logdir ./runs
```

## References

```bibtex
@article{xie2026interleaved,
  title={Interleaved Chain-of-Thought: Grounding Large Language Models via Step-Wise Verifiable Action Loops},
  author={Xie, Yanzhe and Chen, Bo and Zhang, Min and Liu, Qian},
  journal={arXiv preprint arXiv:2602.04812},
  year={2026}
}

@article{nanbeige2025toolmind,
  title={ToolMind: A Large-Scale Dataset and Benchmark for Multi-Turn Agentic Tool Execution and Reasoning},
  author={Nanbeige Research Team},
  journal={arXiv preprint arXiv:2511.10394},
  year={2025}
}

@article{bytedance2026agentworld,
  title={Agent-World: From Static Mocks to Executable Synthetic Environments for Multi-Turn RL Training},
  author={Wang, Shuo and Cao, Ruisheng and Zheng, Zilong and Local, Team ByteDance},
  journal={Transactions on Machine Learning Research},
  volume={14},
  number={2},
  pages={204--221},
  year={2026}
}

@inproceedings{verl2025grpo,
  title={verl: Volcano Engine Reinforcement Learning Framework for Large-Scale Generative Policy Optimization},
  author={Volcano Engine Distributed AI Team},
  booktitle={Proceedings of the Systems and Machine Learning Platform Conference (SysML)},
  year={2025}
}

@article{deepseek2025r1,
  title={DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning},
  author={DeepSeek-AI and inside authors},
  journal={arXiv preprint arXiv:2501.12948},
  year={2025}
}
```
