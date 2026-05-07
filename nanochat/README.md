# nanochat

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `PROGRAM.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `PROGRAM.md` in this workstream is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069) and [this tweet](https://x.com/karpathy/status/2031135152349524125).

## How it works

This workstream is deliberately kept small and only really has four files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`eval_suite.py`** — EleutherAI lm-evaluation-harness adapter and the primary benchmark suite/score definition.
- **`PROGRAM.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. After training, `train.py` runs the benchmark suite in `eval_suite.py` and reports **`benchmark_score`** — higher is better. The default scorecard includes MMLU, MMLU STEM/non-STEM splits, GSM8k, MATH_6k, BBH, IFEval-nile, HumanEval, HellaSwag, ARC-Challenge, WinoGrande, TruthfulQA, LAMBADA, and WikiText-2. Legacy `val_bpb` can still be computed as a diagnostic with `NANOCHAT_EVAL_BPB=1`, but it is no longer the comparison metric.

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

## Quick start

**Requirements:** A single NVIDIA GPU (tested on H100), Python 3.10+, [mise](https://mise.jdx.dev/).

```bash
# 1. Enter the workstream folder
cd nanochat

# 2. Install uv project manager globally with mise
mise use -g asdf:asdf-community/asdf-uv@0.11.9

# 3. Install dependencies
uv sync

# 4. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 5. Enable W&B logging by exporting your host key (optional)
export WANDB_API_KEY=...

# 6. Manually run a single training experiment plus benchmark eval
uv run train.py
```

For a faster smoke run of the benchmark harness, set a per-task limit:

```bash
NANOCHAT_BENCHMARK_LIMIT=8 uv run train.py
```

When `WANDB_API_KEY` is set, `train.py` logs training metrics, benchmark progress, and eval benchmark results to a W&B project named from the workstream folder. The W&B run name comes from the experiment tag when `RUN_TAG`, `RUN_DIR`, or `WANDB_DIR=experiments/<run-tag>/wandb` is set.

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Docker

This workstream includes its own Dockerfile so dependencies are isolated at the workstream folder level.

```bash
cd nanochat
docker build -t autoresearch-nanochat:local .
docker run --rm --gpus all \
  -e WANDB_API_KEY \
  -v "$PWD":/workspace \
  -w /workspace \
  autoresearch-nanochat:local \
  uv run train.py
```

The `-e WANDB_API_KEY` flag forwards the host environment variable into the container for W&B auth.

To run data preparation in the container:

```bash
docker run --rm --gpus all \
  -v "$PWD":/workspace \
  -w /workspace \
  autoresearch-nanochat:local \
  uv run prepare.py
```

## Running the agent

Spin up your Claude/Codex or whatever you want in this workstream folder (and disable all permissions), then you can prompt something like:

```text
Hi have a look at PROGRAM.md and let's kick off a new experiment! let's do the setup first.
```

The `PROGRAM.md` file is essentially a super lightweight "skill".

Run artifacts belong under `experiments/<run-tag>/`. Commit `results.tsv` as the compact experiment record; keep local logs and W&B cache files untracked. Hyperparameter sweeps are not required for this experiment, and hyperparameter-only trials should be recorded there rather than committed one-by-one.

## Project structure

```text
Dockerfile      — isolated container for this workstream
experiments/    — per-run results folders (results.tsv, logs, local W&B files)
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
eval_suite.py   — benchmark harness adapter and primary score definition
PROGRAM.md      — agent-facing research program
pyproject.toml  — dependencies
uv.lock         — locked dependency versions
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed training budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. Benchmark evaluation runs after training, so total wall time depends on the selected benchmark suite and any `NANOCHAT_BENCHMARK_LIMIT`. Generation-heavy full-suite evals can run much longer than training; W&B progress metrics are emitted under `eval/progress/*`.
- **Benchmark-first metric.** The primary comparison metric is `benchmark_score`, a macro-average over the configured lm-evaluation-harness tasks. WikiText-2 bits-per-byte is converted to a higher-is-better score with `1 / (1 + bits_per_byte)`.
- **Self-contained training.** Training remains a single-GPU setup. Benchmark evaluation uses `lm-eval` plus Hugging Face datasets and HumanEval's unsafe-code path, so run the full suite inside the experiment container.
- **Folder-level isolation.** The Dockerfile, lockfile, and project metadata live inside this workstream folder so future workstreams can carry different dependencies without affecting this one.

## Platform support

This code currently requires that you have a single NVIDIA GPU. In principle it is quite possible to support CPU, MPS and other platforms but this would also bloat the code. I'm not 100% sure that I want to take this on personally right now. People can reference (or have their agents reference) the full/parent nanochat repository that has wider platform support and shows the various solutions (e.g. a Flash Attention 3 kernels fallback implementation, generic device support, autodetection, etc.), feel free to create forks or discussions for other platforms and I'm happy to link to them here in the README in some new notable forks section or etc.

Seeing as there seems to be a lot of interest in tinkering with autoresearch on much smaller compute platforms than an H100, a few extra words. If you're going to try running autoresearch on smaller computers (Macbooks etc.), I'd recommend one of the forks below. On top of this, here are some recommendations for how to tune the defaults for much smaller models for aspiring forks:

1. To get half-decent results I'd use a dataset with a lot less entropy, e.g. this [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean). These are GPT-4 generated short stories. Because the data is a lot narrower in scope, you will see reasonable results with a lot smaller models (if you try to sample from them after training).
2. You might experiment with decreasing `vocab_size`, e.g. from 8192 down to 4096, 2048, 1024, or even - simply byte-level tokenizer with 256 possibly bytes after utf-8 encoding.
3. In `prepare.py`, you'll want to lower `MAX_SEQ_LEN` a lot, depending on the computer even down to 256 etc. As you lower `MAX_SEQ_LEN`, you may want to experiment with increasing `DEVICE_BATCH_SIZE` in `train.py` slightly to compensate. The number of tokens per fwd/bwd pass is the product of these two.
4. Also in `prepare.py`, you'll want to decrease `EVAL_TOKENS` so that your validation loss is evaluated on a lot less data.
5. In `train.py`, the primary single knob that controls model complexity is the `DEPTH` (default 8, here). A lot of variables are just functions of this, so e.g. lower it down to e.g. 4.
6. You'll want to most likely use `WINDOW_PATTERN` of just "L", because "SSSL" uses alternating banded attention pattern that may be very inefficient for you. Try it.
7. You'll want to lower `TOTAL_BATCH_SIZE` a lot, but keep it powers of 2, e.g. down to `2**14` (~16K) or so even, hard to tell.

I think these would be the reasonable hyperparameters to play with. Ask your favorite coding agent for help and copy paste them this guide, as well as the full source code.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)
- [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) (AMD)

## License

MIT
