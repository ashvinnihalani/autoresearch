# async-rl

The idea: give an AI agent a real 8-GPU RL training setup and let it experiment autonomously on asynchronous reinforcement learning. It modifies the launch and rollout code, runs Qwen/Qwen3.5-4B RL experiments, checks whether reward or throughput improved, keeps or discards, and repeats. The training stack here is based on [slime](https://github.com/THUDM/slime), specifically its [fully asynchronous rollout example](https://github.com/THUDM/slime/tree/main/examples/fully_async), adapted into the same workstream structure as `nanochat`.

The default target model is [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). The default run uses 8 GPUs split between Megatron actor training and SGLang rollout generation.

## How it works

This workstream is deliberately kept small and has four files that matter:

- **`prepare.py`** - fixed setup script. It downloads slime, the Qwen/Qwen3.5-4B Hugging Face checkpoint, DAPO math prompts, and converts the model to Megatron `torch_dist` format. Not modified during experiments.
- **`train.py`** - the main file the agent edits. It builds the Ray/slime command, model args, RL hyperparameters, GPU split, and logging/runtime configuration. **This file is edited and iterated on by the agent**.
- **`fully_async_rollout.py`** - the persistent rollout worker adapted from slime's fully async example. It is part of the research surface when the experiment is about rollout scheduling, queueing, retry behavior, or async efficiency.
- **`PROGRAM.md`** - baseline instructions for one autonomous research agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By default, training runs GRPO-style RL on DAPO math prompts with a 4 GPU actor and 4 GPU rollout split. The comparison metric is the final DAPO reward from the last completed rollout window, with throughput and stability treated as secondary metrics.

## Quick start

**Requirements:** 8 NVIDIA GPUs, Docker with NVIDIA GPU support, Hugging Face access for `Qwen/Qwen3.5-4B`, and enough storage for model weights and converted checkpoints.

```bash
# 1. Enter the workstream folder
cd async-rl

# 2. Build the workstream image
docker build -t autoresearch-async-rl:local .

# 3. Start the reusable dev container
INSTANCE_STORAGE_ARGS=()
if [ -d /tmp/instance_storage ]; then
  INSTANCE_STORAGE_ARGS=(-v /tmp/instance_storage:/tmp/instance_storage)
fi

docker run -dit \
  --name autoresearch-async-rl-dev \
  --label autoresearch.workstream=async-rl \
  --label autoresearch.role=dev \
  --gpus all \
  --ipc=host \
  --shm-size=64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_TOKEN \
  -e WANDB_API_KEY \
  -v "$PWD":/workspace \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  "${INSTANCE_STORAGE_ARGS[@]}" \
  -w /workspace \
  autoresearch-async-rl:local \
  bash

# 4. Verify GPUs
docker exec autoresearch-async-rl-dev nvidia-smi

# 5. Download model/data and convert checkpoints
docker exec -it autoresearch-async-rl-dev python prepare.py

# 6. Inspect the Ray/slime command
docker exec -it autoresearch-async-rl-dev python train.py --dry-run

# 7. Manually run a smoke experiment
docker exec -it autoresearch-async-rl-dev python train.py --num-rollout 2 --save-interval 1
```

When `WANDB_API_KEY` is set, slime logs to a W&B project named `async-rl` by default.

If the above commands all work ok, your setup is working and you can go into autonomous RL research mode.

## Docker

This workstream includes its own Dockerfile so dependencies are isolated at the workstream folder level. It is based on `slimerl/slime:latest`, because slime carries Megatron, SGLang, Ray, and the model plugins needed for Qwen3.5.

```bash
cd async-rl
docker build -t autoresearch-async-rl:local .
docker run --rm --gpus all --ipc=host --shm-size=64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_TOKEN \
  -e WANDB_API_KEY \
  -v "$PWD":/workspace \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -w /workspace \
  autoresearch-async-rl:local \
  python train.py --dry-run
```

The `-e HF_TOKEN` flag forwards Hugging Face auth if needed. The `-e WANDB_API_KEY` flag forwards W&B auth for experiment tracking.

To run data preparation in a one-off container, mount a persistent workstream cache:

```bash
mkdir -p .cache
docker run --rm --gpus all --ipc=host --shm-size=64g \
  -e HF_TOKEN \
  -e ASYNC_RL_STORAGE=/workspace/.cache \
  -v "$PWD":/workspace \
  -v "$PWD/.cache":/workspace/.cache \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -w /workspace \
  autoresearch-async-rl:local \
  python prepare.py
```

## Running the agent

Spin up your Claude/Codex or whatever you want in this workstream folder (and disable all permissions), then you can prompt something like:

```text
Hi have a look at PROGRAM.md and let's kick off a new async RL experiment! let's do the setup first.
```

The `PROGRAM.md` file is essentially a super lightweight "skill".

Run artifacts belong under `experiments/<run-tag>/`. Commit `results.tsv` as the compact experiment record; keep local logs, Ray state, W&B cache files, and checkpoints untracked. Full comparable runs use the documented 8-GPU Qwen/Qwen3.5-4B setup; short runs are smoke tests unless the program says otherwise.

## Project structure

```text
Dockerfile             - slime-based isolated container for this workstream
experiments/           - per-run results folders (results.tsv, logs, W&B, Ray state)
prepare.py             - model/data prep and checkpoint conversion (do not modify)
train.py               - Ray/slime launch config and RL hyperparameters (agent modifies this)
fully_async_rollout.py - persistent async rollout worker (agent may modify for async ideas)
async_rl_config.py     - shared fixed model/data defaults
PROGRAM.md             - agent-facing research program
```

## Design choices

- **Autonomous RL research.** This workstream is meant for agents to run repeated 8-GPU RL experiments, record outcomes, and keep only changes that improve reward, throughput, or reliability.
- **slime as the substrate.** The workstream uses slime rather than reimplementing distributed RL. That keeps the research focused on async RL behavior and launch-level decisions instead of rebuilding Megatron, SGLang, and Ray plumbing.
- **Fixed model and budget.** The default target is Qwen/Qwen3.5-4B on 8 GPUs. Keeping model, prompt data, and GPU budget fixed makes experiment results easier to compare.
- **Separated setup and research surface.** `prepare.py` handles heavyweight one-time setup. `train.py` and, for async rollout ideas, `fully_async_rollout.py` are the files agents iterate on.
- **Folder-level isolation.** The Dockerfile, setup scripts, and experiment artifacts live inside this workstream folder so future workstreams can carry different dependencies without affecting this one.
