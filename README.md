# autoresearch

This repository is organized as a list of folder-level workstreams. Each workstream owns its code, dependencies, documentation, and container setup so it can evolve independently.

## Workstreams

### nanochat

[nanochat](nanochat/) is a simplified single-GPU LLM training workstream based on [nanochat](https://github.com/karpathy/nanochat). It gives an AI agent a compact training setup, lets the agent iterate on `train.py`, and compares normal runs with validation BPB after a fixed 5-minute training budget. Heavier benchmark evaluation is available for direction-level checks.

Human docs: [nanochat/README.md](nanochat/README.md). Agent program: [nanochat/PROGRAM.md](nanochat/PROGRAM.md).

### async-rl

[async-rl](async-rl/) is an 8-GPU RL workstream for [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) based on slime's [fully asynchronous rollout example](https://github.com/THUDM/slime/tree/main/examples/fully_async). It keeps a persistent async rollout worker running through slime while Megatron consumes completed rollout batches.

Human docs: [async-rl/README.md](async-rl/README.md). Agent program: [async-rl/PROGRAM.md](async-rl/PROGRAM.md).

## Repository structure

```text
nanochat/
  Dockerfile        — isolated container for this workstream
  README.md         — human-facing setup and context
  PROGRAM.md        — agent-facing research program
  eval_suite.py     — benchmark harness adapter and score definition
  prepare.py        — data prep and runtime utilities
  train.py          — model, optimizer, and training loop
  pyproject.toml    — workstream dependencies
  uv.lock           — locked dependency versions

async-rl/
  Dockerfile             — slime-based 8-GPU RL container
  README.md              — human-facing setup and context
  PROGRAM.md             — agent-facing research program
  prepare.py             — model/data download and checkpoint conversion
  train.py               — 8-GPU fully async slime launch configuration
  fully_async_rollout.py — persistent async rollout worker
```
