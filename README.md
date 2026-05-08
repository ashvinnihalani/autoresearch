# autoresearch

This repository is organized as a list of folder-level workstreams. Each workstream owns its code, dependencies, documentation, and container setup so it can evolve independently.

## Workstreams

### nanochat

[nanochat](nanochat/) is a simplified single-GPU LLM training workstream based on [nanochat](https://github.com/karpathy/nanochat). It gives an AI agent a compact training setup, lets the agent iterate on `train.py`, and compares experiments with a benchmark score after a fixed 5-minute training budget.

Human docs: [nanochat/README.md](nanochat/README.md). Agent program: [nanochat/PROGRAM.md](nanochat/PROGRAM.md).

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
```
