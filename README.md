# autoresearch

This repository is organized as a list of folder-level experiments. Each experiment owns its code, dependencies, documentation, and container setup so it can evolve independently.

## Experiments

### nanochat

[nanochat](nanochat/) is a simplified single-GPU LLM training experiment based on [nanochat](https://github.com/karpathy/nanochat). It gives an AI agent a compact training setup, lets the agent iterate on `train.py`, and compares experiments with a fixed 5-minute training budget.

Read the folder-level docs: [nanochat/README.md](nanochat/README.md)

## Repository structure

```text
nanochat/
  Dockerfile        — isolated container for this experiment
  README.md         — experiment-specific setup and context
  prepare.py        — data prep and runtime utilities
  train.py          — model, optimizer, and training loop
  program.md        — agent instructions
  pyproject.toml    — experiment dependencies
  uv.lock           — locked dependency versions
```
