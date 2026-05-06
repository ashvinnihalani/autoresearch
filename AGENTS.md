# autoresearch Agent Guide

## Purpose

This repo is an autonomous research workspace for running focused, measurable experiments with AI agents. The current baseline is a compact LLM training loop, but the broader purpose is to make it easy for agents to propose changes, run bounded experiments, evaluate results against clear metrics, and preserve only changes that demonstrate improvement.

## Core Files

- `prepare.py` contains fixed constants, one-time data preparation, tokenizer training, dataloading, and evaluation utilities. Do not modify it unless explicitly asked.
- `train.py` contains the GPT model, optimizer, and training loop. This is the main file agents may modify during experiments.
- `program.md` contains the agent instructions/research program. Humans iterate on this file to change the research organization behavior.
- `pyproject.toml` contains Python dependencies.

## Setup

- Requires a single NVIDIA GPU, Python 3.10+, and `mise`.
- Install `uv` globally with `mise use -g asdf:asdf-community/asdf-uv@0.11.9`.
- Install dependencies with `uv sync`.
- Prepare data once with `uv run prepare.py`.
- Run one experiment with `uv run train.py`.

## Experiment Rules

- Optimize for lower `val_bpb`; lower is better.
- Training uses a fixed 5-minute wall-clock budget, excluding startup and compilation.
- Keep experiments comparable across runs by respecting the fixed time budget.
- Prefer small, reviewable changes focused on `train.py`.
- Do not add distributed training or broad framework changes unless explicitly requested.
- If targeting smaller compute platforms, consider lower model depth, sequence length, vocab size, eval tokens, and batch size.

## Commit Standard

Use the [Conventional Commits](https://www.conventionalcommits.org/) standard for all commit messages.
