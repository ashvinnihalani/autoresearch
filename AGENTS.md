# autoresearch Agent Guide

## Purpose

This file is for coding agents working in this repository. It defines operating rules, experiment boundaries, and commit expectations.

Do not treat this as user-facing project documentation. The root `README.md` is the public experiment index, and each experiment folder owns its own `README.md` for setup, usage, and experiment-specific context.

## Documentation Roles

- `AGENTS.md` — agent-facing instructions for how to work in the repo.
- `README.md` — repo-facing overview and index of available experiment folders.
- `<experiment>/README.md` — human-facing documentation for one experiment, including setup, commands, design notes, and container usage.
- `<experiment>/program.md` — research-program instructions used by agents running that experiment.

## Repository Layout Rules

This repo is organized as folder-level experiments. The root should stay lightweight: keep the experiment catalog in `README.md`, and keep detailed setup or usage instructions inside each experiment's own `README.md`.

## Working Rules

- Make changes inside the relevant experiment folder unless the user asks for a repo-wide change.
- Keep root-level files focused on repo organization, contributor guidance, and experiment indexing.
- When adding a new experiment, create a new folder and include an experiment-level `README.md`.
- Update the root `README.md` whenever experiment folders are added, renamed, or removed.
- Keep experiment dependencies isolated in that experiment folder.
- Put experiment-specific Dockerfiles inside the experiment folder, not at the repo root.

## Experiment-Specific Rules

Avoid duplicating experiment setup, metrics, or usage notes here. Put those details in `<experiment>/README.md` so humans and agents have one canonical place to read them.

Agent-only constraints that apply while editing an experiment:

- Read the experiment-level `README.md` before making changes inside an experiment folder.
- Follow any file ownership rules described by that experiment.
- Keep dependency changes scoped to the experiment folder.
- Keep Docker changes scoped to the experiment folder.
- Preserve the experiment's documented comparison metric and run budget unless the user explicitly asks to change them.

## Docker Dev Containers

Each experiment should have at most one long-running dev container that agents reuse for commands and research runs. The container must be easy for a newly started agent to discover from Docker alone.

Use this convention:

- Container name: `autoresearch-<experiment>-dev`
- Image tag: `autoresearch-<experiment>:local`
- Working directory inside the container: `/workspace`
- Bind mount: experiment folder mounted to `/workspace`
- Instance storage: if `/tmp/instance_storage` exists on the host, mount it at `/tmp/instance_storage` in the container.
- GPU access: required via `--gpus all`
- Required labels:
  - `autoresearch.experiment=<experiment>`
  - `autoresearch.role=dev`

Before executing commands or research related to a specific experiment, check whether its dev container is already running:

```bash
docker ps --filter "name=^/autoresearch-<experiment>-dev$" --filter "label=autoresearch.experiment=<experiment>" --filter "label=autoresearch.role=dev"
```

If no matching container is running, build the experiment image if needed and start the dev container from inside the experiment folder:

```bash
docker build -t autoresearch-<experiment>:local .
INSTANCE_STORAGE_ARGS=()
if [ -d /tmp/instance_storage ]; then
  INSTANCE_STORAGE_ARGS=(-v /tmp/instance_storage:/tmp/instance_storage)
fi

docker run -dit \
  --name autoresearch-<experiment>-dev \
  --label autoresearch.experiment=<experiment> \
  --label autoresearch.role=dev \
  --gpus all \
  -v "$PWD":/workspace \
  "${INSTANCE_STORAGE_ARGS[@]}" \
  -w /workspace \
  autoresearch-<experiment>:local \
  bash
```

After starting the container, verify GPU visibility before running experiments:

```bash
docker exec autoresearch-<experiment>-dev nvidia-smi
```

Run all experiment commands and research runs inside the experiment container, for example:

```bash
docker exec -it autoresearch-<experiment>-dev uv run train.py
```

## Commit Standard

Use the [Conventional Commits](https://www.conventionalcommits.org/) standard for all commit messages.

Before generating a commit message, inspect the entire current git diff state and base the message on the actual file changes. Do not derive commit messages from the current conversation alone.
