# autoresearch Agent Guide

## Purpose

This file is for coding agents working in this repository. It defines operating rules, workstream boundaries, and commit expectations.

Do not treat this as user-facing project documentation. The root `README.md` is the public workstream index, and each workstream folder owns its own `README.md` for human-facing setup, usage, and context plus its own `PROGRAM.md` for agent-facing research instructions.

## Documentation Roles

- `AGENTS.md` — agent-facing instructions for how to work in the repo.
- `README.md` — repo-facing overview and index of available workstream folders.
- `<workstream>/README.md` — human-facing documentation for one workstream, including setup, commands, design notes, and container usage.
- `<workstream>/PROGRAM.md` — agent-facing research-program instructions used by agents running experiments in that workstream.

## Repository Layout Rules

This repo is organized as folder-level workstreams. The root should stay lightweight: keep the workstream catalog in `README.md`, and keep detailed human-facing setup or usage instructions inside each workstream's own `README.md`.

## Working Rules

- Make changes inside the relevant workstream folder unless the user asks for a repo-wide change.
- Keep root-level files focused on repo organization, contributor guidance, and workstream indexing.
- When adding a new workstream, create a new folder and include a workstream-level `README.md` and `PROGRAM.md`.
- Update the root `README.md` whenever workstream folders are added, renamed, or removed.
- Keep workstream dependencies isolated in that workstream folder.
- Put workstream-specific Dockerfiles inside the workstream folder, not at the repo root.

## Workstream-Specific Rules

Avoid duplicating workstream setup, metrics, run budgets, or usage notes here. Put human-facing details in `<workstream>/README.md` and agent-facing research instructions in `<workstream>/PROGRAM.md`.

Agent-only constraints that apply while editing or running a workstream:

- Read the workstream-level `PROGRAM.md` before making changes or running experiments inside a workstream folder.
- Use the workstream-level `README.md` for human-facing setup and context when needed, but do not treat it as the agent program.
- Follow any file ownership rules described by that workstream.
- Keep dependency changes scoped to the workstream folder.
- Keep Docker changes scoped to the workstream folder.
- Preserve the workstream's documented comparison metric and experiment run budget unless the user explicitly asks to change them.

## WandB MCP Server

A WandB MCP server is available for inspecting experiment runs, metrics, artifacts, and comparisons. When the task involves WandB/W&B run state, logged metrics, dashboards, or experiment history, try to use the WandB MCP tools before relying on local logs or manual summaries.

If the WandB MCP server is unavailable, inaccessible, unauthenticated, or returns a permissions error, warn the user clearly that WandB MCP access is not working and explain what could not be checked. Continue with local files, terminal output, or code inspection only when that still provides useful progress.

## Docker Dev Containers

Each workstream should have at most one long-running dev container that agents reuse for commands and research runs. The container must be easy for a newly started agent to discover from Docker alone.

Use this convention:

- Container name: `autoresearch-<workstream>-dev`
- Image tag: `autoresearch-<workstream>:local`
- Working directory inside the container: `/workspace`
- Bind mount: workstream folder mounted to `/workspace`
- Instance storage: if `/tmp/instance_storage` exists on the host, mount it at `/tmp/instance_storage` in the container.
- GPU access: required via `--gpus all`
- Required labels:
  - `autoresearch.workstream=<workstream>`
  - `autoresearch.role=dev`

Before executing commands or research related to a specific workstream, check whether its dev container is already running:

```bash
docker ps --filter "name=^/autoresearch-<workstream>-dev$" --filter "label=autoresearch.workstream=<workstream>" --filter "label=autoresearch.role=dev"
```

If no matching container is running, build the workstream image if needed and start the dev container from inside the workstream folder:

```bash
docker build -t autoresearch-<workstream>:local .
INSTANCE_STORAGE_ARGS=()
if [ -d /tmp/instance_storage ]; then
  INSTANCE_STORAGE_ARGS=(-v /tmp/instance_storage:/tmp/instance_storage)
fi

docker run -dit \
  --name autoresearch-<workstream>-dev \
  --label autoresearch.workstream=<workstream> \
  --label autoresearch.role=dev \
  --gpus all \
  -v "$PWD":/workspace \
  "${INSTANCE_STORAGE_ARGS[@]}" \
  -w /workspace \
  autoresearch-<workstream>:local \
  bash
```

After starting the container, verify GPU visibility before running experiments:

```bash
docker exec autoresearch-<workstream>-dev nvidia-smi
```

Run all experiment commands and research runs inside the workstream container, for example:

```bash
docker exec -it autoresearch-<workstream>-dev uv run train.py
```

## Commit Standard

Use the [Conventional Commits](https://www.conventionalcommits.org/) standard for all commit messages.

Before generating a commit message, inspect the entire current git diff state and base the message on the actual file changes. Do not derive commit messages from the current conversation alone.

When an AI attribution trailer is available and known for the agent that made the changes, include it in commit messages:

- Codex: `Co-Authored-By: Codex <codex@openai.com>`
- Claude Code: `Co-Authored-By: Claude <noreply@anthropic.com>`

Do not hard-code an unrelated assistant identity or invent unknown attribution details.
