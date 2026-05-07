"""Shared constants for the async-rl workstream."""

from __future__ import annotations

import os
from pathlib import Path

SLIME_REPO = "https://github.com/THUDM/slime.git"
SLIME_COMMIT = "82007faf4b398abd32bd8e07f9638f6cfeb70729"

MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_LOCAL_NAME = "Qwen3.5-4B"

DATASET_ID = "zhuzilin/dapo-math-17k"
DATASET_LOCAL_NAME = "dapo-math-17k"
DATASET_FILENAME = "dapo-math-17k.jsonl"

# Mirrored from THUDM/slime scripts/models/qwen3.5-4B.sh at SLIME_COMMIT.
MODEL_ARGS = [
    "--spec",
    "slime_plugins.models.qwen3_5",
    "get_qwen3_5_spec",
    "--disable-bias-linear",
    "--qk-layernorm",
    "--group-query-attention",
    "--num-attention-heads",
    "16",
    "--num-query-groups",
    "4",
    "--kv-channels",
    "256",
    "--num-layers",
    "32",
    "--hidden-size",
    "2560",
    "--ffn-hidden-size",
    "9216",
    "--use-gated-attention",
    "--normalization",
    "RMSNorm",
    "--apply-layernorm-1p",
    "--position-embedding-type",
    "rope",
    "--norm-epsilon",
    "1e-6",
    "--rotary-percent",
    "0.25",
    "--swiglu",
    "--vocab-size",
    "248320",
    "--rotary-base",
    "10000000",
    "--attention-output-gate",
]


def path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def default_storage_root() -> Path:
    if storage := os.environ.get("ASYNC_RL_STORAGE"):
        return Path(storage).expanduser()
    instance_storage = Path("/tmp/instance_storage")
    if instance_storage.is_dir():
        return instance_storage / "async-rl"
    return Path.home() / ".cache" / "autoresearch" / "async-rl"


def default_slime_dir() -> Path:
    if slime_dir := os.environ.get("SLIME_DIR"):
        return Path(slime_dir).expanduser()
    root_slime = Path("/root/slime")
    if path_exists(root_slime / "train_async.py"):
        return root_slime
    return default_storage_root() / "slime"


def default_megatron_dir() -> Path:
    return Path(os.environ.get("MEGATRON_DIR", "/root/Megatron-LM")).expanduser()


def default_model_dir() -> Path:
    return default_storage_root() / "models" / MODEL_LOCAL_NAME


def default_dataset_dir() -> Path:
    return default_storage_root() / "data" / DATASET_LOCAL_NAME


def default_prompt_data() -> Path:
    return default_dataset_dir() / DATASET_FILENAME


def default_ref_load() -> Path:
    return default_storage_root() / "checkpoints" / f"{MODEL_LOCAL_NAME}_torch_dist"


def default_train_checkpoint_dir() -> Path:
    return default_storage_root() / "checkpoints" / f"{MODEL_LOCAL_NAME}_slime"
