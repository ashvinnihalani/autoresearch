#!/usr/bin/env python3
"""Prepare Qwen/Qwen3.5-4B assets for the async-rl workstream."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from async_rl_config import (
    DATASET_FILENAME,
    DATASET_ID,
    DATASET_LOCAL_NAME,
    MODEL_ARGS,
    MODEL_ID,
    MODEL_LOCAL_NAME,
    SLIME_COMMIT,
    SLIME_REPO,
    default_megatron_dir,
    default_storage_root,
    path_exists,
)


def path_arg(value: str) -> Path:
    return Path(value).expanduser()


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, dry_run: bool = False):
    printable = shlex.join(str(part) for part in cmd)
    if cwd is not None:
        printable = f"(cd {shlex.quote(str(cwd))} && {printable})"
    print(f"$ {printable}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def hf_cli(*, dry_run: bool = False) -> str:
    for name in ("hf", "huggingface-cli"):
        if shutil.which(name):
            return name
    if dry_run:
        return "hf"
    raise SystemExit("Missing Hugging Face CLI. Install huggingface_hub or use the slime Docker image.")


def ensure_slime(slime_dir: Path, *, update: bool, dry_run: bool):
    train_async = slime_dir / "train_async.py"
    if path_exists(train_async):
        print(f"slime: using existing checkout at {slime_dir}")
        if update and path_exists(slime_dir / ".git"):
            run(["git", "fetch", "origin", SLIME_COMMIT], cwd=slime_dir, dry_run=dry_run)
            run(["git", "checkout", SLIME_COMMIT], cwd=slime_dir, dry_run=dry_run)
            run([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"], cwd=slime_dir, dry_run=dry_run)
        return

    if path_exists(slime_dir):
        try:
            has_files = any(slime_dir.iterdir())
        except OSError as exc:
            raise SystemExit(f"Cannot inspect {slime_dir}: {exc}") from exc
        if has_files:
            raise SystemExit(f"{slime_dir} exists but does not look like a slime checkout")

    run(["git", "clone", "--filter=blob:none", SLIME_REPO, str(slime_dir)], dry_run=dry_run)
    run(["git", "checkout", SLIME_COMMIT], cwd=slime_dir, dry_run=dry_run)
    run([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"], cwd=slime_dir, dry_run=dry_run)


def download_hf(repo_id: str, local_dir: Path, *, repo_type: str | None, dry_run: bool):
    if not dry_run:
        local_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [hf_cli(dry_run=dry_run), "download"]
    if repo_type:
        cmd += ["--repo-type", repo_type]
    cmd += [repo_id, "--local-dir", str(local_dir)]
    run(cmd, dry_run=dry_run)


def convert_checkpoint(args):
    if args.ref_load.exists() and any(args.ref_load.iterdir()) and not args.force_convert:
        print(f"checkpoint: converted Megatron checkpoint already exists at {args.ref_load}")
        return

    if not (args.model_dir / "config.json").exists() and not args.dry_run:
        raise SystemExit(f"Missing model files in {args.model_dir}; run without --skip-model-download first")

    if not args.dry_run:
        args.ref_load.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    pythonpath = [str(args.megatron_dir), str(args.slime_dir)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    convert_script = args.slime_dir / "tools" / "convert_hf_to_torch_dist.py"
    cmd = [
        sys.executable,
        str(convert_script),
        *MODEL_ARGS,
        "--hf-checkpoint",
        str(args.model_dir),
        "--save",
        str(args.ref_load),
    ]
    run(cmd, env=env, dry_run=args.dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=path_arg, default=default_storage_root())
    parser.add_argument("--slime-dir", type=path_arg)
    parser.add_argument("--megatron-dir", type=path_arg, default=default_megatron_dir())
    parser.add_argument("--model-dir", type=path_arg)
    parser.add_argument("--dataset-dir", type=path_arg)
    parser.add_argument("--ref-load", type=path_arg)
    parser.add_argument("--skip-slime", action="store_true", help="Do not clone or update slime.")
    parser.add_argument("--update-slime", action="store_true", help="Checkout the pinned slime commit if possible.")
    parser.add_argument("--skip-model-download", action="store_true")
    parser.add_argument("--skip-dataset-download", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.storage_root = args.storage_root.expanduser()
    if args.slime_dir is None:
        root_slime = Path("/root/slime")
        args.slime_dir = root_slime if path_exists(root_slime / "train_async.py") else args.storage_root / "slime"
    if args.model_dir is None:
        args.model_dir = args.storage_root / "models" / MODEL_LOCAL_NAME
    if args.dataset_dir is None:
        args.dataset_dir = args.storage_root / "data" / DATASET_LOCAL_NAME
    if args.ref_load is None:
        args.ref_load = args.storage_root / "checkpoints" / f"{MODEL_LOCAL_NAME}_torch_dist"
    return args


def main():
    args = parse_args()
    print(f"storage: {args.storage_root}")
    print(f"model:   {MODEL_ID} -> {args.model_dir}")
    print(f"dataset: {DATASET_ID} -> {args.dataset_dir}")
    print(f"slime:   {args.slime_dir}")

    if not args.skip_slime:
        ensure_slime(args.slime_dir, update=args.update_slime, dry_run=args.dry_run)

    if not args.skip_model_download:
        download_hf(MODEL_ID, args.model_dir, repo_type=None, dry_run=args.dry_run)

    if not args.skip_dataset_download:
        download_hf(DATASET_ID, args.dataset_dir, repo_type="dataset", dry_run=args.dry_run)

    if not args.skip_convert:
        convert_checkpoint(args)

    prompt_path = args.dataset_dir / DATASET_FILENAME
    print("")
    print("Done. Expected training inputs:")
    print(f"  --model-dir   {args.model_dir}")
    print(f"  --ref-load    {args.ref_load}")
    print(f"  --prompt-data {prompt_path}")


if __name__ == "__main__":
    main()
