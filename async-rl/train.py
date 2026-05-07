#!/usr/bin/env python3
"""Launch slime fully async RL for Qwen/Qwen3.5-4B on 8 GPUs."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

from async_rl_config import (
    MODEL_ARGS,
    default_megatron_dir,
    default_model_dir,
    default_prompt_data,
    default_ref_load,
    default_slime_dir,
    default_train_checkpoint_dir,
)


def path_arg(value: str) -> Path:
    return Path(value).expanduser()


def run(cmd: list[str], *, dry_run: bool = False, check: bool = True):
    print(f"$ {shlex.join(str(part) for part in cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=check)


def detect_nvlink() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return "0"
    return "1" if re.search(r"NV[0-9]*", result.stdout) else "0"


def build_runtime_env(args: argparse.Namespace) -> str:
    workstream_dir = Path(__file__).resolve().parent
    pythonpath = [str(args.megatron_dir), str(workstream_dir), str(args.slime_dir)]
    if os.environ.get("PYTHONPATH"):
        pythonpath.append(os.environ["PYTHONPATH"])

    env_vars = {
        "PYTHONPATH": os.pathsep.join(pythonpath),
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": detect_nvlink(),
        "PYTHONUNBUFFERED": "1",
    }
    if os.environ.get("HF_HOME"):
        env_vars["HF_HOME"] = os.environ["HF_HOME"]
    if os.environ.get("WANDB_PROJECT"):
        env_vars["WANDB_PROJECT"] = os.environ["WANDB_PROJECT"]
    else:
        env_vars["WANDB_PROJECT"] = "async-rl"

    return json.dumps({"env_vars": env_vars})


def build_slime_args(args: argparse.Namespace) -> list[str]:
    ckpt_args = [
        "--hf-checkpoint",
        str(args.model_dir),
        "--ref-load",
        str(args.ref_load),
        "--load",
        str(args.load),
        "--save",
        str(args.save),
        "--save-interval",
        str(args.save_interval),
    ]

    rollout_args = [
        "--rollout-function-path",
        "fully_async_rollout.generate_rollout_fully_async",
        "--prompt-data",
        str(args.prompt_data),
        "--input-key",
        args.input_key,
        "--label-key",
        args.label_key,
        "--apply-chat-template",
        "--rollout-shuffle",
        "--rm-type",
        args.rm_type,
        "--reward-key",
        args.reward_key,
        "--num-rollout",
        str(args.num_rollout),
        "--rollout-batch-size",
        str(args.rollout_batch_size),
        "--n-samples-per-prompt",
        str(args.n_samples_per_prompt),
        "--rollout-max-response-len",
        str(args.rollout_max_response_len),
        "--rollout-temperature",
        str(args.rollout_temperature),
        "--global-batch-size",
        str(args.global_batch_size),
        "--balance-data",
    ]

    perf_args = [
        "--tensor-model-parallel-size",
        str(args.tensor_model_parallel_size),
        "--sequence-parallel",
        "--pipeline-model-parallel-size",
        "1",
        "--context-parallel-size",
        "1",
        "--expert-model-parallel-size",
        "1",
        "--expert-tensor-parallel-size",
        "1",
        "--recompute-granularity",
        "full",
        "--recompute-method",
        "uniform",
        "--recompute-num-layers",
        "1",
        "--use-dynamic-batch-size",
        "--max-tokens-per-gpu",
        str(args.max_tokens_per_gpu),
    ]

    grpo_args = [
        "--advantage-estimator",
        "grpo",
        "--use-kl-loss",
        "--kl-loss-coef",
        str(args.kl_loss_coef),
        "--kl-loss-type",
        "low_var_kl",
        "--entropy-coef",
        str(args.entropy_coef),
        "--eps-clip",
        str(args.eps_clip),
        "--eps-clip-high",
        str(args.eps_clip_high),
        "--use-tis",
    ]

    optimizer_args = [
        "--optimizer",
        "adam",
        "--lr",
        str(args.lr),
        "--lr-decay-style",
        "constant",
        "--weight-decay",
        str(args.weight_decay),
        "--adam-beta1",
        str(args.adam_beta1),
        "--adam-beta2",
        str(args.adam_beta2),
    ]

    sglang_args = [
        "--rollout-num-gpus-per-engine",
        str(args.rollout_num_gpus_per_engine),
    ]

    misc_args = [
        "--attention-dropout",
        "0.0",
        "--hidden-dropout",
        "0.0",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
        "--attention-backend",
        args.attention_backend,
    ]

    extra_args = args.extra_arg or []
    return [
        "--actor-num-nodes",
        str(args.actor_num_nodes),
        "--actor-num-gpus-per-node",
        str(args.actor_num_gpus_per_node),
        "--rollout-num-gpus",
        str(args.rollout_num_gpus),
        *MODEL_ARGS,
        *ckpt_args,
        *rollout_args,
        *optimizer_args,
        *grpo_args,
        *perf_args,
        *sglang_args,
        *misc_args,
        *extra_args,
    ]


def build_ray_job_command(args: argparse.Namespace) -> list[str]:
    train_async = args.slime_dir / "train_async.py"
    return [
        "ray",
        "job",
        "submit",
        f"--address={args.ray_address}",
        f"--runtime-env-json={build_runtime_env(args)}",
        "--",
        "python3",
        str(train_async),
        *build_slime_args(args),
    ]


def cleanup_ray(dry_run: bool):
    run(["pkill", "-9", "sglang"], dry_run=dry_run, check=False)
    run(["ray", "stop", "--force"], dry_run=dry_run, check=False)
    run(["pkill", "-9", "ray"], dry_run=dry_run, check=False)
    if not dry_run:
        time.sleep(3)


def validate_inputs(args: argparse.Namespace):
    missing = []
    if not (args.slime_dir / "train_async.py").exists():
        missing.append(f"slime train_async.py under {args.slime_dir}")
    if not args.prompt_data.exists():
        missing.append(f"prompt data at {args.prompt_data}")
    if not (args.model_dir / "config.json").exists():
        missing.append(f"Hugging Face model files in {args.model_dir}")
    if not args.ref_load.exists() or not any(args.ref_load.iterdir()):
        missing.append(f"Megatron torch_dist checkpoint in {args.ref_load}")

    if missing:
        message = "Missing required inputs:\n  - " + "\n  - ".join(missing)
        message += "\nRun `python prepare.py` inside the async-rl container first."
        if args.dry_run:
            print(message)
        else:
            raise SystemExit(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slime-dir", type=path_arg, default=default_slime_dir())
    parser.add_argument("--megatron-dir", type=path_arg, default=default_megatron_dir())
    parser.add_argument("--model-dir", type=path_arg, default=default_model_dir())
    parser.add_argument("--ref-load", type=path_arg, default=default_ref_load())
    parser.add_argument("--load", type=path_arg, default=default_train_checkpoint_dir())
    parser.add_argument("--save", type=path_arg, default=default_train_checkpoint_dir())
    parser.add_argument("--prompt-data", type=path_arg, default=default_prompt_data())
    parser.add_argument("--input-key", default="prompt")
    parser.add_argument("--label-key", default="label")
    parser.add_argument("--rm-type", default="dapo")
    parser.add_argument("--reward-key", default="score")

    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--actor-num-nodes", type=int, default=1)
    parser.add_argument("--actor-num-gpus-per-node", type=int, default=4)
    parser.add_argument("--rollout-num-gpus", type=int, default=4)
    parser.add_argument("--rollout-num-gpus-per-engine", type=int, default=1)
    parser.add_argument("--tensor-model-parallel-size", type=int, default=2)
    parser.add_argument("--max-tokens-per-gpu", type=int, default=9216)

    parser.add_argument("--num-rollout", type=int, default=3000)
    parser.add_argument("--rollout-batch-size", type=int, default=32)
    parser.add_argument("--n-samples-per-prompt", type=int, default=8)
    parser.add_argument("--rollout-max-response-len", type=int, default=8192)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--global-batch-size", type=int, default=256)

    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.98)
    parser.add_argument("--kl-loss-coef", type=float, default=0.0)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--eps-clip-high", type=float, default=0.28)
    parser.add_argument("--save-interval", type=int, default=20)
    parser.add_argument("--attention-backend", default="flash")

    parser.add_argument("--master-addr", default=os.environ.get("MASTER_ADDR", "127.0.0.1"))
    parser.add_argument("--ray-address", default="http://127.0.0.1:8265")
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--no-ray-start", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra-arg", action="append", default=[], help="Append one raw slime argument.")
    return parser.parse_args()


def main():
    args = parse_args()
    validate_inputs(args)

    if not args.no_cleanup:
        cleanup_ray(args.dry_run)

    if not args.no_ray_start:
        run(
            [
                "ray",
                "start",
                "--head",
                "--node-ip-address",
                args.master_addr,
                "--num-gpus",
                str(args.num_gpus),
                "--disable-usage-stats",
            ],
            dry_run=args.dry_run,
        )

    run(build_ray_job_command(args), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
