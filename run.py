#!/usr/bin/env python3
"""Unified entrypoint for TP-Spikformer workflows."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class TaskSpec:
    script: str
    description: str
    inject_eval_flag: bool = False


TASKS = {
    "qk-train": TaskSpec(
        script="qk_drop/train.py",
        description="Train QKFormer.",
    ),
    "qk-eval": TaskSpec(
        script="qk_drop/train.py",
        description="Evaluate QKFormer (auto adds --eval if missing).",
        inject_eval_flag=True,
    ),
    "sdt-train": TaskSpec(
        script="sdt_drop/train_drop.py",
        description="Train SDT drop variant.",
    ),
    "sdt-eval": TaskSpec(
        script="sdt_drop/train_drop.py",
        description="Evaluate SDT drop variant (auto adds --eval if missing).",
        inject_eval_flag=True,
    ),
    "sdtv3-train": TaskSpec(
        script="sdtv3_drop/main_finetune.py",
        description="Train SDTv3 drop variant.",
    ),
    "sdtv3-eval": TaskSpec(
        script="sdtv3_drop/main_finetune.py",
        description="Evaluate SDTv3 drop variant (auto adds --eval if missing).",
        inject_eval_flag=True,
    ),
}


def _has_flag(args: List[str], flag: str) -> bool:
    return any(token == flag or token.startswith(flag + "=") for token in args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified launcher for TP-Spikformer tasks."
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASKS.keys()),
        help="Task name to run.",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available tasks and exit.",
    )
    parser.add_argument(
        "--launcher",
        choices=("python", "torchrun"),
        default="python",
        help="Execution backend (default: python).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter path (default: current interpreter).",
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=1,
        help="Processes per node for torchrun.",
    )
    parser.add_argument("--nnodes", type=int, default=1, help="Node count for torchrun.")
    parser.add_argument(
        "--node_rank", type=int, default=0, help="Node rank for torchrun."
    )
    parser.add_argument(
        "--master_addr",
        default="127.0.0.1",
        help="Master address for torchrun.",
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=29500,
        help="Master port for torchrun.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print final command and exit without executing.",
    )
    return parser


def _print_tasks() -> None:
    print("Available tasks:")
    for name in sorted(TASKS):
        print(f"  - {name:12s} {TASKS[name].description}")


def main() -> int:
    parser = _build_parser()
    args, forwarded_args = parser.parse_known_args()

    if args.list_tasks:
        _print_tasks()
        return 0

    if not args.task:
        parser.error("--task is required unless --list-tasks is used.")

    repo_root = Path(__file__).resolve().parent
    task = TASKS[args.task]
    script_path = repo_root / task.script
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    forwarded_args = list(forwarded_args)
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]

    if task.inject_eval_flag and not _has_flag(forwarded_args, "--eval"):
        forwarded_args = ["--eval"] + forwarded_args

    command: List[str] = []
    if args.launcher == "python":
        command.extend([args.python, str(script_path)])
    else:
        command.extend(
            [
                args.python,
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={args.nproc_per_node}",
                f"--nnodes={args.nnodes}",
                f"--node_rank={args.node_rank}",
                f"--master_addr={args.master_addr}",
                f"--master_port={args.master_port}",
                str(script_path),
            ]
        )

    command.extend(forwarded_args)

    print(f"[run.py] task: {args.task}")
    print(f"[run.py] cwd:  {repo_root}")
    print(f"[run.py] cmd:  {shlex.join(command)}")

    if args.dry_run:
        return 0

    completed = subprocess.run(command, cwd=str(repo_root), check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
