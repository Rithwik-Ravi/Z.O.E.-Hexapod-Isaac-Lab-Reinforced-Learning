# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import os
import random
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint file path (direct path or filename/pattern) to resume from. Automatically sets resume=True.",
    )
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )
    # -- watchdog arguments
    arg_group.add_argument(
        "--watchdog_iterations",
        type=int,
        default=150,
        help="Early-abort watchdog iteration count (default: 150, 0 to disable).",
    )


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlOnPolicyRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs."""
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    rslrl_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def update_rsl_rl_cfg(agent_cfg: RslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for RSL-RL agent based on inputs."""
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
        agent_cfg.resume = True
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
        agent_cfg.resume = True
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg


def resolve_resume_checkpoint(
    log_root_path: str,
    agent_cfg,
    args_cli: argparse.Namespace,
    get_checkpoint_path_fn: Callable[[str, str, str], str] | None = None,
) -> str | None:
    """Resolve and validate checkpoint path for --resume / --checkpoint.

    - Accepts a direct file path in --checkpoint regardless of experiment_name (and sets agent_cfg.resume = True).
    - If --resume, --load_run, or --checkpoint is given and the file/run does not exist, raises FileNotFoundError.
    - Returns absolute checkpoint path if resuming, or None if training from scratch.
    """
    wants_resume = bool(
        getattr(agent_cfg, "resume", False)
        or getattr(args_cli, "resume", False)
        or getattr(args_cli, "checkpoint", None) is not None
        or getattr(args_cli, "load_run", None) is not None
    )
    if not wants_resume:
        agent_cfg.resume = False
        agent_cfg.resume_from = None
        return None

    agent_cfg.resume = True
    ckpt_arg = getattr(args_cli, "checkpoint", None)
    load_run_arg = getattr(args_cli, "load_run", None)

    # 1. Direct file path via --checkpoint (works across any experiment_name)
    if ckpt_arg is not None:
        if os.path.isfile(ckpt_arg):
            resolved = os.path.abspath(ckpt_arg)
            agent_cfg.resume_from = resolved
            return resolved
        # If it looks like an explicit path (contains slash or parent dir), fail immediately if missing
        if os.sep in ckpt_arg or (os.altsep and os.altsep in ckpt_arg):
            raise FileNotFoundError(
                f"Checkpoint file specified via --checkpoint does not exist: '{ckpt_arg}' "
                f"(resolved: '{os.path.abspath(ckpt_arg)}')"
            )

    # 2. Direct run directory path via --load_run
    search_root = log_root_path
    run_pattern = getattr(agent_cfg, "load_run", ".*")
    ckpt_pattern = getattr(agent_cfg, "load_checkpoint", "model_.*.pt")

    if load_run_arg is not None and os.path.isdir(load_run_arg):
        abs_run = os.path.abspath(load_run_arg)
        search_root = os.path.dirname(abs_run)
        run_pattern = os.path.basename(abs_run)

    if not os.path.isdir(search_root):
        raise FileNotFoundError(
            f"Resume requested (--resume/--load_run/--checkpoint), but log root directory does not exist: '{search_root}'"
        )

    if get_checkpoint_path_fn is None:
        raise FileNotFoundError(
            f"Cannot resolve checkpoint pattern (root='{search_root}', run='{run_pattern}', ckpt='{ckpt_pattern}')"
        )

    try:
        resolved = get_checkpoint_path_fn(search_root, run_pattern, ckpt_pattern)
    except Exception as exc:
        raise FileNotFoundError(
            f"Failed to locate checkpoint in '{search_root}' (load_run='{run_pattern}', checkpoint='{ckpt_pattern}'): {exc}"
        ) from exc

    if not resolved or not os.path.isfile(resolved):
        raise FileNotFoundError(f"Resolved checkpoint path is not an existing file: '{resolved}'")

    resolved = os.path.abspath(resolved)
    agent_cfg.resume_from = resolved
    return resolved


def evaluate_watchdog(
    iters_run: int,
    watchdog_iters: int,
    mean_ep_len: float,
    time_out_rate: float,
    reward_history: list[float],
    learning_rate: float | None = None,
) -> tuple[bool, list[str], str | None]:
    """Evaluate A5(d) early-abort watchdog conditions and LR warning.

    Returns:
        (should_abort, abort_reasons, lr_warning)
    """
    lr_warning = None
    if learning_rate is not None and learning_rate < 1e-5:
        lr_warning = f"[WATCHDOG WARNING] Iteration {iters_run}: learning_rate ({learning_rate:.2e}) collapsed below 1e-5!"

    if watchdog_iters <= 0 or iters_run < watchdog_iters:
        return False, [], lr_warning

    reasons = []
    if mean_ep_len < 100.0:
        reasons.append(f"mean_episode_length ({mean_ep_len:.2f}) < 100.0 steps")

    if time_out_rate < 0.05:
        reasons.append(f"time_out ({time_out_rate * 100.0:.2f}%) < 5.0%")

    if iters_run == watchdog_iters and len(reward_history) >= 3:
        n = len(reward_history)
        third = max(1, n // 3)
        w1 = sum(reward_history[:third]) / float(third)
        w2 = sum(reward_history[third : 2 * third]) / float(third)
        w3 = sum(reward_history[2 * third :]) / float(len(reward_history[2 * third :]))
        if w3 < w2 < w1:
            reasons.append(f"mean_reward is still falling (early={w1:.3f} -> mid={w2:.3f} -> late={w3:.3f})")

    return len(reasons) > 0, reasons, lr_warning
