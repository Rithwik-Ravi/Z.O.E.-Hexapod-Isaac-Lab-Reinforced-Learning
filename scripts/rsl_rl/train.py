# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument(
    "--disable-nan-sanitizer", action="store_true", default=False, help="Disable physics NaN/Inf sanitizer."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# A5(b) Early validation before launching Omniverse: if an explicit file path is passed to --checkpoint
# and does not exist, fail loudly immediately with exit code 2.
if args_cli.checkpoint is not None and (
    os.sep in args_cli.checkpoint or (os.altsep and os.altsep in args_cli.checkpoint)
):
    if not os.path.isfile(args_cli.checkpoint):
        print(
            f"[ERROR] Checkpoint file specified via --checkpoint does not exist: '{args_cli.checkpoint}'",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(2)

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import collections
import gymnasium as gym
import pickle
import statistics
import torch
import yaml
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml


def dump_pickle(filepath: str, data: any):
    """Save data to pickle file."""
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import spdrbot3.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # A5(a) & A5(b): Resolve resume checkpoint BEFORE creating log_dir or environment.
    # Fails loudly with exit code 2 if --resume / --load_run / --checkpoint cannot be resolved.
    resume_path = None
    if (
        agent_cfg.resume
        or args_cli.resume
        or args_cli.checkpoint is not None
        or args_cli.load_run is not None
        or agent_cfg.algorithm.class_name == "Distillation"
    ):
        try:
            resume_path = cli_args.resolve_resume_checkpoint(
                log_root_path, agent_cfg, args_cli, get_checkpoint_path_fn=get_checkpoint_path
            )
        except Exception as exc:
            print(f"\n[ERROR] Resume checkpoint resolution failed: {exc}", file=sys.stderr, flush=True)
            sys.stdout.flush()
            os._exit(2)
    else:
        agent_cfg.resume_from = None

    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Monkey-patch environment to sanitize NaN/Inf from physics explosions unless disabled
    if not args_cli.disable_nan_sanitizer:
        nan_count = [0]
        _original_step = env.unwrapped.step

        def _safe_step(action):
            obs, rew, terminated, truncated, info = _original_step(action)
            has_nan = False
            for k, v in obs.items():
                if torch.isnan(v).any() or torch.isinf(v).any():
                    has_nan = True
                    obs[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            if torch.isnan(rew).any() or torch.isinf(rew).any():
                has_nan = True
                rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
            if has_nan:
                nan_count[0] += 1
            return obs, rew, terminated, truncated, info

        env.unwrapped.step = _safe_step

        _original_reset = env.unwrapped.reset

        def _safe_reset(**kwargs):
            obs, info = _original_reset(**kwargs)
            for k, v in obs.items():
                obs[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            return obs, info

        env.unwrapped.reset = _safe_reset

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # Monkey-patch ActorCritic to prevent NaN std from crashing training.
    import types
    from torch.distributions import Normal as _Normal

    def _safe_update_distribution(self, obs):
        if self.state_dependent_std:
            mean_and_std = self.actor(obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
        else:
            mean = self.actor(obs)
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        std = torch.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
        std = torch.clamp(std, min=1e-6)
        self.distribution = _Normal(mean, std)

    runner.alg.policy._update_distribution = types.MethodType(_safe_update_distribution, runner.alg.policy)

    # write git state to logs
    runner.add_git_repo_to_log(__file__)

    # A5(c): Load the checkpoint and print prominent banner
    if resume_path is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)
        loaded_iter = int(getattr(runner, "current_learning_iteration", 0))
        print("\n" + "=" * 80)
        print(f"RESUMED from {resume_path} (iteration {loaded_iter})")
        print("=" * 80 + "\n", flush=True)
    else:
        print("\n" + "=" * 80)
        print("FROM SCRATCH")
        print("=" * 80 + "\n", flush=True)

    # dump the configuration into log-directory
    agent_yaml_path = os.path.join(log_dir, "params", "agent.yaml")
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(agent_yaml_path, agent_cfg)
    # Ensure resume_from is explicitly persisted in params/agent.yaml
    if os.path.isfile(agent_yaml_path):
        with open(agent_yaml_path, "r") as f:
            agent_yaml_data = yaml.safe_load(f) or {}
        agent_yaml_data["resume_from"] = resume_path
        with open(agent_yaml_path, "w") as f:
            yaml.safe_dump(agent_yaml_data, f, sort_keys=False)

    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # A5(d): Early-abort watchdog hook
    watchdog_iters = int(getattr(args_cli, "watchdog_iterations", 150))
    start_iter = int(getattr(runner, "current_learning_iteration", 0))
    reward_history: list[float] = []
    timeout_window: collections.deque[float] = collections.deque(maxlen=25)
    _orig_runner_log = runner.log

    def _watchdog_log(locs: dict):
        _orig_runner_log(locs)
        cur_it = int(locs.get("it", start_iter))
        iters_run = cur_it - start_iter + 1

        lenbuf = locs.get("lenbuffer", getattr(runner, "lenbuffer", []))
        rewbuf = locs.get("rewbuffer", getattr(runner, "rewbuffer", []))
        mean_ep_len = float(statistics.mean(lenbuf)) if len(lenbuf) > 0 else 0.0
        mean_rew = float(statistics.mean(rewbuf)) if len(rewbuf) > 0 else 0.0
        reward_history.append(mean_rew)

        ep_infos = locs.get("ep_infos", [])
        to_vals = []
        for ep_info in ep_infos:
            if "Terrain/Survival" in ep_info:
                val = ep_info["Terrain/Survival"]
                to_vals.append(float(val.item() if hasattr(val, "item") else val))
                continue
            term_keys = [k for k in ep_info.keys() if k.startswith("Episode_Termination/")]
            if term_keys:
                tot_term = 0.0
                to_term = 0.0
                for k in term_keys:
                    v = ep_info[k]
                    fv = float(v.item() if hasattr(v, "item") else v)
                    tot_term += fv
                    if k.endswith("time_out"):
                        to_term = fv
                if tot_term > 1e-6:
                    to_vals.append(to_term / tot_term)
        if to_vals:
            timeout_window.append(sum(to_vals) / float(len(to_vals)))

        if len(timeout_window) > 0:
            time_out_rate = sum(timeout_window) / float(len(timeout_window))
        else:
            time_out_rate = 1.0 if mean_ep_len >= 100.0 else 0.0

        lr = getattr(runner.alg, "learning_rate", None)
        should_abort, reasons, lr_warning = cli_args.evaluate_watchdog(
            iters_run=iters_run,
            watchdog_iters=watchdog_iters,
            mean_ep_len=mean_ep_len,
            time_out_rate=time_out_rate,
            reward_history=reward_history,
            learning_rate=lr,
        )
        if lr_warning is not None:
            print(lr_warning, flush=True)
        if should_abort:
            print("\n" + "!" * 80, file=sys.stderr)
            print(
                f"[WATCHDOG ABORT] Training stopped at iteration {cur_it} ({iters_run} iterations run):",
                file=sys.stderr,
            )
            for reason in reasons:
                print(f"  - {reason}", file=sys.stderr)
            print("!" * 80 + "\n", file=sys.stderr, flush=True)
            sys.stdout.flush()
            os._exit(3)

    runner.log = _watchdog_log

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"[INFO] Training complete. Saved checkpoints to: {log_dir}")
    sys.stdout.flush()

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
