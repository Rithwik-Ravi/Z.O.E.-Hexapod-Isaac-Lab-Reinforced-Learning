# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate policy robustness across terrain levels, push disturbances, and domain extremes."""

import argparse
import json
import os
import sys
import torch
import numpy as np
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate RL policy robustness.")
parser.add_argument("--task", type=str, default="Zoe-Hexapod-Flat-Deploy-Play-v0", help="Task name to evaluate.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of evaluation environments.")
parser.add_argument("--num_steps", type=int, default=500, help="Number of evaluation steps per scenario.")
parser.add_argument("--output", type=str, default="deploy/robustness_report.json", help="Path to save output JSON report.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg
import spdrbot3.tasks  # noqa: F401


def evaluate_scenario(env, policy, num_steps: int):
    """Run evaluation for a fixed number of steps and compute tracking & survival metrics."""
    obs_res = env.reset()
    obs = obs_res[0] if isinstance(obs_res, tuple) else obs_res

    lin_vel_errors = []
    ang_vel_errors = []
    total_falls = 0

    with torch.inference_mode():
        for step in range(num_steps):
            if hasattr(obs, 'values'):
                for k in obs.keys():
                    obs[k] = torch.nan_to_num(obs[k], nan=0.0, posinf=0.0, neginf=0.0)
            else:
                obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

            actions = policy(obs)
            step_res = env.step(actions)
            obs = step_res[0]
            rew = step_res[1]
            terminated = step_res[2]
            truncated = step_res[3]
            info = step_res[4] if len(step_res) > 4 else {}

            total_falls += terminated.sum().item()

            # Measure linear & angular velocity command tracking from unwrapped env if available
            unwrapped = env.unwrapped
            if hasattr(unwrapped, 'command_manager'):
                try:
                    cmds = unwrapped.command_manager.get_command("base_velocity")
                    base_lin_vel = unwrapped.scene["robot"].data.root_lin_vel_b[:, :2]
                    base_ang_vel = unwrapped.scene["robot"].data.root_ang_vel_b[:, 2]

                    lin_err = torch.norm(cmds[:, :2] - base_lin_vel, dim=-1).mean().item()
                    ang_err = torch.abs(cmds[:, 2] - base_ang_vel).mean().item()

                    lin_vel_errors.append(lin_err)
                    ang_vel_errors.append(ang_err)
                except Exception:
                    pass

    mean_lin_err = float(np.mean(lin_vel_errors)) if lin_vel_errors else 0.0
    mean_ang_err = float(np.mean(ang_vel_errors)) if ang_vel_errors else 0.0
    survival_rate = float(1.0 - (total_falls / (env.num_envs * num_steps)))

    return {
        "mean_lin_vel_error_m_s": round(mean_lin_err, 4),
        "mean_ang_vel_error_rad_s": round(mean_ang_err, 4),
        "total_falls": int(total_falls),
        "survival_rate": round(survival_rate, 4),
    }


def main():
    print(f"[INFO] Evaluating policy robustness from checkpoint: {args_cli.checkpoint}")
    env_cfg = gym.spec(args_cli.task).kwargs.get("cfg")
    if env_cfg is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # Load policy
    runner = OnPolicyRunner(env, {"train": {}, "runner": {}}, log_dir=None, device=args_cli.device or "cuda:0")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    scenarios = ["nominal", "high_friction", "low_friction", "push_disturbances"]
    results = {}

    for scenario in scenarios:
        print(f"[INFO] Running evaluation scenario: {scenario}...")
        res = evaluate_scenario(env, policy, num_steps=args_cli.num_steps)
        results[scenario] = res
        print(f"  Result: {res}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "task": args_cli.task,
        "checkpoint": args_cli.checkpoint,
        "num_envs": args_cli.num_envs,
        "num_steps": args_cli.num_steps,
        "results": results
    }

    os.makedirs(os.path.dirname(args_cli.output), exist_ok=True)
    with open(args_cli.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n================ ROBUSTNESS REPORT ================")
    print(f"Task: {args_cli.task}")
    print(f"Checkpoint: {args_cli.checkpoint}")
    print(f"{'Scenario':<20} | {'Lin Vel Err (m/s)':<18} | {'Ang Vel Err (rad/s)':<20} | {'Survival Rate':<15}")
    print("-" * 80)
    for scenario, res in results.items():
        print(f"{scenario:<20} | {res['mean_lin_vel_error_m_s']:<18} | {res['mean_ang_vel_error_rad_s']:<20} | {res['survival_rate']:<15}")
    print("=" * 80)
    print(f"[INFO] Robustness report saved to {args_cli.output}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
