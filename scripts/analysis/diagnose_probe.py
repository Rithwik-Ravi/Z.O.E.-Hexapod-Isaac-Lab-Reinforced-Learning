# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import sys
import numpy as np
import torch

sys.path.append("scripts/rsl_rl")


def diagnose_probe(args_cli):
    from isaaclab.app import AppLauncher

    args_cli.headless = True
    app_launcher = AppLauncher(args_cli)
    sim_app = app_launcher.app

    import gymnasium as gym
    import cli_args
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg
    from isaaclab.utils.assets import retrieve_file_path
    import spdrbot3.tasks

    ckpt_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[DIAG] Loading {ckpt_path}")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=4)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task.split(":")[-1], args_cli)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(args_cli.device)

    robot = env.unwrapped.scene["robot"]
    contact_sensor = env.unwrapped.scene["contact_forces"]

    obs_res = env.get_observations()
    obs = obs_res[0] if isinstance(obs_res, tuple) else obs_res

    # 1. Zero Action Test (30 steps)
    zero_heights = []
    for _ in range(30):
        step_res = env.step(torch.zeros((4, 18), device=args_cli.device))
        obs = step_res[0]
        zero_heights.append(robot.data.root_pos_w[:, 2].cpu().numpy().copy())

    print(f"[DIAG] Zero Action Mean Height: {np.mean(zero_heights):.4f} m (Min: {np.min(zero_heights):.4f}, Max: {np.max(zero_heights):.4f})")
    print(f"[DIAG] Zero Action Joint Pos (env 0): {robot.data.joint_pos[0].cpu().numpy()}")

    # Reset
    env.reset()
    obs_res = env.get_observations()
    obs = obs_res[0] if isinstance(obs_res, tuple) else obs_res

    # 2. Policy Action Test (100 steps)
    policy_heights = []
    actions_list = []
    with torch.inference_mode():
        for step in range(100):
            pol_obs = obs["policy"] if isinstance(obs, dict) else obs
            actions = policy(pol_obs)
            actions_list.append(actions[0].cpu().numpy())
            step_res = env.step(actions)
            obs = step_res[0]
            policy_heights.append(robot.data.root_pos_w[:, 2].cpu().numpy().copy())

    print(f"[DIAG] Policy Mean Height: {np.mean(policy_heights):.4f} m (Min: {np.min(policy_heights):.4f}, Max: {np.max(policy_heights):.4f})")
    print(f"[DIAG] Policy Joint Pos (env 0): {robot.data.joint_pos[0].cpu().numpy()}")
    print(f"[DIAG] Policy Mean Actions (env 0): {np.mean(actions_list, axis=0)}")

    env.close()
    sim_app.close()


if __name__ == "__main__":
    import cli_args
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Zoe-Hexapod-Flat-Deploy-Play-v0")
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()

    diagnose_probe(args_cli)
