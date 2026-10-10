# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Stance Gate Evaluation Script: Validates posture, ground clearance, and foot contact metrics for Z.O.E. Hexapod policy."""

import argparse
import sys
import torch
import numpy as np


def evaluate_stance_gate(args_cli):
    """Run rollout evaluation and enforce posture stance gates."""
    import gymnasium as gym
    from isaaclab.app import AppLauncher

    args_cli.headless = True
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import cli_args
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg
    import spdrbot3.tasks

    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[STANCE GATE] Loading checkpoint from: {checkpoint_path}")

    # 1. Parse Env Config & Agent Config
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task.split(":")[-1], args_cli)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # 2. Load Policy with OnPolicyRunner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=args_cli.device)

    # 3. Step evaluation loop
    num_steps = args_cli.num_steps
    print(f"[STANCE GATE] Running stance evaluation over {num_steps} steps across {args_cli.num_envs} envs...")

    obs_res = env.get_observations()
    obs = obs_res[0] if isinstance(obs_res, tuple) else obs_res

    base_heights = []
    non_foot_contacts = []
    feet_down_counts = []

    unwrapped_env = env.unwrapped
    robot = unwrapped_env.scene["robot"]
    contact_sensor = unwrapped_env.scene["contact_forces"]

    # Body IDs
    body_names = contact_sensor.body_names
    non_foot_ids = [i for i, name in enumerate(body_names) if not ("NFOOT" in name or "foot" in name)]
    foot_ids = [i for i, name in enumerate(body_names) if ("NFOOT" in name or "foot" in name)]

    with torch.inference_mode():
        for step in range(num_steps):
            pol_obs = obs["policy"] if isinstance(obs, dict) else obs
            actions = policy(pol_obs)

            # Record metrics before step
            root_z = robot.data.root_pos_w[:, 2].cpu().numpy()
            base_heights.extend(root_z)

            # Contact forces [N_envs, N_bodies, 3]
            forces_z = contact_sensor.data.net_forces_w[:, :, 2]
            non_foot_force = (forces_z[:, non_foot_ids] > 1.0).any(dim=1).cpu().numpy()
            non_foot_contacts.extend(non_foot_force)

            feet_contacts = (forces_z[:, foot_ids] > 1.0).sum(dim=1).cpu().numpy()
            feet_down_counts.extend(feet_contacts)

            step_res = env.step(actions)
            obs = step_res[0]

    # 4. Metric Aggregation
    mean_height = float(np.mean(base_heights))
    non_foot_contact_rate = float(np.mean(non_foot_contacts)) * 100.0
    valid_stance_steps = float(np.mean(np.array(feet_down_counts) >= 3)) * 100.0
    mean_feet_down = float(np.mean(feet_down_counts))

    # Thresholds (Nominal settled standing height = 0.158m)
    NOMINAL_HEIGHT = 0.158
    MIN_HEIGHT = 0.125   # ~80% of nominal standing height (accounts for dynamic tripod gait)
    MAX_NON_FOOT_CONTACT_PCT = 1.0  # < 1% non-foot contact
    MIN_STANCE_STEPS_PCT = 95.0     # >= 3 feet down in >= 95% steps

    pass_height = mean_height >= MIN_HEIGHT
    pass_contacts = non_foot_contact_rate < MAX_NON_FOOT_CONTACT_PCT
    pass_stance = valid_stance_steps >= MIN_STANCE_STEPS_PCT

    gate_passed = pass_height and pass_contacts and pass_stance

    print("\n" + "=" * 65)
    print("                STANCE GATE EVALUATION REPORT")
    print("=" * 65)
    print(f"  Checkpoint           : {checkpoint_path}")
    print(f"  Total Steps Evaluated: {num_steps} steps x {args_cli.num_envs} envs = {num_steps * args_cli.num_envs}")
    print("-" * 65)
    print(f"  Mean Base Height     : {mean_height:.4f} m   (Target >= {MIN_HEIGHT:.3f} m) -> [{'PASS' if pass_height else 'FAIL'}]")
    print(f"  Non-Foot Contact Rate: {non_foot_contact_rate:.2f} %     (Target <  {MAX_NON_FOOT_CONTACT_PCT:.1f} %) -> [{'PASS' if pass_contacts else 'FAIL'}]")
    print(f"  >=3 Feet Contact Rate: {valid_stance_steps:.2f} %     (Target >= {MIN_STANCE_STEPS_PCT:.1f} %) -> [{'PASS' if pass_stance else 'FAIL'}]")
    print(f"  Mean Feet Down Count : {mean_feet_down:.2f} / 6 feet")
    print("=" * 65)
    if gate_passed:
        print("  RESULT: GATE PASSED - POSTURE & GROUND CLEARANCE VERIFIED ✅")
    else:
        print("  RESULT: GATE FAILED ❌")
    print("=" * 65 + "\n")

    env.close()
    simulation_app.close()

    if not gate_passed:
        sys.exit(1)


if __name__ == "__main__":
    import cli_args
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Stance Gate Evaluation for Z.O.E. Hexapod Policy.")
    parser.add_argument("--task", type=str, default="Zoe-Hexapod-Flat-Deploy-Play-v0", help="Task name")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments")
    parser.add_argument("--num_steps", type=int, default=500, help="Number of rollout evaluation steps")
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()

    evaluate_stance_gate(args_cli)
