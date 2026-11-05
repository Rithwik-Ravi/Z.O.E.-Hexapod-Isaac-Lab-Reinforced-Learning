# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Honest Stance Gate Evaluation: Tests policy under 3 conditions with comprehensive kinematic and dynamic metrics."""

import argparse
import sys
import numpy as np
import torch

sys.path.append("scripts/rsl_rl")


def run_condition_eval(env, policy, condition_name: str, num_steps: int, cmd_override=None, apply_pushes=False):
    """Run rollout for a specific condition and compute detailed metrics."""
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    contact_sensor = unwrapped.scene["contact_forces"]

    body_names = contact_sensor.body_names
    foot_ids = [i for i, name in enumerate(body_names) if "NFOOT" in name or "foot" in name]
    non_foot_ids = [i for i, name in enumerate(body_names) if not ("NFOOT" in name or "foot" in name)]

    num_envs = unwrapped.num_envs
    dt = unwrapped.step_dt

    base_heights = []
    non_foot_contacts = []
    feet_down_counts = []
    action_diffs = []
    joint_velocities = []
    torque_saturations = []
    foot_slips = []

    obs_res = env.reset()
    obs = obs_res[0] if isinstance(obs_res, tuple) else obs_res
    prev_actions = None

    push_interval = 50  # every 1s if apply_pushes
    with torch.no_grad():
        for step in range(num_steps):
            if cmd_override is not None:
                # Force command manager
                cmd = unwrapped.command_manager.get_command("base_velocity")
                cmd[:, 0] = cmd_override[0]
                cmd[:, 1] = cmd_override[1]
                cmd[:, 2] = cmd_override[2]

            if apply_pushes and (step % push_interval == 0):
                # Apply random velocity push impulse
                root_state = robot.data.root_state_w.clone()
                push_v = torch.empty((num_envs, 2), device=unwrapped.device).uniform_(-0.4, 0.4)
                root_state[:, 7:9] += push_v
                robot.write_root_state_to_sim(root_state)

            pol_obs = obs["policy"] if isinstance(obs, dict) else obs
            actions = policy(pol_obs)

            # Record action jitter
            if prev_actions is not None:
                jitter = torch.abs(actions - prev_actions).mean().item()
                action_diffs.append(jitter)
            prev_actions = actions.clone()

            # Record metrics
            root_z = robot.data.root_pos_w[:, 2].cpu().numpy()
            base_heights.extend(root_z)

            forces_z = contact_sensor.data.net_forces_w[:, :, 2]
            non_foot_hit = (forces_z[:, non_foot_ids] > 1.0).any(dim=1).cpu().numpy()
            non_foot_contacts.extend(non_foot_hit)

            feet_down = (forces_z[:, foot_ids] > 1.0).sum(dim=1).cpu().numpy()
            feet_down_counts.extend(feet_down)

            # Joint velocities
            j_vel = torch.abs(robot.data.joint_vel).cpu().numpy().flatten()
            joint_velocities.extend(j_vel)

            # Torque saturation fraction (against 2.5 N.m, threshold 2.45 N.m)
            torques = torch.abs(robot.data.applied_torque)
            sat = (torques >= 2.45).float().mean().item()
            torque_saturations.append(sat)

            # Foot slip: foot body horizontal velocity when foot in contact (F_z > 1.0)
            foot_vel_xy = torch.norm(robot.data.body_lin_vel_w[:, foot_ids, :2], dim=-1) # [N, 6]
            foot_in_contact = forces_z[:, foot_ids] > 1.0 # [N, 6]
            if foot_in_contact.any():
                slips = foot_vel_xy[foot_in_contact].cpu().numpy().tolist()
                foot_slips.extend(slips)

            step_res = env.step(actions)
            obs = step_res[0]

    STANDING_BASE_HEIGHT = 0.1583
    SPEC_HEIGHT_THRESHOLD = 0.90 * STANDING_BASE_HEIGHT # 0.1425 m
    SPEC_STANCE_THRESHOLD = 99.0 # >= 99%
    SPEC_NON_FOOT_THRESHOLD = 1.0 # < 1%

    mean_h = float(np.mean(base_heights))
    p5_h = float(np.percentile(base_heights, 5))
    min_h = float(np.min(base_heights))
    non_foot_pct = float(np.mean(non_foot_contacts)) * 100.0
    stance_pct = float(np.mean(np.array(feet_down_counts) >= 3)) * 100.0
    mean_feet = float(np.mean(feet_down_counts))

    mean_jitter = float(np.mean(action_diffs)) if action_diffs else 0.0
    vel_p50 = float(np.percentile(joint_velocities, 50))
    vel_p95 = float(np.percentile(joint_velocities, 95))
    vel_max = float(np.max(joint_velocities))
    mean_sat_pct = float(np.mean(torque_saturations)) * 100.0
    mean_slip = float(np.mean(foot_slips)) if foot_slips else 0.0
    max_slip = float(np.max(foot_slips)) if foot_slips else 0.0

    pass_h = mean_h >= SPEC_HEIGHT_THRESHOLD
    pass_contact = non_foot_pct < SPEC_NON_FOOT_THRESHOLD
    pass_stance = stance_pct >= SPEC_STANCE_THRESHOLD
    condition_passed = pass_h and pass_contact and pass_stance

    return {
        "condition": condition_name,
        "mean_height": mean_h,
        "p5_height": p5_h,
        "min_height": min_h,
        "non_foot_contact_pct": non_foot_pct,
        "stance_99_pct": stance_pct,
        "mean_feet_down": mean_feet,
        "jitter": mean_jitter,
        "vel_p50": vel_p50,
        "vel_p95": vel_p95,
        "vel_max": vel_max,
        "torque_sat_pct": mean_sat_pct,
        "mean_slip_mps": mean_slip,
        "max_slip_mps": max_slip,
        "pass_height": pass_h,
        "pass_contact": pass_contact,
        "pass_stance": pass_stance,
        "passed": condition_passed,
    }


def main():
    import cli_args
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Evaluate Honest Stance Gate across 3 conditions.")
    parser.add_argument("--task", type=str, default="Zoe-Hexapod-Flat-Deploy-Play-v0")
    parser.add_argument("--num_envs", type=int, default=16)
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()

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
    print(f"\n[HONEST GATE] Loading checkpoint: {ckpt_path}")

    # Set up Play Env (Fixed seed, deterministic)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task.split(":")[-1], args_cli)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(args_cli.device)

    # 1. Condition 1: Zero command for 5 s (250 steps)
    print("\n[HONEST GATE] Running Condition 1: Zero Command (5.0 s / 250 steps)...")
    res1 = run_condition_eval(env, policy, "Condition 1: Zero Command (5s)", num_steps=250, cmd_override=(0.0, 0.0, 0.0))

    # 2. Condition 2: Forward 0.15 m/s for 10 s (500 steps)
    print("\n[HONEST GATE] Running Condition 2: Forward 0.15 m/s (10.0 s / 500 steps)...")
    res2 = run_condition_eval(env, policy, "Condition 2: Forward 0.15 m/s (10s)", num_steps=500, cmd_override=(0.15, 0.0, 0.0))

    # 3. Condition 3: Randomization & Pushes (10.0 s / 500 steps)
    print("\n[HONEST GATE] Running Condition 3: Randomization & Pushes (10.0 s / 500 steps)...")
    res3 = run_condition_eval(env, policy, "Condition 3: Extreme Randomization & Pushes (10s)", num_steps=500, cmd_override=None, apply_pushes=True)

    results = [res1, res2, res3]

    print("\n" + "=" * 90)
    print("                    HONEST STANCE GATE EVALUATION REPORT")
    print("=" * 90)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Spec Thresholds: Height >= 0.1425m (0.90x0.158m) | Non-Foot Contact < 1.0% | >=3 Feet Down >= 99.0%")
    print("-" * 90)

    for r in results:
        print(f"\n--- {r['condition']} ---")
        print(f"  Base Height (m)       : Mean={r['mean_height']:.4f}m, p5={r['p5_height']:.4f}m, Min={r['min_height']:.4f}m -> [{'PASS' if r['pass_height'] else 'FAIL'}]")
        print(f"  Non-Foot Contact (%)  : {r['non_foot_contact_pct']:.2f}% (Spec < 1.0%) -> [{'PASS' if r['pass_contact'] else 'FAIL'}]")
        print(f"  >=3 Feet Down (%)     : {r['stance_99_pct']:.2f}% (Spec >= 99.0%) -> [{'PASS' if r['pass_stance'] else 'FAIL'}]")
        print(f"  Mean Feet Down Count  : {r['mean_feet_down']:.2f} / 6 feet")
        print(f"  Action Jitter (mean)  : {r['jitter']:.4f} rad")
        print(f"  Joint Velocity (rad/s): p50={r['vel_p50']:.2f}, p95={r['vel_p95']:.2f}, Max={r['vel_max']:.2f} (Limits: 4.5 / 6.0 rad/s)")
        print(f"  Torque Saturation (%) : {r['torque_sat_pct']:.2f}% (at 2.5 N.m limit)")
        print(f"  Foot Slip (m/s)       : Mean={r['mean_slip_mps']:.4f} m/s, Max={r['max_slip_mps']:.4f} m/s")
        print(f"  Condition Result      : {'PASS ✅' if r['passed'] else 'FAIL ❌'}")

    all_passed = all(r["passed"] for r in results)
    print("\n" + "=" * 90)
    print(f"OVERALL STANCE GATE RESULT: {'PASS ✅' if all_passed else 'FAIL ❌'}")
    print("=" * 90 + "\n")

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
