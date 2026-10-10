# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Gait Report: Evaluates single-environment contact timeline, duty factors, phase alternation, slip, and tracking."""

import argparse
import sys
import numpy as np
import torch

sys.path.append("scripts/rsl_rl")


def generate_gait_report(args_cli):
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
    print(f"\n[GAIT REPORT] Loading checkpoint: {ckpt_path}")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task.split(":")[-1], args_cli)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(args_cli.device)

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    contact_sensor = unwrapped.scene["contact_forces"]

    body_names = contact_sensor.body_names
    # Order legs 1..6
    foot_ids = [body_names.index(f"NFOOT{i}") for i in range(1, 7)]

    num_steps = 600  # 12.0 s at 50Hz
    dt = unwrapped.step_dt

    obs_res = env.reset()
    obs = obs_res[0] if isinstance(obs_res, tuple) else obs_res

    # Storage arrays
    contacts = np.zeros((num_steps, 6), dtype=bool)
    foot_heights = np.zeros((num_steps, 6))
    foot_slips = [[] for _ in range(6)]
    base_lin_vels = np.zeros((num_steps, 3))
    base_ang_vels = np.zeros((num_steps, 3))
    cmd_records = np.zeros((num_steps, 3))

    with torch.no_grad():
        for step in range(num_steps):
            # Command steps:
            # 0..100 (0..2s): 0 m/s
            # 100..300 (2..6s): 0.2 m/s forward
            # 300..400 (6..8s): 0 m/s
            # 400..500 (8..10s): yaw 0.5 rad/s
            # 500..600 (10..12s): strafe (0, 0.15)
            if step < 100:
                cmd_val = (0.0, 0.0, 0.0)
            elif step < 300:
                cmd_val = (0.2, 0.0, 0.0)
            elif step < 400:
                cmd_val = (0.0, 0.0, 0.0)
            elif step < 500:
                cmd_val = (0.0, 0.0, 0.5)
            else:
                cmd_val = (0.0, 0.15, 0.0)

            cmd = unwrapped.command_manager.get_command("base_velocity")
            cmd[:, 0] = cmd_val[0]
            cmd[:, 1] = cmd_val[1]
            cmd[:, 2] = cmd_val[2]
            cmd_records[step] = cmd_val

            pol_obs = obs["policy"] if isinstance(obs, dict) else obs
            actions = policy(pol_obs)

            # Record contacts and foot kinematics
            forces_z = contact_sensor.data.net_forces_w[0, foot_ids, 2].cpu().numpy()
            is_contact = forces_z > 1.0
            contacts[step] = is_contact

            foot_pos = robot.data.body_pos_w[0, foot_ids, :].cpu().numpy()
            foot_heights[step] = foot_pos[:, 2]

            foot_vel_xy = np.linalg.norm(robot.data.body_lin_vel_w[0, foot_ids, :2].cpu().numpy(), axis=-1)
            for leg_i in range(6):
                if is_contact[leg_i]:
                    foot_slips[leg_i].append(foot_vel_xy[leg_i])

            base_lin_vels[step] = robot.data.root_lin_vel_b[0].cpu().numpy()
            base_ang_vels[step] = robot.data.root_ang_vel_b[0].cpu().numpy()

            step_res = env.step(actions)
            obs = step_res[0]

    # Analysis during forward walking phase (steps 100..300, 2..6s)
    fwd_contacts = contacts[100:300]
    duty_factors = np.mean(fwd_contacts, axis=0) * 100.0

    # Swing clearance during forward walking
    swing_clearances = []
    for leg_i in range(6):
        is_swing = ~fwd_contacts[:, leg_i]
        if np.any(is_swing):
            swing_clearances.append(float(np.max(foot_heights[100:300, leg_i][is_swing])))
        else:
            swing_clearances.append(0.0)

    # Tripod group correlation (Legs 1,3,5 vs Legs 2,4,6)
    tripod_A = np.mean(fwd_contacts[:, [0, 2, 4]], axis=1) # Legs 1, 3, 5
    tripod_B = np.mean(fwd_contacts[:, [1, 3, 5]], axis=1) # Legs 2, 4, 6
    if np.std(tripod_A) > 1e-5 and np.std(tripod_B) > 1e-5:
        group_correlation = float(np.corrcoef(tripod_A, tripod_B)[0, 1])
    else:
        group_correlation = 0.0

    # Mean slip during stance
    mean_slips = [float(np.mean(foot_slips[i])) if foot_slips[i] else 0.0 for i in range(6)]

    # Tracking metrics in each phase
    # Phase 2 (forward 0.2 m/s):
    fwd_vx_mean = float(np.mean(base_lin_vels[100:300, 0]))
    # Phase 4 (yaw 0.5 rad/s):
    yaw_wz_mean = float(np.mean(base_ang_vels[400:500, 2]))
    # Phase 5 (strafe 0.15 m/s):
    strafe_vy_mean = float(np.mean(base_lin_vels[500:600, 1]))

    print("\n" + "=" * 80)
    print("                     GAIT EVALUATION REPORT (10s ROLLOUT)")
    print("=" * 80)
    print(f"Checkpoint: {ckpt_path}")
    print("-" * 80)

    print("\n1. DUTY FACTOR PER LEG (Forward Walking Phase 2.0s - 6.0s):")
    for i in range(6):
        print(f"   Leg {i+1} (NFOOT{i+1}): Duty Factor = {duty_factors[i]:.1f} %, Swing Clearance = {swing_clearances[i]*100:.2f} cm, Mean Stance Slip = {mean_slips[i]:.4f} m/s")

    print(f"\n2. ALTERNATION & PHASE ANALYSIS:")
    print(f"   Tripod Group A (Legs 1, 3, 5) Mean Duty Factor: {np.mean(duty_factors[[0, 2, 4]]):.1f} %")
    print(f"   Tripod Group B (Legs 2, 4, 6) Mean Duty Factor: {np.mean(duty_factors[[1, 3, 5]]):.1f} %")
    print(f"   Cross-Group Correlation (Tripod A vs B): {group_correlation:+.4f}")
    if group_correlation < -0.30:
        print("   -> Alternation Assessment: STRONG ALTERNATING GAIT CONFIRMED ✅ (True Tripod)")
    elif group_correlation < 0.0:
        print("   -> Alternation Assessment: PARTIAL ALTERNATION (Wave / Ripple Gait)")
    else:
        print("   -> Alternation Assessment: NO CLEAR ALTERNATION DETECTED ❌ (Legs move synchronously or randomly)")

    print(f"\n3. COMMAND TRACKING THROUGH STEP SEQUENCE:")
    print(f"   Forward Step (cmd: vx=0.20 m/s) -> Achieved Mean vx: {fwd_vx_mean:+.4f} m/s (Error: {abs(fwd_vx_mean - 0.20):.4f} m/s)")
    print(f"   Yaw Step     (cmd: wz=0.50 rad/s) -> Achieved Mean wz: {yaw_wz_mean:+.4f} rad/s (Error: {abs(yaw_wz_mean - 0.50):.4f} rad/s)")
    print(f"   Strafe Step  (cmd: vy=0.15 m/s) -> Achieved Mean vy: {strafe_vy_mean:+.4f} m/s (Error: {abs(strafe_vy_mean - 0.15):.4f} m/s)")

    print("\n4. CONTACT TIMELINE (10 Hz downsampled across 12 seconds, '#' = In Contact, '.' = Air):")
    timeline_steps = contacts[::5] # 50Hz downsampled to 10Hz
    print("Time (s): " + "".join(f"{int(t/10)%10}" for t in range(len(timeline_steps))))
    for leg_i in range(6):
        row_str = "".join("#" if timeline_steps[t, leg_i] else "." for t in range(len(timeline_steps)))
        print(f"Leg {leg_i+1} : {row_str}")

    print("=" * 80 + "\n")

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

    generate_gait_report(args_cli)
