# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Gate Verification Suite (Chunk Check).

Evaluates 3 conditions on a policy checkpoint with 128 envs in parallel:
- Condition 1: Zero command, 5s duration.
- Condition 2: Forward command (0.15 m/s), 10s duration.
- Condition 3: Forward command (0.15 m/s) with pushes at 2.0s and 5.0s.

Logs summary table to docs/CHUNK_LOG.md.
"""

import argparse
import datetime
import math
import os
import subprocess
import sys
import numpy as np
import torch
from scipy.spatial import ConvexHull, QhullError

# AppLauncher must be invoked before importing omniverse / isaaclab modules
from isaaclab.app import AppLauncher
sys.path.append(os.path.abspath("scripts/rsl_rl"))
import cli_args

parser = argparse.ArgumentParser(description="Chunk Check: 3-Condition Gate Suite for Zoe Hexapod.")
parser.add_argument("--task", type=str, default="Zoe-Hexapod-Flat-Deploy-v0")
parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel environments (default: 128)")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = True
app_launcher = AppLauncher(args_cli)
sim_app = app_launcher.app

import gymnasium as gym
sys.path.append(os.path.abspath("scripts/rsl_rl"))
import cli_args
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg
from isaaclab.utils.assets import retrieve_file_path
import spdrbot3.tasks
from spdrbot3.tasks.manager_based.zoe_deploy.params import ZOE_PARAMS, get_foot_tip_offsets_tensor

GATE_MIN_HEIGHT = ZOE_PARAMS.GATE_MIN_HEIGHT
TORQUE_SAT_THRESHOLD = ZOE_PARAMS.TORQUE_SAT_THRESHOLD_NM


def compute_foot_tip_positions(robot, foot_ids, foot_names=None):
    """Compute foot tip 3D coordinates in world frame for specified foot bodies using per-leg 3D local tip vectors."""
    body_pos = robot.data.body_pos_w[:, foot_ids, :]
    body_quat = robot.data.body_quat_w[:, foot_ids, :]

    if foot_names is None:
        foot_names = [f"NFOOT{i}" for i in range(1, len(foot_ids) + 1)]
    r_local = get_foot_tip_offsets_tensor(foot_names, body_pos.device, body_pos.dtype)
    r_local = r_local.unsqueeze(0).expand(body_pos.shape[0], -1, 3)

    q_w = body_quat[..., 0:1]
    q_vec = body_quat[..., 1:4]
    a = torch.cross(q_vec, r_local, dim=-1)
    v_rot = r_local + 2.0 * q_w * a + 2.0 * torch.cross(q_vec, a, dim=-1)

    return body_pos + v_rot


def compute_stability_margin(com_2d: np.ndarray, foot_pts_2d: np.ndarray) -> float:
    """Calculate signed horizontal distance from COM to closest edge of support polygon formed by grounded feet.

    Positive if COM is strictly inside polygon, negative if outside.
    """
    if len(foot_pts_2d) < 3:
        return -0.1
    try:
        hull = ConvexHull(foot_pts_2d)
        eq = hull.equations  # normal [nx, ny] pointing outward, d
        dists = -(eq[:, :2] @ com_2d + eq[:, 2])
        return float(dists.min())
    except (QhullError, ValueError):
        return -0.1


def run_chunk_check():
    ckpt_path = retrieve_file_path(args_cli.checkpoint)
    print(f"\n[CHUNK CHECK] Running gate verification on checkpoint: {ckpt_path}")
    print(f"[CHUNK CHECK] Environments: {args_cli.num_envs}, Device: {args_cli.device}")

    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        git_hash = "unknown"

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task.split(":")[-1], args_cli)

    # Disable random push interval during conditions 1 & 2 (pushes controlled manually in condition 3)
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(args_cli.device)

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    contact_sensor = unwrapped.scene["contact_forces"]

    body_names = contact_sensor.body_names
    foot_names = [f"NFOOT{i}" for i in range(1, 7)]
    foot_ids = [body_names.index(n) for n in foot_names]
    robot_foot_ids = [robot.body_names.index(n) for n in foot_names]
    non_foot_ids = [
        idx for idx, name in enumerate(body_names)
        if any(part in name for part in ["base_link", "NCCUP", "NTHIGH"])
    ]

    results = []

    # =========================================================================
    # CONDITION 1: Zero Command (5.0s = 250 steps)
    # =========================================================================
    print("\n--> Evaluating Condition 1: Zero Command (5.0s)...")
    obs_res = env.reset()
    obs = obs_res[0] if isinstance(obs_res, tuple) else obs_res

    c1_steps = 250
    c1_height_pass = 0
    c1_feet_pass = 0
    c1_non_foot_contacts = 0
    c1_torque_sat_count = 0
    c1_total_joint_steps = c1_steps * args_cli.num_envs * 18

    with torch.no_grad():
        for step in range(c1_steps):
            cmd = unwrapped.command_manager.get_command("base_velocity")
            cmd[:] = 0.0

            pol_obs = obs["policy"] if isinstance(obs, dict) else obs
            actions = policy(pol_obs)

            # Record metrics
            base_height = robot.data.root_pos_w[:, 2]
            c1_height_pass += (base_height >= GATE_MIN_HEIGHT).sum().item()

            forces_z = contact_sensor.data.net_forces_w[:, foot_ids, 2]
            feet_down = (forces_z > 1.0).sum(dim=1)
            c1_feet_pass += (feet_down >= 5).sum().item()

            if non_foot_ids:
                non_foot_forces = torch.norm(contact_sensor.data.net_forces_w[:, non_foot_ids, :], dim=-1)
                c1_non_foot_contacts += (non_foot_forces > 1.0).any(dim=1).sum().item()

            torques = torch.abs(robot.data.applied_torque)
            c1_torque_sat_count += (torques >= TORQUE_SAT_THRESHOLD).sum().item()

            step_res = env.step(actions)
            obs = step_res[0]

    c1_total_frames = c1_steps * args_cli.num_envs
    c1_height_pct = 100.0 * c1_height_pass / c1_total_frames
    c1_feet_pct = 100.0 * c1_feet_pass / c1_total_frames
    c1_non_foot_pct = 100.0 * c1_non_foot_contacts / c1_total_frames
    c1_torque_sat_pct = 100.0 * c1_torque_sat_count / c1_total_joint_steps

    results.append({
        "cond": "1. Zero Command",
        "metric": f"Base height >= {GATE_MIN_HEIGHT:.4f}m",
        "measured": f"{c1_height_pct:.2f}% frames",
        "threshold": ">= 95%",
        "pass": c1_height_pct >= 95.0,
    })
    results.append({
        "cond": "1. Zero Command",
        "metric": ">= 5 feet in contact",
        "measured": f"{c1_feet_pct:.2f}% frames",
        "threshold": ">= 95%",
        "pass": c1_feet_pct >= 95.0,
    })
    results.append({
        "cond": "1. Zero Command",
        "metric": "Non-foot contact",
        "measured": f"{c1_non_foot_pct:.2f}% frames",
        "threshold": "< 1%",
        "pass": c1_non_foot_pct < 1.0,
    })
    results.append({
        "cond": "1. Zero Command",
        "metric": f"Torque sat (|tau| >= {TORQUE_SAT_THRESHOLD:.2f} N.m)",
        "measured": f"{c1_torque_sat_pct:.2f}% joint-steps",
        "threshold": "<= 2%",
        "pass": c1_torque_sat_pct <= 2.0,
    })

    # =========================================================================
    # CONDITION 2: Forward Command 0.15 m/s (10.0s = 500 steps)
    # =========================================================================
    print("\n--> Evaluating Condition 2: Forward Command 0.15 m/s (10.0s)...")
    obs_res = env.reset()
    obs = obs_res[0] if isinstance(obs_res, tuple) else obs_res

    c2_steps = 500
    c2_height_pass = 0
    c2_feet_pass = 0
    c2_non_foot_contacts = 0
    c2_torque_sat_count = 0
    c2_vel_x_sum = 0.0
    c2_total_joint_steps = c2_steps * args_cli.num_envs * 18

    stability_margins = []
    foot_names = [f"NFOOT{i}" for i in range(1, 7)]

    with torch.no_grad():
        for step in range(c2_steps):
            cmd = unwrapped.command_manager.get_command("base_velocity")
            cmd[:, 0] = 0.15
            cmd[:, 1] = 0.0
            cmd[:, 2] = 0.0

            pol_obs = obs["policy"] if isinstance(obs, dict) else obs
            actions = policy(pol_obs)

            base_height = robot.data.root_pos_w[:, 2]
            c2_height_pass += (base_height >= GATE_MIN_HEIGHT).sum().item()

            forces_z = contact_sensor.data.net_forces_w[:, foot_ids, 2]
            in_contact = forces_z > 1.0
            feet_down = in_contact.sum(dim=1)
            c2_feet_pass += (feet_down >= 3).sum().item()

            if non_foot_ids:
                non_foot_forces = torch.norm(contact_sensor.data.net_forces_w[:, non_foot_ids, :], dim=-1)
                c2_non_foot_contacts += (non_foot_forces > 1.0).any(dim=1).sum().item()

            torques = torch.abs(robot.data.applied_torque)
            c2_torque_sat_count += (torques >= TORQUE_SAT_THRESHOLD).sum().item()

            # Liveness: forward velocity in body frame
            c2_vel_x_sum += robot.data.root_lin_vel_b[:, 0].sum().item()

            # Stability margin calculation sampled every 5 steps (10 Hz)
            if step % 5 == 0:
                tip_pos_w = compute_foot_tip_positions(robot, robot_foot_ids, foot_names)
                com_2d_all = robot.data.root_pos_w[:, :2].cpu().numpy()
                in_contact_cpu = in_contact.cpu().numpy()
                tip_pos_cpu = tip_pos_w.cpu().numpy()

                for env_i in range(args_cli.num_envs):
                    pts_2d = tip_pos_cpu[env_i, in_contact_cpu[env_i], :2]
                    margin = compute_stability_margin(com_2d_all[env_i], pts_2d)
                    stability_margins.append(margin)

            step_res = env.step(actions)
            obs = step_res[0]

    c2_total_frames = c2_steps * args_cli.num_envs
    c2_height_pct = 100.0 * c2_height_pass / c2_total_frames
    c2_feet_pct = 100.0 * c2_feet_pass / c2_total_frames
    c2_non_foot_pct = 100.0 * c2_non_foot_contacts / c2_total_frames
    c2_torque_sat_pct = 100.0 * c2_torque_sat_count / c2_total_joint_steps
    c2_mean_speed = c2_vel_x_sum / c2_total_frames

    mean_margin = float(np.mean(stability_margins)) if stability_margins else -1.0
    min_margin = float(np.min(stability_margins)) if stability_margins else -1.0
    p5_margin = float(np.percentile(stability_margins, 5)) if stability_margins else -1.0

    results.append({
        "cond": "2. Forward Walk 0.15m/s",
        "metric": f"Base height >= {GATE_MIN_HEIGHT:.4f}m",
        "measured": f"{c2_height_pct:.2f}% frames",
        "threshold": ">= 90%",
        "pass": c2_height_pct >= 90.0,
    })
    results.append({
        "cond": "2. Forward Walk 0.15m/s",
        "metric": ">= 3 feet in contact",
        "measured": f"{c2_feet_pct:.2f}% frames",
        "threshold": ">= 99%",
        "pass": c2_feet_pct >= 99.0,
    })
    results.append({
        "cond": "2. Forward Walk 0.15m/s",
        "metric": "Non-foot contact",
        "measured": f"{c2_non_foot_pct:.2f}% frames",
        "threshold": "< 1%",
        "pass": c2_non_foot_pct < 1.0,
    })
    results.append({
        "cond": "2. Forward Walk 0.15m/s",
        "metric": "Static stability margin (p5)",
        "measured": f"p5 {p5_margin:.4f}m (mean {mean_margin:.4f}m, min {min_margin:.4f}m)",
        "threshold": "p5 >= 0.0m",
        "pass": p5_margin >= 0.0,
    })
    results.append({
        "cond": "2. Forward Walk 0.15m/s",
        "metric": "Forward speed (not-frozen)",
        "measured": f"{c2_mean_speed:.4f} m/s",
        "threshold": ">= 0.1200 m/s (80% of 0.15m/s)",
        "pass": c2_mean_speed >= 0.120,
    })
    results.append({
        "cond": "2. Forward Walk 0.15m/s",
        "metric": f"Torque sat (|tau| >= {TORQUE_SAT_THRESHOLD:.2f} N.m)",
        "measured": f"{c2_torque_sat_pct:.2f}% joint-steps",
        "threshold": "<= 2%",
        "pass": c2_torque_sat_pct <= 2.0,
    })

    # =========================================================================
    # CONDITION 3: Forward Command with Pushes at 2.0s and 5.0s (8.0s = 400 steps)
    # =========================================================================
    print("\n--> Evaluating Condition 3: Forward Walk with Pushes (8.0s)...")
    obs_res = env.reset()
    obs = obs_res[0] if isinstance(obs_res, tuple) else obs_res

    c3_steps = 400
    c3_non_foot_contacts = 0
    c3_torque_sat_count = 0
    c3_total_joint_steps = c3_steps * args_cli.num_envs * 18
    terminated_count = 0

    # Height monitoring at recovery checkpoints:
    # Push 1 at step 100 (t=2.0s) -> recover by step 150 (t=3.0s), hold steps 150..250
    # Push 2 at step 250 (t=5.0s) -> recover by step 300 (t=6.0s), hold steps 300..400
    post_push1_height_pass = 0
    post_push2_height_pass = 0

    with torch.no_grad():
        for step in range(c3_steps):
            cmd = unwrapped.command_manager.get_command("base_velocity")
            cmd[:, 0] = 0.15
            cmd[:, 1] = 0.0
            cmd[:, 2] = 0.0

            # Apply impulse pushes at 2.0s (step 100) and 5.0s (step 250)
            if step in [100, 250]:
                print(f"    [Push at step {step} (t={step * 0.02:.1f}s)] Applying velocity impulse 0.3-0.5 m/s")
                current_root_state = robot.data.root_state_w.clone()
                angle = torch.rand(args_cli.num_envs, device=args_cli.device) * (2 * math.pi)
                mag = torch.rand(args_cli.num_envs, device=args_cli.device) * 0.2 + 0.3  # 0.3 to 0.5 m/s
                current_root_state[:, 7] += mag * torch.cos(angle)
                current_root_state[:, 8] += mag * torch.sin(angle)
                robot.write_root_state_to_sim(current_root_state)

            pol_obs = obs["policy"] if isinstance(obs, dict) else obs
            actions = policy(pol_obs)

            base_height = robot.data.root_pos_w[:, 2]
            if 150 <= step < 250:
                post_push1_height_pass += (base_height >= GATE_MIN_HEIGHT).sum().item()
            if 300 <= step < 400:
                post_push2_height_pass += (base_height >= GATE_MIN_HEIGHT).sum().item()

            if non_foot_ids:
                non_foot_forces = torch.norm(contact_sensor.data.net_forces_w[:, non_foot_ids, :], dim=-1)
                c3_non_foot_contacts += (non_foot_forces > 1.0).any(dim=1).sum().item()

            torques = torch.abs(robot.data.applied_torque)
            c3_torque_sat_count += (torques >= TORQUE_SAT_THRESHOLD).sum().item()

            step_res = env.step(actions)
            obs, rewards, dones, truncated, extras = step_res if len(step_res) == 5 else (*step_res, None)
            terminated_count += dones.sum().item()

    c3_total_frames = c3_steps * args_cli.num_envs
    c3_non_foot_pct = 100.0 * c3_non_foot_contacts / c3_total_frames
    c3_torque_sat_pct = 100.0 * c3_torque_sat_count / c3_total_joint_steps

    push1_hold_pct = 100.0 * post_push1_height_pass / (100 * args_cli.num_envs)
    push2_hold_pct = 100.0 * post_push2_height_pass / (100 * args_cli.num_envs)

    results.append({
        "cond": "3. Push Recovery",
        "metric": "Terminations (falls)",
        "measured": f"{terminated_count} falls",
        "threshold": "0 falls",
        "pass": terminated_count == 0,
    })
    results.append({
        "cond": "3. Push Recovery",
        "metric": "Non-foot contact",
        "measured": f"{c3_non_foot_pct:.2f}% frames",
        "threshold": "< 1%",
        "pass": c3_non_foot_pct < 1.0,
    })
    results.append({
        "cond": "3. Push Recovery",
        "metric": "Push 1 Height Recovery (t=3..5s)",
        "measured": f"{push1_hold_pct:.2f}% frames >= {GATE_MIN_HEIGHT:.4f}m",
        "threshold": ">= 90%",
        "pass": push1_hold_pct >= 90.0,
    })
    results.append({
        "cond": "3. Push Recovery",
        "metric": "Push 2 Height Recovery (t=6..8s)",
        "measured": f"{push2_hold_pct:.2f}% frames >= {GATE_MIN_HEIGHT:.4f}m",
        "threshold": ">= 90%",
        "pass": push2_hold_pct >= 90.0,
    })
    results.append({
        "cond": "3. Push Recovery",
        "metric": f"Torque sat (|tau| >= {TORQUE_SAT_THRESHOLD:.2f} N.m)",
        "measured": f"{c3_torque_sat_pct:.2f}% joint-steps",
        "threshold": "<= 2%",
        "pass": c3_torque_sat_pct <= 2.0,
    })

    # =========================================================================
    # REPORT-ONLY METRICS: Yaw (0.4 rad/s) & Strafe (0.12 m/s) (150 steps each)
    # =========================================================================
    print("\n--> Evaluating Report-Only Yaw & Strafe Tracking (3.0s each)...")
    obs_res = env.reset()
    obs = obs_res[0] if isinstance(obs_res, tuple) else obs_res
    yaw_steps = 150
    yaw_wz_sum = 0.0
    with torch.no_grad():
        for _ in range(yaw_steps):
            cmd = unwrapped.command_manager.get_command("base_velocity")
            cmd[:, 0] = 0.0
            cmd[:, 1] = 0.0
            cmd[:, 2] = 0.40
            pol_obs = obs["policy"] if isinstance(obs, dict) else obs
            actions = policy(pol_obs)
            yaw_wz_sum += robot.data.root_ang_vel_b[:, 2].sum().item()
            obs = env.step(actions)[0]
    mean_yaw_wz = yaw_wz_sum / (yaw_steps * args_cli.num_envs)

    obs_res = env.reset()
    obs = obs_res[0] if isinstance(obs_res, tuple) else obs_res
    strafe_steps = 150
    strafe_vy_sum = 0.0
    with torch.no_grad():
        for _ in range(strafe_steps):
            cmd = unwrapped.command_manager.get_command("base_velocity")
            cmd[:, 0] = 0.0
            cmd[:, 1] = 0.12
            cmd[:, 2] = 0.0
            pol_obs = obs["policy"] if isinstance(obs, dict) else obs
            actions = policy(pol_obs)
            strafe_vy_sum += robot.data.root_lin_vel_b[:, 1].sum().item()
            obs = env.step(actions)[0]
    mean_strafe_vy = strafe_vy_sum / (strafe_steps * args_cli.num_envs)

    # =========================================================================
    # SUMMARY TABLE & OVERALL GATE EVALUATION
    # =========================================================================
    all_passed = all(r["pass"] for r in results)
    overall_status = "PASS" if all_passed else "FAIL"

    table_header = "| Condition | Metric | Measured | Threshold | Result |"
    table_divider = "|---|---|---|---|---|"
    table_rows = []
    for r in results:
        res_str = "**PASS**" if r["pass"] else "**FAIL**"
        table_rows.append(f"| {r['cond']} | {r['metric']} | {r['measured']} | {r['threshold']} | {res_str} |")

    full_table = "\n".join([table_header, table_divider] + table_rows)

    print("\n" + "=" * 90)
    print("                     CHUNK CHECK GATE EVALUATION SUMMARY")
    print("=" * 90)
    print(full_table)
    print("-" * 90)
    print(f"REPORT-ONLY TRACKING: Yaw (cmd 0.40 rad/s) = {mean_yaw_wz:+.4f} rad/s | Strafe (cmd 0.12 m/s) = {mean_strafe_vy:+.4f} m/s")
    print("=" * 90)
    print(f"OVERALL GATE VERDICT: {overall_status}")
    print("=" * 90 + "\n")

    # Append to docs/CHUNK_LOG.md
    log_path = "docs/CHUNK_LOG.md"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_entry = f"""
## Chunk Check Run - {timestamp}
- **Git Commit**: `{git_hash}`
- **Checkpoint**: `{ckpt_path}`
- **Overall Verdict**: **{overall_status}**
- **Report-Only Metrics**: Yaw (cmd `0.40 rad/s`) = `{mean_yaw_wz:+.4f} rad/s`, Strafe (cmd `0.12 m/s`) = `{mean_strafe_vy:+.4f} m/s`

{full_table}

---
"""
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("# Stage 1 Chunk Verification Log\n\n")

    with open(log_path, "a") as f:
        f.write(log_entry)

    print(f"[CHUNK CHECK] Log appended to {log_path}")
    sys.stdout.flush()
    os._exit(0 if all_passed else 1)


if __name__ == "__main__":
    sys.exit(run_chunk_check())
