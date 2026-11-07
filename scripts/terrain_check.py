# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Stage 2 Multi-Terrain Traversal & Gate Verification Suite (scripts/terrain_check.py).

Evaluates a policy checkpoint on Zoe-Hexapod-Terrain-v0 over 20.0s (1000-step) episodes
with forward command 0.20 m/s and random initial headings across all 7 terrain types
and difficulty levels [0, 2, 4, 6, 9] (>= 64 envs per terrain type; default 65 = 13 x 5).

Frozen Deployment Gate (B4):
- Target deployment levels: 0..4 (levels 0, 2, 4: slopes up to ~7 deg, steps up to ~3 cm,
  rough and boxes in lowest half of range).
- Pass criterion: survival >= 90.0% for EVERY terrain type across target levels 0..4.

Appends summary tables to docs/CHUNK_LOG.md.
"""

import argparse
import datetime
import math
import os
import subprocess
import sys
import time
import numpy as np
import torch

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath("scripts/rsl_rl"))
import cli_args

parser = argparse.ArgumentParser(description="Stage 2 Multi-Terrain Traversal & Gate Check.")
parser.add_argument("--task", type=str, default="Zoe-Hexapod-Terrain-v0")
parser.add_argument(
    "--envs_per_cell",
    type=int,
    default=13,
    help="Envs per (terrain_type, level) cell. Default 13 x 5 levels = 65 envs/terrain (>=64), 455 total.",
)
parser.add_argument(
    "--levels",
    type=str,
    default="0,2,4,6,9",
    help="Comma-separated terrain levels to evaluate (default: 0,2,4,6,9).",
)
parser.add_argument("--cmd_vx", type=float, default=0.20, help="Forward velocity command [m/s] (default: 0.20).")
parser.add_argument("--episode_s", type=float, default=20.0, help="Episode duration [s] (default: 20.0).")
parser.add_argument("--label", type=str, default="", help="Optional label for CHUNK_LOG.md entry.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

t0_gpu = time.time()
app_launcher = AppLauncher(args_cli)
sim_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg
from isaaclab.utils.assets import retrieve_file_path
import spdrbot3.tasks  # noqa: F401
from spdrbot3.tasks.manager_based.zoe_deploy.params import ZOE_PARAMS
from spdrbot3.tasks.manager_based.zoe_deploy.mdp.curriculum import get_column_terrain_names
from spdrbot3.tasks.manager_based.zoe_deploy.mdp.rewards import height_above_ground

TORQUE_SAT_THRESHOLD = ZOE_PARAMS.TORQUE_SAT_THRESHOLD_NM


def extract_upright_roll_pitch_deg(quat_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract roll and pitch angles (in degrees) relative to the upright 180-deg X-rolled base frame.

    Uses projected gravity in the robot's body frame (where upright has g_b = [0, 0, +1]):
    pitch = asin(clamp(g_bx, -1, 1)), roll = atan2(g_by, g_bz).
    """
    q_w = quat_wxyz[:, 0:1]
    q_vec = quat_wxyz[:, 1:4]
    g_w = torch.tensor([0.0, 0.0, -1.0], device=quat_wxyz.device, dtype=quat_wxyz.dtype).unsqueeze(0).expand(
        quat_wxyz.shape[0], 3
    )
    # Inverse rotation q^* * g_w * q
    q_vec_conj = -q_vec
    a = torch.cross(q_vec_conj, g_w, dim=-1)
    g_b = g_w + 2.0 * q_w * a + 2.0 * torch.cross(q_vec_conj, a, dim=-1)
    # In hexa_ir's inverted base_link frame, upright has g_b = (0, 0, +1)
    roll_rad = torch.atan2(g_b[:, 1], g_b[:, 2])
    pitch_rad = torch.asin(torch.clamp(g_b[:, 0], -1.0, 1.0))
    return torch.rad2deg(roll_rad), torch.rad2deg(pitch_rad)


def run_terrain_check():
    if not args_cli.checkpoint:
        print("[ERROR] --checkpoint is required for scripts/terrain_check.py", file=sys.stderr)
        os._exit(2)
    if not os.path.isfile(args_cli.checkpoint):
        print(f"[ERROR] Checkpoint file does not exist: {args_cli.checkpoint}", file=sys.stderr)
        os._exit(2)

    ckpt_path = os.path.abspath(retrieve_file_path(args_cli.checkpoint))
    levels = [int(x.strip()) for x in args_cli.levels.split(",") if x.strip()]
    terrain_order = ["flat", "rough", "slope_up", "slope_down", "stairs_up", "stairs_down", "boxes"]
    num_terrains = len(terrain_order)
    num_levels = len(levels)
    envs_per_cell = args_cli.envs_per_cell
    envs_per_terrain = envs_per_cell * num_levels
    total_envs = num_terrains * envs_per_terrain

    print(f"\n[TERRAIN CHECK] Checkpoint      : {ckpt_path}")
    print(f"[TERRAIN CHECK] Levels          : {levels} (Frozen Gate Target: levels <= 4)")
    print(f"[TERRAIN CHECK] Envs per terrain: {envs_per_terrain} ({envs_per_cell} per level x {num_levels} levels)")
    print(f"[TERRAIN CHECK] Total Envs      : {total_envs}, Episode: {args_cli.episode_s:.1f}s, Command: {args_cli.cmd_vx:.2f} m/s")

    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        git_hash = "unknown"

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=total_envs)
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    # Disable curriculum transitions during evaluation so env_origins stay fixed to the assigned tile
    if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "terrain_levels"):
        env_cfg.curriculum.terrain_levels = None

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task.split(":")[-1], args_cli)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=args_cli.device)

    unwrapped = env.unwrapped
    terrain = unwrapped.scene.terrain
    robot = unwrapped.scene["robot"]
    contact_sensor = unwrapped.scene["contact_forces"]
    scanner = unwrapped.scene["height_scanner"]

    col_names = get_column_terrain_names(terrain)
    rep_cols = {}
    for col_idx, name in enumerate(col_names):
        if name not in rep_cols:
            rep_cols[name] = col_idx

    # Build deterministic (terrain_idx, level) assignment for all total_envs
    cell_terrain_idx = []
    cell_level_val = []
    cell_col_idx = []
    for t_idx, t_name in enumerate(terrain_order):
        col = rep_cols[t_name]
        for lvl in levels:
            for _ in range(envs_per_cell):
                cell_terrain_idx.append(t_idx)
                cell_level_val.append(lvl)
                cell_col_idx.append(col)

    cell_terrain_idx_t = torch.tensor(cell_terrain_idx, device=unwrapped.device, dtype=torch.long)
    cell_level_val_t = torch.tensor(cell_level_val, device=unwrapped.device, dtype=torch.long)
    cell_col_idx_t = torch.tensor(cell_col_idx, device=unwrapped.device, dtype=torch.long)

    # Reset environment and assign tiles + random yaw headings at tile centers
    obs_res = env.reset()
    obs = obs_res[0] if isinstance(obs_res, tuple) else obs_res

    terrain.terrain_levels[:] = cell_level_val_t
    terrain.terrain_types[:] = cell_col_idx_t
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    assigned_origins_xy = terrain.env_origins[:, :2].clone()

    torch.manual_seed(42)
    yaw_rand = (torch.rand(total_envs, device=unwrapped.device) * 2.0 - 1.0) * math.pi
    half_yaw = yaw_rand * 0.5

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0] = terrain.env_origins[:, 0]
    root_states[:, 1] = terrain.env_origins[:, 1]
    root_states[:, 2] = terrain.env_origins[:, 2] + ZOE_PARAMS.STANDING_BASE_HEIGHT + 0.015
    root_states[:, 3] = 0.0
    root_states[:, 4] = torch.cos(half_yaw)
    root_states[:, 5] = torch.sin(half_yaw)
    root_states[:, 6] = 0.0
    root_states[:, 7:] = 0.0

    robot.write_root_state_to_sim(root_states)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.joint_vel))
    unwrapped.scene.write_data_to_sim()
    unwrapped.sim.forward()
    scanner.update(dt=unwrapped.physics_dt, force_recompute=True)
    h_init = height_above_ground(unwrapped, method="raycaster")
    ground_z_init = robot.data.root_pos_w[:, 2] - h_init
    root_states[:, 2] = ground_z_init + ZOE_PARAMS.STANDING_BASE_HEIGHT + 0.015
    robot.write_root_state_to_sim(root_states)
    unwrapped.scene.write_data_to_sim()
    unwrapped.sim.forward()
    obs = unwrapped.observation_manager.compute()

    body_names = contact_sensor.body_names
    foot_ids = [body_names.index(f"NFOOT{i}") for i in range(1, 7)]
    non_foot_ids = [
        idx for idx, name in enumerate(body_names) if any(part in name for part in ["base_link", "NCCUP", "NTHIGH"])
    ]

    num_steps = int(round(args_cli.episode_s / unwrapped.step_dt))  # 1000 steps for 20.0s
    alive = torch.ones(total_envs, device=unwrapped.device, dtype=torch.bool)
    active_steps = torch.zeros(total_envs, device=unwrapped.device, dtype=torch.long)
    final_dist = torch.zeros(total_envs, device=unwrapped.device, dtype=torch.float32)
    max_dist = torch.zeros(total_envs, device=unwrapped.device, dtype=torch.float32)
    off_platform_steps = torch.zeros(total_envs, device=unwrapped.device, dtype=torch.long)
    feet_down_sum = torch.zeros(total_envs, device=unwrapped.device, dtype=torch.float32)
    non_foot_steps = torch.zeros(total_envs, device=unwrapped.device, dtype=torch.long)
    torque_sat_joint_steps = torch.zeros(total_envs, device=unwrapped.device, dtype=torch.long)

    term_bad_ori = torch.zeros(total_envs, device=unwrapped.device, dtype=torch.bool)
    term_base_contact = torch.zeros(total_envs, device=unwrapped.device, dtype=torch.bool)
    term_nan = torch.zeros(total_envs, device=unwrapped.device, dtype=torch.bool)

    slope_mask = (cell_terrain_idx_t == terrain_order.index("slope_up")) | (
        cell_terrain_idx_t == terrain_order.index("slope_down")
    )
    slope_rolls = []
    slope_pitches = []

    with torch.no_grad():
        for step in range(num_steps):
            cmd = unwrapped.command_manager.get_command("base_velocity")
            cmd[:, 0] = args_cli.cmd_vx
            cmd[:, 1] = 0.0
            cmd[:, 2] = 0.0

            actions = policy(obs)

            # Record pre-step state for currently alive envs
            dist_xy = torch.norm(robot.data.root_pos_w[:, :2] - assigned_origins_xy, dim=-1)
            final_dist = torch.where(alive, dist_xy, final_dist)
            max_dist = torch.where(alive, torch.maximum(max_dist, dist_xy), max_dist)
            off_platform_steps += (alive & (dist_xy > 1.20)).long()
            active_steps += alive.long()

            forces_z = contact_sensor.data.net_forces_w[:, foot_ids, 2]
            f_down = (forces_z > 1.0).float().sum(dim=1)
            feet_down_sum += torch.where(alive, f_down, torch.zeros_like(f_down))

            if non_foot_ids:
                nf_forces = torch.norm(contact_sensor.data.net_forces_w[:, non_foot_ids, :], dim=-1)
                nf_hit = (nf_forces > 1.0).any(dim=1)
                non_foot_steps += (alive & nf_hit).long()

            torques = torch.abs(robot.data.applied_torque)
            t_sat = (torques >= TORQUE_SAT_THRESHOLD).sum(dim=1)
            torque_sat_joint_steps += torch.where(alive, t_sat, torch.zeros_like(t_sat))

            # Collect roll/pitch on slope tiles every 5 steps for alive envs
            if step % 5 == 0:
                alive_slopes = alive & slope_mask
                if alive_slopes.any():
                    r_deg, p_deg = extract_upright_roll_pitch_deg(robot.data.root_quat_w[alive_slopes])
                    slope_rolls.append(r_deg.cpu().numpy())
                    slope_pitches.append(p_deg.cpu().numpy())

            step_res = env.step(actions)
            obs = step_res[0]

            # Check termination terms on this step for alive envs
            tm = unwrapped.termination_manager
            newly_terminated = alive & tm.terminated & (~tm.time_outs)
            if newly_terminated.any():
                try:
                    bo = tm.get_term("bad_orientation")
                    term_bad_ori |= newly_terminated & bo
                except Exception:
                    pass
                try:
                    bc = tm.get_term("base_contact")
                    term_base_contact |= newly_terminated & bc
                except Exception:
                    pass
                try:
                    nan_t = tm.get_term("invalid_state_nan")
                    term_nan |= newly_terminated & nan_t
                except Exception:
                    pass
                alive = alive & (~newly_terminated)

    # Aggregate per (terrain_type, level)
    cell_rows = []
    for t_idx, t_name in enumerate(terrain_order):
        for lvl in levels:
            mask = (cell_terrain_idx_t == t_idx) & (cell_level_val_t == lvl)
            n_cell = mask.sum().item()
            surv_rate = 100.0 * alive[mask].float().mean().item()
            mean_final_d = final_dist[mask].mean().item()
            mean_max_d = max_dist[mask].mean().item()
            tot_steps_cell = active_steps[mask].sum().clamp(min=1).item()
            off_plat_pct = 100.0 * off_platform_steps[mask].sum().item() / tot_steps_cell
            mean_fd = feet_down_sum[mask].sum().item() / tot_steps_cell
            nf_pct = 100.0 * non_foot_steps[mask].sum().item() / tot_steps_cell
            tsat_pct = 100.0 * torque_sat_joint_steps[mask].sum().item() / (tot_steps_cell * 18)
            n_bo = term_bad_ori[mask].sum().item()
            n_bc = term_base_contact[mask].sum().item()
            n_nan = term_nan[mask].sum().item()
            cell_rows.append({
                "terrain": t_name,
                "level": lvl,
                "n": n_cell,
                "survival_pct": surv_rate,
                "final_dist_m": mean_final_d,
                "max_dist_m": mean_max_d,
                "off_plat_pct": off_plat_pct,
                "feet_down": mean_fd,
                "non_foot_pct": nf_pct,
                "torque_sat_pct": tsat_pct,
                "bad_ori": n_bo,
                "base_contact": n_bc,
                "nan": n_nan,
            })

    # Aggregate per terrain type (overall and target levels 0..4)
    summary_rows = []
    for t_idx, t_name in enumerate(terrain_order):
        mask_all = cell_terrain_idx_t == t_idx
        mask_target = mask_all & (cell_level_val_t <= 4)
        mask_l02 = mask_all & (cell_level_val_t <= 2)

        surv_all = 100.0 * alive[mask_all].float().mean().item()
        surv_target = 100.0 * alive[mask_target].float().mean().item() if mask_target.any() else surv_all
        surv_l02 = 100.0 * alive[mask_l02].float().mean().item() if mask_l02.any() else surv_all

        tot_steps_t = active_steps[mask_all].sum().clamp(min=1).item()
        dist_all = final_dist[mask_all].mean().item()
        dist_target = final_dist[mask_target].mean().item() if mask_target.any() else dist_all
        off_plat_t = 100.0 * off_platform_steps[mask_all].sum().item() / tot_steps_t
        fd_t = feet_down_sum[mask_all].sum().item() / tot_steps_t
        nf_t = 100.0 * non_foot_steps[mask_all].sum().item() / tot_steps_t
        tsat_t = 100.0 * torque_sat_joint_steps[mask_all].sum().item() / (tot_steps_t * 18)
        n_bo = term_bad_ori[mask_all].sum().item()
        n_bc = term_base_contact[mask_all].sum().item()

        summary_rows.append({
            "terrain": t_name,
            "surv_l02": surv_l02,
            "surv_target_0_4": surv_target,
            "surv_all": surv_all,
            "dist_target_m": dist_target,
            "dist_all_m": dist_all,
            "off_plat_pct": off_plat_t,
            "feet_down": fd_t,
            "non_foot_pct": nf_t,
            "torque_sat_pct": tsat_t,
            "bad_ori": n_bo,
            "base_contact": n_bc,
            "pass_target": surv_target >= 90.0,
            "pass_probe_l02": surv_l02 >= 50.0,
        })

    # Format Markdown tables
    sum_header = (
        "| Terrain Type | L0-2 Survival | Target L0-4 Survival (Gate >=90%) | All-Level Survival | "
        "Dist Walked L0-4 (m) | Dist Walked All (m) | Off-Platform (>1.2m) | Mean Feet Down | "
        "Non-Foot Contact | Torque Sat | Terminations (ori/base) | Gate L0-4 |"
    )
    sum_sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"
    sum_lines = [sum_header, sum_sep]
    for r in summary_rows:
        gate_str = "**PASS**" if r["pass_target"] else "**FAIL**"
        sum_lines.append(
            f"| `{r['terrain']}` | `{r['surv_l02']:.1f}%` | **`{r['surv_target_0_4']:.1f}%`** | `{r['surv_all']:.1f}%` | "
            f"`{r['dist_target_m']:.2f}` | `{r['dist_all_m']:.2f}` | `{r['off_plat_pct']:.1f}%` | `{r['feet_down']:.2f}` | "
            f"`{r['non_foot_pct']:.2f}%` | `{r['torque_sat_pct']:.2f}%` | `{r['bad_ori']} / {r['base_contact']}` | {gate_str} |"
        )
    summary_table_md = "\n".join(sum_lines)

    cell_header = (
        "| Terrain | Level | Survival | Final Dist (m) | Max Dist (m) | Off-Plat (>1.2m) | "
        "Feet Down | Non-Foot | Torque Sat | Term (ori/base) |"
    )
    cell_sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    cell_lines = [cell_header, cell_sep]
    for r in cell_rows:
        cell_lines.append(
            f"| `{r['terrain']}` | {r['level']} | `{r['survival_pct']:.1f}%` | `{r['final_dist_m']:.2f}` | "
            f"`{r['max_dist_m']:.2f}` | `{r['off_plat_pct']:.1f}%` | `{r['feet_down']:.2f}` | "
            f"`{r['non_foot_pct']:.2f}%` | `{r['torque_sat_pct']:.2f}%` | `{r['bad_ori']}/{r['base_contact']}` |"
        )
    cell_table_md = "\n".join(cell_lines)

    # Slope Roll/Pitch summary (A6)
    if slope_rolls:
        all_r = np.concatenate(slope_rolls)
        all_p = np.concatenate(slope_pitches)
        abs_r = np.abs(all_r)
        abs_p = np.abs(all_p)
        slope_ori_str = (
            f"Roll: mean={np.mean(all_r):+.2f} deg, std={np.std(all_r):.2f} deg, "
            f"|roll| p50={np.percentile(abs_r, 50):.2f} deg, p95={np.percentile(abs_r, 95):.2f} deg, max={np.max(abs_r):.2f} deg | "
            f"Pitch: mean={np.mean(all_p):+.2f} deg, std={np.std(all_p):.2f} deg, "
            f"|pitch| p50={np.percentile(abs_p, 50):.2f} deg, p95={np.percentile(abs_p, 95):.2f} deg, max={np.max(abs_p):.2f} deg"
        )
    else:
        slope_ori_str = "N/A"

    all_target_pass = all(r["pass_target"] for r in summary_rows)
    all_probe_l02_pass = all(r["pass_probe_l02"] for r in summary_rows)
    verdict_str = "PASS" if all_target_pass else "FAIL"
    elapsed = time.time() - t0_gpu

    print("\n" + "=" * 120)
    print("                           STAGE 2 MULTI-TERRAIN SUMMARY BY TERRAIN TYPE")
    print("=" * 120)
    print(summary_table_md)
    print("\n" + "=" * 120)
    print("                        PER-LEVEL TRAVERSAL BREAKDOWN (LEVELS 0, 2, 4, 6, 9)")
    print("=" * 120)
    print(cell_table_md)
    print("-" * 120)
    print(f"SLOPE ORIENTATION DISTRIBUTION (A6): {slope_ori_str}")
    print(f"B3 PROBE L0-2 SURVIVAL >= 50% CHECK: {'PASS' if all_probe_l02_pass else 'FAIL'}")
    print(f"B4 DEPLOYMENT TARGET (L0-4 >= 90%) : {verdict_str} (Elapsed GPU time: {elapsed:.1f}s)")
    print("=" * 120 + "\n")

    # Append to docs/CHUNK_LOG.md
    log_path = "docs/CHUNK_LOG.md"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    label_suffix = f" ({args_cli.label})" if args_cli.label else ""
    log_entry = f"""
## Stage 2 Terrain Check Run - {timestamp}{label_suffix}
- **Git Commit**: `{git_hash}`
- **Checkpoint**: `{ckpt_path}`
- **Command**: `{args_cli.cmd_vx:.2f} m/s` forward, random headings, `{args_cli.episode_s:.1f}s` (`{num_steps}` steps), `{envs_per_terrain}` envs/terrain (`{total_envs}` total)
- **B3 Probe Criterion (All Terrains L0-2 Survival >= 50%)**: **{"PASS" if all_probe_l02_pass else "FAIL"}**
- **B4 Deployment Gate Verdict (All Terrains L0-4 Survival >= 90%)**: **{verdict_str}**
- **Slope Roll/Pitch Distribution (A6)**: `{slope_ori_str}`

### Summary by Terrain Type
{summary_table_md}

<details>
<summary>Per-Level Traversal Breakdown (Levels {args_cli.levels})</summary>

{cell_table_md}

</details>

---
"""
    with open(log_path, "a") as f:
        f.write(log_entry)

    print(f"[TERRAIN CHECK] Appended results to {log_path}")
    sys.stdout.flush()
    env.close()
    os._exit(0)


if __name__ == "__main__":
    run_terrain_check()
