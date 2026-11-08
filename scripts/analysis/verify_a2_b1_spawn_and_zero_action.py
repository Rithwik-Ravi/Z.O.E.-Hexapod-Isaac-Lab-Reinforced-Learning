# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Verification script for A1 cross-check, A2 spawn table, and B1 (diagnosis sections E & F re-run)."""

import argparse
import os
import sys
import time
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify A1, A2, and B1 on Zoe-Hexapod-Terrain-v0.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

t0_gpu = time.time()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import spdrbot3.tasks  # noqa: F401
from spdrbot3.tasks.manager_based.zoe_deploy.params import ZOE_PARAMS
from spdrbot3.tasks.manager_based.zoe_deploy.mdp.rewards import (
    base_height_shaped,
    height_above_ground,
)
from spdrbot3.tasks.manager_based.zoe_deploy.mdp.curriculum import get_column_terrain_names


def main():
    num_envs = 70  # 7 terrain types x 10 levels (0..9)
    env_cfg = parse_env_cfg("Zoe-Hexapod-Terrain-v0", device="cuda:0", num_envs=num_envs)
    # Disable random pushes during spawn/zero-action audit
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None

    env = gym.make("Zoe-Hexapod-Terrain-v0", cfg=env_cfg)
    unwrapped = env.unwrapped
    terrain = unwrapped.scene.terrain
    robot = unwrapped.scene["robot"]
    scanner = unwrapped.scene["height_scanner"]

    col_names = get_column_terrain_names(terrain)
    print(f"[INFO] Column terrain names (0..9): {list(enumerate(col_names))}")

    # Pick representative column index for each of the 7 sub-terrain types
    rep_cols = {}
    for col_idx, name in enumerate(col_names):
        if name not in rep_cols:
            rep_cols[name] = col_idx
    terrain_order = ["flat", "rough", "slope_up", "slope_down", "stairs_up", "stairs_down", "boxes"]
    assert all(t in rep_cols for t in terrain_order), f"Missing terrain type in {rep_cols}"

    # Assign the 70 envs to (terrain_idx in 0..6, level in 0..9)
    env.reset()
    levels_list = []
    types_list = []
    for t_idx, t_name in enumerate(terrain_order):
        col = rep_cols[t_name]
        for lvl in range(10):
            levels_list.append(lvl)
            types_list.append(col)

    terrain.terrain_levels[:] = torch.tensor(levels_list, device=unwrapped.device, dtype=torch.long)
    terrain.terrain_types[:] = torch.tensor(types_list, device=unwrapped.device, dtype=torch.long)
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]

    # =========================================================================
    # A2: Spawn with zero action at deterministic standing height across all 7x10 tiles
    # =========================================================================
    # Place robots at tile center (env_origins xy) with default joint pose and zero velocity,
    # then step 1 frame with zero action so RayCaster updates on the exact spawn pose,
    # and set root_z = ground_z + STANDING_BASE_HEIGHT (deterministic spawn height without random z_rand jitter)
    root_states = robot.data.default_root_state.clone()
    root_states[:, 0] = terrain.env_origins[:, 0]
    root_states[:, 1] = terrain.env_origins[:, 1]
    root_states[:, 2] = terrain.env_origins[:, 2] + ZOE_PARAMS.STANDING_BASE_HEIGHT
    root_states[:, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=unwrapped.device)
    root_states[:, 7:] = 0.0
    robot.write_root_state_to_sim(root_states)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.joint_vel))
    unwrapped.scene.write_data_to_sim()
    unwrapped.sim.forward()
    scanner.update(dt=unwrapped.physics_dt, force_recompute=True)

    # Snap spawn Z to exact ground_z + STANDING_BASE_HEIGHT (as reset_root_near_ground does when z_rand is fixed)
    h_ray_initial = height_above_ground(unwrapped, method="raycaster")
    ground_z_spawn = robot.data.root_pos_w[:, 2] - h_ray_initial
    root_states[:, 2] = ground_z_spawn + ZOE_PARAMS.STANDING_BASE_HEIGHT
    robot.write_root_state_to_sim(root_states)
    unwrapped.scene.write_data_to_sim()
    unwrapped.sim.forward()
    scanner.update(dt=unwrapped.physics_dt, force_recompute=True)

    bhs_spawn = base_height_shaped(unwrapped)
    h_spawn = height_above_ground(unwrapped, method="raycaster")

    print("\n" + "=" * 95)
    print("A2 TABLE 1: base_height_shaped AT SPAWN (t=0, zero action, deterministic z_offset=0.1427m)")
    print("=" * 95)
    header = "| Terrain Type | " + " | ".join([f"L{l}" for l in range(10)]) + " | Mean | Min | Max |"
    sep = "|---|" + "|".join(["---:" for _ in range(13)]) + "|"
    print(header)
    print(sep)
    for t_idx, t_name in enumerate(terrain_order):
        vals = [bhs_spawn[t_idx * 10 + l].item() for l in range(10)]
        row_str = " | ".join([f"{v:.4f}" for v in vals])
        print(f"| `{t_name}` | {row_str} | {sum(vals)/10:.4f} | {min(vals):.4f} | {max(vals):.4f} |")

    spawn_spread = (bhs_spawn.max() - bhs_spawn.min()).item()
    print(f"\n[A2 SPAWN CHECK] Global min={bhs_spawn.min().item():.4f}, max={bhs_spawn.max().item():.4f}, spread={spawn_spread:.6f} (Requirement: within +-0.02)")
    assert spawn_spread <= 0.02, f"A2 spawn spread {spawn_spread} > 0.02!"

    # Now let robots settle for 30 steps (0.60s) with ZERO action across all 70 tiles
    zero_actions = torch.zeros((num_envs, 18), device=unwrapped.device)
    with torch.no_grad():
        for _ in range(30):
            cmd = unwrapped.command_manager.get_command("base_velocity")
            cmd[:] = 0.0
            env.step(zero_actions)

    bhs_settled = base_height_shaped(unwrapped)
    h_ray_settled = height_above_ground(unwrapped, method="raycaster")
    h_feet_settled = height_above_ground(unwrapped, method="feet")
    diff_ray_vs_feet = torch.abs(h_ray_settled - h_feet_settled)

    print("\n" + "=" * 95)
    print("A2 TABLE 2: base_height_shaped AFTER SETTLING WITH ZERO ACTION (t=0.6s, 30 steps)")
    print("=" * 95)
    print(header)
    print(sep)
    for t_idx, t_name in enumerate(terrain_order):
        vals = [bhs_settled[t_idx * 10 + l].item() for l in range(10)]
        row_str = " | ".join([f"{v:.4f}" for v in vals])
        print(f"| `{t_name}` | {row_str} | {sum(vals)/10:.4f} | {min(vals):.4f} | {max(vals):.4f} |")

    # Cross-check RayCaster vs Foot-Contact ground height on flat, slope, stairs, boxes
    print("\n" + "=" * 95)
    print("A1 CROSS-CHECK: height_above_ground (RayCaster 3x3 vs Foot-Tip Contact Fallback)")
    print("=" * 95)
    print("| Terrain Type | RayCaster Mean Height (m) | Foot-Tip Mean Height (m) | Mean |Diff| (m) | Max |Diff| (m) | Threshold | Status |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for t_idx, t_name in enumerate(terrain_order):
        sl = slice(t_idx * 10, (t_idx + 1) * 10)
        r_m = h_ray_settled[sl].mean().item()
        f_m = h_feet_settled[sl].mean().item()
        d_mean = diff_ray_vs_feet[sl].mean().item()
        d_max = diff_ray_vs_feet[sl].max().item()
        thresh = 0.010 if t_name == "flat" else 0.020
        ok = d_max <= thresh
        print(f"| `{t_name}` | {r_m:.4f} | {f_m:.4f} | {d_mean:.4f} | {d_max:.4f} | <= {thresh:.3f}m | {'PASS' if ok else 'FAIL'} |")
        if t_name in ("flat", "slope_up", "slope_down", "stairs_up", "stairs_down", "boxes"):
            assert ok, f"Cross-check failed on {t_name}: max diff {d_max:.4f}m > {thresh:.3f}m"

    # =========================================================================
    # B1: Re-run Diagnosis Section E (world Z span vs height_above_ground) & Section F (250-step zero-action survival)
    # =========================================================================
    origin_z = terrain.env_origins[:, 2]
    world_z = robot.data.root_pos_w[:, 2]
    print("\n" + "=" * 95)
    print("B1 (SECTION E RE-RUN): WORLD-Z vs LOCAL HEIGHT_ABOVE_GROUND ACROSS ALL 70 TILES")
    print("=" * 95)
    print(f"  Tile origin_z span       : min = {origin_z.min().item():+.4f} m, max = {origin_z.max().item():+.4f} m")
    print(f"  Robot world root_z span  : min = {world_z.min().item():+.4f} m, max = {world_z.max().item():+.4f} m")
    print(f"  height_above_ground span : min = {h_ray_settled.min().item():+.4f} m, max = {h_ray_settled.max().item():+.4f} m (mean = {h_ray_settled.mean().item():.4f} m)")
    print(f"  base_height_shaped span  : min = {bhs_settled.min().item():+.4f}, max = {bhs_settled.max().item():+.4f} (mean = {bhs_settled.mean().item():.4f})")
    print("  (Before fix: min base_height_shaped was -68.30/step on negative-Z tiles and 0.0000 on positive-Z tiles!)")

    # Section F: Full 250-step (5.0s) zero-action rollout
    fall_counts = 0
    bhs_sum = 0.0
    with torch.no_grad():
        for _ in range(220):  # remaining 220 steps to complete 250 steps (5.0s)
            cmd = unwrapped.command_manager.get_command("base_velocity")
            cmd[:] = 0.0
            env.step(zero_actions)
            bhs_sum += base_height_shaped(unwrapped).mean().item()
            early_term = unwrapped.termination_manager.terminated & (~unwrapped.termination_manager.time_outs)
            fall_counts += early_term.sum().item()

    survival_pct = 100.0 * (num_envs - min(num_envs, fall_counts)) / num_envs
    print("\n" + "=" * 95)
    print("B1 (SECTION F RE-RUN): 250-STEP (5.0s) ZERO-ACTION ROLLOUT ACROSS ALL 70 TILES")
    print("=" * 95)
    print(f"  Environments evaluated   : {num_envs} (7 terrain types x 10 levels)")
    print(f"  Early terminations/falls : {fall_counts}")
    print(f"  Zero-action survival rate: {survival_pct:.1f}%")
    print(f"  Mean base_height_shaped  : {bhs_sum / 220.0:+.4f} / step")
    assert fall_counts == 0, f"Expected 0 falls during zero-action stance, got {fall_counts}"

    elapsed = time.time() - t0_gpu
    print(f"\n[SUCCESS] A1, A2, and B1 verification passed in {elapsed:.1f}s GPU time.")
    env.close()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)

