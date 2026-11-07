# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Audit Stage 2 Terrain World: origins, local elevation, spawn heights, t=0 penetrations, and world-z reward math."""

import argparse
import os
import sys
import torch
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Audit Terrain World for Zoe Hexapod Stage 2.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
sim_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks
import spdrbot3.tasks
from spdrbot3.tasks.manager_based.zoe_deploy.params import ZOE_PARAMS

def audit_terrain_world():
    task_name = "Zoe-Hexapod-Terrain-v0"
    print(f"\n========================================================")
    print(f"AUDITING TERRAIN WORLD: {task_name} (num_envs={args_cli.num_envs})")
    print(f"========================================================\n")

    from isaaclab_tasks.utils import parse_env_cfg
    env_cfg = parse_env_cfg(task_name, num_envs=args_cli.num_envs)
    env = gym.make(task_name, cfg=env_cfg)
    env.reset()

    unwrapped = env.unwrapped
    terrain = unwrapped.scene.terrain
    robot = unwrapped.scene["robot"]
    contact_sensor = unwrapped.scene["contact_forces"]

    print(f"Terrain type: {terrain.cfg.terrain_type}")
    print(f"Num rows: {terrain.cfg.terrain_generator.num_rows}, Num cols: {terrain.cfg.terrain_generator.num_cols}")
    print(f"Env origins shape: {unwrapped.scene.env_origins.shape}")
    print(f"Robot root pos w: min={robot.data.root_pos_w[:, 2].min().item():.4f}, max={robot.data.root_pos_w[:, 2].max().item():.4f}, mean={robot.data.root_pos_w[:, 2].mean().item():.4f}")

    # Inspect terrain sub-terrains and levels
    if hasattr(terrain, "terrain_types"):
        print(f"Terrain types per env: {terrain.terrain_types}")
    if hasattr(terrain, "terrain_levels"):
        print(f"Terrain levels per env: min={terrain.terrain_levels.min().item()}, max={terrain.terrain_levels.max().item()}, mean={terrain.terrain_levels.float().mean().item():.2f}")

    # Step 1 physics step to evaluate initial forces and penetrations
    actions = torch.zeros((unwrapped.num_envs, 18), device=unwrapped.device)
    obs, rew, terminated, truncated, info = unwrapped.step(actions)

    net_forces = contact_sensor.data.net_forces_w
    max_net_force_per_env = torch.max(torch.norm(net_forces, dim=-1), dim=1)[0]
    
    print("\n--- Initial Step 0/1 Dynamics ---")
    print(f"Max contact force across bodies at t=0: min={max_net_force_per_env.min().item():.1f} N, max={max_net_force_per_env.max().item():.1f} N, mean={max_net_force_per_env.mean().item():.1f} N")

    # Check non-foot contact at t=0
    illegal_body_names = [name for name in contact_sensor.body_names if any(b in name for b in ["base_link", "NCCUP", "NTHIGH"])]
    illegal_body_ids = [contact_sensor.body_names.index(name) for name in illegal_body_names]
    illegal_forces = net_forces[:, illegal_body_ids, :]
    illegal_contacts = (torch.norm(illegal_forces, dim=-1) > 1.0).any(dim=1)
    print(f"Envs with non-foot contact at t=0: {illegal_contacts.sum().item()} / {unwrapped.num_envs} ({illegal_contacts.float().mean().item()*100:.1f}%)")

    # Check root_pos_w vs base_height_shaped reward
    root_z = robot.data.root_pos_w[:, 2]
    target_h = ZOE_PARAMS.TARGET_STANDING_HEIGHT
    min_h = ZOE_PARAMS.GATE_MIN_HEIGHT
    std = 0.015
    attractor = torch.exp(-torch.square((root_z - target_h) / std))
    deficit = torch.clamp(min_h - root_z, min=0.0)
    barrier = 10.0 * deficit + 150.0 * torch.square(deficit)
    reward_base_height = attractor - barrier

    print("\n--- World-Z Base Height Reward Evaluation ---")
    print(f"Target Height: {target_h:.4f} m, Min Height: {min_h:.4f} m")
    print(f"Root World Z: min={root_z.min().item():.4f} m, median={root_z.median().item():.4f} m, max={root_z.max().item():.4f} m")
    print(f"base_height_shaped Reward: min={reward_base_height.min().item():.4f}, median={reward_base_height.median().item():.4f}, max={reward_base_height.max().item():.4f}")
    
    # Check breakdown by terrain type if available
    origins_z = unwrapped.scene.env_origins[:, 2]
    print(f"Env origins Z: min={origins_z.min().item():.4f} m, max={origins_z.max().item():.4f} m")
    height_above_origin = root_z - origins_z
    print(f"Base height above Env Origin: min={height_above_origin.min().item():.4f} m, median={height_above_origin.median().item():.4f} m, max={height_above_origin.max().item():.4f} m")

    env.close()

if __name__ == "__main__":
    audit_terrain_world()
    sim_app.close()
