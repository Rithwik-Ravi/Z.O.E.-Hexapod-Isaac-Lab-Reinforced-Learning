# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect stance geometry, link frames, foot tip positions, and joint torques."""

import argparse
import math
import numpy as np
import torch


def inspect_geometry():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(["--headless"])
    app_launcher = AppLauncher(args)
    sim_app = app_launcher.app

    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from pxr import Usd, UsdGeom
    import spdrbot3.tasks

    env_cfg = parse_env_cfg("Zoe-Hexapod-Flat-Deploy-Play-v0", device="cuda:0", num_envs=1)
    env = gym.make("Zoe-Hexapod-Flat-Deploy-Play-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    contact_sensor = unwrapped.scene["contact_forces"]

    # Step simulation with zero actions to settle on ground
    for _ in range(50):
        env.step(torch.zeros((1, 18), device="cuda:0"))

    root_pos_w = robot.data.root_pos_w[0].cpu().numpy()
    root_quat_w = robot.data.root_quat_w[0].cpu().numpy() # (w, x, y, z)
    body_names = robot.body_names
    body_pos_w = robot.data.body_pos_w[0].cpu().numpy()
    body_quat_w = robot.data.body_quat_w[0].cpu().numpy()

    # Standing base height
    base_height = float(root_pos_w[2])
    print(f"=== STANDING BASE HEIGHT: {base_height:.4f} m ===")

    # Inspect USD stage for NFOOT* geometry and bounding boxes / tip offsets
    stage = Usd.Stage.Open("hexapod/hexa/hexa.usd")
    bbox_cache = UsdGeom.BBoxCache(0, ["default", "render"])

    print("\n=== FOOT (NFOOT1..6) BODIES & TIPS IN BASE FRAME ===")
    foot_body_indices = [i for i, name in enumerate(body_names) if name.startswith("NFOOT")]
    
    # Base link index
    base_idx = body_names.index("base_link_v1")
    base_p = body_pos_w[base_idx]

    # For each leg 1..6:
    for leg_i in range(1, 7):
        name = f"NFOOT{leg_i}"
        idx = body_names.index(name)
        p_w = body_pos_w[idx]
        q_w = body_quat_w[idx]

        # In base frame (since base is roughly level and yaw aligned)
        rel_p = p_w - base_p
        r = math.sqrt(rel_p[0]**2 + rel_p[1]**2)
        angle_rad = math.atan2(rel_p[1], rel_p[0])
        angle_deg = math.degrees(angle_rad)

        # Get USD prim bounds for this foot
        prim_path = None
        for prim in stage.Traverse():
            if prim.GetName() == name:
                prim_path = prim.GetPath()
                break
        
        shank_len = 0.0
        z_min_local = 0.0
        if prim_path:
            prim = stage.GetPrimAtPath(prim_path)
            bound = bbox_cache.ComputeLocalBound(prim)
            r_box = bound.ComputeAlignedBox()
            min_local = r_box.GetMin()
            max_local = r_box.GetMax()
            z_min_local = min_local[2]
            shank_len = max_local[2] - min_local[2]

        print(f"Leg {leg_i} ({name}):")
        print(f"  Tibia CoM in Base Frame: x={rel_p[0]:.4f}m, y={rel_p[1]:.4f}m, z={rel_p[2]:.4f}m")
        print(f"  Polar: radius={r:.4f}m, angle={angle_deg:.2f}° (rad: {angle_rad:.4f}), z={rel_p[2]:.4f}m")
        print(f"  Tibia CoM World z: {p_w[2]:.4f}m")
        print(f"  Local Bounds: z_min={z_min_local:.4f}m, z_max={max_local[2]:.4f}m, length={shank_len:.4f}m")

    # Check Coxa positions
    print("\n=== COXA (NCCUP1..6) IN BASE FRAME ===")
    coxa_radii = []
    coxa_angles = []
    for leg_i in range(1, 7):
        name = f"NCCUP{leg_i}"
        idx = body_names.index(name)
        p_w = body_pos_w[idx]
        rel_p = p_w - base_p
        r = math.sqrt(rel_p[0]**2 + rel_p[1]**2)
        angle_deg = math.degrees(math.atan2(rel_p[1], rel_p[0]))
        coxa_radii.append(r)
        coxa_angles.append(angle_deg)
        print(f"Leg {leg_i} ({name}): radius={r:.4f}m, angle={angle_deg:.2f}°, z={rel_p[2]:.4f}m")

    # Check symmetry
    print("\n=== SYMMETRY ANALYSIS ===")
    print(f"Coxa radii: {coxa_radii}")
    print(f"Coxa angles: {coxa_angles}")
    print(f"Angle deltas: {[coxa_angles[i] - coxa_angles[i-1] for i in range(1, 6)]}")

    # Inspect standing torques
    torques = robot.data.applied_torque[0].cpu().numpy()
    joint_names = robot.joint_names
    print("\n=== STANDING JOINT TORQUES ===")
    for j_name, tau in zip(joint_names, torques):
        print(f"  {j_name:30s}: {tau:+.4f} N.m")

    env.close()
    sim_app.close()


if __name__ == "__main__":
    inspect_geometry()
