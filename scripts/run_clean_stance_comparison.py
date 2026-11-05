# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Clean 1b Stance Comparison Script."""

import math
import numpy as np
import torch
from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import spdrbot3.tasks  # noqa: F401
from spdrbot3.tasks.manager_based.zoe_deploy.env_cfg_flat import ZoeHexapodFlatDeployPlayEnvCfg, EventCfg

print("\n" + "=" * 115)
print(f"{'Case':<32} | {'t=0 h':<8} | {'Settled h':<10} | {'Feet Down':<9} | {'Torque % (C/T/F)':<22} | {'Colliding Bodies'}")
print("=" * 115)

test_cases = [
    ("flat_deploy", 0.18, 2.5),
    ("flat_deploy", 0.18, 5.0),
    ("flat_deploy", 0.28, 2.5),
    ("flat_deploy", 0.28, 5.0),
    ("usd_zero", 0.18, 2.5),
    ("usd_zero", 0.18, 5.0),
    ("usd_zero", 0.28, 2.5),
    ("usd_zero", 0.28, 5.0),
]

for pose_name, spawn_z, effort_lim in test_cases:
    env_cfg = ZoeHexapodFlatDeployPlayEnvCfg()
    env_cfg.scene.num_envs = 16
    env_cfg.events = EventCfg()

    if pose_name == "flat_deploy":
        DEFAULT_COXA_POS = math.radians(45.0)
        DEFAULT_THIGH_POS = math.radians(7.7)
        DEFAULT_FOOT_POS = math.radians(7.4)
        env_cfg.scene.robot.init_state.joint_pos = {
            ".*base_link_v1_Revolute.*": DEFAULT_COXA_POS,
            ".*NCCUP.*_Revolute.*": DEFAULT_THIGH_POS,
            ".*NTHIGH.*_Revolute.*": DEFAULT_FOOT_POS,
        }
    else:
        env_cfg.scene.robot.init_state.joint_pos = {".*Revolute.*": 0.0}

    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, spawn_z)
    env_cfg.scene.robot.actuators["coxa_joints"].effort_limit_sim = effort_lim
    env_cfg.scene.robot.actuators["thigh_joints"].effort_limit_sim = effort_lim
    env_cfg.scene.robot.actuators["foot_joints"].effort_limit_sim = effort_lim

    env = gym.make("Zoe-Hexapod-Flat-Deploy-v0", cfg=env_cfg)
    env.reset()

    robot = env.unwrapped.scene["robot"]
    contact_sensor = env.unwrapped.scene["contact_forces"]

    h_t0 = robot.data.root_pos_w[:, 2].mean().item()

    h_settled_list = []
    for step in range(200):
        obs, rew, term, trunc, info = env.step(torch.zeros((16, 18), device=env.unwrapped.device))
        h = robot.data.root_pos_w[:, 2].mean().item()
        if step >= 150:
            h_settled_list.append(h)

    h_settled = np.mean(h_settled_list)

    net_forces = contact_sensor.data.net_forces_w[:, :, 2]
    contact_body_names = contact_sensor.body_names

    colliding_bodies = []
    feet_down_count = 0

    for b_idx, b_name in enumerate(contact_body_names):
        mean_f = net_forces[:, b_idx].mean().item()
        if mean_f > 1.0:
            colliding_bodies.append(b_name)
            if "NFOOT" in b_name or "FOOT" in b_name:
                feet_down_count += 1

    torques = robot.data.applied_torque.abs()
    joint_names = robot.data.joint_names
    c_t, t_t, f_t = [], [], []

    for j_idx, j_name in enumerate(joint_names):
        t_pct = torques[:, j_idx].mean().item() / effort_lim * 100.0
        if "base_link_v1" in j_name:
            c_t.append(t_pct)
        elif "NCCUP" in j_name:
            t_t.append(t_pct)
        elif "NTHIGH" in j_name:
            f_t.append(t_pct)

    c_mean = np.mean(c_t) if c_t else 0.0
    t_mean = np.mean(t_t) if t_t else 0.0
    f_mean = np.mean(f_t) if f_t else 0.0

    case_str = f"{pose_name} | z={spawn_z:.2f} | {effort_lim:.1f}Nm"
    coll_str = ", ".join(colliding_bodies[:3]) if colliding_bodies else "None"
    print(f"{case_str:<32} | {h_t0:.4f}m | {h_settled:.4f}m   | {feet_down_count:^9} | C:{c_mean:.1f}% T:{t_mean:.1f}% F:{f_mean:.1f}% | {coll_str}")

    env.close()

simulation_app.close()
