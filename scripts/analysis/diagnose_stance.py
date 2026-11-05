# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Streamlined stance diagnosis script for Z.O.E. Hexapod."""

import argparse
import math
import os
import sys
import numpy as np
import torch

# Ensure scripts/rsl_rl is in sys.path
sys.path.append(os.path.abspath("scripts/rsl_rl"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diagnose Z.O.E. Hexapod stance.")
parser.add_argument("--test", type=str, default="probe", choices=["grid", "probe", "tb", "conventions", "all"])
parser.add_argument("--checkpoint", type=str, default="logs/rsl_rl/zoe_hexapod_deploy/2026-10-09_00-29-35/model_9999.pt")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import spdrbot3.tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner


def run_1b_grid():
    print("=" * 80)
    print("PHASE 1b: ZERO-ACTION STANCE TEST GRID (16 envs, 300 steps)")
    print("=" * 80)

    from spdrbot3.tasks.manager_based.zoe_deploy.env_cfg_flat import ZoeHexapodFlatDeployPlayEnvCfg, EventCfg

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

    print(f"\n{'Case':<32} | {'t=0 h':<7} | {'Settled h':<9} | {'Feet Down':<9} | {'Torque % (C/T/F)':<20} | {'Colliding Bodies'}")
    print("-" * 115)

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
        for step in range(300):
            obs, rew, term, trunc, info = env.step(torch.zeros((16, 18), device=env.unwrapped.device))
            h = robot.data.root_pos_w[:, 2].mean().item()
            if step >= 250:
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

        # Torques
        torques = robot.data.applied_torque.abs() # (16, 18)
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

        case_str = f"{pose_name} | z={spawn_z} | {effort_lim}Nm"
        coll_str = ", ".join(colliding_bodies[:3]) if colliding_bodies else "None"
        print(f"{case_str:<32} | {h_t0:.4f}m | {h_settled:.4f}m   | {feet_down_count:^9} | C:{c_mean:.1f}% T:{t_mean:.1f}% F:{f_mean:.1f}% | {coll_str}")
        
        env.close()


def run_1d_probe(checkpoint_path):
    print("\n" + "=" * 80)
    print(f"PHASE 1d: PROBE TRAINED CHECKPOINT ({checkpoint_path})")
    print("=" * 80)

    from spdrbot3.tasks.manager_based.zoe_deploy.env_cfg_flat import ZoeHexapodFlatDeployPlayEnvCfg
    import cli_args

    env_cfg = ZoeHexapodFlatDeployPlayEnvCfg()
    env_cfg.scene.num_envs = 64

    env = gym.make("Zoe-Hexapod-Flat-Deploy-v0", cfg=env_cfg)
    env_wrapped = RslRlVecEnvWrapper(env)

    agent_cfg = cli_args.parse_rsl_rl_cfg("Zoe-Hexapod-Flat-Deploy-v0", argparse.Namespace(
        task="Zoe-Hexapod-Flat-Deploy-v0",
        resume=True,
        load_run=".*",
        load_checkpoint=None,
        checkpoint=checkpoint_path,
        device="cuda:0",
        run_name=None,
        experiment_name=None,
        logger=None,
        seed=None,
        distributed=False,
        max_iterations=None,
    ))

    ppo_runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    ppo_runner.load(checkpoint_path)
    policy = ppo_runner.get_inference_policy(device="cuda:0")

    env_wrapped.reset()
    obs = env_wrapped.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    robot = env.unwrapped.scene["robot"]
    contact_sensor = env.unwrapped.scene["contact_forces"]

    heights = []
    body_contacts = {b: 0 for b in contact_sensor.body_names}
    body_forces = {b: [] for b in contact_sensor.body_names}
    feet_down_steps = []
    vel_errors = []

    for step in range(750):
        cmd = torch.zeros((64, 3), device="cuda:0")
        if step >= 250:
            cmd[:, 0] = 0.15 # vx = 0.15 m/s

        env.unwrapped.command_manager.get_term("base_velocity").vel_command_b[:, 0] = cmd[:, 0]

        with torch.inference_mode():
            actions = policy(obs)

        obs, rewards, dones, infos = env_wrapped.step(actions)
        
        h = robot.data.root_pos_w[:, 2].cpu().numpy()
        heights.extend(h)
        
        net_forces = contact_sensor.data.net_forces_w[:, :, 2].cpu().numpy()
        step_feet = 0
        for b_idx, b_name in enumerate(contact_sensor.body_names):
            f_vals = net_forces[:, b_idx]
            touching = (f_vals > 1.0)
            cnt = touching.sum().item()
            body_contacts[b_name] += cnt
            if cnt > 0:
                body_forces[b_name].extend(f_vals[touching])
            if ("NFOOT" in b_name or "FOOT" in b_name):
                step_feet += cnt
                
        feet_down_steps.append(step_feet / 64.0)

        if step >= 250:
            actual_vx = robot.data.root_lin_vel_b[:, 0].cpu().numpy()
            vel_errors.extend(np.abs(actual_vx - 0.15))

    env.close()

    h_mean = np.mean(heights)
    h_p5 = np.percentile(heights, 5)
    h_p95 = np.percentile(heights, 95)

    print(f"Base Height Distribution: Mean={h_mean:.4f} m, p5={h_p5:.4f} m, p95={h_p95:.4f} m")
    print(f"Velocity Tracking Error (vx=0.15 m/s): Mean Error={np.mean(vel_errors):.4f} m/s")
    print(f"Mean Feet in Contact per step: {np.mean(feet_down_steps):.2f} / 6")
    print("\nGround Contact Fraction & Forces by Body:")
    print(f"{'Body Name':<30} | {'Contact % of Steps':<20} | {'Mean Force (N)'}")
    print("-" * 65)
    total_samples = 64 * 750
    for b_name, count in body_contacts.items():
        pct = (count / total_samples) * 100.0
        mean_f = np.mean(body_forces[b_name]) if len(body_forces[b_name]) > 0 else 0.0
        if pct > 0.1:
            print(f"{b_name:<30} | {pct:<20.2f}% | {mean_f:.2f} N")


def main():
    if args_cli.test == "grid":
        run_1b_grid()
    elif args_cli.test == "probe":
        run_1d_probe(args_cli.checkpoint)

    simulation_app.close()


if __name__ == "__main__":
    main()
