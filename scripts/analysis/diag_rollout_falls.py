# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Rollout evaluation & fall dynamics audit for Zoe-Hexapod-Terrain-v0."""

import argparse
import os
import sys
import torch
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Rollout and fall dynamics analysis on multi-terrain.")
parser.add_argument("--condition", type=str, required=True, choices=["zero", "flat", "terrain"], help="Condition to run.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments.")
parser.add_argument("--steps", type=int, default=250, help="Steps per rollout (250 steps = 5.0 s).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
sim_app = app_launcher.app

import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from spdrbot3.tasks.manager_based.zoe_deploy.agents.rsl_rl_ppo_cfg import ZoeTerrainPPORunnerCfg

def load_policy(checkpoint_path, env):
    print(f"Loading checkpoint via OnPolicyRunner: {checkpoint_path}")
    wrapped_env = RslRlVecEnvWrapper(env)
    agent_cfg = ZoeTerrainPPORunnerCfg()
    ppo_runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device)
    ppo_runner.load(checkpoint_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    return policy

def main():
    flat_ckpt = "/home/rithwik/SpdrBot-main/logs/rsl_rl/zoe_hexapod_deploy/2026-11-05_19-51-02/model_599.pt"
    terrain_ckpt = "/home/rithwik/SpdrBot-main/logs/rsl_rl/zoe_hexapod_terrain/2026-11-06_22-18-05/model_9999.pt"

    env_cfg = parse_env_cfg("Zoe-Hexapod-Terrain-v0", num_envs=args_cli.num_envs)
    env = gym.make("Zoe-Hexapod-Terrain-v0", cfg=env_cfg)
    obs, _ = env.reset()

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    contact_sensor = unwrapped.scene["contact_forces"]
    terrain = unwrapped.scene.terrain

    policy = None
    if args_cli.condition == "flat":
        policy = load_policy(flat_ckpt, env)
        cmd_tensor = torch.tensor([0.15, 0.0, 0.0], device=unwrapped.device).repeat(args_cli.num_envs, 1)
        unwrapped.command_manager.get_term("base_velocity").vel_command_b[:] = cmd_tensor
    elif args_cli.condition == "terrain":
        policy = load_policy(terrain_ckpt, env)

    print(f"\n========================================================")
    print(f"RUNNING ROLLOUT: Condition={args_cli.condition} ({args_cli.num_envs} envs, {args_cli.steps} steps = {args_cli.steps*0.02:.1f}s)")
    print(f"========================================================")

    # Tracking buffers
    num_envs = args_cli.num_envs
    num_steps = args_cli.steps
    survived = torch.ones(num_envs, dtype=torch.bool, device=unwrapped.device)
    term_step = torch.full((num_envs,), num_steps, dtype=torch.long, device=unwrapped.device)
    term_reasons = { "time_out": 0, "base_contact": 0, "bad_orientation": 0, "invalid_state_nan": 0 }

    history_roll = []
    history_pitch = []
    history_actions = []
    history_torques = []
    history_joint_pos = []
    history_feet_contact = []

    feet_body_names = [name for name in contact_sensor.body_names if "NFOOT" in name]
    feet_body_ids = [contact_sensor.body_names.index(name) for name in feet_body_names]

    illegal_body_names = [name for name in contact_sensor.body_names if any(b in name for b in ["base_link", "NCCUP", "NTHIGH"])]
    illegal_body_ids = [contact_sensor.body_names.index(name) for name in illegal_body_names]

    for step in range(num_steps):
        if policy is None:
            actions = torch.zeros((num_envs, 18), device=unwrapped.device)
        else:
            with torch.no_grad():
                actions = policy(obs)

        # Record root roll/pitch
        quat = robot.data.root_quat_w
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

        history_roll.append(roll.clone())
        history_pitch.append(pitch.clone())
        history_actions.append(actions.clone())
        history_torques.append(robot.data.applied_torque.clone())
        history_joint_pos.append(robot.data.joint_pos.clone())

        net_f = contact_sensor.data.net_forces_w
        f_down = (torch.norm(net_f[:, feet_body_ids, :], dim=-1) > 1.0)
        history_feet_contact.append(f_down.clone())

        obs, rew, terminated, truncated, info = unwrapped.step(actions)

        just_terminated = (terminated | truncated) & survived
        if just_terminated.any():
            is_bad_ori = (torch.acos(torch.clamp(robot.data.projected_gravity_b[:, 2], -1.0, 1.0)).abs() > 1.2)
            illegal_forces = contact_sensor.data.net_forces_w[:, illegal_body_ids, :]
            is_base_contact = (torch.norm(illegal_forces, dim=-1) > 1.0).any(dim=1)
            for env_i in torch.where(just_terminated)[0]:
                survived[env_i] = False
                term_step[env_i] = step
                if is_bad_ori[env_i]:
                    term_reasons["bad_orientation"] = term_reasons.get("bad_orientation", 0) + 1
                elif is_base_contact[env_i]:
                    term_reasons["base_contact"] = term_reasons.get("base_contact", 0) + 1
                else:
                    term_reasons["other"] = term_reasons.get("other", 0) + 1

    survival_pct = survived.float().mean().item() * 100.0
    mean_time_to_fall = term_step.float().mean().item() * 0.02
    print(f"\n--- Final Rollout Report [{args_cli.condition}] ---")
    print(f"Survival Rate: {survival_pct:.1f}% ({survived.sum().item()}/{num_envs})")
    print(f"Mean Time to Fall / End: {mean_time_to_fall:.2f} s")
    print(f"Termination Reasons: {term_reasons}")

    all_rolls = torch.stack(history_roll, dim=0) # [T, N]
    all_pitch = torch.stack(history_pitch, dim=0)
    all_acts = torch.stack(history_actions, dim=0)
    all_torques = torch.stack(history_torques, dim=0)
    all_jpos = torch.stack(history_joint_pos, dim=0)
    all_fdown = torch.stack(history_feet_contact, dim=0)

    final_rolls = [all_rolls[min(term_step[i].item(), num_steps - 1), i].item() for i in range(num_envs)]
    final_pitch = [all_pitch[min(term_step[i].item(), num_steps - 1), i].item() for i in range(num_envs)]
    mean_roll = np.mean(final_rolls)
    mean_pitch = np.mean(final_pitch)
    left_falls = sum(1 for r in final_rolls if r > 0.3)
    right_falls = sum(1 for r in final_rolls if r < -0.3)
    
    print(f"Final Roll: mean={np.degrees(mean_roll):+.1f}° | Left tilts (>+17°): {left_falls}, Right tilts (<-17°): {right_falls}")
    print(f"Final Pitch: mean={np.degrees(mean_pitch):+.1f}°")

    # Joint limit proximity
    joint_limits = robot.data.joint_pos_limits
    dist_to_lower = torch.abs(all_jpos - joint_limits[:, :, 0].unsqueeze(0))
    dist_to_upper = torch.abs(all_jpos - joint_limits[:, :, 1].unsqueeze(0))
    near_limit = (dist_to_lower < np.radians(3.0)) | (dist_to_upper < np.radians(3.0))
    limit_hit_frac = near_limit.float().mean(dim=(0, 1)).cpu().numpy()

    joint_names = unwrapped.scene["robot"].joint_names
    print("\nJoint Limit Proximity (<3 deg from limit):")
    for j_i, j_name in enumerate(joint_names):
        print(f"  Joint {j_i:2d} ({j_name:<25}): {limit_hit_frac[j_i]*100:5.1f}%")

    # Mean action per joint
    mean_actions = all_acts.mean(dim=(0, 1)).cpu().numpy()
    action_std = all_acts.std(dim=(0, 1)).cpu().numpy()
    print("\nMean Action & Std per joint:")
    for j_i, j_name in enumerate(joint_names):
        print(f"  Joint {j_i:2d} ({j_name:<25}): mean={mean_actions[j_i]:+6.3f}, std={action_std[j_i]:6.3f}")

    # Mirrored legs asymmetry
    # Leg order is [6, 1, 5, 4, 3, 2]
    # Left legs: Leg 1, 2, 3 (indices: Coxa [1, 5, 4], Thigh [7, 11, 10], Foot [13, 17, 16])
    # Right legs: Leg 6, 5, 4 (indices: Coxa [0, 2, 3], Thigh [6, 8, 9], Foot [12, 14, 15])
    left_coxa_act = mean_actions[[1, 5, 4]].mean()
    right_coxa_act = mean_actions[[0, 2, 3]].mean()
    left_thigh_act = mean_actions[[7, 11, 10]].mean()
    right_thigh_act = mean_actions[[6, 8, 9]].mean()
    left_foot_act = mean_actions[[13, 17, 16]].mean()
    right_foot_act = mean_actions[[12, 14, 15]].mean()
    print("\n--- Left vs Right Mirrored Action Asymmetry ---")
    print(f"Coxa Actions:  Left (Legs 1,2,3) = {left_coxa_act:+6.3f} vs Right (Legs 6,5,4) = {right_coxa_act:+6.3f} (diff = {left_coxa_act - right_coxa_act:+6.3f})")
    print(f"Thigh Actions: Left (Legs 1,2,3) = {left_thigh_act:+6.3f} vs Right (Legs 6,5,4) = {right_thigh_act:+6.3f} (diff = {left_thigh_act - right_thigh_act:+6.3f})")
    print(f"Foot Actions:  Left (Legs 1,2,3) = {left_foot_act:+6.3f} vs Right (Legs 6,5,4) = {right_foot_act:+6.3f} (diff = {left_foot_act - right_foot_act:+6.3f})")

    # Torque saturation per joint
    torque_sat_per_joint = (torch.abs(all_torques) >= 2.45).float().mean(dim=(0, 1)).cpu().numpy()
    print("\nTorque Saturation (>= 2.45 Nm) per joint:")
    for j_i, j_name in enumerate(joint_names):
        print(f"  Joint {j_i:2d} ({j_name:<25}): {torque_sat_per_joint[j_i]*100:5.1f}%")

    # Feet loading asymmetry
    feet_loading = all_fdown.float().mean(dim=(0, 1)).cpu().numpy()
    print("\nFeet Ground Contact Duty Cycle:")
    # feet order: NFOOT1..6
    for f_i, f_name in enumerate(feet_body_names):
        print(f"  {f_name}: {feet_loading[f_i]*100:5.1f}%")

    env.close()

if __name__ == "__main__":
    main()
    sim_app.close()
