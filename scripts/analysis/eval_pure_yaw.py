# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-yaw evaluation script on Stage 1 checkpoint."""

import argparse
import math
import os
import sys
import torch

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath("scripts/rsl_rl"))
import cli_args

parser = argparse.ArgumentParser(description="Evaluate pure yaw on checkpoint.")
parser.add_argument("--task", type=str, default="Zoe-Hexapod-Flat-Deploy-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--cmd_wz", type=float, default=0.40)
parser.add_argument("--steps", type=int, default=250)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = True
app_launcher = AppLauncher(args_cli)
sim_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from spdrbot3.tasks.manager_based.zoe_deploy.params import ZOE_PARAMS


def main():
    ckpt_path = os.path.abspath(args_cli.checkpoint or "logs/rsl_rl/zoe_hexapod_deploy/2026-11-05_19-51-02/model_599.pt")
    print(f"\n[EVAL PURE YAW] Checkpoint: {ckpt_path}")
    print(f"[EVAL PURE YAW] Target Yaw Rate: {args_cli.cmd_wz} rad/s (vx=0, vy=0), Steps: {args_cli.steps} ({args_cli.steps*0.02:.1f}s)")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None

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
    foot_names = [f"NFOOT{i}" for i in range(1, 7)]
    foot_ids = [body_names.index(n) for n in foot_names]

    # Joint limits [lower, upper] for 18 joints
    joint_limits = robot.data.joint_limits  # [num_envs, 18, 2]
    # simulator joint order: 0..5 coxa, 6..11 thigh, 12..17 foot
    coxa_ids = list(range(0, 6))
    thigh_ids = list(range(6, 12))
    foot_ids_joints = list(range(12, 18))

    obs_res = env.reset()
    obs = obs_res[0] if isinstance(obs_res, tuple) else obs_res

    wz_records = []
    feet_down_records = []
    torque_sat_records = []

    # Proximity threshold: 3 deg = 0.05236 rad
    PROX_RAD = math.radians(3.0)
    coxa_near_limit = 0
    thigh_near_limit = 0
    foot_near_limit = 0
    total_joint_steps_per_type = args_cli.steps * args_cli.num_envs * 6

    with torch.no_grad():
        for step in range(args_cli.steps):
            cmd = unwrapped.command_manager.get_command("base_velocity")
            cmd[:, 0] = 0.0
            cmd[:, 1] = 0.0
            cmd[:, 2] = args_cli.cmd_wz

            pol_obs = obs["policy"] if isinstance(obs, dict) else obs
            actions = policy(pol_obs)

            # Record metrics
            wz = robot.data.root_ang_vel_b[:, 2]
            wz_records.append(wz.cpu())

            forces_z = contact_sensor.data.net_forces_w[:, foot_ids, 2]
            f_down = (forces_z > 1.0).sum(dim=1).float()
            feet_down_records.append(f_down.cpu())

            torques = torch.abs(robot.data.applied_torque)
            sat = (torques >= 2.45).float().mean(dim=-1)
            torque_sat_records.append(sat.cpu())

            # Joint limit proximity
            q = robot.data.joint_pos
            lower = joint_limits[:, :, 0]
            upper = joint_limits[:, :, 1]
            near = (q - lower <= PROX_RAD) | (upper - q <= PROX_RAD)  # [N, 18]

            coxa_near_limit += near[:, coxa_ids].sum().item()
            thigh_near_limit += near[:, thigh_ids].sum().item()
            foot_near_limit += near[:, foot_ids_joints].sum().item()

            step_res = env.step(actions)
            obs = step_res[0]

    all_wz = torch.cat(wz_records)
    all_feet = torch.cat(feet_down_records)
    all_sat = torch.cat(torque_sat_records)

    mean_wz = all_wz.mean().item()
    std_wz = all_wz.std().item()
    mean_feet = all_feet.mean().item()
    mean_sat_pct = all_sat.mean().item() * 100.0

    coxa_prox_pct = 100.0 * coxa_near_limit / total_joint_steps_per_type
    thigh_prox_pct = 100.0 * thigh_near_limit / total_joint_steps_per_type
    foot_prox_pct = 100.0 * foot_near_limit / total_joint_steps_per_type

    # Reward budget comparison for pure yaw
    # Target wz = 0.40, kernel std = 0.25
    # If wz = 0.038: error = 0.362 -> exp(-(0.362/0.25)^2) = exp(-2.0967) = 0.1228 -> reward = 0.5 * 0.1228 = 0.0614/step = +3.07/s
    # If wz = 0.400: error = 0.000 -> exp(0) = 1.0000 -> reward = 0.5 * 1.0000 = 0.5000/step = +25.00/s
    # Max reward gain from perfect turning = 25.00 - 3.07 = +21.93/s

    print("\n=======================================================")
    print("           PURE YAW EVALUATION RESULTS                 ")
    print("=======================================================")
    print(f"Commanded Yaw Rate        : {args_cli.cmd_wz:.2f} rad/s")
    print(f"Measured Base Yaw Rate    : {mean_wz:+.4f} +- {std_wz:.4f} rad/s ({mean_wz/args_cli.cmd_wz*100.1:.1f}% tracking)")
    print(f"Mean Feet Down            : {mean_feet:.2f} / 6.0")
    print(f"Torque Saturation (|tau|>=2.45): {mean_sat_pct:.2f}% joint-steps")
    print(f"Coxa  Near Limit (<=3 deg): {coxa_prox_pct:.2f}% joint-steps")
    print(f"Thigh Near Limit (<=3 deg): {thigh_prox_pct:.2f}% joint-steps")
    print(f"Foot  Near Limit (<=3 deg): {foot_prox_pct:.2f}% joint-steps")
    print("-------------------------------------------------------")
    print("Reward Budget Analysis:")
    curr_yaw_gain = 0.5 * math.exp(-((mean_wz - args_cli.cmd_wz)/0.25)**2) * 50.0
    max_yaw_gain = 0.5 * 1.0 * 50.0
    delta_gain = max_yaw_gain - curr_yaw_gain
    print(f"  Current Yaw Reward Rate : {curr_yaw_gain:+.2f} / s")
    print(f"  Max Potential Yaw Reward: {max_yaw_gain:+.2f} / s")
    print(f"  Net Incentive to Turn   : {delta_gain:+.2f} / s")
    print("=======================================================\n")

    sim_app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
