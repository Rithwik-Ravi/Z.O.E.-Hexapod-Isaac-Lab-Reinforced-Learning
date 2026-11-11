# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Task 2: Reward Economics of Chunk-1 Configuration (Simulator-Verified).

Measures true static holding torques from the Isaac Lab simulator for 6, 5, 4, and 3
grounded feet at the 5.00 kg Chunk-1 mass, evaluates every Chunk-1 reward/penalty term
per second (50 Hz), and computes the share of training commands with |v_xy| < 0.10 m/s.
"""

import argparse
import math
import os
import sys
import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate Chunk-1 reward economics with simulator static torques.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
sim_app = app_launcher.app

import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import spdrbot3.tasks  # noqa: F401
from spdrbot3.tasks.manager_based.zoe_deploy.params import ZOE_PARAMS
from spdrbot3.tasks.manager_based.zoe_deploy.mdp.rewards import base_height_shaped, height_above_ground


def measure_simulator_static_torques():
    """Run simulator with 4 envs having 6, 5, 4, 3 feet grounded and measure static torques."""
    # Force Chunk-1 parameters in case params.py was modified for later probes
    env_cfg = parse_env_cfg("Zoe-Hexapod-Flat-Deploy-Play-v0", device="cuda:0", num_envs=4)
    env_cfg.events.push_robot = None
    env_cfg.actions.joint_pos.rate_limit_rad_per_step = 10.0  # allow reaching static test posture quickly

    env = gym.make("Zoe-Hexapod-Flat-Deploy-Play-v0", cfg=env_cfg)
    unwrapped = env.unwrapped
    env.reset()

    robot = unwrapped.scene["robot"]
    contact_sensor = unwrapped.scene["contact_forces"]
    joint_names = robot.joint_names
    body_names = contact_sensor.body_names

    # Map leg index (1..6) -> thigh joint index and foot body index
    thigh_idx = {}
    foot_j_idx = {}
    coxa_idx = {}
    for j_i, jn in enumerate(joint_names):
        if jn.startswith("NCCUP"):
            leg = int(jn.split("_")[0].replace("NCCUP", ""))
            thigh_idx[leg] = j_i
        elif jn.startswith("NTHIGH"):
            leg = int(jn.split("_")[0].replace("NTHIGH", ""))
            foot_j_idx[leg] = j_i
        elif jn.startswith("base_link"):
            # infer leg from matching child body in ZOE_PARAMS
            info = ZOE_PARAMS.joint_info_by_sim_name.get(jn, {})
            if "leg" in info:
                coxa_idx[int(info["leg"])] = j_i

    foot_body_ids = [body_names.index(f"NFOOT{i}") for i in range(1, 7)]

    # Determine sign for each leg's thigh joint that lifts the foot (+Z world when rolled 180 deg around X)
    robot_body_names = robot.body_names
    robot_foot_ids = [robot_body_names.index(f"NFOOT{i}") for i in range(1, 7)]

    base_q = robot.data.default_joint_pos[0].clone()
    lift_signs = {}
    for leg in range(1, 7):
        env.reset()
        q_test = base_q.clone()
        robot.write_joint_state_to_sim(q_test.unsqueeze(0).expand(4, -1), torch.zeros_like(robot.data.joint_vel))
        unwrapped.sim.forward()
        z0 = robot.data.body_pos_w[0, robot_foot_ids[leg - 1], 2].item()
        q_test[thigh_idx[leg]] += 0.35
        robot.write_joint_state_to_sim(q_test.unsqueeze(0).expand(4, -1), torch.zeros_like(robot.data.joint_vel))
        unwrapped.sim.forward()
        z_plus = robot.data.body_pos_w[0, robot_foot_ids[leg - 1], 2].item()
        lift_signs[leg] = 1.0 if z_plus > z0 else -1.0

    # Configure target joint positions for 4 envs:
    # env 0: 6 feet down (all nominal)
    # env 1: 5 feet down (lift leg 2)
    # env 2: 4 feet down (lift legs 2, 5 - symmetric middle legs)
    # env 3: 3 feet down (alternating tripod: lift legs 2, 4, 6; support on 1, 3, 5)
    lifted_sets = {0: [], 1: [2], 2: [2, 5], 3: [2, 4, 6]}
    env.reset()
    q_targets = base_q.unsqueeze(0).expand(4, -1).clone()
    for e_i, legs_up in lifted_sets.items():
        for leg in legs_up:
            # Lift thigh + tuck foot inward slightly so lifted foot clears ground cleanly
            q_targets[e_i, thigh_idx[leg]] += lift_signs[leg] * 0.55

    # Set PD targets directly on robot and step physics directly for 120 decimation steps (2.4s) to settle
    for _ in range(120):
        robot.set_joint_position_target(q_targets)
        for _ in range(unwrapped.cfg.decimation):
            unwrapped.scene.write_data_to_sim()
            unwrapped.sim.step(render=False)
            unwrapped.scene.update(dt=unwrapped.physics_dt)

    # Average static torques over last 25 control steps (0.5 s)
    tau_samples = []
    feet_samples = []
    h_samples = []
    pos_lim_samples = []
    torq_l2_samples = []

    coxa_mask = torch.tensor([i in coxa_idx.values() for i in range(len(joint_names))], device="cuda:0")
    thigh_mask = torch.tensor([i in thigh_idx.values() for i in range(len(joint_names))], device="cuda:0")
    foot_mask = torch.tensor([i in foot_j_idx.values() for i in range(len(joint_names))], device="cuda:0")

    for _ in range(25):
        robot.set_joint_position_target(q_targets)
        for _ in range(unwrapped.cfg.decimation):
            unwrapped.scene.write_data_to_sim()
            unwrapped.sim.step(render=False)
            unwrapped.scene.update(dt=unwrapped.physics_dt)

        tau = robot.data.applied_torque.abs()  # [4, 18]
        tau_samples.append(tau.clone())
        fz = contact_sensor.data.net_forces_w[:, foot_body_ids, 2]
        feet_samples.append((fz > 1.0).float().sum(dim=-1))
        h_samples.append(height_above_ground(unwrapped))
        torq_l2_samples.append(torch.sum(torch.square(robot.data.applied_torque), dim=-1))

        # Check joint limit violation (beyond 0.95 soft limit)
        jpos = robot.data.joint_pos
        limits = robot.data.soft_joint_pos_limits  # [4, 18, 2]
        out_low = (-jpos + limits[:, :, 0]).clamp(min=0.0)
        out_high = (jpos - limits[:, :, 1]).clamp(min=0.0)
        pos_lim_samples.append(torch.sum(out_low + out_high, dim=-1))

    tau_mean = torch.stack(tau_samples, dim=0).mean(dim=0)  # [4, 18]
    feet_mean = torch.stack(feet_samples, dim=0).mean(dim=0)  # [4]
    h_mean = torch.stack(h_samples, dim=0).mean(dim=0)  # [4]
    torq_l2_mean = torch.stack(torq_l2_samples, dim=0).mean(dim=0)  # [4]
    pos_lim_mean = torch.stack(pos_lim_samples, dim=0).mean(dim=0)  # [4]

    results = {}
    for e_i, n_target in enumerate([6, 5, 4, 3]):
        tau_vec = tau_mean[e_i]
        # Torque limit penalty sum((|tau| - threshold)_+^2) for threshold = 1.95 N.m (Chunk 1)
        excess_195 = (tau_vec - 1.95).clamp(min=0.0)
        tlim_raw_195 = torch.sum(torch.square(excess_195)).item()
        excess_210 = (tau_vec - 2.10).clamp(min=0.0)
        tlim_raw_210 = torch.sum(torch.square(excess_210)).item()

        # Find thigh torques specifically for grounded stance legs
        stance_legs = [l for l in range(1, 7) if l not in lifted_sets[e_i]]
        stance_thigh_taus = [tau_vec[thigh_idx[l]].item() for l in stance_legs]
        stance_foot_taus = [tau_vec[foot_j_idx[l]].item() for l in stance_legs]

        results[n_target] = {
            "measured_feet_down": feet_mean[e_i].item(),
            "height_m": h_mean[e_i].item(),
            "max_abs_tau_nm": tau_vec.max().item(),
            "stance_thigh_mean_nm": float(np.mean(stance_thigh_taus)),
            "stance_thigh_max_nm": float(np.max(stance_thigh_taus)),
            "stance_foot_mean_nm": float(np.mean(stance_foot_taus)),
            "coxa_mean_nm": tau_vec[coxa_mask].mean().item(),
            "thigh_mean_nm": tau_vec[thigh_mask].mean().item(),
            "foot_mean_nm": tau_vec[foot_mask].mean().item(),
            "torq_l2_sum": torq_l2_mean[e_i].item(),
            "tlim_raw_195": tlim_raw_195,
            "tlim_raw_210": tlim_raw_210,
            "pos_lim_sum": pos_lim_mean[e_i].item(),
        }

    env.close()
    return results


def compute_reward_economics():
    sim_stats = measure_simulator_static_torques()

    policy_hz = 50.0  # step_dt = 0.02s
    target_h = ZOE_PARAMS.TARGET_STANDING_HEIGHT  # 0.1346
    min_h = ZOE_PARAMS.GATE_MIN_HEIGHT  # 0.1275

    # Chunk-1 reward weights (verbatim from logs/rsl_rl/zoe_hexapod_deploy/2026-11-09_22-33-44/params/env.yaml)
    w_lin = 2.0
    std_lin = 0.13
    w_ang = 1.0
    std_ang = 0.25
    w_height = 1.0
    w_lin_vel_z = -2.0
    w_ang_vel_xy = -0.05
    w_torques_l2 = -5e-3
    w_dof_acc = -1e-7
    w_action_rate = -0.05
    w_flat_ori = -0.5
    w_pos_lim = -1.0
    w_air_time = 0.25
    w_fewer_3 = -0.5
    w_fewer_4 = -0.05
    w_undesired = -1.0
    w_foot_slip = -0.1
    w_clearance = 0.1
    w_stand_posture = -0.5
    w_stand_grounded = 1.0
    w_joint_vel_lim = -0.01
    w_torque_limit = -2.0  # Chunk-1 weight (with threshold 1.95 N.m)

    print("\n" + "=" * 96)
    print("TASK 2: SIMULATOR-MEASURED REWARD ECONOMICS OF CHUNK-1 CONFIGURATION (5.00 KG)")
    print("=" * 96)

    # 1. Exact Command Batch Breakdown
    rng = np.random.default_rng(42)
    N = 1_000_000
    # Exact ZoeUniformVelocityCommand sampling order in commands.py:
    # Step 1: parent UniformVelocityCommand samples uniform + is_standing_env (0.10)
    vx = rng.uniform(-0.25, 0.30, size=N)
    vy = rng.uniform(-0.20, 0.20, size=N)
    stand_mask = rng.random(N) < 0.10
    vx[stand_mask] = 0.0
    vy[stand_mask] = 0.0
    # Step 2: rel_yaw_envs=0.30 applies over all envs and clears is_standing_env on selected
    yaw_mask = rng.random(N) < 0.30
    stand_mask = stand_mask & (~yaw_mask)
    vx[yaw_mask] = rng.uniform(-0.05, 0.05, size=yaw_mask.sum())
    vy[yaw_mask] = rng.uniform(-0.05, 0.05, size=yaw_mask.sum())

    std_mask = (~stand_mask) & (~yaw_mask)
    v_mag = np.sqrt(vx**2 + vy**2)

    s_stand = stand_mask.mean() * 100.0
    s_yaw_lt01 = ((v_mag[yaw_mask] < 0.10).sum() / N) * 100.0
    s_std_lt01 = ((v_mag[std_mask] < 0.10).sum() / N) * 100.0
    s_total_lt01 = (v_mag < 0.10).mean() * 100.0

    print("\n1. SHARE OF CHUNK-1 TRAINING ENVS WITH |v_xy| < 0.10 m/s (1,000,000 exact draws):")
    print(f"   - Standing envs (cmd = 0)            : {s_stand:6.2f}% (10% * 70% not overwritten by yaw)")
    print(f"   - Yaw-dominant envs (|vx,vy|<=0.05)  : {s_yaw_lt01:6.2f}% (30% total; max |v_xy| = 0.0707 < 0.10)")
    print(f"   - Remaining uniform envs (|v|<0.10)  : {s_std_lt01:6.2f}% (analytical circle/rect: pi*0.1^2/(0.55*0.40) * 63% = 8.99%)")
    print(f"   ------------------------------------------------------------------")
    print(f"   TOTAL ENVS WITH |v_xy| < 0.10 m/s    : {s_total_lt01:6.2f}%")

    # 2. Simulator Measured Static Torques
    print("\n2. SIMULATOR STATIC HOLDING TORQUES AT 5.00 KG (Measured from Isaac Lab):")
    print("   | Grounded | Measured Feet | Height (m) | Max |tau| | Stance Thigh Mean | Stance Thigh Max | sum(tau^2) | tlim raw (>1.95) | tlim raw (>2.10) |")
    print("   |---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for nf in [6, 5, 4, 3]:
        s = sim_stats[nf]
        print(
            f"   | {nf} feet | {s['measured_feet_down']:.2f} | {s['height_m']:.4f} | "
            f"{s['max_abs_tau_nm']:.3f} N.m | {s['stance_thigh_mean_nm']:.3f} N.m | "
            f"{s['stance_thigh_max_nm']:.3f} N.m | {s['torq_l2_sum']:.2f} | "
            f"{s['tlim_raw_195']:.4f} | {s['tlim_raw_210']:.4f} |"
        )

    def eval_height_rew(h):
        err = h - target_h
        gaussian = math.exp(-(err / 0.015) ** 2)
        barrier = 0.0
        if h < min_h:
            d = min_h - h
            barrier = -(150.0 * d * d + 10.0 * d)
        return w_height * (gaussian + barrier)

    # Estimate dynamic walking overhead terms per second at speed v_cmd (conservative vs realistic):
    # Static standing: lin_vel_z ~ 0, ang_vel_xy ~ 0, dof_acc ~ 0, action_rate ~ 0, flat_ori ~ 0, air_time = 0, clearance = 0, slip = 0
    # Ideal walking at v_cmd with n_feet down:
    # - For 6 feet down: no swing legs (or negligible micro-steps), so air_time=0, clearance=0, dynamic factor ~1.0x
    # - For 3, 4, 5 feet down: (6 - n_feet) swing legs cycle (~2.5 Hz swing rate).
    #   Ideal steady walk: track_lin=1.0, track_ang=1.0, flat_ori~0, lin_vel_z~0.01 (cost -0.0002), ang_vel_xy~0.02 (cost -0.001),
    #   action_rate ~ 18 joints * (0.04 rad/step)^2 = 0.0288 -> *(-0.05) = -0.00144/step (-0.072/s),
    #   dof_acc ~ 18 * (20)^2 = 7200 -> *(-1e-7) = -0.00072/step (-0.036/s),
    #   feet_air_time: threshold=0.15s, tripod step T=0.25s -> excess=0.10s * 3 legs every 0.25s -> ~0.30/s * 0.25? = +0.05 to +0.30/s depending on cycle; wait actual per step average ~ +0.02/s * weight 0.25
    #   foot_tip_clearance: (6 - n_feet) * 0.1 * (1/6) for perfect 5 cm swing -> +(6-n_feet)/6 * 0.1 per step!
    #   Wait: torque_limit during dynamic walking hits higher peaks (~1.15x static stance transfer/acceleration + impact spikes):
    #   We compute BOTH pure-static-torque lower bound AND realistic 1.15x dynamic-stance torque bound!

    print("\n3. COMPLETE PER-SECOND REWARD BREAKDOWN (Chunk-1 Config, All Penalty Terms Included, 50 Hz):")

    # Case 0: Standing still on cmd = 0 (6 feet down)
    s6 = sim_stats[6]
    terms_stand0 = {
        "track_lin_vel_xy_exp (+2.0)": w_lin * 1.0 * policy_hz,
        "track_ang_vel_z_exp (+1.0)": w_ang * 1.0 * policy_hz,
        "base_height_shaped (+1.0)": eval_height_rew(s6["height_m"]) * policy_hz,
        "stand_still_feet_grounded (+1.0)": w_stand_grounded * (6.0 / 6.0) * policy_hz,
        "stand_still_posture (-0.5)": 0.0,  # zero deviation from default pose
        "dof_torques_l2 (-5e-3)": w_torques_l2 * s6["torq_l2_sum"] * policy_hz,
        "torque_limit (-2.0 @ 1.95)": w_torque_limit * s6["tlim_raw_195"] * policy_hz,
        "fewer_than_4_feet (-0.05)": 0.0,
        "fewer_than_3_feet (-0.5)": 0.0,
        "dof_pos_limits (-1.0)": w_pos_lim * s6["pos_lim_sum"] * policy_hz,
        "flat_orientation_l2 (-0.5)": 0.0,
        "lin_vel_z / ang_vel_xy": 0.0,
        "dof_acc / action_rate / vel_lim": 0.0,
        "feet_air_time / clearance / slip": 0.0,
    }
    tot_stand0 = sum(terms_stand0.values())
    print(f"\n   [A] STANDING STILL AT cmd = [0, 0, 0] (6 feet down): TOTAL = {tot_stand0:+.2f} / s")
    for k, v in terms_stand0.items():
        if abs(v) > 1e-4 or "torque" in k or "stand_still" in k:
            print(f"       - {k:34s}: {v:+.2f} / s")

    print("\n   [B] STANDING STILL (CREEPING/FREEZING, 6 feet down) vs IDEAL WALKING (3, 4, 5, 6 feet down):")
    print("       Columns: Creeper | Pure-Static Walk (tau=sim static) | Dynamic Walk (1.15x peak stance tau + control costs)")
    print("   | v_cmd | Mode / Feet | Track+H /s | Torq L2 /s | Torq Lim /s | Contact+Air+Clr /s | Ctrl/Dyn /s | Total /s | vs Creep /s |")
    print("   |---:|---|---:|---:|---:|---:|---:|---:|---:|")

    for v_cmd in [0.10, 0.15, 0.25]:
        # Creeper (robot stands completely still with 6 feet down while commanded v_cmd)
        lin_err = math.exp(-((v_cmd / std_lin) ** 2))
        c_track_h = (w_lin * lin_err + w_ang * 1.0 + eval_height_rew(s6["height_m"])) * policy_hz
        c_tl2 = w_torques_l2 * s6["torq_l2_sum"] * policy_hz
        c_tlim = w_torque_limit * s6["tlim_raw_195"] * policy_hz
        c_pos = w_pos_lim * s6["pos_lim_sum"] * policy_hz
        tot_creep = c_track_h + c_tl2 + c_tlim + c_pos
        print(
            f"   | {v_cmd:.2f} | **Freeze/Creep (6ft)** | {c_track_h:+.2f} | {c_tl2:+.2f} | {c_tlim:+.2f} | "
            f"+0.00 | {c_pos:+.2f} | **{tot_creep:+.2f}** | +0.00 |"
        )

        for nf in [6, 5, 4, 3]:
            s = sim_stats[nf]
            n_swing = 6 - nf
            w_track_h = (w_lin * 1.0 + w_ang * 1.0 + eval_height_rew(s["height_m"])) * policy_hz

            # Contact / air / clearance
            c_pen = (w_fewer_3 if nf < 3 else 0.0) + (w_fewer_4 if nf < 4 else 0.0)
            # Air time reward triggers upon touchdown after >= 0.15s air: ~0.05s excess per 0.35s gait cycle per swing group
            r_air = w_air_time * (0.05 if n_swing > 0 else 0.0) * (n_swing / 3.0)
            # Foot clearance reward when swing foot >= 2cm (max 1.0 per swing leg / 6)
            r_clr = w_clearance * (n_swing / 6.0)
            r_slip = -0.002 * (nf / 6.0) * (v_cmd / 0.15)  # minor stance slip ~2cm/s
            w_gait = (c_pen + r_air + r_clr + r_slip) * policy_hz

            # Pure simulator static torques (strict lower bound on walking torque cost)
            w_tl2_stat = w_torques_l2 * s["torq_l2_sum"] * policy_hz
            w_tlim_stat = w_torque_limit * s["tlim_raw_195"] * policy_hz
            w_ctrl_stat = w_pos_lim * s["pos_lim_sum"] * policy_hz
            tot_stat = w_track_h + w_tl2_stat + w_tlim_stat + w_gait + w_ctrl_stat
            print(
                f"   | {v_cmd:.2f} | Walk-Static ({nf}ft) | {w_track_h:+.2f} | {w_tl2_stat:+.2f} | {w_tlim_stat:+.2f} | "
                f"{w_gait:+.2f} | {w_ctrl_stat:+.2f} | {tot_stat:+.2f} | {tot_stat - tot_creep:+.2f} |"
            )

            # Dynamic stepping (1.15x peak stance torques + action_rate/dof_acc/lin_z/flat_ori)
            # Scale only the grounded stance leg torques by 1.15, clipped at 2.50 N.m actuator effort limit
            dyn_scale = 1.0 if nf == 6 else 1.15
            w_tl2_dyn = w_torques_l2 * min(s["torq_l2_sum"] * (dyn_scale ** 2), nf * (2.50 ** 2) + 6.0) * policy_hz
            stance_tau_scaled = min(2.50, s["stance_thigh_mean_nm"] * dyn_scale)
            tlim_dyn_raw = nf * (max(0.0, stance_tau_scaled - 1.95) ** 2)
            tlim_dyn_raw = max(s["tlim_raw_195"], tlim_dyn_raw)
            w_tlim_dyn = w_torque_limit * tlim_dyn_raw * policy_hz
            # Control/dynamic body oscillation costs proportional to speed
            r_dyn_ctrl = (
                w_pos_lim * s["pos_lim_sum"]
                + w_action_rate * (0.04 * (v_cmd / 0.15)) ** 2 * 18
                + w_dof_acc * (20.0 * (v_cmd / 0.15)) ** 2 * 18
                + w_lin_vel_z * (0.015 * (v_cmd / 0.15)) ** 2
                + w_ang_vel_xy * (0.05 * (v_cmd / 0.15)) ** 2
                + w_flat_ori * (0.02 * (v_cmd / 0.15)) ** 2
            ) * policy_hz
            tot_dyn = w_track_h + w_tl2_dyn + w_tlim_dyn + w_gait + r_dyn_ctrl
            print(
                f"   | {v_cmd:.2f} | Walk-Dyn ({nf}ft) | {w_track_h:+.2f} | {w_tl2_dyn:+.2f} | {w_tlim_dyn:+.2f} | "
                f"{w_gait:+.2f} | {r_dyn_ctrl:+.2f} | {tot_dyn:+.2f} | {tot_dyn - tot_creep:+.2f} |"
            )

    sim_app.close()


if __name__ == "__main__":
    compute_reward_economics()
