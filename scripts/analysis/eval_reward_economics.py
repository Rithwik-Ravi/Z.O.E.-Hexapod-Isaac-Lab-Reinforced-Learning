# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Task 2: Reward Economics of Chunk-1 Configuration.

Computes:
1. Per-second reward of standing still vs ideal walking at 0.10, 0.15, 0.25 m/s
   with 3, 4, 5, and 6 feet down using simulator static torques.
2. Exact share of training environments with |v_xy| < 0.10 m/s.
"""

import math
import numpy as np
import torch

from spdrbot3.tasks.manager_based.zoe_deploy.params import ZOE_PARAMS


def compute_reward_economics():
    policy_hz = 50.0  # dt = 0.02s
    total_mass = 5.00  # kg
    g = 9.81
    total_weight_N = total_mass * g  # 49.05 N

    # Geometry at nominal standing height 0.142m:
    # Lever arm to thigh joint axis is approx 0.115m
    lever_arm_m = 0.115

    # Static torque per leg for N feet down:
    # tau_leg = (W / N) * lever_arm
    def get_static_tau(n_feet):
        return (total_weight_N / n_feet) * lever_arm_m

    # Chunk-1 reward weights and parameters:
    w_lin = 2.0
    std_lin = ZOE_PARAMS.rewards.track_lin_vel_std  # 0.13 m/s
    w_ang = 1.0  # Chunk 1 yaw weight was 1.0
    std_ang = ZOE_PARAMS.rewards.track_ang_vel_std  # 0.25 rad/s
    w_height = 1.0
    w_torques_l2 = -0.005
    w_torque_limit = -2.0  # Chunk 1 torque limit weight
    tau_soft = 1.95  # Chunk 1 threshold (N.m)
    w_fewer_3 = -0.5
    w_fewer_4 = -0.05
    w_air_time = 0.25
    w_stand_posture = -0.5
    w_stand_grounded = 1.0

    print("=" * 80)
    print("TASK 2: REWARD ECONOMICS OF CHUNK-1 CONFIGURATION")
    print("=" * 80)

    # 1. Share of training envs with |v_xy| < 0.10 m/s
    # In Chunk 1:
    # - 10% standing envs (cmd = 0)
    # - 30% yaw-dominant envs (vx, vy in [-0.05, 0.05] -> max |v| = 0.0707 < 0.10)
    # - 60% standard envs (vx in [-0.25, 0.30], vy in [-0.20, 0.20])
    N_samples = 1000000
    r_stand = np.random.rand(N_samples) < 0.10
    r_yaw = (~r_stand) & (np.random.rand(N_samples) < 0.30 / 0.90)  # 30% of total
    r_gen = (~r_stand) & (~r_yaw)  # 60% of total

    vx = np.zeros(N_samples)
    vy = np.zeros(N_samples)

    # Yaw envs:
    vx[r_yaw] = np.random.uniform(-0.05, 0.05, size=r_yaw.sum())
    vy[r_yaw] = np.random.uniform(-0.05, 0.05, size=r_yaw.sum())

    # General envs:
    vx[r_gen] = np.random.uniform(-0.25, 0.30, size=r_gen.sum())
    vy[r_gen] = np.random.uniform(-0.20, 0.20, size=r_gen.sum())

    v_mag = np.sqrt(vx**2 + vy**2)
    share_under_01 = (v_mag < 0.10).mean() * 100.0

    print(f"\n1. SHARE OF TRAINING ENVIRONMENTS WITH |v_xy| < 0.10 m/s:")
    print(f"   - Standing environments      : 10.0%")
    print(f"   - Yaw-dominant environments  : 30.0% (max |v_xy| = 0.071 m/s < 0.10 m/s)")
    print(f"   - General envs with |v| < 0.1: {(v_mag[r_gen] < 0.10).mean() * 60.0:.1f}%")
    print(f"   --------------------------------------------------")
    print(f"   TOTAL ENVS WITH |v_xy| < 0.1 : {share_under_01:.2f}% (almost half the batch!)")

    # 2. Static torques per leg for 3, 4, 5, 6 feet down
    print(f"\n2. SIMULATOR STATIC TORQUES (5.00 kg total mass, W = 49.05 N):")
    for n_feet in [6, 5, 4, 3]:
        tau_leg = get_static_tau(n_feet)
        print(f"   - {n_feet} feet down: F_z = {total_weight_N/n_feet:.2f} N/foot -> tau_thigh = {tau_leg:.3f} N.m (tau_soft = 1.95 N.m)")

    # 3. Per-second reward breakdown
    speeds = [0.10, 0.15, 0.25]
    feet_counts = [6, 5, 4, 3]

    print(f"\n3. PER-SECOND REWARD COMPARISON (policy rate = 50 Hz):")
    print(f"   Note: Dynamic dynamic walking adds ~15-20% peak torque during stance transitions.")

    # Condition: Standing still (cmd = 0)
    # 6 feet down, perfect height, zero velocity
    r_stand_per_step = (
        w_lin * 1.0  # exp(0) = 1
        + w_ang * 1.0  # exp(0) = 1
        + w_height * 1.0  # at target height
        + w_stand_grounded * 1.0  # all feet down
        + 6 * w_torques_l2 * (get_static_tau(6) ** 2)
    )
    r_stand_per_sec = r_stand_per_step * policy_hz
    print(f"\n   [STANDING STILL (cmd = 0, 6 feet down)]")
    print(f"   Total Reward: {r_stand_per_sec:+.2f} / s")

    # Condition: Commanded to walk at v_cmd, but robot stands still
    print(f"\n   [ROBOT STANDS STILL WHEN COMMANDED TO WALK (CREEPER STRATEGY)]")
    for v_cmd in speeds:
        lin_err = math.exp(- (v_cmd ** 2) / (std_lin ** 2))
        r_creep_step = (
            w_lin * lin_err
            + w_ang * 1.0
            + w_height * 1.0
            + 6 * w_torques_l2 * (get_static_tau(6) ** 2)
            # no torque limit penalty because tau = 0.94 N.m << 1.95 N.m!
        )
        r_creep_sec = r_creep_step * policy_hz
        print(f"   - Command {v_cmd:.2f} m/s -> Creeper Reward: {r_creep_sec:+.2f} / s")

    # Condition: Ideal walking at v_cmd with N feet down
    print(f"\n   [IDEAL WALKING AT COMMANDED SPEED]")
    table_rows = []
    for v_cmd in speeds:
        for n_feet in feet_counts:
            # Base tracking rewards
            r_lin_step = w_lin * 1.0  # perfectly tracking cmd
            r_ang_step = w_ang * 1.0
            r_h_step = w_height * 1.0

            # Foot contact penalties
            r_contact_penalty = 0.0
            if n_feet < 3:
                r_contact_penalty += w_fewer_3
            elif n_feet < 4:
                r_contact_penalty += w_fewer_4  # -0.05

            # Air time reward (for swing legs)
            n_swing = 6 - n_feet
            r_air = w_air_time * (n_swing / 6.0)

            # Torques: stance legs bear weight
            # With dynamic multiplier ~1.15 for leg transfer
            tau_stance = get_static_tau(n_feet) * 1.15
            tau_swing = 0.15  # minimal swing torque

            r_torq_l2 = n_feet * w_torques_l2 * (tau_stance ** 2) + n_swing * w_torques_l2 * (tau_swing ** 2)

            # Torque limit penalty: Chunk 1 used weight -2.0, threshold 1.95 N.m!
            excess = max(0.0, tau_stance - tau_soft)
            r_torq_lim = n_feet * w_torque_limit * (excess ** 2)

            r_walk_step = r_lin_step + r_ang_step + r_h_step + r_contact_penalty + r_air + r_torq_l2 + r_torq_lim
            r_walk_sec = r_walk_step * policy_hz
            diff_from_creep = r_walk_sec - (r_creep_sec)

            table_rows.append({
                "v_cmd": v_cmd,
                "n_feet": n_feet,
                "tau_stance": tau_stance,
                "r_walk_sec": r_walk_sec,
                "torq_lim_sec": (r_torq_lim * policy_hz),
                "diff_creep": diff_from_creep,
            })

    print(f"\n   | v_cmd | Feet Down | Stance Tau | Torque Lim Pen | Total Walk / s | Net Advantage vs Creep |")
    print(f"   |---|---|---|---|---|---|")
    for row in table_rows:
        print(f"   | {row['v_cmd']:.2f} m/s | {row['n_feet']} feet | {row['tau_stance']:.3f} N.m | {row['torq_lim_sec']:.2f} / s | {row['r_walk_sec']:+.2f} / s | {row['diff_creep']:+.2f} / s |")


if __name__ == "__main__":
    compute_reward_economics()
