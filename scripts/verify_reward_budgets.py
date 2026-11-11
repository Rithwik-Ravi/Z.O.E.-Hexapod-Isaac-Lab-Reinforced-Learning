# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Verifies reward term per-second budgets for ideal walking vs standing still under a walk command."""

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../source/spdrbot3")))
from spdrbot3.tasks.manager_based.zoe_deploy.params import ZOE_PARAMS


def print_reward_comparison():
    hz = 50.0  # Policy rate at dt=0.005, decimation=4
    cmd_vx = 0.20  # m/s commanded forward velocity
    cmd_wz = 0.00  # rad/s

    print("=" * 80)
    print(f"REWARD BUDGET AUDIT: IDEAL WALKING VS STANDING STILL (Cmd vx = {cmd_vx} m/s)")
    print("=" * 80)
    print(f"{'Reward Term':<32} | {'Ideal Walking (/s)':<20} | {'Standing Still (/s)':<20}")
    print("-" * 80)

    # 1. track_lin_vel_xy_exp
    # Ideal walking: v_actual = 0.20 -> error = 0
    rew_walk_lin = ZOE_PARAMS.rewards.track_lin_vel_xy_exp * math.exp(-0.0) * hz
    # Standing still: v_actual = 0.0 -> error = 0.20
    std_lin = ZOE_PARAMS.rewards.track_lin_vel_std  # 0.13
    rew_stand_lin = ZOE_PARAMS.rewards.track_lin_vel_xy_exp * math.exp(-((cmd_vx / std_lin) ** 2)) * hz
    print(f"{'track_lin_vel_xy_exp':<32} | {rew_walk_lin:+18.2f}  | {rew_stand_lin:+18.2f}")

    # 2. track_ang_vel_z_exp
    rew_walk_ang = ZOE_PARAMS.rewards.track_ang_vel_z_exp * math.exp(-0.0) * hz
    rew_stand_ang = ZOE_PARAMS.rewards.track_ang_vel_z_exp * math.exp(-0.0) * hz
    print(f"{'track_ang_vel_z_exp':<32} | {rew_walk_ang:+18.2f}  | {rew_stand_ang:+18.2f}")

    # 3. base_height_shaped (at target height ZOE_PARAMS.TARGET_STANDING_HEIGHT)
    rew_walk_height = 1.0 * math.exp(-0.0) * hz
    rew_stand_height = 1.0 * math.exp(-0.0) * hz
    print(f"{'base_height_shaped':<32} | {rew_walk_height:+18.2f}  | {rew_stand_height:+18.2f}")

    # 4. feet_air_time (walking: 3 feet in swing with air time > 0.15s, standing: 0 air time)
    # Walking: ~3 feet lifting with air time 0.35s, landing ~1.5 Hz frequency
    rew_walk_air = ZOE_PARAMS.rewards.feet_air_time * (3 * (0.35 - 0.15)) * 1.5
    rew_stand_air = 0.0  # Gated by cmd_norm and first_contact
    print(f"{'feet_air_time':<32} | {rew_walk_air:+18.2f}  | {rew_stand_air:+18.2f}")

    # 5. foot_tip_clearance (walking: 3 swing feet at 6 cm, standing: 0 swing feet)
    rew_walk_clear = ZOE_PARAMS.rewards.foot_tip_clearance * 3.0 * math.exp(-0.0) * hz
    rew_stand_clear = 0.0  # 0 feet in air / swing
    print(f"{'foot_tip_clearance':<32} | {rew_walk_clear:+18.2f}  | {rew_stand_clear:+18.2f}")

    # 6. foot_tip_slip (walking: slight stance slip ~0.02 m/s on 3 grounded feet, standing: 0 slip)
    rew_walk_slip = ZOE_PARAMS.rewards.foot_tip_slip * (3 * (0.02 ** 2)) * hz
    rew_stand_slip = 0.0
    print(f"{'foot_tip_slip':<32} | {rew_walk_slip:+18.2f}  | {rew_stand_slip:+18.2f}")

    # 7. dof_torques_l2 (walking: torques ~1.2 N.m, standing: torques ~0.8 N.m across 18 joints)
    rew_walk_tau = ZOE_PARAMS.rewards.dof_torques_l2 * (18 * (1.2 ** 2)) * hz
    rew_stand_tau = ZOE_PARAMS.rewards.dof_torques_l2 * (18 * (0.8 ** 2)) * hz
    print(f"{'dof_torques_l2':<32} | {rew_walk_tau:+18.2f}  | {rew_stand_tau:+18.2f}")

    # 8. action_rate_l2 (walking: action rate ~0.04 rad/step, standing: 0.0)
    rew_walk_act = ZOE_PARAMS.rewards.action_rate_l2 * (18 * (0.04 ** 2)) * hz
    rew_stand_act = 0.0
    print(f"{'action_rate_l2':<32} | {rew_walk_act:+18.2f}  | {rew_stand_act:+18.2f}")

    # 9. fewer_than_4_feet_contact (walking: 3 feet grounded -> (4-3)^2 = 1.0, standing: 6 feet grounded -> 0.0)
    rew_walk_f4 = ZOE_PARAMS.rewards.fewer_than_4_feet_contact * (1.0 ** 2) * hz
    rew_stand_f4 = 0.0
    print(f"{'fewer_than_4_feet_contact':<32} | {rew_walk_f4:+18.2f}  | {rew_stand_f4:+18.2f}")

    # 10. torque_limit (walking torques ~1.2 N.m < 2.10 N.m soft limit -> 0.0, standing ~0.8 N.m -> 0.0)
    rew_walk_tlim = 0.0
    rew_stand_tlim = 0.0
    print(f"{'torque_limit':<32} | {rew_walk_tlim:+18.2f}  | {rew_stand_tlim:+18.2f}")

    # 11. stand_still_posture / stand_still_feet_grounded (active only when |cmd_xy| < 0.05 AND |cmd_yaw| < 0.05)
    rew_walk_stand = 0.0
    rew_stand_stand = 0.0  # Not awarded because |cmd_xy| = 0.20 m/s >= 0.05
    print(f"{'stand_still_rewards':<32} | {rew_walk_stand:+18.2f}  | {rew_stand_stand:+18.2f}")

    # TOTALS
    total_walk = (
        rew_walk_lin + rew_walk_ang + rew_walk_height + rew_walk_air +
        rew_walk_clear + rew_walk_slip + rew_walk_tau + rew_walk_act +
        rew_walk_f4 + rew_walk_tlim
    )
    total_stand = (
        rew_stand_lin + rew_stand_ang + rew_stand_height + rew_stand_air +
        rew_stand_clear + rew_stand_slip + rew_stand_tau + rew_stand_act +
        rew_stand_f4 + rew_stand_tlim
    )

    print("-" * 80)
    print(f"{'TOTAL REWARD PER SECOND':<32} | {total_walk:+18.2f}  | {total_stand:+18.2f}")
    print("=" * 80)

    delta = total_walk - total_stand
    ratio = total_walk / total_stand if total_stand > 0 else float("inf")
    print(f"Walking Advantage: {delta:+.2f} /s ({ratio:.2f}x reward of standing still)")

    assert total_walk > total_stand + 50.0, "Ideal walking must come out clearly ahead by at least +50/s!"
    print("ASSERTION PASSED: Walking comes out clearly ahead of standing still! ✅\n")


if __name__ == "__main__":
    print_reward_comparison()
