# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Single source of truth for all Z.O.E. Hexapod deployable task hyper-parameters."""

from dataclasses import dataclass, field
import math


@dataclass
class ZoeCommandParams:
    lin_vel_x_range: tuple[float, float] = (-0.25, 0.30)  # m/s
    lin_vel_y_range: tuple[float, float] = (-0.20, 0.20)  # m/s
    ang_vel_z_range: tuple[float, float] = (-0.60, 0.60)  # rad/s
    rel_standing_envs: float = 0.10
    rel_heading_envs: float = 0.0
    heading_command: bool = False
    resampling_time_range: tuple[float, float] = (3.0, 5.0)  # seconds


@dataclass
class ZoeObsNoiseParams:
    gyro_noise_std: float = 0.2
    gravity_noise_std: float = 0.05
    joint_pos_noise_std: float = 0.01
    joint_vel_noise_std: float = 1.5


@dataclass
class ZoeObsScaleParams:
    base_ang_vel_scale: float = 0.25
    base_ang_vel_clip: tuple[float, float] = (-5.0, 5.0)
    projected_gravity_scale: float = 1.0
    joint_pos_rel_scale: float = 1.0
    joint_pos_rel_clip: tuple[float, float] = (-5.0, 5.0)
    joint_vel_scale: float = 0.05
    joint_vel_clip: tuple[float, float] = (-5.0, 5.0)
    last_action_scale: float = 1.0


@dataclass
class ZoeEventParams:
    static_friction_range: tuple[float, float] = (0.4, 1.2)
    dynamic_friction_range: tuple[float, float] = (0.3, 1.0)
    base_payload_mass_range: tuple[float, float] = (-0.2, 0.5)  # kg
    body_mass_scale_range: tuple[float, float] = (0.85, 1.15)
    base_com_offset_range: tuple[float, float] = (-0.015, 0.015)  # m
    actuator_kp_scale_range: tuple[float, float] = (0.8, 1.2)
    actuator_kd_scale_range: tuple[float, float] = (0.8, 1.2)
    reset_joint_offset_range: tuple[float, float] = (-0.05, 0.05)  # rad
    calibration_zero_offset_deg: float = 3.0  # max calibration error +-3 deg per joint
    max_action_delay_steps: int = 2  # 0-2 policy steps (0-40 ms at 50Hz)
    push_interval_s_range: tuple[float, float] = (8.0, 15.0)
    push_velocity_x_y_range: tuple[float, float] = (-0.3, 0.3)  # m/s


@dataclass
class ZoeRewardParams:
    track_lin_vel_xy_exp: float = 2.0
    track_ang_vel_z_exp: float = 0.5
    termination_penalty: float = -200.0
    dof_torques_l2: float = -2e-4
    action_rate_l2: float = -0.05
    dof_acc_l2: float = -1e-7
    dof_pos_limits: float = -1.0
    feet_air_time: float = 0.25
    feet_air_time_threshold: float = 0.15  # seconds
    fewer_than_3_feet_contact: float = -0.5
    undesired_contacts: float = -1.0
    base_height_l2: float = -5.0
    flat_orientation_l2_flat: float = -0.5
    flat_orientation_l2_terrain: float = 0.0
    lin_vel_z_l2_flat: float = -2.0
    lin_vel_z_l2_terrain: float = -0.5


@dataclass
class ZoeParams:
    STANDING_BASE_HEIGHT: float = 0.23
    commands: ZoeCommandParams = field(default_factory=ZoeCommandParams)
    obs_noise: ZoeObsNoiseParams = field(default_factory=ZoeObsNoiseParams)
    obs_scale: ZoeObsScaleParams = field(default_factory=ZoeObsScaleParams)
    events: ZoeEventParams = field(default_factory=ZoeEventParams)
    rewards: ZoeRewardParams = field(default_factory=ZoeRewardParams)


# Global instance
ZOE_PARAMS = ZoeParams()
