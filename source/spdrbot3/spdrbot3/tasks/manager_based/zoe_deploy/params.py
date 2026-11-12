# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Single source of truth for all Z.O.E. Hexapod deployable task hyper-parameters.

Reads geometric and mass properties derived by `scripts/audit_asset.py` from
`deploy/asset_report.json` and actuator specs from `deploy/actuator_model.yaml`.
"""

from dataclasses import dataclass, field
import json
import math
import os
import yaml

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../.."))
ASSET_REPORT_PATH = os.path.join(_REPO_ROOT, "deploy/asset_report.json")
ACTUATOR_MODEL_PATH = os.path.join(_REPO_ROOT, "deploy/actuator_model.yaml")


def _load_asset_report() -> dict:
    for cand in [os.path.abspath("deploy/asset_report.json"), ASSET_REPORT_PATH]:
        if os.path.exists(cand):
            with open(cand, "r") as f:
                return json.load(f)
    return {}


def _load_actuator_model() -> dict:
    for cand in [os.path.abspath("deploy/actuator_model.yaml"), ACTUATOR_MODEL_PATH]:
        if os.path.exists(cand):
            with open(cand, "r") as f:
                return yaml.safe_load(f) or {}
    return {}


_ASSET_DATA = _load_asset_report()
_ACTUATOR_DATA = _load_actuator_model()
_KIN_DATA = _ASSET_DATA.get("kinematics", {})

# Derived geometric & mass constants from deploy/asset_report.json
DERIVED_BASE_BODY_NAME: str = _ASSET_DATA.get("base_body_name", "base_link")
DERIVED_TOTAL_ASSET_MASS_KG: float = float(_ASSET_DATA.get("total_mass_kg", 5.0))
DERIVED_BASE_LINK_MASS_KG: float = float(_ASSET_DATA.get("base_mass_kg", 3.3365))
ASSUMED_TOTAL_MASS_KG: float = float(_ASSET_DATA.get("assumed_real_mass_kg", 5.0))

DERIVED_KINEMATIC_STANDING_HEIGHT: float = float(_KIN_DATA.get("kinematic_standing_height_m", 0.1476))
DERIVED_STANDING_BASE_HEIGHT: float = float(_KIN_DATA.get("settled_standing_height_m", 0.1417))
DERIVED_ALL_ZERO_SETTLED_HEIGHT: float = float(_KIN_DATA.get("all_zero_settled_height_m", 0.0032))

# Frozen gate and reward height ratios relative to measured settled standing height
FROZEN_GATE_HEIGHT_RATIO: float = 0.90
FROZEN_TARGET_HEIGHT_RATIO: float = 0.95

DERIVED_GATE_MIN_HEIGHT: float = round(FROZEN_GATE_HEIGHT_RATIO * DERIVED_STANDING_BASE_HEIGHT, 4)
DERIVED_TARGET_STANDING_HEIGHT: float = round(FROZEN_TARGET_HEIGHT_RATIO * DERIVED_STANDING_BASE_HEIGHT, 4)
DERIVED_SPAWN_Z_RANGE: tuple[float, float] = (
    round(DERIVED_KINEMATIC_STANDING_HEIGHT + 0.012, 3),
    round(DERIVED_KINEMATIC_STANDING_HEIGHT + 0.032, 3),
)

_DEFAULT_POS_DICT = _KIN_DATA.get("default_joint_pos_rad", {})
DERIVED_DEFAULT_COXA_POS_RAD: float = float(_DEFAULT_POS_DICT.get("coxa", 0.0))
DERIVED_DEFAULT_THIGH_POS_RAD: float = float(_DEFAULT_POS_DICT.get("thigh", 0.0))
DERIVED_DEFAULT_FOOT_POS_RAD: float = float(_DEFAULT_POS_DICT.get("foot", -math.pi / 2.0))

_UPRIGHT_QUAT = _KIN_DATA.get("upright_root_quat_wxyz", [0.0, 1.0, 0.0, 0.0])
DERIVED_UPRIGHT_ROOT_QUAT_WXYZ: tuple[float, float, float, float] = tuple(float(x) for x in _UPRIGHT_QUAT)
DERIVED_UPRIGHT_PROJ_GRAV_Z_SIGN: float = float(_KIN_DATA.get("upright_projected_gravity_z_sign", 1.0))

# Per-leg 3D local foot tip vectors in NFOOT1..6 body frames (derived from STL meshes via audit_asset.py)
DEFAULT_FOOT_TIP_LOCAL_VECTORS: dict[str, list[float]] = {
    "NFOOT1": [0.06777, 0.0, -0.15162],
    "NFOOT2": [0.16520, 0.0, -0.01712],
    "NFOOT3": [0.09742, 0.0, 0.13450],
    "NFOOT4": [-0.06777, 0.0, 0.15162],
    "NFOOT5": [-0.16520, 0.0, 0.01712],
    "NFOOT6": [-0.09742, 0.0, -0.13450],
}
DERIVED_FOOT_TIP_LOCAL_VECTORS: dict[str, list[float]] = _KIN_DATA.get(
    "foot_tip_local_vectors_m", DEFAULT_FOOT_TIP_LOCAL_VECTORS
)

DERIVED_JOINT_INFO: list[dict] = _ASSET_DATA.get("joint_info", [])
DERIVED_JOINT_INFO_BY_SIM_NAME: dict[str, dict] = {j["sim_name"]: j for j in DERIVED_JOINT_INFO if "sim_name" in j}

# Derived/Loaded Actuator Constants from deploy/actuator_model.yaml (Waveshare ST3215 12V)
ACTUATOR_STALL_TORQUE_12V_NM: float = float(_ACTUATOR_DATA.get("stall_torque_12v_nm", 2.942))
ACTUATOR_EFFORT_MARGIN_RATIO: float = float(_ACTUATOR_DATA.get("effort_limit_margin_ratio", 0.85))
ACTUATOR_EFFORT_LIMIT_NM: float = float(
    _ACTUATOR_DATA.get("effort_limit_sim_nm", round(ACTUATOR_STALL_TORQUE_12V_NM * ACTUATOR_EFFORT_MARGIN_RATIO, 2))
)
ACTUATOR_TORQUE_SAT_THRESHOLD_NM: float = float(
    _ACTUATOR_DATA.get("torque_saturation_threshold_nm", round(0.98 * ACTUATOR_EFFORT_LIMIT_NM, 2))
)
ACTUATOR_NO_LOAD_SPEED_RAD_S: float = float(_ACTUATOR_DATA.get("no_load_speed_12v_rad_s", 4.71))
ACTUATOR_KP: float = float(_ACTUATOR_DATA.get("kp", 16.0))
ACTUATOR_KD: float = float(_ACTUATOR_DATA.get("kd", 1.6))

# Hardware joint position limits (radians): coxa [-45..+135 deg], thigh/foot [-90..+90 deg]
JOINT_LIMITS_RAD: dict[str, tuple[float, float]] = {
    "coxa": (-math.pi / 4.0, 3.0 * math.pi / 4.0),
    "thigh": (-math.pi / 2.0, math.pi / 2.0),
    "foot": (-math.pi, 0.0),
}


def get_foot_tip_offsets_tensor(body_names: list[str], device, dtype):
    """Return a [len(body_names), 3] tensor of local 3D tip vectors for the given foot body names."""
    import torch

    vecs = []
    for name in body_names:
        vec = DERIVED_FOOT_TIP_LOCAL_VECTORS.get(name, [0.0, 0.0, -0.1476])
        vecs.append(vec)
    return torch.tensor(vecs, device=device, dtype=dtype)


@dataclass
class ZoeCommandParams:
    lin_vel_x_range: tuple[float, float] = (-0.25, 0.30)  # m/s
    lin_vel_y_range: tuple[float, float] = (-0.20, 0.20)  # m/s
    ang_vel_z_range: tuple[float, float] = (-0.60, 0.60)  # rad/s
    rel_standing_envs: float = 0.10
    rel_heading_envs: float = 0.0
    rel_yaw_envs: float = 0.30  # >= 25% yaw-dominant envs (|wz| in [0.3, 0.6], small vx, vy)
    yaw_dominant_wz_range: tuple[float, float] = (0.30, 0.60)  # rad/s magnitude
    yaw_dominant_lin_vel_range: tuple[float, float] = (-0.05, 0.05)  # m/s
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
    base_mass_scale_range: tuple[float, float] = (0.80, 1.10)  # Multiplicative scale 80-110% (covers 4.0-5.5 kg around 5.0 kg, Task 7)
    body_mass_scale_range: tuple[float, float] = (0.80, 1.10)  # Multiplicative scale 80-110% (Task 7)
    base_com_offset_range: tuple[float, float] = (-0.015, 0.015)  # m
    actuator_kp_scale_range: tuple[float, float] = (0.70, 1.30)  # +-30%
    actuator_kd_scale_range: tuple[float, float] = (0.70, 1.30)  # +-30%
    effort_limit_scale_range: tuple[float, float] = (0.85, 1.15)  # +-15%
    reset_joint_offset_range: tuple[float, float] = (-0.05, 0.05)  # rad
    calibration_zero_offset_deg: float = 3.0  # max calibration error +-3 deg per joint
    max_action_delay_steps: int = 2  # 0-2 policy steps (0-40 ms at 50Hz)
    push_interval_s_range: tuple[float, float] = (8.0, 15.0)
    push_velocity_x_y_range: tuple[float, float] = (-0.3, 0.3)  # m/s


@dataclass
class ZoeRewardParams:
    track_lin_vel_xy_exp: float = 2.0
    track_ang_vel_z_exp: float = 1.0  # Winning Stage 1 config (Probe A: 16/16 PASS)
    track_lin_vel_std: float = 0.13  # Width ~0.13 m/s so standing still earns <= 25% reward at 0.15 m/s cmd
    track_ang_vel_std: float = 0.25  # Winning Stage 1 config (Probe A: 16/16 PASS)
    termination_penalty: float = -200.0
    dof_torques_l2: float = -5e-3
    action_rate_l2: float = -0.05
    dof_acc_l2: float = -1e-7
    dof_pos_limits: float = -1.0
    feet_air_time: float = 0.25
    feet_air_time_threshold: float = 0.15  # seconds
    fewer_than_3_feet_contact: float = -0.5  # Probe A (P1 baseline weight)
    fewer_than_4_feet_contact: float = -0.05
    undesired_contacts: float = -1.0
    base_height_l2: float = -50.0
    flat_orientation_l2_flat: float = -0.5
    flat_orientation_l2_terrain: float = 0.0
    lin_vel_z_l2_flat: float = -2.0
    lin_vel_z_l2_terrain: float = -0.5
    foot_tip_slip: float = -0.1
    foot_tip_clearance: float = 0.1
    stand_still_posture: float = -0.5
    stand_still_feet_grounded: float = 1.0  # Reward fraction of grounded feet when |cmd_xy| < 0.05 AND |cmd_yaw| < 0.05
    joint_vel_limit: float = -0.01
    torque_limit: float = -0.5  # Restored from -1.0 (Probe P4 throttled to 0.094 m/s); retains penalty_scale_ramp + 2.10 N.m threshold
    torque_soft_limit_nm: float = 2.10  # 2.10 N.m soft threshold (84% of 2.50 N.m effort limit)


@dataclass
class ZoeParams:
    BASE_BODY_NAME: str = DERIVED_BASE_BODY_NAME
    TOTAL_ASSET_MASS_KG: float = DERIVED_TOTAL_ASSET_MASS_KG
    BASE_LINK_MASS_KG: float = DERIVED_BASE_LINK_MASS_KG
    ASSUMED_TOTAL_MASS_KG: float = ASSUMED_TOTAL_MASS_KG
    KINEMATIC_STANDING_HEIGHT: float = DERIVED_KINEMATIC_STANDING_HEIGHT
    STANDING_BASE_HEIGHT: float = DERIVED_STANDING_BASE_HEIGHT
    ALL_ZERO_SETTLED_HEIGHT: float = DERIVED_ALL_ZERO_SETTLED_HEIGHT
    GATE_HEIGHT_RATIO: float = FROZEN_GATE_HEIGHT_RATIO
    TARGET_HEIGHT_RATIO: float = FROZEN_TARGET_HEIGHT_RATIO
    GATE_MIN_HEIGHT: float = DERIVED_GATE_MIN_HEIGHT
    TARGET_STANDING_HEIGHT: float = DERIVED_TARGET_STANDING_HEIGHT
    SPAWN_Z_RANGE: tuple[float, float] = DERIVED_SPAWN_Z_RANGE
    UPRIGHT_ROOT_QUAT_WXYZ: tuple[float, float, float, float] = DERIVED_UPRIGHT_ROOT_QUAT_WXYZ
    UPRIGHT_PROJ_GRAV_Z_SIGN: float = DERIVED_UPRIGHT_PROJ_GRAV_Z_SIGN
    DEFAULT_COXA_POS_RAD: float = DERIVED_DEFAULT_COXA_POS_RAD
    DEFAULT_THIGH_POS_RAD: float = DERIVED_DEFAULT_THIGH_POS_RAD
    DEFAULT_FOOT_POS_RAD: float = DERIVED_DEFAULT_FOOT_POS_RAD
    STALL_TORQUE_12V_NM: float = ACTUATOR_STALL_TORQUE_12V_NM
    EFFORT_LIMIT_MARGIN_RATIO: float = ACTUATOR_EFFORT_MARGIN_RATIO
    EFFORT_LIMIT_NM: float = ACTUATOR_EFFORT_LIMIT_NM
    TORQUE_SAT_THRESHOLD_NM: float = ACTUATOR_TORQUE_SAT_THRESHOLD_NM
    NO_LOAD_SPEED_RAD_S: float = ACTUATOR_NO_LOAD_SPEED_RAD_S
    ACTUATOR_KP: float = ACTUATOR_KP
    ACTUATOR_KD: float = ACTUATOR_KD
    foot_tip_local_vectors: dict[str, list[float]] = field(
        default_factory=lambda: dict(DERIVED_FOOT_TIP_LOCAL_VECTORS)
    )
    joint_info: list[dict] = field(default_factory=lambda: list(DERIVED_JOINT_INFO))
    joint_info_by_sim_name: dict[str, dict] = field(default_factory=lambda: dict(DERIVED_JOINT_INFO_BY_SIM_NAME))
    commands: ZoeCommandParams = field(default_factory=ZoeCommandParams)
    obs_noise: ZoeObsNoiseParams = field(default_factory=ZoeObsNoiseParams)
    obs_scale: ZoeObsScaleParams = field(default_factory=ZoeObsScaleParams)
    events: ZoeEventParams = field(default_factory=ZoeEventParams)
    rewards: ZoeRewardParams = field(default_factory=ZoeRewardParams)


# Global instance
ZOE_PARAMS = ZoeParams()
