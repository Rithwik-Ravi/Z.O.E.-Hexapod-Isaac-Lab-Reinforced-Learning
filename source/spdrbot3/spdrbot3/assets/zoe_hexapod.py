# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Articulation and Actuator Configuration for Z.O.E. Hexapod."""

import math
import os
import yaml
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils

# USD path resolution order:
# 1. Environment variable ZOE_USD_PATH
# 2. Check candidate locations relative to current working directory or repo root
candidates = [
    os.environ.get("ZOE_USD_PATH"),
    os.path.abspath("hexapod/hexa/hexa.usd"),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../hexapod/hexa/hexa.usd")),
]

DEFAULT_USD_PATH = "hexapod/hexa/hexa.usd"
for cand in candidates:
    if cand and os.path.exists(cand):
        DEFAULT_USD_PATH = os.path.abspath(cand)
        break

# Load actuator model calibration file if available
ACTUATOR_MODEL_FILE = os.path.abspath("deploy/actuator_model.yaml")

# ST3215 Servo Defaults (UNCALIBRATED placeholders)
DEFAULT_STIFFNESS = 15.0
DEFAULT_DAMPING = 1.5
DEFAULT_EFFORT_LIMIT = 2.5   # N*m (STS3215 stall torque ~2.5 N.m)
DEFAULT_VELOCITY_LIMIT = 6.0 # rad/s (~57 rpm)

if os.path.exists(ACTUATOR_MODEL_FILE):
    try:
        with open(ACTUATOR_MODEL_FILE, "r") as f:
            actuator_data = yaml.safe_load(f)
            DEFAULT_STIFFNESS = actuator_data.get("kp", DEFAULT_STIFFNESS)
            DEFAULT_DAMPING = actuator_data.get("kd", DEFAULT_DAMPING)
    except Exception as e:
        print(f"[ZOE_HEXAPOD] Warning: Failed to load actuator model: {e}")

# Standing stance angles in radians (USD zero stance convention: 0.0 rad plants 6/6 feet at 0.23m height)
DEFAULT_COXA_POS = 0.0   # USD zero angle
DEFAULT_THIGH_POS = 0.0  # USD zero angle
DEFAULT_FOOT_POS = 0.0   # USD zero angle

ZOE_HEXAPOD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=DEFAULT_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.18),  # Spawn height 2cm above standing ground contact (settled height: 0.158m)
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            ".*base_link_v1_Revolute.*": DEFAULT_COXA_POS,  # Coxa joints (legs 1-6)
            ".*NCCUP.*_Revolute.*": DEFAULT_THIGH_POS,       # Thigh joints (legs 1-6)
            ".*NTHIGH.*_Revolute.*": DEFAULT_FOOT_POS,       # Foot joints (legs 1-6)
        },
        joint_vel={
            ".*Revolute.*": 0.0,
        },
    ),
    actuators={
        "coxa_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*base_link_v1_Revolute.*"],
            effort_limit_sim=DEFAULT_EFFORT_LIMIT,
            velocity_limit_sim=DEFAULT_VELOCITY_LIMIT,
            stiffness=DEFAULT_STIFFNESS,
            damping=DEFAULT_DAMPING,
        ),
        "thigh_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*NCCUP.*_Revolute.*"],
            effort_limit_sim=DEFAULT_EFFORT_LIMIT,
            velocity_limit_sim=DEFAULT_VELOCITY_LIMIT,
            stiffness=DEFAULT_STIFFNESS,
            damping=DEFAULT_DAMPING,
        ),
        "foot_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*NTHIGH.*_Revolute.*"],
            effort_limit_sim=DEFAULT_EFFORT_LIMIT,
            velocity_limit_sim=DEFAULT_VELOCITY_LIMIT,
            stiffness=DEFAULT_STIFFNESS,
            damping=DEFAULT_DAMPING,
        ),
    },
    soft_joint_pos_limit_factor=0.95,  # Hardware limit clamp factor (A10 fix)
)
