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

# USD / URDF path resolution order:
# 1. Environment variable ZOE_USD_PATH
# 2. Check candidate locations relative to current working directory or repo root
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
candidates = [
    os.environ.get("ZOE_USD_PATH"),
    os.path.abspath("hexapod/hexa/hexa.usd"),
    os.path.join(_REPO_ROOT, "hexapod/hexa/hexa.usd"),
]

DEFAULT_USD_PATH = os.path.join(_REPO_ROOT, "hexapod/hexa/hexa.usd")
for cand in candidates:
    if cand and os.path.exists(cand):
        DEFAULT_USD_PATH = os.path.abspath(cand)
        break

urdf_candidates = [
    os.path.abspath("hexapod/hexa.urdf"),
    os.path.join(_REPO_ROOT, "hexapod/hexa.urdf"),
]
DEFAULT_URDF_PATH = os.path.join(_REPO_ROOT, "hexapod/hexa.urdf")
for cand in urdf_candidates:
    if cand and os.path.exists(cand):
        DEFAULT_URDF_PATH = os.path.abspath(cand)
        break


def _patch_isaaclab_urdf_converter():
    """Patch UrdfConverter._get_urdf_import_config for Isaac Sim 4.5 compatibility."""
    try:
        import inspect
        import textwrap
        import isaaclab.sim.converters.urdf_converter as urdf_conv_mod
        from isaaclab.sim.converters.urdf_converter import UrdfConverter

        if not getattr(UrdfConverter, "_zoe_patched", False):
            orig_init = UrdfConverter.__init__

            def _patched_init(self, cfg):
                try:
                    from isaacsim.core.utils.extensions import enable_extension
                    enable_extension("isaacsim.asset.importer.urdf")
                except Exception:
                    pass
                return orig_init(self, cfg)

            UrdfConverter.__init__ = _patched_init

            src = textwrap.dedent(inspect.getsource(UrdfConverter._get_urdf_import_config))
            if "import_config.set_merge_fixed_ignore_inertia(" in src:
                src = src.replace(
                    "import_config.set_merge_fixed_ignore_inertia(",
                    "getattr(import_config, 'set_merge_fixed_ignore_inertia', lambda *a, **kw: None)(",
                )
                ns = dict(urdf_conv_mod.__dict__)
                exec("from __future__ import annotations\n" + src, ns)
                UrdfConverter._get_urdf_import_config = ns["_get_urdf_import_config"]

            UrdfConverter._zoe_patched = True
    except Exception as e:
        print(f"[ZOE_HEXAPOD] Warning: UrdfConverter patch skipped: {e}")


def _build_zoe_spawn_cfg():
    _patch_isaaclab_urdf_converter()
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        rigid_body_enabled=True,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
        enable_gyroscopic_forces=True,
        disable_gravity=False,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
        sleep_threshold=0.005,
        stabilization_threshold=0.001,
    )
    if os.path.exists(DEFAULT_USD_PATH):
        return sim_utils.UsdFileCfg(
            usd_path=DEFAULT_USD_PATH,
            activate_contact_sensors=True,
            rigid_props=rigid_props,
            articulation_props=articulation_props,
        )
    return sim_utils.UrdfFileCfg(
        asset_path=DEFAULT_URDF_PATH,
        usd_dir=os.path.dirname(DEFAULT_USD_PATH),
        usd_file_name=os.path.basename(DEFAULT_USD_PATH),
        fix_base=False,
        merge_fixed_joints=True,
        force_usd_conversion=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
        activate_contact_sensors=True,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    )

# Load actuator model calibration file if available
ACTUATOR_MODEL_FILE = os.path.join(_REPO_ROOT, "deploy/actuator_model.yaml")
if not os.path.exists(ACTUATOR_MODEL_FILE):
    ACTUATOR_MODEL_FILE = os.path.abspath("deploy/actuator_model.yaml")

# Waveshare ST3215 Servo Defaults (12V variant, 3S LiPo direct: 30 kg.cm = 2.942 N.m stall, 85% effort limit = 2.50 N.m)
DEFAULT_STIFFNESS = 16.0
DEFAULT_DAMPING = 1.6
DEFAULT_EFFORT_LIMIT = 2.50    # N*m (85% of 2.942 N*m 12V stall torque for thermal/safety margin)
DEFAULT_VELOCITY_LIMIT = 4.71  # rad/s (45 RPM / 0.222 s/60deg no-load speed at 12V)

if os.path.exists(ACTUATOR_MODEL_FILE):
    try:
        with open(ACTUATOR_MODEL_FILE, "r") as f:
            actuator_data = yaml.safe_load(f) or {}
            DEFAULT_STIFFNESS = float(actuator_data.get("kp", DEFAULT_STIFFNESS))
            DEFAULT_DAMPING = float(actuator_data.get("kd", DEFAULT_DAMPING))
            DEFAULT_EFFORT_LIMIT = float(actuator_data.get("effort_limit_sim_nm", DEFAULT_EFFORT_LIMIT))
            DEFAULT_VELOCITY_LIMIT = float(actuator_data.get("no_load_speed_12v_rad_s", DEFAULT_VELOCITY_LIMIT))
    except Exception as e:
        print(f"[ZOE_HEXAPOD] Warning: Failed to load actuator model: {e}")

# Standing stance angles in radians (in hexa.urdf with 180 deg roll so -Z is up, foot=-pi/2 points all 6 feet down)
DEFAULT_COXA_POS = 0.0           # Neutral coxa angle
DEFAULT_THIGH_POS = 0.0          # Horizontal thigh angle
DEFAULT_FOOT_POS = -math.pi / 2  # 90 deg downward foot angle when upright (plants 6/6 feet)

ZOE_HEXAPOD_CFG = ArticulationCfg(
    spawn=_build_zoe_spawn_cfg(),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.17),  # Spawn height ~2-3 cm above 0.1427m settled standing height
        rot=(0.0, 1.0, 0.0, 0.0),  # (w, x, y, z): 180 deg roll about X so base_link is right-side up (-Z up)
        joint_pos={
            ".*base_link.*Revolute.*": DEFAULT_COXA_POS,  # Coxa joints (legs 1-6)
            ".*NCCUP.*Revolute.*": DEFAULT_THIGH_POS,     # Thigh joints (legs 1-6)
            ".*NTHIGH.*Revolute.*": DEFAULT_FOOT_POS,     # Foot joints (legs 1-6)
        },
        joint_vel={
            ".*Revolute.*": 0.0,
        },
    ),
    actuators={
        "coxa_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*base_link.*Revolute.*"],
            effort_limit_sim=DEFAULT_EFFORT_LIMIT,
            velocity_limit_sim=DEFAULT_VELOCITY_LIMIT,
            stiffness=DEFAULT_STIFFNESS,
            damping=DEFAULT_DAMPING,
        ),
        "thigh_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*NCCUP.*Revolute.*"],
            effort_limit_sim=DEFAULT_EFFORT_LIMIT,
            velocity_limit_sim=DEFAULT_VELOCITY_LIMIT,
            stiffness=DEFAULT_STIFFNESS,
            damping=DEFAULT_DAMPING,
        ),
        "foot_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*NTHIGH.*Revolute.*"],
            effort_limit_sim=DEFAULT_EFFORT_LIMIT,
            velocity_limit_sim=DEFAULT_VELOCITY_LIMIT,
            stiffness=DEFAULT_STIFFNESS,
            damping=DEFAULT_DAMPING,
        ),
    },
    soft_joint_pos_limit_factor=0.95,  # Hardware limit clamp factor (A10 fix)
)
