"""Configuration for the Spdrbot robot."""

import os
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils

_USD_PATH = "/home/rithwik/SpdrBot-main/hexapod/hexa/hexa.usd"
_URDF_PATH = "/home/rithwik/SpdrBot-main/hexapod/hexa.urdf"


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
        print(f"[SPDRBOT] Warning: UrdfConverter patch skipped: {e}")


def _build_spdrbot_spawn_cfg():
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
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
        sleep_threshold=0.005,
        stabilization_threshold=0.001,
    )
    if os.path.exists(_USD_PATH):
        return sim_utils.UsdFileCfg(
            usd_path=_USD_PATH,
            activate_contact_sensors=True,
            rigid_props=rigid_props,
            articulation_props=articulation_props,
        )
    return sim_utils.UrdfFileCfg(
        asset_path=_URDF_PATH,
        usd_dir=os.path.dirname(_USD_PATH),
        usd_file_name=os.path.basename(_USD_PATH),
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


SPDRBOT_CFG = ArticulationCfg(
    spawn=_build_spdrbot_spawn_cfg(),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(0.0, 1.0, 0.0, 0.0),
        joint_pos={
            ".*base_link.*Revolute.*": 0.0,
            ".*NCCUP.*Revolute.*": 0.0,
            ".*NTHIGH.*Revolute.*": -1.57079632679,
        },
        joint_vel={
            ".*Revolute.*": 0.0,
        },
    ),
    actuators={
        "leg_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*Revolute.*"],
            effort_limit_sim=5.0,
            velocity_limit_sim=1.0,
            stiffness=5.0,
            damping=0.2,
        ),
    },
    soft_joint_pos_limit_factor=2,
)
"""Configuration of Spdrbot robot using implicit actuators."""