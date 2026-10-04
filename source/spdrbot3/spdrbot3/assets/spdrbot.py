"""Configuration for the Spdrbot robot."""

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils

SPDRBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/rithwik/SpdrBot-main/hexapod/hexa/hexa.usd",
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
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            ".*Revolute.*": 0.0,
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