# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Stage 1 Flat Deployable Task Configuration for Z.O.E. Hexapod (Zoe-Hexapod-Flat-Deploy-v0)."""

import math
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from spdrbot3.assets.zoe_hexapod import ZOE_HEXAPOD_CFG
from .params import ZOE_PARAMS
from .terrains import ZOE_FLAT_TERRAIN_CFG
from . import mdp as zoe_mdp


@configclass
class ZoeSceneCfg(InteractiveSceneCfg):
    """Scene configuration for flat ground environment."""
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )


    robot = ZOE_HEXAPOD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )


@configclass
class ActionsCfg:
    """Action configuration for 18 joint position targets with delay and rate limiting."""
    joint_pos = zoe_mdp.DelayedRateLimitedJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*Revolute.*"],
        scale=0.50,
        max_delay_steps=ZOE_PARAMS.events.max_action_delay_steps,
        rate_limit_rad_per_step=math.radians(6.0),
    )



@configclass
class ObservationsCfg:
    """Observation configuration with separate Actor (63-D, noise on) and Privileged Critic (90-D) groups."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observation group (63-D, deployable, no linear velocity)."""
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=ZOE_PARAMS.obs_scale.base_ang_vel_scale,
            clip=ZOE_PARAMS.obs_scale.base_ang_vel_clip,
            noise=UniformNoiseCfg(
                n_min=-ZOE_PARAMS.obs_noise.gyro_noise_std,
                n_max=ZOE_PARAMS.obs_noise.gyro_noise_std,
            ),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            scale=ZOE_PARAMS.obs_scale.projected_gravity_scale,
            noise=UniformNoiseCfg(
                n_min=-ZOE_PARAMS.obs_noise.gravity_noise_std,
                n_max=ZOE_PARAMS.obs_noise.gravity_noise_std,
            ),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(
            func=zoe_mdp.joint_pos_rel_calibrated,
            scale=ZOE_PARAMS.obs_scale.joint_pos_rel_scale,
            clip=ZOE_PARAMS.obs_scale.joint_pos_rel_clip,
            noise=UniformNoiseCfg(
                n_min=-ZOE_PARAMS.obs_noise.joint_pos_noise_std,
                n_max=ZOE_PARAMS.obs_noise.joint_pos_noise_std,
            ),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=ZOE_PARAMS.obs_scale.joint_vel_scale,
            clip=ZOE_PARAMS.obs_scale.joint_vel_clip,
            noise=UniformNoiseCfg(
                n_min=-ZOE_PARAMS.obs_noise.joint_vel_noise_std,
                n_max=ZOE_PARAMS.obs_noise.joint_vel_noise_std,
            ),
        )
        actions = ObsTerm(
            func=mdp.last_action,
            scale=ZOE_PARAMS.obs_scale.last_action_scale,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic privileged observation group (90-D)."""
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=ZOE_PARAMS.obs_scale.base_ang_vel_scale,
            clip=ZOE_PARAMS.obs_scale.base_ang_vel_clip,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            scale=ZOE_PARAMS.obs_scale.projected_gravity_scale,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(
            func=zoe_mdp.joint_pos_rel_calibrated,
            scale=ZOE_PARAMS.obs_scale.joint_pos_rel_scale,
            clip=ZOE_PARAMS.obs_scale.joint_pos_rel_clip,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=ZOE_PARAMS.obs_scale.joint_vel_scale,
            clip=ZOE_PARAMS.obs_scale.joint_vel_clip,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            scale=ZOE_PARAMS.obs_scale.last_action_scale,
        )
        # Privileged terms (critic only)
        base_lin_vel = ObsTerm(
            func=zoe_mdp.base_lin_vel_privileged,
            scale=1.0,
        )
        foot_contacts = ObsTerm(
            func=zoe_mdp.foot_contact_flags,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*NFOOT.*")},
        )
        applied_torques = ObsTerm(
            func=zoe_mdp.applied_joint_torques,
            scale=0.1,
        )
        base_height = ObsTerm(
            func=zoe_mdp.base_height_privileged,
        )
        belly_contact = ObsTerm(
            func=zoe_mdp.belly_contact_flag,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*base_link.*|.*NCCUP.*|.*NTHIGH.*")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Domain Randomization and Reset Events."""
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": ZOE_PARAMS.events.static_friction_range,
            "dynamic_friction_range": ZOE_PARAMS.events.dynamic_friction_range,
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    scale_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*base_link.*"),
            "mass_distribution_params": ZOE_PARAMS.events.base_mass_scale_range,
            "operation": "scale",
        },
    )
    scale_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*N.*"),
            "mass_distribution_params": ZOE_PARAMS.events.body_mass_scale_range,
            "operation": "scale",
        },
    )
    base_com_offset = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*base_link.*"),
            "com_range": {
                "x": ZOE_PARAMS.events.base_com_offset_range,
                "y": ZOE_PARAMS.events.base_com_offset_range,
                "z": ZOE_PARAMS.events.base_com_offset_range,
            },
        },
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*Revolute.*"),
            "stiffness_distribution_params": ZOE_PARAMS.events.actuator_kp_scale_range,
            "damping_distribution_params": ZOE_PARAMS.events.actuator_kd_scale_range,
            "operation": "scale",
        },
    )
    joint_calibration_offsets = EventTerm(
        func=zoe_mdp.randomize_joint_calibration_offsets,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*Revolute.*"),
            "offset_range_rad": (-math.radians(ZOE_PARAMS.events.calibration_zero_offset_deg),
                                 math.radians(ZOE_PARAMS.events.calibration_zero_offset_deg)),
        },
    )
    reset_base = EventTerm(
        func=zoe_mdp.reset_root_near_ground,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": ZOE_PARAMS.SPAWN_Z_RANGE, "yaw": (-math.pi, math.pi)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": ZOE_PARAMS.events.reset_joint_offset_range,
            "velocity_range": (0.0, 0.0),
        },
    )
    push_robot = EventTerm(
        func=zoe_mdp.push_robot_base,
        mode="interval",
        interval_range_s=ZOE_PARAMS.events.push_interval_s_range,
        params={
            "velocity_range": {"x": ZOE_PARAMS.events.push_velocity_x_y_range, "y": ZOE_PARAMS.events.push_velocity_x_y_range},
        },
    )


@configclass
class RewardsCfg:
    """Reward configuration for Stage 1 flat deployable task."""
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=ZOE_PARAMS.rewards.track_lin_vel_xy_exp,
        params={"command_name": "base_velocity", "std": ZOE_PARAMS.rewards.track_lin_vel_std},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=ZOE_PARAMS.rewards.track_ang_vel_z_exp,
        params={"command_name": "base_velocity", "std": ZOE_PARAMS.rewards.track_ang_vel_std},
    )
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=ZOE_PARAMS.rewards.termination_penalty,
    )
    base_height_shaped = RewTerm(
        func=zoe_mdp.base_height_shaped,
        weight=1.0,
        params={
            "target_height": ZOE_PARAMS.TARGET_STANDING_HEIGHT,
            "min_height": ZOE_PARAMS.GATE_MIN_HEIGHT,
            "std": 0.015,
            "barrier_quad_weight": 150.0,
            "barrier_lin_weight": 10.0,
        },
    )
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=ZOE_PARAMS.rewards.lin_vel_z_l2_flat,
    )
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
    )
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=ZOE_PARAMS.rewards.dof_torques_l2,
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=ZOE_PARAMS.rewards.dof_acc_l2,
    )
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=ZOE_PARAMS.rewards.action_rate_l2,
    )
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=ZOE_PARAMS.rewards.flat_orientation_l2_flat,
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=ZOE_PARAMS.rewards.dof_pos_limits,
    )
    feet_air_time = RewTerm(
        func=zoe_mdp.feet_air_time_gated,
        weight=ZOE_PARAMS.rewards.feet_air_time,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*NFOOT.*"),
            "threshold": ZOE_PARAMS.rewards.feet_air_time_threshold,
        },
    )
    fewer_than_3_feet_contact = RewTerm(
        func=zoe_mdp.fewer_than_3_feet_contact,
        weight=ZOE_PARAMS.rewards.fewer_than_3_feet_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*NFOOT.*")},
    )
    fewer_than_4_feet_contact = RewTerm(
        func=zoe_mdp.fewer_than_4_feet_contact,
        weight=ZOE_PARAMS.rewards.fewer_than_4_feet_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*NFOOT.*")},
    )
    undesired_contacts = RewTerm(
        func=zoe_mdp.undesired_contacts,
        weight=ZOE_PARAMS.rewards.undesired_contacts,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*NTHIGH.*")},
    )
    foot_tip_slip = RewTerm(
        func=zoe_mdp.foot_tip_slip_penalty,
        weight=ZOE_PARAMS.rewards.foot_tip_slip,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*NFOOT.*"),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    foot_tip_clearance = RewTerm(
        func=zoe_mdp.foot_tip_clearance_reward,
        weight=ZOE_PARAMS.rewards.foot_tip_clearance,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*NFOOT.*"),
            "command_name": "base_velocity",
            "target_clearance": 0.05,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    stand_still_posture = RewTerm(
        func=zoe_mdp.stand_still_posture_penalty,
        weight=ZOE_PARAMS.rewards.stand_still_posture,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*NFOOT.*"),
            "command_name": "base_velocity",
            "cmd_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    stand_still_feet_grounded = RewTerm(
        func=zoe_mdp.stand_still_feet_grounded_reward,
        weight=ZOE_PARAMS.rewards.stand_still_feet_grounded,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*NFOOT.*"),
            "command_name": "base_velocity",
            "cmd_threshold": 0.05,
        },
    )
    joint_vel_limit = RewTerm(
        func=zoe_mdp.joint_vel_limit_penalty,
        weight=ZOE_PARAMS.rewards.joint_vel_limit,
        params={
            "limit": ZOE_PARAMS.NO_LOAD_SPEED_RAD_S,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    torque_limit = RewTerm(
        func=zoe_mdp.torque_limit_penalty,
        weight=ZOE_PARAMS.rewards.torque_limit,
        params={
            "threshold": ZOE_PARAMS.rewards.torque_soft_limit_nm,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # Unweighted TensorBoard Metrics (weight=0.0)
    tb_mean_base_height = RewTerm(
        func=zoe_mdp.metric_base_height,
        weight=0.0,
    )
    tb_non_foot_contact = RewTerm(
        func=zoe_mdp.metric_non_foot_contact,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*base_link.*|.*NCCUP.*|.*NTHIGH.*")},
    )
    tb_torque_saturation = RewTerm(
        func=zoe_mdp.metric_torque_saturation,
        weight=0.0,
        params={"threshold": ZOE_PARAMS.TORQUE_SAT_THRESHOLD_NM},
    )
    tb_mean_feet_down = RewTerm(
        func=zoe_mdp.metric_feet_down,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*NFOOT.*")},
    )


@configclass
class CommandsCfg:
    """Command generator configuration (with >=25% yaw-dominant envs, 10% standing envs, no heading command)."""
    base_velocity = zoe_mdp.ZoeUniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=ZOE_PARAMS.commands.resampling_time_range,
        rel_standing_envs=ZOE_PARAMS.commands.rel_standing_envs,
        rel_heading_envs=ZOE_PARAMS.commands.rel_heading_envs,
        rel_yaw_envs=ZOE_PARAMS.commands.rel_yaw_envs,
        yaw_dominant_wz_range=ZOE_PARAMS.commands.yaw_dominant_wz_range,
        yaw_dominant_lin_vel_range=ZOE_PARAMS.commands.yaw_dominant_lin_vel_range,
        heading_command=ZOE_PARAMS.commands.heading_command,
        debug_vis=True,
        ranges=zoe_mdp.ZoeUniformVelocityCommandCfg.Ranges(
            lin_vel_x=ZOE_PARAMS.commands.lin_vel_x_range,
            lin_vel_y=ZOE_PARAMS.commands.lin_vel_y_range,
            ang_vel_z=ZOE_PARAMS.commands.ang_vel_z_range,
        ),
    )


@configclass
class TerminationsCfg:
    """Episode terminations: timeout, base contact, bad tilt, and NaN state."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*base_link.*|.*NCCUP.*"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(
        func=zoe_mdp.bad_orientation,
        params={"limit_angle": 1.2},
    )
    invalid_state_nan = DoneTerm(
        func=zoe_mdp.invalid_state_nan,
    )


@configclass
class FlatCurriculumCfg:
    """Curriculum schedule ramping harsh penalty weights from 0.1 (iter 0) to 1.0 (iter 800)."""

    penalty_scale = CurrTerm(func=zoe_mdp.penalty_scale_ramp)


@configclass
class ZoeHexapodFlatDeployEnvCfg(ManagerBasedRLEnvCfg):
    """Stage 1 Environment configuration."""
    scene: ZoeSceneCfg = ZoeSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: FlatCurriculumCfg = FlatCurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.viewer.eye = [1.2, 1.2, 0.8]
        self.viewer.lookat = [0.0, 0.0, 0.2]
        self.sim.dt = 0.005
        self.sim.render_interval = 4
        self.sim.physx.enable_external_forces_every_iteration = True


@configclass
class ZoeHexapodFlatDeployPlayEnvCfg(ZoeHexapodFlatDeployEnvCfg):
    """Play variant (-Play-v0): 16 envs, fixed seed, noise and domain randomization disabled."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events = EventCfg()  # Clear randomization events
        self.actions.joint_pos.fixed_delay_steps = 0  # Configurable default 0 in play mode

