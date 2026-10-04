import math
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns, ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, HfRandomUniformTerrainCfg
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from spdrbot3.assets.spdrbot import SPDRBOT_CFG

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv

def foot_clearance(env: ManagerBasedRLEnv, target_height: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    # Get the height of the feet
    foot_height = foot_pos_w[:, :, 2]
    # Penalize feet that go above target height
    return torch.sum(torch.square(torch.clamp(foot_height - target_height, min=0.0)), dim=1)

def tripod_gait_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor = env.scene[sensor_cfg.name]
    # Net forces in Z direction
    foot_forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    # Check which feet are in contact (force > 1.0)
    in_contact = (foot_forces_z > 1.0).float()
    # Number of feet in contact per environment
    num_feet_in_contact = torch.sum(in_contact, dim=1)
    # Penalize deviation from exactly 3 feet
    return torch.square(num_feet_in_contact - 3.0)

def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)

FLAT_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "flat": HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=(0.0, 0.0), noise_step=0.02, border_width=0.25
        ),
    },
)

@configclass
class SpdrBotSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=FLAT_TERRAIN_CFG,
        max_init_terrain_level=5,
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
            color=(0.75, 0.75, 0.75)
        )
    )

    robot = SPDRBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.actuators["leg_joints"].stiffness = 15.0
    robot.actuators["leg_joints"].damping = 1.5
    robot.actuators["leg_joints"].effort_limit_sim = 5.0
    robot.actuators["leg_joints"].velocity_limit_sim = 4.5

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True
    )

@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*Revolute.*"], scale=0.25)

def safe_base_lin_vel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return torch.nan_to_num(mdp.base_lin_vel(env, asset_cfg), nan=0.0, posinf=0.0, neginf=0.0)

def safe_base_ang_vel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return torch.nan_to_num(mdp.base_ang_vel(env, asset_cfg), nan=0.0, posinf=0.0, neginf=0.0)

def safe_projected_gravity(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return torch.nan_to_num(mdp.projected_gravity(env, asset_cfg), nan=0.0, posinf=0.0, neginf=0.0)

def safe_joint_pos_rel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return torch.nan_to_num(mdp.joint_pos_rel(env, asset_cfg), nan=0.0, posinf=0.0, neginf=0.0)

def safe_joint_vel_rel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return torch.nan_to_num(mdp.joint_vel_rel(env, asset_cfg), nan=0.0, posinf=0.0, neginf=0.0)

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=safe_base_lin_vel, scale=2.0, clip=(-5.0, 5.0))
        base_ang_vel = ObsTerm(func=safe_base_ang_vel, scale=0.25, clip=(-5.0, 5.0))
        projected_gravity = ObsTerm(func=safe_projected_gravity, scale=1.0)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=safe_joint_pos_rel, scale=1.0, clip=(-5.0, 5.0))
        joint_vel = ObsTerm(func=safe_joint_vel_rel, scale=0.05, clip=(-5.0, 5.0))
        actions = ObsTerm(func=mdp.last_action, scale=1.0)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link_v1"),
            "mass_distribution_params": (0.0, 0.5),
            "operation": "add",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.6, 0.8), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

@configclass
class RewardsCfg:
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=2.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1e-7)
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
    is_alive = RewTerm(func=is_alive, weight=10.0)
    
    foot_clearance = RewTerm(
        func=foot_clearance,
        weight=0.0,
        params={"target_height": 0.05, "asset_cfg": SceneEntityCfg("robot", body_names=".*FOOT.*")},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    
    tripod_gait_penalty = RewTerm(
        func=tripod_gait_penalty,
        weight=-0.2,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT.*")},
    )

@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link_v1"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.pi / 2.0}
    )

@configclass
class Spdrbot3RoughEnvCfg(ManagerBasedRLEnvCfg):
    scene: SpdrBotSceneCfg = SpdrBotSceneCfg(num_envs=100, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.viewer.eye = [1.2, 1.2, 0.8]
        self.viewer.lookat = [0.0, 0.0, 0.3]
        self.sim.dt = 0.005
        self.sim.render_interval = 4
        self.sim.physx.enable_external_forces_every_iteration = True
