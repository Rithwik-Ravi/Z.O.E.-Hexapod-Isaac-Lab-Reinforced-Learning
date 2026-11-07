# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Stage 2 Multi-Terrain Task Configuration for Z.O.E. Hexapod (Zoe-Hexapod-Terrain-v0)."""

import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .env_cfg_flat import ZoeHexapodFlatDeployEnvCfg, ZoeSceneCfg, RewardsCfg
from .params import ZOE_PARAMS
from .terrains import ZOE_TERRAINS_CFG
from . import mdp as zoe_mdp


@configclass
class ZoeTerrainSceneCfg(ZoeSceneCfg):
    """Multi-terrain scene configuration."""
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ZOE_TERRAINS_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.15, size=[0.3, 0.3]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class ZoeTerrainRewardsCfg(RewardsCfg):
    """Rewards overridden for terrain navigation."""
    # Relax orientation and Z-velocity penalties on slopes/stairs (A19 fix)
    flat_orientation_l2 = RewTerm(
        func=zoe_mdp.flat_orientation_l2,
        weight=ZOE_PARAMS.rewards.flat_orientation_l2_terrain,
    )
    lin_vel_z_l2 = RewTerm(
        func=zoe_mdp.lin_vel_z_l2,
        weight=ZOE_PARAMS.rewards.lin_vel_z_l2_terrain,
    )
    # Add undesired body contacts penalty (thigh/coxa touching terrain)
    undesired_contacts = RewTerm(
        func=zoe_mdp.undesired_contacts,
        weight=ZOE_PARAMS.rewards.undesired_contacts,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*(NCCUP|NTHIGH).*")},
    )


@configclass
class CurriculumCfg:
    """Terrain curriculum configuration."""
    terrain_levels = CurrTerm(func=zoe_mdp.zoe_terrain_curriculum)


@configclass
class ZoeHexapodTerrainEnvCfg(ZoeHexapodFlatDeployEnvCfg):
    """Stage 2 Multi-Terrain Environment Configuration."""
    scene: ZoeTerrainSceneCfg = ZoeTerrainSceneCfg(num_envs=1024, env_spacing=2.5)
    rewards: ZoeTerrainRewardsCfg = ZoeTerrainRewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()


@configclass
class ZoeHexapodTerrainPlayEnvCfg(ZoeHexapodTerrainEnvCfg):
    """Play variant (-Play-v0): 16 envs, fixed seed, noise and domain randomization disabled."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events = type(self.events)()  # Clear randomization events
