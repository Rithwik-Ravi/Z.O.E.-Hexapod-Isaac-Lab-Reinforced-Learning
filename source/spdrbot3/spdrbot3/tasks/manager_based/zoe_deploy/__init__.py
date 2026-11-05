# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Gym Task registration for Z.O.E. Hexapod deployable tasks."""

try:
    import gymnasium as gym
except ImportError:
    gym = None

# Register Gym environments using string entry points to support lightweight imports
if gym is not None:
    gym.register(
        id="Zoe-Hexapod-Flat-Deploy-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg_flat:ZoeHexapodFlatDeployEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ZoePPORunnerCfg",
        },
    )

    gym.register(
        id="Zoe-Hexapod-Flat-Deploy-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg_flat:ZoeHexapodFlatDeployPlayEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ZoePPORunnerCfg",
        },
    )

    gym.register(
        id="Zoe-Hexapod-Terrain-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg_terrain:ZoeHexapodTerrainEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ZoeTerrainPPORunnerCfg",
        },
    )

    gym.register(
        id="Zoe-Hexapod-Terrain-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg_terrain:ZoeHexapodTerrainPlayEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ZoeTerrainPPORunnerCfg",
        },
    )
