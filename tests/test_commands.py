# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Zoe yaw command generation and distribution."""

import pytest
import torch

from spdrbot3.tasks.manager_based.zoe_deploy.params import ZOE_PARAMS
from spdrbot3.tasks.manager_based.zoe_deploy.mdp.commands import (
    ZoeUniformVelocityCommand,
    ZoeUniformVelocityCommandCfg,
)


def test_yaw_command_params():
    """Verify parameters guarantee at least 25% yaw-dominant environments."""
    assert ZOE_PARAMS.commands.rel_yaw_envs >= 0.25, f"rel_yaw_envs={ZOE_PARAMS.commands.rel_yaw_envs} < 0.25"
    assert ZOE_PARAMS.commands.yaw_dominant_wz_range[0] >= 0.30
    assert ZOE_PARAMS.commands.yaw_dominant_wz_range[1] <= 0.60
    assert abs(ZOE_PARAMS.commands.yaw_dominant_lin_vel_range[0]) <= 0.05
    assert abs(ZOE_PARAMS.commands.yaw_dominant_lin_vel_range[1]) <= 0.05
    assert ZOE_PARAMS.rewards.track_ang_vel_z_exp >= 1.0


def test_zoe_uniform_velocity_command_distribution():
    """Verify ZoeUniformVelocityCommand samples at least 25% yaw-dominant environments."""
    cfg = ZoeUniformVelocityCommandCfg(
        asset_name="robot",
        rel_yaw_envs=0.30,
        rel_standing_envs=0.10,
        ranges=ZoeUniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.25, 0.30),
            lin_vel_y=(-0.20, 0.20),
            ang_vel_z=(-0.60, 0.60),
        ),
    )

    class DummyRobot:
        pass

    class DummyScene:
        def __getitem__(self, key):
            return DummyRobot()

    class DummyEnv:
        def __init__(self, num_envs=2000):
            self.num_envs = num_envs
            self.device = "cpu"
            self.step_dt = 0.02
            self.scene = DummyScene()

    dummy_env = DummyEnv(num_envs=2000)
    cmd_gen = ZoeUniformVelocityCommand(cfg, dummy_env)

    # Resample all envs
    cmd_gen._resample_command(list(range(dummy_env.num_envs)))

    cmds = cmd_gen.command  # [2000, 3] -> (vx, vy, wz)
    vx = cmds[:, 0]
    vy = cmds[:, 1]
    wz = cmds[:, 2]

    # Yaw dominant condition: |wz| in [0.3, 0.6] and small vx, vy
    yaw_dominant = (wz.abs() >= 0.299) & (vx.abs() <= 0.051) & (vy.abs() <= 0.051)
    yaw_fraction = yaw_dominant.float().mean().item()

    # Must be at least 25% (nominal 30% +- binomial variance for N=2000 is ~1%)
    assert yaw_fraction >= 0.25, f"Expected >= 25% yaw-dominant envs, got {yaw_fraction*100:.1f}%"
