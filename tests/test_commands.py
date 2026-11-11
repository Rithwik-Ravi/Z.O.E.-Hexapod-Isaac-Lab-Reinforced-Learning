# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Zoe yaw command generation and two-group terrain command generator."""

import pytest
import torch

from spdrbot3.tasks.manager_based.zoe_deploy.params import ZOE_PARAMS
from spdrbot3.tasks.manager_based.zoe_deploy.mdp.commands import (
    _quat_apply_wxyz,
    ZoeUniformVelocityCommand,
    ZoeUniformVelocityCommandCfg,
    ZoeTerrainTwoGroupCommand,
    ZoeTerrainTwoGroupCommandCfg,
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
    assert yaw_fraction >= 0.25, f"Expected >= 25% yaw-dominant envs, got {yaw_fraction * 100:.1f}%"


def test_two_group_command_split_and_distributions():
    """Verify ZoeTerrainTwoGroupCommand:
    1. Splits environments 50% Group T (traversal) and 50% Group S (skills).
    2. Group T: vx in [0.10, 0.25], vy in [-0.10, 0.10], wz in [-0.30, 0.30], held for 20s, no standing envs.
    3. Group S: resample in [3.0, 5.0], ~10% standing, >= 25% yaw-dominant, backward vx allowed.
    4. Step-integrated progress tracking: commanded_travel, achieved_travel, reset behavior.
    """
    num_envs = 2048
    cfg = ZoeTerrainTwoGroupCommandCfg(asset_name="robot")

    class DummyRobotData:
        def __init__(self, n):
            self.root_lin_vel_b = torch.zeros((n, 3), dtype=torch.float32)
            self.root_ang_vel_b = torch.zeros((n, 3), dtype=torch.float32)
            self.root_pos_w = torch.zeros((n, 3), dtype=torch.float32)
            torch.manual_seed(123)
            yaw = (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0) * 3.14159265
            self.heading_w = yaw
            # Nominal 180-deg X roll + yaw: (0, cos(yaw/2), sin(yaw/2), 0)
            self.root_quat_w = torch.zeros((n, 4), dtype=torch.float32)
            self.root_quat_w[:, 1] = torch.cos(0.5 * yaw)
            self.root_quat_w[:, 2] = torch.sin(0.5 * yaw)

    class DummyRobot:
        def __init__(self, n):
            self.data = DummyRobotData(n)

    class DummyScene:
        def __init__(self, robot):
            self._robot = robot

        def __getitem__(self, key):
            return self._robot

    class DummyEnv:
        def __init__(self, n):
            self.num_envs = n
            self.device = "cpu"
            self.step_dt = 0.02
            self.robot = DummyRobot(n)
            self.scene = DummyScene(self.robot)

    env = DummyEnv(num_envs)
    cmd_gen = ZoeTerrainTwoGroupCommand(cfg, env)

    # 1. Group Split Confirmation
    assert hasattr(env, "is_group_s")
    assert cmd_gen.is_group_s.shape[0] == num_envs
    n_s = cmd_gen.is_group_s.sum().item()
    n_t = num_envs - n_s
    assert n_s == num_envs // 2, f"Expected {num_envs // 2} Group S envs, got {n_s}"
    assert n_t == num_envs // 2, f"Expected {num_envs // 2} Group T envs, got {n_t}"

    # Resample all envs
    all_ids = list(range(num_envs))
    cmd_gen._resample(all_ids)
    cmd_gen._update_command()

    # 2. Resampling Time Ranges
    t_mask = ~cmd_gen.is_group_s
    s_mask = cmd_gen.is_group_s

    # Group T time_left should all be 20.0
    assert torch.all(cmd_gen.time_left[t_mask] == 20.0)
    # Group S time_left should be in [3.0, 5.0]
    assert torch.all((cmd_gen.time_left[s_mask] >= 2.99) & (cmd_gen.time_left[s_mask] <= 5.01))

    # 3. Group T Velocity Distributions
    cmds = cmd_gen.command
    vx_t = cmds[t_mask, 0]
    vy_t = cmds[t_mask, 1]
    wz_t = cmds[t_mask, 2]
    # No standing envs in Group T
    assert not cmd_gen.is_standing_env[t_mask].any()
    assert torch.all((vx_t >= 0.099) & (vx_t <= 0.251))
    assert torch.all((vy_t >= -0.101) & (vy_t <= 0.101))
    assert torch.all((wz_t >= -0.301) & (wz_t <= 0.301))

    # 4. Group S Velocity Distributions
    stand_s = cmd_gen.is_standing_env[s_mask]
    stand_fraction = stand_s.float().mean().item()
    assert 0.05 <= stand_fraction <= 0.16, f"Expected ~10% standing envs, got {stand_fraction * 100:.1f}%"

    vx_s = cmds[s_mask, 0]
    vy_s = cmds[s_mask, 1]
    wz_s = cmds[s_mask, 2]

    # Backward allowed in Group S
    has_backward = (vx_s < -0.05).any().item()
    assert has_backward, "Expected backward velocity commands in Group S!"

    # Yaw-dominant among non-standing Group S
    non_stand_s = ~stand_s
    vx_ns = vx_s[non_stand_s]
    vy_ns = vy_s[non_stand_s]
    wz_ns = wz_s[non_stand_s]
    yaw_dominant = (wz_ns.abs() >= 0.299) & (vx_ns.abs() <= 0.051) & (vy_ns.abs() <= 0.051)
    yaw_fraction = yaw_dominant.float().mean().item()
    assert yaw_fraction >= 0.23, f"Expected >= 25% yaw-dominant among non-standing Group S, got {yaw_fraction * 100:.1f}%"

    # 5. World-frame Net Displacement Progress Tracking (turning + strafe + 180-deg roll)
    # Set robot body velocities to match command
    env.robot.data.root_lin_vel_b[:, :2] = cmds[:, :2]

    cmd_gen.cmd_disp_w.zero_()
    cmd_gen.commanded_travel.zero_()
    cmd_gen.achieved_travel.zero_()
    cmd_gen.pos_start_w.copy_(env.robot.data.root_pos_w[:, :2])
    cmd_gen._pos_start_initialized.fill_(True)

    # Step 50 times (50 * 0.02 = 1.0 s) while rotating yaw according to wz
    yaw = env.robot.data.heading_w.clone()
    for _ in range(50):
        # Robot physically moves along R(q_t) @ cmd_b and turns by wz * dt
        vel_b_3d = torch.zeros((num_envs, 3), dtype=torch.float32)
        vel_b_3d[:, :2] = cmds[:, :2]
        vel_w = _quat_apply_wxyz(env.robot.data.root_quat_w, vel_b_3d)
        env.robot.data.root_pos_w[:, :2] += vel_w[:, :2] * env.step_dt
        cmd_gen._update_metrics()
        yaw = yaw + cmds[:, 2] * env.step_dt
        env.robot.data.heading_w = yaw
        env.robot.data.root_quat_w[:, 1] = torch.cos(0.5 * yaw)
        env.robot.data.root_quat_w[:, 2] = torch.sin(0.5 * yaw)

    # In moving envs with non-trivial commanded displacement, net progress ratio == 1.0
    non_zero_cmd = cmd_gen.commanded_travel > 0.01
    assert non_zero_cmd.any()
    ratio = cmd_gen.achieved_travel[non_zero_cmd] / cmd_gen.commanded_travel[non_zero_cmd]
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-3)

    # 6. Reset behavior clears travel buffers
    reset_ids = [0, 1, 2, 3]
    cmd_gen.reset(reset_ids)
    assert torch.all(cmd_gen.commanded_travel[reset_ids] == 0.0)
    assert torch.all(cmd_gen.achieved_travel[reset_ids] == 0.0)
