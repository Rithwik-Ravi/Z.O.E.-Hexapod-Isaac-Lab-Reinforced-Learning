# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for A1: height_above_ground(env) across flat, raised, lowered, and sloped tiles."""

import inspect
import math
import pytest
import torch

from spdrbot3.tasks.manager_based.zoe_deploy.params import ZOE_PARAMS, get_foot_tip_offsets_tensor
from spdrbot3.tasks.manager_based.zoe_deploy.mdp import rewards as zoe_rewards
from spdrbot3.tasks.manager_based.zoe_deploy.mdp import observations as zoe_obs
from spdrbot3.tasks.manager_based.zoe_deploy.mdp.rewards import (
    base_height_shaped,
    foot_clearance_relative,
    foot_tip_clearance_reward,
    ground_height_from_feet,
    ground_height_from_raycaster,
    height_above_ground,
    metric_base_height,
)
from spdrbot3.tasks.manager_based.zoe_deploy.mdp.observations import base_height_privileged


class _DummyScannerData:
    def __init__(self, ray_hits_z: torch.Tensor):
        # ray_hits_z: [N, num_rays]
        n, m = ray_hits_z.shape
        self.ray_hits_w = torch.zeros((n, m, 3), dtype=torch.float32)
        self.ray_hits_w[:, :, 2] = ray_hits_z


class _DummyScanner:
    def __init__(self, ray_hits_z: torch.Tensor):
        self.data = _DummyScannerData(ray_hits_z)


class _DummyContactData:
    def __init__(self, forces_z: torch.Tensor):
        n, b = forces_z.shape
        self.net_forces_w = torch.zeros((n, b, 3), dtype=torch.float32)
        self.net_forces_w[:, :, 2] = forces_z


class _DummyContactSensor:
    def __init__(self, forces_z: torch.Tensor):
        self.data = _DummyContactData(forces_z)
        self.body_names = [f"NFOOT{i}" for i in range(1, 7)]


class _DummyRobotData:
    def __init__(self, root_z: torch.Tensor, ground_z_at_feet: torch.Tensor):
        n = root_z.shape[0]
        self.root_pos_w = torch.zeros((n, 3), dtype=torch.float32)
        self.root_pos_w[:, 2] = root_z

        foot_names = [f"NFOOT{i}" for i in range(1, 7)]
        tip_offsets = get_foot_tip_offsets_tensor(foot_names, "cpu", torch.float32)  # [6, 3]
        # With identity body_quat_w (1, 0, 0, 0), tip_pos_w = body_pos_w + tip_offsets.
        # Set body_pos_w so tip_pos_w[..., 2] == ground_z_at_feet
        self.body_pos_w = torch.zeros((n, 6, 3), dtype=torch.float32)
        self.body_pos_w[:, :, 2] = ground_z_at_feet - tip_offsets[:, 2].unsqueeze(0)
        self.body_quat_w = torch.zeros((n, 6, 4), dtype=torch.float32)
        self.body_quat_w[:, :, 0] = 1.0
        self.body_lin_vel_w = torch.zeros((n, 6, 3), dtype=torch.float32)
        self.body_ang_vel_w = torch.zeros((n, 6, 3), dtype=torch.float32)


class _DummyRobot:
    def __init__(self, root_z: torch.Tensor, ground_z_at_feet: torch.Tensor):
        self.data = _DummyRobotData(root_z, ground_z_at_feet)
        self.body_names = [f"NFOOT{i}" for i in range(1, 7)]


class _DummyTerrainCfg:
    def __init__(self, terrain_type: str = "generator"):
        self.terrain_type = terrain_type


class _DummyTerrain:
    def __init__(self, terrain_type: str = "generator"):
        self.cfg = _DummyTerrainCfg(terrain_type)


class _DummyScene(dict):
    def __init__(self, *args, terrain_type="generator", env_origins=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.terrain = _DummyTerrain(terrain_type)
        self.env_origins = env_origins


class _DummyCommandManager:
    def __init__(self, cmd: torch.Tensor):
        self._cmd = cmd

    def get_command(self, name: str) -> torch.Tensor:
        return self._cmd


class _DummyEnv:
    def __init__(self, scene: _DummyScene, cmd: torch.Tensor | None = None):
        self.scene = scene
        if cmd is not None:
            self.command_manager = _DummyCommandManager(cmd)


class _DummyCfg:
    def __init__(self, name="robot", body_ids=None):
        self.name = name
        self.body_ids = body_ids if body_ids is not None else [0, 1, 2, 3, 4, 5]


def test_flat_plane_equals_old_value():
    """On the flat plane (z=0), height_above_ground(env) must equal the old root_pos_w[:, 2] exactly."""
    root_z = torch.tensor([0.1356, 0.1427, 0.1650, 0.1100], dtype=torch.float32)
    ground_feet = torch.zeros((4, 6), dtype=torch.float32)
    robot = _DummyRobot(root_z, ground_feet)

    # Case 1: Stage 1 analytical plane (no height_scanner in scene)
    scene_plane = _DummyScene({"robot": robot}, terrain_type="plane")
    env_plane = _DummyEnv(scene_plane)
    h_plane = height_above_ground(env_plane)
    assert torch.equal(h_plane, robot.data.root_pos_w[:, 2])

    # Case 2: Flat tile at z=0 with height_scanner RayCaster
    ray_hits_flat = torch.zeros((4, 9), dtype=torch.float32)
    scene_ray = _DummyScene(
        {"robot": robot, "height_scanner": _DummyScanner(ray_hits_flat)},
        terrain_type="generator",
    )
    env_ray = _DummyEnv(scene_ray)
    h_ray = height_above_ground(env_ray)
    assert torch.equal(h_ray, robot.data.root_pos_w[:, 2])


def test_height_above_ground_flat_raised_lowered_slope():
    """Test height_above_ground and base_height_shaped on:
    0: Flat tile (ground_z = 0.0 m)
    1: Raised tile (ground_z = +0.608 m, origin_z = +0.608 m)
    2: Lowered pit tile (ground_z = -0.681 m, origin_z = -0.681 m)
    3: Slope tile walked away from origin (origin_z = 0.0 m, ground under robot = +0.600 m on 0.30 slope)
    """
    target_h = ZOE_PARAMS.TARGET_STANDING_HEIGHT  # 0.1356 m
    ground_z_true = torch.tensor([0.0, 0.608, -0.681, 0.600], dtype=torch.float32)
    root_z = ground_z_true + target_h

    # Simulate 3x3 RayCaster patch (9 rays along X in [-0.15, 0, +0.15] m)
    # On the 0.30 slope (env 3), the 3x3 patch has z hits [0.555, 0.600, 0.645] -> mean = 0.600 m
    ray_hits = ground_z_true.unsqueeze(1).repeat(1, 9)
    slope_offsets = torch.tensor([-0.045, 0.0, 0.045, -0.045, 0.0, 0.045, -0.045, 0.0, 0.045])
    ray_hits[3] = 0.600 + slope_offsets

    # Simulate foot contact heights (6 feet around base, symmetric on slope with tiny <5mm variance)
    foot_ground = ground_z_true.unsqueeze(1).repeat(1, 6)
    foot_ground[3] = 0.600 + torch.tensor([-0.015, 0.0, 0.015, 0.015, 0.0, -0.015])
    forces_z = torch.full((4, 6), 8.0, dtype=torch.float32)

    # Note env 3 has env_origin_z = 0.0 even though robot walked up slope to ground_z = +0.600 m!
    env_origins = torch.tensor([
        [0.0, 0.0, 0.0],
        [8.0, 0.0, 0.608],
        [16.0, 0.0, -0.681],
        [24.0, 0.0, 0.0],
    ], dtype=torch.float32)

    robot = _DummyRobot(root_z, foot_ground)
    scene = _DummyScene(
        {
            "robot": robot,
            "height_scanner": _DummyScanner(ray_hits),
            "contact_forces": _DummyContactSensor(forces_z),
        },
        terrain_type="generator",
        env_origins=env_origins,
    )
    env = _DummyEnv(scene, cmd=torch.tensor([[0.2, 0.0, 0.0]] * 4))

    # 1. Verify RayCaster height_above_ground equals target_h across all 4 tiles (including slope away from origin)
    h_ray = height_above_ground(env, method="raycaster")
    assert torch.allclose(h_ray, torch.full((4,), target_h), atol=1e-5)

    # 2. Cross-check against foot-contact fallback: within 1 cm on flat, 2 cm on slope
    h_feet = height_above_ground(env, method="feet")
    assert torch.abs(h_ray[0] - h_feet[0]).item() < 0.01
    assert torch.max(torch.abs(h_ray - h_feet)).item() < 0.02

    # 3. Verify base_height_shaped returns ~1.0 across all 4 tiles
    rew = base_height_shaped(env)
    assert torch.allclose(rew, torch.ones(4), atol=1e-4)

    # 4. Verify metric_base_height and base_height_privileged return local height above ground
    assert torch.allclose(metric_base_height(env), torch.full((4,), target_h), atol=1e-5)
    assert torch.allclose(base_height_privileged(env), torch.full((4, 1), target_h), atol=1e-5)

    # 5. Verify foot_clearance_relative and foot_tip_clearance_reward are invariant to tile elevation
    robot_cfg = _DummyCfg("robot", [0, 1, 2, 3, 4, 5])
    contact_cfg = _DummyCfg("contact_forces", [0, 1, 2, 3, 4, 5])
    fc_rel = foot_clearance_relative(env, target_height=0.20, asset_cfg=robot_cfg)
    assert torch.allclose(fc_rel, fc_rel[0].expand(4), atol=1e-3)

    # Put 2 feet in swing at +0.05m above local ground on all 4 tiles
    forces_z[:, [0, 3]] = 0.0
    env.scene["contact_forces"] = _DummyContactSensor(forces_z)
    robot.data.body_pos_w[:, [0, 3], 2] += 0.05
    ft_rew = foot_tip_clearance_reward(env, sensor_cfg=contact_cfg, target_clearance=0.05, asset_cfg=robot_cfg)
    assert torch.allclose(ft_rew[:3], torch.full((3,), 2.0), atol=1e-4)
    expected_slope_ft = 2.0 * math.exp(-((0.015 / 0.02) ** 2))
    assert abs(ft_rew[3].item() - expected_slope_ft) < 1e-4


def test_unused_base_height_l2_deleted_and_no_raw_world_z_in_rewards():
    """Confirm unused base_height_l2 was removed and all height consumers call height_above_ground."""
    assert not hasattr(zoe_rewards, "base_height_l2"), "Unused base_height_l2 should be deleted"
    for fn in (base_height_shaped, metric_base_height, base_height_privileged):
        src = inspect.getsource(fn)
        assert "height_above_ground" in src
        assert "root_pos_w[:, 2]" not in src
