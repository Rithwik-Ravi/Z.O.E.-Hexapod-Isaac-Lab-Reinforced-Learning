# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Z.O.E. Hexapod curriculum logic (A3)."""

import pytest
import torch
from spdrbot3.tasks.manager_based.zoe_deploy.mdp.curriculum import (
    compute_curriculum_step,
    zoe_terrain_curriculum,
)


def test_curriculum_step_scalar():
    # Success case: walked >= 0.75 * commanded, not terminated early
    move_up, move_down = compute_curriculum_step(
        distance_walked=0.8, commanded_distance=1.0, was_terminated_early=False, cmd_speed=0.20
    )
    assert move_up is True
    assert move_down is False

    # Early termination case: walked >= 0.75 * commanded but terminated early
    move_up, move_down = compute_curriculum_step(
        distance_walked=0.8, commanded_distance=1.0, was_terminated_early=True, cmd_speed=0.20
    )
    assert move_up is False
    assert move_down is False

    # Failure case: walked < 0.35 * commanded
    move_up, move_down = compute_curriculum_step(
        distance_walked=0.2, commanded_distance=1.0, was_terminated_early=False, cmd_speed=0.20
    )
    assert move_up is False
    assert move_down is True

    # Keep-level middle band: 0.35 <= walked < 0.75
    move_up, move_down = compute_curriculum_step(
        distance_walked=0.5, commanded_distance=1.0, was_terminated_early=False, cmd_speed=0.20
    )
    assert move_up is False
    assert move_down is False


def test_curriculum_standing_envs_no_decision():
    """Standing envs (|cmd| < 0.1 m/s or commanded_distance < 0.5 m) must NOT promote or demote."""
    # Scalar standing env (cmd_speed = 0.0, distance = 0.0, commanded_distance = 0.0)
    move_up, move_down = compute_curriculum_step(
        distance_walked=0.0, commanded_distance=0.0, was_terminated_early=False, cmd_speed=0.0
    )
    assert move_up is False
    assert move_down is False

    # Scalar small speed (|cmd| = 0.05 < 0.10 m/s even over 20s -> commanded_distance = 1.0 m)
    move_up, move_down = compute_curriculum_step(
        distance_walked=0.9, commanded_distance=1.0, was_terminated_early=False, cmd_speed=0.05
    )
    assert move_up is False
    assert move_down is False

    # Scalar short commanded distance (commanded_distance = 0.40 < 0.50 m even with cmd_speed = 0.20 m/s)
    move_up, move_down = compute_curriculum_step(
        distance_walked=0.0, commanded_distance=0.40, was_terminated_early=True, cmd_speed=0.20
    )
    assert move_up is False
    assert move_down is False

    # Tensor batch with standing and walking envs
    dist_walked = torch.tensor([0.0, 0.9, 0.1, 0.8, 0.2])
    cmd_dist = torch.tensor([0.0, 1.0, 0.4, 1.0, 1.0])
    cmd_speed = torch.tensor([0.0, 0.05, 0.20, 0.20, 0.20])
    early_term = torch.tensor([False, False, True, False, False])

    move_up, move_down = compute_curriculum_step(dist_walked, cmd_dist, early_term, cmd_speed=cmd_speed)
    assert torch.equal(move_up, torch.tensor([False, False, False, True, False]))
    assert torch.equal(move_down, torch.tensor([False, False, False, False, True]))


def test_zoe_terrain_curriculum_far_from_world_origin_clamping_and_logging():
    """Verify zoe_terrain_curriculum:
    (a) Subtracts env_origins xy so robots at (36m, 24m) are measured by local travel, not 43m from (0,0)
    (b) Standing envs keep their level
    (c) Promotes >= 0.75x (not early term) and demotes < 0.35x
    (d) Clamps levels to [0, max_level] (0 does not go to -1; 9 does not wrap or exceed 9)
    (e) Logs per-terrain-type mean_level, promotion_rate, demotion_rate, survival in extras['log']
    """

    class _DummyTerrain:
        def __init__(self):
            self.max_terrain_level = 10  # levels 0..9
            # 6 envs:
            # env 0 (col 0 = flat): level 0, walks only 0.1m of 2.0m at world (36, 24) -> demote clamped at 0
            # env 1 (col 2 = rough): level 5, walks only 0.1m of 2.0m at world (-32, 28) -> demote to 4
            # env 2 (col 4 = slope_up): level 4, walks 1.8m of 2.0m at world (20, -20), timeout -> promote to 5
            # env 3 (col 5 = slope_down): level 9 (max), walks 1.9m of 2.0m, timeout -> promote clamped at 9
            # env 4 (col 6 = stairs_up): level 3, standing (cmd=0, dist=0) at world (40, 40) -> keep level 3
            # env 5 (col 9 = boxes): level 2, walks 1.8m of 2.0m but early terminated -> keep level 2
            self.terrain_levels = torch.tensor([0, 5, 4, 9, 3, 2], dtype=torch.long)
            self.terrain_types = torch.tensor([0, 2, 4, 5, 6, 9], dtype=torch.long)
            self.terrain_origins = torch.zeros((10, 10, 3), dtype=torch.float32)
            for r in range(10):
                for c in range(10):
                    self.terrain_origins[r, c] = torch.tensor([r * 8.0 - 36.0, c * 8.0 - 36.0, r * 0.05])
            self.env_origins = torch.tensor([
                [36.0, 24.0, 0.0],
                [-32.0, 28.0, 0.25],
                [20.0, -20.0, 0.40],
                [-16.0, -24.0, -0.60],
                [40.0, 40.0, 0.15],
                [12.0, 16.0, 0.10],
            ], dtype=torch.float32)

        def update_env_origins(self, env_ids, move_up, move_down):
            # Simulate Isaac Lab's default behavior (including random wrap at >= max_terrain_level)
            self.terrain_levels[env_ids] += 1 * move_up.long() - 1 * move_down.long()
            self.terrain_levels[env_ids] = torch.where(
                self.terrain_levels[env_ids] >= self.max_terrain_level,
                torch.zeros_like(self.terrain_levels[env_ids]),  # would wrap to 0 without A3(d) clamp!
                torch.clip(self.terrain_levels[env_ids], 0),
            )

    class _DummyRobotData:
        def __init__(self, origins, displacements_xy):
            self.root_pos_w = origins.clone()
            self.root_pos_w[:, 0] += displacements_xy[:, 0]
            self.root_pos_w[:, 1] += displacements_xy[:, 1]

    class _DummyRobot:
        def __init__(self, origins, displacements_xy):
            self.data = _DummyRobotData(origins, displacements_xy)

    class _DummyCmdMgr:
        def __init__(self, cmds):
            self._cmds = cmds

        def get_command(self, name):
            return self._cmds

    class _DummyTermMgr:
        def __init__(self, terminated, time_outs):
            self.terminated = terminated
            self.time_outs = time_outs

    class _DummyScene(dict):
        def __init__(self, terrain, robot):
            super().__init__({"robot": robot})
            self.terrain = terrain
            self.env_origins = terrain.env_origins

    class _DummyEnv:
        def __init__(self):
            terrain = _DummyTerrain()
            # Local displacements from each tile's env_origin
            displacements = torch.tensor([
                [0.10, 0.0],  # env 0: 0.10m (< 0.35 * 2.0)
                [0.10, 0.0],  # env 1: 0.10m (< 0.35 * 2.0)
                [1.80, 0.0],  # env 2: 1.80m (>= 0.75 * 2.0)
                [1.90, 0.0],  # env 3: 1.90m (>= 0.75 * 2.0)
                [0.00, 0.0],  # env 4: 0.00m (standing)
                [1.80, 0.0],  # env 5: 1.80m (early term)
            ], dtype=torch.float32)
            robot = _DummyRobot(terrain.env_origins, displacements)
            self.scene = _DummyScene(terrain, robot)
            self.step_dt = 0.02
            self.episode_length_buf = torch.full((6,), 500, dtype=torch.long)  # 10.0s
            cmds = torch.tensor([
                [0.20, 0.0, 0.0],  # cmd_dist = 2.0m
                [0.20, 0.0, 0.0],  # cmd_dist = 2.0m
                [0.20, 0.0, 0.0],  # cmd_dist = 2.0m
                [0.20, 0.0, 0.0],  # cmd_dist = 2.0m
                [0.00, 0.0, 0.0],  # standing: cmd_dist = 0.0m
                [0.20, 0.0, 0.0],  # cmd_dist = 2.0m
            ], dtype=torch.float32)
            self.command_manager = _DummyCmdMgr(cmds)
            self.termination_manager = _DummyTermMgr(
                terminated=torch.tensor([True, True, True, True, True, True]),
                time_outs=torch.tensor([False, False, True, True, True, False]),
            )
            self.extras = {}

    env = _DummyEnv()
    env_ids = torch.arange(6, dtype=torch.long)
    zoe_terrain_curriculum(env, env_ids)

    # Expected levels:
    # env 0: 0 -> clamped at 0
    # env 1: 5 -> demoted to 4 (even though world pos is (-32, 28), 42.5m from world origin!)
    # env 2: 4 -> promoted to 5
    # env 3: 9 -> clamped at 9 (did NOT wrap to 0)
    # env 4: 3 -> kept at 3 (standing env did NOT promote)
    # env 5: 2 -> kept at 2 (terminated early did NOT promote)
    assert torch.equal(env.scene.terrain.terrain_levels, torch.tensor([0, 4, 5, 9, 3, 2], dtype=torch.long))

    # Check per-terrain-type logs in extras["log"]
    log = env.extras["log"]
    assert "Terrain/Mean_Level" in log
    for t_name in ["flat", "rough", "slope_up", "slope_down", "stairs_up", "boxes"]:
        assert f"Terrain/{t_name}/mean_level" in log
        assert f"Terrain/{t_name}/promotion_rate" in log
        assert f"Terrain/{t_name}/demotion_rate" in log
        assert f"Terrain/{t_name}/survival" in log

    assert log["Terrain/rough/demotion_rate"] == 1.0
    assert log["Terrain/slope_up/promotion_rate"] == 1.0
    assert log["Terrain/stairs_up/promotion_rate"] == 0.0
    assert log["Terrain/stairs_up/demotion_rate"] == 0.0
