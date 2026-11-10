# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Z.O.E. Hexapod curriculum logic (Part A).

Verifies:
1. Step-integrated progress accounting (commanded_travel, achieved_travel, progress_ratio).
2. Command reversals: forward then backward moves net 0 displacement from origin,
   but step-integrated achieved travel correctly measures progress and promotes.
3. Standing envs: commanded travel < 0.5 m keeps level (neither promote nor demote if survived).
4. Early termination: robot falling over always demotes.
5. Level clamping: [0, max_level] for Group T (0..9), [0, 3] for Group S (capped at 3).
6. Extras logging: mean level, promotion, demotion, survival, progress_ratio per terrain type.
"""

import pytest
import torch
from spdrbot3.tasks.manager_based.zoe_deploy.mdp.curriculum import (
    compute_curriculum_step,
    zoe_terrain_curriculum,
)


def test_curriculum_step_scalar():
    """Verify scalar curriculum step decisions."""
    # Success case: ratio = 0.8 / 1.0 = 0.8 >= 0.75, not terminated early -> promote
    move_up, move_down = compute_curriculum_step(
        commanded_travel=1.0, achieved_travel=0.8, was_terminated_early=False
    )
    assert move_up is True
    assert move_down is False

    # Early termination case: fell over -> always demotes
    move_up, move_down = compute_curriculum_step(
        commanded_travel=1.0, achieved_travel=0.8, was_terminated_early=True
    )
    assert move_up is False
    assert move_down is True

    # Failure case: ratio = 0.2 / 1.0 = 0.2 < 0.35, survived -> demote
    move_up, move_down = compute_curriculum_step(
        commanded_travel=1.0, achieved_travel=0.2, was_terminated_early=False
    )
    assert move_up is False
    assert move_down is True

    # Middle band (keep level): ratio = 0.5 / 1.0 = 0.5, survived -> keep
    move_up, move_down = compute_curriculum_step(
        commanded_travel=1.0, achieved_travel=0.5, was_terminated_early=False
    )
    assert move_up is False
    assert move_down is False


def test_curriculum_standing_envs_no_decision():
    """Standing envs (commanded travel < 0.5 m) must NOT promote or demote if survived."""
    # Scalar standing env (commanded = 0.0, achieved = 0.0, survived)
    move_up, move_down = compute_curriculum_step(
        commanded_travel=0.0, achieved_travel=0.0, was_terminated_early=False
    )
    assert move_up is False
    assert move_down is False

    # Short commanded distance < 0.50 m (e.g. 0.40 m, survived)
    move_up, move_down = compute_curriculum_step(
        commanded_travel=0.40, achieved_travel=0.38, was_terminated_early=False
    )
    assert move_up is False
    assert move_down is False

    # Standing env that fell over (was_terminated_early=True) -> demote
    move_up, move_down = compute_curriculum_step(
        commanded_travel=0.0, achieved_travel=0.0, was_terminated_early=True
    )
    assert move_up is False
    assert move_down is True

    # Tensor batch with standing and walking envs
    # env 0: standing, survived -> keep
    # env 1: short cmd (0.4m), survived -> keep
    # env 2: short cmd (0.4m), fell over -> demote
    # env 3: walking (cmd 1.0m, ach 0.8m), survived -> promote
    # env 4: walking (cmd 1.0m, ach 0.2m), survived -> demote
    cmd_travel = torch.tensor([0.0, 0.4, 0.4, 1.0, 1.0])
    ach_travel = torch.tensor([0.0, 0.35, 0.1, 0.8, 0.2])
    early_term = torch.tensor([False, False, True, False, False])

    move_up, move_down = compute_curriculum_step(
        commanded_travel=cmd_travel, achieved_travel=ach_travel, was_terminated_early=early_term
    )
    assert torch.equal(move_up, torch.tensor([False, False, False, True, False]))
    assert torch.equal(move_down, torch.tensor([False, False, True, False, True]))


def test_curriculum_command_reversals():
    """Verify that command reversals (forward then backward) work with step-integrated accounting.

    Scenario:
    - Steps 0-250 (5s): commanded +0.20 m/s vx -> commanded travel = 1.0 m, robot walks +0.95 m
    - Steps 250-500 (5s): commanded -0.20 m/s vx -> commanded travel = 1.0 m, robot walks -0.95 m
    Total commanded travel = 2.0 m
    Step-integrated achieved travel along command direction:
      sum max(0, v_xy . unit(cmd_xy)) dt = 0.95 + 0.95 = 1.90 m
    progress_ratio = 1.90 / 2.0 = 0.95 >= 0.75
    Net displacement from origin: |0.95 - 0.95| = 0.0 m (displacement metric would fail!)
    """
    cmd_travel = 2.0
    ach_travel = 1.90  # step-integrated
    net_origin_disp = 0.0  # old flawed metric

    # Old flawed metric would demote (0.0 < 0.35 * 2.0)
    move_up_old, move_down_old = compute_curriculum_step(
        commanded_travel=cmd_travel, achieved_travel=net_origin_disp, was_terminated_early=False
    )
    assert move_up_old is False
    assert move_down_old is True  # FAILED with old metric!

    # New step-integrated progress accounting correctly promotes
    move_up_new, move_down_new = compute_curriculum_step(
        commanded_travel=cmd_travel, achieved_travel=ach_travel, was_terminated_early=False
    )
    assert move_up_new is True
    assert move_down_new is False


def test_zoe_terrain_curriculum_far_from_world_origin_clamping_and_logging():
    """Verify zoe_terrain_curriculum:
    (a) Uses step-integrated command travel and achieved travel
    (b) Standing envs keep their level
    (c) Promotes >= 0.75x (not early term) and demotes < 0.35x or early term
    (d) Clamps levels to [0, max_level] for Group T and [0, 3] for Group S
    (e) Logs per-terrain-type mean_level, promotion_rate, demotion_rate, survival, progress_ratio
    """

    class _DummyTerrain:
        def __init__(self):
            self.max_terrain_level = 10  # levels 0..9
            # 6 envs:
            # env 0 (col 0 = flat, Group T): level 0, achieved 0.2m of 2.0m -> demote clamped at 0
            # env 1 (col 2 = rough, Group S): level 3 (Group S cap), achieved 1.8m of 2.0m -> promote clamped at 3
            # env 2 (col 4 = slope_up, Group T): level 4, achieved 1.8m of 2.0m -> promote to 5
            # env 3 (col 5 = slope_down, Group T): level 9 (max), achieved 1.9m of 2.0m -> promote clamped at 9
            # env 4 (col 6 = stairs_up, Group S): level 2, standing (cmd=0, ach=0) -> keep level 2
            # env 5 (col 9 = boxes, Group T): level 2, achieved 1.8m of 2.0m but early term -> demote to 1
            self.terrain_levels = torch.tensor([0, 3, 4, 9, 2, 2], dtype=torch.long)
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
            self.terrain_levels[env_ids] += 1 * move_up.long() - 1 * move_down.long()
            self.terrain_levels[env_ids] = torch.where(
                self.terrain_levels[env_ids] >= self.max_terrain_level,
                torch.zeros_like(self.terrain_levels[env_ids]),
                torch.clip(self.terrain_levels[env_ids], 0),
            )

    class _DummyRobotData:
        def __init__(self, origins):
            self.root_pos_w = origins.clone()

    class _DummyRobot:
        def __init__(self, origins):
            self.data = _DummyRobotData(origins)

    class _DummyCmd:
        def __init__(self):
            # Step-integrated travel
            self.commanded_travel = torch.tensor([2.0, 2.0, 2.0, 2.0, 0.0, 2.0], dtype=torch.float32)
            self.achieved_travel = torch.tensor([0.2, 1.8, 1.8, 1.9, 0.0, 1.8], dtype=torch.float32)
            # Group S mask: envs 1 and 4 are Group S
            self.is_group_s = torch.tensor([False, True, False, False, True, False], dtype=torch.bool)

    class _DummyCmdMgr:
        def __init__(self, cmd):
            self._cmd = cmd

        def get_term(self, name):
            if name == "base_velocity":
                return self._cmd
            return None

        def get_command(self, name):
            return torch.zeros((6, 3))

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
            robot = _DummyRobot(terrain.env_origins)
            self.scene = _DummyScene(terrain, robot)
            self.step_dt = 0.02
            self.episode_length_buf = torch.full((6,), 500, dtype=torch.long)
            cmd = _DummyCmd()
            self.command_manager = _DummyCmdMgr(cmd)
            self.is_group_s = cmd.is_group_s
            self.termination_manager = _DummyTermMgr(
                terminated=torch.tensor([True, True, True, True, True, True]),
                time_outs=torch.tensor([False, True, True, True, True, False]),  # envs 0 & 5 early terminated
            )
            self.extras = {}

    env = _DummyEnv()
    env_ids = torch.arange(6, dtype=torch.long)
    zoe_terrain_curriculum(env, env_ids)

    # Expected levels:
    # env 0: 0 -> clamped at 0
    # env 1: 3 -> promoted, but Group S capped at 3 -> clamped at 3!
    # env 2: 4 -> promoted to 5
    # env 3: 9 -> clamped at 9 (did NOT wrap to 0)
    # env 4: 2 -> kept at 2 (standing env did NOT promote or demote)
    # env 5: 2 -> demoted to 1 (early terminated!)
    assert torch.equal(
        env.scene.terrain.terrain_levels,
        torch.tensor([0, 3, 5, 9, 2, 1], dtype=torch.long),
    )

    # Check per-terrain-type logs in extras["log"]
    log = env.extras["log"]
    assert "Terrain/Mean_Level" in log
    assert "Terrain/Progress_Ratio" in log
    for t_name in ["flat", "rough", "slope_up", "slope_down", "stairs_up", "boxes"]:
        assert f"Terrain/{t_name}/mean_level" in log
        assert f"Terrain/{t_name}/promotion_rate" in log
        assert f"Terrain/{t_name}/demotion_rate" in log
        assert f"Terrain/{t_name}/survival" in log
        assert f"Terrain/{t_name}/progress_ratio" in log

    assert log["Terrain/flat/demotion_rate"] == 1.0
    assert log["Terrain/rough/promotion_rate"] == 1.0
    assert log["Terrain/slope_up/promotion_rate"] == 1.0
    assert log["Terrain/stairs_up/promotion_rate"] == 0.0
    assert log["Terrain/stairs_up/demotion_rate"] == 0.0
    assert log["Terrain/boxes/demotion_rate"] == 1.0


def test_max_init_terrain_level_not_none():
    """Fail if max_init_terrain_level is None in ZoeTerrainSceneCfg."""
    import ast
    import os

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg_path = os.path.join(
        repo_root, "source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/env_cfg_terrain.py"
    )
    with open(cfg_path, "r") as f:
        source = f.read()

    tree = ast.parse(source)
    found_kw = False
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "max_init_terrain_level":
            found_kw = True
            assert not (isinstance(node.value, ast.Constant) and node.value.value is None), (
                "max_init_terrain_level in ZoeTerrainSceneCfg must NOT be None!"
            )
            assert isinstance(node.value, ast.Constant) and node.value.value == 1, (
                f"Expected max_init_terrain_level == 1, got {ast.dump(node.value)}"
            )
    assert found_kw, "max_init_terrain_level keyword not found in env_cfg_terrain.py"
