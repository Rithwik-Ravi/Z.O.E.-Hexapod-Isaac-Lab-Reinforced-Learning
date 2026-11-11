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

import math
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


def test_net_displacement_oscillating_robot_gives_zero_progress():
    """Verify that an oscillating robot (e.g. creeper rocking back and forth) gives ~0 progress.

    Scenario:
    - Commanded forward at 0.20 m/s for 10s (500 steps, dt=0.02s) -> commanded travel = 2.0 m
    - Robot rocks back and forth: x(t) = 0.05 * sin(2 * pi * t / 2.0)
    - Old step-integrated metric: sum max(0, v) dt accumulated all forward halves -> inflated ~0.50m!
    - New net displacement metric: actual displacement at end = 0.0m -> progress = 0.0 / 2.0 = 0.0!
    - Curriculum must DEMOTE (ratio 0.0 < 0.35 with commanded travel 2.0m >= 0.5m).
    """
    dt = 0.02
    num_steps = 500
    times = torch.linspace(0, num_steps * dt, num_steps)
    v_cmd = 0.20

    # Old step-integrated calculation
    vx_robot = 0.05 * (2 * 3.14159 / 2.0) * torch.cos(2 * 3.14159 * times / 2.0)
    old_achieved = torch.sum(torch.clamp(vx_robot, min=0.0)) * dt
    old_ratio = (old_achieved / (v_cmd * num_steps * dt)).item()
    # Old metric gave inflated positive progress
    assert old_ratio > 0.15, f"Expected old metric to be inflated, got {old_ratio}"

    # New net displacement in world frame
    cmd_disp_w = torch.tensor([v_cmd * num_steps * dt, 0.0])  # [2.0, 0.0]
    cmd_norm = torch.norm(cmd_disp_w).item()
    assert abs(cmd_norm - 2.0) < 1e-4

    # Robot actual position at end of oscillation
    pos_start_w = torch.tensor([0.0, 0.0])
    pos_end_w = torch.tensor([0.05 * math.sin(2 * 3.14159 * (num_steps * dt) / 2.0), 0.0])
    actual_disp_w = pos_end_w - pos_start_w

    cmd_unit_w = cmd_disp_w / cmd_norm
    new_achieved = torch.dot(actual_disp_w, cmd_unit_w).item()
    new_progress = new_achieved / cmd_norm

    # New metric strictly reflects zero net progress
    assert abs(new_progress) < 0.01, f"Expected ~0 net progress, got {new_progress}"

    # Curriculum demotes the oscillating creeper
    move_up, move_down = compute_curriculum_step(
        commanded_travel=cmd_norm, achieved_travel=new_achieved, was_terminated_early=False
    )
    assert move_up is False
    assert move_down is True, "Oscillating robot must be demoted!"


def test_net_displacement_changing_commands():
    """Verify that net displacement progress correctly handles changing / multi-directional commands.

    Scenario:
    - Phase 1 (5s): commanded +0.20 m/s along X -> cmd_disp = [1.0, 0.0]
    - Phase 2 (5s): commanded +0.20 m/s along Y -> cmd_disp = [1.0, 1.0]
    Total commanded displacement = [1.0, 1.0], magnitude = sqrt(2) = 1.4142 m
    Robot tracks well: actual end position = [0.95, 0.95]
    Progress = ([0.95, 0.95] . [1/sqrt(2), 1/sqrt(2)]) / sqrt(2) = 0.95 >= 0.75 -> promote!
    """
    cmd_disp_w = torch.tensor([1.0, 1.0])
    cmd_norm = torch.norm(cmd_disp_w).item()
    cmd_unit_w = cmd_disp_w / cmd_norm

    actual_disp_w = torch.tensor([0.95, 0.95])
    achieved = torch.dot(actual_disp_w, cmd_unit_w).item()
    progress = achieved / cmd_norm

    assert abs(progress - 0.95) < 1e-4

    move_up, move_down = compute_curriculum_step(
        commanded_travel=cmd_norm, achieved_travel=achieved, was_terminated_early=False
    )
    assert move_up is True
    assert move_down is False


def test_net_displacement_command_reversals():
    """Verify net displacement behavior under forward then backward commands.

    Case A: Command 1.0m forward (+X) then 0.4m backward (-X).
    Net commanded displacement = 0.6m >= 0.5m.
    Robot reaches 0.57m -> progress = 0.57 / 0.6 = 0.95 >= 0.75 -> promote!

    Case B: Command 1.0m forward (+X) then 1.0m backward (-X).
    Net commanded displacement = 0.0m < 0.5m -> no decision (keep level).
    """
    # Case A: Partial reversal
    cmd_disp_a = torch.tensor([0.60, 0.0])
    cmd_norm_a = torch.norm(cmd_disp_a).item()
    actual_disp_a = torch.tensor([0.57, 0.0])
    achieved_a = torch.dot(actual_disp_a, cmd_disp_a / cmd_norm_a).item()

    move_up_a, move_down_a = compute_curriculum_step(
        commanded_travel=cmd_norm_a, achieved_travel=achieved_a, was_terminated_early=False
    )
    assert move_up_a is True
    assert move_down_a is False

    # Case B: Complete reversal (net 0 commanded)
    cmd_norm_b = 0.0
    actual_b = 0.0
    move_up_b, move_down_b = compute_curriculum_step(
        commanded_travel=cmd_norm_b, achieved_travel=actual_b, was_terminated_early=False
    )
    assert move_up_b is False
    assert move_down_b is False  # No decision below 0.5m commanded!


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


def test_penalty_scale_ramp():
    """Verify penalty_scale_ramp multiplies penalties by 0.1 at iter 0 to 1.0 at iter 800 (Task 4)."""
    from spdrbot3.tasks.manager_based.zoe_deploy.mdp.curriculum import penalty_scale_ramp

    class _DummyTermCfg:
        def __init__(self, weight):
            self.weight = weight

    class _DummyRewardManager:
        def __init__(self):
            self._terms = {
                "torque_limit": _DummyTermCfg(-2.0),
                "fewer_than_4_feet_contact": _DummyTermCfg(-0.05),
                "dof_acc_l2": _DummyTermCfg(-2.5e-7),
                "action_rate_l2": _DummyTermCfg(-0.005),
                "stand_still_posture": _DummyTermCfg(-0.5),
                "joint_vel_limit": _DummyTermCfg(-0.01),
            }
            self._term_names = list(self._terms.keys())

        def get_term_cfg(self, name):
            return self._terms[name]

        def set_term_cfg(self, name, cfg):
            self._terms[name] = cfg

    class _DummyEnv:
        def __init__(self):
            self.reward_manager = _DummyRewardManager()
            self.common_step_counter = 0
            self.extras = {}

    env = _DummyEnv()

    # Step 0 (iteration 0): scale must be 0.10
    env.common_step_counter = 0
    scale_0 = penalty_scale_ramp(env, None)
    assert abs(scale_0 - 0.10) < 1e-4
    assert abs(env.reward_manager.get_term_cfg("torque_limit").weight - (-0.20)) < 1e-4
    assert abs(env.reward_manager.get_term_cfg("fewer_than_4_feet_contact").weight - (-0.005)) < 1e-4
    assert abs(env.extras["log"]["Curriculum/penalty_scale"] - 0.10) < 1e-4

    # Step 19200 (iteration 400 = 19200 / 48): scale must be 0.10 + 0.90 * (400 / 800) = 0.55
    env.common_step_counter = 19200
    scale_400 = penalty_scale_ramp(env, None)
    assert abs(scale_400 - 0.55) < 1e-4
    assert abs(env.reward_manager.get_term_cfg("torque_limit").weight - (-1.10)) < 1e-4
    assert abs(env.extras["log"]["Curriculum/penalty_scale"] - 0.55) < 1e-4

    # Step 38400 (iteration 800 = 38400 / 48): scale must be 1.00
    env.common_step_counter = 38400
    scale_800 = penalty_scale_ramp(env, None)
    assert abs(scale_800 - 1.00) < 1e-4
    assert abs(env.reward_manager.get_term_cfg("torque_limit").weight - (-2.00)) < 1e-4
    assert abs(env.extras["log"]["Curriculum/penalty_scale"] - 1.00) < 1e-4

    # Step 48000 (iteration 1000): clamped at 1.00
    env.common_step_counter = 48000
    scale_1000 = penalty_scale_ramp(env, None)
    assert abs(scale_1000 - 1.00) < 1e-4
    assert abs(env.reward_manager.get_term_cfg("torque_limit").weight - (-2.00)) < 1e-4


def test_flat_env_wires_penalty_scale_ramp():
    """Verify env_cfg_flat.py wires penalty_scale = CurrTerm(func=zoe_mdp.penalty_scale_ramp) and matches CurriculumManager signature."""
    import ast
    import inspect
    import os
    from spdrbot3.tasks.manager_based.zoe_deploy.mdp.curriculum import penalty_scale_ramp

    # Mirror Isaac Lab ManagerBase._resolve_common_term_cfg(..., min_argc=2)
    args = inspect.signature(penalty_scale_ramp).parameters
    args_with_defaults = [arg for arg in args if args[arg].default is not inspect.Parameter.empty]
    args_without_defaults = [arg for arg in args if args[arg].default is inspect.Parameter.empty]
    ordered = args_without_defaults + args_with_defaults
    assert set(ordered[2:]) == set(args_with_defaults)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg_path = os.path.join(
        repo_root, "source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/env_cfg_flat.py"
    )
    with open(cfg_path, "r") as f:
        source = f.read()

    assert "penalty_scale_ramp" in source
    tree = ast.parse(source)
    flat_cls = None
    curr_cls = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "FlatCurriculumCfg":
            curr_cls = node
        if isinstance(node, ast.ClassDef) and node.name == "ZoeHexapodFlatDeployEnvCfg":
            flat_cls = node

    assert curr_cls is not None, "FlatCurriculumCfg not found in env_cfg_flat.py"
    assert flat_cls is not None, "ZoeHexapodFlatDeployEnvCfg not found in env_cfg_flat.py"
    curr_src = ast.get_source_segment(source, curr_cls)
    flat_src = ast.get_source_segment(source, flat_cls)
    assert "penalty_scale" in curr_src and "penalty_scale_ramp" in curr_src
    assert "curriculum: FlatCurriculumCfg" in flat_src


