# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for action delay buffer, rate limiter, and joint zero-offset semantics."""

import math
import pytest
import torch

from spdrbot3.tasks.manager_based.zoe_deploy.mdp.actions import (
    DelayedRateLimitedJointPositionAction,
    DelayedRateLimitedJointPositionActionCfg,
)


class MockAsset:
    def __init__(self, num_envs=4, num_joints=18, device="cpu"):
        self.num_joints = num_joints
        self.default_joint_pos = torch.zeros((num_envs, num_joints), device=device)
        self.data = self

    def find_joints(self, joint_names, preserve_order=False):
        return list(range(self.num_joints)), [f"joint_{i}" for i in range(self.num_joints)]


class MockEnv:
    def __init__(self, num_envs=4, num_joints=18, device="cpu"):
        self.num_envs = num_envs
        self.device = device
        self.scene = {"robot": MockAsset(num_envs, num_joints, device)}
        self.joint_calibration_offsets = torch.zeros((num_envs, num_joints), device=device)


def test_action_delay_buffer_pure_delays():
    """Test that delay buffer accurately delays commands by 0, 1, and 2 policy steps."""
    device = "cpu"
    num_envs = 3
    num_joints = 18

    mock_env = MockEnv(num_envs=num_envs, num_joints=num_joints, device=device)
    cfg = DelayedRateLimitedJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        offset=0.0,
        max_delay_steps=2,
        rate_limit_rad_per_step=100.0,  # disable rate limit for pure delay test
    )

    action_term = DelayedRateLimitedJointPositionAction(cfg, mock_env)
    action_term.reset([0, 1, 2])

    # Assign deterministic delays: env 0 -> 0 steps, env 1 -> 1 step, env 2 -> 2 steps
    action_term.delay_steps = torch.tensor([0, 1, 2], dtype=torch.long, device=device)

    # Step 1: command 1.0 across all envs
    a1 = torch.full((num_envs, num_joints), 1.0, device=device)
    action_term.process_actions(a1)
    # env 0 (d=0) should see 1.0 immediately
    assert torch.allclose(action_term.processed_actions[0], torch.tensor(1.0))
    # env 1 (d=1) should see initial 0.0
    assert torch.allclose(action_term.processed_actions[1], torch.tensor(0.0))
    # env 2 (d=2) should see initial 0.0
    assert torch.allclose(action_term.processed_actions[2], torch.tensor(0.0))

    # Step 2: command 2.0 across all envs
    a2 = torch.full((num_envs, num_joints), 2.0, device=device)
    action_term.process_actions(a2)
    # env 0 (d=0) should see 2.0 immediately
    assert torch.allclose(action_term.processed_actions[0], torch.tensor(2.0))
    # env 1 (d=1) should see previous 1.0
    assert torch.allclose(action_term.processed_actions[1], torch.tensor(1.0))
    # env 2 (d=2) should still see initial 0.0
    assert torch.allclose(action_term.processed_actions[2], torch.tensor(0.0))

    # Step 3: command 3.0 across all envs
    a3 = torch.full((num_envs, num_joints), 3.0, device=device)
    action_term.process_actions(a3)
    # env 0 (d=0) should see 3.0
    assert torch.allclose(action_term.processed_actions[0], torch.tensor(3.0))
    # env 1 (d=1) should see previous 2.0
    assert torch.allclose(action_term.processed_actions[1], torch.tensor(2.0))
    # env 2 (d=2) should see a1 (1.0)
    assert torch.allclose(action_term.processed_actions[2], torch.tensor(1.0))

    # Raw actions must always equal the current un-delayed policy action
    assert torch.allclose(action_term.raw_actions, a3)


def test_target_rate_limiter_clamping():
    """Test that target rate limiter clamps per-step joint position change to 6 deg (0.1047 rad)."""
    device = "cpu"
    num_envs = 1
    num_joints = 18
    rate_limit = math.radians(6.0)

    mock_env = MockEnv(num_envs=num_envs, num_joints=num_joints, device=device)
    cfg = DelayedRateLimitedJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        offset=0.0,
        max_delay_steps=0,
        fixed_delay_steps=0,
        rate_limit_rad_per_step=rate_limit,
    )

    action_term = DelayedRateLimitedJointPositionAction(cfg, mock_env)
    action_term.reset([0])

    # Command a huge step of +1.0 rad (approx 57 deg)
    large_cmd = torch.full((num_envs, num_joints), 1.0, device=device)
    action_term.process_actions(large_cmd)

    # Processed action must be clamped to exactly 0.0 + rate_limit
    expected = torch.full((num_envs, num_joints), rate_limit, device=device)
    assert torch.allclose(action_term.processed_actions, expected, atol=1e-5)
    assert action_term.clip_fraction == 1.0

    # Next step: command 1.0 again -> advances by another rate_limit
    action_term.process_actions(large_cmd)
    expected_step2 = torch.full((num_envs, num_joints), 2 * rate_limit, device=device)
    assert torch.allclose(action_term.processed_actions, expected_step2, atol=1e-5)


def test_joint_calibration_offset_addition():
    """Test that persistent joint zero-offset delta is added to the commanded targets."""
    device = "cpu"
    num_envs = 2
    num_joints = 18

    mock_env = MockEnv(num_envs=num_envs, num_joints=num_joints, device=device)
    calib_offset = torch.full((num_envs, num_joints), 0.05, device=device)
    mock_env.joint_calibration_offsets = calib_offset

    cfg = DelayedRateLimitedJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        offset=0.0,
        max_delay_steps=0,
        fixed_delay_steps=0,
        rate_limit_rad_per_step=100.0,
    )

    action_term = DelayedRateLimitedJointPositionAction(cfg, mock_env)
    action_term.reset([0, 1])

    cmd = torch.full((num_envs, num_joints), 0.2, device=device)
    action_term.process_actions(cmd)

    # Commanded target should include the calibration offset: 0.2 + 0.05 = 0.25
    expected = torch.full((num_envs, num_joints), 0.25, device=device)
    assert torch.allclose(action_term.processed_actions, expected, atol=1e-5)
