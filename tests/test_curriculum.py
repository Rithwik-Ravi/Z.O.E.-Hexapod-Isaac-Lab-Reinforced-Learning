# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Z.O.E. Hexapod curriculum logic (pure-Python)."""

import pytest
import torch
from spdrbot3.tasks.manager_based.zoe_deploy.mdp.curriculum import compute_curriculum_step


def test_curriculum_step_scalar():
    # Success case: walked >= 0.75 * commanded, not terminated early
    move_up, move_down = compute_curriculum_step(distance_walked=0.8, commanded_distance=1.0, was_terminated_early=False)
    assert move_up is True
    assert move_down is False

    # Early termination case: walked >= 0.75 * commanded but terminated early
    move_up, move_down = compute_curriculum_step(distance_walked=0.8, commanded_distance=1.0, was_terminated_early=True)
    assert move_up is False
    assert move_down is False

    # Failure case: walked < 0.35 * commanded
    move_up, move_down = compute_curriculum_step(distance_walked=0.2, commanded_distance=1.0, was_terminated_early=False)
    assert move_up is False
    assert move_down is True


def test_curriculum_step_tensor():
    dist_walked = torch.tensor([0.8, 0.8, 0.2])
    cmd_dist = torch.tensor([1.0, 1.0, 1.0])
    early_term = torch.tensor([False, True, False])

    move_up, move_down = compute_curriculum_step(dist_walked, cmd_dist, early_term)
    assert torch.equal(move_up, torch.tensor([True, False, False]))
    assert torch.equal(move_down, torch.tensor([False, False, True]))
