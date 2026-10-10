# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Z.O.E. Hexapod params module (pure-Python)."""

import pytest
from spdrbot3.tasks.manager_based.zoe_deploy.params import ZOE_PARAMS, ZoeParams


def test_params_structure():
    assert isinstance(ZOE_PARAMS, ZoeParams)
    assert ZOE_PARAMS.commands.lin_vel_x_range[0] < ZOE_PARAMS.commands.lin_vel_x_range[1]
    assert ZOE_PARAMS.commands.lin_vel_y_range[0] < ZOE_PARAMS.commands.lin_vel_y_range[1]
    assert ZOE_PARAMS.commands.ang_vel_z_range[0] < ZOE_PARAMS.commands.ang_vel_z_range[1]


def test_params_bounds():
    assert ZOE_PARAMS.obs_scale.base_ang_vel_scale > 0.0
    assert ZOE_PARAMS.obs_scale.joint_vel_scale > 0.0
    assert ZOE_PARAMS.events.max_action_delay_steps >= 0
    assert ZOE_PARAMS.rewards.termination_penalty < 0.0
    assert ZOE_PARAMS.rewards.fewer_than_3_feet_contact < 0.0
