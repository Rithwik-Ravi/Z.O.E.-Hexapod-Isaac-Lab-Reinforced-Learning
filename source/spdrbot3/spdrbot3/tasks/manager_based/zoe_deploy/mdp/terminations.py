# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""MDP Termination terms for Z.O.E. Hexapod deployable task."""

import torch
try:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg
except Exception:
    ManagerBasedRLEnv = object
    SceneEntityCfg = object


def invalid_state_nan(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = None) -> torch.Tensor:
    """Terminate episode if NaN or Inf is detected in root or joint states, keeping a counter."""
    name = getattr(asset_cfg, "name", "robot") if asset_cfg is not None else "robot"
    robot = env.scene[name]
    root_has_nan = torch.isnan(robot.data.root_state_w).any(dim=1) | torch.isinf(robot.data.root_state_w).any(dim=1)
    joint_has_nan = torch.isnan(robot.data.joint_pos).any(dim=1) | torch.isinf(robot.data.joint_pos).any(dim=1)

    is_invalid = root_has_nan | joint_has_nan

    nan_count = torch.sum(is_invalid).item()
    if "nan_counter" not in env.extras:
        env.extras["nan_counter"] = 0
    env.extras["nan_counter"] += nan_count

    return is_invalid
