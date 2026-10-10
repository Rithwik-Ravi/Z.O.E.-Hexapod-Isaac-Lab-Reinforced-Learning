# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""MDP Observation terms for Z.O.E. Hexapod deployable task."""

import torch
try:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg
except Exception:
    ManagerBasedRLEnv = object
    SceneEntityCfg = object


def foot_contact_flags(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Privileged observation: Binary foot contact flags [N, 6]."""
    contact_sensor = env.scene[sensor_cfg.name]
    net_forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    return (net_forces_z > 1.0).float()


def applied_joint_torques(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = None) -> torch.Tensor:
    """Privileged observation: Applied joint torques [N, 18]."""
    if asset_cfg is None and hasattr(SceneEntityCfg, "__call__"):
        asset_cfg = SceneEntityCfg("robot")
    name = getattr(asset_cfg, "name", "robot")
    robot = env.scene[name]
    return robot.data.applied_torque.clone()


def base_lin_vel_privileged(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = None) -> torch.Tensor:
    """Privileged observation: Base linear velocity [N, 3]."""
    if asset_cfg is None and hasattr(SceneEntityCfg, "__call__"):
        asset_cfg = SceneEntityCfg("robot")
    name = getattr(asset_cfg, "name", "robot")
    robot = env.scene[name]
    return robot.data.root_lin_vel_b.clone()


def base_height_privileged(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = None) -> torch.Tensor:
    """Privileged observation: Base height above ground [N, 1]."""
    if asset_cfg is None and hasattr(SceneEntityCfg, "__call__"):
        asset_cfg = SceneEntityCfg("robot")
    name = getattr(asset_cfg, "name", "robot")
    robot = env.scene[name]
    return robot.data.root_pos_w[:, 2].unsqueeze(1)


def belly_contact_flag(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Privileged observation: Belly/coxa/thigh ground contact flag [N, 1]."""
    contact_sensor = env.scene[sensor_cfg.name]
    net_forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    return (net_forces_z > 1.0).any(dim=1, keepdim=True).float()
