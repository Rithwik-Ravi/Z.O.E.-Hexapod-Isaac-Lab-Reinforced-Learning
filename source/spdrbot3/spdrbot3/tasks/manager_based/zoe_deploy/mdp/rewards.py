# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""MDP Reward terms for Z.O.E. Hexapod deployable task."""

import torch
try:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg
except Exception:
    ManagerBasedRLEnv = object
    SceneEntityCfg = object


def fewer_than_3_feet_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize having fewer than 3 feet in contact with ground: -0.5 * (3 - n)^2 for n < 3."""
    contact_sensor = env.scene[sensor_cfg.name]
    net_forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    in_contact = (net_forces_z > 1.0).float()
    num_feet_in_contact = torch.sum(in_contact, dim=1)
    def_count = torch.clamp(3.0 - num_feet_in_contact, min=0.0)
    return torch.square(def_count)


def feet_air_time_gated(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.15,
) -> torch.Tensor:
    """Reward air time of feet, gated by command velocity magnitude."""
    contact_sensor = env.scene[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    air_time_reward = torch.sum((last_air_time - threshold) * first_contact.float(), dim=1)

    cmd = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(cmd[:, :2], dim=-1)
    gated_reward = air_time_reward * (cmd_norm > 0.05).float()
    return gated_reward


def foot_clearance_relative(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize feet that go significantly above or below clearance target relative to root/ground."""
    robot = env.scene[asset_cfg.name]
    foot_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]
    root_pos_z = robot.data.root_pos_w[:, 2].unsqueeze(1)
    ground_clearance = foot_pos_w[:, :, 2]
    return torch.sum(torch.square(torch.clamp(ground_clearance - target_height, min=0.0)), dim=1)


def undesired_contacts(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Penalize undesired contacts (thigh, coxa, or base touching ground)."""
    contact_sensor = env.scene[sensor_cfg.name]
    net_forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    is_contact = (net_forces_z > threshold).float()
    return torch.sum(is_contact, dim=1)


def base_height_l2(env: ManagerBasedRLEnv, target_height: float, asset_cfg: SceneEntityCfg = None) -> torch.Tensor:
    """Penalize base height error relative to target standing height: (root_z - target_height)^2."""
    if asset_cfg is None and hasattr(SceneEntityCfg, "__call__"):
        asset_cfg = SceneEntityCfg("robot")
    name = getattr(asset_cfg, "name", "robot")
    robot = env.scene[name]
    root_z = robot.data.root_pos_w[:, 2]
    return torch.square(root_z - target_height)


def base_height_shaped(
    env: ManagerBasedRLEnv,
    target_height: float = 0.155,
    min_height: float = 0.148,
    std: float = 0.015,
    barrier_quad_weight: float = 150.0,
    barrier_lin_weight: float = 10.0,
    asset_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """Base height shaping: positive exponential attractor near target height plus steep barrier below min_height."""
    if asset_cfg is None and hasattr(SceneEntityCfg, "__call__"):
        asset_cfg = SceneEntityCfg("robot")
    name = getattr(asset_cfg, "name", "robot")
    robot = env.scene[name]
    root_z = robot.data.root_pos_w[:, 2]

    # Positive attractor (1.0 when at target_height)
    attractor = torch.exp(-torch.square((root_z - target_height) / std))

    # One-sided barrier below min_height
    deficit = torch.clamp(min_height - root_z, min=0.0)
    barrier = barrier_lin_weight * deficit + barrier_quad_weight * torch.square(deficit)

    return attractor - barrier


def metric_base_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = None) -> torch.Tensor:
    """Metric monitor: Mean base height [m]."""
    if asset_cfg is None and hasattr(SceneEntityCfg, "__call__"):
        asset_cfg = SceneEntityCfg("robot")
    name = getattr(asset_cfg, "name", "robot")
    robot = env.scene[name]
    return robot.data.root_pos_w[:, 2]


def metric_non_foot_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Metric monitor: Non-foot contact fraction [0 or 1 per env]."""
    contact_sensor = env.scene[sensor_cfg.name]
    net_forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    return (net_forces_z > 1.0).any(dim=1).float()


def metric_torque_saturation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = None, threshold: float = 2.45) -> torch.Tensor:
    """Metric monitor: Joint torque saturation fraction [0 to 1]."""
    if asset_cfg is None and hasattr(SceneEntityCfg, "__call__"):
        asset_cfg = SceneEntityCfg("robot")
    name = getattr(asset_cfg, "name", "robot")
    robot = env.scene[name]
    torques = torch.abs(robot.data.applied_torque)
    return (torques >= threshold).float().mean(dim=-1)


def metric_feet_down(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Metric monitor: Mean feet in contact [0 to 6]."""
    contact_sensor = env.scene[sensor_cfg.name]
    net_forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    return (net_forces_z > 1.0).sum(dim=1).float()
