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
