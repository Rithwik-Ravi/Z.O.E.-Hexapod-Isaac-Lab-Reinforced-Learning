# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""MDP Event terms for domain randomization in Z.O.E. Hexapod deployable task."""

import torch
try:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import SceneEntityCfg
except Exception:
    ManagerBasedEnv = object
    SceneEntityCfg = object


from ..params import ZOE_PARAMS
from .rewards import _get_scene_entity, ground_height_from_raycaster


def reset_root_near_ground(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = None,
):
    """Reset robot root state near ground (avoiding unrealistic >1m drops)."""
    name = getattr(asset_cfg, "name", "robot") if asset_cfg is not None else "robot"
    robot = env.scene[name]
    num_resets = len(env_ids)

    root_states = robot.data.default_root_state[env_ids].clone()

    x_rand = torch.FloatTensor(num_resets).uniform_(*pose_range.get("x", (0.0, 0.0))).to(env.device)
    y_rand = torch.FloatTensor(num_resets).uniform_(*pose_range.get("y", (0.0, 0.0))).to(env.device)
    z_rand = torch.FloatTensor(num_resets).uniform_(*pose_range.get("z", ZOE_PARAMS.SPAWN_Z_RANGE)).to(env.device)

    root_states[:, 0] = env.scene.env_origins[env_ids, 0] + x_rand
    root_states[:, 1] = env.scene.env_origins[env_ids, 1] + y_rand
    root_states[:, 2] = env.scene.env_origins[env_ids, 2] + z_rand

    yaw_rand = torch.FloatTensor(num_resets).uniform_(*pose_range.get("yaw", (-3.14, 3.14))).to(env.device)
    half_yaw = yaw_rand * 0.5
    # Compose world yaw (cos(psi/2), 0, 0, sin(psi/2)) with upright 180-deg X-roll (0, 1, 0, 0)
    root_states[:, 3] = 0.0
    root_states[:, 4] = torch.cos(half_yaw)
    root_states[:, 5] = torch.sin(half_yaw)
    root_states[:, 6] = 0.0

    root_states[:, 7] = torch.FloatTensor(num_resets).uniform_(*velocity_range.get("x", (0.0, 0.0))).to(env.device)
    root_states[:, 8] = torch.FloatTensor(num_resets).uniform_(*velocity_range.get("y", (0.0, 0.0))).to(env.device)
    root_states[:, 9] = torch.FloatTensor(num_resets).uniform_(*velocity_range.get("z", (0.0, 0.0))).to(env.device)

    robot.write_root_state_to_sim(root_states, env_ids=env_ids)

    if not hasattr(env, "_last_reset_step") or env._last_reset_step is None:
        env._last_reset_step = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    env._last_reset_step[env_ids] = int(getattr(env, "common_step_counter", 0))

    scanner = _get_scene_entity(env.scene, "height_scanner")
    if scanner is not None:
        try:
            if hasattr(scanner, "update"):
                scanner.update(dt=0.0, force_recompute=True)
            ground_z = ground_height_from_raycaster(env, sensor_name="height_scanner")
            if ground_z is not None and torch.isfinite(ground_z[env_ids]).all():
                root_states[:, 2] = ground_z[env_ids] + z_rand
                robot.write_root_state_to_sim(root_states, env_ids=env_ids)

    if not hasattr(env, "_last_reset_step") or env._last_reset_step is None:
        env._last_reset_step = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    env._last_reset_step[env_ids] = int(getattr(env, "common_step_counter", 0))
                if hasattr(scanner, "update"):
                    scanner.update(dt=0.0, force_recompute=True)
        except Exception:
            pass


def push_robot_base(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = None,
):
    """Random velocity impulse on robot base."""
    name = getattr(asset_cfg, "name", "robot") if asset_cfg is not None else "robot"
    robot = env.scene[name]
    num_envs = len(env_ids)

    vel_x = torch.FloatTensor(num_envs).uniform_(*velocity_range.get("x", (-0.3, 0.3))).to(env.device)
    vel_y = torch.FloatTensor(num_envs).uniform_(*velocity_range.get("y", (-0.3, 0.3))).to(env.device)

    current_root_state = robot.data.root_state_w[env_ids].clone()
    current_root_state[:, 7] += vel_x
    current_root_state[:, 8] += vel_y

    robot.write_root_state_to_sim(current_root_state, env_ids=env_ids)


def randomize_joint_calibration_offsets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    offset_range_rad: tuple[float, float] = (-0.0523, 0.0523),
    asset_cfg: SceneEntityCfg = None,
):
    """Randomize joint calibration zero-offsets per environment reset."""
    name = getattr(asset_cfg, "name", "robot") if asset_cfg is not None else "robot"
    robot = env.scene[name]
    num_resets = len(env_ids)
    rand_offsets = torch.FloatTensor(num_resets, robot.num_joints).uniform_(*offset_range_rad).to(env.device)

    if not hasattr(env, "joint_calibration_offsets"):
        env.joint_calibration_offsets = torch.zeros((env.num_envs, robot.num_joints), device=env.device)
    env.joint_calibration_offsets[env_ids] = rand_offsets
