# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""MDP Curriculum terms for Z.O.E. Hexapod terrain task."""

import torch
try:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg
except Exception:
    ManagerBasedRLEnv = object
    SceneEntityCfg = object


def compute_curriculum_step(
    distance_walked: float | torch.Tensor,
    commanded_distance: float | torch.Tensor,
    was_terminated_early: bool | torch.Tensor,
    up_threshold: float = 0.75,
    down_threshold: float = 0.35,
):
    """Pure-Python / PyTorch core logic for terrain curriculum updates.

    - Move UP if distance_walked >= up_threshold * commanded_distance AND not terminated early
    - Move DOWN if distance_walked < down_threshold * commanded_distance
    - Otherwise KEEP current level
    """
    if isinstance(distance_walked, (float, int)):
        if was_terminated_early:
            move_up = False
        else:
            move_up = distance_walked >= up_threshold * commanded_distance
        move_down = distance_walked < down_threshold * commanded_distance
        return move_up, move_down

    target_dist = up_threshold * commanded_distance
    min_dist = down_threshold * commanded_distance

    move_up = (distance_walked >= target_dist) & (~was_terminated_early)
    move_down = distance_walked < min_dist
    return move_up, move_down


def zoe_terrain_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = None,
):
    """Curriculum manager function invoked on environment resets."""
    if not hasattr(env.scene.terrain, "update_env_origins"):
        return

    name = getattr(asset_cfg, "name", "robot") if asset_cfg is not None else "robot"
    robot = env.scene[name]

    episode_length = env.episode_length_buf[env_ids].float() * env.step_dt
    vel_cmd = env.command_manager.get_command("base_velocity")[env_ids]
    cmd_speed = torch.norm(vel_cmd[:, :2], dim=-1)
    commanded_distance = cmd_speed * episode_length

    root_pos = robot.data.root_pos_w[env_ids, :2]
    distance_walked = torch.norm(root_pos, dim=-1)

    was_terminated_early = env.termination_manager.terminated[env_ids] & (~env.termination_manager.time_outs[env_ids])

    move_up, move_down = compute_curriculum_step(distance_walked, commanded_distance, was_terminated_early)
    env.scene.terrain.update_env_origins(env_ids, move_up=move_up, move_down=move_down)

    # Per-terrain-type logging into extras["log"]
    if hasattr(env.scene.terrain, "terrain_types") and hasattr(env.scene.terrain, "terrain_levels"):
        terrain_types = env.scene.terrain.terrain_types[env_ids]
        terrain_levels = env.scene.terrain.terrain_levels[env_ids]

        if "log" not in env.extras:
            env.extras["log"] = {}

        # Log mean level across reset envs
        env.extras["log"]["Terrain/Mean_Level"] = torch.mean(env.scene.terrain.terrain_levels.float()).item()
