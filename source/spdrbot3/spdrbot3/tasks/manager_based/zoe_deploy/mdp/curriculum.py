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

DEFAULT_SUB_TERRAIN_PROPORTIONS: dict[str, float] = {
    "flat": 0.10,
    "rough": 0.20,
    "slope_up": 0.12,
    "slope_down": 0.12,
    "stairs_up": 0.12,
    "stairs_down": 0.10,
    "boxes": 0.14,
}


def get_column_terrain_names(terrain=None, num_cols: int = 10) -> list[str]:
    """Return the sub-terrain name for each column index [0..num_cols-1] matching TerrainGenerator's cumsum rule."""
    props_dict = DEFAULT_SUB_TERRAIN_PROPORTIONS
    gen_cfg = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    if gen_cfg is not None and hasattr(gen_cfg, "sub_terrains") and gen_cfg.sub_terrains:
        props_dict = {k: float(v.proportion) for k, v in gen_cfg.sub_terrains.items()}
        num_cols = int(getattr(gen_cfg, "num_cols", num_cols))

    names = list(props_dict.keys())
    vals = [props_dict[k] for k in names]
    total = sum(vals) if sum(vals) > 0 else 1.0
    norm_props = [v / total for v in vals]

    cumsum = []
    acc = 0.0
    for p in norm_props:
        acc += p
        cumsum.append(acc)

    col_names = []
    for col in range(num_cols):
        frac = col / float(num_cols) + 0.001
        chosen = names[-1]
        for idx, c_val in enumerate(cumsum):
            if frac < c_val:
                chosen = names[idx]
                break
        col_names.append(chosen)
    return col_names


def compute_curriculum_step(
    distance_walked: float | torch.Tensor,
    commanded_distance: float | torch.Tensor,
    was_terminated_early: bool | torch.Tensor,
    cmd_speed: float | torch.Tensor | None = None,
    up_threshold: float = 0.75,
    down_threshold: float = 0.35,
    min_cmd_speed: float = 0.10,
    min_commanded_distance: float = 0.50,
):
    """Pure-Python / PyTorch core logic for terrain curriculum updates.

    - Make NO promotion or demotion decision when commanded travel distance is too small
      (|cmd| < 0.10 m/s or commanded_distance < 0.50 m): keep current level.
    - Move UP if not terminated early AND distance_walked >= up_threshold * commanded_distance
    - Move DOWN if distance_walked < down_threshold * commanded_distance
    - Otherwise KEEP current level
    """
    if isinstance(distance_walked, (float, int)):
        if commanded_distance < min_commanded_distance:
            return False, False
        if cmd_speed is not None and cmd_speed < min_cmd_speed:
            return False, False
        if was_terminated_early:
            move_up = False
        else:
            move_up = bool(distance_walked >= up_threshold * commanded_distance)
        move_down = bool(distance_walked < down_threshold * commanded_distance)
        return move_up, move_down

    eligible = commanded_distance >= min_commanded_distance
    if cmd_speed is not None:
        eligible = eligible & (cmd_speed >= min_cmd_speed)

    target_dist = up_threshold * commanded_distance
    min_dist = down_threshold * commanded_distance

    move_up = eligible & (~was_terminated_early) & (distance_walked >= target_dist)
    move_down = eligible & (distance_walked < min_dist)
    return move_up, move_down


def zoe_terrain_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = None,
):
    """Curriculum manager function invoked on environment resets."""
    if not hasattr(env.scene.terrain, "update_env_origins"):
        return

    if len(env_ids) == 0:
        return

    name = getattr(asset_cfg, "name", "robot") if asset_cfg is not None else "robot"
    robot = env.scene[name]

    steps_walked = env.episode_length_buf[env_ids]
    if hasattr(env, "_last_reset_step") and env._last_reset_step is not None and hasattr(env, "common_step_counter"):
        actual_steps = (int(env.common_step_counter) - env._last_reset_step[env_ids]).clamp(min=0)
        steps_walked = torch.minimum(steps_walked, actual_steps)
    episode_length = steps_walked.float() * env.step_dt
    vel_cmd = env.command_manager.get_command("base_velocity")[env_ids]
    cmd_speed = torch.norm(vel_cmd[:, :2], dim=-1)
    commanded_distance = cmd_speed * episode_length

    # (a) Subtract env_origins xy from robot root_pos_w xy before computing distance walked
    root_pos_xy = robot.data.root_pos_w[env_ids, :2]
    origin_xy = env.scene.env_origins[env_ids, :2]
    distance_walked = torch.norm(root_pos_xy - origin_xy, dim=-1)

    was_terminated_early = env.termination_manager.terminated[env_ids] & (~env.termination_manager.time_outs[env_ids])

    # (b) & (c) Compute move_up / move_down with standing-env protection
    move_up, move_down = compute_curriculum_step(
        distance_walked=distance_walked,
        commanded_distance=commanded_distance,
        was_terminated_early=was_terminated_early,
        cmd_speed=cmd_speed,
    )

    prev_levels = (
        env.scene.terrain.terrain_levels[env_ids].clone()
        if hasattr(env.scene.terrain, "terrain_levels")
        else None
    )

    env.scene.terrain.update_env_origins(env_ids, move_up=move_up, move_down=move_down)

    # (d) Strictly clamp terrain_levels to [0, max_level] and sync env_origins
    if prev_levels is not None and hasattr(env.scene.terrain, "terrain_levels"):
        max_terrain_level = int(getattr(env.scene.terrain, "max_terrain_level", 10))
        max_level = max(0, max_terrain_level - 1)
        clamped_levels = torch.clamp(prev_levels + move_up.long() - move_down.long(), min=0, max=max_level)
        env.scene.terrain.terrain_levels[env_ids] = clamped_levels
        if hasattr(env.scene.terrain, "terrain_origins") and hasattr(env.scene.terrain, "terrain_types"):
            env.scene.terrain.env_origins[env_ids] = env.scene.terrain.terrain_origins[
                clamped_levels, env.scene.terrain.terrain_types[env_ids]
            ]

    # (e) Per-terrain-type logging into extras["log"]: mean level, promotion rate, demotion rate, survival
    if hasattr(env.scene.terrain, "terrain_types") and hasattr(env.scene.terrain, "terrain_levels"):
        if "log" not in env.extras:
            env.extras["log"] = {}

        all_levels = env.scene.terrain.terrain_levels.float()
        all_types = env.scene.terrain.terrain_types
        reset_types = all_types[env_ids]
        survived = (~was_terminated_early).float()

        env.extras["log"]["Terrain/Mean_Level"] = torch.mean(all_levels).item()
        env.extras["log"]["Terrain/Promotion_Rate"] = torch.mean(move_up.float()).item()
        env.extras["log"]["Terrain/Demotion_Rate"] = torch.mean(move_down.float()).item()
        env.extras["log"]["Terrain/Survival"] = torch.mean(survived).item()

        col_names = get_column_terrain_names(env.scene.terrain)
        unique_names = list(dict.fromkeys(col_names))
        for t_name in unique_names:
            matching_cols = [c for c, cname in enumerate(col_names) if cname == t_name]
            col_tensor = torch.tensor(matching_cols, device=all_types.device, dtype=all_types.dtype)
            all_mask = torch.isin(all_types, col_tensor)
            if all_mask.any():
                env.extras["log"][f"Terrain/{t_name}/mean_level"] = torch.mean(all_levels[all_mask]).item()
            reset_mask = torch.isin(reset_types, col_tensor)
            if reset_mask.any():
                env.extras["log"][f"Terrain/{t_name}/promotion_rate"] = torch.mean(move_up[reset_mask].float()).item()
                env.extras["log"][f"Terrain/{t_name}/demotion_rate"] = torch.mean(move_down[reset_mask].float()).item()
                env.extras["log"][f"Terrain/{t_name}/survival"] = torch.mean(survived[reset_mask]).item()
