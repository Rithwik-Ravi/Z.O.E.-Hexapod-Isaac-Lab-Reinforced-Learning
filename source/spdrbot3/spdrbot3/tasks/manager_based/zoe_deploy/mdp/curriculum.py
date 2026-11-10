# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""MDP Curriculum terms for Z.O.E. Hexapod terrain task."""

from collections.abc import Sequence
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
    commanded_travel: float | torch.Tensor = None,
    achieved_travel: float | torch.Tensor = None,
    was_terminated_early: bool | torch.Tensor = False,
    up_threshold: float = 0.75,
    down_threshold: float = 0.35,
    min_commanded_travel: float = 0.50,
    # Backward compatibility arguments
    distance_walked: float | torch.Tensor = None,
    commanded_distance: float | torch.Tensor = None,
    cmd_speed: float | torch.Tensor | None = None,
    min_cmd_speed: float = 0.10,
    min_commanded_distance: float = 0.50,
):
    """Core logic for terrain curriculum updates with step-integrated progress accounting.

    - commanded_travel = sum |cmd_xy| dt
    - achieved_travel = sum max(0, v_xy . unit(cmd_xy)) dt (along command direction)
    - progress_ratio = achieved_travel / commanded_travel (if commanded_travel >= 1e-4)
    - Promote if: no early termination AND commanded_travel >= 0.5 m AND progress_ratio >= 0.75
    - Demote if: early termination OR (commanded_travel >= 0.5 m AND progress_ratio < 0.35)
    - Otherwise KEEP current level.
    """
    if commanded_travel is None:
        commanded_travel = commanded_distance if commanded_distance is not None else 0.0
    if achieved_travel is None:
        achieved_travel = distance_walked if distance_walked is not None else 0.0
    if min_commanded_travel is None:
        min_commanded_travel = min_commanded_distance

    if isinstance(commanded_travel, (float, int)):
        if was_terminated_early:
            return False, True
        if commanded_travel < min_commanded_travel:
            return False, False
        if cmd_speed is not None and cmd_speed < min_cmd_speed:
            return False, False
        ratio = achieved_travel / max(float(commanded_travel), 1e-4)
        move_up = bool(ratio >= up_threshold)
        move_down = bool(ratio < down_threshold)
        return move_up, move_down

    eligible = commanded_travel >= min_commanded_travel
    if cmd_speed is not None:
        eligible = eligible & (cmd_speed >= min_cmd_speed)

    safe_cmd = torch.clamp(commanded_travel, min=1e-4)
    ratio = torch.where(eligible, achieved_travel / safe_cmd, torch.zeros_like(achieved_travel))

    if isinstance(was_terminated_early, bool):
        was_terminated_early = torch.tensor(was_terminated_early, device=commanded_travel.device, dtype=torch.bool)

    move_up = eligible & (~was_terminated_early) & (ratio >= up_threshold)
    move_down = was_terminated_early | (eligible & (ratio < down_threshold))
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

    # (a) Check if command generator tracks commanded_travel and achieved_travel
    cmd_gen = getattr(env, "command_manager", None)
    base_vel_cmd = None
    if cmd_gen is not None:
        if hasattr(cmd_gen, "get_term"):
            try:
                base_vel_cmd = cmd_gen.get_term("base_velocity")
            except Exception:
                pass
        elif hasattr(cmd_gen, "_terms") and "base_velocity" in cmd_gen._terms:
            base_vel_cmd = cmd_gen._terms["base_velocity"]

    if base_vel_cmd is not None and hasattr(base_vel_cmd, "commanded_travel") and hasattr(base_vel_cmd, "achieved_travel"):
        commanded_travel = base_vel_cmd.commanded_travel[env_ids]
        achieved_travel = base_vel_cmd.achieved_travel[env_ids]
    elif hasattr(env, "commanded_travel") and hasattr(env, "achieved_travel"):
        commanded_travel = env.commanded_travel[env_ids]
        achieved_travel = env.achieved_travel[env_ids]
    else:
        # Fallback to local origin distance
        steps_walked = env.episode_length_buf[env_ids]
        if hasattr(env, "_last_reset_step") and env._last_reset_step is not None and hasattr(env, "common_step_counter"):
            actual_steps = (int(env.common_step_counter) - env._last_reset_step[env_ids]).clamp(min=0)
            steps_walked = torch.minimum(steps_walked, actual_steps)
        episode_length = steps_walked.float() * env.step_dt
        vel_cmd = env.command_manager.get_command("base_velocity")[env_ids]
        cmd_speed = torch.norm(vel_cmd[:, :2], dim=-1)
        commanded_travel = cmd_speed * episode_length

        root_pos_xy = robot.data.root_pos_w[env_ids, :2]
        origin_xy = env.scene.env_origins[env_ids, :2]
        achieved_travel = torch.norm(root_pos_xy - origin_xy, dim=-1)

    was_terminated_early = env.termination_manager.terminated[env_ids] & (~env.termination_manager.time_outs[env_ids])

    # (b) & (c) Compute move_up / move_down using progress ratio
    move_up, move_down = compute_curriculum_step(
        commanded_travel=commanded_travel,
        achieved_travel=achieved_travel,
        was_terminated_early=was_terminated_early,
    )

    prev_levels = (
        env.scene.terrain.terrain_levels[env_ids].clone()
        if hasattr(env.scene.terrain, "terrain_levels")
        else None
    )

    env.scene.terrain.update_env_origins(env_ids, move_up=move_up, move_down=move_down)

    # (d) Strictly clamp terrain_levels to [0, max_level] (Group S capped at level 3)
    is_group_s = getattr(env, "is_group_s", None)
    if is_group_s is None and base_vel_cmd is not None and hasattr(base_vel_cmd, "is_group_s"):
        is_group_s = base_vel_cmd.is_group_s

    if prev_levels is not None and hasattr(env.scene.terrain, "terrain_levels"):
        max_terrain_level = int(getattr(env.scene.terrain, "max_terrain_level", 10))
        max_level = max(0, max_terrain_level - 1)
        new_levels = prev_levels + move_up.long() - move_down.long()

        if is_group_s is not None:
            group_s_env = is_group_s[env_ids]
            cap = torch.where(
                group_s_env,
                torch.tensor(3, device=new_levels.device, dtype=new_levels.dtype),
                torch.tensor(max_level, device=new_levels.device, dtype=new_levels.dtype),
            )
            zero_tensor = torch.zeros_like(cap)
            clamped_levels = torch.clamp(new_levels, min=zero_tensor, max=cap)
        else:
            clamped_levels = torch.clamp(new_levels, min=0, max=max_level)

        env.scene.terrain.terrain_levels[env_ids] = clamped_levels
        if hasattr(env.scene.terrain, "terrain_origins") and hasattr(env.scene.terrain, "terrain_types"):
            env.scene.terrain.env_origins[env_ids] = env.scene.terrain.terrain_origins[
                clamped_levels, env.scene.terrain.terrain_types[env_ids]
            ]

    # (e) Per-terrain-type logging into extras["log"]: mean level, promotion, demotion, survival, progress_ratio
    if hasattr(env.scene.terrain, "terrain_types") and hasattr(env.scene.terrain, "terrain_levels"):
        if "log" not in env.extras:
            env.extras["log"] = {}

        all_levels = env.scene.terrain.terrain_levels.float()
        all_types = env.scene.terrain.terrain_types
        reset_types = all_types[env_ids]
        survived = (~was_terminated_early).float()

        safe_cmd = torch.clamp(commanded_travel, min=1e-4)
        progress_ratio = torch.where(commanded_travel >= 0.50, achieved_travel / safe_cmd, torch.zeros_like(achieved_travel))

        env.extras["log"]["Terrain/Mean_Level"] = torch.mean(all_levels).item()
        env.extras["log"]["Terrain/Promotion_Rate"] = torch.mean(move_up.float()).item()
        env.extras["log"]["Terrain/Demotion_Rate"] = torch.mean(move_down.float()).item()
        env.extras["log"]["Terrain/Survival"] = torch.mean(survived).item()
        env.extras["log"]["Terrain/Progress_Ratio"] = torch.mean(progress_ratio).item()

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
                env.extras["log"][f"Terrain/{t_name}/progress_ratio"] = torch.mean(progress_ratio[reset_mask]).item()


PENALTY_RAMP_TERMS: list[tuple[str, ...]] = [
    ("torque_limit", "torque_limit_penalty"),
    ("fewer_than_4_feet_contact",),
    ("dof_acc_l2", "dof_acc"),
    ("action_rate_l2", "action_rate"),
    ("stand_still_posture", "stand_still_posture_penalty"),
    ("joint_vel_limit", "joint_vel_limit_penalty"),
]


def penalty_scale_ramp(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | None = None,
    num_steps_per_env: int = 48,
    max_iter: float = 800.0,
    min_scale: float = 0.1,
    max_scale: float = 1.0,
) -> float:
    """Curriculum term that scales harsh penalty reward terms over the first 800 iterations (Task 4).

    Schedule: scale = min(1.0, 0.1 + 0.9 * (iteration / 800.0))
    Multiplies base weights of:
    - torque_limit_penalty (or torque_limit)
    - fewer_than_4_feet_contact
    - dof_acc (or dof_acc_l2)
    - action_rate (or action_rate_l2)
    - stand_still_posture (or stand_still_posture_penalty)
    - joint_vel_limit (or joint_vel_limit_penalty)

    Derives iteration correctly from common_step_counter / num_steps_per_env (48).
    Logs 'Curriculum/penalty_scale' to TensorBoard via env.extras['log'].
    """
    if not hasattr(env, "reward_manager") or env.reward_manager is None:
        return 1.0

    # Cache base weights on first call
    if not hasattr(env, "_penalty_ramp_base_weights"):
        env._penalty_ramp_base_weights = {}
        for name_tuple in PENALTY_RAMP_TERMS:
            for candidate in name_tuple:
                if candidate in env.reward_manager._term_names:
                    term_cfg = env.reward_manager.get_term_cfg(candidate)
                    env._penalty_ramp_base_weights[candidate] = float(term_cfg.weight)
                    break

    # Derive iteration from step counter
    common_step_counter = getattr(env, "common_step_counter", 0)
    current_iter = float(common_step_counter) / float(num_steps_per_env)

    # Compute ramp scale: 0.1 at iter 0 rising linearly to 1.0 at iter 800
    progress = max(0.0, min(1.0, current_iter / max_iter))
    scale = min_scale + (max_scale - min_scale) * progress

    # Update active weights in reward manager
    for term_name, base_weight in env._penalty_ramp_base_weights.items():
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        term_cfg.weight = base_weight * scale
        env.reward_manager.set_term_cfg(term_name, term_cfg)

    # Log to TensorBoard
    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["Curriculum/penalty_scale"] = scale

    return scale
