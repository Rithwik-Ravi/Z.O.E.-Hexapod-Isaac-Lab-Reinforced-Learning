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

from ..params import ZOE_PARAMS, get_foot_tip_offsets_tensor


def fewer_than_3_feet_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize having fewer than 3 feet in contact with ground: (3 - n)^2 for n < 3."""
    contact_sensor = env.scene[sensor_cfg.name]
    net_forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    in_contact = (net_forces_z > 1.0).float()
    num_feet_in_contact = torch.sum(in_contact, dim=1)
    def_count = torch.clamp(3.0 - num_feet_in_contact, min=0.0)
    return torch.square(def_count)


def fewer_than_4_feet_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize having fewer than 4 feet in contact with ground: (4 - n)^2 for n < 4."""
    contact_sensor = env.scene[sensor_cfg.name]
    net_forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    in_contact = (net_forces_z > 1.0).float()
    num_feet_in_contact = torch.sum(in_contact, dim=1)
    def_count = torch.clamp(4.0 - num_feet_in_contact, min=0.0)
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


def _get_scene_entity(scene, name: str):
    """Safely retrieve an entity or sensor from env.scene by name."""
    if hasattr(scene, "sensors") and isinstance(scene.sensors, dict) and name in scene.sensors:
        return scene.sensors[name]
    if hasattr(scene, "articulations") and isinstance(scene.articulations, dict) and name in scene.articulations:
        return scene.articulations[name]
    try:
        return scene[name]
    except (KeyError, IndexError, TypeError):
        return None


def ground_height_from_raycaster(env: ManagerBasedRLEnv, sensor_name: str = "height_scanner") -> torch.Tensor | None:
    """Compute mean terrain elevation [N] under the robot from a downward RayCaster sensor."""
    scanner = _get_scene_entity(env.scene, sensor_name)
    if scanner is None or not hasattr(scanner, "data") or not hasattr(scanner.data, "ray_hits_w"):
        return None
    ray_hits_z = scanner.data.ray_hits_w[..., 2]
    finite_mask = torch.isfinite(ray_hits_z)
    safe_z = torch.where(finite_mask, ray_hits_z, torch.zeros_like(ray_hits_z))
    valid_counts = finite_mask.sum(dim=-1).clamp(min=1)
    mean_z = safe_z.sum(dim=-1) / valid_counts.float()
    all_missed = finite_mask.sum(dim=-1) == 0
    if all_missed.any():
        fallback_z = (
            env.scene.env_origins[:, 2]
            if hasattr(env.scene, "env_origins") and env.scene.env_origins is not None
            else torch.zeros_like(mean_z)
        )
        mean_z = torch.where(all_missed, fallback_z, mean_z)
    return mean_z


def ground_height_from_feet(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = None,
    sensor_cfg: SceneEntityCfg = None,
) -> torch.Tensor | None:
    """Fallback ground height estimator: mean Z of contacting foot tips (or lowest foot tip if airborne)."""
    name = getattr(asset_cfg, "name", "robot") if asset_cfg is not None else "robot"
    robot = _get_scene_entity(env.scene, name)
    contact_name = getattr(sensor_cfg, "name", "contact_forces") if sensor_cfg is not None else "contact_forces"
    contact_sensor = _get_scene_entity(env.scene, contact_name)
    if robot is None or contact_sensor is None:
        return None

    if (
        sensor_cfg is not None
        and getattr(sensor_cfg, "body_ids", None) is not None
        and not isinstance(sensor_cfg.body_ids, slice)
    ):
        foot_body_ids = list(sensor_cfg.body_ids)
    elif hasattr(contact_sensor, "body_names") and contact_sensor.body_names is not None:
        foot_body_ids = [i for i, n in enumerate(contact_sensor.body_names) if "NFOOT" in n]
    else:
        foot_body_ids = list(range(6))

    if not foot_body_ids:
        return None

    class _FootCfg:
        def __init__(self, cname, bids):
            self.name = cname
            self.body_ids = bids

    fcfg = _FootCfg(contact_name, foot_body_ids)
    tip_pos_w, _ = _compute_foot_tip_state(robot, fcfg, contact_sensor=contact_sensor)
    tip_z = tip_pos_w[..., 2]
    net_forces_z = contact_sensor.data.net_forces_w[:, foot_body_ids, 2]
    in_contact = net_forces_z > 1.0
    contact_count = in_contact.sum(dim=1)

    contact_z_sum = torch.where(in_contact, tip_z, torch.zeros_like(tip_z)).sum(dim=1)
    mean_contact_z = contact_z_sum / contact_count.clamp(min=1).float()
    min_tip_z = tip_z.min(dim=1).values
    return torch.where(contact_count > 0, mean_contact_z, min_tip_z)


def ground_height_under_robot(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = None,
    sensor_cfg: SceneEntityCfg = None,
    method: str = "auto",
) -> torch.Tensor:
    """Return estimated terrain ground Z [N] directly underneath each robot."""
    name = getattr(asset_cfg, "name", "robot") if asset_cfg is not None else "robot"
    robot = env.scene[name]
    root_z = robot.data.root_pos_w[:, 2]

    if method in ("auto", "raycaster"):
        ray_z = ground_height_from_raycaster(env, sensor_name="height_scanner")
        if ray_z is not None:
            return ray_z
        if method == "raycaster":
            raise RuntimeError("RayCaster 'height_scanner' requested but not found in env.scene")

    if method == "feet":
        feet_z = ground_height_from_feet(env, asset_cfg=asset_cfg, sensor_cfg=sensor_cfg)
        if feet_z is not None:
            return feet_z
        raise RuntimeError("Foot contact fallback requested but robot/contact_forces not found in env.scene")

    terrain = getattr(env.scene, "terrain", None)
    terrain_type = getattr(getattr(terrain, "cfg", None), "terrain_type", "plane")
    if terrain_type != "plane":
        feet_z = ground_height_from_feet(env, asset_cfg=asset_cfg, sensor_cfg=sensor_cfg)
        if feet_z is not None:
            return feet_z
        if hasattr(env.scene, "env_origins") and env.scene.env_origins is not None:
            return env.scene.env_origins[:, 2]

    return torch.zeros_like(root_z)


def height_above_ground(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = None,
    sensor_cfg: SceneEntityCfg = None,
    method: str = "auto",
) -> torch.Tensor:
    """Return base height above the local terrain surface [N] under the robot."""
    if asset_cfg is None and hasattr(SceneEntityCfg, "__call__"):
        asset_cfg = SceneEntityCfg("robot")
    name = getattr(asset_cfg, "name", "robot")
    robot = env.scene[name]
    root_z = robot.data.root_pos_w[:, 2]
    ground_z = ground_height_under_robot(env, asset_cfg=asset_cfg, sensor_cfg=sensor_cfg, method=method)
    return root_z - ground_z


def foot_clearance_relative(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize feet that go significantly above clearance target relative to local ground under robot."""
    robot = env.scene[asset_cfg.name]
    foot_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]
    ground_z = (robot.data.root_pos_w[:, 2] - height_above_ground(env, asset_cfg=asset_cfg)).unsqueeze(1)
    ground_clearance = foot_pos_w[:, :, 2] - ground_z
    return torch.sum(torch.square(torch.clamp(ground_clearance - target_height, min=0.0)), dim=1)


def undesired_contacts(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Penalize undesired contacts (thigh, coxa, or base touching ground)."""
    contact_sensor = env.scene[sensor_cfg.name]
    net_forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    is_contact = (net_forces_z > threshold).float()
    return torch.sum(is_contact, dim=1)


def base_height_shaped(
    env: ManagerBasedRLEnv,
    target_height: float = ZOE_PARAMS.TARGET_STANDING_HEIGHT,
    min_height: float = ZOE_PARAMS.GATE_MIN_HEIGHT,
    std: float = 0.015,
    barrier_quad_weight: float = 150.0,
    barrier_lin_weight: float = 10.0,
    asset_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """Base height shaping: positive exponential attractor near target height plus steep barrier below min_height."""
    height = height_above_ground(env, asset_cfg=asset_cfg)

    # Positive attractor (1.0 when at target_height)
    attractor = torch.exp(-torch.square((height - target_height) / std))

    # One-sided barrier below min_height
    deficit = torch.clamp(min_height - height, min=0.0)
    barrier = barrier_lin_weight * deficit + barrier_quad_weight * torch.square(deficit)

    return attractor - barrier


def metric_base_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = None) -> torch.Tensor:
    """Metric monitor: Mean base height above local ground [m]."""
    return height_above_ground(env, asset_cfg=asset_cfg)


def metric_non_foot_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Metric monitor: Non-foot contact fraction [0 or 1 per env]."""
    contact_sensor = env.scene[sensor_cfg.name]
    net_forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    return (net_forces_z > 1.0).any(dim=1).float()


def metric_torque_saturation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = None,
    threshold: float = ZOE_PARAMS.TORQUE_SAT_THRESHOLD_NM,
) -> torch.Tensor:
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


def _compute_foot_tip_state(
    robot,
    sensor_cfg: SceneEntityCfg,
    tip_offset_z: float | None = None,
    contact_sensor=None,
):
    """Computes foot tip positions [N, 6, 3] and velocities [N, 6, 3] in world frame using per-leg 3D tip vectors.

    Maps `sensor_cfg.body_ids` (indices in `contact_sensor.body_names`) to the corresponding body indices in
    `robot.body_names` so that foot contacts and foot kinematics are strictly aligned leg-for-leg.
    """
    cache_key = (
        tuple(sensor_cfg.body_ids),
        str(robot.data.body_pos_w.device),
        robot.data.body_pos_w.dtype,
        tip_offset_z,
    )
    cache = getattr(robot, "_foot_tip_cache", None)
    if cache is None:
        cache = {}
        robot._foot_tip_cache = cache

    if cache_key not in cache:
        if (
            contact_sensor is not None
            and hasattr(contact_sensor, "body_names")
            and contact_sensor.body_names is not None
        ):
            foot_names = [contact_sensor.body_names[i] for i in sensor_cfg.body_ids]
        elif hasattr(robot, "body_names") and robot.body_names is not None and len(robot.body_names) > max(sensor_cfg.body_ids):
            foot_names = [robot.body_names[i] for i in sensor_cfg.body_ids]
        else:
            foot_names = [f"NFOOT{i}" for i in range(1, len(sensor_cfg.body_ids) + 1)]

        if hasattr(robot, "body_names") and robot.body_names is not None and all(n in robot.body_names for n in foot_names):
            robot_body_ids = [robot.body_names.index(n) for n in foot_names]
        else:
            robot_body_ids = list(sensor_cfg.body_ids)

        dev = robot.data.body_pos_w.device
        dt = robot.data.body_pos_w.dtype
        if tip_offset_z is not None:
            r_loc = torch.tensor([0.0, 0.0, tip_offset_z], device=dev, dtype=dt).view(1, 1, 3).expand(1, len(robot_body_ids), 3)
        else:
            r_loc = get_foot_tip_offsets_tensor(foot_names, dev, dt).unsqueeze(0)
        cache[cache_key] = (robot_body_ids, r_loc)

    robot_body_ids, r_local_1 = cache[cache_key]
    body_pos = robot.data.body_pos_w[:, robot_body_ids, :]
    body_quat = robot.data.body_quat_w[:, robot_body_ids, :]
    body_lin_vel = robot.data.body_lin_vel_w[:, robot_body_ids, :]
    body_ang_vel = robot.data.body_ang_vel_w[:, robot_body_ids, :]
    r_local = r_local_1.expand(body_pos.shape[0], -1, 3)

    q_w = body_quat[..., 0:1]
    q_vec = body_quat[..., 1:4]
    a = torch.cross(q_vec, r_local, dim=-1)
    v_rot = r_local + 2.0 * q_w * a + 2.0 * torch.cross(q_vec, a, dim=-1)

    tip_pos_w = body_pos + v_rot
    tip_vel_w = body_lin_vel + torch.cross(body_ang_vel, v_rot, dim=-1)
    return tip_pos_w, tip_vel_w


def foot_tip_slip_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = None,
    tip_offset_z: float | None = None,
) -> torch.Tensor:
    """Penalize horizontal velocity of foot tips while in ground contact."""
    if asset_cfg is None and hasattr(SceneEntityCfg, "__call__"):
        asset_cfg = SceneEntityCfg("robot")
    name = getattr(asset_cfg, "name", "robot")
    robot = env.scene[name]
    contact_sensor = env.scene[sensor_cfg.name]
    net_forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    in_contact = (net_forces_z > 1.0).float()

    _, tip_vel_w = _compute_foot_tip_state(robot, sensor_cfg, tip_offset_z, contact_sensor=contact_sensor)
    tip_speed_xy_sq = torch.sum(torch.square(tip_vel_w[..., :2]), dim=-1)
    return torch.sum(in_contact * tip_speed_xy_sq, dim=1)


def foot_tip_clearance_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    target_clearance: float = 0.05,
    std: float = 0.02,
    tip_offset_z: float | None = None,
    asset_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """Encourage foot tips to reach target swing clearance (4-7 cm) during swing phase when walking."""
    if asset_cfg is None and hasattr(SceneEntityCfg, "__call__"):
        asset_cfg = SceneEntityCfg("robot")
    name = getattr(asset_cfg, "name", "robot")
    robot = env.scene[name]
    cmd = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(cmd[:, :2], dim=1)
    is_walking = (cmd_norm > 0.05).float()

    contact_sensor = env.scene[sensor_cfg.name]
    net_forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    in_swing = (net_forces_z <= 1.0).float()

    tip_pos_w, _ = _compute_foot_tip_state(robot, sensor_cfg, tip_offset_z, contact_sensor=contact_sensor)
    ground_z = (robot.data.root_pos_w[:, 2] - height_above_ground(env, asset_cfg=asset_cfg)).unsqueeze(1)
    tip_height = tip_pos_w[..., 2] - ground_z
    height_error = torch.square((tip_height - target_clearance) / std)
    reward_per_foot = torch.exp(-height_error)
    return is_walking * torch.sum(in_swing * reward_per_foot, dim=1)


def stand_still_posture_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """When velocity command is near zero, penalize joint deviation from default pose and feet in air."""
    if asset_cfg is None and hasattr(SceneEntityCfg, "__call__"):
        asset_cfg = SceneEntityCfg("robot")
    name = getattr(asset_cfg, "name", "robot")
    robot = env.scene[name]
    cmd = env.command_manager.get_command(command_name)
    cmd_speed = torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])
    is_stopped = (cmd_speed < cmd_threshold).float()

    joint_dev = torch.sum(torch.square(robot.data.joint_pos - robot.data.default_joint_pos), dim=-1)

    contact_sensor = env.scene[sensor_cfg.name]
    net_forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    feet_down = torch.sum((net_forces_z > 1.0).float(), dim=1)
    feet_in_air = torch.clamp(6.0 - feet_down, min=0.0)

    return is_stopped * (joint_dev + 0.5 * torch.square(feet_in_air))


def stand_still_feet_grounded_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.05,
) -> torch.Tensor:
    """When velocity command is near zero, reward the fraction of feet on the ground (N_down / 6.0)."""
    cmd = env.command_manager.get_command(command_name)
    cmd_speed = torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])
    is_stopped = (cmd_speed < cmd_threshold).float()

    contact_sensor = env.scene[sensor_cfg.name]
    net_forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    feet_down = torch.sum((net_forces_z > 1.0).float(), dim=1)
    fraction_grounded = feet_down / 6.0
    return is_stopped * fraction_grounded


def joint_vel_limit_penalty(
    env: ManagerBasedRLEnv,
    limit: float = ZOE_PARAMS.NO_LOAD_SPEED_RAD_S,
    asset_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """Penalize joint velocities exceeding the Waveshare ST3215 12V servo speed limit (4.71 rad/s)."""
    if asset_cfg is None and hasattr(SceneEntityCfg, "__call__"):
        asset_cfg = SceneEntityCfg("robot")
    name = getattr(asset_cfg, "name", "robot")
    robot = env.scene[name]
    excess_vel = torch.clamp(torch.abs(robot.data.joint_vel) - limit, min=0.0)
    return torch.sum(torch.square(excess_vel), dim=-1)


def torque_limit_penalty(
    env: ManagerBasedRLEnv,
    threshold: float = ZOE_PARAMS.rewards.torque_soft_limit_nm,
    asset_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """Soft quadratic barrier penalizing joint torques exceeding 80% of the 12V effort limit (2.00 N.m)."""
    if asset_cfg is None and hasattr(SceneEntityCfg, "__call__"):
        asset_cfg = SceneEntityCfg("robot")
    name = getattr(asset_cfg, "name", "robot")
    robot = env.scene[name]
    excess_torque = torch.clamp(torch.abs(robot.data.applied_torque) - threshold, min=0.0)
    return torch.sum(torch.square(excess_torque), dim=-1)

