# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Command terms for Zoe hexapod deployment."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass


class ZoeUniformVelocityCommand(UniformVelocityCommand):
    """Command generator ensuring a guaranteed fraction of yaw-dominant commands.

    In standard uniform velocity commands, linear velocity commands dominate the reward
    budget (+2.0 vs +0.5/1.0), and with uniform sampling, pure yaw commands are virtually
    absent (measure 0), leading to the robot learning to move linearly and lock its feet
    during pure yaw teleop commands.

    This command generator ensures that at least `rel_yaw_envs` fraction of environments
    receive yaw-dominant commands (|wz| in `yaw_dominant_wz_range`, small vx, vy in
    `yaw_dominant_lin_vel_range`).
    """

    cfg: ZoeUniformVelocityCommandCfg

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        if len(env_ids) == 0:
            return

        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        r = torch.rand(len(env_ids_tensor), device=self.device)
        yaw_mask = r < self.cfg.rel_yaw_envs

        if yaw_mask.any():
            selected_ids = env_ids_tensor[yaw_mask]
            N = len(selected_ids)

            # Not a standing environment
            self.is_standing_env[selected_ids] = False

            # Small linear velocity vx, vy
            vx_min, vx_max = self.cfg.yaw_dominant_lin_vel_range
            vy_min, vy_max = self.cfg.yaw_dominant_lin_vel_range
            self.vel_command_b[selected_ids, 0] = torch.empty(N, device=self.device).uniform_(vx_min, vx_max)
            self.vel_command_b[selected_ids, 1] = torch.empty(N, device=self.device).uniform_(vy_min, vy_max)

            # Yaw magnitude in [0.3, 0.6] with random sign
            wz_min, wz_max = self.cfg.yaw_dominant_wz_range
            wz_mag = torch.empty(N, device=self.device).uniform_(wz_min, wz_max)
            sign = torch.where(torch.rand(N, device=self.device) < 0.5, 1.0, -1.0)
            self.vel_command_b[selected_ids, 2] = sign * wz_mag


@configclass
class ZoeUniformVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for Zoe uniform velocity command generator."""

    class_type: type = ZoeUniformVelocityCommand

    rel_yaw_envs: float = 0.30
    """Fraction of environments that receive yaw-dominant commands (>= 25%)."""

    yaw_dominant_wz_range: tuple[float, float] = (0.30, 0.60)
    """Angular velocity magnitude range for yaw-dominant envs (rad/s)."""

    yaw_dominant_lin_vel_range: tuple[float, float] = (-0.05, 0.05)
    """Small linear velocity range for yaw-dominant envs (m/s)."""


class ZoeTerrainTwoGroupCommand(UniformVelocityCommand):
    """Two-group command generator for Stage 2 terrain training (Part B).

    - Group T (Traversal, ~50%):
        Command held for whole 20s episode.
        vx in [0.10, 0.25], vy in [-0.10, 0.10], wz in [-0.30, 0.30].
        No standing envs. Full terrain curriculum (levels 0..9).
    - Group S (Skills, ~50%):
        Stage 1 distribution (resample every 3-5 s, 10% standing, >= 25% yaw-dominant, backward allowed).
        Terrain level capped at 3, spread across all terrain types.
    - Step-integrated progress tracking:
        commanded_travel = sum |cmd_xy| dt
        achieved_travel = sum max(0, v_xy . unit(cmd_xy)) dt
    """

    cfg: ZoeTerrainTwoGroupCommandCfg

    def __init__(self, cfg: ZoeTerrainTwoGroupCommandCfg, env: any):
        super().__init__(cfg, env)

        # Alternating even/odd env indices guarantees exactly 50% split across all terrain columns
        self.is_group_s = (torch.arange(self.num_envs, device=self.device) % 2 == 1)
        self._env.is_group_s = self.is_group_s

        # Buffers for world-frame net displacement progress accounting (Task 6)
        self.cmd_disp_w = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self.pos_start_w = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self._pos_start_initialized = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.commanded_travel = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.achieved_travel = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.progress_ratio = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._env.commanded_travel = self.commanded_travel
        self._env.achieved_travel = self.achieved_travel
        self._env.progress_ratio = self.progress_ratio

        # Confirm group sizes at runtime
        n_s = self.is_group_s.sum().item()
        n_t = self.num_envs - n_s
        print(
            f"[ZoeTerrainTwoGroupCommand] Group T (Traversal): {n_t} envs ({n_t / self.num_envs * 100:.1f}%) | "
            f"Group S (Skills): {n_s} envs ({n_s / self.num_envs * 100:.1f}%) [Total: {self.num_envs}]"
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset the command generator and clear accumulated travel for reset envs."""
        if env_ids is None:
            env_ids = slice(None)
        # Note: curriculum was evaluated before reset in ManagerBasedRLEnv.reset(),
        # so commanded_travel and achieved_travel have already been consumed.
        self.cmd_disp_w[env_ids] = 0.0
        self.commanded_travel[env_ids] = 0.0
        self.achieved_travel[env_ids] = 0.0
        self.progress_ratio[env_ids] = 0.0
        if hasattr(self.robot, "data") and hasattr(self.robot.data, "root_pos_w"):
            self.pos_start_w[env_ids] = self.robot.data.root_pos_w[env_ids, :2].clone()
            self._pos_start_initialized[env_ids] = True
        return super().reset(env_ids)

    def _resample(self, env_ids: Sequence[int]):
        """Resample time_left independently for Group T (20s) and Group S (3-5s)."""
        if len(env_ids) == 0:
            return
        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        group_s_mask = self.is_group_s[env_ids_tensor]
        t_ids = env_ids_tensor[~group_s_mask]
        s_ids = env_ids_tensor[group_s_mask]

        if len(t_ids) > 0:
            self.time_left[t_ids] = torch.empty(len(t_ids), device=self.device).uniform_(
                *self.cfg.group_t_resampling_time_range
            )
        if len(s_ids) > 0:
            self.time_left[s_ids] = torch.empty(len(s_ids), device=self.device).uniform_(
                *self.cfg.group_s_resampling_time_range
            )

        self._resample_command(env_ids)
        self.command_counter[env_ids] += 1

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample velocity commands according to group distributions."""
        if len(env_ids) == 0:
            return
        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        group_s_mask = self.is_group_s[env_ids_tensor]
        t_ids = env_ids_tensor[~group_s_mask]
        s_ids = env_ids_tensor[group_s_mask]

        # Group T: Traversal (held for 20s episode, forward traversal with mild heading variations)
        if len(t_ids) > 0:
            n_t = len(t_ids)
            self.is_standing_env[t_ids] = False
            self.vel_command_b[t_ids, 0] = torch.empty(n_t, device=self.device).uniform_(
                *self.cfg.group_t_ranges.lin_vel_x
            )
            self.vel_command_b[t_ids, 1] = torch.empty(n_t, device=self.device).uniform_(
                *self.cfg.group_t_ranges.lin_vel_y
            )
            self.vel_command_b[t_ids, 2] = torch.empty(n_t, device=self.device).uniform_(
                *self.cfg.group_t_ranges.ang_vel_z
            )
            if self.cfg.heading_command:
                self.heading_target[t_ids] = torch.empty(n_t, device=self.device).uniform_(-3.14159, 3.14159)
                self.is_heading_env[t_ids] = True
            else:
                self.is_heading_env[t_ids] = False

        # Group S: Skills (Stage 1 distribution: 10% standing, >= 25% yaw-dominant, backward allowed)
        if len(s_ids) > 0:
            n_s = len(s_ids)
            # 10% standing envs
            r_stand = torch.rand(n_s, device=self.device)
            stand_mask = r_stand < self.cfg.group_s_rel_standing_envs
            self.is_standing_env[s_ids] = stand_mask

            # Sample general skill commands
            self.vel_command_b[s_ids, 0] = torch.empty(n_s, device=self.device).uniform_(
                *self.cfg.group_s_ranges.lin_vel_x
            )
            self.vel_command_b[s_ids, 1] = torch.empty(n_s, device=self.device).uniform_(
                *self.cfg.group_s_ranges.lin_vel_y
            )
            self.vel_command_b[s_ids, 2] = torch.empty(n_s, device=self.device).uniform_(
                *self.cfg.group_s_ranges.ang_vel_z
            )

            # >= 25% yaw-dominant among non-standing envs
            non_stand_mask = ~stand_mask
            if non_stand_mask.any():
                non_stand_ids = s_ids[non_stand_mask]
                n_ns = len(non_stand_ids)
                r_yaw = torch.rand(n_ns, device=self.device)
                yaw_mask = r_yaw < self.cfg.group_s_rel_yaw_envs
                if yaw_mask.any():
                    yaw_ids = non_stand_ids[yaw_mask]
                    n_y = len(yaw_ids)
                    vx_min, vx_max = self.cfg.group_s_yaw_dominant_lin_vel_range
                    vy_min, vy_max = self.cfg.group_s_yaw_dominant_lin_vel_range
                    self.vel_command_b[yaw_ids, 0] = torch.empty(n_y, device=self.device).uniform_(vx_min, vx_max)
                    self.vel_command_b[yaw_ids, 1] = torch.empty(n_y, device=self.device).uniform_(vy_min, vy_max)
                    wz_min, wz_max = self.cfg.group_s_yaw_dominant_wz_range
                    wz_mag = torch.empty(n_y, device=self.device).uniform_(wz_min, wz_max)
                    sign = torch.where(torch.rand(n_y, device=self.device) < 0.5, 1.0, -1.0)
                    self.vel_command_b[yaw_ids, 2] = sign * wz_mag

            if self.cfg.heading_command:
                heading_range = self.cfg.ranges.heading if self.cfg.ranges.heading is not None else (-3.14159, 3.14159)
                self.heading_target[s_ids] = torch.empty(n_s, device=self.device).uniform_(*heading_range)
                self.is_heading_env[s_ids] = torch.rand(n_s, device=self.device) < self.cfg.rel_heading_envs
            else:
                self.is_heading_env[s_ids] = False

    def _update_metrics(self):
        super()._update_metrics()
        # Initialize start position if not initialized
        if hasattr(self.robot, "data") and hasattr(self.robot.data, "root_pos_w"):
            if not self._pos_start_initialized.all():
                uninit = ~self._pos_start_initialized
                self.pos_start_w[uninit] = self.robot.data.root_pos_w[uninit, :2].clone()
                self._pos_start_initialized[uninit] = True

        dt = float(self._env.step_dt)
        # 1. Integrate commanded displacement vector in world frame: R(yaw_t) @ cmd_b
        yaw = self.robot.data.heading_w
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        vx_b = self.vel_command_b[:, 0]
        vy_b = self.vel_command_b[:, 1]
        cmd_w_x = cos_yaw * vx_b - sin_yaw * vy_b
        cmd_w_y = sin_yaw * vx_b + cos_yaw * vy_b

        self.cmd_disp_w[:, 0] += cmd_w_x * dt
        self.cmd_disp_w[:, 1] += cmd_w_y * dt

        # 2. Commanded travel magnitude: D_cmd = ||cmd_disp_w||
        cmd_norm = torch.norm(self.cmd_disp_w, dim=-1)
        self.commanded_travel[:] = cmd_norm

        # 3. Actual net displacement from reset position in world frame
        actual_disp_w = self.robot.data.root_pos_w[:, :2] - self.pos_start_w

        # 4. Projected net displacement along unit commanded vector: progress = (actual . unit(cmd)) / |cmd|
        safe_cmd_norm = torch.clamp(cmd_norm, min=1e-4)
        cmd_unit_w = self.cmd_disp_w / safe_cmd_norm.unsqueeze(-1)
        achieved = torch.sum(actual_disp_w * cmd_unit_w, dim=-1)
        self.achieved_travel[:] = achieved

        self.progress_ratio[:] = torch.where(
            cmd_norm >= 1e-4,
            achieved / safe_cmd_norm,
            torch.zeros_like(achieved),
        )

    def _update_command(self):
        super()._update_command()
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        if len(standing_env_ids) > 0:
            self.vel_command_b[standing_env_ids, :] = 0.0


@configclass
class ZoeTerrainTwoGroupCommandCfg(UniformVelocityCommandCfg):
    """Configuration for ZoeTerrainTwoGroupCommand."""

    class_type: type = ZoeTerrainTwoGroupCommand

    resampling_time_range: tuple[float, float] = (20.0, 20.0)
    rel_standing_envs: float = 0.0
    rel_heading_envs: float = 0.0
    heading_command: bool = False

    # Default fallback ranges for parent class init
    ranges: UniformVelocityCommandCfg.Ranges = UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(0.10, 0.25),
        lin_vel_y=(-0.10, 0.10),
        ang_vel_z=(-0.30, 0.30),
    )

    # Group T (Traversal, ~50%)
    group_t_resampling_time_range: tuple[float, float] = (20.0, 20.0)
    group_t_ranges: UniformVelocityCommandCfg.Ranges = UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(0.10, 0.25),
        lin_vel_y=(-0.10, 0.10),
        ang_vel_z=(-0.30, 0.30),
    )

    # Group S (Skills, ~50%)
    group_s_resampling_time_range: tuple[float, float] = (3.0, 5.0)
    group_s_rel_standing_envs: float = 0.10
    group_s_rel_yaw_envs: float = 0.30  # >= 25% yaw-dominant envs
    group_s_yaw_dominant_wz_range: tuple[float, float] = (0.30, 0.60)
    group_s_yaw_dominant_lin_vel_range: tuple[float, float] = (-0.05, 0.05)
    group_s_ranges: UniformVelocityCommandCfg.Ranges = UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.25, 0.30),  # backward allowed!
        lin_vel_y=(-0.20, 0.20),
        ang_vel_z=(-0.60, 0.60),
    )
