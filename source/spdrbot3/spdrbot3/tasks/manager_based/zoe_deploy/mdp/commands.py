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
