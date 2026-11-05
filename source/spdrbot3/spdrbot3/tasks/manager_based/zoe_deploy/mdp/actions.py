# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom MDP Actions for Z.O.E. Hexapod: Action Delay Buffer and Target Rate Limiter."""

from __future__ import annotations
import math
from dataclasses import MISSING
from typing import TYPE_CHECKING, Sequence
import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DelayedRateLimitedJointPositionAction(JointPositionAction):
    """Joint position action term with per-environment action delay buffer and rate limiter.
    
    Features:
    1. Action Delay Buffer: Stores historical targets for 0 to max_delay_steps policy cycles.
       Delay is randomly assigned per environment on reset.
    2. Zero-Offset Calibration: Adds persistent calibration offset delta to commanded target.
    3. Target Rate Limiter: Clamps per-step change to rate_limit_rad_per_step (matching ROS runtime).
    4. Clipping Monitor: Logs rate limiter clipping frequency.
    5. Preserves un-delayed policy action in self.raw_actions for mdp.last_action observation.
    """

    cfg: DelayedRateLimitedJointPositionActionCfg

    def __init__(self, cfg: DelayedRateLimitedJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.max_delay = cfg.max_delay_steps
        self.rate_limit = cfg.rate_limit_rad_per_step

        # Per-environment sampled delay
        self.delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Buffer shape: [max_delay + 1, num_envs, num_joints]
        self.buffer_len = self.max_delay + 1
        self.action_buffer = torch.zeros(self.buffer_len, self.num_envs, self.action_dim, device=self.device)

        # Previous rate-limited target
        self.prev_targets = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        # Metrics: clipping counter
        self.total_clipped_joints = 0
        self.total_commanded_joints = 0

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)

        # Sample delay steps
        if self.cfg.fixed_delay_steps is not None:
            self.delay_steps[env_ids] = self.cfg.fixed_delay_steps
        else:
            if isinstance(env_ids, slice):
                n_resets = self.num_envs
            else:
                n_resets = len(env_ids)
            rand_delays = torch.randint(0, self.max_delay + 1, (n_resets,), device=self.device)
            self.delay_steps[env_ids] = rand_delays

        # Initialize buffer and prev_targets with current default/offset position
        default_pos = self._offset
        if isinstance(default_pos, torch.Tensor):
            init_val = default_pos[env_ids] if not isinstance(env_ids, slice) else default_pos
        else:
            init_val = default_pos

        self.action_buffer[:, env_ids] = init_val
        self.prev_targets[env_ids] = init_val

    def process_actions(self, actions: torch.Tensor):
        # 1. Store raw un-delayed policy output (for mdp.last_action observation)
        self._raw_actions[:] = actions

        # 2. Compute un-delayed joint target from policy action
        target_undelayed = self._raw_actions * self._scale + self._offset

        # 3. Update delay ring buffer (shift history forward)
        self.action_buffer[1:] = self.action_buffer[:-1].clone()
        self.action_buffer[0] = target_undelayed

        # 4. Gather delayed targets per environment based on self.delay_steps
        # delay_steps = 0 -> action_buffer[0] (no delay)
        # delay_steps = 1 -> action_buffer[1] (1 step delay)
        # delay_steps = 2 -> action_buffer[2] (2 step delay)
        delayed_targets = torch.zeros_like(target_undelayed)
        for d in range(self.buffer_len):
            mask = self.delay_steps == d
            if mask.any():
                delayed_targets[mask] = self.action_buffer[d, mask]

        # 5. Apply Joint Zero-Offset Calibration if active in env
        if hasattr(self._env, "joint_calibration_offsets"):
            calibrated_targets = delayed_targets + self._env.joint_calibration_offsets
        else:
            calibrated_targets = delayed_targets

        # 6. Apply Target Rate Limiter (maximum delta per 50 Hz step)
        delta = calibrated_targets - self.prev_targets
        clipped_delta = torch.clamp(delta, min=-self.rate_limit, max=self.rate_limit)

        # Log clipping frequency
        is_clipped = (delta != clipped_delta)
        self.total_clipped_joints += int(is_clipped.sum().item())
        self.total_commanded_joints += int(delta.numel())

        # Update applied processed actions and previous state
        self._processed_actions = self.prev_targets + clipped_delta
        self.prev_targets[:] = self._processed_actions

    @property
    def clip_fraction(self) -> float:
        """Fraction of joint target updates that were rate-limited."""
        if self.total_commanded_joints == 0:
            return 0.0
        return self.total_clipped_joints / self.total_commanded_joints


@configclass
class DelayedRateLimitedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for delayed and rate-limited joint position action term."""
    class_type: type = DelayedRateLimitedJointPositionAction

    max_delay_steps: int = 2
    """Maximum policy-step action delay (0 to max_delay_steps). Default is 2 steps (40 ms at 50 Hz)."""

    fixed_delay_steps: int | None = None
    """If set, all environments use this fixed delay (e.g. 0 for Play variant). If None, delay is randomized."""

    rate_limit_rad_per_step: float = math.radians(6.0)
    """Maximum joint position change per policy step (default 6 deg = 0.10472 rad per 50 Hz step)."""
