# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for A5: scripts/rsl_rl/train.py and cli_args.py hardening."""

import argparse
import os
import sys
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_RSL_RL_SCRIPTS = os.path.join(_REPO_ROOT, "scripts/rsl_rl")
if _RSL_RL_SCRIPTS not in sys.path:
    sys.path.insert(0, _RSL_RL_SCRIPTS)

import cli_args


class _DummyAgentCfg:
    def __init__(self):
        self.seed = 42
        self.resume = False
        self.load_run = ".*"
        self.load_checkpoint = "model_.*.pt"
        self.run_name = None
        self.logger = "tensorboard"
        self.resume_from = None


def test_direct_checkpoint_cross_experiment_sets_resume_true(tmp_path):
    """A5(a): --checkpoint accepts a direct file path from any experiment_name and sets resume=True."""
    other_exp_dir = tmp_path / "logs" / "rsl_rl" / "zoe_hexapod_deploy" / "2026-11-05_19-51-02"
    other_exp_dir.mkdir(parents=True)
    ckpt_file = other_exp_dir / "model_599.pt"
    ckpt_file.write_bytes(b"dummy_weights")

    terrain_log_root = str(tmp_path / "logs" / "rsl_rl" / "zoe_hexapod_terrain")

    parser = argparse.ArgumentParser()
    cli_args.add_rsl_rl_args(parser)
    args = parser.parse_args(["--checkpoint", str(ckpt_file)])

    agent_cfg = _DummyAgentCfg()
    cli_args.update_rsl_rl_cfg(agent_cfg, args)
    assert agent_cfg.resume is True

    resolved = cli_args.resolve_resume_checkpoint(terrain_log_root, agent_cfg, args)
    assert resolved == str(ckpt_file.resolve())
    assert agent_cfg.resume is True
    assert agent_cfg.resume_from == str(ckpt_file.resolve())


def test_missing_checkpoint_or_run_fails_loudly(tmp_path):
    """A5(b): If --resume or --checkpoint is given and file/run does not exist, fail loudly."""
    terrain_log_root = str(tmp_path / "logs" / "rsl_rl" / "zoe_hexapod_terrain")
    parser = argparse.ArgumentParser()
    cli_args.add_rsl_rl_args(parser)

    # Case 1: Non-existent direct --checkpoint path
    args_bad_ckpt = parser.parse_args(["--checkpoint", str(tmp_path / "missing" / "model_599.pt")])
    agent_cfg1 = _DummyAgentCfg()
    cli_args.update_rsl_rl_cfg(agent_cfg1, args_bad_ckpt)
    with pytest.raises(FileNotFoundError):
        cli_args.resolve_resume_checkpoint(terrain_log_root, agent_cfg1, args_bad_ckpt)

    # Case 2: --resume when experiment directory does not exist
    args_resume_missing_dir = parser.parse_args(["--resume", "--load_run", "2026-11-05_19-51-02"])
    agent_cfg2 = _DummyAgentCfg()
    cli_args.update_rsl_rl_cfg(agent_cfg2, args_resume_missing_dir)
    with pytest.raises(FileNotFoundError):
        cli_args.resolve_resume_checkpoint(terrain_log_root, agent_cfg2, args_resume_missing_dir)

    # Case 3: --resume when experiment directory exists but requested run does not
    os.makedirs(terrain_log_root, exist_ok=True)

    def _failing_get_ckpt(root, run, ckpt):
        raise ValueError(f"No runs present in {root} match {run}")

    with pytest.raises(FileNotFoundError):
        cli_args.resolve_resume_checkpoint(
            terrain_log_root, agent_cfg2, args_resume_missing_dir, get_checkpoint_path_fn=_failing_get_ckpt
        )


def test_from_scratch_when_no_resume_flags(tmp_path):
    """A5(c): Without --resume / --checkpoint / --load_run, resolve_resume_checkpoint returns None."""
    terrain_log_root = str(tmp_path / "logs" / "rsl_rl" / "zoe_hexapod_terrain")
    parser = argparse.ArgumentParser()
    cli_args.add_rsl_rl_args(parser)
    args = parser.parse_args([])

    agent_cfg = _DummyAgentCfg()
    cli_args.update_rsl_rl_cfg(agent_cfg, args)
    resolved = cli_args.resolve_resume_checkpoint(terrain_log_root, agent_cfg, args)
    assert resolved is None
    assert agent_cfg.resume is False
    assert agent_cfg.resume_from is None


def test_early_abort_watchdog_conditions():
    """A5(d): Test watchdog abort conditions at iteration 150 and LR collapse warning."""
    rising_rewards = [float(i) * 0.1 for i in range(150)]
    falling_rewards = [-float(i) * 0.1 for i in range(150)]

    # 1. Before iteration 150, do not abort even if metrics are poor
    abort, reasons, warn = cli_args.evaluate_watchdog(
        iters_run=100,
        watchdog_iters=150,
        mean_ep_len=22.0,
        time_out_rate=0.0,
        reward_history=falling_rewards[:100],
        learning_rate=3e-4,
    )
    assert abort is False
    assert reasons == []
    assert warn is None

    # 2. At iteration 150: healthy metrics -> do not abort
    abort, reasons, warn = cli_args.evaluate_watchdog(
        iters_run=150,
        watchdog_iters=150,
        mean_ep_len=850.0,
        time_out_rate=0.82,
        reward_history=rising_rewards,
        learning_rate=3e-4,
    )
    assert abort is False
    assert reasons == []

    # 3. At iteration 150: mean_episode_length < 100 -> abort
    abort, reasons, _ = cli_args.evaluate_watchdog(
        iters_run=150,
        watchdog_iters=150,
        mean_ep_len=22.3,
        time_out_rate=0.50,
        reward_history=rising_rewards,
        learning_rate=3e-4,
    )
    assert abort is True
    assert any("mean_episode_length" in r for r in reasons)

    # 4. At iteration 150: time_out < 5% -> abort
    abort, reasons, _ = cli_args.evaluate_watchdog(
        iters_run=150,
        watchdog_iters=150,
        mean_ep_len=250.0,
        time_out_rate=0.02,
        reward_history=rising_rewards,
        learning_rate=3e-4,
    )
    assert abort is True
    assert any("time_out" in r for r in reasons)

    # 5. At iteration 150: mean_reward still falling -> abort
    abort, reasons, _ = cli_args.evaluate_watchdog(
        iters_run=150,
        watchdog_iters=150,
        mean_ep_len=500.0,
        time_out_rate=0.50,
        reward_history=falling_rewards,
        learning_rate=3e-4,
    )
    assert abort is True
    assert any("mean_reward is still falling" in r for r in reasons)

    # 6. LR < 1e-5 -> warning emitted
    _, _, warn = cli_args.evaluate_watchdog(
        iters_run=50,
        watchdog_iters=150,
        mean_ep_len=500.0,
        time_out_rate=0.50,
        reward_history=rising_rewards[:50],
        learning_rate=5e-6,
    )
    assert warn is not None and "collapsed below 1e-5" in warn
