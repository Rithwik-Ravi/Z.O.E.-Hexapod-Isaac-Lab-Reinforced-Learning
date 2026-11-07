# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Isolation repro for Stage 2 task:
(1) Run 100 iterations with 256 envs on single flat sub-terrain.
(2) If that works, run 100 iterations on slopes levels 0-2.
"""

import argparse
import os
import sys
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isolation repro for Stage 2 diagnosis.")
parser.add_argument("--mode", type=str, default="flat_only", choices=["flat_only", "slopes_only"])
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--iterations", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
sim_app = app_launcher.app

import gymnasium as gym
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

import spdrbot3.tasks
from spdrbot3.tasks.manager_based.zoe_deploy.agents.rsl_rl_ppo_cfg import ZoeTerrainPPORunnerCfg

def run_isolation():
    print(f"\n========================================================")
    print(f"RUNNING ISOLATION REPRO: {args_cli.mode} ({args_cli.num_envs} envs, {args_cli.iterations} iters)")
    print(f"========================================================\n")

    env_cfg = parse_env_cfg("Zoe-Hexapod-Terrain-v0", num_envs=args_cli.num_envs)

    if args_cli.mode == "flat_only":
        # Replace terrain generator with 100% flat
        env_cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=10,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "flat": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=1.0, noise_range=(0.0, 0.0), noise_step=0.02, border_width=0.25
                ),
            },
        )
    elif args_cli.mode == "slopes_only":
        # Slopes only with gentle difficulty
        env_cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=10,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
                    proportion=0.5, slope_range=(0.0, 0.15), platform_width=2.0
                ),
                "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                    proportion=0.5, slope_range=(0.0, 0.15), platform_width=2.0
                ),
            },
        )

    # Disable max_init_terrain_level = None to start at level 0
    # BUT let us see how it behaves if kept as-is or set to 0
    # In this repro, we keep env_cfg as configured in Stage 2 to isolate exactly what fails
    agent_cfg = ZoeTerrainPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.iterations
    agent_cfg.num_steps_per_env = 48
    agent_cfg.save_interval = 50
    agent_cfg.experiment_name = f"diag_isolation_{args_cli.mode}"

    log_dir = os.path.join("/home/rithwik/SpdrBot-main/logs/rsl_rl", agent_cfg.experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make("Zoe-Hexapod-Terrain-v0", cfg=env_cfg)
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    wrapped_env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.learn(num_learning_iterations=args_cli.iterations, init_at_random_ep_len=True)

    print("\nIsolation run finished. Reading results...")
    env.close()

if __name__ == "__main__":
    run_isolation()
    sim_app.close()

