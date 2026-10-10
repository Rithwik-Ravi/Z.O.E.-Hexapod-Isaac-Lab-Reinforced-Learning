# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Export trained RSL-RL policy to deploy/ directory (ONNX, NumPy NPZ, YAML metadata, and reference rollout)."""

import argparse
import copy
import datetime
import os
import yaml
import numpy as np
import torch
import torch.nn as nn


class PolicyOnnxExporter(nn.Module):
    """Exporter wrapper combining empirical normalizer and actor MLP into a single ONNX graph."""

    def __init__(self, actor_mlp, normalizer=None):
        super().__init__()
        self.actor = copy.deepcopy(actor_mlp)
        if normalizer is not None and hasattr(normalizer, "_mean"):
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = nn.Identity()

    def forward(self, x):
        return self.actor(self.normalizer(x))

    def export(self, path: str, filename: str = "policy.onnx"):
        self.cpu()
        self.eval()
        in_dim = self.actor[0].in_features
        dummy_obs = torch.zeros(1, in_dim)
        os.makedirs(path, exist_ok=True)
        out_file = os.path.join(path, filename)

        torch.onnx.export(
            self,
            dummy_obs,
            out_file,
            export_params=True,
            opset_version=14,
            dynamo=False,
            input_names=["obs"],
            output_names=["actions"],
        )
        return out_file


def export_numpy_mlp(actor_module, normalizer_module, out_path: str):
    """Export policy MLP weights and normalizer statistics to NumPy .npz file."""
    actor_module.eval()
    actor_module.cpu()

    weights_dict = {}
    linear_idx = 0

    for layer in actor_module:
        if isinstance(layer, torch.nn.Linear):
            w = layer.weight.detach().cpu().numpy().T
            b = layer.bias.detach().cpu().numpy()
            weights_dict[f"W{linear_idx}"] = w
            weights_dict[f"b{linear_idx}"] = b
            linear_idx += 1

    if normalizer_module is not None and hasattr(normalizer_module, "_mean"):
        obs_mean = normalizer_module._mean.detach().cpu().numpy().squeeze(0)
        obs_var = normalizer_module._var.detach().cpu().numpy().squeeze(0)
        eps = float(getattr(normalizer_module, "eps", 1e-2))
    else:
        in_dim = weights_dict["W0"].shape[0]
        obs_mean = np.zeros(in_dim, dtype=np.float32)
        obs_var = np.ones(in_dim, dtype=np.float32)
        eps = 1e-2

    weights_dict["obs_mean"] = obs_mean
    weights_dict["obs_var"] = obs_var
    weights_dict["eps"] = np.array(eps, dtype=np.float32)
    weights_dict["activation"] = np.array("elu")

    np.savez(out_path, **weights_dict)
    print(f"[EXPORT] Saved NumPy MLP weights to: {out_path}")


def numpy_mlp_inference(npz_path: str, obs: np.ndarray) -> np.ndarray:
    """Pure NumPy forward pass for dependency-free Jetson inference."""
    data = np.load(npz_path)
    obs_mean = data["obs_mean"]
    obs_var = data["obs_var"]
    eps = float(data["eps"])

    obs_std = np.sqrt(obs_var)
    norm_obs = (obs - obs_mean) / (obs_std + eps)

    x = norm_obs
    i = 0
    while f"W{i}" in data:
        W = data[f"W{i}"]
        b = data[f"b{i}"]
        x = x @ W + b
        if f"W{i+1}" in data:
            x = np.where(x > 0, x, np.expm1(x))
        i += 1

    return x


def generate_reference_rollout(env, policy, num_steps=500) -> dict:
    """Generate 500-step golden simulation rollout with noise off."""
    env.reset()
    obs_list = []
    action_list = []
    cmd_list = []

    obs_res = env.get_observations()
    obs = obs_res[0] if isinstance(obs_res, tuple) else obs_res

    with torch.inference_mode():
        for _ in range(num_steps):
            if isinstance(obs, dict):
                pol_obs = obs["policy"]
            else:
                pol_obs = obs

            actions = policy(pol_obs)

            obs_list.append(pol_obs.cpu().numpy())
            action_list.append(actions.cpu().numpy())

            cmd = env.unwrapped.command_manager.get_command("base_velocity")
            cmd_list.append(cmd.cpu().numpy())

            step_res = env.step(actions)
            obs = step_res[0]

    rollout_data = {
        "obs": np.concatenate(obs_list, axis=0),
        "actions": np.concatenate(action_list, axis=0),
        "commands": np.concatenate(cmd_list, axis=0),
    }
    return rollout_data


def run_export(args_cli):
    """Main export execution function."""
    import git
    import gymnasium as gym
    from isaaclab.app import AppLauncher

    args_cli.headless = True
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import cli_args
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg
    import spdrbot3.tasks

    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[EXPORT] Loading checkpoint from: {checkpoint_path}")

    run_name = os.path.basename(os.path.dirname(checkpoint_path))
    out_dir = args_cli.out or os.path.join("deploy", run_name)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Parse Env & Agent Configs
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task.split(":")[-1], args_cli)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # 2. Load Policy with OnPolicyRunner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    runner.load(checkpoint_path)

    actor_critic = runner.alg.policy
    actor_mlp = actor_critic.actor
    normalizer = getattr(actor_critic, "actor_obs_normalizer", None)

    # 3. Export ONNX Policy with robust opset_version=14
    onnx_exporter = PolicyOnnxExporter(actor_mlp, normalizer)
    onnx_path = onnx_exporter.export(out_dir, filename="policy.onnx")
    print(f"[EXPORT] Saved ONNX policy to: {onnx_path}")

    # 4. Export NumPy MLP
    npz_path = os.path.join(out_dir, "policy_mlp.npz")
    export_numpy_mlp(actor_mlp, normalizer, npz_path)

    # 5. Extract Metadata for deploy.yaml
    unwrapped_env = env.unwrapped
    robot = unwrapped_env.scene["robot"]

    sim_joint_names = robot.joint_names
    num_joints = len(sim_joint_names)

    # Build Joint Order entries matching the ROS contract
    joint_order_entries = []
    for idx, j_name in enumerate(sim_joint_names):
        leg_num = None
        for c in j_name:
            if c.isdigit():
                leg_num = int(c)
                break

        j_type = "coxa"
        if "CUP" in j_name or "COXA" in j_name or "38" in j_name or "41" in j_name or "44" in j_name or "47" in j_name or "50" in j_name or "53" in j_name:
            j_type = "coxa"
            ros_name = f"NCCUP{leg_num}_joint"
        elif "THIGH" in j_name or "FEMUR" in j_name or "37" in j_name or "40" in j_name or "43" in j_name or "46" in j_name or "49" in j_name or "52" in j_name:
            j_type = "foot" if "THIGH" in j_name else "thigh"
            ros_name = f"NFOOT{leg_num}_joint" if j_type == "foot" else f"NTHIGH{leg_num}_joint"
        else:
            ros_name = f"{j_name}_joint"

        joint_order_entries.append({
            "sim": j_name,
            "ros": ros_name,
            "leg": leg_num or (idx // 3 + 1),
            "type": j_type,
        })

    default_joint_pos_list = robot.data.default_joint_pos[0].cpu().numpy().tolist()

    joint_limits_lower = [-1.5708] * num_joints
    joint_limits_upper = [1.5708] * num_joints

    obs_spec = [
        {"name": "base_ang_vel", "dim": 3, "scale": 0.25, "clip": [-5.0, 5.0], "frame": "base", "unit": "rad/s"},
        {"name": "projected_gravity", "dim": 3, "scale": 1.0},
        {"name": "velocity_commands", "dim": 3, "order": ["vx", "vy", "wz"]},
        {"name": "joint_pos_rel", "dim": num_joints, "scale": 1.0, "clip": [-5.0, 5.0]},
        {"name": "joint_vel", "dim": num_joints, "scale": 0.05, "clip": [-5.0, 5.0]},
        {"name": "last_action", "dim": num_joints},
    ]

    cmd_ranges = unwrapped_env.cfg.commands.base_velocity.ranges
    command_limits = {
        "vx": list(cmd_ranges.lin_vel_x),
        "vy": list(cmd_ranges.lin_vel_y),
        "wz": list(cmd_ranges.ang_vel_z),
    }

    try:
        repo = git.Repo(search_parent_directories=True)
        git_commit = repo.head.object.hexsha
    except Exception:
        git_commit = "unknown"

    deploy_yaml_data = {
        "schema_version": 1,
        "policy": {
            "onnx": "policy.onnx",
            "numpy": "policy_mlp.npz",
            "control_hz": int(1.0 / unwrapped_env.step_dt),
            "action_scale": float(unwrapped_env.cfg.actions.joint_pos.scale),
            "action_clip": None,
        },
        "joint_order": joint_order_entries,
        "default_joint_pos_rad": default_joint_pos_list,
        "joint_limits_rad": {
            "lower": joint_limits_lower,
            "upper": joint_limits_upper,
        },
        "obs_spec": obs_spec,
        "command_limits": command_limits,
        "actuator_assumptions": {
            "delay_policy_steps": [0, 2],
            "kp_scale": [0.8, 1.2],
            "status": "UNCALIBRATED",
        },
        "provenance": {
            "task": args_cli.task,
            "checkpoint": checkpoint_path,
            "git_commit": git_commit,
            "isaaclab_version": "v2.3.2",
            "date": datetime.datetime.now().isoformat(),
        },
    }

    yaml_path = os.path.join(out_dir, "deploy.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(deploy_yaml_data, f, sort_keys=False)
    print(f"[EXPORT] Saved deployment config to: {yaml_path}")

    print("[EXPORT] Generating reference simulation rollout (500 steps)...")
    inference_policy = runner.get_inference_policy(device=unwrapped_env.device)
    rollout_data = generate_reference_rollout(env, inference_policy, num_steps=500)
    ref_path = os.path.join(out_dir, "reference_rollout.npz")
    np.savez(ref_path, **rollout_data)
    print(f"[EXPORT] Saved reference rollout to: {ref_path}")

    print(f"\n✅ Successfully exported policy package to: {os.path.abspath(out_dir)}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    import cli_args
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Export RSL-RL policy for deployment.")
    parser.add_argument("--task", type=str, required=True, help="Task name e.g. Zoe-Hexapod-Flat-Deploy-v0")
    parser.add_argument("--out", type=str, default=None, help="Output directory e.g. deploy/run_01")
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()

    run_export(args_cli)
