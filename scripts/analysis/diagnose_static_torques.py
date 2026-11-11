# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Read-only simulator diagnostic explaining static holding torques for 6/5/4/3-foot stances at 5.0 kg and 4.0 kg."""

import argparse
import os
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect per-joint static holding torques at 5.0 kg and 4.0 kg.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
sim_app = app_launcher.app

import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import spdrbot3.tasks  # noqa: F401
from spdrbot3.tasks.manager_based.zoe_deploy.params import ZOE_PARAMS
from spdrbot3.tasks.manager_based.zoe_deploy.mdp.rewards import height_above_ground


def main():
    env_cfg = parse_env_cfg("Zoe-Hexapod-Flat-Deploy-Play-v0", device="cuda:0", num_envs=5)
    env_cfg.events.push_robot = None
    env_cfg.actions.joint_pos.rate_limit_rad_per_step = 10.0
    env = gym.make("Zoe-Hexapod-Flat-Deploy-Play-v0", cfg=env_cfg)
    unwrapped = env.unwrapped
    env.reset()

    robot = unwrapped.scene["robot"]
    base_masses = robot.root_physx_view.get_masses().clone()
    contact_sensor = unwrapped.scene["contact_forces"]
    joint_names = robot.joint_names
    body_names = contact_sensor.body_names

    thigh_idx, foot_j_idx, coxa_idx = {}, {}, {}
    for j_i, jn in enumerate(joint_names):
        if jn.startswith("NCCUP"):
            thigh_idx[int(jn.split("_")[0].replace("NCCUP", ""))] = j_i
        elif jn.startswith("NTHIGH"):
            foot_j_idx[int(jn.split("_")[0].replace("NTHIGH", ""))] = j_i
        elif jn.startswith("base_link"):
            info = ZOE_PARAMS.joint_info_by_sim_name.get(jn, {})
            if "leg" in info:
                coxa_idx[int(info["leg"])] = j_i

    foot_body_ids = [body_names.index(f"NFOOT{i}") for i in range(1, 7)]
    robot_foot_ids = [robot.body_names.index(f"NFOOT{i}") for i in range(1, 7)]
    thigh_body_ids = [robot.body_names.index(f"NTHIGH{i}") for i in range(1, 7)]
    base_q = robot.data.default_joint_pos[0].clone()
    r_local = torch.tensor(
        [ZOE_PARAMS.foot_tip_local_vectors[f"NFOOT{i}"] for i in range(1, 7)],
        device=robot.device,
        dtype=torch.float32,
    )

    # Probe thigh lift sign per leg
    lift_signs = {}
    for leg in range(1, 7):
        env.reset()
        q_test = base_q.clone()
        robot.write_joint_state_to_sim(q_test.unsqueeze(0).expand(5, -1), torch.zeros_like(robot.data.joint_vel))
        unwrapped.sim.forward()
        z0 = robot.data.body_pos_w[0, robot_foot_ids[leg - 1], 2].item()
        q_test[thigh_idx[leg]] += 0.35
        robot.write_joint_state_to_sim(q_test.unsqueeze(0).expand(5, -1), torch.zeros_like(robot.data.joint_vel))
        unwrapped.sim.forward()
        z_plus = robot.data.body_pos_w[0, robot_foot_ids[leg - 1], 2].item()
        lift_signs[leg] = 1.0 if z_plus > z0 else -1.0

    # Env 0..3 match eval_reward_economics; Env 4 tests alternate tripod (lift A=[1,3,5], stand on B=[2,4,6])
    lifted_sets = {0: [], 1: [2], 2: [2, 5], 3: [2, 4, 6], 4: [1, 3, 5]}
    stance_labels = [
        "6 feet (all down)",
        "5 feet (lift L2)",
        "4 feet (lift L2,L5)",
        "3 feet Tripod A down (lift L2,L4,L6)",
        "3 feet Tripod B down (lift L1,L3,L5)",
    ]

    for mass_scale in (1.0, 0.8):
        robot.root_physx_view.set_masses(base_masses * mass_scale, torch.arange(5, device="cpu"))
        total_mass = robot.root_physx_view.get_masses()[0].sum().item()
        env.reset()

        q_targets = base_q.unsqueeze(0).expand(5, -1).clone()
        for e_i, legs_up in lifted_sets.items():
            for leg in legs_up:
                q_targets[e_i, thigh_idx[leg]] += lift_signs[leg] * 0.55

        for _ in range(200):
            robot.set_joint_position_target(q_targets)
            for _ in range(unwrapped.cfg.decimation):
                unwrapped.scene.write_data_to_sim()
                unwrapped.sim.step(render=False)
                unwrapped.scene.update(dt=unwrapped.physics_dt)

        tau_s, fxyz_s, h_s, q_s = [], [], [], []
        for _ in range(25):
            robot.set_joint_position_target(q_targets)
            for _ in range(unwrapped.cfg.decimation):
                unwrapped.scene.write_data_to_sim()
                unwrapped.sim.step(render=False)
                unwrapped.scene.update(dt=unwrapped.physics_dt)
            tau_s.append(robot.data.applied_torque.clone())
            fxyz_s.append(contact_sensor.data.net_forces_w[:, foot_body_ids, :].clone())
            h_s.append(height_above_ground(unwrapped).clone())
            q_s.append(robot.data.joint_pos.clone())

        tau_m = torch.stack(tau_s).mean(dim=0)
        fxyz_m = torch.stack(fxyz_s).mean(dim=0)
        h_m = torch.stack(h_s).mean(dim=0)
        q_m = torch.stack(q_s).mean(dim=0)

        print(f"\n=== MASS SCALE {mass_scale:.2f} (TOTAL MASS = {total_mass:.4f} kg, Mg = {total_mass * 9.81:.2f} N) ===")
        for e_i, label in enumerate(stance_labels):
            bpos = robot.data.body_pos_w[e_i, robot_foot_ids, :]
            bquat = robot.data.body_quat_w[e_i, robot_foot_ids, :]
            qw, qv = bquat[:, 0:1], bquat[:, 1:4]
            a = torch.cross(qv, r_local, dim=-1)
            tip_w = bpos + r_local + 2.0 * qw * a + 2.0 * torch.cross(qv, a, dim=-1)
            thigh_w = robot.data.body_pos_w[e_i, thigh_body_ids, :]

            fz = fxyz_m[e_i, :, 2]
            fxy = torch.norm(fxyz_m[e_i, :, :2], dim=-1)
            max_abs_tau = torch.abs(tau_m[e_i]).max().item()
            feasible_250 = max_abs_tau < 2.499
            feasible_200 = max_abs_tau <= 2.000
            print(
                f"\n--- {label} | height={h_m[e_i].item():.4f} m | sum(Fz)={fz.sum().item():.2f} N | "
                f"max|tau|={max_abs_tau:.3f} N.m (unsat<2.50: {feasible_250}, <=2.00: {feasible_200}) ---"
            )
            print("Leg | Fz (N) | Fxy (N) | Coxa tau | Thigh tau | Foot tau | q_thigh | q_foot  | r_xy(tip-thigh) | r_xy(tip-foot)")
            for leg in range(1, 7):
                l0 = leg - 1
                r_th = torch.norm(tip_w[l0, :2] - thigh_w[l0, :2]).item()
                r_ft = torch.norm(tip_w[l0, :2] - bpos[l0, :2]).item()
                print(
                    f" L{leg} | {fz[l0]:6.2f} | {fxy[l0]:7.2f} | "
                    f"{tau_m[e_i, coxa_idx[leg]]:+8.3f} | "
                    f"{tau_m[e_i, thigh_idx[leg]]:+9.3f} | "
                    f"{tau_m[e_i, foot_j_idx[leg]]:+8.3f} | "
                    f"{q_m[e_i, thigh_idx[leg]]:+7.4f} | "
                    f"{q_m[e_i, foot_j_idx[leg]]:+7.4f} | "
                    f"{r_th:15.4f} | {r_ft:12.4f}"
                )

    os._exit(0)


if __name__ == "__main__":
    main()
