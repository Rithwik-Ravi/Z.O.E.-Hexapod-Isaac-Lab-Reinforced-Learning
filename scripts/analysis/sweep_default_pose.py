# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Default Pose Sweep, Zero-Action Stance Test, and 4/5/6 kg Torque Feasibility for Zoe Hexapod (hexa_ir)."""

import argparse
import math
import os
import sys
import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sweep default poses and mass/torque feasibility for Zoe Hexapod.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = True
app_launcher = AppLauncher(args_cli)
sim_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor, ContactSensorCfg

for _rel in ("../source/spdrbot3", "../../source/spdrbot3"):
    _cand = os.path.abspath(os.path.join(os.path.dirname(__file__), _rel))
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)
from spdrbot3.assets.zoe_hexapod import ZOE_HEXAPOD_CFG
from spdrbot3.tasks.manager_based.zoe_deploy.params import ZOE_PARAMS, get_foot_tip_offsets_tensor


def compute_foot_tip_positions(robot, foot_ids, foot_names=None):
    """Compute foot tip 3D coordinates in world frame using per-leg 3D local tip vectors."""
    body_pos = robot.data.body_pos_w[:, foot_ids, :]
    body_quat = robot.data.body_quat_w[:, foot_ids, :]

    if foot_names is None:
        foot_names = [f"NFOOT{i}" for i in range(1, len(foot_ids) + 1)]
    r_local = get_foot_tip_offsets_tensor(foot_names, body_pos.device, body_pos.dtype)
    r_local = r_local.unsqueeze(0).expand(body_pos.shape[0], -1, 3)

    q_w = body_quat[..., 0:1]
    q_vec = body_quat[..., 1:4]
    a = torch.cross(q_vec, r_local, dim=-1)
    v_rot = r_local + 2.0 * q_w * a + 2.0 * torch.cross(q_vec, a, dim=-1)
    return body_pos + v_rot


def set_robot_total_mass(robot, nominal_masses_cpu: torch.Tensor, target_total_mass_kg: float):
    """Scale all link masses proportionally to achieve target_total_mass_kg."""
    scale = target_total_mass_kg / float(nominal_masses_cpu.sum().item())
    new_masses = (nominal_masses_cpu * scale).clone()
    env_indices = torch.arange(robot.num_instances, dtype=torch.int32, device="cpu")
    robot.root_physx_view.set_masses(new_masses, env_indices)


def evaluate_stance(
    sim,
    robot,
    contact_sensor,
    nominal_masses_cpu,
    candidate_name: str,
    joint_targets_dict: dict,
    total_mass_kg: float = 5.0,
    kp_val: float = ZOE_PARAMS.ACTUATOR_KP,
    kd_val: float = ZOE_PARAMS.ACTUATOR_KD,
):
    """Evaluate 6-foot, 4-foot, and 3-foot stance at given mass and gains."""
    set_robot_total_mass(robot, nominal_masses_cpu, total_mass_kg)

    for act_name in ["coxa_joints", "thigh_joints", "foot_joints"]:
        robot.actuators[act_name].stiffness[:] = kp_val
        robot.actuators[act_name].damping[:] = kd_val

    body_names = contact_sensor.body_names
    foot_names = [f"NFOOT{i}" for i in range(1, 7)]
    foot_ids = [body_names.index(n) for n in foot_names]
    robot_foot_ids = [robot.body_names.index(n) for n in foot_names]
    non_foot_ids = [i for i, n in enumerate(body_names) if not n.startswith("NFOOT")]

    def _reset_and_settle(target_jpos):
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] = torch.tensor([0.0, 0.0, 0.17], device=sim.device)
        root_state[:, 3:7] = torch.tensor(list(ZOE_PARAMS.UPRIGHT_ROOT_QUAT_WXYZ), device=sim.device)
        root_state[:, 7:] = 0.0
        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(target_jpos, torch.zeros_like(target_jpos))
        robot.set_joint_position_target(target_jpos)
        for _ in range(150):
            robot.write_data_to_sim()
            sim.step()
            robot.update(dt=0.005)
            contact_sensor.update(dt=0.005)

    # 1. 6-Foot Zero-Action Stance
    joint_pos_6ft = torch.zeros((1, robot.num_joints), device=sim.device)
    for j_idx, j_name in enumerate(robot.joint_names):
        for pattern, val in joint_targets_dict.items():
            if pattern in j_name:
                joint_pos_6ft[0, j_idx] = val

    _reset_and_settle(joint_pos_6ft)

    base_pos = robot.data.root_pos_w[0].cpu().numpy()
    actual_jpos = robot.data.joint_pos[0].cpu().numpy()
    target_jpos_np = joint_pos_6ft[0].cpu().numpy()
    torques_6ft = np.abs(robot.data.applied_torque[0].cpu().numpy())

    sag_deg = np.rad2deg(np.abs(actual_jpos - target_jpos_np))
    max_sag_deg = float(np.max(sag_deg))
    mean_sag_deg = float(np.mean(sag_deg))

    tip_pos_w = compute_foot_tip_positions(robot, robot_foot_ids, foot_names)[0].cpu().numpy()
    foot_radii = np.linalg.norm(tip_pos_w[:, :2] - base_pos[:2], axis=-1)

    net_forces_foot = contact_sensor.data.net_forces_w[0, foot_ids].cpu().numpy()
    fz = net_forces_foot[:, 2]
    feet_down_6ft = int(np.sum(fz > 1.0))
    non_foot_forces_z = contact_sensor.data.net_forces_w[0, non_foot_ids, 2].cpu().numpy()
    non_foot_contact = bool(np.any(non_foot_forces_z > 1.0))

    sum_fz = float(np.sum(fz))
    gravity_weight = total_mass_kg * 9.81
    peak_tau_6ft = float(np.max(torques_6ft))

    # 2. 4-Foot Stance (lift opposite legs 1 and 4 by rotating thigh upward: in hexa.urdf, -thigh lifts foot up)
    joint_pos_4ft = joint_pos_6ft.clone()
    for j_idx, j_name in enumerate(robot.joint_names):
        if any(f"NCCUP{leg}" in j_name for leg in [1, 4]):
            joint_pos_4ft[0, j_idx] = -0.45
    robot.set_joint_position_target(joint_pos_4ft)
    for _ in range(120):
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt=0.005)
        contact_sensor.update(dt=0.005)
    torques_4ft = np.abs(robot.data.applied_torque[0].cpu().numpy())
    grounded_4ft_joint_mask = [
        idx for idx, j_name in enumerate(robot.joint_names)
        if not any(f"{p}{leg}" in j_name or (leg == 1 and "_34" in j_name) or (leg == 4 and "_42" in j_name) for leg in [1, 4] for p in ["NCCUP", "NTHIGH"])
    ]
    peak_tau_4ft = float(np.max(torques_4ft[grounded_4ft_joint_mask]))
    height_4ft = float(robot.data.root_pos_w[0, 2].item())

    # 3. 3-Foot Tripod Stance (lift legs 2, 4, 6; keep tripod {1, 3, 5} grounded)
    _reset_and_settle(joint_pos_6ft)
    joint_pos_3ft = joint_pos_6ft.clone()
    for j_idx, j_name in enumerate(robot.joint_names):
        if any(f"NCCUP{leg}" in j_name for leg in [2, 4, 6]):
            joint_pos_3ft[0, j_idx] = -0.45
    robot.set_joint_position_target(joint_pos_3ft)
    for _ in range(120):
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt=0.005)
        contact_sensor.update(dt=0.005)
    torques_3ft = np.abs(robot.data.applied_torque[0].cpu().numpy())
    grounded_3ft_joint_mask = [
        idx for idx, j_name in enumerate(robot.joint_names)
        if any(f"NCCUP{leg}" in j_name or f"NTHIGH{leg}" in j_name or (leg == 1 and "_34" in j_name) or (leg == 3 and "_45" in j_name) or (leg == 5 and "_39" in j_name) for leg in [1, 3, 5])
    ]
    peak_tau_3ft = float(np.max(torques_3ft[grounded_3ft_joint_mask]))
    height_3ft = float(robot.data.root_pos_w[0, 2].item())

    # Analytical static thigh moment-arm torque (L_thigh = 0.0754 m)
    l_thigh = 0.0754
    static_tau_6ft = (gravity_weight / 6.0) * l_thigh
    static_tau_4ft = (gravity_weight / 4.0) * l_thigh
    static_tau_3ft = (gravity_weight / 3.0) * l_thigh

    effort_lim = ZOE_PARAMS.EFFORT_LIMIT_NM       # 2.50 N.m
    stall_lim = ZOE_PARAMS.STALL_TORQUE_12V_NM    # 2.942 N.m

    return {
        "candidate": candidate_name,
        "mass_kg": total_mass_kg,
        "kp": kp_val,
        "kd": kd_val,
        "base_height_6ft": float(base_pos[2]),
        "base_height_4ft": height_4ft,
        "base_height_3ft": height_3ft,
        "feet_down_6ft": feet_down_6ft,
        "non_foot_contact": non_foot_contact,
        "max_sag_deg": max_sag_deg,
        "mean_sag_deg": mean_sag_deg,
        "radii_mean": float(np.mean(foot_radii)),
        "radii_span": float(np.ptp(foot_radii)),
        "peak_tau_6ft": peak_tau_6ft,
        "peak_tau_4ft": peak_tau_4ft,
        "peak_tau_3ft": peak_tau_3ft,
        "static_tau_6ft": static_tau_6ft,
        "static_tau_4ft": static_tau_4ft,
        "static_tau_3ft": static_tau_3ft,
        "margin_effort_6ft_pct": (1.0 - peak_tau_6ft / effort_lim) * 100.0,
        "margin_effort_4ft_pct": (1.0 - peak_tau_4ft / effort_lim) * 100.0,
        "margin_effort_3ft_pct": (1.0 - peak_tau_3ft / effort_lim) * 100.0,
        "margin_stall_3ft_pct": (1.0 - peak_tau_3ft / stall_lim) * 100.0,
        "sum_fz": sum_fz,
        "fz_ratio": sum_fz / gravity_weight,
    }


def run_sweep():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.005)
    sim = sim_utils.SimulationContext(sim_cfg)

    plane_cfg = sim_utils.GroundPlaneCfg()
    plane_cfg.func("/World/ground", plane_cfg)

    art_cfg = ZOE_HEXAPOD_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(art_cfg)

    contact_cfg = ContactSensorCfg(
        prim_path="/World/Robot/.*",
        update_period=0.0,
        history_length=3,
        force_threshold=1.0,
    )
    contact_sensor = ContactSensor(contact_cfg)

    sim.reset()
    robot.update(dt=0.005)
    contact_sensor.update(dt=0.005)

    nominal_masses_cpu = robot.root_physx_view.get_masses().clone()
    default_pose_targets = {
        "base_link_Revolute": 0.0,
        "NCCUP": 0.0,
        "NTHIGH": -math.pi / 2.0,
    }

    # Part A: Mass & Torque Feasibility Sweep across 4.0, 4.1662 (CAD), 5.0 (Assumed), and 6.0 kg
    mass_results = []
    for m_kg in [4.0, ZOE_PARAMS.TOTAL_ASSET_MASS_KG, 5.0, 6.0]:
        res = evaluate_stance(
            sim,
            robot,
            contact_sensor,
            nominal_masses_cpu,
            candidate_name=f"Default (0, 0, -pi/2) @ {m_kg:.2f}kg",
            joint_targets_dict=default_pose_targets,
            total_mass_kg=m_kg,
            kp_val=16.0,
            kd_val=1.6,
        )
        mass_results.append(res)

    # Part B: Candidate Pose & Gain Sweep at Assumed 5.0 kg
    pose_results = []
    for kp in [16.0, 25.0, 35.0]:
        pose_results.append(
            evaluate_stance(
                sim,
                robot,
                contact_sensor,
                nominal_masses_cpu,
                candidate_name=f"Cand 0: (0, 0, -pi/2) Kp={kp:.0f}",
                joint_targets_dict=default_pose_targets,
                total_mass_kg=5.0,
                kp_val=kp,
                kd_val=kp / 10.0,
            )
        )

    cand_list = [
        ("Cand 1: (0, -0.10, -pi/2+0.10)", {"base_link_Revolute": 0.0, "NCCUP": -0.10, "NTHIGH": -math.pi / 2.0 + 0.10}),
        ("Cand 2: (0, +0.10, -pi/2-0.10)", {"base_link_Revolute": 0.0, "NCCUP": 0.10, "NTHIGH": -math.pi / 2.0 - 0.10}),
        ("Cand 3: (0, 0.00, -pi/2+0.10)", {"base_link_Revolute": 0.0, "NCCUP": 0.0, "NTHIGH": -math.pi / 2.0 + 0.10}),
    ]
    for cname, cdict in cand_list:
        pose_results.append(
            evaluate_stance(
                sim,
                robot,
                contact_sensor,
                nominal_masses_cpu,
                candidate_name=cname,
                joint_targets_dict=cdict,
                total_mass_kg=5.0,
                kp_val=16.0,
                kd_val=1.6,
            )
        )

    print("\n" + "=" * 115)
    print("  PART A: TORQUE FEASIBILITY AT 4.0, 4.17 (CAD), 5.0 (ASSUMED), AND 6.0 KG (12V Effort=2.50 Nm, Stall=2.942 Nm)")
    print("=" * 115)
    print(
        f"{'Mass (kg)':<10} | {'H_6ft (m)':<9} | {'Feet':<4} | {'Sag(max)':<8} | "
        f"{'6ft Tau (Margin)':<20} | {'4ft Tau (Margin)':<20} | {'3ft Tau (Margin vs 2.5/2.94Nm)':<30}"
    )
    print("-" * 115)
    for r in mass_results:
        print(
            f"{r['mass_kg']:<10.4f} | {r['base_height_6ft']:<9.4f} | {r['feet_down_6ft']}/6  | {r['max_sag_deg']:<6.2f}° | "
            f"{r['peak_tau_6ft']:.3f} Nm ({r['margin_effort_6ft_pct']:+5.1f}%) | "
            f"{r['peak_tau_4ft']:.3f} Nm ({r['margin_effort_4ft_pct']:+5.1f}%) | "
            f"{r['peak_tau_3ft']:.3f} Nm ({r['margin_effort_3ft_pct']:+5.1f}% / {r['margin_stall_3ft_pct']:+5.1f}%)"
        )
    print("=" * 115)

    print("\n" + "=" * 115)
    print("  PART B: DEFAULT POSE & GAIN SWEEP AT 5.0 KG")
    print("=" * 115)
    print(
        f"{'Candidate':<32} | {'H_6ft':<7} | {'Feet':<4} | {'Sag(max)':<8} | {'Radius(span)':<15} | "
        f"{'6ft Tau':<9} | {'4ft Tau':<9} | {'3ft Tau':<9} | {'Fz/Mg':<6}"
    )
    print("-" * 115)
    for r in pose_results:
        print(
            f"{r['candidate']:<32} | {r['base_height_6ft']:<7.4f} | {r['feet_down_6ft']}/6  | {r['max_sag_deg']:<6.2f}° | "
            f"{r['radii_mean']:.4f} ({r['radii_span']:.4f}) | "
            f"{r['peak_tau_6ft']:<7.3f}Nm | {r['peak_tau_4ft']:<7.3f}Nm | {r['peak_tau_3ft']:<7.3f}Nm | {r['fz_ratio']:<6.3f}"
        )
    print("=" * 115 + "\n")

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    run_sweep()

