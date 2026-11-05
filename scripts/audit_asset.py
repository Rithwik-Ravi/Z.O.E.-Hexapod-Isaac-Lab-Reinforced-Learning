# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Asset audit script for Z.O.E. Hexapod USD asset."""

import argparse
import json
import math
import os
import struct
import sys
import xml.etree.ElementTree as ET
import numpy as np

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Audit USD asset for Z.O.E. Hexapod.")
parser.add_argument(
    "--usd",
    type=str,
    default=os.environ.get("ZOE_USD_PATH", "hexapod/hexa/hexa.usd"),
    help="Path to the USD file.",
)
parser.add_argument(
    "--urdf",
    type=str,
    default="hexapod/hexa.urdf",
    help="Path to companion URDF file for exact joint axes and mesh tip vectors.",
)
parser.add_argument(
    "--real-mass-kg",
    type=float,
    default=5.0,
    help="Expected real/assumed mass of the robot in kg for comparison.",
)
parser.add_argument(
    "--out",
    type=str,
    default="deploy/asset_report.json",
    help="Output JSON path for the asset report.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# default to headless if not specified
if "--headless" not in sys.argv:
    args_cli.headless = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
from pxr import Usd, UsdGeom, UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg


def derive_ros_joint_name(child_body_name: str) -> str:
    """Auto-derive ROS joint name from child body name."""
    if child_body_name and not child_body_name.startswith("base_link"):
        return f"{child_body_name}_joint"
    return "unknown_joint"


def determine_joint_type(child_body: str) -> str:
    """Determine joint type (coxa|thigh|foot) from child body name."""
    child_upper = child_body.upper()
    if "CUP" in child_upper or "COXA" in child_upper:
        return "coxa"
    elif "THIGH" in child_upper or "FEMUR" in child_upper:
        return "thigh"
    elif "FOOT" in child_upper or "TIBIA" in child_upper or "SHANK" in child_upper:
        return "foot"
    return "unknown"


def compute_servo_id(leg: int | None, j_type: str) -> int | None:
    """Compute Waveshare ST3215 servo ID: 3*(leg-1) + k, k=1 coxa, 2 thigh, 3 foot."""
    if leg is None:
        return None
    k_map = {"coxa": 1, "thigh": 2, "foot": 3}
    k = k_map.get(j_type)
    if k is None:
        return None
    return 3 * (leg - 1) + k


def _rpy_to_mat(r: float, p: float, y: float) -> np.ndarray:
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ Ry @ Rx


def _axis_angle_to_mat(axis, angle: float) -> np.ndarray:
    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        return np.eye(3)
    u = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    ux, uy, uz = u
    K = np.array([[0.0, -uz, uy], [uz, 0.0, -ux], [-uy, ux, 0.0]])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def _read_stl_verts(stl_path: str) -> np.ndarray:
    with open(stl_path, "rb") as f:
        data = f.read()
    num_tri = struct.unpack("<I", data[80:84])[0]
    verts = np.zeros((num_tri * 3, 3), dtype=np.float64)
    offset = 84
    for i in range(num_tri):
        v = struct.unpack("<12f", data[offset : offset + 48])[3:]
        verts[i * 3] = v[0:3]
        verts[i * 3 + 1] = v[3:6]
        verts[i * 3 + 2] = v[6:9]
        offset += 50
    return verts * 0.001


def extract_urdf_kinematics(urdf_path: str) -> dict:
    """Extract exact joint metadata, link lengths, and local foot-tip vectors from URDF + STL meshes."""
    if not os.path.exists(urdf_path):
        return {}
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    root = ET.parse(urdf_path).getroot()

    links_vis = {}
    for l_el in root.findall("link"):
        name = l_el.attrib["name"]
        vis = l_el.find("visual")
        if vis is not None:
            orig = vis.find("origin")
            xyz = [float(x) for x in orig.attrib.get("xyz", "0 0 0").split()] if orig is not None else [0.0, 0.0, 0.0]
            rpy = [float(x) for x in orig.attrib.get("rpy", "0 0 0").split()] if orig is not None else [0.0, 0.0, 0.0]
            mesh_el = vis.find("geometry/mesh")
            mesh_fn = mesh_el.attrib["filename"] if mesh_el is not None else ""
            links_vis[name] = {"xyz": np.array(xyz), "rpy": rpy, "mesh": os.path.join(urdf_dir, mesh_fn)}

    urdf_joints_by_sim_name = {}
    joints_by_child = {}
    for j_el in root.findall("joint"):
        if j_el.attrib.get("type") == "fixed":
            continue
        raw_name = j_el.attrib["name"]
        sim_name = raw_name.replace("-", "_")
        parent = j_el.find("parent").attrib["link"]
        child = j_el.find("child").attrib["link"]
        orig = j_el.find("origin")
        xyz = [float(x) for x in orig.attrib.get("xyz", "0 0 0").split()] if orig is not None else [0.0, 0.0, 0.0]
        rpy = [float(x) for x in orig.attrib.get("rpy", "0 0 0").split()] if orig is not None else [0.0, 0.0, 0.0]
        axis_el = j_el.find("axis")
        axis = [float(x) for x in axis_el.attrib.get("xyz", "0 0 1").split()] if axis_el is not None else [0.0, 0.0, 1.0]
        lim_el = j_el.find("limit")
        if lim_el is not None:
            lim = [float(lim_el.attrib.get("lower", "-inf")), float(lim_el.attrib.get("upper", "inf"))]
        else:
            lim = [-float("inf"), float("inf")]

        entry = {
            "urdf_name": raw_name,
            "sim_name": sim_name,
            "parent": parent,
            "child": child,
            "xyz": xyz,
            "rpy": rpy,
            "axis": axis,
            "limits_rad": lim,
        }
        urdf_joints_by_sim_name[sim_name] = entry
        joints_by_child[child] = entry

    # Compute link lengths, foot radii, and 3D foot tip vectors in NFOOT1..6 link frames
    # In the new URDF, base_link has -Z as UP, so upright world orientation is 180-deg X-roll: diag(1, -1, -1)
    R_x180 = np.diag([1.0, -1.0, -1.0])
    q_stand_foot = -math.pi / 2.0

    coxa_mount_radii = {}
    coxa_mount_angles_deg = {}
    coxa_lengths_3d = {}
    thigh_lengths_3d = {}
    tibia_lengths_max = {}
    foot_tip_local_vectors = {}
    standing_foot_radii = {}
    all_zero_foot_radii = {}
    standing_tip_z_rel_base = {}
    all_zero_tip_z_rel_base = {}

    for leg in range(1, 7):
        c_name, t_name, f_name = f"NCCUP{leg}", f"NTHIGH{leg}", f"NFOOT{leg}"
        if c_name not in joints_by_child or t_name not in joints_by_child or f_name not in joints_by_child:
            continue
        jc, jt, jf = joints_by_child[c_name], joints_by_child[t_name], joints_by_child[f_name]

        p_c = R_x180 @ np.array(jc["xyz"])
        R_c = R_x180 @ _rpy_to_mat(*jc["rpy"])

        p_t = p_c + R_c @ np.array(jt["xyz"])
        R_t = R_c @ _rpy_to_mat(*jt["rpy"])

        p_f = p_t + R_t @ np.array(jf["xyz"])
        R_f_zero = R_t @ _rpy_to_mat(*jf["rpy"])
        R_f_stand = R_f_zero @ _axis_angle_to_mat(jf["axis"], q_stand_foot)

        coxa_mount_radii[f"leg_{leg}"] = round(float(np.linalg.norm(p_c[:2])), 4)
        coxa_mount_angles_deg[f"leg_{leg}"] = round(float(math.degrees(math.atan2(p_c[1], p_c[0]))), 2)
        coxa_lengths_3d[f"leg_{leg}"] = round(float(np.linalg.norm(jt["xyz"])), 4)
        thigh_lengths_3d[f"leg_{leg}"] = round(float(np.linalg.norm(jf["xyz"])), 4)

        if f_name in links_vis and os.path.exists(links_vis[f_name]["mesh"]):
            f_vis = links_vis[f_name]
            verts_raw = _read_stl_verts(f_vis["mesh"])
            R_vis = _rpy_to_mat(*f_vis["rpy"])
            verts_link = (verts_raw @ R_vis.T) + f_vis["xyz"]

            dists = np.linalg.norm(verts_link, axis=1)
            tibia_lengths_max[f"leg_{leg}"] = round(float(np.max(dists)), 4)

            # World vertices at standing pose (q_foot = -pi/2)
            verts_w_stand = (verts_link @ R_f_stand.T) + p_f
            min_z_idx = int(np.argmin(verts_w_stand[:, 2]))
            tip_link_vec = verts_link[min_z_idx]
            tip_w_stand = verts_w_stand[min_z_idx]

            foot_tip_local_vectors[f_name] = [round(float(x), 5) for x in tip_link_vec]
            standing_foot_radii[f"leg_{leg}"] = round(float(np.linalg.norm(tip_w_stand[:2])), 4)
            standing_tip_z_rel_base[f"leg_{leg}"] = round(float(tip_w_stand[2]), 4)

            # World vertices at all-zero joints (q_foot = 0)
            verts_w_zero = (verts_link @ R_f_zero.T) + p_f
            tip_w_zero = verts_w_zero[min_z_idx]
            all_zero_foot_radii[f"leg_{leg}"] = round(float(np.linalg.norm(tip_w_zero[:2])), 4)
            all_zero_tip_z_rel_base[f"leg_{leg}"] = round(float(tip_w_zero[2]), 4)

    return {
        "urdf_joints_by_sim_name": urdf_joints_by_sim_name,
        "coxa_mount_radii_m": coxa_mount_radii,
        "coxa_mount_angles_deg": coxa_mount_angles_deg,
        "coxa_lengths_3d_m": coxa_lengths_3d,
        "thigh_lengths_3d_m": thigh_lengths_3d,
        "tibia_lengths_max_m": tibia_lengths_max,
        "foot_tip_local_vectors_m": foot_tip_local_vectors,
        "standing_foot_radii_m": standing_foot_radii,
        "all_zero_foot_radii_m": all_zero_foot_radii,
        "standing_tip_z_rel_base_m": standing_tip_z_rel_base,
        "all_zero_tip_z_rel_base_m": all_zero_tip_z_rel_base,
        "kinematic_standing_height_m": round(-float(np.mean(list(standing_tip_z_rel_base.values()))), 4)
        if standing_tip_z_rel_base
        else 0.1476,
    }


def main():
    usd_path = os.path.abspath(args_cli.usd)
    urdf_path = os.path.abspath(args_cli.urdf)
    print(f"\n==================================================")
    print(f"[AUDIT] Auditing USD asset: {usd_path}")
    print(f"==================================================\n")

    if not os.path.exists(usd_path):
        print(f"[ERROR] USD file not found at: {usd_path}")
        simulation_app.close()
        os._exit(1)

    urdf_kin = extract_urdf_kinematics(urdf_path)
    urdf_joints = urdf_kin.get("urdf_joints_by_sim_name", {})

    # 1. Open USD with PXR for stage-level inspection
    stage = Usd.Stage.Open(usd_path)
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    up_axis = UsdGeom.GetStageUpAxis(stage)
    print(f"[USD Metadata] Meters Per Unit: {mpu}, Up Axis: {up_axis}")

    bbox_cache = UsdGeom.BBoxCache(0, ["default", "render"])
    aligned_range = bbox_cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    min_pt = aligned_range.GetMin()
    max_pt = aligned_range.GetMax()
    bbox_dim = [float(max_pt[0] - min_pt[0]), float(max_pt[1] - min_pt[1]), float(max_pt[2] - min_pt[2])]
    print(f"[USD Bounding Box] Min: {list(min_pt)}, Max: {list(max_pt)}")
    print(f"[USD Bounding Box Dimensions] X: {bbox_dim[0]:.4f}m, Y: {bbox_dim[1]:.4f}m, Z: {bbox_dim[2]:.4f}m")

    # 2. Set up sim scene with ground plane and spawn Articulation
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device or "cuda:0", dt=0.005)
    sim = sim_utils.SimulationContext(sim_cfg)

    plane_cfg = sim_utils.GroundPlaneCfg()
    plane_cfg.func("/World/ground", plane_cfg)

    art_cfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
                enable_gyroscopic_forces=True,
                disable_gravity=False,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.18),
            rot=(0.0, 1.0, 0.0, 0.0),
            joint_pos={
                ".*base_link.*Revolute.*": 0.0,
                ".*NCCUP.*Revolute.*": 0.0,
                ".*NTHIGH.*Revolute.*": -math.pi / 2.0,
            },
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*Revolute.*"],
                effort_limit_sim=2.50,
                velocity_limit_sim=4.71,
                stiffness=16.0,
                damping=1.6,
            ),
        },
    )
    art_cfg.prim_path = "/World/Robot"
    robot = Articulation(art_cfg)

    sim.reset()
    robot.update(dt=0.005)

    sim_joint_names = list(robot.joint_names)
    sim_body_names = list(robot.body_names)
    num_joints = robot.num_joints
    num_bodies = robot.num_bodies

    print(f"\n[Articulation Summary]")
    print(f"  Total Joints: {num_joints}")
    print(f"  Total Rigid Bodies: {num_bodies}")
    print(f"  Body Names ({num_bodies}): {sim_body_names}")

    masses = robot.root_physx_view.get_masses().cpu().numpy()[0]
    total_mass = float(np.sum(masses))

    body_mass_dict = {}
    base_mass = 0.0
    base_body_name = "base_link"
    for idx, b_name in enumerate(sim_body_names):
        m = float(masses[idx])
        body_mass_dict[b_name] = m
        if "base" in b_name.lower():
            base_mass += m
            base_body_name = b_name

    print(f"\n[Mass Breakdown]")
    print(f"  Total Robot Mass: {total_mass:.4f} kg")
    print(f"  Base Link Mass ({base_body_name}): {base_mass:.4f} kg")
    for b_name, m in body_mass_dict.items():
        print(f"    - {b_name:20s}: {m:.4f} kg")

    # Read Joint details from USD + URDF fallback
    usd_joints = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(prim)
            j_name = str(prim.GetPath()).split("/")[-1]
            b0_list = j.GetBody0Rel().GetTargets()
            b1_list = j.GetBody1Rel().GetTargets()
            b0_name = str(b0_list[0]).split("/")[-1] if b0_list else "None"
            b1_name = str(b1_list[0]).split("/")[-1] if b1_list else "None"
            axis = "D6"
            lower, upper = -float("inf"), float("inf")
            if prim.IsA(UsdPhysics.RevoluteJoint):
                rev = UsdPhysics.RevoluteJoint(prim)
                axis = rev.GetAxisAttr().Get() or "Z"
                lo_attr = rev.GetLowerLimitAttr().Get()
                hi_attr = rev.GetUpperLimitAttr().Get()
                if lo_attr is not None:
                    lower = float(lo_attr)
                if hi_attr is not None:
                    upper = float(hi_attr)
            usd_joints[j_name] = {
                "parent_body": b0_name,
                "child_body": b1_name,
                "axis": axis,
                "usd_limits": [lower, upper],
            }

    sim_joint_info = []
    is_leg_grouped = True
    leg_indices_found = []

    for idx, j_name in enumerate(sim_joint_names):
        info = usd_joints.get(j_name, {})
        u_info = urdf_joints.get(j_name, {})

        parent_b = u_info.get("parent") or info.get("parent_body", "unknown")
        child_b = u_info.get("child") or info.get("child_body", "")
        ros_j = derive_ros_joint_name(child_b)
        j_type = determine_joint_type(child_b)

        usd_lim = info.get("usd_limits", u_info.get("limits_rad", [-float("inf"), float("inf")]))
        lo, hi = float(usd_lim[0]), float(usd_lim[1])

        leg_num = None
        for char in child_b:
            if char.isdigit():
                leg_num = int(char)
                break
        if leg_num is not None:
            leg_indices_found.append(leg_num)

        urdf_axis = u_info.get("axis", [0.0, 0.0, 1.0])
        axis_label = info.get("axis") or f"[{urdf_axis[0]:.3f},{urdf_axis[1]:.3f},{urdf_axis[2]:.3f}]"
        default_pos = -math.pi / 2.0 if j_type == "foot" else 0.0

        j_entry = {
            "sim_index": idx,
            "sim_name": j_name,
            "ros_name": ros_j,
            "parent_body": parent_b,
            "child_body": child_b,
            "axis": axis_label,
            "urdf_axis_xyz": [round(float(x), 6) for x in urdf_axis],
            "limits_rad": [lo, hi],
            "default_pos_rad": round(default_pos, 6),
            "leg": leg_num,
            "type": j_type,
            "servo_id": compute_servo_id(leg_num, j_type),
        }
        sim_joint_info.append(j_entry)

    if len(leg_indices_found) == 18:
        if leg_indices_found != sorted(leg_indices_found):
            is_leg_grouped = False

    print(f"\n[Simulator Joint Order ({len(sim_joint_info)} joints)]")
    print(f"{'Idx':<4} {'Simulator Joint Name':<26} {'ROS Joint Name':<16} {'ServoID':<8} {'Parent':<12} {'Child':<10} {'URDF Axis':<24} {'Default(rad)'}")
    print("-" * 120)
    for j in sim_joint_info:
        ax_str = str(j["urdf_axis_xyz"])
        print(f"{j['sim_index']:<4} {j['sim_name']:<26} {j['ros_name']:<16} {str(j['servo_id']):<8} {j['parent_body']:<12} {j['child_body']:<10} {ax_str:<24} {j['default_pos_rad']:+.4f}")

    # Measure settled height at all-zero joints vs standing pose (foot = -pi/2)
    # A) All-zero joints (0, 0, 0)
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] = torch.tensor([[0.0, 0.0, 0.18]], device=sim.device)
    root_state[:, 3:7] = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=sim.device)
    root_state[:, 7:] = 0.0
    zero_j = torch.zeros((1, num_joints), device=sim.device)
    robot.write_root_state_to_sim(root_state)
    robot.write_joint_state_to_sim(zero_j, zero_j)
    robot.set_joint_position_target(zero_j)
    for _ in range(150):
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt=0.005)
    all_zero_settled_height_m = round(float(robot.data.root_pos_w[0, 2].item()), 4)

    # B) Standing pose (coxa=0, thigh=0, foot=-pi/2)
    stand_j = robot.data.default_joint_pos.clone()
    robot.write_root_state_to_sim(root_state)
    robot.write_joint_state_to_sim(stand_j, zero_j)
    robot.set_joint_position_target(stand_j)
    for _ in range(200):
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt=0.005)
    settled_standing_height_m = round(float(robot.data.root_pos_w[0, 2].item()), 4)

    # Verify foot tip world positions in standing pose using foot_tip_local_vectors_m
    tip_vecs = urdf_kin.get("foot_tip_local_vectors_m", {})
    foot_tip_world_z = {}
    foot_tip_world_radii = {}
    base_xy = robot.data.root_pos_w[0, :2].cpu().numpy()
    for leg in range(1, 7):
        f_name = f"NFOOT{leg}"
        if f_name in sim_body_names and f_name in tip_vecs:
            b_idx = sim_body_names.index(f_name)
            b_pos = robot.data.body_pos_w[0, b_idx].cpu().numpy()
            b_quat = robot.data.body_quat_w[0, b_idx].cpu().numpy()  # (w, x, y, z)
            r_loc = np.array(tip_vecs[f_name], dtype=np.float64)
            qw, qvec = b_quat[0], b_quat[1:4]
            a = np.cross(qvec, r_loc)
            v_rot = r_loc + 2.0 * qw * a + 2.0 * np.cross(qvec, a)
            tip_w = b_pos + v_rot
            foot_tip_world_z[f_name] = round(float(tip_w[2]), 4)
            foot_tip_world_radii[f_name] = round(float(np.linalg.norm(tip_w[:2] - base_xy)), 4)

    print(f"\n[Standing Stance & Geometry Audit]")
    print(f"  All-Zero Joints (0,0,0) Settled Base Height: {all_zero_settled_height_m:.4f} m (T-pose, belly on ground)")
    print(f"  Kinematic Zero-Sag Standing Height (q_foot=-pi/2): {urdf_kin.get('kinematic_standing_height_m', 0.1476):.4f} m")
    print(f"  PhysX Settled Standing Base Height (Kp=16, Kd=1.6): {settled_standing_height_m:.4f} m")
    print(f"  Settled Foot Tip World Z: {foot_tip_world_z}")
    print(f"  Settled Foot Tip Radii (m): {foot_tip_world_radii}")

    warnings = []
    mass_diff_pct = abs(total_mass - args_cli.real_mass_kg) / args_cli.real_mass_kg * 100.0
    if mass_diff_pct > 20.0:
        msg = f"WARNING: Total USD mass ({total_mass:.3f} kg) differs from expected real mass ({args_cli.real_mass_kg:.3f} kg) by {mass_diff_pct:.1f}% (>20% threshold)."
        warnings.append(msg)

    has_unconstrained_limits = any(
        j["limits_rad"][0] == -float("inf") or j["limits_rad"][1] == float("inf") for j in sim_joint_info
    )
    if has_unconstrained_limits:
        warnings.append(
            "WARNING: USD joints have unconstrained/infinite limits (-inf, inf). Hardware clamps (coxa -45..+135 deg, thigh/foot -90..+90 deg) must be enforced in ArticulationCfg."
        )
    if not is_leg_grouped:
        warnings.append(
            f"WARNING: Simulator joint order is NOT leg-grouped! Actual order: {[j['sim_name'] for j in sim_joint_info]}"
        )

    report_data = {
        "usd_path": usd_path,
        "urdf_path": urdf_path,
        "total_mass_kg": total_mass,
        "assumed_real_mass_kg": args_cli.real_mass_kg,
        "base_body_name": base_body_name,
        "base_mass_kg": base_mass,
        "body_masses": body_mass_dict,
        "bounding_box_dim_m": bbox_dim,
        "sim_joint_names": sim_joint_names,
        "sim_body_names": sim_body_names,
        "joint_info": sim_joint_info,
        "is_leg_grouped": is_leg_grouped,
        "kinematics": {
            "upright_root_quat_wxyz": [0.0, 1.0, 0.0, 0.0],
            "upright_projected_gravity_z_sign": 1.0,
            "joint_zero_convention": "T-pose horizontal at q=(0,0,0); standing pose uses q_foot=-pi/2 (-1.570796 rad)",
            "default_joint_pos_rad": {"coxa": 0.0, "thigh": 0.0, "foot": -1.5707963267948966},
            "all_zero_settled_height_m": all_zero_settled_height_m,
            "kinematic_standing_height_m": urdf_kin.get("kinematic_standing_height_m", 0.1476),
            "settled_standing_height_m": settled_standing_height_m,
            "coxa_mount_radii_m": urdf_kin.get("coxa_mount_radii_m", {}),
            "coxa_mount_angles_deg": urdf_kin.get("coxa_mount_angles_deg", {}),
            "coxa_lengths_3d_m": urdf_kin.get("coxa_lengths_3d_m", {}),
            "thigh_lengths_3d_m": urdf_kin.get("thigh_lengths_3d_m", {}),
            "tibia_lengths_max_m": urdf_kin.get("tibia_lengths_max_m", {}),
            "standing_foot_radii_m": urdf_kin.get("standing_foot_radii_m", {}),
            "all_zero_foot_radii_m": urdf_kin.get("all_zero_foot_radii_m", {}),
            "foot_tip_local_vectors_m": tip_vecs,
            "settled_foot_tip_world_z_m": foot_tip_world_z,
            "settled_foot_tip_radii_m": foot_tip_world_radii,
        },
        "tip_body_situation": "NFOOT1..6 bodies exist (tibia links). Per-body 3D local tip vectors computed from STL meshes.",
        "warnings": warnings,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n[INFO] Asset report written to: {os.path.abspath(args_cli.out)}\n")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
