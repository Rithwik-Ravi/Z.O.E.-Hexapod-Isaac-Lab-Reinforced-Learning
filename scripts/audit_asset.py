# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Asset audit script for Z.O.E. Hexapod USD asset."""

import argparse
import json
import os
import sys
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
    "--real-mass-kg",
    type=float,
    default=2.0,
    help="Expected real mass of the robot in kg for comparison.",
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
from isaaclab.assets import Articulation, ArticulationCfg


def derive_ros_joint_name(child_body_name: str) -> str:
    """Auto-derive ROS joint name from child body name."""
    # NCCUP1 -> NCCUP1_joint, NTHIGH1 -> NTHIGH1_joint, NFOOT1 -> NFOOT1_joint
    if child_body_name and child_body_name != "base_link_v1":
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



def main():
    usd_path = os.path.abspath(args_cli.usd)
    print(f"\n==================================================")
    print(f"[AUDIT] Auditing USD asset: {usd_path}")
    print(f"==================================================\n")

    if not os.path.exists(usd_path):
        print(f"[ERROR] USD file not found at: {usd_path}")
        simulation_app.close()
        sys.exit(1)

    # 1. Open USD with PXR for stage-level inspection
    stage = Usd.Stage.Open(usd_path)
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    up_axis = UsdGeom.GetStageUpAxis(stage)
    print(f"[USD Metadata] Meters Per Unit: {mpu}, Up Axis: {up_axis}")

    # Bounding Box calculation
    bbox_cache = UsdGeom.BBoxCache(0, ["default", "render"])
    aligned_range = bbox_cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    min_pt = aligned_range.GetMin()
    max_pt = aligned_range.GetMax()
    bbox_dim = [max_pt[0] - min_pt[0], max_pt[1] - min_pt[1], max_pt[2] - min_pt[2]]
    print(f"[USD Bounding Box] Min: {list(min_pt)}, Max: {list(max_pt)}")
    print(f"[USD Bounding Box Dimensions] X: {bbox_dim[0]:.4f}m, Y: {bbox_dim[1]:.4f}m, Z: {bbox_dim[2]:.4f}m")

    # 2. Set up sim scene and spawn Articulation to query Isaac Lab / PhysX view
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device or "cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.5, 1.5, 1.0], [0.0, 0.0, 0.2])

    art_cfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
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
            pos=(0.0, 0.0, 0.5),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={},
    )
    art_cfg.prim_path = "/World/Robot"
    robot = Articulation(art_cfg)

    sim.reset()
    robot.update(dt=0.005)

    # Gather Sim Joint Names and Body Names
    sim_joint_names = robot.joint_names
    sim_body_names = robot.body_names
    num_joints = robot.num_joints
    num_bodies = robot.num_bodies

    print(f"\n[Articulation Summary]")
    print(f"  Total Joints: {num_joints}")
    print(f"  Total Rigid Bodies: {num_bodies}")
    print(f"  Body Names ({num_bodies}): {sim_body_names}")

    # Gather Masses
    # robot.root_physx_view provides body masses
    masses = robot.root_physx_view.get_masses().cpu().numpy()[0]
    total_mass = float(np.sum(masses))

    body_mass_dict = {}
    base_mass = 0.0
    for idx, b_name in enumerate(sim_body_names):
        m = float(masses[idx])
        body_mass_dict[b_name] = m
        if b_name == "base_link_v1" or "base" in b_name.lower():
            base_mass += m

    print(f"\n[Mass Breakdown]")
    print(f"  Total Robot Mass: {total_mass:.4f} kg")
    print(f"  Base Link Mass:   {base_mass:.4f} kg")
    for b_name, m in body_mass_dict.items():
        print(f"    - {b_name:20s}: {m:.4f} kg")

    # Read Joint details from USD
    joint_details = []
    usd_joints = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint):
            j = UsdPhysics.RevoluteJoint(prim)
            j_path = str(prim.GetPath())
            j_name = j_path.split("/")[-1]

            b0_list = j.GetBody0Rel().GetTargets()
            b1_list = j.GetBody1Rel().GetTargets()
            b0_name = str(b0_list[0]).split("/")[-1] if b0_list else "None"
            b1_name = str(b1_list[0]).split("/")[-1] if b1_list else "None"

            axis = j.GetAxisAttr().Get()
            lower = j.GetLowerLimitAttr().Get()
            upper = j.GetUpperLimitAttr().Get()

            usd_joints[j_name] = {
                "parent_body": b0_name,
                "child_body": b1_name,
                "axis": axis,
                "usd_limits": [lower, upper],
            }

    # Map joint order in Simulator
    sim_joint_info = []
    ros_mapping_warnings = []
    joint_limit_warnings = []
    is_leg_grouped = True

    # Check leg grouping order
    leg_indices_found = []
    for idx, j_name in enumerate(sim_joint_names):
        # find matching USD joint
        info = usd_joints.get(j_name, {})
        child_b = info.get("child_body", "")
        ros_j = derive_ros_joint_name(child_b)

        # Check joint limits from simulation / USD
        usd_lim = info.get("usd_limits", [-float("inf"), float("inf")])
        lo, hi = usd_lim[0], usd_lim[1]

        # Determine leg number
        leg_num = None
        for char in child_b:
            if char.isdigit():
                leg_num = int(char)
                break

        if leg_num is not None:
            leg_indices_found.append(leg_num)

        j_entry = {
            "sim_index": idx,
            "sim_name": j_name,
            "ros_name": ros_j,
            "parent_body": info.get("parent_body", "unknown"),
            "child_body": child_b,
            "axis": info.get("axis", "unknown"),
            "limits_rad": [lo, hi],
            "default_pos_rad": 0.0,
            "leg": leg_num,
            "type": determine_joint_type(child_b),
        }
        sim_joint_info.append(j_entry)

    # Verify if leg indices are grouped monotonically e.g. leg 1 (3 joints), leg 2 (3 joints)...
    if len(leg_indices_found) == 18:
        expected_grouped = sorted(leg_indices_found)
        if leg_indices_found != expected_grouped:
            is_leg_grouped = False

    print(f"\n[Simulator Joint Order ({len(sim_joint_info)} joints)]")
    print(f"{'Idx':<4} {'Simulator Joint Name':<30} {'ROS Joint Name':<20} {'Parent':<15} {'Child':<15} {'Axis':<6} {'Limits (rad)'}")
    print("-" * 110)
    for j in sim_joint_info:
        lim_str = f"[{j['limits_rad'][0]}, {j['limits_rad'][1]}]"
        print(f"{j['sim_index']:<4} {j['sim_name']:<30} {j['ros_name']:<20} {j['parent_body']:<15} {j['child_body']:<15} {str(j['axis']):<6} {lim_str}")

    # Foot Tip Identification Audit
    foot_bodies = [b for b in sim_body_names if "NFOOT" in b or "FOOT" in b or "TIP" in b]
    print(f"\n[Foot Tip Audit]")
    print(f"  Foot/Shank bodies found ({len(foot_bodies)}): {foot_bodies}")
    tip_body_situation = ""
    if all(b.startswith("NFOOT") for b in foot_bodies) and len(foot_bodies) == 6:
        tip_body_situation = "NFOOT1..6 bodies exist (tibia links). No dedicated zero-radius tip prims found in USD."
        print(f"  NOTICE: {tip_body_situation}")
        print("  RECOMMENDATION: Measure ground contact on NFOOT1..6 bodies and apply clearance height target relative to tibia position offset or local ground.")
    else:
        tip_body_situation = f"Custom foot bodies detected: {foot_bodies}"

    # Calculate standing height with default joint positions (all zeros or standing pose)
    # Standing pose in IK convention: coxa 45 deg (0.7854 rad), thigh 7.7 deg (0.1344 rad), foot 7.4 deg (0.1292 rad)
    # USD zero pose height:
    default_pos = robot.data.default_root_state[0, :3].cpu().numpy()
    root_z = float(default_pos[2])
    print(f"\n[Standing Stance Audit]")
    print(f"  Root Spawn Z Position: {root_z:.4f} m")

    # 3. Warnings and Rule Enforcement
    warnings = []
    mass_diff_pct = abs(total_mass - args_cli.real_mass_kg) / args_cli.real_mass_kg * 100.0
    if mass_diff_pct > 20.0:
        msg = f"WARNING: Total USD mass ({total_mass:.3f} kg) differs from expected real mass ({args_cli.real_mass_kg:.3f} kg) by {mass_diff_pct:.1f}% (>20% threshold)."
        warnings.append(msg)

    # Joint limit check: coxa [-45..+135 deg] -> [-0.785, 2.356 rad], thigh/foot [-90..+90 deg] -> [-1.571, 1.571 rad]
    has_unconstrained_limits = False
    for j in sim_joint_info:
        lo, hi = j["limits_rad"]
        if lo == -float("inf") or hi == float("inf") or (lo == 0.0 and hi == 0.0):
            has_unconstrained_limits = True

    if has_unconstrained_limits:
        msg = "WARNING: USD joints have unconstrained/infinite limits (-inf, inf). Hardware clamps (coxa -45..+135 deg, thigh/foot -90..+90 deg) must be enforced in ArticulationCfg."
        warnings.append(msg)

    if not is_leg_grouped:
        msg = f"WARNING: Simulator joint order is NOT leg-grouped! Actual order: {[j['sim_name'] for j in sim_joint_info]}"
        warnings.append(msg)

    print(f"\n==================================================")
    print(f"[AUDIT RESULTS & WARNINGS]")
    print(f"==================================================")
    if warnings:
        for w in warnings:
            print(f"  ⚠️  {w}")
    else:
        print("  ✅ All asset checks passed without warnings!")

    # Write report JSON
    report_data = {
        "usd_path": usd_path,
        "total_mass_kg": total_mass,
        "base_mass_kg": base_mass,
        "body_masses": body_mass_dict,
        "bounding_box_dim_m": bbox_dim,
        "sim_joint_names": sim_joint_names,
        "sim_body_names": sim_body_names,
        "joint_info": sim_joint_info,
        "is_leg_grouped": is_leg_grouped,
        "tip_body_situation": tip_body_situation,
        "warnings": warnings,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n[INFO] Asset report written to: {os.path.abspath(args_cli.out)}\n")

    # Close sim app
    simulation_app.close()


if __name__ == "__main__":
    main()
