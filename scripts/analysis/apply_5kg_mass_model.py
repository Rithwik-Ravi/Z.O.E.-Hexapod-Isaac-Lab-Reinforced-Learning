# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Applies the 5.0 kg assumed mass model to hexapod/hexa/hexa.usd."""

import argparse
import os
import shutil
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Apply 5.0 kg mass model to hexa.usd")
parser.add_argument("--usd_path", type=str, default="hexapod/hexa/hexa.usd")
parser.add_argument("--total_mass", type=float, default=5.000)
parser.add_argument("--chassis_mass", type=float, default=3.290)
parser.add_argument("--link_mass", type=float, default=0.095)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = True
app_launcher = AppLauncher(args_cli)
sim_app = app_launcher.app

from pxr import Gf, Usd, UsdGeom, UsdPhysics

usd_file = os.path.abspath(args_cli.usd_path)
backup_file = os.path.abspath("hexapod/hexa/hexa_orig.usd")

if not os.path.exists(backup_file):
    print(f"[MASS_APPLY] Creating backup: {backup_file}")
    shutil.copyfile(usd_file, backup_file)
else:
    print(f"[MASS_APPLY] Backup already exists at: {backup_file}")

stage = Usd.Stage.Open(usd_file)
total_mass = 0.0

# 1. Update Base Link
base_prim = stage.GetPrimAtPath("/hexa_body/base_link_v1")
if base_prim.IsValid() and base_prim.HasAPI(UsdPhysics.MassAPI):
    mass_api = UsdPhysics.MassAPI(base_prim)
    old_m = mass_api.GetMassAttr().Get()
    new_m = args_cli.chassis_mass
    s = new_m / old_m
    old_i = mass_api.GetDiagonalInertiaAttr().Get()
    new_i = Gf.Vec3f(old_i[0] * s, old_i[1] * s, old_i[2] * s)
    mass_api.GetMassAttr().Set(new_m)
    mass_api.GetDiagonalInertiaAttr().Set(new_i)
    total_mass += new_m
    print(f"[MASS_APPLY] /hexa_body/base_link_v1: {old_m:.4f} kg -> {new_m:.4f} kg (inertia scale {s:.2f})")
else:
    print("[ERROR] Base link not found or missing MassAPI!")

# 2. Update Leg Links (NCCUP, NTHIGH, NFOOT for legs 1..6)
for leg_i in range(1, 7):
    for prefix in ["NCCUP", "NTHIGH", "NFOOT"]:
        prim_path = f"/hexa_body/{prefix}{leg_i}"
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid() and prim.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI(prim)
            old_m = mass_api.GetMassAttr().Get()
            new_m = args_cli.link_mass
            s = new_m / old_m
            old_i = mass_api.GetDiagonalInertiaAttr().Get()
            new_i = Gf.Vec3f(old_i[0] * s, old_i[1] * s, old_i[2] * s)
            mass_api.GetMassAttr().Set(new_m)
            mass_api.GetDiagonalInertiaAttr().Set(new_i)
            total_mass += new_m
            print(f"[MASS_APPLY] {prim_path}: {old_m:.4f} kg -> {new_m:.4f} kg (inertia scale {s:.2f})")
        else:
            print(f"[ERROR] Prim {prim_path} not found or missing MassAPI!")

stage.Save()
print(f"\n[MASS_APPLY] Stage saved successfully! Total explicit mass in USD: {total_mass:.4f} kg")
assert abs(total_mass - 5.000) < 0.001, f"Total mass {total_mass} does not match 5.000 kg!"

sim_app.close()
sys.exit(0)
