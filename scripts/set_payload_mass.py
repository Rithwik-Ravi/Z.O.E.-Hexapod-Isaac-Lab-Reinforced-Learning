#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Rebuild base payload mass, inertia, URDF, and config properties from measured total robot mass (Task 7).

Usage:
    python scripts/set_payload_mass.py --total-mass-kg 5.0
"""

import argparse
import json
import math
import os
import re
import sys
import yaml

CAD_BASE_MASS_KG = 2.502728704259523
CAD_SIX_LEGS_MASS_KG = 1.66348730  # 6 * (0.015043186 + 0.21543324 + 0.04671158)
CAD_TOTAL_MASS_KG = CAD_BASE_MASS_KG + CAD_SIX_LEGS_MASS_KG  # ~4.166216 kg

CAD_BASE_INERTIA = {
    "ixx": 0.010209323161935659,
    "iyy": 0.009936168413282058,
    "izz": 0.019355228545140914,
    "ixy": -2.7795540266991648e-12,
    "iyz": 7.976531684940807e-14,
    "ixz": -1.0281441658134396e-13,
}


def compute_payload_mass_properties(total_mass_kg: float):
    """Compute base link mass, payload mass, and scaled inertia from total robot mass."""
    if total_mass_kg <= CAD_SIX_LEGS_MASS_KG:
        raise ValueError(
            f"Total mass ({total_mass_kg:.3f} kg) must exceed legs mass ({CAD_SIX_LEGS_MASS_KG:.3f} kg)!"
        )

    base_link_mass_kg = total_mass_kg - CAD_SIX_LEGS_MASS_KG
    payload_added_kg = base_link_mass_kg - CAD_BASE_MASS_KG
    inertia_scale = base_link_mass_kg / CAD_BASE_MASS_KG

    scaled_inertia = {k: v * inertia_scale for k, v in CAD_BASE_INERTIA.items()}

    return {
        "total_mass_kg": total_mass_kg,
        "base_link_mass_kg": base_link_mass_kg,
        "payload_added_kg": payload_added_kg,
        "six_legs_mass_kg": CAD_SIX_LEGS_MASS_KG,
        "inertia_scale": inertia_scale,
        "scaled_inertia": scaled_inertia,
    }


def update_urdf(props: dict, urdf_path: str = "hexapod/hexa.urdf"):
    """Update base_link mass and inertia in URDF."""
    if not os.path.exists(urdf_path):
        return False

    with open(urdf_path, "r") as f:
        content = f.read()

    base_m = props["base_link_mass_kg"]
    iner = props["scaled_inertia"]

    # Replace base_link mass
    mass_pattern = r'(<link name="base_link">\s*<inertial>\s*<origin [^>]+/>\s*<mass value=")[^"]+(" />)'
    content = re.sub(mass_pattern, rf'\g<1>{base_m:.15f}\g<2>', content)

    # Replace base_link inertia
    inertia_pattern = r'(<inertia ixx=")[^"]+(" iyy=")[^"]+(" izz=")[^"]+(" ixy=")[^"]+(" iyz=")[^"]+(" ixz=")[^"]+(" />)'
    new_inertia_str = (
        rf'\g<1>{iner["ixx"]:.8e}\g<2>{iner["iyy"]:.8e}\g<3>{iner["izz"]:.8e}'
        rf'\g<4>{iner["ixy"]:.8e}\g<5>{iner["iyz"]:.8e}\g<6>{iner["ixz"]:.8e}\g<7>'
    )
    content = re.sub(inertia_pattern, new_inertia_str, content)

    with open(urdf_path, "w") as f:
        f.write(content)
    return True


def update_yaml(props: dict, yaml_path: str = "deploy/mass_properties.yaml"):
    """Update deploy/mass_properties.yaml."""
    if not os.path.exists(yaml_path):
        return False

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}

    data["calibrated"] = True
    data["mass_status"] = "MEASURED_TOTAL_RECONSTRUCTED"
    if "assumed_masses" not in data:
        data["assumed_masses"] = {}
    data["assumed_masses"]["assumed_total_robot_kg"] = round(props["total_mass_kg"], 4)
    data["assumed_masses"]["base_payload_added_kg"] = round(props["payload_added_kg"], 4)

    if "applied_model_masses" not in data:
        data["applied_model_masses"] = {}
    data["applied_model_masses"]["base_link_with_payload_kg"] = round(props["base_link_mass_kg"], 4)
    data["applied_model_masses"]["six_legs_total_kg"] = round(props["six_legs_mass_kg"], 4)
    data["applied_model_masses"]["total_robot_mass_kg"] = round(props["total_mass_kg"], 4)

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)
    return True


def update_asset_report(props: dict, report_path: str = "deploy/asset_report.json"):
    """Update deploy/asset_report.json."""
    if not os.path.exists(report_path):
        return False

    with open(report_path, "r") as f:
        data = json.load(f)

    data["total_mass_kg"] = props["total_mass_kg"]
    data["assumed_real_mass_kg"] = props["total_mass_kg"]
    data["base_mass_kg"] = props["base_link_mass_kg"]
    if "body_masses" in data:
        data["body_masses"]["base_link"] = props["base_link_mass_kg"]

    with open(report_path, "w") as f:
        json.dump(data, f, indent=2)
    return True


def update_usd_physics(props: dict, usd_path: str = "hexapod/hexa/configuration/hexa_physics.usd"):
    """Update USD physics base_link mass and inertia if pxr is available."""
    try:
        from pxr import Usd, UsdPhysics, Gf
        if not os.path.exists(usd_path):
            return False
        stage = Usd.Stage.Open(usd_path)
        base_prim = stage.GetPrimAtPath("/hexa/base_link")
        if not base_prim.IsValid():
            # Try alternate path
            base_prim = stage.GetPrimAtPath("/base_link")
        if base_prim.IsValid():
            mass_api = UsdPhysics.MassAPI.Apply(base_prim)
            mass_api.GetMassAttr().Set(float(props["base_link_mass_kg"]))
            iner = props["scaled_inertia"]
            diag_inertia = Gf.Vec3f(float(iner["ixx"]), float(iner["iyy"]), float(iner["izz"]))
            mass_api.GetDiagonalInertiaAttr().Set(diag_inertia)
            stage.GetRootLayer().Save()
            return True
    except Exception as e:
        print(f"[WARN] USD update via pxr skipped: {e}")
    return False


def set_payload_mass(total_mass_kg: float):
    props = compute_payload_mass_properties(total_mass_kg)

    print(f"\n[SET PAYLOAD MASS] Setting robot total mass to: {total_mass_kg:.4f} kg")
    print(f"  - CAD Link Mass (6 legs) : {props['six_legs_mass_kg']:.4f} kg")
    print(f"  - Base Link Mass         : {props['base_link_mass_kg']:.4f} kg (CAD: {CAD_BASE_MASS_KG:.4f} kg)")
    print(f"  - Base Payload Added     : {props['payload_added_kg']:+.4f} kg")
    print(f"  - Base Inertia Scale     : {props['inertia_scale']:.4f}x")

    urdf_ok = update_urdf(props)
    yaml_ok = update_yaml(props)
    json_ok = update_asset_report(props)
    usd_ok = update_usd_physics(props)

    print(f"  - Updated hexapod/hexa.urdf        : {'OK' if urdf_ok else 'FAILED'}")
    print(f"  - Updated deploy/mass_properties.yaml: {'OK' if yaml_ok else 'FAILED'}")
    print(f"  - Updated deploy/asset_report.json   : {'OK' if json_ok else 'FAILED'}")
    print(f"  - Updated USD physics layer         : {'OK' if usd_ok else 'SKIPPED'}")
    print(f"[SET PAYLOAD MASS] Successfully updated mass model to {total_mass_kg:.4f} kg.\n")
    return props


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild base payload mass and inertia from weighed number.")
    parser.add_argument("--total-mass-kg", type=float, default=5.000, help="Total robot mass in kg (e.g. 5.0).")
    args = parser.parse_args()
    set_payload_mass(args.total_mass_kg)
