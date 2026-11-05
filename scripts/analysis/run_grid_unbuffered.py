# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import subprocess
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

test_cases = [
    ("flat_deploy", 0.18, 2.5),
    ("flat_deploy", 0.18, 5.0),
    ("flat_deploy", 0.28, 2.5),
    ("flat_deploy", 0.28, 5.0),
    ("usd_zero", 0.18, 2.5),
    ("usd_zero", 0.18, 5.0),
    ("usd_zero", 0.28, 2.5),
    ("usd_zero", 0.28, 5.0),
]

print("=" * 110)
print(f"{'Case':<32} | {'t=0 h':<8} | {'Settled h':<10} | {'Feet Down':<9} | {'Torque % (C/T/F)':<22} | {'Colliding Bodies'}")
print("=" * 110)
sys.stdout.flush()

for pose, z, effort in test_cases:
    cmd = [
        "python", os.path.join(_SCRIPT_DIR, "run_single_grid_case.py"),
        "--pose", pose,
        "--spawn_z", str(z),
        "--effort_limit", str(effort)
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in iter(proc.stdout.readline, ''):
        if "GRID_RESULT:" in line:
            clean_line = line.replace("GRID_RESULT:", "").strip()
            print(clean_line)
            sys.stdout.flush()
    proc.stdout.close()
    proc.wait()
