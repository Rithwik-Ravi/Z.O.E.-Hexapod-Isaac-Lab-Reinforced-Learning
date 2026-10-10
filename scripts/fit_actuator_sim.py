# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to fit actuator PD parameters (kp, kd, delay) from step-response CSV data."""

import argparse
import os
import yaml
import numpy as np

# CSV Format Specification:
# Header: t_s,joint,target_deg,pos_deg,vel_deg_s,load_permille
# Example:
# 0.00,NCCUP1_joint,45.0,45.0,0.0,0
# 0.01,NCCUP1_joint,90.0,46.2,120.0,150

def parse_args():
    parser = argparse.ArgumentParser(description="Fit actuator PD model parameters from step-response CSV.")
    parser.add_argument("--csv", type=str, required=False, help="Path to step response CSV file.")
    parser.add_argument("--out", type=str, default="deploy/actuator_model.yaml", help="Output YAML file path.")
    return parser.parse_args()


def simulate_pd_step(target_deg, initial_pos_deg, t_steps, dt, kp, kd, delay_steps):
    """Simulate 2nd order PD step response with delay."""
    num_steps = len(t_steps)
    pos = np.zeros(num_steps)
    vel = np.zeros(num_steps)

    pos[0] = initial_pos_deg
    target_buffer = np.full(delay_steps + 1, target_deg)

    for i in range(1, num_steps):
        # Target with delay
        target_active = target_buffer[0]
        err = target_active - pos[i - 1]
        acc = kp * err - kd * vel[i - 1]

        vel[i] = vel[i - 1] + acc * dt
        pos[i] = pos[i - 1] + vel[i] * dt

        # Update delay buffer
        target_buffer[:-1] = target_buffer[1:]
        target_buffer[-1] = target_deg

    return pos


def fit_actuator_from_csv(csv_path):
    """Grid search kp, kd, delay_ms to fit recorded step response."""
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    t_s = data["t_s"]
    target_deg = data["target_deg"]
    pos_deg = data["pos_deg"]

    dt = np.mean(np.diff(t_s)) if len(t_s) > 1 else 0.01

    best_error = float("inf")
    best_params = {"kp": 15.0, "kd": 1.5, "delay_ms": 10.0}

    # Grid search space
    kp_range = np.linspace(5.0, 30.0, 10)
    kd_range = np.linspace(0.1, 5.0, 10)
    delay_steps_range = [0, 1, 2, 3, 4]  # 0 to 40 ms at 100 Hz

    target_val = target_deg[-1]
    initial_val = pos_deg[0]

    for kp in kp_range:
        for kd in kd_range:
            for delay in delay_steps_range:
                sim_pos = simulate_pd_step(target_val, initial_val, t_s, dt, kp, kd, delay)
                rmse = np.sqrt(np.mean((sim_pos - pos_deg) ** 2))
                if rmse < best_error:
                    best_error = rmse
                    best_params = {"kp": float(kp), "kd": float(kd), "delay_ms": float(delay * dt * 1000.0)}

    return best_params


def main():
    args = parse_args()

    actuator_model = {
        "status": "UNCALIBRATED",
        "actuator_assumptions": {
            "stall_torque_nm": 2.5,
            "max_speed_rad_s": 6.0,
            "deadband_deg": 0.5,
            "tau_ms": 10.0,
        },
        "coxa": {
            "delay_ms": 20.0,
            "tau_ms": 10.0,
            "max_speed_rad_s": 6.0,
            "deadband_deg": 0.5,
            "stall_torque_nm": 2.5,
            "kp": 15.0,
            "kd": 1.5,
        },
        "thigh": {
            "delay_ms": 20.0,
            "tau_ms": 10.0,
            "max_speed_rad_s": 6.0,
            "deadband_deg": 0.5,
            "stall_torque_nm": 2.5,
            "kp": 15.0,
            "kd": 1.5,
        },
        "foot": {
            "delay_ms": 20.0,
            "tau_ms": 10.0,
            "max_speed_rad_s": 6.0,
            "deadband_deg": 0.5,
            "stall_torque_nm": 2.5,
            "kp": 15.0,
            "kd": 1.5,
        },
    }

    if args.csv and os.path.exists(args.csv):
        print(f"[FIT_ACTUATOR] Fitting parameters from CSV: {args.csv}")
        fitted = fit_actuator_from_csv(args.csv)
        print(f"[FIT_ACTUATOR] Fitted params: {fitted}")
        for j_type in ["coxa", "thigh", "foot"]:
            actuator_model[j_type].update(fitted)
        actuator_model["status"] = "CALIBRATED"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        yaml.dump(actuator_model, f, default_flow_style=False)

    print(f"[FIT_ACTUATOR] Written actuator model to {args.out}")


if __name__ == "__main__":
    main()
