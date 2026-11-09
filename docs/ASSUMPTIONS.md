# Owner Facts & Engineering Assumptions (`Z.O.E. Hexapod`)

This document is the single authoritative record of **KNOWN** owner hardware facts, **DATASHEET** parameters, and **UNVERIFIED** engineering assumptions for the new Z.O.E. Hexapod (`hexa_ir` / `hexapod/hexa.urdf`). All superseded guesses (such as 7.4 V or 8.4–11.1 V servo supply) have been deleted.

---

## 1. Authoritative Owner Facts & Datasheet Specifications (`KNOWN`)

| Parameter | Value | Status | Source / Notes |
| :--- | :--- | :---: | :--- |
| **Battery & Power Distribution** | **3S LiPo**, powers servos directly; servos see **11.4–12.0 V**; total current **12–18 A** | **KNOWN** | Owner hardware fact. |
| **Servo Model** | **Waveshare ST3215**, **12 V** serial bus servo variant | **KNOWN** | Owner hardware fact. |
| **Datasheet Stall Torque (12 V)** | **30 kg·cm = 2.942 N·m** @ 12 V | **KNOWN (DATASHEET)** | Waveshare ST3215 Official Wiki/Datasheet (`https://www.waveshare.com/wiki/ST3215_Servo`, `https://www.waveshare.com/st3215-servo.htm`). |
| **Simulation Effort Limit (85% Margin)** | **2.50 N·m** ($0.85 \times 2.942\text{ N}\cdot\text{m} = 2.5007\text{ N}\cdot\text{m}$) | **KNOWN (DERIVED)** | 85% of 12 V stall torque retained as thermal and mechanical safety margin per owner directive. Saturation gate threshold is $0.98 \times 2.50 = 2.45\text{ N}\cdot\text{m}$. |
| **Datasheet No-Load Speed (12 V)** | **0.222 s / 60° = 45 RPM = 4.71 rad/s** @ 12 V | **KNOWN (DATASHEET)** | Waveshare ST3215 Official Wiki/Datasheet (`https://www.waveshare.com/wiki/ST3215_Servo`, 1:345 gear ratio, 12 V variant). |
| **Servo ID Addressing** | $\text{ID} = 3(\text{leg} - 1) + k$, where $k = 1$ (coxa), $2$ (thigh), $3$ (foot), $\text{leg} \in \{1..6\}$ | **KNOWN** | Owner hardware fact (IDs 1..18). |
| **Servo Bus Baud Rate** | **1 Mbaud** (`1,000,000` bps) | **KNOWN** | Owner hardware fact. |
| **Sensors & E-Stop Status** | **None yet**: no servo feedback path, no IMU, and no hardware emergency stop exist yet | **KNOWN** | Owner hardware fact. |
| **Fault Behavior** | **Hold position** on any fault | **KNOWN** | Owner safety requirement. |
| **Depth Camera (`DCAM`)** | Commented out in [`hexapod/hexa.urdf`](file:///home/rithwik/SpdrBot-main/hexapod/hexa.urdf) | **KNOWN** | Owner directive (`<link name="DCAM">` and `<joint name="base_link_Rigid-57">` commented out). |

---

## 2. Unverified Engineering Assumptions (`UNVERIFIED`)

| Parameter | Assumed Value | Status | Rationale & How to Replace with Real Data |
| :--- | :--- | :---: | :--- |
| **Total Robot Mass** | **5.00 kg** (CAD links: **1.6635 kg**, `base_link` with payload: **3.3365 kg**) | **CALIBRATED ASSUMPTION** | 5.0 kg upper-bound mass model applied by keeping native CAD link masses ($1.6635\text{ kg}$) and adding $0.8338\text{ kg}$ payload mass (representing battery, Jetson, carrier, wiring, and standoffs) to `base_link` in `hexa_physics.usd` and `hexa.urdf`. Total mass verified at $5.0000\text{ kg} \pm 0.05$. Weigh the physical robot to replace. |
| **Mass Domain Randomization** | **$\pm 10\%$** multiplicative (`[0.90, 1.10]`) | **ASSUMPTION** | Covers battery weight and wiring variance around the nominal mass. |
| **Link Moments of Inertia** | Native Fusion 360 CAD inertia tensors (scaled proportionally if mass is scaled) | **UNVERIFIED ASSUMPTION** | Computed from CAD geometry in `hexa.urdf`. |
| **Joint PD Stiffness ($K_p$)** | **16.0 N·m/rad** | **UNVERIFIED ASSUMPTION** | Keeps static standing sag $< 3^\circ$ while providing compliant give at footfall to relieve closed-chain internal forces. Replace via `scripts/fit_actuator_sim.py` once servo feedback is wired. |
| **Joint PD Damping ($K_d$)** | **1.6 N·m·s/rad** ($K_p / 10$) | **UNVERIFIED ASSUMPTION** | Near-critical damping to prevent oscillation during swing phase. |
| **Actuator Gain Randomization** | **$\pm 30\%$** multiplicative (`[0.70, 1.30]`) | **ASSUMPTION** | Covers unit-to-unit gearbox friction and thermal variance across the 18 servos. |
| **Effort Limit Randomization** | **$\pm 15\%$** multiplicative (`[0.85, 1.15]`) | **ASSUMPTION** | Covers 3S LiPo voltage droop under 12–18 A load. |
| **Target Rate Limiter** | **6.0° / step** ($0.10472\text{ rad/step} = 5.236\text{ rad/s}$ at 50 Hz) | **ASSUMPTION** | Matches safety rate clamp in the ROS 2 driver node. |
| **Action Latency / Delay** | **0–2 policy steps** (0–40 ms at 50 Hz) | **UNVERIFIED ASSUMPTION** | Models USB-serial bridge and ESP32 forwarding latency at 1 Mbaud. |
| **Joint Calibration Zero-Offset** | **$\pm 3.0^\circ$** ($\pm 0.0524\text{ rad}$) | **UNVERIFIED ASSUMPTION** | Models spline horn alignment tolerance (1 tooth on 25T spline = $14.4^\circ$; fine offset calibrated in software). |
| **Ground Friction** | Static `[0.4, 1.2]`, Dynamic `[0.3, 1.0]` | **ASSUMPTION** | Spans slick tile to high-grip rubber mat. |
