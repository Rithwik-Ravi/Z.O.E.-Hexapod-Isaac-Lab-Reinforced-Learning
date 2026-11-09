# Stage 1 Readiness & Verification Report: `Zoe-Hexapod-Flat-Deploy-v0` (Chunk 1)

**Date**: 2026-11-09  
**Git Commit**: `2df368c`  
**Task**: `Zoe-Hexapod-Flat-Deploy-v0` (Stage 1 Flat Terrain Locomotion)  
**Run Directory**: `logs/rsl_rl/zoe_hexapod_deploy/2026-11-09_22-33-44/`  
**Evaluated Checkpoints**: `model_500.pt`, `model_1000.pt`, `model_1499.pt` (1500 iterations total)  
**Overall Stage 1 Gate Verdict**: **FAIL ON FROZEN CRITERIA** (Forward speed $0.0065\text{ m/s}$ vs $\ge 0.120\text{ m/s}$; Torque saturation $6.33\%$ vs $\le 2.0\%$)  
**Locomotion Quality & Safety Verdict**: **ULTRA-STABLE 6-LEG CREEPING & STANDING; STATICS TENSION AT 5.0 KG**  

---

## 1. Summary of Changes Since Previous Stage 1 Probes

1. **New Asset Integration & Mass Calibration**:
   - Switched to new CAD model (`hexa_ir`): CAD raw links total $4.1662\text{ kg}$ (with servos embedded in CAD links: `NTHIGH` $0.2155\text{ kg}$, `NFOOT` $0.0467\text{ kg}$, `NCCUP` $0.0150\text{ kg}$).
   - Added $0.8338\text{ kg}$ payload mass directly to `base_link` ($3.3365\text{ kg}$) in `hexapod/hexa/configuration/hexa_physics.usd` and `hexapod/hexa.urdf` to represent real-world battery (~200g), Jetson carrier (~280g), PDB/buck (~80g), ESP32 boards & wiring (~70g), and standoffs/harness (~200g).
   - Strictly verified asset mass: **$5.0000\text{ kg}$** ($\pm 0.05\text{ kg}$ assertion verified via `scripts/audit_asset.py`).
2. **Torque Saturation Penalty Recalibration**:
   - Servos: Waveshare ST3215 12V variant ($2.94\text{ N}\cdot\text{m}$ stall torque at 12V nominal, 12V direct 3S LiPo battery, effort limit $2.50\text{ N}\cdot\text{m}$).
   - Set soft torque limit $\tau_{\text{soft}} = 1.95\text{ N}\cdot\text{m}$ (above the $1.880\text{ N}\cdot\text{m}$ static tripod holding torque, avoiding penalties during quiescent standing) with penalty weight $-2.0$.
3. **Yaw-Dominant Command Sampling**:
   - Created `ZoeUniformVelocityCommand` and `ZoeUniformVelocityCommandCfg` in `source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/mdp/commands.py`.
   - Guaranteed that $\ge 25\%$ (configured to $30\%$) of non-standing environments receive yaw-dominant commands ($|\omega_z| \in [0.30, 0.60]\text{ rad/s}$ with small linear drift $|v_{xy}| \le 0.05\text{ m/s}$).
   - Increased yaw reward weight `track_ang_vel_z_exp` from $0.5$ to $1.0$ in `params.py`.
4. **Enhanced Report-Only Gate Suite (`scripts/chunk_check.py`)**:
   - Added joint-limit proximity ($\le 3^\circ$) for coxa, thigh, and foot.
   - Added per-leg swing clearance ($L_1..L_6$) during Condition 2 walking.
   - Added dedicated 5.0s pure-yaw test ($v_{xy} = 0, \omega_z = 0.40\text{ rad/s}$).
   - Kept all gate thresholds frozen.

---

## 2. Gate Verification Table Across Chunk 1 Checkpoints

All evaluations were executed with 128 parallel environments on frozen thresholds via [`scripts/chunk_check.py`](file:///home/rithwik/SpdrBot-main/scripts/chunk_check.py):

| Condition | Metric | Threshold | Iteration 500 (`model_500.pt`) | Iteration 1000 (`model_1000.pt`) | Iteration 1500 (`model_1499.pt`) | Trend / Final Result |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Zero Command** | Base Height $\ge 0.1275\text{ m}$ | $\ge 95\%$ | $100.00\%$ | $100.00\%$ | $100.00\%$ | **PASS** (Rock solid) |
| **1. Zero Command** | $\ge 5$ Feet in Contact | $\ge 95\%$ | $98.28\%$ | $98.33\%$ | $98.34\%$ | **PASS** (Zero sag) |
| **1. Zero Command** | Non-foot Contact | $< 1\%$ | $0.00\%$ | $0.00\%$ | $0.00\%$ | **PASS** (Zero body drag) |
| **1. Zero Command** | Torque Saturation ($|\tau| \ge 2.45\text{ N}\cdot\text{m}$) | $\le 2\%$ | $3.32\%$ | $2.54\%$ | **$2.15\%$** | **FAIL** (Substantial drop: $3.3\% \to 2.1\%$) |
| **2. Forward Walk** | Base Height $\ge 0.1275\text{ m}$ | $\ge 90\%$ | $100.00\%$ | $100.00\%$ | $100.00\%$ | **PASS** (100% upright) |
| **2. Forward Walk** | $\ge 3$ Feet in Contact | $\ge 99\%$ | $99.40\%$ | $99.42\%$ | $99.40\%$ | **PASS** (Continuous support) |
| **2. Forward Walk** | Non-foot Contact | $< 1\%$ | $0.00\%$ | $0.00\%$ | $0.00\%$ | **PASS** (Zero body contact) |
| **2. Forward Walk** | Static Stability Margin ($p5$) | $p5 \ge 0.0\text{ m}$ | $+0.1142\text{ m}$ | $+0.0801\text{ m}$ | **$+0.0718\text{ m}$** | **PASS** (COM fully enclosed) |
| **2. Forward Walk** | Forward Speed | $\ge 0.120\text{ m/s}$ | $0.0022\text{ m/s}$ | $0.0049\text{ m/s}$ | **$0.0065\text{ m/s}$** | **FAIL** (Robot creeps/freezes) |
| **2. Forward Walk** | Torque Saturation ($|\tau| \ge 2.45\text{ N}\cdot\text{m}$) | $\le 2\%$ | $5.36\%$ | $6.75\%$ | **$6.33\%$** | **FAIL** (Down from 14.25% in Probe 6) |
| **3. Push Recovery**| Terminations (Falls) | $0\text{ falls}$ | $0\text{ falls}$ | $0\text{ falls}$ | **$0\text{ falls}$** | **PASS** (Zero falls in all tests) |
| **3. Push Recovery**| Non-foot Contact | $< 1\%$ | $0.00\%$ | $0.00\%$ | $0.00\%$ | **PASS** |
| **3. Push Recovery**| Push 1 Recovery ($t=3..5\text{s}$) | $\ge 90\%$ | $100.00\%$ | $100.00\%$ | $100.00\%$ | **PASS** (Instant recovery) |
| **3. Push Recovery**| Push 2 Recovery ($t=6..8\text{s}$) | $\ge 90\%$ | $100.00\%$ | $100.00\%$ | **$99.86\%$** | **PASS** (Instant recovery) |
| **3. Push Recovery**| Torque Saturation ($|\tau| \ge 2.45\text{ N}\cdot\text{m}$) | $\le 2\%$ | $5.76\%$ | $7.10\%$ | **$6.65\%$** | **FAIL** |
| **OVERALL GATE** | **All 15 Conditions** | **100% PASS**| **FAIL** | **FAIL** | **FAIL** | **FAIL** |

---

## 3. Report-Only Diagnostics & Trends Across Checkpoints

| Diagnostic Metric | Iteration 500 | Iteration 1000 | Iteration 1500 (`model_1499.pt`) | Target / Physical Context |
| :--- | :---: | :---: | :---: | :--- |
| **Dedicated Pure-Yaw ($\omega_z = 0.40\text{ rad/s}$)** | $+0.0264\text{ rad/s}$ ($6.6\%$) | $+0.0377\text{ rad/s}$ ($9.4\%$) | **$+0.0400\text{ rad/s}$ ($10.0\%$)** | Up from 6.4% in old model; single-env rollout reached **$+0.2049\text{ rad/s}$** |
| **Pure-Yaw Mean Feet Grounded** | $5.89 / 6.0$ | $5.92 / 6.0$ | $5.92 / 6.0$ | Robot keeps nearly all feet planted |
| **Pure-Yaw Torque Saturation** | $4.17\%$ | $2.86\%$ | **$2.71\%$** | Steady decrease toward the 2.0% boundary |
| **Strafe Velocity ($v_y = 0.12\text{ m/s}$)** | $+0.0163\text{ m/s}$ ($13.6\%$) | $+0.0207\text{ m/s}$ ($17.2\%$) | **$+0.0230\text{ m/s}$ ($19.1\%$)** | Gradual increase |
| **Joint Limit Proximity ($\le 3^\circ$, Coxa)** | $0.00\%$ | $0.00\%$ | **$0.00\%$** | Never near mechanical stop |
| **Joint Limit Proximity ($\le 3^\circ$, Thigh)**| $0.00\%$ | $0.00\%$ | **$0.00\%$** | Never near mechanical stop |
| **Joint Limit Proximity ($\le 3^\circ$, Foot)** | $0.00\%$ | $0.00\%$ | **$0.00\%$** | Never near mechanical stop |
| **Swing Clearance $L_1$ (Leg 1)** | $1.99\text{ cm}$ | $2.07\text{ cm}$ | **$2.08\text{ cm}$** | *1–3 cm target (flat ground only)*: **MET** |
| **Swing Clearance $L_2$ (Leg 2)** | $2.05\text{ cm}$ | $2.02\text{ cm}$ | **$2.04\text{ cm}$** | *1–3 cm target (flat ground only)*: **MET** |
| **Swing Clearance $L_3$ (Leg 3)** | $2.05\text{ cm}$ | $2.02\text{ cm}$ | **$2.12\text{ cm}$** | *1–3 cm target (flat ground only)*: **MET** |
| **Swing Clearance $L_4$ (Leg 4)** | $2.06\text{ cm}$ | $2.06\text{ cm}$ | **$2.12\text{ cm}$** | *1–3 cm target (flat ground only)*: **MET** |
| **Swing Clearance $L_5$ (Leg 5)** | $2.05\text{ cm}$ | $2.06\text{ cm}$ | **$2.07\text{ cm}$** | *1–3 cm target (flat ground only)*: **MET** |
| **Swing Clearance $L_6$ (Leg 6)** | $2.02\text{ cm}$ | $2.05\text{ cm}$ | **$2.07\text{ cm}$** | *1–3 cm target (flat ground only)*: **MET** |

---

## 4. Why Forward Speed Failed & The Statics Bottleneck

### 4.1 The Mechanism
In Chunk 1, the policy converged to an **ultra-conservative, highly stable, 6-legged creeping / standing posture** ($v_x = 0.0065\text{ m/s}$):
1. **Statics vs Penalty Threshold**:
   A 5.0 kg hexapod resting horizontally with thigh lever arm $0.115\text{ m}$ requires:
   $$\tau_{\text{tripod}} = \frac{5.0 \cdot 9.81 \cdot 0.115}{3} = 1.880\text{ N}\cdot\text{m}$$
   The torque soft limit was set to $\tau_{\text{soft}} = 1.95\text{ N}\cdot\text{m}$ with penalty weight $-2.0$. The margin between static tripod holding and penalty trigger is only $0.07\text{ N}\cdot\text{m}$ ($3.6\%$).
2. **Dynamic Penalty Overwhelming Forward Reward**:
   Lifting 3 legs in an alternating tripod gait inevitably causes dynamic acceleration and impact torques above $2.0\text{--}2.4\text{ N}\cdot\text{m}$. With weight $-2.0$, this imposes a $-20\text{ to }-40\text{/s}$ penalty.
   Conversely, keeping 5 to 6 legs planted divides the $49.05\text{ N}$ load across 6 legs:
   $$\tau_{\text{6-leg}} = \frac{5.0 \cdot 9.81 \cdot 0.115}{6} = 0.940\text{ N}\cdot\text{m}$$
   $0.940\text{ N}\cdot\text{m}$ is far below $1.95\text{ N}\cdot\text{m}$, incurring **zero torque penalty**.
3. **Reward Landscape Optimization**:
   The policy learned that by standing firm or creeping at $< 0.01\text{ m/s}$:
   - Zero falls (avoids $-200$ penalty)
   - Zero non-foot contacts (avoids $-1.0$ penalty)
   - 100% height target achieved ($+50\text{/s}$)
   - Low torque saturation ($2.15\%$)
   - In 10% standing environments and 30% yaw environments, it collects substantial reward without needing to walk forward.
   Hence, the optimizer chose stability and torque compliance over forward speed.

---

## 5. Honest Gate Verdicts

### Verdict (a): Continuing to 3000 iterations?
> **VERDICT: NO (not without adjusting the reward/statics balance).**  
> Continuing training for another 1500 iterations with the current reward formulation will merely solidify the creeping/standing minimum. The policy has already settled into a shallow loss basin ($V_{\text{loss}} \approx 0.05, S_{\text{loss}} \approx -0.013$) where creeping satisfies safety and torque penalties. 

### Verdict (b): Starting Stage 2 Terrain?
> **VERDICT: NO.**  
> A policy that creeps at $0.0065\text{ m/s}$ on flat ground will instantly stall on terrain hurdles, $0.30$ slopes, and box stairs. Stage 2 curriculum requires vigorous stepping with active foot clearance.

### Verdict (c): Putting Policy on Real Robot?
> **VERDICT: ABSOLUTE NO.**  
> Under no circumstances should this checkpoint be deployed to physical hardware:
> 1. Real robot has no verified servo telemetry loop, IMU filtering pipeline, or physical emergency stop.
> 2. Zero-calibration offsets on physical servos are unmeasured.
> 3. The policy does not walk dynamically forward.

---

## 6. Real-Robot Measurement Checklist

Before any hardware deployment can be considered, the following physical measurements MUST be recorded on the real robot:

1. **Physical Total Mass Measurement**:
   - Place fully assembled robot (chassis, 18 servos, battery, Jetson Orin Nano, buck converters, cables, cameras) on a digital scale.
   - Record exact total mass $M_{\text{real}}$. If $M_{\text{real}} \approx 3.2\text{--}3.8\text{ kg}$ rather than $5.0\text{ kg}$, holding torque drops to $1.20\text{--}1.43\text{ N}\cdot\text{m}$, eliminating the statics bottleneck.
2. **Servo Stance Telemetry (18 Servos)**:
   - Power up the robot in nominal standing posture with feet resting on the ground.
   - Query each of the 18 Waveshare ST3215 servos via serial bus for:
     - Raw position tick: $P_i \in [0, 4095]$
     - Calculated angle in degrees / radians
     - Present load / current: $I_i\text{ (mA)}$
     - Operating voltage: $V_i\text{ (V)}$ (confirm whether servos see 11.4V–12.6V direct 3S LiPo).
3. **Joint Sign & Direction Alignment**:
   - For each leg (1 to 6) and each joint (coxa, thigh, foot), command a $+5^\circ$ positive displacement:
     - Verify coxa rotates counter-clockwise viewed from top.
     - Verify thigh elevates leg upward.
     - Verify foot extends outward/downward.
     - Document sign flips required in ROS 2 hardware bridge.
4. **Mechanical Limit Verification**:
   - Gently back-drive each joint by hand with power off to mechanical hard stops:
     - Measure coxa mechanical stop range vs simulated $[\pm 45^\circ]$ ($\pm 0.785\text{ rad}$).
     - Measure thigh mechanical stop range vs simulated $[\pm 90^\circ]$ ($\pm 1.571\text{ rad}$).
     - Measure foot mechanical stop range vs simulated $[-\pi, 0]$ ($-3.1416\text{ to } 0\text{ rad}$).
5. **Emergency Stop (E-Stop) & Safety Interlocks**:
   - Implement physical hardware kill switch cutting battery power to the servo bus.
   - Implement software watchdog in ROS 2 node cutting PWM / serial torque enable if heartbeat drops for $> 100\text{ ms}$.
