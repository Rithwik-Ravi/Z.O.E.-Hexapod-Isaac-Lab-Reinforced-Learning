# Stage 1 Readiness & Verification Report: `Zoe-Hexapod-Flat-Deploy-v0`

**Date**: 2026-10-10  
**Task**: `Zoe-Hexapod-Flat-Deploy-v0` (Stage 1 Flat Terrain Locomotion)  
**Evaluation Target**: Verification of Stance, MDP Specification, Rewards, and Gate Criteria before Full Training.  

---

## 1. AS-BUILT vs SPEC Audit

### 1.1 Specification Compliance Matrix

| Item | Status | File & Line | Implemented / Actual Value | Spec Requirement |
| :--- | :---: | :--- | :--- | :--- |
| **Actor Observation Terms & Dimension** | **DONE** | [`env_cfg_flat.py:80-129`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/env_cfg_flat.py#L80-L129) | **63-D total**: `base_ang_vel` (3), `projected_gravity` (3), `velocity_commands` (3), `joint_pos` (18), `joint_vel` (18), `actions` (18). **No `base_lin_vel`**. | 63-D, deployable, no base linear velocity |
| **Observation Noise** | **DONE** | [`env_cfg_flat.py:86-127`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/env_cfg_flat.py#L86-L127)<br>[`params.py:23-28`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/params.py#L23-L28) | `enable_corruption = True`.<br>Gyro: Uniform($\pm 0.2\text{ rad/s}$)<br>Gravity: Uniform($\pm 0.05$)<br>Joint Pos: Uniform($\pm 0.01\text{ rad}$)<br>Joint Vel: Uniform($\pm 1.5\text{ rad/s}$) | Active observation corruption with realistic sensor noise |
| **Critic Group & Privileged Terms** | **DONE** | [`env_cfg_flat.py:131-187`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/env_cfg_flat.py#L131-L187) | **92-D total**: 63-D policy terms + `base_lin_vel` (3), `foot_contacts` (6), `applied_torques` (18), `base_height` (1), `belly_contact` (1). | Separate asymmetric critic with privileged state |
| **Domain Rand: Friction** | **DONE** | [`env_cfg_flat.py:192-202`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/env_cfg_flat.py#L192-L202)<br>[`params.py:44-45`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/params.py#L44-L45) | Static: $[0.4, 1.2]$, Dynamic: $[0.3, 1.0]$, Restitution: $0.0$, 64 buckets, startup mode. | Randomized ground friction |
| **Domain Rand: Per-Body Mass** | **DONE** | [`env_cfg_flat.py:203-220`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/env_cfg_flat.py#L203-L220)<br>[`params.py:46-47`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/params.py#L46-L47) | Base payload: $[-0.2, +0.5]\text{ kg}$ on `base_link_v1`.<br>Body mass scale: $[0.85, 1.15]\times$ on `.*N.*`, startup mode. | Mass perturbation across chassis and limbs |
| **Domain Rand: Center of Mass (COM)** | **DONE** | [`env_cfg_flat.py:221-232`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/env_cfg_flat.py#L221-L232)<br>[`params.py:48`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/params.py#L48) | Offset: $\pm 1.5\text{ cm}$ ($\pm 0.015\text{ m}$) on $x, y, z$ of `base_link_v1`, startup mode. | COM displacement |
| **Domain Rand: Actuator Gains** | **DONE** | [`env_cfg_flat.py:233-242`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/env_cfg_flat.py#L233-L242)<br>[`params.py:49-50`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/params.py#L49-L50) | $K_p$ scale: $[0.8, 1.2]\times$, $K_d$ scale: $[0.8, 1.2]\times$ on `.*Revolute.*`, startup mode. | Stiffness & damping variation |
| **Domain Rand: Joint Zero-Offset** | **DONE** | [`env_cfg_flat.py:243-251`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/env_cfg_flat.py#L243-L251)<br>[`params.py:52`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/params.py#L52) | Offset: $\pm 3.0^\circ$ ($\pm 0.0524\text{ rad}$) added per joint on reset. | Servo calibration offset |
| **Action Latency / Delay** | **MISSING** | [`params.py:53`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/params.py#L53)<br>[`env_cfg_flat.py:68-72`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/env_cfg_flat.py#L68-L72) | Defined in `params.py` as `max_action_delay_steps: 2` (0-40 ms), but **NOT wired into `env_cfg_flat.py`** (`ActionsCfg.joint_pos` uses standard unbuffered `JointPositionActionCfg`). | 0-2 policy step randomized action delay |
| **Domain Rand: Random Pushes** | **DONE** | [`env_cfg_flat.py:268-275`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/env_cfg_flat.py#L268-L275)<br>[`params.py:54-55`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/params.py#L54-L55) | Interval: $[8.0, 15.0]\text{ s}$, velocity impulse $\Delta v_{xy} \in [-0.3, +0.3]\text{ m/s}$. | Periodic lateral perturbation |
| **Domain Rand: Reset Joint Noise** | **DONE** | [`env_cfg_flat.py:260-267`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/env_cfg_flat.py#L260-L267)<br>[`params.py:51`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/params.py#L51) | Offset: $\pm 0.05\text{ rad}$ ($\pm 2.86^\circ$) on reset. Spawn $z \in [0.17, 0.19]\text{ m}$. | Small initial posture perturbation |
| **Command Generator** | **DONE** | [`env_cfg_flat.py:378-390`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/env_cfg_flat.py#L378-L390)<br>[`params.py:13-19`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/params.py#L13-L19) | $v_x \in [-0.25, 0.30]\text{ m/s}$, $v_y \in [-0.20, 0.20]\text{ m/s}$, $\omega_z \in [-0.60, 0.60]\text{ rad/s}$. Resample $[3.0, 5.0]\text{ s}$, `rel_standing_envs=0.10`, `heading_command=False`. | Feasible command envelopes, 10% standing, no heading lock |
| **Reward: Velocity Tracking** | **DONE** | [`env_cfg_flat.py:281-290`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/env_cfg_flat.py#L281-L290) | `track_lin_vel_xy_exp`: $+2.0$ ($\sigma=0.5$), `track_ang_vel_z_exp`: $+0.5$ ($\sigma=0.5$). | Exp velocity tracking kernel |
| **Reward: Terminations** | **DONE** | [`env_cfg_flat.py:291-294`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/env_cfg_flat.py#L291-L294) | `termination_penalty`: $-200.0$. | Negative penalty for illegal termination |
| **Reward: Base Height Shaping** | **DONE** | [`env_cfg_flat.py:295-305`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/env_cfg_flat.py#L295-L305) | `base_height_shaped`: $+1.0$ (exponential kernel centered at $0.155\text{ m}$ + steep barrier below $0.148\text{ m}$). | Prevents crouching equilibrium |
| **Reward: Regularizers** | **DONE** | [`env_cfg_flat.py:306-333`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/env_cfg_flat.py#L306-L333) | `lin_vel_z_l2`: $-2.0$, `ang_vel_xy_l2`: $-0.05$, `dof_torques_l2`: $-2\times 10^{-4}$, `dof_acc_l2`: $-1\times 10^{-7}$, `action_rate_l2`: $-0.05$, `flat_orientation_l2`: $-0.5$, `dof_pos_limits`: $-1.0$. | Smoothness, pitch/roll suppression, effort penalty |
| **Reward: Contact Shaping** | **DONE** | [`env_cfg_flat.py:334-352`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/env_cfg_flat.py#L334-L352) | `feet_air_time`: $+0.25$ ($t_{\text{threshold}}=0.15\text{ s}$), `fewer_than_3_feet_contact`: $-0.5$, `undesired_contacts`: $-1.0$ on `.*NTHIGH.*`. | Step clearance, gait stability, thigh contact penalty |
| **Reward: TensorBoard Metrics** | **DONE** | [`env_cfg_flat.py:354-372`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/env_cfg_flat.py#L354-L372) | Weight $0.0$: `tb_mean_base_height`, `tb_non_foot_contact`, `tb_torque_saturation` ($\tau \ge 2.45\text{ N.m}$), `tb_mean_feet_down`. | Auxiliary metrics in TensorBoard |
| **Terminations** | **DONE** | [`env_cfg_flat.py:396-408`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/env_cfg_flat.py#L396-L408) | `time_out` (20 s), `base_contact` ($F > 1.0\text{ N}$ on `.*base_link.*\|.*NCCUP.*`), `bad_orientation` ($\theta_{\text{tilt}} > 1.2\text{ rad}$), `invalid_state_nan`. | Terminate on belly/coxa impact, flip, or divergence |
| **Play Variant** | **DONE** | [`env_cfg_flat.py:432-440`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/env_cfg_flat.py#L432-L440) | Registered `Zoe-Hexapod-Flat-Deploy-Play-v0`: 16 envs, noise disabled (`enable_corruption=False`), randomization disabled. | Deterministic visualization environment |

---

### 1.2 Milestone Status & Commit Tracking

| Milestone | Description | Status | Relevant Commit Hashes |
| :--- | :--- | :---: | :--- |
| **M0** | USD Asset Audit Script & Asset Specification Report | **DONE** | [`d42e24f`](https://github.com/Rithwik-Ravi/Z.O.E.-Hexapod-Isaac-Lab-Reinforced-Learning/commit/d42e24f) (`scripts/audit_asset.py`, `deploy/asset_report.json`) |
| **M0b** | Neutral Stance Verification & Coordinate Audit | **DONE (Sim)** | [`843cd2f`](https://github.com/Rithwik-Ravi/Z.O.E.-Hexapod-Isaac-Lab-Reinforced-Learning/commit/843cd2f), [`9d70813`](https://github.com/Rithwik-Ravi/Z.O.E.-Hexapod-Isaac-Lab-Reinforced-Learning/commit/9d70813), [`231b0eb`](https://github.com/Rithwik-Ravi/Z.O.E.-Hexapod-Isaac-Lab-Reinforced-Learning/commit/231b0eb) (Pending physical confirmation on hardware) |
| **M1** | Actuator Step-Response & Calibration Model Fit | **DONE** | [`d42e24f`](https://github.com/Rithwik-Ravi/Z.O.E.-Hexapod-Isaac-Lab-Reinforced-Learning/commit/d42e24f) (`scripts/fit_actuator_sim.py`, `deploy/actuator_model.yaml`) |
| **M3** | Stage 1 Flat Terrain Policy Training | **IN PROGRESS** | Blocked on sign-off. Probes completed: `model_299.pt` ([`c0c505a`](https://github.com/Rithwik-Ravi/Z.O.E.-Hexapod-Isaac-Lab-Reinforced-Learning/commit/c0c505a)). |
| **M4** | Policy Deployment Exporter with Stance Gate Refusal | **DONE** | [`865b224`](https://github.com/Rithwik-Ravi/Z.O.E.-Hexapod-Isaac-Lab-Reinforced-Learning/commit/865b224), [`c0c505a`](https://github.com/Rithwik-Ravi/Z.O.E.-Hexapod-Isaac-Lab-Reinforced-Learning/commit/c0c505a) (`scripts/rsl_rl/export.py`, `tests/test_export.py`) |
| **M5** | Robustness Evaluation & Gate Validation Suite | **DONE** | [`d42e24f`](https://github.com/Rithwik-Ravi/Z.O.E.-Hexapod-Isaac-Lab-Reinforced-Learning/commit/d42e24f), [`c0c505a`](https://github.com/Rithwik-Ravi/Z.O.E.-Hexapod-Isaac-Lab-Reinforced-Learning/commit/c0c505a) (`scripts/eval_honest_gate.py`, `eval_robustness.py`) |
| **M6** | Stage 2 Rough Terrain Curriculum Training | **NOT STARTED** | Configs created (`env_cfg_terrain.py`), blocked on Stage 1 sign-off. |
| **M7** | Jetson / ROS 2 Hardware Deployment Integration | **NOT STARTED** | Blocked on Stage 1 & 2 trained checkpoints and hardware bridge. |

---

### 1.3 Robot Mass Provenance & Discrepancy

> [!WARNING]
> **Mass Provenance Disclosure**:
> - **Simulator's Total Mass**: **`0.6305 kg`** (630.5 grams) as defined inside [`hexapod/hexa/hexa.usd`](file:///home/rithwik/SpdrBot-main/hexapod/hexa/hexa.usd) (`base_link_v1`: 0.1779 kg; 6 coxa links: 0.0278 kg each; 6 femur links: 0.0206 kg each; 6 tibia links: 0.0271 kg each).
> - **Source of the 2.0 kg Figure**: The 2.0 kg figure did **NOT** come from a physical measurement of the robot. It was an unverified command-line default argument (`parser.add_argument("--robot_mass", default=2.0)`) in `scripts/audit_asset.py:32`, which was recorded into `deploy/asset_report.json` and propagated into analytical calculations.
> - **Physical Reality**: 18 Feetech STS3215 servos weigh $\approx 55\text{ g} \times 18 = 0.99\text{ kg}$ on their own, before accounting for aluminum brackets, 3D printed parts, battery pack, Jetson Orin Nano, and dual ESP32 carrier boards. The true assembled robot is expected to weigh substantially more than the 0.63 kg in USD.
> - **Action Required**: **Please provide the actual weighed value of the physical robot on a scale.**

---

## 2. Stance Geometry & Kinematics Audit

### 2.1 Default Pose Foot Tip Coordinates (Base Frame)

Simulated at default pose (`0.0 rad` USD zero angles across all 18 joints) with zero action for 50 settling steps:
- **Standing Base Height**: **`0.1583 m`** (Nominal spec: $0.158\text{ m}$).

| Leg | Body Prim | Radius ($r$) | Polar Angle ($\theta$) | Height ($z_{\text{rel}}$) | Tibia CoM World $z$ | Local Shank Bounding Box |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Leg 1** | `NFOOT1` | $0.2095\text{ m}$ | $+107.27^\circ$ ($+1.872\text{ rad}$) | $+0.0270\text{ m}$ | $0.1854\text{ m}$ | $z \in [-0.1598, +0.0352]\text{ m}$, len $= 0.195\text{ m}$ |
| **Leg 2** | `NFOOT2` | $0.1957\text{ m}$ | $+55.80^\circ$ ($+0.974\text{ rad}$) | $+0.0267\text{ m}$ | $0.1851\text{ m}$ | $z \in [-0.1598, +0.0352]\text{ m}$, len $= 0.195\text{ m}$ |
| **Leg 3** | `NFOOT3` | $0.2220\text{ m}$ | $-22.64^\circ$ ($-0.395\text{ rad}$) | $+0.0269\text{ m}$ | $0.1853\text{ m}$ | $z \in [-0.1598, +0.0352]\text{ m}$, len $= 0.195\text{ m}$ |
| **Leg 4** | `NFOOT4` | $0.2104\text{ m}$ | $-73.41^\circ$ ($-1.281\text{ rad}$) | $+0.0271\text{ m}$ | $0.1854\text{ m}$ | $z \in [-0.1598, +0.0352]\text{ m}$, len $= 0.195\text{ m}$ |
| **Leg 5** | `NFOOT5` | $0.1948\text{ m}$ | $-123.69^\circ$ ($-2.159\text{ rad}$) | $+0.0268\text{ m}$ | $0.1851\text{ m}$ | $z \in [-0.1598, +0.0352]\text{ m}$, len $= 0.195\text{ m}$ |
| **Leg 6** | `NFOOT6` | $0.2221\text{ m}$ | $+157.30^\circ$ ($+2.745\text{ rad}$) | $+0.0266\text{ m}$ | $0.1850\text{ m}$ | $z \in [-0.1598, +0.0352]\text{ m}$, len $= 0.195\text{ m}$ |

### 2.2 Radial vs Bilateral Symmetry Analysis

- **Chassis Coxa Mounts (`NCCUP1..6`)**:
  - Radii: $r = [0.0966, 0.0966, 0.0966, 0.0966, 0.0966, 0.0966]\text{ m}$.
  - Mount Angles: $[+75.43^\circ, +15.43^\circ, -44.58^\circ, -104.58^\circ, -164.58^\circ, +135.42^\circ]$.
  - Angular Deltas: $\Delta \theta = 60.00^\circ$ uniformly.  
  **The chassis mounting ring is 100% six-fold radially symmetric ($C_6$).**
- **Foot Stance Footprint (`NFOOT1..6`)**:
  - Radii vary from $0.195\text{ m}$ (Legs 2 & 5) to $0.210\text{ m}$ (Legs 1 & 4) to $0.222\text{ m}$ (Legs 3 & 6).
  - The ground contact footprint is **bilaterally symmetric** along the robot's longitudinal sagittal plane, but **NOT six-fold radially symmetric** because the USD zero rest pose carries non-radial leg orientations from CAD.

### 2.3 Definition of Foot Tips & Contact Location

- `NFOOT1..6` is the rigid **tibia shank link**, having a total length of $0.1950\text{ m}$.
- The physical tip is defined as the distal spherical cap at local coordinate $z_{\text{min}} = -0.1598\text{ m}$.
- During standing contact at default pose, the tibia CoM is at world height $0.1852\text{ m}$, while the distal tip reaches the ground ($z \approx 0.0\text{ m}$).
- Because the tibia is pitched outward at an angle of $\approx 35^\circ$ from vertical, contact occurs right at the rounded distal tip cap, with shank ground clearance increasing rapidly as you move upward toward the knee joint.

### 2.4 Why Foot-Joint Torque Exceeds Thigh Torque at Standing

In static standing, measured joint torques are:
- Thigh joints (`NCCUP*_Revolute_*`): $-0.22\text{ to } -0.38\text{ N.m}$
- Foot / knee joints (`NTHIGH*_Revolute_*`): **$-0.30\text{ to } -0.77\text{ N.m}$**

**Mechanical Cause**:
In the kinematic chain: $\text{Base} \to \text{Coxa} \to \text{Thigh} \to \text{Femur} \to \text{Knee/Foot Joint} \to \text{Tibia} \to \text{Tip}$.  
The ground reaction force $F_z$ acts vertically upward at the foot tip.  
The torque on each joint is $\tau = F_z \times d_{\perp}$, where $d_{\perp}$ is the horizontal distance from the joint rotation axis to the vertical ground force line of action.  
In the USD zero pose, the femur is angled slightly downward, placing the knee joint at radius $r \approx 0.14\text{ m}$. However, the tibia is slanted outward to radius $r \approx 0.21\text{ m}$.  
The horizontal moment arm from the knee joint to the ground contact point is $d_{\text{knee-to-foot}} \approx 0.07\text{--}0.09\text{ m}$. The foot joint acts as a cantilever holding the entire horizontal outreach of the slanted tibia. In contrast, the femur link spans a smaller net horizontal moment arm between the hip and knee, resulting in lower torque at the thigh joint than at the foot joint.

### 2.5 Comparison with Real Neutral Stance (Appendix B IK Convention)

In Appendix B of the build plan, the neutral stance specifies:
- **Legs radial**: Coxa rotated to point each leg straight out along its radial vector.
- **Femur near horizontal**: Thigh angle small ($\approx +7.7^\circ$).
- **Tibia near vertical**: Tibia angle $\approx +7.4^\circ$ relative to gravity, placing the tip directly under the knee joint.
- **Coxa plane height**: $\approx 0.15\text{ m}$ above ground.

**Physical Differences**:
1. In the USD zero pose, the tibia is slanted outwards at $\sim 35^\circ$ rather than near vertical. This widens the stance radius to $0.22\text{ m}$ (versus $0.16\text{ m}$ in IK) and increases static knee torque.
2. In the USD zero pose, coxa joint angles are all $0.0\text{ rad}$, whereas the IK convention commands non-zero coxa angles to align the femur radially with the body center.

---

## 3. Honest Stance Gate Evaluation

Specified evaluation thresholds:
- Base Height: $\ge 0.90 \times \text{STANDING\_BASE\_HEIGHT} = 0.90 \times 0.1583\text{ m} = \mathbf{0.1425\text{ m}}$.
- Ground Contact Posture: $\ge 3$ feet down in $\ge \mathbf{99.0\%}$ of steps.
- Body Shell Clearance: Non-foot contact $< \mathbf{1.0\%}$ of steps.

### 3.1 Old Probe Checkpoint (`2026-10-10_13-17-50/model_299.pt`)

Evaluated with [`scripts/eval_honest_gate.py`](file:///home/rithwik/SpdrBot-main/scripts/eval_honest_gate.py) (16 envs):

| Condition | Base Height (Mean, Min) | Non-Foot Contact | $\ge 3$ Feet Down | Mean Feet Down | Action Jitter | Joint Vel ($p50/p95/\max$) | Torque Sat ($\ge 2.5\text{ N.m}$) | Foot Slip | Result |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cond 1: Zero Cmd (5s)** | $0.1284\text{ m}$ (Min: $0.1214$) | **0.00%** | 98.17% | 2.97 / 6 | 0.0179 rad | $0.05 / 0.84 / 6.56$ rad/s | 7.86% | Mean: 0.011 m/s | **FAIL ❌** |
| **Cond 2: Fwd 0.15 m/s (10s)** | $0.1296\text{ m}$ (Min: $0.1183$) | **0.00%** | 98.96% | 3.09 / 6 | 0.0741 rad | $0.43 / 1.56 / 7.74$ rad/s | 10.08% | Mean: 0.127 m/s | **FAIL ❌** |
| **Cond 3: Extremes & Pushes (10s)** | $0.1293\text{ m}$ (Min: $0.1175$) | **0.00%** | 96.66% | 3.12 / 6 | 0.1003 rad | $0.56 / 2.13 / 9.56$ rad/s | 9.84% | Mean: 0.148 m/s | **FAIL ❌** |
| **OVERALL RESULT** | — | — | — | — | — | — | — | — | **FAIL ❌** |

*Diagnosis*: Old probe never contacts the floor with its belly (non-foot contact 0.00%), but it squats at $0.128\text{--}0.130\text{ m}$, failing the height threshold ($0.1425\text{ m}$) across all 3 conditions.

---

### 3.2 Shaped Probe Checkpoint (`2026-10-10_19-01-52/model_299.pt`)

Evaluated with [`scripts/eval_honest_gate.py`](file:///home/rithwik/SpdrBot-main/scripts/eval_honest_gate.py) (16 envs):

| Condition | Base Height (Mean, Min) | Non-Foot Contact | $\ge 3$ Feet Down | Mean Feet Down | Action Jitter | Joint Vel ($p50/p95/\max$) | Torque Sat ($\ge 2.5\text{ N.m}$) | Foot Slip | Result |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cond 1: Zero Cmd (5s)** | **$0.1541\text{ m}$** (Min: $0.1512$) | **0.00%** | 98.12% | 3.03 / 6 | 0.0158 rad | $0.06 / 0.55 / 7.85$ rad/s | 13.46% | Mean: 0.012 m/s | **FAIL ❌** ($\ge 3$ ft) |
| **Cond 2: Fwd 0.15 m/s (10s)** | **$0.1539\text{ m}$** (Min: $0.1507$) | **0.00%** | **99.09%** | 3.19 / 6 | 0.0763 rad | $0.38 / 1.25 / 12.05$ rad/s | 12.62% | Mean: 0.105 m/s | **PASS ✅** |
| **Cond 3: Extremes & Pushes (10s)** | **$0.1541\text{ m}$** (Min: $0.1490$) | **0.00%** | 96.99% | 3.16 / 6 | 0.0902 rad | $0.44 / 1.83 / 11.07$ rad/s | 12.14% | Mean: 0.129 m/s | **FAIL ❌** ($\ge 3$ ft) |
| **OVERALL RESULT** | — | — | — | — | — | — | — | — | **FAIL ❌** |

*Findings*:
1. Base height increased from $0.1296\text{ m}$ to **$0.1539\text{ m}$** (+2.43 cm), completely clearing the height gate ($> 0.1425\text{ m}$).
2. **Condition 2 (Dynamic Walking) PASSED all criteria**: Height $0.1539\text{ m}$, $\ge 3$ feet down $99.09\%$, non-foot contact $0.00\%$.
3. Condition 1 (Zero command) missed the $\ge 3$ feet down threshold at $98.12\%$ due to minor idle stepping/weight transfer.
4. Condition 3 missed at $96.99\%$ because abrupt $\pm 0.3\text{ m/s}$ pushes knock multiple legs into transient recovery.

---

## 4. Base-Height Shaping Analysis & Probe Comparison

### 4.1 Quadratic Deficit Penalty Analysis

In the previous unshaped configuration with `base_height_l2` ($w = -50.0$):
$$\text{Penalty}_{\text{step}} = -50.0 \times (h - h^*)^2, \quad \text{Penalty}_{\text{sec}} = 50\text{ Hz} \times \text{Penalty}_{\text{step}} = -2500 \times \Delta h^2\text{ / s}$$
The maximum forward velocity tracking reward is $+2.0\text{ / step} = \mathbf{+100.0\text{ / s}}$.

| Height Deficit ($\Delta h$) | Per-Step Penalty | Per-Second Penalty | Share of Max Tracking Reward |
| :---: | :---: | :---: | :---: |
| **3 cm deficit** ($h = 0.128\text{ m}$) | $-0.0450$ | $\mathbf{-2.25\text{ / s}}$ | **2.25%** |
| **5 cm deficit** ($h = 0.108\text{ m}$) | $-0.1250$ | $\mathbf{-6.25\text{ / s}}$ | **6.25%** |
| **8 cm deficit** ($h = 0.078\text{ m}$) | $-0.3200$ | $\mathbf{-16.00\text{ / s}}$ | **16.00%** |

*Why Quadratic Penalty Failed*: Because quadratic loss has zero derivative at $\Delta h = 0$, the penalty at a 3 cm deficit is a trivial 2.25% of the tracking reward. The policy gained 95%+ of its velocity tracking reward by crouching and dragging, easily overpowering the tiny height gradient.

### 4.2 Proposed Shaping Function

We replaced `base_height_l2` with `base_height_shaped` ([`mdp/rewards.py:91-115`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/mdp/rewards.py#L91-L115)):
$$R_{\text{height}} = \exp\left(-\left(\frac{h - h_{\text{target}}}{\sigma}\right)^2\right) - \mathbb{I}(h < h_{\text{min}}) \left[ w_{\text{quad}} (h_{\text{min}} - h)^2 + w_{\text{lin}} (h_{\text{min}} - h) \right]$$
with $h_{\text{target}} = 0.155\text{ m}$, $h_{\text{min}} = 0.148\text{ m}$, $\sigma = 0.015\text{ m}$, $w_{\text{quad}} = 150.0$, $w_{\text{lin}} = 10.0$.

### 4.3 Probe Comparison (Old Probe vs Shaped Probe)

| Metric | Old Probe (`13-17-50/model_299.pt`) | Shaped Probe (`19-01-52/model_299.pt`) | Improvement |
| :--- | :---: | :---: | :---: |
| **Mean Walking Height** | $0.1296\text{ m}$ | **$0.1539\text{ m}$** | **+2.43 cm** (Target $\ge 0.145\text{ m}$ achieved) |
| **Cond 2 $\ge 3$ Feet Down** | 98.96% | **99.09%** | **PASS ✅** |
| **Forward Velocity Tracking** | Achieved $\sim 0.11\text{ m/s}$ | Achieved **$0.1895\text{ m/s}$** on $0.20\text{ m/s}$ cmd | **Tracking error reduced to 5.2%** |
| **Non-Foot Contact** | 0.00% | **0.00%** | Maintained zero body shell contact |

---

## 5. TensorBoard Metrics & Export Gate Refusal

### 5.1 TensorBoard Metric Monitors

Added to `RewardsCfg` ([`env_cfg_flat.py:354-372`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/env_cfg_flat.py#L354-L372)) with weight $0.0$:
- `tb_mean_base_height`: logs average robot base z height.
- `tb_non_foot_contact`: logs fraction of steps with contact force on `base_link`, `NCCUP`, or `NTHIGH`.
- `tb_torque_saturation`: logs fraction of joint steps where $|\tau| \ge 2.45\text{ N.m}$ (98% of 2.5 N.m stall limit).
- `tb_mean_feet_down`: logs average number of planted feet per step.

### 5.2 Deployment Export Refusal

Updated [`scripts/rsl_rl/export.py`](file:///home/rithwik/SpdrBot-main/scripts/rsl_rl/export.py) to automatically execute the stance gate prior to exporting ONNX / TorchScript:
- **Refusal Test Log** ([`task-1225.log`](file:///home/rithwik/.gemini/antigravity/brain/d189b8ab-2fb1-4d3d-912c-b8d65b2f6f30/.system_generated/tasks/task-1225.log)):
  ```text
  [EXPORT] Stance Gate Check: Mean Height=0.1323m (Threshold: >=0.1425m), Non-Foot Contacts=0.00% (Threshold: <1.0%)
  [EXPORT ERROR] Policy FAILED stance gate! Export refused to prevent deploying belly-dragging policies. Use --force to override.
  ```
- Exits cleanly with status code 1. Export is permitted only if `--force` is explicitly passed.

---

## 6. Single-Env 10s Gait Evaluation Report

Evaluated using [`scripts/eval_gait_report.py`](file:///home/rithwik/SpdrBot-main/scripts/eval_gait_report.py) on checkpoint `logs/rsl_rl/zoe_hexapod_deploy/2026-10-10_19-01-52/model_299.pt`:

### 6.1 Per-Leg Duty Factors & Foot Metrics (Forward Walking 2.0s - 6.0s)

| Leg Index | Body Name | Duty Factor | Swing Clearance | Mean Stance Slip |
| :---: | :---: | :---: | :---: | :---: |
| **Leg 1** | `NFOOT1` | 46.5% | $16.18\text{ cm}$ | $0.0969\text{ m/s}$ |
| **Leg 2** | `NFOOT2` | 55.5% | $16.34\text{ cm}$ | $0.1026\text{ m/s}$ |
| **Leg 3** | `NFOOT3` | 49.5% | $18.13\text{ cm}$ | $0.0802\text{ m/s}$ |
| **Leg 4** | `NFOOT4` | 52.5% | $18.49\text{ cm}$ | $0.0901\text{ m/s}$ |
| **Leg 5** | `NFOOT5` | 47.5% | $19.49\text{ cm}$ | $0.0334\text{ m/s}$ |
| **Leg 6** | `NFOOT6` | 55.0% | $18.73\text{ cm}$ | $0.0704\text{ m/s}$ |

### 6.2 Alternation & Tripod Phase Analysis

- **Tripod Group A (Legs 1, 3, 5)** Mean Duty Factor: **47.8%**
- **Tripod Group B (Legs 2, 4, 6)** Mean Duty Factor: **54.3%**
- **Cross-Group Correlation**: **`-0.9813`** (Nearly perfect antiphase alternation).
- **Assessment**: **STRONG ALTERNATING TRIPOD GAIT CONFIRMED ✅**. The data show clear alternating tripod coordination without dragging.

### 6.3 Command Step Tracking

- **Forward Step** ($v_x = 0.20\text{ m/s}$ cmd): Achieved mean $v_x = \mathbf{+0.1895\text{ m/s}}$ (Error: $0.0105\text{ m/s}$).
- **Yaw Step** ($\omega_z = 0.50\text{ rad/s}$ cmd): Achieved mean $\omega_z = \mathbf{+0.4879\text{ rad/s}}$ (Error: $0.0121\text{ rad/s}$).
- **Strafe Step** ($v_y = 0.15\text{ m/s}$ cmd): Achieved mean $v_y = \mathbf{+0.1716\text{ m/s}}$ (Error: $0.0216\text{ m/s}$).

### 6.4 12-Second Contact Timeline (10 Hz Downsampled, `#` = Contact, `.` = Air)

```text
Time (s): 000000000011111111112222222222333333333344444444445555555555666666666677777777778888888888999999999900000000001111111111
Leg 1 : .#....................###...###..###...###..###...###..###...#####################....#####....#####....###....####....#
Leg 2 : ..####################...###..###...###..###...###...##...###.....................####.....####.....####...####....####.
Leg 3 : .#....................###...###..###...###..###...###..###...######################...#####....#####....###....####....#
Leg 4 : ..####################...###...##...###...##...###...##...###......................###.....####.....####...####....####.
Leg 5 : .#....................###...###..###...###..###...###..###...######################...#####....#####....###.....###....#
Leg 6 : ..#####################..###...###..###...###..###...###..###......................###.....####.....####...#####...####.
```

---

## 7. Stage 1 Training Plan & CLI Commands

### 7.1 Measured Simulation Throughput

From probe training logs ([`task-1231.log`](file:///home/rithwik/.gemini/antigravity/brain/d189b8ab-2fb1-4d3d-912c-b8d65b2f6f30/.system_generated/tasks/task-1231.log)):
- **Hardware**: NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM), Intel i9-14900HX (24 cores).
- **Environment Count**: 1024 envs.
- **Measured Iteration Time**: **3.63s – 3.83s** (Sampling: 3.48s, PPO update: 0.35s, $\approx 12,850\text{ steps/s}$).
- **Wall Clock Time**: **4.15 seconds per iteration** (including reset handling).

### 7.2 Chunked Training Schedule (Total Budget: 3,000 – 4,500 Iterations)

- **Chunk 1 (Iterations 0 – 1,500)**:
  - Duration: $\approx 103\text{ minutes}$ ($\sim 1.7\text{ hours}$).
  - Milestone checkpoint: `model_1499.pt`.
  - Gate check: Run `scripts/eval_honest_gate.py` on `model_1499.pt`.
- **Chunk 2 (Iterations 1,500 – 3,000)**:
  - Duration: $\approx 103\text{ minutes}$ ($\sim 1.7\text{ hours}$).
  - Milestone checkpoint: `model_2999.pt`.
  - Gate & Gait check: Verify alternating tripod duty cycle and velocity tracking error $< 5\%$.
- **Chunk 3 (Iterations 3,000 – 4,500, Optional Fine-Tuning)**:
  - Duration: $\approx 103\text{ minutes}$ ($\sim 1.7\text{ hours}$).
  - Milestone checkpoint: `model_4499.pt`.
- **Stop Criteria**:
  1. Mean base height $\ge 0.150\text{ m}$ under full velocity command range.
  2. Non-foot contact rate $= 0.00\%$.
  3. Tripod phase cross-correlation $\le -0.95$ with duty factors within $[45\%, 55\%]$.
  4. Honest stance gate PASS across Condition 1 and Condition 2.

### 7.3 Exact CLI Train and Resume Commands

> [!IMPORTANT]
> Real CLI flags inspected and confirmed against [`scripts/rsl_rl/train.py`](file:///home/rithwik/SpdrBot-main/scripts/rsl_rl/train.py) and [`scripts/rsl_rl/cli_args.py`](file:///home/rithwik/SpdrBot-main/scripts/rsl_rl/cli_args.py).

#### Initial Training Command (Chunk 1: 0 to 1,500 iterations):
```bash
python scripts/rsl_rl/train.py \
  --task Zoe-Hexapod-Flat-Deploy-v0 \
  --num_envs 1024 \
  --max_iterations 1500 \
  --headless
```

#### Resume Training Command (Chunk 2: 1,500 to 3,000 iterations):
```bash
python scripts/rsl_rl/train.py \
  --task Zoe-Hexapod-Flat-Deploy-v0 \
  --num_envs 1024 \
  --max_iterations 3000 \
  --resume \
  --load_run <RUN_TIMESTAMP_DIRECTORY> \
  --checkpoint model_1499.pt \
  --headless
```

*(Where `<RUN_TIMESTAMP_DIRECTORY>` is the exact folder name under `logs/rsl_rl/zoe_hexapod_deploy/`, e.g., `2026-10-10_19-01-52`)*.

---

## 8. Final Verdict & Blocking Items

**NOT READY for full Stage 1 training.**

### Blocking Items to Resolve Before Launch:
1. **Action Delay Not Implemented in MDP**: `max_action_delay_steps = 2` is specified in `params.py:53`, but is missing from `env_cfg_flat.py:68-72`. Action delay is critical for sim-to-real transfer over the 50 Hz serial servo bus; adding delay post-training will invalidate learned policies.
2. **Unverified Robot Mass in Simulation**: USD total mass is `0.6305 kg`, whereas the 18 servos alone weigh $\approx 0.99\text{ kg}$. The 2.0 kg figure was an unweighed CLI default. The true physical mass of the assembled robot must be obtained from the owner and calibrated in USD before conducting long training runs.
3. **Stance Gate Threshold Refinement**: While Condition 2 (Walking) passes all gates (Height $0.1539\text{ m}$, 99.09% $\ge 3$ feet down, 0.00% non-foot contacts), Condition 1 (98.12%) and Condition 3 (96.99%) slightly missed the $\ge 99.0\%$ feet down threshold due to idle stepping and external push recoveries. The policy requires longer convergence or a slight relax of the push recovery gate threshold.
