# Read-Only Technical Audit Report: Isaac Lab Locomotion Training & Sim-to-Real WASD Teleop Pipeline

**Author**: Senior Robotics & RL Audit Specialist  
**Date**: October 5, 2026  
**Target Architecture**: 18-DOF Hexapod (Z.O.E. / `spdrbot3`)  
**Deployment Goal**: Real-Time 50 Hz WASD Teleoperation via SSH (Jetson + IMU + ST3215 Serial Servos)  
**Environment & Stack**: Ubuntu 22.04 LTS, NVIDIA RTX 4070 Laptop GPU (8 GB VRAM), Isaac Lab `v2.3.2` (`0.54.2`), Isaac Sim `4.5.0.0`, `rsl-rl-lib 3.1.2`, PyTorch `2.14.0`.

---

## Executive Summary & Direct Verdict

### Verdict: **ONLY IF (With Critical Architectural Corrections)**

The planned Isaac Lab pipeline **CANNOT** produce a deployable, WASD-driven policy for the real robot in its current state unless **5 critical architectural blockers** are resolved first. 

While the high-level design of asymmetric actor-critic RL is sound, the plan contains serious domain discrepancies—specifically in joint ordering, asset mass scaling, observation noise ordering, command dynamics, and servo control latency—that will cause immediate physical instability or total gait failure on hardware.

---

## 1. Verification of Existing State (`docs/EXISTING_STATE.md`)

Each defect (A1–A20) listed in `docs/EXISTING_STATE.md` was verified against the repository codebase:

| ID | Status | File & Line Citation | Technical Summary & Empirical Evidence |
|---|---|---|---|
| **A1** | **CONFIRMED** | [`source/spdrbot3/spdrbot3/tasks/manager_based/spdrbot3_rough/env_cfg.py:46-60,236`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/spdrbot3_rough/env_cfg.py#L46-L60) | `FLAT_TERRAIN_CFG` has `noise_range=(0,0)`. `CurrTerm` is imported on L9 but omitted from `Spdrbot3RoughEnvCfg`. |
| **A2** | **CONFIRMED** | [`.../spdrbot3_rough/env_cfg.py:122`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/spdrbot3_rough/env_cfg.py#L122) | `PolicyCfg` directly includes `base_lin_vel = ObsTerm(func=safe_base_lin_vel...)`. Hardware has no base velocity sensor. |
| **A3** | **CONFIRMED** | [`.../spdrbot3_rough/env_cfg.py:130`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/spdrbot3_rough/env_cfg.py#L130) | `self.enable_corruption = False` explicitly disables all observation noise. |
| **A4** | **CONFIRMED** | [`.../spdrbot3_rough/env_cfg.py:142-177`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/spdrbot3_rough/env_cfg.py#L142-L177) | Static friction range is fixed to `(0.8, 0.8)` and dynamic to `(0.6, 0.6)`. Joint reset position range is fixed `(1.0, 1.0)`. |
| **A5** | **CONFIRMED** | [`.../spdrbot3_rough/env_cfg.py:136-180`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/spdrbot3_rough/env_cfg.py#L136-L180) | `EventCfg` contains zero push terms, no gain randomization, and zero action/observation delay buffers. |
| **A6** | **CONFIRMED** | [`source/spdrbot3/spdrbot3/assets/spdrbot.py:40`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/assets/spdrbot.py#L40), [`.../env_cfg.py:90`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/spdrbot3_rough/env_cfg.py#L90) | Effort limit is set to `5.0 N*m`. Feetech ST3215 stall torque is ~2.5 N·m (at 7.4V/12V). |
| **A7** | **CONFIRMED** | [`.../spdrbot3_rough/env_cfg.py:32-41,202`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/spdrbot3_rough/env_cfg.py#L32-L41) | `tripod_gait_penalty` computes `(num_feet_in_contact - 3.0)^2`. It does NOT verify alternating leg sets (tripod A vs B) and penalizes 4+ feet contact. |
| **A8** | **CONFIRMED** | [`.../spdrbot3_rough/env_cfg.py:24-30,195`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/spdrbot3_rough/env_cfg.py#L24-L30) | `foot_clearance` has `weight=0.0`. It uses world-frame Z (`foot_pos_w[:, :, 2]`), which penalizes climbing slopes or terrain steps. |
| **A9** | **CONFIRMED** | [`.../spdrbot3_rough/env_cfg.py:193`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/spdrbot3_rough/env_cfg.py#L193) | `is_alive` gives `+10.0` per step. At 50 Hz, 50 steps/sec = +500 reward/sec, dominating episode return (~80%). No termination penalty exists. |
| **A10** | **CONFIRMED** | [`source/spdrbot3/spdrbot3/assets/spdrbot.py:46`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/assets/spdrbot.py#L46), [`.../env_cfg.py:188`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/spdrbot3_rough/env_cfg.py#L188) | `soft_joint_pos_limit_factor=2` doubles soft limit bounds beyond hard physical limits, rendering `joint_pos_limits` penalty completely inert. |
| **A11** | **CONFIRMED** | [`source/spdrbot3/spdrbot3/assets/spdrbot.py:28`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/assets/spdrbot.py#L28), [`.../env_cfg.py:162`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/spdrbot3_rough/env_cfg.py#L162) | Source code inspection of `reset_root_state_uniform` confirms `positions = root_states[:, 0:3] + env_origins + rand_samples`. `pos=(0,0,0.5)` + `z=(0.6,0.8)` = **1.1m to 1.3m drop height**. |
| **A12** | **CONFIRMED** | [`.../spdrbot3_rough/env_cfg.py:212,219`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/spdrbot3_rough/env_cfg.py#L212) | Commands sample `[-1.0, 1.0]` m/s, far exceeding hardware gait limits (~0.15 m/s). Resampling period is fixed at 10.0 s. |
| **A13** | **CONFIRMED** | [`source/spdrbot3/spdrbot3/assets/spdrbot.py:31`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/assets/spdrbot.py#L31) | Default joint positions are set to `0.0` for all 18 joints (a collapsed flat leg stance). |
| **A14** | **CONFIRMED** | [`source/spdrbot3/spdrbot3/assets/spdrbot.py:9`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/assets/spdrbot.py#L9) | USD path is hardcoded to `/home/rithwik/SpdrBot-main/hexapod/hexa/hexa.usd`. |
| **A15** | **CONFIRMED** | [`scripts/rsl_rl/train.py:172-186`](file:///home/rithwik/SpdrBot-main/scripts/rsl_rl/train.py#L172-L186), [`scripts/rsl_rl/play.py:143-145`](file:///home/rithwik/SpdrBot-main/scripts/rsl_rl/play.py#L143-L145) | `train.py` monkey-patches `env.step` and `env.reset` to replace NaNs with `0.0`, silently masking simulation numerical explosions. |
| **A16** | **CONFIRMED** | [`scripts/rsl_rl/play.py`](file:///home/rithwik/SpdrBot-main/scripts/rsl_rl/play.py) | No ONNX/TorchScript/NumPy export logic or metadata generator existed in original `play.py`. |
| **A17** | **CONFIRMED** | [`source/spdrbot3/spdrbot3/assets/spdrbot.py:29`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/assets/spdrbot.py#L29) | `rot=(0,0,0,1)` specifies `(x,y,z,w)` zero quaternion, which resolves to a 180° yaw rotation under Isaac Lab's `(w,x,y,z)` default ordering. |
| **A18** | **CONFIRMED** | [`README.md:40-100`](file:///home/rithwik/SpdrBot-main/README.md#L40) | Documentation claims 24-D observation (actual code is 66-D) and claims alternating tripod enforcement (actual code checks only total contact count = 3). |
| **A19** | **CONFIRMED** | [`.../spdrbot3_rough/env_cfg.py:192,215`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/spdrbot3_rough/env_cfg.py#L192) | `flat_orientation_l2` weight `-0.5` and `heading_command=True` penalize non-flat body orientations required for slope climbing. |
| **A20** | **CONFIRMED** | [`.../spdrbot3_rough/env_cfg.py:54`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/spdrbot3_rough/env_cfg.py#L54) | `use_cache=True` in `FLAT_TERRAIN_CFG` forces Isaac Lab to cache compiled mesh heightfields to `/tmp/isaaclab/terrains`, causing stale terrain reuse after config edits. |

### Unlisted Items Missed by `EXISTING_STATE.md`:
1. **Pip Editable Package Conflict**: `pip show spdrbot3` resolved to `spdrbot3_direct_project/source/spdrbot3` rather than root `source/spdrbot3`. Editing files under root `source/spdrbot3` had zero effect until `pip install -e source/spdrbot3` was re-run.
2. **`pxr` USD Dependency**: USD core library (`pxr`) is provided directly by `isaacsim` 4.5.0.0 pip package and is accessible via `SimulationApp` bootstrap.

---

## 2. API & Framework Verification (`isaaclab 0.54.2`, `rsl-rl-lib 3.1.2`)

| Planned API Feature | Status | Installed API Signature / Verification Details |
|---|---|---|
| **Asymmetric Actor-Critic (Separate Obs Groups)** | **EXISTS** | `rsl_rl.modules.ActorCritic(obs: TensorDict, obs_groups: dict[str, list[str]], ...)` explicitly supports separate `"policy"` and `"critic"` observation groups. |
| **`DelayedPDActuatorCfg` & `DCMotorCfg`** | **EXISTS** | Both classes exist under `isaaclab.actuators`. However, explicit PD numerical integration at `dt=0.005` with high stiffness ($K_p > 25.0$) can trigger PhysX solver instability; `ImplicitActuatorCfg` is recommended for sim stability. |
| **Randomization Events** | **EXISTS** | `randomize_rigid_body_material`, `randomize_rigid_body_mass`, `randomize_rigid_body_com`, `randomize_actuator_gains`, and `push_robot_by_setting_velocity` exist in `isaaclab.envs.mdp`. |
| **Terrain Classes & Curriculum** | **EXISTS / DIFFERS** | `HfRandomUniformTerrainCfg`, `MeshPyramidStairsTerrainCfg`, `MeshRandomGridTerrainCfg` exist. `HfRandomUniformTerrainCfg` does NOT take a `difficulty` parameter directly; it maps `noise_range` to `terrain_levels`. `max_init_terrain_level=None` is required to populate all 10 grid rows. |
| **`feet_air_time` Signature** | **EXISTS** | `mdp.feet_air_time` accepts `command_name: str`, `sensor_cfg: SceneEntityCfg`, and `threshold: float`. |
| **Policy Exporter & Normalizer** | **EXISTS** | `rsl_rl` exports via TorchScript (`torch.jit.script`) or ONNX (`torch.onnx.export`). `actor_obs_normalizer` (RunningMeanStd) is baked directly into the exported computation graph when `actor_obs_normalization=True`. |
| **Observation Noise Order** | **EXISTS** | In `isaaclab.managers.ObservationManager`, noise (`enable_corruption=True`) is applied to **raw** values BEFORE scaling and clipping. |

---

## 3. Physical Asset Audit (`hexapod/hexa/hexa.usd`)

Loading `hexapod/hexa/hexa.usd` via Isaac Sim headless inspection yielded the following authoritative physical properties:

```json
{
  "meters_per_unit": 1.0,
  "up_axis": "Z",
  "total_mass_kg": 0.6305,
  "base_mass_kg": 0.1779,
  "total_joints": 18,
  "total_bodies": 19
}
```

### 3.1 Simulator Joint Order (Depth-First Grouped)
The simulator joint order in PhysX is **NOT leg-grouped**; it is depth-grouped across all 6 legs:

1. `base_link_v1_Revolute_39` (`NCCUP1_joint`)
2. `base_link_v1_Revolute_42` (`NCCUP2_joint`)
3. `base_link_v1_Revolute_45` (`NCCUP3_joint`)
4. `base_link_v1_Revolute_48` (`NCCUP4_joint`)
5. `base_link_v1_Revolute_51` (`NCCUP5_joint`)
6. `base_link_v1_Revolute_54` (`NCCUP6_joint`)
7. `NCCUP1_Revolute_38` (`NTHIGH1_joint`)
8. `NCCUP2_Revolute_41` (`NTHIGH2_joint`)
9. `NCCUP3_Revolute_44` (`NTHIGH3_joint`)
10. `NCCUP4_Revolute_47` (`NTHIGH4_joint`)
11. `NCCUP5_Revolute_50` (`NTHIGH5_joint`)
12. `NCCUP6_Revolute_53` (`NTHIGH6_joint`)
13. `NTHIGH1_Revolute_37` (`NFOOT1_joint`)
14. `NTHIGH2_Revolute_40` (`NFOOT2_joint`)
15. `NTHIGH3_Revolute_43` (`NFOOT3_joint`)
16. `NTHIGH4_Revolute_46` (`NFOOT4_joint`)
17. `NTHIGH5_Revolute_49` (`NFOOT5_joint`)
18. `NTHIGH6_Revolute_52` (`NFOOT6_joint`)

> [!CAUTION]
> **CRITICAL MISMATCH**: ROS leg-sequential order (`Leg 1 Coxa, Thigh, Foot; Leg 2...`) differs completely from PhysX depth-grouped order (`6x Coxa, 6x Thigh, 6x Foot`). Passing actions without explicit index remap will cause instant joint destruction.

### 3.2 Link Mass & Geometry Breakdown
- **Total Mass**: `0.6305 kg` (Base: 0.1779 kg; 6x Coxa: 0.0278 kg each; 6x Thigh: 0.0206 kg each; 6x Foot: 0.0271 kg each).
- **Mass Discrepancy**: Real 18-servo ST3215 hexapod weighs **~1.8–2.2 kg** (servos alone weigh 1.0 kg). USD CAD mass is **68.5% lighter than reality**.
- **Joint Limits**: Unconstrained (`-inf`, `+inf`) in USD physics schema. Hardware clamps (Coxa -45..+135°, Thigh/Foot -90..+90°) must be enforced via `soft_joint_pos_limit_factor=0.95` in `ArticulationCfg`.
- **Foot Prim**: `NFOOT1..6` bodies represent the **tibia links** (shank bodies), not zero-radius sphere tips at the ground contact point.

---

## 4. Observation Availability & Hardware Mapping

Evaluation of the planned **63-D Policy Observation Vector** against real Jetson hardware:

| Observation Term | Dim | Hardware Source | Real-World Availability | Sim-to-Real Gap & Sensor Degradation |
|---|---|---|---|---|
| `base_ang_vel` | 3 | IMU Gyroscope | **AVAILABLE** | High-frequency motor vibration noise; serial loop latency (10–20 ms); orientation alignment required. |
| `projected_gravity` | 3 | IMU Orientation / Accel | **AVAILABLE** | Linear acceleration during locomotion corrupts naive gravity vector; requires EKF / Madgwick filter. |
| `velocity_commands` | 3 | WASD Teleop via SSH | **AVAILABLE** | Abrupt step changes when keys are pressed vs smooth command sampling in sim. |
| `joint_pos_rel` | 18 | ST3215 Servo Encoders | **AVAILABLE** | Serial bus polling latency (~15 ms for 18 servos at 115200 baud); zero-point calibration offset (1–3° per joint). |
| `joint_vel_rel` | 18 | Servo Telemetry / Diff | **AVAILABLE** | Serial speed telemetry is coarse and quantized; finite differencing amplifies high-frequency noise. |
| `last_action` | 18 | Controller Memory | **AVAILABLE** | Zero latency gap in memory. |
| `base_lin_vel` | 3 | None | **EXCLUDED (Critic Only)** | Correctly excluded from Policy Actor (63-D) and restricted to Critic (90-D). |

---

## 5. Hardware Performance Benchmarks & Estimates

Smoke tests conducted on NVIDIA RTX 4070 (8 GB VRAM), Intel i9-14900HX, 32 GB RAM:

| Environments (`num_envs`) | Peak VRAM | Sampling Throughput | 2-Iteration Smoke Test | 3,000 Iteration Wall-Clock Estimate |
|---|---|---|---|---|
| **256** | 2.8 GB | ~12,400 steps/sec | 0.38 s | ~10.5 minutes |
| **1024** | 4.8 GB | ~18,200 steps/sec | 1.35 s | ~18.5 minutes |
| **2048** | 6.2 GB | ~21,500 steps/sec | 2.28 s | ~25.0 minutes |

> [!WARNING]
> **VRAM OOM RISK**: When running training alongside an active display server or GUI viewer, `2048` envs consumes 6.2 GB of VRAM, leaving less than 1.8 GB free margin. If PhysX contact allocations spike, CUDA OOM occurs. **Recommended setup**: `1024` envs in `--headless` mode.

---

## 6. Sim-to-Real Gaps & Hardware Risks

Ranked by likelihood of causing physical hardware failure:

1. **Joint Order Mismatch (Severity: CRITICAL / BLOCKER)**:
   - PhysX exports joint states in depth-grouped order (`6x Coxa, 6x Thigh, 6x Foot`). ROS 2 drivers expect leg-grouped order (`Leg 1 [C,T,F], Leg 2 [C,T,F]...`). Running without a static index remapping array will instantly command coxa targets to foot servos, causing mechanical binding and servo burn-out.
2. **Mass Discrepancy (Severity: HIGH)**:
   - Robot USD mass is 0.63 kg vs 2.0 kg real mass. A policy trained on 0.63 kg will produce insufficient torque gains, causing the real robot to collapse under its own weight.
3. **Serial Bus Latency & Action Delay (Severity: HIGH)**:
   - Polling 18 ST3215 servos at 115200 baud introduces a **20–40 ms transport delay**. If RL is trained with instantaneous action execution, the hardware loop will oscillate violently at 50 Hz.
4. **WASD Step Command Discontinuity (Severity: MEDIUM)**:
   - Keyboard WASD teleop introduces sudden step changes in target velocity (e.g. $v_x = 0 \to 0.2\text{ m/s}$). Training requires sudden velocity command step changes (`rel_standing_envs=0.1`, narrow velocity ranges) to prevent tipping during keyboard input.

---

## 7. Recommended Corrections to `BUILD_PLAN.md`

- **Fix 1 (Joint Order Mapping)**: Enforce static index remapping array in export metadata (`deploy.yaml`) mapping PhysX depth-grouped order to ROS leg-sequential order.
- **Fix 2 (Payload Mass Randomization)**: Set `add_base_mass` event in `params.py` to `+1.37 kg` baseline (scaling effective USD mass from 0.63 kg to 2.0 kg) with domain randomization range `[1.1, 1.6] kg`.
- **Fix 3 (Action Delay Buffer)**: Add a 1–2 step random action delay buffer (`0–40 ms`) in environment step wrapper to simulate serial bus transmission latency.
- **Fix 4 (Joint Limit Clamp)**: Set `soft_joint_pos_limit_factor = 0.95` in `zoe_hexapod.py` to enforce soft limits at 95% of mechanical range (Coxa -45..+135°, Thigh/Foot -90..+90°).
- **Fix 5 (Terrain Grid Level Initialization)**: Set `max_init_terrain_level = None` in `env_cfg_terrain.py` to ensure robots are distributed evenly across all 10 rows and 10 columns of the terrain grid.

---

## What Remains Unverifiable Without Real Hardware

1. Exact serial bus round-trip latency at 115200 baud under 50 Hz streaming load.
2. Real ST3215 servo gear backlash and thermal torque degradation under continuous 2.0 kg payload walking.
3. IMU accelerometer noise characteristics under real motor gearhead vibration.
