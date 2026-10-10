# 01 — EXISTING STATE (what has already been made)

Snapshot: 2026-10-05. This file is **context, not a task**.
Do not rebuild, "improve", rename or delete anything described here unless a build prompt (02A / 02B) explicitly tells you to.
Anything marked **UNVERIFIED** must be checked by you before you rely on it.

---

## 1. The project in one paragraph

Z.O.E. is an 18-DOF hexapod (6 legs x coxa / thigh / foot) driven by a Jetson that the owner reaches over SSH.
Two codebases exist and they are **not connected**:

| | **A. Training repo** | **B. ROS 2 package `rover`** |
|---|---|---|
| Purpose | Train an RL locomotion policy in simulation | Drive the real robot with a hand-coded gait |
| Source | `github.com/Rithwik-Ravi/Z.O.E.-Hexapod-Isaac-Lab-Reinforced-Learning` (single commit `cc66599`) | `gz_hex/src/rover` (colcon, `ament_python`) |
| Stack | Isaac Sim + Isaac Lab, `rsl-rl` 2.x (README says RSL_RL_VERSION 2.3.1), Python 3.10 | ROS 2 Humble, Python 3.10 |
| Runs on | Owner's laptop: RTX 4070 (8 GB VRAM), Intel i9, 32 GB RAM, Ubuntu | Jetson (Orin Nano assumed, **UNVERIFIED**), commands issued via SSH |

**No trained policy has ever run on the real robot.** There is no export step, no runtime that can host a policy, and no servo feedback path.

Hardware (from README and code; **UNVERIFIED** until the owner confirms):
- 18 serial-bus servos, README says "ST3215" (Feetech STS3215 class; these report position, speed, load, voltage, temperature).
- ESP32 #1 on `/dev/ttyUSB0`: receives servo angles, drives the servo bus. **Firmware is not in either repo.**
- ESP32 #2 (CP2102, addressed by `/dev/serial/by-id/...`): streams IMU text lines. **Firmware is not in either repo.**
- 2D LiDAR is planned but is not read by any code today.

---

## 2. Codebase A — training repo (Isaac Lab)

### 2.1 What is in it

| Path | What it is | Status |
|---|---|---|
| `scripts/rsl_rl/train.py` | RSL-RL launcher. Contains NaN "sanitizer" monkey-patches (about L170-220) that replace NaN/Inf in obs/reward and clamp the policy std | Works. The patches mask instability instead of fixing it |
| `scripts/rsl_rl/play.py` | Loads latest checkpoint, runs in viewer, sanitizes NaN obs | Works. **Does not export anything** |
| `scripts/rsl_rl/cli_args.py`, `scripts/{zero,random}_agent.py`, `scripts/list_envs.py`, `scripts/skrl/*` | Isaac Lab template files | Untouched template |
| `source/spdrbot3/spdrbot3/assets/spdrbot.py` | `SPDRBOT_CFG` articulation config | Hard-coded to `/home/rithwik/SpdrBot-main/hexapod/hexa/hexa.usd`. **USD IS present locally in the repo at `hexapod/hexa/hexa.usd`** |
| `.../tasks/manager_based/spdrbot3_rough/env_cfg.py` + `__init__.py` | The only task actually used: **`Zoe-Hexapod-Rough-v0`** | Trains; see 2.2 and 2.3 |
| `.../tasks/direct/spdrbot3/` | Leftover template task `Template-Spdrbot3-Direct-v0`. **It also holds `agents/rsl_rl_ppo_cfg.py`, which the Zoe task imports** | Legacy; keep it, the Zoe task depends on it |
| `check_mdp.py`, `check_model.py`, `check_tb.py` | One-off debug scripts with hard-coded paths on the author's machine (`check_model.py` checks checkpoints for NaN weights, evidence of past instability) | Dead |
| `README.md` | Documentation | Contains incorrect claims, see A18 |

### 2.2 The MDP as implemented today (`env_cfg.py`)

| Item | Current value |
|---|---|
| Sim / policy rate | `dt=0.005` (200 Hz), `decimation=4` -> **50 Hz policy**, episode 20 s |
| Action | 18 joint-position targets, `scale=0.25`, offset = default joint pos (all joints **0.0**), joints matched by regex `.*Revolute.*` |
| Actuator | `ImplicitActuator`, stiffness 15, damping 1.5, effort limit 5.0 N·m, velocity limit 4.5 rad/s, same for all joints. Not measured from hardware |
| Observation (66-D, policy group only) | `base_lin_vel`(3), `base_ang_vel`(3), `projected_gravity`(3), `velocity_commands`(3), `joint_pos_rel`(18), `joint_vel_rel`(18), `last_action`(18). `enable_corruption=False` (no noise). No height scan |
| Terrain | `TerrainImporter` with generator containing **one** sub-terrain `flat`, `noise_range=(0,0)`. `use_cache=True`. 8x8 m tiles, `border_width=20` |
| Commands | `lin_vel_x/y` in [-1, 1] m/s, `ang_vel_z` in [-1, 1] rad/s, resample every 10 s, `heading_command=True` (`rel_heading_envs=1.0`), `rel_standing_envs=0.02` |
| Rewards (weight) | `track_lin_vel_xy_exp` +2.0, `track_ang_vel_z_exp` +0.5, **`is_alive` +10.0**, `dof_torques_l2` -1e-5, `dof_acc_l2` -1e-7, `action_rate_l2` -0.05, `lin_vel_z_l2`, `ang_vel_xy_l2`, `flat_orientation_l2` -0.5, `dof_pos_limits`, **`tripod_gait_penalty` -0.2**, `foot_clearance` **weight 0.0** (disabled) |
| Terminations | `time_out`, `base_contact` (body `base_link_v1`, threshold 1 N), `bad_orientation` (limit pi/2) |
| Events | `physics_material` (static 0.8 / dynamic 0.8 ranges, i.e. no variation), `add_base_mass` (0 to +0.5 kg), `reset_base` (xy +-0.5, **z +0.6 to +0.8**, yaw uniform), `reset_robot_joints` (scale 1.0 to 1.0, i.e. no variation). No pushes, no actuator/gain randomization, no latency |
| Contact sensor | on `{ENV}/Robot/.*`, history 3, `track_air_time=True` |
| PPO (`rsl_rl_ppo_cfg.py`) | actor/critic MLP 512-256-128 ELU, `init_noise_std=1.0`, `actor_obs_normalization=True`, `num_steps_per_env=48`, `max_iterations=5000`, `save_interval=50`, lr 1e-3 adaptive |
| Recommended `num_envs` | README suggests 1024-2048 |

### 2.3 Known defects in A (referenced by ID from 02A)

| ID | Defect | Where |
|---|---|---|
| A1 | Terrain is perfectly flat. No terrain variety, no curriculum (`CurrTerm` is imported but unused) | `env_cfg.py` terrain section |
| A2 | Actor observes `base_lin_vel`. The real robot has no sensor for it | ObservationsCfg |
| A3 | No observation noise | `enable_corruption=False` |
| A4 | Domain randomization is effectively off (degenerate friction range, tiny mass range, zero joint-reset variation) | EventCfg |
| A5 | No pushes, no actuator-gain randomization, no action/obs latency, no calibration-offset randomization | EventCfg |
| A6 | Actuator numbers are not measured. Effort limit 5 N·m is above what an STS3215 can deliver (roughly 1.9-2.9 N·m stall depending on 7.4 V / 12 V variant; check the owner's datasheet) | env_cfg / spdrbot.py |
| A7 | `tripod_gait_penalty` is `(feet_in_contact - 3)^2`. It does **not** enforce alternating tripods (README claims it does) and it punishes 4-6 feet down, which is the right behaviour on rough ground | rewards |
| A8 | `foot_clearance` uses world-frame z and is disabled. Also the `NFOOT*` body frame is the **knee** in the ROS URDF, not the foot tip (UNVERIFIED for the USD) | rewards |
| A9 | `is_alive` +10/step is about 80% of all reward; encourages freezing on hard terrain. No termination penalty | rewards |
| A10 | `soft_joint_pos_limit_factor=2` makes the joint-limit penalty effectively inert; torque penalty 1e-5 is negligible | spdrbot.py / rewards |
| A11 | Spawn: init `z=0.5` plus reset offset `z +0.6..0.8` is additive in Isaac Lab, so robots drop from about 1.1-1.3 m (real stance height is about 0.15 m). Probably related to the NaN problems | spdrbot.py + EventCfg |
| A12 | Command ranges (+-1 m/s) are far beyond the real gait (a few cm/s today). Commands change only every 10 s; the owner will change them abruptly from a keyboard | CommandsCfg |
| A13 | Default joint pose is 0 for all joints. The real robot's standing pose is coxa 45 deg, thigh about 8 deg, foot about 7 deg in the IK convention (see 3.3) | spdrbot.py |
| A14 | USD path hard-coded and USD missing; model scale, mass, joint limits, joint names/order are **unknown** | spdrbot.py |
| A15 | NaN/Inf sanitizing patches in train.py and play.py hide physics blow-ups | train.py, play.py |
| A16 | No policy export (ONNX/JIT/NumPy) and no deployment config | play.py |
| A17 | `init_state.rot=(0,0,0,1)` is a 180 deg yaw in Isaac Lab's (w,x,y,z) convention. Harmless because reset randomizes yaw | spdrbot.py |
| A18 | README errors: says 24-D observation (actual 66), says alternating tripod is enforced (it is not), lists files that are not in the repo, quotes 7,500+ iterations while `max_iterations=5000` | README.md |
| A19 | `flat_orientation_l2` and `heading_command` will fight slopes/stairs if reused unchanged on terrain | rewards / commands |
| A20 | `use_cache=True` on the terrain generator will silently reuse stale terrain after config edits (default cache dir `/tmp/isaaclab/terrains`) | terrain cfg |

---

## 3. Codebase B — ROS 2 package `rover` (`gz_hex/src/rover`)

Built with ROS 2 Humble, Python 3.10, in `/home/aanimesh/gz_hex`. The package began as a wheeled rover and was adapted for the hexapod, so it carries rover-era leftovers.

### 3.1 Nodes and their status

| Entry point | File | What it does | Status |
|---|---|---|---|
| `hxp_cmd` | `hxp_command.py` | **Current teleop path.** Keyboard via blocking `input()` (needs Enter); keys `f/b/l/r` only, **no stop command**. Plays a hard-coded 4-keyframe gait: foot targets in mm -> IK -> `JointTrajectory` on `/hexapod_controller/joint_trajectory` at **8 Hz** (`step_time=0.5 s`) | In use. Open-loop |
| `esp_bdg` | `esp.py` | **Current serial bridge.** Subscribes the trajectory, converts radians to servo degrees, writes one ASCII line `d1 d2 ... d18\n` to `/dev/ttyUSB0` at **115200 baud**. `time_from_start` is ignored (no interpolation) | In use. **One-way: reads nothing back** |
| `lidar_esp` | `lidar_esp.py` | Despite the name it is the **IMU reader** for ESP32 #2. Parses `ROLL:..,PITCH:..,YAW:..,AX:..,AY:..,AZ:..,VX:..,VY:..` (deg, accel in g). Publishes `/imu/data` (orientation + linear accel only, **no angular velocity**) and `/odom` (integrates VX/VY with fixed `dt=0.01`, drifts). Subscribes `/scan` but never uses it | In use |
| `twist` | `twist.py` | `/cmd_vel` -> same gait. Uses only the **sign** of velocity, ignores magnitude | Exists |
| `cal_servo` | `test_esp.py` | `/joint_states` -> servos (RViz-to-robot calibration helper) | Helper |
| `rviz_ik` | `rviz_hexa_ik.py` | RViz-only IK gait visualiser; publishes trajectory and `/joint_states`; gait z=120 | Helper |
| `imu_stabilizer` | `imu_stabalizer.py` | Old P-controller holding a fixed stance; publishes to the same joint topic as the gait | Legacy. **Conflicts** with other publishers |
| `hxp_ik`, `t_gait` | `hexapod_ik_controller.py`, `tripod_gait.py` | Entirely commented out | Dead |
| `teleop` | listed in `setup.py` | **Module `rover/teleop.py` does not exist** | Broken entry point |
| (not reviewed) | `arm_control.py`, `arm_ik_commander.py`, `box_tracker.py`, `object_detector.py` | Rover-era arm / vision nodes | Out of scope |
| Launch | `1_rviz`, `2_gazebo`, `3_gazebo` | RViz / Gazebo (ros2_control) simulation of `hexa.urdf` | Simulation only, not used for hardware |

### 3.2 Data flow today

```
keyboard (input(), Enter required)
  -> hxp_command.py   (8 Hz, open-loop 4-keyframe gait + IK)
  -> /hexapod_controller/joint_trajectory  (18 joints, radians)
  -> esp.py           ("d1 ... d18\n", 115200 baud, /dev/ttyUSB0)   ONE WAY
  -> ESP32 #1 -> servos

ESP32 #2 (IMU) -> "ROLL:..,PITCH:..,..\n" -> lidar_esp.py -> /imu/data (no gyro), /odom (drifting)
```

### 3.3 Numeric conventions you must preserve or consciously replace

- **Joint order** (18), used by every script: for leg i in 1..6: `NCCUP{i}_joint` (coxa), `NTHIGH{i}_joint` (thigh), `NFOOT{i}_joint` (foot/tibia).
- **Angle -> servo mapping** (`esp.py`): `servo_deg = q_deg + 45` for coxa; `servo_deg = q_deg + 90` for thigh and foot; for **leg 5 thigh and foot only** (joint indices 13, 14): `servo_deg = 180 - (q_deg + 90)`; then clamp to `[0, 180]`. Implied joint range: coxa -45..+135 deg, thigh/foot -90..+90 deg.
- **IK constants** (`hxp_command.py`): coxa 35 mm, thigh 75 mm, tibia 160 mm (older values in comments: 55 / 81 / 185). The URDF joint origins imply coxa about 58 mm and thigh 81 mm. **They disagree; measure the robot.**
- **Neutral stance**: foot target (77.8, 77.8, 150) mm gives q = (coxa 45.0 deg, thigh 7.7 deg, foot 7.4 deg), i.e. servo about (90, 97.7, 97.4). Swing lift uses z = 100 mm. The hand gait moves coxa roughly 21-69 deg and thigh/foot up to about 50 deg.
- **Gait timing**: 4 phases x 0.5 s = 2 s cycle, foot stroke about 50 mm, so roughly a few cm/s (estimate, not measured).
- **Leg numbering is inconsistent**: the per-leg gait tables differ between `hxp_command.py`, `twist.py` and `rviz_hexa_ik.py` (legs 2/6 and 3/5 are swapped between files). Do not infer leg-to-wiring mapping from any single script.
- **URDF (`hexa.urdf`)**: hexapod + rover leftovers (wheels/casters); coxa mounts at radius about 0.0966 m; every leg joint limited to +-1.5 rad with effort 30 / velocity 6 (not real servo limits); **link masses sum to about 58.5 kg** (CAD-export artifact, certainly not the real mass).

### 3.4 Known defects in B (referenced by ID from 02B)

| ID | Defect |
|---|---|
| B1 | No servo feedback path anywhere. "Joint state" is only ever the command |
| B2 | ASCII line at 115200 baud: about 108 bytes x 50 Hz already uses roughly half the link in one direction; no room for feedback |
| B3 | IMU stream has no gyro rates; odometry integration drifts; node name `lidar_esp` is misleading |
| B4 | Gait is open-loop, 8 Hz, blocking `input()`, no stop command, last command persists if the SSH session dies |
| B5 | Leg numbering and gait tables inconsistent across three scripts (see 3.3) |
| B6 | `imu_stabilizer` publishes to the same topic as the gait and will fight it |
| B7 | `teleop` entry point has no module |
| B8 | URDF masses, offsets and limits do not match reality or the IK constants |
| B9 | Offsets and the leg-5 inversion are hard-coded in `esp.py`; there is no calibration table |
| B10 | No safety layer: no watchdog, no stale-data checks, no tilt/temperature/voltage limits, no ramp to a safe pose |
| B11 | No `time_from_start` handling: servos jump to each target at their own max speed |

---

## 4. What does NOT exist (do not assume it does)

- A policy export (ONNX / TorchScript / NumPy) and any deployment config.
- Any ROS node that runs a learned policy, builds an observation vector, or reads servo encoders.
- Gyro angular velocity from the IMU.
- A joint calibration table (sim joint <-> ROS joint <-> servo ID, sign, offset, limits).
- A measured actuator model (delay, effective stiffness, speed limit) for the real servos.
- A key-by-key SSH teleop node publishing `/cmd_vel` with deadman behaviour.
- Terrain variety, domain randomization, observation noise, or a terrain curriculum in training.
- The `hexa.usd` file, the ESP32 firmware (both boards), and any sim-to-real validation tooling.

## 5. Unverified assumptions (verify, do not assume)

1. Sim model matches the real robot in scale, mass, link lengths, joint axes/signs/limits, and body names (`base_link_v1`, `NCCUP*`, `NTHIGH*`, `NFOOT*`).
2. Isaac Lab's joint order for this USD (likely grouped by joint depth, not by leg).
3. Servo model, supply voltage, and what the ESP32 #1 firmware does with the 0-180 deg values.
4. Which Jetson is in use (ROS 2 Humble on Ubuntu 22.04 implies Orin-class, not the original Nano).
5. The installed versions of Isaac Lab, Isaac Sim and `rsl-rl-lib` (API names in the build prompts come from memory; introspect the installed packages).

---

## 6. Added by this build plan

### 6.1 Newly Created Files & Tasks
- **New Task IDs**:
  - `Zoe-Hexapod-Flat-Deploy-v0` & `Zoe-Hexapod-Flat-Deploy-Play-v0` (Stage 1 Deployable Task)
  - `Zoe-Hexapod-Terrain-v0` & `Zoe-Hexapod-Terrain-Play-v0` (Stage 2 Multi-Terrain Task)
- **New Files Created**:
  - `source/spdrbot3/spdrbot3/assets/zoe_hexapod.py` (New articulation config with dynamic portable USD resolution)
  - `source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/__init__.py` (Task registrations)
  - `source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/params.py` (Dataclass single source of truth for all tunables)
  - `source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/terrains.py` (Flat + Multi-terrain generator configs)
  - `source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/env_cfg_flat.py` & `env_cfg_terrain.py` (Stage 1 & Stage 2 env configs)
  - `source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/mdp/*` (`observations.py`, `rewards.py`, `events.py`, `curriculum.py`, `actions.py`, `terminations.py`)
  - `source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/agents/rsl_rl_ppo_cfg.py` (PPO runner config)
  - `scripts/audit_asset.py` (M0 USD asset auditor script)
  - `scripts/fit_actuator_sim.py` (M1 Actuator step-response calibration script)
  - `scripts/rsl_rl/export.py` (M4 Policy exporter script)
  - `scripts/eval_robustness.py` (M5 Policy robustness evaluation script)
  - `tests/test_params.py`, `tests/test_curriculum.py`, `tests/test_export.py` (Unit tests)
  - `deploy/actuator_model.yaml` & `deploy/asset_report.json` (Contract metadata and asset report)
  - `scripts/diagnose_stance.py` & `scripts/diagnose_offline.py` (Stance diagnostic scripts)
  - `scripts/eval_honest_gate.py` (Quantitative stance gate across 3 conditions)
  - `scripts/eval_gait_report.py` (Single-env 10s gait analysis & tripod alternation verification)
  - `scripts/inspect_stance_geometry.py` (Link frame, foot tip polar coordinates, and symmetry inspector)
  - `docs/TRAINING_RECIPE.md` (Reproduction guide)
  - `docs/AUDIT_A_REPORT.md` (Read-only technical audit report)
  - `docs/DIAG_STANCE_REPORT.md` (Phase 1 stance diagnosis report)
  - `docs/STAGE1_READINESS_REPORT.md` (Stage 1 readiness, stance geometry, honest gate, and probe evaluation report)

---

## 7. Changelog

- **2026-10-06**: Verified ground truth codebase state against defects A1-A26. Updated `docs/EXISTING_STATE.md` with Section 6 listing all files, tasks, and contract specs built for the deployable pipeline. All 7 unit tests verified passing (`pytest`).
- **2026-10-10 (AM)**: Completed Phase 1 read-only stance diagnosis. Identified root cause of belly-dragging gait (H1 confirmed: IK angles applied as USD joint targets; H2 confirmed: 0.17m reset height; H3 refuted: 2.5 N.m servo limit provides >78% torque headroom). Created `scripts/diagnose_stance.py`, `scripts/diagnose_offline.py`, and `docs/DIAG_STANCE_REPORT.md`.
- **2026-10-10 (PM)**: Implemented Phase 2 stance fixes: calibrated nominal standing base height to 0.158m, restored USD zero default pose (0.0 rad), expanded illegal contact terminations to chassis and coxa cups, added thigh contact penalty, implemented shaped base height reward (exponential attractor + barrier below 0.148m), added 4 unweighted TensorBoard metrics, and updated `export.py` to refuse export if the checkpoint fails the stance gate (overrideable via `--force`). Ran 300-iteration probe (`logs/rsl_rl/zoe_hexapod_deploy/2026-10-10_19-01-52/model_299.pt`); achieved mean walking height 0.1539m, confirmed true alternating tripod gait (cross-correlation -0.9813), and documented full audit in `docs/STAGE1_READINESS_REPORT.md`.


