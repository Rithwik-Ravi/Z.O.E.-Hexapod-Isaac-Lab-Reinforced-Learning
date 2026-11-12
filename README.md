# Z.O.E. Hexapod – Isaac Lab Reinforcement Learning

### Autonomous 18-DOF Hexapod Locomotion & Multi-Terrain Navigation in NVIDIA Isaac Sim

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Isaac Sim 4.5](https://img.shields.io/badge/Isaac--Sim-4.5-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/isaac-sim)
[![Isaac Lab](https://img.shields.io/badge/Isaac--Lab-Framework-76B900?logo=nvidia&logoColor=white)](https://isaac-sim.github.io/IsaacLab/)
[![RSL-RL PPO](https://img.shields.io/badge/RSL--RL-PPO-725cd1)](#reinforcement-learning--mdp-design)
[![Unit Tests](https://img.shields.io/badge/Tests-34%20Passed-brightgreen)](tests/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](#license-and-acknowledgments)
[![Gait: Alternating Tripod](https://img.shields.io/badge/Gait-Alternating%20Tripod-168f77)](#empirical-verification--gate-performance)

**Z.O.E.** (Zone Operational Entity) Hexapod is an end-to-end reinforcement learning (RL) framework built on **NVIDIA Isaac Lab** and **RSL-RL**. It trains an 18-degree-of-freedom (18-DOF) spider robot to master agile, deployable locomotion on flat ground and traverse complex procedurally generated terrains (pyramid slopes, rough heightfields, stairs, and obstacle boxes) using a high-stability **alternating tripod gait**.

Z.O.E. combines hardware-measured actuator dynamics (Waveshare ST3215 12V bus servos), realistic observation noise, delayed rate-limited actions, relative ground height sensing, and strict multi-condition physical stability gates to ensure policies trained in simulation are directly transferable to physical hardware.

---

> [!NOTE]
> **Active Production Tasks**:
> - **Stage 1 (Flat Deployable Locomotion)**: `Zoe-Hexapod-Flat-Deploy-v0` (Play: `Zoe-Hexapod-Flat-Deploy-Play-v0`)
> - **Stage 2 (Multi-Terrain Navigation & Curriculum)**: `Zoe-Hexapod-Terrain-v0` (Play: `Zoe-Hexapod-Terrain-Play-v0`)
>
> *(The legacy task `Zoe-Hexapod-Rough-v0` under `source/spdrbot3/spdrbot3/tasks/manager_based/spdrbot3_rough/` is deprecated).*

---

## Contents

- [Key Capabilities](#key-capabilities)
- [Quick Start](#quick-start)
  - [1. Playback Pre-Trained Policy in GUI](#1-playback-pre-trained-policy-in-gui)
  - [2. Headless GPU Training](#2-headless-gpu-training)
  - [3. Automated Gate & Traversal Verification](#3-automated-gate--traversal-verification)
  - [4. Unit Testing Suite](#4-unit-testing-suite)
- [Hardware & Actuator Specifications](#hardware--actuator-specifications)
- [System Architecture](#system-architecture)
- [Reinforcement Learning & MDP Design](#reinforcement-learning--mdp-design)
  - [Deployable Observation Spaces](#deployable-observation-spaces)
  - [Action Space & Actuation Dynamics](#action-space--actuation-dynamics)
  - [Reward Formulation](#reward-formulation)
  - [Stage 2 Ground Height Sensing & Terrain Curriculum](#stage-2-ground-height-sensing--terrain-curriculum)
- [Empirical Verification & Gate Performance](#empirical-verification--gate-performance)
  - [Stage 1 Flat Locomotion Gate Results (`2026-11-12`, `model_1499.pt`)](#stage-1-flat-locomotion-gate-results-2026-11-12-model_1499pt)
  - [Stage 2 Zero-Shot Calibration & Status (`NOT READY`)](#stage-2-zero-shot-calibration--status-not-ready)
- [Repository Structure](#repository-structure)
- [Sim-to-Real Policy Export](#sim-to-real-policy-export)
- [License and Acknowledgments](#license-and-acknowledgments)

---

## Key Capabilities

| Capability | Technical Implementation |
|---|---|
| **18-DOF Kinematics** | Validated USD asset (`hexapod/hexa/hexa.usd`) matching CAD mass properties (4.17 kg native CAD, 5.0 kg nominal model, 19 rigid bodies, 18 revolute joints). |
| **Hardware-Matched Actuation** | Explicit modeling of Waveshare ST3215 12V bus servos: 2.50 N·m effort limit (85% thermal margin from 2.94 N·m stall), 4.71 rad/s speed limit, $K_p = 16.0$ N·m/rad, $K_d = 1.6$ N·m·s/rad. |
| **Sim-to-Real Deployable Actor** | 63-D observation space strictly excluding unmeasurable base linear velocity, with realistic IMU gyro, gravity vector, and joint encoder noise. |
| **Latency & Slew-Rate Limiting** | `DelayedRateLimitedJointPositionActionCfg` modeling 0–2 control step communication delay (0–40 ms) and a 6.0°/step slew-rate clamp. |
| **Terrain Ground Height Sensing** | Unified `height_above_ground` using a 3x3 downward raycaster on `/World/ground` with dynamic foot-contact fallback. |
| **Adaptive Terrain Curriculum** | Procedural terrain generation across 7 terrain types (flat, rough, slopes, stairs, boxes) across 10 difficulty levels with world-frame net-displacement progress tracking (`ZoeTerrainTwoGroupCommand`). |
| **Automated Verification Suite** | Standalone evaluation tools: `scripts/chunk_check.py` (16-condition Stage 1 parallel gate incl. Check #16 leg-duty & tripod balance, pure yaw, strafe, backward, push recovery), `scripts/eval_gait_report.py`, and `scripts/terrain_check.py` (7-terrain Stage 2 traversal + skill retention). |
| **Comprehensive Unit Tests** | 34 unit tests across 7 files verifying actions, command groups, `is_cmd_stopped` gating, curriculum logic, height calculation, YAML deduplication, export integrity, and training hardening. |

---

## Quick Start

### Prerequisites

- **OS**: Ubuntu 20.04 / 22.04 LTS (Linux x86_64)
- **GPU**: NVIDIA RTX Series (8GB+ VRAM recommended, e.g. RTX 4070 Laptop / RTX 3080+)
- **NVIDIA Driver**: 525+ with CUDA 12.x
- **Environment**: Conda environment with NVIDIA Isaac Sim 4.5 and Isaac Lab installed

```bash
# Enable GPU persistence mode for maximum PhysX throughput
sudo nvidia-smi -pm 1

# Activate Isaac Lab environment
conda activate env_isaaclab

# Install local package in editable mode
cd ~/SpdrBot-main
pip install -e source/spdrbot3
```

---

### 1. Playback Pre-Trained Policy in GUI

To visualize a trained policy navigating in the Isaac Sim interactive viewer:

```bash
# Stage 1 Flat Deployable Walking (16 parallel robots)
python scripts/rsl_rl/play.py --task Zoe-Hexapod-Flat-Deploy-Play-v0 --num_envs 16

# Stage 2 Multi-Terrain Navigation (16 parallel robots on terrain tiles)
python scripts/rsl_rl/play.py --task Zoe-Hexapod-Terrain-Play-v0 --num_envs 16
```

---

### 2. Headless GPU Training

#### Stage 1: Flat Ground Locomotion
Trains the base walking policy from scratch (1024 parallel environments, 50 Hz control rate, 20s episodes):
```bash
python scripts/rsl_rl/train.py \
  --task Zoe-Hexapod-Flat-Deploy-v0 \
  --num_envs 1024 \
  --max_iterations 1500 \
  --headless
```

#### Stage 2: Multi-Terrain Traversal (Warm-Started — Not Yet Started)
Warm-starts training from the validated Stage 1 Chunk 1 checkpoint (`2026-11-12_05-47-14/model_1499.pt`) and progresses through the 10-level terrain curriculum:
```bash
python scripts/rsl_rl/train.py \
  --task Zoe-Hexapod-Terrain-v0 \
  --checkpoint logs/rsl_rl/zoe_hexapod_deploy/2026-11-12_05-47-14/model_1499.pt \
  --num_envs 1024 \
  --max_iterations 1500 \
  --headless
```

> [!TIP]
> The training launcher includes an automatic early-abort watchdog (`--watchdog_iterations 150` by default). If training fails to achieve basic survival or reward progression within 150 iterations, it aborts cleanly with exit code 3 to save GPU compute.

---

### 3. Automated Gate & Traversal Verification

#### Stage 1 Physical Gate Evaluation (`scripts/chunk_check.py`)
Evaluates a checkpoint against the frozen **16-condition** Stage 1 gate (Zero Command 5s, Forward Walk 0.15 m/s 10s including Check #16 leg duty $\ge 40\%$ & tripod balance $\le 15\text{pp}$, Lateral Push Recovery 8s, Pure Yaw 0.40 rad/s, Strafe 0.12 m/s, Backward Walk -0.12 m/s):
```bash
python scripts/chunk_check.py \
  --task Zoe-Hexapod-Flat-Deploy-v0 \
  --checkpoint logs/rsl_rl/zoe_hexapod_deploy/2026-11-12_05-47-14/model_1499.pt \
  --headless
```

#### Stage 2 Multi-Terrain Traversal Evaluation (`scripts/terrain_check.py`)
Evaluates 455 parallel environments across all 7 terrain types and difficulty levels (0, 2, 4, 6, 9) under a 0.20 m/s traversal command (world-frame net-displacement progress) plus a 5-skill retention suite:
```bash
python scripts/terrain_check.py \
  --checkpoint logs/rsl_rl/zoe_hexapod_deploy/2026-11-12_05-47-14/model_1499.pt \
  --headless
```

---

### 4. Unit Testing Suite

Run the 34 automated unit tests verifying the MDP terms, actions, commands, gating, export, and curriculum:
```bash
PYTHONPATH="" pytest -p no:launch_testing tests/
```

---

## Hardware & Actuator Specifications

Authoritative hardware parameters aligned between physical hardware, URDF/USD assets, and simulation parameters:

| Component | Hardware Specification | Simulation Representation |
|---|---|---|
| **Actuators** | 18x Waveshare ST3215 12V Serial Bus Servos | `ImplicitActuator` ($K_p = 16.0$ N·m/rad, $K_d = 1.6$ N·m·s/rad) |
| **Effort Limit** | 2.942 N·m stall torque @ 12V (30 kg·cm) | **2.50 N·m** (85% thermal safety margin; saturation threshold 2.45 N·m) |
| **Speed Limit** | 0.222 s / 60° = 45 RPM @ 12V | **4.71 rad/s** |
| **Power Supply** | 3S LiPo Battery (11.4 V – 12.0 V, 12 A – 18 A total) | Multiplicative effort limit domain randomization ($\pm 15\%$) |
| **Total Mass** | 4.166 kg native CAD (without DCAM), 5.0 kg assumed payload model | **5.00 kg** nominal mass model (`+0.8338 kg` on `base_link`, adjustable via `scripts/set_payload_mass.py`) with `[0.80, 1.10]` domain randomization (`4.00–5.50 kg`) |
| **Kinematic Stance** | Coxa: $0.0^\circ$, Thigh: $0.0^\circ$, Foot: $-90.0^\circ$ ($-\pi/2$ rad) | Settled standing height **`0.1417 m`** (`5.00 kg`); Target **`0.1346 m`**; Gate minimum **`0.1275 m`** |
| **Control Rate** | 50 Hz policy frequency ($\text{dt} = 0.005\text{s}$, $\text{decimation} = 4$) | Delay: 0–2 steps (0–40 ms); Slew rate limit: 6.0°/step (5.24 rad/s) |
| **Servo Bus** | 1 Mbaud serial bus (`/dev/ttyUSB0`) | Joint ID addressing: $\text{ID} = 3(\text{leg} - 1) + k$ ($k \in \{1, 2, 3\}$) |

---

## System Architecture

```mermaid
flowchart TD
    subgraph Sim[NVIDIA Isaac Sim / Isaac Lab 4.5]
        ROBOT["18-DOF Hexapod USD (hexapod/hexa/hexa.usd)"]
        PHYSX["PhysX GPU Dynamics (dt=0.005s)"]
        TERRAINS["Procedural Mesh Generator (7 Terrains x 10 Levels)"]
        RAY["3x3 Downward RayCaster (height_scanner)"]
        CONTACTS["Contact Sensor (Body & Foot Net Forces)"]
    end

    subgraph MDP[Markov Decision Process (50 Hz, Decimation 4)]
        OBS_ACT["Actor Obs (63-D, No LinVel, Corrupted)"]
        OBS_CRI["Critic Obs (92-D, Privileged State)"]
        REW["Shaped Height + Stability + Torque Penalties"]
        CURR["Origin-Relative Distance Curriculum"]
    end

    subgraph Agent[RSL-RL PPO Architecture]
        ACTOR["Actor MLP (512-256-128)"]
        CRITIC["Critic MLP (512-256-128)"]
        ACTION_FILTER["Delayed Rate-Limiter (0-40ms, 6 deg/step)"]
    end

    ROBOT --> PHYSX
    TERRAINS --> PHYSX
    PHYSX --> CONTACTS & RAY
    CONTACTS & RAY --> OBS_ACT & OBS_CRI & REW
    OBS_ACT --> ACTOR
    OBS_CRI --> CRITIC
    ACTOR --> ACTION_FILTER
    ACTION_FILTER -->|Joint Position Targets| ROBOT
    REW --> Agent
    CURR -->|Terrain Level Promotion/Demotion| TERRAINS
```

---

## Reinforcement Learning & MDP Design

### Deployable Observation Spaces

To guarantee direct sim-to-real transferability, the Actor network does not observe base linear velocity $\mathbf{v}_{xy}$, which cannot be measured directly on the real robot without drifting state estimation:

- **Actor Policy Observation ($63\text{-D}$, deployable with noise)**:
  - Base angular velocity $\boldsymbol{\omega}_{xyz}$ (3) with Gaussian gyro noise ($\sigma = 0.05$ rad/s)
  - Projected gravity vector $\mathbf{g}_{\text{proj}}$ (3) with accelerometer noise ($\sigma = 0.02$)
  - Velocity commands $(v_{x,\text{cmd}}, v_{y,\text{cmd}}, \omega_{z,\text{cmd}})$ (3)
  - Calibrated joint position offsets $(q - q_{\text{default}})$ (18) with encoder noise ($\sigma = 0.01$ rad)
  - Relative joint velocities $\dot{q}$ (18) with velocity noise ($\sigma = 0.5$ rad/s)
  - Last executed actions $a_{t-1}$ (18)

- **Critic Privileged Observation ($92\text{-D}$)**:
  - Includes all 63 Actor terms (uncorrupted)
  - Base linear velocity $\mathbf{v}_{xyz}$ (3)
  - Binary foot contact flags $\mathbf{c}_{\text{feet}}$ (6)
  - Applied joint torques $\boldsymbol{\tau}$ (18)
  - Base height above ground $h_{\text{ground}}$ (1)
  - Belly/thigh ground contact flag $c_{\text{belly}}$ (1)

---

### Action Space & Actuation Dynamics

- **Target Scaling**: Actions $a_t \in [-1, 1]^{18}$ are scaled by $0.50$ rad ($28.6^\circ$ kinematic range):
  $$q_{\text{target}} = q_{\text{default}} + 0.50 \cdot a_t$$
- **Slew-Rate Clamping**: Position target differences between consecutive policy steps are clamped to $6.0^\circ$ ($0.1047$ rad/step = $5.236$ rad/s) to prevent servo over-current.
- **Latency Modeling**: Actions are delayed by a randomized FIFO buffer of 0–2 steps (0–40 ms).

---

### Reward Formulation

*(All standing vs. moving reward gates use `is_cmd_stopped`: `|cmd_xy| < 0.05 AND |cmd_yaw| < 0.05`, ensuring pure-yaw and small diagonal commands receive full swing/clearance rewards.)*

| Reward Term | Weight | Purpose |
|---|---|---|
| `track_lin_vel_xy_exp` | $+2.0$ | Exponential tracking of commanded forward and lateral velocity ($\sigma = 0.13$ m/s). |
| `track_ang_vel_z_exp` | $+1.0$ | Exponential tracking of commanded yaw turning rate ($\sigma = 0.25$ rad/s). |
| `base_height_shaped` | $+1.0$ | Quadratic exponential attractor to target standing height ($0.1346$ m) with barrier penalty below gate height ($0.1275$ m). |
| `termination_penalty` | $-200.0$ | Severe penalty for roll/pitch flip ($>68.8^\circ$), chassis contact, or NaN state. |
| `torque_limit` | $-0.5$ | Soft quadratic penalty when joint torque exceeds $2.10$ N·m (84% of 2.50 N·m limit). |
| `fewer_than_3_feet_contact` | $-0.5$ | Penalizes having fewer than 3 feet on the ground to preserve tripod support without blocking swing handover. |
| `fewer_than_4_feet_contact` | $-0.05$ | Mild regularization encouraging robust support polygon formation. |
| `feet_air_time` | $+0.25$ | Rewards swing-phase foot air time (`~is_cmd_stopped`) to establish alternating stepping cadence. |
| `foot_tip_clearance` | $+0.1$ | Rewards foot clearance (`~is_cmd_stopped`) up to $5.0$ cm during leg swing phase. |
| `foot_tip_slip` | $-0.1$ | Penalizes horizontal velocity of feet while in ground contact. |
| `stand_still_feet_grounded`| $+1.0$ | Rewards planting 5–6 feet firmly on the ground when stopped (`is_cmd_stopped`). |
| `stand_still_posture` | $-0.5$ | Penalizes joint deviations from default standing pose when stopped (`is_cmd_stopped`). |
| `dof_torques_l2` | $-0.005$ | L2 torque minimization to reduce electrical heat dissipation. |
| `action_rate_l2` | $-0.05$ | First-difference action smoothing to prevent joint vibration. |

---

### Stage 2 Ground Height Sensing & Terrain Curriculum

#### Ground Height Computation
The unified `height_above_ground(env)` function calculates true terrain clearance directly beneath the chassis:
1. **Primary Sensor**: A 3x3 downward grid RayCaster (`height_scanner`) attached to `base_link` projecting onto `/World/ground`.
2. **Fallback**: If no raycaster is attached, it computes base world $z$ minus the average world $z$ of grounded foot tips (accounting for foot sphere radius $0.0085$ m).

#### Adaptive Terrain Curriculum
The terrain generator (`source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/terrains.py`) features 7 terrain columns across 10 progressive difficulty rows (deployment targets at L0–3/4):
- `flat`: Flat reference plane.
- `rough`: Random uniform heightfield ($0.5$ cm to $4.5$ cm; deployment target $\le 2.5$ cm, L0–4).
- `slope_up` / `slope_down`: Pyramid slopes ($0.0$ to $0.40$ grade; deployment target $\le 0.21$ rad / $12^\circ$, L0–4).
- `stairs_up` / `stairs_down`: Stepped pyramid stairs ($1.0$ cm to $9.0$ cm step rise; deployment target $\le 4.0$ cm, L0–3).
- `boxes`: Discrete obstacle step platforms ($1.0$ cm to $7.0$ cm box heights; deployment target $\le 4.0$ cm, L0–4).

**Progression & Two-Group Command Logic**:
- `ZoeTerrainTwoGroupCommandCfg` splits environments 50/50 between traversal (`Group T`, 20s held episodes across levels 0–9) and skill retention (`Group S`, multi-directional + turning/standing commands capped at level 3).
- Evaluation progress uses **world-frame net displacement** $(p_{\text{end},w} - p_{\text{start},w})_{xy} \cdot \hat{cmd}_w$ with pre-reset endpoint capture (`scripts/terrain_check.py`).
- **Promotion / Demotion**: Survive without early termination and achieve $\ge 75\%$ commanded travel ($\ge 0.5$ m) $\to +1$; early termination or $< 35\%$ $\to -1$.

---

## Empirical Verification & Gate Performance

> [!NOTE]
> **Current Status (`2026-11-12`)**:
> - **Stage 1 (`Zoe-Hexapod-Flat-Deploy-v0`)**: **`PASS` (`16/16`)** — Full `1500`-iter Chunk 1 (`logs/rsl_rl/zoe_hexapod_deploy/2026-11-12_05-47-14/model_1499.pt`) passes all 16 frozen checks cleanly on the `5.0000 kg` payload model with `[0.80, 1.10]` mass randomization.
> - **Stage 2 (`Zoe-Hexapod-Terrain-v0`)**: **`NOT READY`** — Earlier `GO` (`model_2098.pt`) was retracted (`8.7%–32.2%` world-frame net-displacement progress, `100% FAIL`), and Stage 2 training warm-started from `2026-11-12_05-47-14/model_1499.pt` has not yet started.
> - **Physical Hardware**: **`NO`** — Policy is cleared for Stage 2 simulation training and software-pipeline verification only (`deploy/test_export_not_for_hardware/deploy.yaml` is flagged `NOT_FOR_HARDWARE_UNCALIBRATED`). See [`docs/STAGE1_READINESS_REPORT.md`](docs/STAGE1_READINESS_REPORT.md) and [`docs/STAGE2_DIAG_REPORT.md`](docs/STAGE2_DIAG_REPORT.md).

All gate thresholds are strictly frozen and verified automatically across simulation checkpoints (`scripts/chunk_check.py`).

### Stage 1 Flat Locomotion Gate Results (`2026-11-12`, `model_1499.pt`)

Evaluated with `scripts/chunk_check.py` on the final Stage 1 Chunk 1 checkpoint (`2026-11-12_05-47-14/model_1499.pt`, `5.0000 kg` model):

| Condition | Evaluated Metric | Measured Value | Frozen Gate Threshold | Verdict |
|---|---|---|---|:---:|
| **1. Zero Command (5.0s)** | Base height $\ge 0.1275$ m | **100.00%** frames | $\ge 95\%$ frames | **PASS** |
| | $\ge 5$ feet in contact | **98.36%** frames | $\ge 95\%$ frames | **PASS** |
| | Non-foot chassis contact | **0.00%** frames | $< 1\%$ frames | **PASS** |
| **2. Forward Walk 0.15 m/s (10.0s)** | Base height $\ge 0.1275$ m | **100.00%** frames | $\ge 90\%$ frames | **PASS** |
| | $\ge 3$ feet in contact | **99.42%** frames | $\ge 99\%$ frames | **PASS** |
| | Non-foot chassis contact | **0.00%** frames | $< 1\%$ frames | **PASS** |
| | Static stability margin ($p5$) | **$+0.0571$ m** (mean $+0.0864$ m) | $p5 \ge 0.0000$ m | **PASS** |
| | Forward speed tracking | **0.1544 m/s** (`102.9%`) | $\ge 0.1200$ m/s (`80%`) | **PASS** |
| | Check #16: Leg duty & tripod diff | **min `54.7%` [L1], diff `3.1pp`** (`A:63.3%, B:66.4%`) | `min >= 40% & diff <= 15pp` | **PASS** |
| **3. Push Recovery (8.0s)** | Lateral push survival (`0.3–0.5 m/s`) | **0 falls** (`0.00%` non-foot) | `0 falls`, `< 1%` non-foot | **PASS** |
| | Push 1 & 2 Height Recovery | **100.00% / 100.00%** frames | $\ge 90\%$ frames | **PASS** |
| **4. Pure Yaw 0.40 rad/s (5.0s)** | Yaw rate tracking | **+0.4172 rad/s** (`104.3%`) | $\ge 0.2000$ rad/s (`50%`) | **PASS** |
| **5. Strafe 0.12 m/s (3.0s)** | Lateral velocity tracking | **+0.1077 m/s** (`89.8%`) | $\ge 0.0600$ m/s (`50%`) | **PASS** |
| **6. Backward Walk -0.12 m/s (3.0s)** | Backward velocity tracking | **-0.1033 m/s** (`86.1%`) | $\le -0.0600$ m/s (`50%`) | **PASS** |

*Additional Verified Diagnostics (`model_1499.pt`)*:
- **Single-Env Gait Report (`scripts/eval_gait_report.py`)**: Tripod A `52.5%` vs Tripod B `63.7%`, cross-group correlation **`-0.8422`** (alternating tripod confirmed); step tracking `vx +0.2298 m/s` (`cmd 0.20`), `wz +0.6487 rad/s` (`cmd 0.50`), `vy +0.1493 m/s` (`cmd 0.15`).
- **Report-Only Torque & Support Breakdown (`C2`)**: `TorqSat (>= 2.45 N·m)` = `4.14%` (Zero Cmd) / `13.98%` (Forward) / `13.74%` (Push); `> 80%` effort (`> 2.00 N·m`) = `1.01%` Coxa / `34.98%` Thigh / `21.74%` Foot (`19.24%` joint-steps overall); mean feet down `3.89 / 6`; limit proximity (`<= 3 deg`) `0.00%`.

---

### Stage 2 Zero-Shot Calibration & Status (`NOT READY`)

Zero-shot multi-terrain calibration (`scripts/terrain_check.py`, `65` envs/terrain, `20.0s`) on **Probe A** (`2026-11-11_23-36-34/model_599.pt`) under the world-frame net-displacement metric (`NEW`) vs. retracted Stage 2 `model_2098.pt`:

| Terrain Type | Deployment Target Survival | `Probe A` Zero-Shot Prog (`NEW / OLD`) | `Probe A` Gate Target | Retracted `model_2098.pt` (`NEW`) |
|---|---:|---:|:---:|---:|
| `flat` | `100.0%` | **`90.6%`** (`90.0%`) | **PASS** | `32.2%` |
| `rough` | `100.0%` | **`38.9%`** (`46.1%`, `23.1%` stuck) | **FAIL** | `8.7%` |
| `slope_up` | `100.0%` | **`96.2%`** (`91.2%`) | **PASS** | `31.6%` |
| `slope_down` | `100.0%` | **`72.0%`** (`77.9%`) | **PASS** | `28.2%` |
| `stairs_up` | `100.0%` | **`80.1%`** (`78.2%`) | **PASS** | `26.1%` |
| `stairs_down` | `100.0%` | **`42.1%`** (`53.1%`, `46.2%` stuck) | **FAIL** | `17.0%` |
| `boxes` | `100.0%` | **`57.0%`** (`61.0%`) | **FAIL** | `25.4%` |

---

## Repository Structure

```text
SpdrBot-main/
├── README.md                                  # Production documentation & usage guide
├── hexapod/                                   # Robot 3D assets & URDF/USD models
│   ├── hexa.urdf                              # Audited 18-DOF robot URDF (DCAM commented out)
│   └── hexa/
│       └── hexa.usd                           # Optimized PhysX USD articulation asset
├── deploy/                                    # Deployment metadata & hardware contracts
│   ├── actuator_model.yaml                    # ST3215 actuator parameters & limits
│   ├── asset_report.json                      # Machine-audited robot link & joint metadata
│   ├── mass_properties.yaml                   # Link-by-link mass & inertia specifications
│   └── test_export_not_for_hardware/
│       └── deploy.yaml                        # Software-test export metadata (Stage 1 model_1499)
├── docs/                                      # Engineering reports & specifications
│   ├── EXISTING_STATE.md                      # Complete system context & changelog
│   ├── ASSUMPTIONS.md                         # Authoritative hardware facts & engineering assumptions
│   ├── CHUNK_LOG.md                           # Verification logs for all chunk and terrain checks
│   ├── STAGE1_READINESS_REPORT.md             # Stage 1 probe, Chunk 1 (16/16), torque & export report
│   └── STAGE2_DIAG_REPORT.md                  # Stage 2 root-cause diagnosis, fixes & metric retraction
├── scripts/
│   ├── rsl_rl/
│   │   ├── train.py                           # Hardened PPO training launcher with watchdog
│   │   ├── play.py                            # GUI visualization viewer
│   │   ├── export.py                          # Policy exporter (ONNX, NumPy NPZ, deploy.yaml, rollout)
│   │   └── cli_args.py                        # CLI argument parser with cross-task resume
│   ├── chunk_check.py                         # 16-condition Stage 1 parallel gate verification suite
│   ├── terrain_check.py                       # 7-terrain Stage 2 traversal & skill retention suite
│   ├── eval_gait_report.py                    # Single-env 10s gait & tripod alternation evaluator
│   ├── set_payload_mass.py                    # One-command total-mass recalibration CLI
│   └── analysis/                              # Verification & diagnostic analysis scripts
│       ├── diagnose_static_torques.py         # Simulator static holding torque audit (5 kg & 4 kg)
│       ├── eval_reward_economics.py           # Per-second standing vs walking reward audit
│       └── verify_a2_b1_spawn_and_zero_action.py
├── source/spdrbot3/spdrbot3/
│   ├── assets/
│   │   └── zoe_hexapod.py                     # Articulation & actuator configuration
│   └── tasks/manager_based/zoe_deploy/        # Production deployment pipeline tasks
│       ├── env_cfg_flat.py                    # Stage 1 Flat Deployable environment configuration
│       ├── env_cfg_terrain.py                 # Stage 2 Multi-Terrain environment configuration
│       ├── terrains.py                        # Procedural terrain generator definitions
│       ├── commands.py                        # ZoeUniformVelocity & TwoGroup command generators
│       ├── params.py                          # Single-source-of-truth tunable parameter registry
│       ├── agents/rsl_rl_ppo_cfg.py           # PPO runner hyperparameters
│       └── mdp/                               # Modular MDP definitions
│           ├── actions.py                     # Delayed rate-limited joint actions
│           ├── curriculum.py                  # Net-displacement terrain curriculum
│           ├── events.py                      # Domain randomization & ground reset logic
│           ├── observations.py                # 63-D actor & 92-D critic observations
│           ├── rewards.py                     # Height attractor, gating, torque & stability rewards
│           └── terminations.py                # Illegal contact, flip, and NaN terminations
└── tests/                                     # Automated test suite (34 tests passing)
    ├── test_actions.py                        # Unit tests for action rate-limiter & delay buffer
    ├── test_commands.py                       # Unit tests for command distributions & group splits
    ├── test_curriculum.py                     # Unit tests for terrain curriculum promotion/demotion
    ├── test_export.py                         # Unit tests for ONNX & NumPy MLP export equivalence
    ├── test_height_above_ground.py            # Unit tests for raycaster, foot height & command gating
    ├── test_params.py                         # Unit tests for YAML deduplication & derived constants
    └── test_train_hardening.py                # Unit tests for checkpoint loader & watchdog logic
```

---

## Sim-to-Real Policy Export

To export a trained policy package for software pipeline verification (hardware deployment remains **blocked** until physical calibration, IMU, and servo feedback exist):

```bash
python scripts/rsl_rl/export.py \
  --task Zoe-Hexapod-Flat-Deploy-v0 \
  --checkpoint logs/rsl_rl/zoe_hexapod_deploy/2026-11-12_05-47-14/model_1499.pt \
  --out deploy/test_export_not_for_hardware \
  --force
```

The export script performs:
1. **Automated Stance Gate Verification**: Evaluates mean height ($\ge 0.1275$ m) and non-foot contacts ($< 1.0\%$) before serializing (`model_1499.pt` passed at `0.1361 m`, `0.00%`).
2. **ONNX & NumPy MLP Serialization**: Writes `policy.onnx` and dependency-free `policy_mlp.npz` (with embedded observation normalizer stats).
3. **Reference Simulation Rollout**: Generates `reference_rollout.npz` (500 noise-free steps verified matching ONNX and NumPy inference to $< 1.2 \times 10^{-6}$).
4. **Metadata & Deployment Contract**: Writes `deploy.yaml` with `obs_spec`, `joint_order`, limits, and `status: NOT_FOR_HARDWARE_UNCALIBRATED`.

---

## License and Acknowledgments

- **License**: BSD-3-Clause License.
- **Frameworks**: Built using [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/) and [RSL-RL](https://github.com/leggedrobotics/rsl_rl).
- **Robot Platform**: Z.O.E. Hexapod with Waveshare ST3215 12V bus servos and 3S LiPo power distribution.
