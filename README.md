# Z.O.E. Hexapod – Isaac Lab Reinforcement Learning

### Autonomous 18-DOF Hexapod Locomotion & Multi-Terrain Navigation in NVIDIA Isaac Sim

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Isaac Sim 4.5](https://img.shields.io/badge/Isaac--Sim-4.5-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/isaac-sim)
[![Isaac Lab](https://img.shields.io/badge/Isaac--Lab-Framework-76B900?logo=nvidia&logoColor=white)](https://isaac-sim.github.io/IsaacLab/)
[![RSL-RL PPO](https://img.shields.io/badge/RSL--RL-PPO-725cd1)](#reinforcement-learning--mdp-design)
[![Unit Tests](https://img.shields.io/badge/Tests-23%20Passed-brightgreen)](tests/)
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
  - [Stage 1 Flat Locomotion Gate Results](#stage-1-flat-locomotion-gate-results)
  - [Stage 2 Multi-Terrain Traversal Results](#stage-2-multi-terrain-traversal-results)
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
| **Adaptive Terrain Curriculum** | Procedural terrain generation across 7 terrain types (flat, rough, slopes, stairs, boxes) across 10 difficulty levels with origin-referenced displacement tracking. |
| **Automated Verification Suite** | Two standalone evaluation tools: `scripts/chunk_check.py` (3-condition parallel stability and push recovery) and `scripts/terrain_check.py` (multi-terrain traversal). |
| **Comprehensive Unit Tests** | 23 unit tests verifying actions, curriculum logic, height calculation, yaml deduplication, export integrity, and training hardening. |

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

#### Stage 2: Multi-Terrain Traversal (Warm-Started)
Warm-starts training from a validated Stage 1 checkpoint and progresses through the 10-level terrain curriculum:
```bash
python scripts/rsl_rl/train.py \
  --task Zoe-Hexapod-Terrain-v0 \
  --checkpoint logs/rsl_rl/zoe_hexapod_deploy/2026-11-05_19-51-02/model_599.pt \
  --num_envs 1024 \
  --max_iterations 1500 \
  --headless
```

> [!TIP]
> The training launcher includes an automatic early-abort watchdog (`--watchdog_iterations 150` by default). If training fails to achieve basic survival or reward progression within 150 iterations, it aborts cleanly with exit code 3 to save GPU compute.

---

### 3. Automated Gate & Traversal Verification

#### Stage 1 Physical Gate Evaluation (`scripts/chunk_check.py`)
Evaluates a checkpoint across 3 parallel conditions (Zero Command 5s, Forward Walk 0.15 m/s 10s, and Lateral Push Recovery with 0.3–0.5 m/s impulses):
```bash
python scripts/chunk_check.py \
  --task Zoe-Hexapod-Flat-Deploy-v0 \
  --checkpoint logs/rsl_rl/zoe_hexapod_terrain/2026-11-07_22-49-47/model_1198.pt \
  --headless
```

#### Stage 2 Multi-Terrain Traversal Evaluation (`scripts/terrain_check.py`)
Evaluates 455 parallel environments across all 7 terrain types and difficulty levels (0, 2, 4, 6, 9) under a 0.20 m/s traversal command:
```bash
python scripts/terrain_check.py \
  --checkpoint logs/rsl_rl/zoe_hexapod_terrain/2026-11-07_22-49-47/model_1198.pt \
  --headless
```

---

### 4. Unit Testing Suite

Run the 23 automated unit tests verifying the MDP terms, actions, export, and curriculum:
```bash
pytest -p no:launch_testing -p no:launch_testing_ros_pytest_entrypoint tests/
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
| **Total Mass** | 4.166 kg native CAD (without DCAM), 5.0 kg weighed robot | **5.00 kg** nominal mass model with $\pm 10\%$ domain randomization |
| **Kinematic Stance** | Coxa: $0.0^\circ$, Thigh: $0.0^\circ$, Foot: $-90.0^\circ$ ($-\pi/2$ rad) | Settled standing height $0.1427\text{ m}$; Gate minimum height $0.1284\text{ m}$ |
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
        OBS_CRI["Critic Obs (90-D, Privileged State)"]
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

- **Critic Privileged Observation ($90\text{-D}$)**:
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

| Reward Term | Weight | Purpose |
|---|---|---|
| `track_lin_vel_xy_exp` | $+2.0$ | Exponential tracking of commanded forward and lateral velocity ($\sigma = 0.13$ m/s). |
| `track_ang_vel_z_exp` | $+0.5$ | Exponential tracking of commanded yaw turning rate ($\sigma = 0.25$ rad/s). |
| `base_height_shaped` | $+1.0$ | Quadratic exponential attractor to target standing height ($0.1356$ m) with barrier penalty below gate height ($0.1284$ m). |
| `termination_penalty` | $-200.0$ | Severe penalty for roll/pitch flip ($>68.8^\circ$), chassis contact, or NaN state. |
| `torque_limit` | $-0.5$ | Soft quadratic penalty when joint torque exceeds $2.45$ N·m (98% of 2.50 N·m limit). |
| `fewer_than_3_feet_contact` | $-0.5$ | Penalizes having fewer than 3 feet on the ground to preserve static stability. |
| `fewer_than_4_feet_contact` | $-0.05$ | Regularization encouraging robust support polygon formation. |
| `feet_air_time` | $+0.25$ | Rewires swing-phase foot air time during locomotion to establish stepping cadence. |
| `foot_tip_clearance` | $+0.1$ | Rewards foot clearance up to $5.0$ cm during leg swing phase. |
| `foot_tip_slip` | $-0.1$ | Penalizes horizontal velocity of feet while in ground contact. |
| `stand_still_feet_grounded`| $+1.0$ | Rewards planting 5–6 feet firmly on the ground when velocity command is zero. |
| `stand_still_posture` | $-0.5$ | Penalizes joint deviations from default standing pose when commanded to stop. |
| `dof_torques_l2` | $-0.005$ | L2 torque minimization to minimize electrical heat dissipation. |
| `action_rate_l2` | $-0.05$ | First-difference action smoothing to prevent joint vibration. |

---

### Stage 2 Ground Height Sensing & Terrain Curriculum

#### Ground Height Computation
The unified `height_above_ground(env)` function calculates true terrain clearance directly beneath the chassis:
1. **Primary Sensor**: A 3x3 downward grid RayCaster (`height_scanner`) attached to `base_link` projecting onto `/World/ground`.
2. **Fallback**: If no raycaster is attached, it computes base world $z$ minus the average world $z$ of grounded foot tips (accounting for foot sphere radius $0.0085$ m).

#### Adaptive Terrain Curriculum
The terrain generator features 7 terrain columns across 10 progressive difficulty rows:
- `flat`: Flat reference plane.
- `rough`: Random uniform heightfield ($0.5$ cm to $4.0$ cm).
- `slope_up` / `slope_down`: Pyramid slopes ($5.7^\circ$ to $16.7^\circ$, slopes $0.10$ to $0.30$).
- `stairs_up` / `stairs_down`: Stepped pyramid stairs ($1.5$ cm to $4.5$ cm step rise).
- `boxes`: Discrete obstacle step platforms ($2.0$ cm to $6.0$ cm box heights).

**Progression Logic**:
- Displacement is calculated relative to each robot's tile origin `env_origins`.
- Robots traverse outward under consistent 20s episode commands (`ZoeTerrainCommandsCfg`).
- **Promotion**: If the robot survives without early termination and achieves $\ge 75\%$ of the commanded distance, its terrain level increases by $+1$.
- **Demotion**: If the robot achieves $< 35\%$ of the commanded distance, its terrain level decreases by $-1$.

---

## Empirical Verification & Gate Performance

All gate thresholds are strictly frozen and verified automatically across simulation checkpoints.

### Stage 1 Flat Locomotion Gate Results

Evaluated with `scripts/chunk_check.py` on the Stage 2 multi-terrain checkpoint (`model_1198.pt`):

| Condition | Evaluated Metric | Measured Value | Frozen Gate Threshold | Verdict |
|---|---|---|---|:---:|
| **1. Zero Command (5.0s)** | Base height $\ge 0.1284$ m | **100.00%** frames | $\ge 95\%$ frames | **PASS** |
| | $\ge 5$ feet in contact | **98.30%** frames | $\ge 95\%$ frames | **PASS** |
| | Non-foot chassis contact | **0.00%** frames | $< 1\%$ frames | **PASS** |
| | Torque saturation ($\|\tau\| \ge 2.45$ N·m) | **1.62%** joint-steps | $\le 2.0\%$ joint-steps | **PASS** |
| **2. Forward Walk 0.15 m/s (10.0s)** | Base height $\ge 0.1284$ m | **100.00%** frames | $\ge 90\%$ frames | **PASS** |
| | $\ge 3$ feet in contact | **99.43%** frames | $\ge 99\%$ frames | **PASS** |
| | Static stability margin ($p5$) | **$+0.0682$ m** (mean $+0.0940$ m) | $p5 \ge 0.0000$ m | **PASS** |
| | Forward speed tracking | **0.1515 m/s** (101% of cmd) | $\ge 0.1200$ m/s | **PASS** |
| | Non-foot chassis contact | **0.00%** frames | $< 1\%$ frames | **PASS** |
| **3. Push Recovery (8.0s)** | Lateral push survival (0.3–0.5 m/s) | **0 falls** | 0 falls | **PASS** |
| | Push 1 & 2 Height Recovery | **100.00%** frames | $\ge 90\%$ frames | **PASS** |
| | Non-foot chassis contact | **0.00%** frames | $< 1\%$ frames | **PASS** |

---

### Stage 2 Multi-Terrain Traversal Results

Evaluated with `scripts/terrain_check.py` across 455 environments spanning all 7 terrain types and difficulty levels (0, 2, 4, 6, 9) at 0.20 m/s:

| Terrain Type | Level 0–2 Survival | Target Level 0–4 Survival (Gate $\ge 90\%$) | All-Level Survival (L0–9) | Traversal Distance (m) | Non-Foot Contact | Gate Verdict |
|---|---:|---:|---:|---:|---:|:---:|
| `flat` | `100.0%` | **`100.0%`** | `100.0%` | `3.90 m` | `0.02%` | **PASS** |
| `rough` | `100.0%` | **`100.0%`** | `100.0%` | `3.46 m` | `0.02%` | **PASS** |
| `slope_up` | `100.0%` | **`100.0%`** | `100.0%` | `3.93 m` | `0.02%` | **PASS** |
| `slope_down` | `100.0%` | **`100.0%`** | `100.0%` | `3.46 m` | `0.02%` | **PASS** |
| `stairs_up` | `100.0%` | **`100.0%`** | `100.0%` | `3.58 m` | `0.02%` | **PASS** |
| `stairs_down` | `100.0%` | **`100.0%`** | `100.0%` | `3.12 m` | `0.01%` | **PASS** |
| `boxes` | `100.0%` | **`100.0%`** | `100.0%` | `3.27 m` | `0.01%` | **PASS** |

During 600 iterations of Stage 2 curriculum training, the average curriculum difficulty level climbed steadily from **`0.51` $\to$ `8.44` / 9.0**, with **`99.94%` timeout survival**.

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
│   └── mass_properties.yaml                   # Link-by-link mass & inertia specifications
├── docs/                                      # Engineering reports & specifications
│   ├── EXISTING_STATE.md                      # Complete system context & changelog
│   ├── ASSUMPTIONS.md                         # Authoritative hardware facts & engineering assumptions
│   ├── CHUNK_LOG.md                           # Verification logs for all chunk evaluations
│   ├── STAGE1_READINESS_REPORT.md             # Stage 1 probe evaluations & honest gate analysis
│   └── STAGE2_DIAG_REPORT.md                  # Stage 2 root-cause diagnosis, fixes & probe results
├── scripts/
│   ├── rsl_rl/
│   │   ├── train.py                           # Hardened PPO training launcher with watchdog
│   │   ├── play.py                            # GUI visualization viewer
│   │   ├── export.py                          # Policy exporter (TorchScript, ONNX, NumPy)
│   │   └── cli_args.py                        # CLI argument parser with cross-task resume
│   ├── chunk_check.py                         # 3-condition Stage 1 parallel gate verification suite
│   ├── terrain_check.py                       # 7-terrain Stage 2 traversal verification suite
│   └── analysis/                              # Verification & diagnostic analysis scripts
│       └── verify_a2_b1_spawn_and_zero_action.py
├── source/spdrbot3/spdrbot3/
│   ├── assets/
│   │   └── zoe_hexapod.py                     # Articulation & actuator configuration
│   └── tasks/manager_based/zoe_deploy/        # Production deployment pipeline tasks
│       ├── env_cfg_flat.py                    # Stage 1 Flat Deployable environment configuration
│       ├── env_cfg_terrain.py                 # Stage 2 Multi-Terrain environment configuration
│       ├── terrains.py                        # Procedural terrain generator definitions
│       ├── params.py                          # Single-source-of-truth tunable parameter registry
│       ├── agents/rsl_rl_ppo_cfg.py           # PPO runner hyperparameters
│       └── mdp/                               # Modular MDP definitions
│           ├── actions.py                     # Delayed rate-limited joint actions
│           ├── curriculum.py                  # Local-displacement terrain curriculum
│           ├── events.py                      # Domain randomization & ground reset logic
│           ├── observations.py                # 63-D actor & 90-D critic observations
│           ├── rewards.py                     # Height attractor, torque, and stability rewards
│           └── terminations.py                # Illegal contact, flip, and NaN terminations
└── tests/                                     # Automated test suite (23 tests passing)
    ├── test_actions.py                        # Unit tests for action rate-limiter & delay buffer
    ├── test_curriculum.py                     # Unit tests for terrain curriculum promotion/demotion
    ├── test_export.py                         # Unit tests for ONNX & TorchScript export equivalence
    ├── test_height_above_ground.py            # Unit tests for raycaster & foot-contact ground height
    ├── test_params.py                         # Unit tests for YAML deduplication & derived constants
    └── test_train_hardening.py                # Unit tests for checkpoint loader & watchdog logic
```

---

## Sim-to-Real Policy Export

To export a trained policy for deployment on the real robot's onboard controller:

```bash
python scripts/rsl_rl/export.py \
  --checkpoint logs/rsl_rl/zoe_hexapod_terrain/2026-11-07_22-49-47/model_1198.pt \
  --output_dir deploy/exported_policy/
```

The export script performs:
1. **Automated Stance Gate Verification**: Evaluates the policy on the standing gate.
2. **TorchScript & ONNX Serialization**: Exports clean inference models for Jetson deployment.
3. **NumPy Parameter Weights**: Exports raw layer weights and biases for embedded microcontrollers.
4. **Metadata & Deployment Contract**: Writes `deployment_contract.yaml` with input tensor scales, joint order, and safety bounds.

---

## License and Acknowledgments

- **License**: BSD-3-Clause License.
- **Frameworks**: Built using [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/) and [RSL-RL](https://github.com/leggedrobotics/rsl_rl).
- **Robot Platform**: Z.O.E. Hexapod with Waveshare ST3215 12V bus servos and 3S LiPo power distribution.
