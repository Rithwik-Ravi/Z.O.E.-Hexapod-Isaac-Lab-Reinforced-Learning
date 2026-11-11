# Stage 1 & Deployment Readiness Report

- **Date**: 2026-11-11
- **Repository Baseline**: Branch `main`, code state `9662a60` / evaluation log `744ea37` (plus documentation & `deploy/mass_properties.yaml` updates)
- **Active Environment / Robot Asset**: `Zoe-Hexapod-Flat-Deploy-v0` (`5.0000 kg` payload model in `hexapod/hexa/configuration/hexa_physics.usd`, settled standing height `0.1417 m`, target `0.1346 m`, minimum `0.1275 m`)

---

## Executive Verdicts

| Scope | Verdict | Basis (`VERIFIED`) |
|---|---|---|
| **Stage 1 (`Zoe-Hexapod-Flat-Deploy-v0`) Frozen 15-Condition Gate (`scripts/chunk_check.py`, `52685d4`)** | **`FAIL` (`NOT READY`)** | Best of 4 GPU probes (`1024` envs, `600` iters, from scratch): **Probe 4** (`2026-11-11_22-30-05/model_599.pt`) scored **`13/15 PASS`**, failing `Forward` (`0.0079 m/s` vs `>= 0.1200 m/s`, `80%` of `0.15 m/s`) and `Strafe` (`+0.0420 m/s = 35.0%` vs `>= 0.0600 m/s`, `50%` of `0.12 m/s`). Per the strict GPU budget (`<= 4` probes; 1500-iter Chunk 1 only if a probe passes), no 5th probe and no Chunk 1 run were started. |
| **Stage 2 (`Zoe-Hexapod-Terrain-v0`)** | **`NOT READY` (Previous `GO` Retracted)** | The earlier Stage 2 `GO` verdict rested on a Stage 2 probe (`2026-11-10_18-11-19/model_2098.pt`) warm-started from the Stage 1 Chunk-1 creeper (`model_1499.pt`) and evaluated with the inflated step-integrated progress metric. Under Task 6's fixed world-frame net-displacement metric (`1462e4f`, `5cfc643`, `dbe0b49`), `model_2098.pt` failed all 7 terrains (`flat 32.2%`, `rough 8.7%`, `slope_up 31.6%`, `slope_down 28.2%`, `stairs_up 26.1%`, `stairs_down 17.0%`, `boxes 25.4%`). |
| **Physical Hardware Sim-to-Real Deployment** | **`NO-GO`** | No Stage 1 policy passes the frozen Stage 1 gate; furthermore the physical platform has no onboard IMU, no joint servo position/velocity/torque feedback, uncalibrated mechanical joint zeros, no hardware E-stop, and an unweighed physical mass (`5.00 kg` is a calibrated assumption). |

---

## 1. Task 1: Exact Diff Between Walking Baseline (`2026-11-05_19-51-02`) and Chunk-1 Creeper (`2026-11-09_22-33-44`)

`2026-11-05_19-51-02/model_599.pt` (`600` iters) walked at **`0.1518 m/s`** (`tripod_corr -0.7527`). Chunk 1 (`2026-11-09_22-33-44/model_1499.pt`, `1500` iters) crept at **`0.0065 m/s`** (`4.3%` of `0.15 m/s`), `Pure Yaw +0.0400 rad/s (10.0%)`, and `Strafe +0.0230 m/s (19.1%)`.

Exact parameter & asset changes between the two configs (`VERIFIED`):

| Category | `2026-11-05_19-51-02` (Walking) | Chunk 1 `2026-11-09_22-33-44` (Creeping) | Commit | Status |
|---|---|---|---|---|
| **Robot Total Mass (`hexa_physics.usd`)** | `4.1662 kg` CAD mass (before payload edit) | **`5.0000 kg`** (`+0.8338 kg` payload on `base_link`; settled height `0.1417 m`, target `0.1346 m`, min `0.1275 m`) | `b58c52c` | **CHANGED** |
| **`torque_limit` Penalty** | Weight `-0.5`, soft threshold `2.10 N·m` | Weight **`-2.0`**, soft threshold **`1.95 N·m`** | `8bbb08d` | **CHANGED** |
| **Velocity Command Distribution** | `UniformVelocityCommand` (standard uniform sampling) | **`ZoeUniformVelocityCommand`** (`rel_yaw_envs: 0.30`, `wz: 0.30..0.60`, `vx, vy: -0.05..0.05` on yaw envs) | `38cd329` | **CHANGED** |
| **`track_ang_vel_z_exp` Weight** | `0.5` | **`1.0`** | `38cd329` | **CHANGED** |
| **`max_iterations`** | `600` | **`1500`** | Chunk 1 CLI/cfg | **CHANGED** |
| **Actuator Model** | `Kp=16.0`, `Kd=1.6`, `effort_limit=2.5 N·m` | `Kp=16.0`, `Kd=1.6`, `effort_limit=2.5 N·m` | — | **SAME** |
| **Action Rate Limiter** | `6.0 deg/step` (`300 deg/s` at `50 Hz`) | `6.0 deg/step` (`300 deg/s` at `50 Hz`) | — | **SAME** |
| **Action Delay** | `0..2` steps | `0..2` steps | — | **SAME** |
| **Mass Domain Randomization** | `[0.90, 1.10]` | `[0.90, 1.10]` | — | **SAME** |

*(Note: `[0.90, 1.10]` was kept in Chunk 1 and Probe `P1`, then expanded to `[0.80, 1.10]` in Task 7 [`6087218`, `d3da62c`] for Probes `P4`, `Probe 3`, and `Probe 4`.)*

---

## 2. Task 2: Simulator Reward Economics (`5.00 kg` Model)

Measured directly in the Isaac Lab physics simulator using [`scripts/analysis/eval_reward_economics.py`](file:///home/rithwik/SpdrBot-main/scripts/analysis/eval_reward_economics.py) (`33d2e8c`, `VERIFIED`):

### 2.1 Training Command Distribution Share

Under Chunk-1's `ZoeUniformVelocityCommand` configuration (`VERIFIED`):
- **`46.10%`** of training environments have translational command speed `|v_xy| < 0.10 m/s`:
  - `7.00%` standing environments (`cmd = [0, 0, 0]`)
  - `30.04%` yaw-dominant environments (`rel_yaw_envs`)
  - `9.05%` uniform draws with `|v_xy| < 0.10 m/s`

### 2.2 Simulator Static Holding Torques (`5.00 kg`, Feet Planted vs Raised)

| Support Stance | Settled Base Height (`m`) | Max Joint Torque `|tau|` (`N·m`) | Mean Thigh Torque (`N·m`) | `torque_limit` Penalty Metric (`> 1.95 N·m`) | `torque_limit` Penalty Metric (`> 2.10 N·m`) |
|---|---|---|---|---|---|
| **6 feet down** | `0.1416` | `1.751` | `1.180` | `0.0000` | `0.0000` |
| **5 feet down** | — | `2.500` | `1.602` | `0.3839` | — |
| **4 feet down** | — | `2.445` | `1.668` | `0.4370` | — |
| **3 feet down (tripod)** | `0.1337` | `2.500` (at hard effort limit) | `2.313` | `1.4730` | `0.7716` |

### 2.3 Per-Second Chunk-1 Reward Economics (`50 Hz`)

Using the measured simulator static torques under Chunk-1 weights (`torque_limit: -2.0 @ 1.95 N·m`, `VERIFIED`):

| Behavior / Command Speed | Per-Second Net Reward (`/s`) | Key Mechanism |
|---|---|---|
| **Standing still (`cmd = [0, 0, 0]`)** | **`+235.80 /s`** | Zero torque penalty (`0.0000`), high stillness/orientation/tracking reward |
| **Freeze / Creep (6 feet down) at `0.10 / 0.15 / 0.25 m/s`** | **`+141.14` / `+112.22` / `+88.28 /s`** | Pays zero `torque_limit` penalty (`6ft max |tau| = 1.751 N·m < 1.95 N·m`) |
| **Dynamic Tripod Walk (3 feet down) at `0.10 / 0.15 / 0.25 m/s`** | **`+46.84` / `+46.75` / `+46.45 /s`** | Incurs **`-147.30 /s`** `torque_limit` penalty (`-2.0 * 1.4730 * 50`), losing **`-94.30` / `-65.47` / `-41.83 /s`** relative to creeping |

Conclusion (`VERIFIED`): At `5.00 kg`, Chunk-1's `-2.0 @ 1.95 N·m` `torque_limit` penalty made lifting 3 legs into a tripod stance strongly reward-negative compared to keeping all 6 feet planted.

---

## 3. Task 3 & Task 4: Bisection (`P1`) and Stage 1 Improvement Probes (`P4`, `Probe 3`, `Probe 4`)

All 4 GPU probes used the fixed budget of `1024` environments, `600` iterations, trained **from scratch**, and evaluated on the frozen 15-condition gate in [`scripts/chunk_check.py`](file:///home/rithwik/SpdrBot-main/scripts/chunk_check.py) (`52685d4`, no thresholds modified).

### 3.1 Probe Comparison Summary (`VERIFIED`)

| Run / Label | Key Config Delta | Frozen Gate Score | Forward (`>= 0.1200 m/s`) | Pure Yaw (`>= 50%`) | Strafe (`>= 50%`) | Backward (`>= 50%`) | Contacts (`>= 3ft` / `>= 5ft` ZeroCmd) | Stability `p5` (`>= 0.0 m`) | Notable TB / Gait Diagnostics |
|---|---|---|---|---|---|---|---|---|---|
| **Chunk 1 (`2026-11-09_22-33-44/model_1499.pt`)** | `torque -2.0@1.95`, `yaw 1.0`, `fewer_3 -0.5`, `rand [0.90, 1.10]` | Freeze baseline | **`0.0065 m/s`** | `+0.0400 rad/s (10.0%)` | `+0.0230 m/s (19.1%)` | — | — | — | Stuck in 6-foot creep |
| **Probe 1 (`P1`, `06d4486`, `25e194f`, `2026-11-11_11-46-21/model_599.pt`)** | **ONLY** `torque_limit` reverted to `-0.5 @ 2.10 N·m`; no ramp; `[0.90, 1.10]` | **`12/15 PASS`** | **`0.1595 m/s` (PASS)** | `+0.1515 rad/s (37.9%)` (FAIL) | **`+0.1091 m/s (90.9%)` (PASS)** | **`-0.1148 m/s (95.7%)` (PASS)** | `>=3ft 95.70%` (FAIL) / `C1 >=5ft 98.38%` (PASS) | `-0.0156 m` (FAIL) | `tripod_corr -0.5049`, `Leg 5` `2.0%` duty (`38.41` TB reward), `C2 TorqSat 14.21%` |
| **Probe 2 (`P4`, `a7882f2`, `9383efb`, `2026-11-11_16-01-30/model_599.pt`)** | `torque -1.0 @ 2.10`, **`penalty_scale_ramp` (`0.1->1.0 @ 800`)**, `[0.80, 1.10]` | **`11/15 PASS`** | `0.0940 m/s (62.7%)` (FAIL) | `+0.0358 rad/s (8.9%)` (FAIL) | `+0.0582 m/s (48.5%)` (FAIL) | **`-0.1186 m/s (98.8%)` (PASS)** | `>=3ft 98.83%` (FAIL) / `C1 >=5ft 97.09%` (PASS) | **`+0.0258 m` (PASS)** | Tripod A `65.2%` vs B `64.8%`, `corr -0.6980`, `5.55/6` feet on yaw, `C2 TorqSat 15.43%`; TB stalled (`26.30->21.11->26.04`, `noise_std 0.5936`, `action_rate_l2 -0.5279`) |
| **Probe 3 (`5693d4d`, `bc040bb`, `2026-11-11_18-50-08/model_599.pt`)** | Ramp + **`rewards.py` yaw gating fix** + `torque -0.5 @ 2.10` + `yaw 1.5@0.40` + **`fewer_3 -1.0`**, `[0.80, 1.10]` | **`12/15 PASS`** | `0.0066 m/s` (FAIL) | `+0.1819 rad/s (45.5%)` (FAIL) | **`+0.0877 m/s (73.1%)` (PASS)** | `-0.0527 m/s (43.9%)` (FAIL) | **`>=3ft 99.42%` (PASS)** (`>=4ft 97.45%`) / `C1 >=5ft 96.79%` (PASS) | **`+0.0415 m` (PASS)** | `4.94/6` feet on yaw (`63.9%` on `0.50 rad/s` step), `C2 TorqSat 12.99%`; TB stalled (`35.21->33.20->35.98`, `noise_std 0.6196`, `action_rate_l2 -0.5950`) |
| **Probe 4 (`9662a60`, `744ea37`, `2026-11-11_22-30-05/model_599.pt`)** | **NO ramp** + `rewards.py` yaw fix + `torque -0.5 @ 2.10` + `yaw 1.25@0.40` + `fewer_3 -1.0`, `[0.80, 1.10]` | **`13/15 PASS`** | `0.0079 m/s` (FAIL) | **`+0.2011 rad/s (50.3%)` (PASS)** | `+0.0420 m/s (35.0%)` (FAIL) | **`-0.1087 m/s (90.6%)` (PASS)** | **`>=3ft 99.41%` (PASS)** (`>=4ft 99.36%`) / `C1 >=5ft 98.33%` (PASS) | **`+0.0741 m` (PASS)** (`Push 100%` PASS) | Forward duty A `90.0%` / B `98.0%`, `corr -0.0253`, Yaw step `0.2599 rad/s` on `0.50`, `C2 TorqSat 9.51%`; TB monotonic (`-0.7028->18.2780->30.5465->40.7297`, `noise_std 0.4538`) |

### 3.2 Key Mechanistic Findings From the 4 Probes

1. **Chunk-1 Freeze Root Cause Confirmed (`P1`, `VERIFIED`)**: Reverting only `torque_limit` from `-2.0 @ 1.95 N·m` to `-0.5 @ 2.10 N·m` jumped forward speed **`24.5×`** (`0.0065 -> 0.1595 m/s`), so bisection stopped after `P1` per rule.
2. **Yaw Gait-Gating Bug Fixed (`5693d4d`, `VERIFIED`)**: `feet_air_time_gated` and `foot_tip_clearance_reward` in `rewards.py` previously gated on `torch.norm(cmd[:, :2]) > 0.05`, withholding stepping/clearance reward during pure-yaw commands (`P4`: `8.9%` yaw, `5.55/6` feet planted). Updating the gate to `torch.norm(cmd[:, :2]) + torch.abs(cmd[:, 2]) > 0.05` plus `track_ang_vel_z_exp` `std=0.40` enabled `Probe 4` to achieve **`50.3%` Pure Yaw (`PASS`)**. Verified by unit test `test_stepping_rewards_active_on_pure_yaw_and_off_when_stopped` in [`tests/test_height_above_ground.py`](file:///home/rithwik/SpdrBot-main/tests/test_height_above_ground.py).
3. **`penalty_scale_ramp` Stalled Exploration & Mid-Run Reward (`P4` & `Probe 3`, `VERIFIED`)**: Implementing Task 4's `0.1 -> 1.0` linear ramp over `800` iterations (`6bc0acd`) scaled `action_rate_l2` down early, allowing `Policy/mean_noise_std` to rise to `1.03–1.04` around iteration `100` while continuous negative penalty drift flattened `Train/mean_reward` (`P4`: `26.30 -> 26.04`; `Probe 3`: `35.21 -> 35.98`; `noise_std` stayed at `0.59–0.62`). Disabling the ramp in `Probe 4` (`9662a60`, leaving `FlatCurriculumCfg` / `penalty_scale_ramp` available in code/tests) restored monotonic learning (`-0.7028 -> 40.7297`, `noise_std -> 0.4538`).
4. **Remaining Forward Suppression Mechanism (`INFERRED` from `VERIFIED` contrast)**: Every run with `fewer_than_3_feet_contact = -0.5` (`2026-11-05` at `0.1518 m/s`, `P1` at `0.1595 m/s`, `P4` at `0.0940 m/s` with `corr -0.6980`) actively alternated tripods forward, whereas both runs with `fewer_than_3_feet_contact = -1.0` (`Probe 3`: `97.45% >= 4ft`, `0.0066 m/s`; `Probe 4`: `99.36% >= 4ft`, `0.0079 m/s`, duty `90.0% / 98.0%`) kept `>= 4` feet planted forward to avoid transient `< 3`-foot penalties during tripod handover. **Recommended next candidate once GPU budget is renewed**: keep `9662a60` (no ramp, yaw gating fix, `yaw 1.25 @ 0.40`, `torque -0.5 @ 2.10`) and restore `fewer_than_3_feet_contact` from `-1.0` to **`-0.5`** (untested in this session).

---

## 4. Task 5 (Frozen Gate) & Task 6 (Net-Displacement Progress Metric)

### 4.1 Task 5 — Frozen Stage 1 Gate (`scripts/chunk_check.py`, `52685d4`, `VERIFIED`)

- Converted `C2 TorqSat` from hard pass/fail to **report-only** (reflecting legitimate dynamic walking spikes: in tripod stance at `5.0 kg`, static thigh torque is already `~1.88–2.31 N·m` vs `2.50 N·m` effort limit).
- Added directional command tracking gates (**15 pass/fail checks total**), including `C1 >= 5` feet in contact (`>= 95%` on zero cmd), `C2 Forward >= 0.1200 m/s` (`>= 80%` of `0.15 m/s`), `C4 Pure Yaw >= 0.2000 rad/s` (`>= 50%` of `0.40 rad/s`), `C5 Strafe >= 0.0600 m/s` (`>= 50%` of `0.12 m/s`), and `C6 Backward <= -0.0600 m/s` (`>= 50%` of `-0.12 m/s`). All four probes were evaluated against this unchanged frozen gate.

### 4.2 Task 6 — World-Frame Net-Displacement Progress Metric (`1462e4f`, `5cfc643`, `dbe0b49`, `VERIFIED`)

The Stage 2 curriculum previously integrated instantaneous step progress (`sum max(0, v_xy . unit(cmd_xy)) dt`), which let policies accumulate high fake progress by shuffling or oscillating in place. Task 6 replaced it with net world-frame displacement `(pos_end_w - pos_start_w)[:2] . unit(cmd_dir_w)` using full 3D `root_quat_w` rotation and pre-reset `pos_end_w` capture.

Exact `OLD` vs `NEW` progress across the 7 terrains (`dbe0b49`, `VERIFIED`):

| Terrain | `2026-11-05/model_599.pt` (`NEW` / `OLD`) | Stage 1 Chunk-1 `model_1499.pt` (`NEW` / `OLD`) | Stage 2 Probe `model_2098.pt` (`NEW` / `OLD`) |
|---|---|---|---|
| **`flat`** | `102.5%` / `102.0%` | `2.3%` / `5.6%` | **`32.2%`** / `58.3%` |
| **`rough`** | `53.3%` / `60.0%` | `2.9%` / `6.9%` | **`8.7%`** / `43.9%` (`94.9%` stuck) |
| **`slope_up`** | `106.4%` / `105.7%` | `2.4%` / `6.0%` | **`31.6%`** / `57.1%` |
| **`slope_down`** | `90.3%` / `94.0%` | `2.4%` / `5.7%` | **`28.2%`** / `56.6%` |
| **`stairs_up`** | `98.1%` / `97.6%` | `2.4%` / `5.9%` | **`26.1%`** / `54.1%` |
| **`stairs_down`** | `37.1%` / `51.4%` | `2.2%` / `5.8%` | **`17.0%`** / `46.2%` (`76.9%` stuck) |
| **`boxes`** | `63.1%` / `69.0%` | `2.2%` / `5.8%` | **`25.4%`** / `53.4%` |
| **Overall** | `FAIL` (rough / stairs_down / boxes) | `100% FAIL` | **`100% FAIL`** (old metric hid shuffling in place) |

---

## 5. Task 7: Mass Randomization `[0.80, 1.10]` & Payload Recalibration Tooling (`6087218`, `d3da62c`)

- **Expanded Randomization Range (`params.py`, `deploy/mass_properties.yaml`, `docs/ASSUMPTIONS.md`, `VERIFIED`)**: Set multiplicative mass scale range to **`(0.80, 1.10)`** (`4.00–5.50 kg` around `5.000 kg`) so the trained policy spans lighter unloaded CAD configurations (`~4.17 kg`) through heavy payload configurations (`5.50 kg`).
- **One-Command Recalibration Script (`scripts/set_payload_mass.py`, `VERIFIED`)**: Added CLI `--total-mass-kg` (default `5.000`) to recompute payload mass & inertia cleanly and synchronize `hexapod/hexa.urdf`, `deploy/mass_properties.yaml`, `deploy/asset_report.json`, and `hexapod/hexa/configuration/hexa_physics.usd` once the physical robot is weighed.

---

## 6. Task 9: Workspace & Package Hygiene Verification (`VERIFIED`)

Executed directly in the Isaac Lab environment and git history:
- `python -c "import spdrbot3; print(spdrbot3.__file__)"` -> `/home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/__init__.py`
- `pip show spdrbot3` -> `Name: spdrbot3`, `Version: 0.1.0`, `Editable project location: /home/rithwik/SpdrBot-main/source/spdrbot3`
- Stage B legacy cleanup commit `3fbbdf6`: `34 files changed, 2 insertions(+), 1845 deletions(-)` (removed `check_cam.py`, `check_mdp.py`, `check_model.py`, `spdr.usd`, `spdr_stage.usd`, `spdrbot.py`, `spdrbot3_direct_project`, `spyderbot_minimal URDF/*`, `spyderbot_test.py`).
