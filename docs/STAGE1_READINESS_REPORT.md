# Stage 1 & Deployment Readiness Report

- **Date**: 2026-11-12
- **Repository Baseline**: Branch `main`, export commit `28b83b8` (`Stage 1 Chunk 1` trained on config `ecd2c66`, evaluation logs `fab4649` / `6e4a795`)
- **Active Environment / Robot Asset**: `Zoe-Hexapod-Flat-Deploy-v0` (`5.0000 kg` payload model in `hexapod/hexa/configuration/hexa_physics.usd`, settled standing height `0.1417 m`, target `0.1346 m`, minimum `0.1275 m`)
- **Final Stage 1 Checkpoint**: `/home/rithwik/SpdrBot-main/logs/rsl_rl/zoe_hexapod_deploy/2026-11-12_05-47-14/model_1499.pt`

---

## Executive Verdicts

| Scope | Verdict | Basis (`VERIFIED`) |
|---|---|---|
| **Stage 1 (`Zoe-Hexapod-Flat-Deploy-v0`) 16-Check Frozen Gate ([`scripts/chunk_check.py`](file:///home/rithwik/SpdrBot-main/scripts/chunk_check.py))** | **`PASS` (`16/16`)** | Winning probe config **Probe A** (`ecd2c66`: `P1` rewards + `[0.80, 1.10]` mass randomization + [`is_cmd_stopped`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/mdp/rewards.py#L38-L40) gating fix `ff97785`, no penalty ramp) passed `16/16` at 600 iters (`18b317b`), then ran full `1500`-iter **Stage 1 Chunk 1** (`2026-11-12_05-47-14`, `1024` envs, `73,728,000` steps, `35m 26s`). Both `model_1000.pt` and `model_1499.pt` pass all **16/16** checks (`Forward 0.1544 m/s`, `Pure Yaw +0.4172 rad/s [104.3%]`, `Strafe +0.1077 m/s [89.8%]`, `Backward -0.1033 m/s [86.1%]`, `Check #16 min 54.7%, diff 3.1pp`, `p5 +0.0571 m`, `>=3ft 99.42%`, `0 falls`, tripod `r = -0.8422`). |
| **Stage 2 (`Zoe-Hexapod-Terrain-v0`)** | **`NOT READY`** (Previous `GO` Remains Retracted) | Earlier `GO` on `2026-11-10_18-11-19/model_2098.pt` remains **retracted** (`100% FAIL` under Task 6 net-displacement progress `1462e4f`, `5cfc643`, `dbe0b49`). Multi-terrain Stage 2 training warm-started from `2026-11-12_05-47-14/model_1499.pt` has **not yet started** (`Zoe-Hexapod-Terrain-v0` cannot use a Stage 1 flat policy zero-shot: `Probe A` passed `4/7` terrain targets and failed `rough 38.9%`, `stairs_down 42.1%`, `boxes 57.0%`). |
| **Physical Hardware Sim-to-Real Deployment** | **`NO`** | Cleared **only** for Stage 2 simulation training and software pipeline testing ([`deploy/test_export_not_for_hardware/deploy.yaml`](file:///home/rithwik/SpdrBot-main/deploy/test_export_not_for_hardware/deploy.yaml) flagged `status: NOT_FOR_HARDWARE_UNCALIBRATED`). Physical robot lacks onboard IMU, joint servo position/velocity/torque feedback, calibrated mechanical joint zeros, hardware E-stop, and a measured body weight (`5.00 kg` is a calibrated assumption). |

---

## 1. Command Gating Audit & Fix (`ff97785`)

Every reward and penalty gating standing vs walking was audited so `"stopped"` means `|cmd_xy| < 0.05 AND |cmd_yaw| < 0.05` everywhere (`VERIFIED`):

- Added [`is_cmd_stopped`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/mdp/rewards.py#L38-L40): `(torch.norm(cmd[:, :2], dim=-1) < cmd_threshold) & (torch.abs(cmd[:, 2]) < cmd_threshold)`.
- Audited & updated all 4 command-gated reward terms in [`rewards.py`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/mdp/rewards.py):
  - [`feet_air_time_gated`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/mdp/rewards.py#L43-L60): `~is_cmd_stopped(cmd, 0.05)` (changed from additive `|v_xy| + |wz| > 0.05`).
  - [`foot_tip_clearance_reward`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/mdp/rewards.py#L351-L378): `~is_cmd_stopped(cmd, 0.05)`.
  - [`stand_still_posture_penalty`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/mdp/rewards.py#L381-L403): `is_cmd_stopped(cmd, cmd_threshold)` (previously `(norm(cmd_xy) + abs(cmd_yaw)) < 0.05`, which silently treated combined small commands e.g. `[0.03, 0, 0.03]` or boundary commands inconsistently relative to axis-wise checks).
  - [`stand_still_feet_grounded_reward`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/mdp/rewards.py#L406-L426): `is_cmd_stopped(cmd, cmd_threshold)`.
- Note: [`fewer_than_3_feet_contact`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/mdp/rewards.py#L18-L25) and [`fewer_than_4_feet_contact`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/mdp/rewards.py#L28-L35) are intentionally ungated across all commands (3- or 4-foot support floor applies regardless of command).
- Unit-tested in `test_command_stopped_definition_and_all_gated_rewards` in [`tests/test_height_above_ground.py`](file:///home/rithwik/SpdrBot-main/tests/test_height_above_ground.py) across `[0,0,0]`, `[0.03,0,0.03]`, `[0.049,0,0.049]`, `[0.03,0.03,0.04]`, `[0.05,0,0]`, `[0,0,0.05]`, `[0.04,0.04,0]`, `[0.15,0,0]`, `[0,0,0.40]`.

---

## 2. Check #16, Historical Re-Score, Probe A/B Selection, and Chunk 1 Results

### 2.1 New Frozen Check #16 (`7576351`)

Added to [`scripts/chunk_check.py`](file:///home/rithwik/SpdrBot-main/scripts/chunk_check.py#L409) before any new runs:
- **Condition**: During Condition 2 (`Forward Walk 0.15 m/s`, 10 s), every leg's duty factor (`L1..L6`) must be **`>= 40.0%`**, and the two tripod groups' mean duties (`Tripod A [1,3,5]` vs `Tripod B [2,4,6]`) must differ by **`<= 15.0 percentage points`**.
- **Hardware Rationale**: A policy that leaves one leg idle in the air walks on 5 legs, concentrating load and thermal stress onto adjacent actuators.
- Report-only torque & support metrics were also added (`EFFORT_80_THRESHOLD = 2.00 N·m`, per-joint-type mean/RMS torque & `>80%` effort fractions, mean feet down, and `<3/3/4/5/6` feet support distribution).

### 2.2 Full 16-Check Gate Table (`VERIFIED`, Frozen Thresholds)

| Run / Checkpoint | Commit | Config / Notes | Gate Score | Forward (`>=0.1200`) | Pure Yaw (`>=50%`) | Strafe (`>=50%`) | Backward (`>=50%`) | Contacts (`C2 >=3ft >=99%` / `C1 >=5ft >=95%`) | Stability `p5` (`>=0.0 m`) | Check #16 (`min>=40%`, `diff<=15pp`) | L1–L6 Duty Factors (`%`) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **P1** (`2026-11-11_11-46-21/model_599.pt`) | `4ef9a86` (re-score) | `torque -0.5@2.10`, `yaw 1.0@0.25`, `fewer_3 -0.5`, `[0.90,1.10]`, old gating | **`12/16 FAIL`** | **`0.1632`** (PASS) | `+0.1559` `[39.0%]` (FAIL) | **`+0.1093`** `[91.1%]` (PASS) | **`-0.1102`** `[91.9%]` (PASS) | `94.35%` (FAIL) / `98.38%` (PASS) | `-0.0370` (FAIL) | `min 9.5% [L5]`, `diff 29.9pp (49.1/79.0)` (**FAIL**) | `76.4, 65.6, 61.5, 83.2, 9.5, 88.3` |
| **P4** (`2026-11-11_16-01-30/model_599.pt`) | `4ef9a86` (re-score) | Ramp `0.1->1.0@800`, `torque -1.0@2.10`, `[0.80,1.10]`, old gating | **`11/16 FAIL`** | `0.0959` (FAIL) | `+0.0356` `[8.9%]` (FAIL) | `+0.0544` `[45.4%]` (FAIL) | **`-0.1118`** `[93.2%]` (PASS) | `98.85%` (FAIL) / `97.09%` (PASS) | **`+0.0245`** (PASS) | `min 32.0% [L5]`, `diff 22.3pp (57.5/79.7)` (**FAIL**) | `70.4, 85.7, 69.9, 64.5, 32.0, 89.0` |
| **Probe 3** (`2026-11-11_18-50-08/model_599.pt`) | `4ef9a86` (re-score) | Ramp, `fewer_3 -1.0`, `yaw 1.5@0.40`, `5693d4d` gating, `[0.80,1.10]` | **`12/16 FAIL`** | `0.0067` (FAIL) | `+0.1734` `[43.3%]` (FAIL) | **`+0.0845`** `[70.4%]` (PASS) | `-0.0523` `[43.6%]` (FAIL) | **`99.40%`** / `96.79%` (PASS) | **`+0.0444`** (PASS) | `min 9.5% [L5]`, `diff 34.2pp (62.1/96.3)` (**FAIL**) | `98.7, 98.0, 78.2, 97.3, 9.5, 93.6` |
| **Probe 4** (`2026-11-11_22-30-05/model_599.pt`) | `4ef9a86` (re-score) | No ramp, `fewer_3 -1.0`, `yaw 1.25@0.40`, `5693d4d` gating | **`14/16 FAIL`** | `0.0073` (FAIL) | **`+0.2017`** `[50.4%]` (PASS) | `+0.0430` `[35.8%]` (FAIL) | **`-0.1042`** `[86.9%]` (PASS) | **`99.42%`** / `98.33%` (PASS) | **`+0.0729`** (PASS) | `min 62.1% [L6]`, `diff 8.9pp (94.9/86.1)` (PASS via creep) | `96.6, 98.9, 91.6, 97.3, 96.6, 62.1` |
| **Probe A** (`2026-11-11_23-36-34/model_599.pt`) | `1c8ecc5` (`18b317b`) | **P1 weights + `[0.80,1.10]` + `ff97785` gating + no ramp** | **`16/16 PASS`** | **`0.1432`** `[95.5%]` (PASS) | **`+0.3552`** `[88.8%]` (PASS) | **`+0.0924`** `[77.0%]` (PASS) | **`-0.1107`** `[92.2%]` (PASS) | **`99.28%`** / **`98.28%`** (PASS) | **`+0.0288`** (PASS) | **`min 57.5% [L1], diff 0.2pp (66.6/66.8)`** (**PASS**) | `57.5, 64.8, 60.5, 76.6, 81.7, 58.9` |
| **Probe B** (`2026-11-11_23-58-51/model_599.pt`) | `4a6cf4b` (`d0a2718`) | Probe 4 `yaw 1.25@0.40` + `fewer_3 -0.5` + `ff97785` gating | **`14/16 FAIL`** | **`0.1361`** `[90.7%]` (PASS) | **`+0.3511`** `[87.8%]` (PASS) | `+0.0409` `[34.0%]` (FAIL) | **`-0.0901`** `[75.1%]` (PASS) | `98.77%` (FAIL) / `98.20%` (PASS) | **`+0.0006`** (PASS) | `min 45.1% [L5]`, `diff 10.8pp (59.2/70.0)` (PASS) | `66.9, 71.6, 65.5, 79.6, 45.1, 58.8` |
| **Chunk 1 @ 500** (`2026-11-12_05-47-14/model_500.pt`) | `ecd2c66` (`6e4a795`) | Probe A config @ 500 iters (early milestone) | **`14/16 FAIL`** | **`0.1384`** `[92.3%]` (PASS) | `+0.0689` `[17.2%]` (FAIL) | `+0.0435` `[36.3%]` (FAIL) | **`-0.1030`** `[85.8%]` (PASS) | **`99.06%`** / **`98.24%`** (PASS) | **`+0.0110`** (PASS) | `min 56.3% [L1]`, `diff 3.4pp (63.9/67.2)` (PASS) | `56.3, 69.7, 59.4, 69.1, 76.0, 62.9` |
| **Chunk 1 @ 1000** (`2026-11-12_05-47-14/model_1000.pt`) | `ecd2c66` (`6e4a795`) | Probe A config @ 1000 iters | **`16/16 PASS`** | **`0.1489`** `[99.3%]` (PASS) | **`+0.4277`** `[106.9%]` (PASS) | **`+0.1128`** `[94.0%]` (PASS) | **`-0.1142`** `[95.2%]` (PASS) | **`99.40%`** / **`98.34%`** (PASS) | **`+0.0520`** (PASS) | **`min 55.3% [L1], diff 4.4pp (62.0/66.4)`** (**PASS**) | `55.3, 61.3, 59.3, 65.9, 71.2, 71.9` |
| **Chunk 1 @ 1499** (`2026-11-12_05-47-14/model_1499.pt`) | `ecd2c66` (`6e4a795`) | **Final Stage 1 Chunk 1 @ 1500 iters** | **`16/16 PASS`** | **`0.1544`** `[102.9%]` (PASS) | **`+0.4172`** `[104.3%]` (PASS) | **`+0.1077`** `[89.8%]` (PASS) | **`-0.1033`** `[86.1%]` (PASS) | **`99.42%`** / **`98.36%`** (PASS) | **`+0.0571`** (`mean 0.0864`, PASS) | **`min 54.7% [L1], diff 3.1pp (63.3/66.4)`** (**PASS**) | `54.7, 58.8, 59.9, 61.6, 75.3, 78.8` |

*(Push Recovery C3: all runs `P1`..`Chunk 1` passed `0 falls`, `0.00%` non-foot contact, and `99.95%–100.00%` recovery `>= 0.1275 m`; C1/C2 height `100.00% >= 0.1275 m` on Probe A and Chunk 1.)*

### 2.3 Why Check #16 & Probe A/B Mattered

1. **Five-legged limp caught by Check #16 (`VERIFIED`)**: `P1` (`L5 = 9.5%`), `P4` (`L5 = 32.0%`), and `Probe 3` (`L5 = 9.5%`) all carried 5-legged limps and were rejected under Check #16, whereas `Probe 4` satisfied Check #16 only by keeping `5.43/6` feet planted (`0.0073 m/s` creep).
2. **Unified gating (`ff97785`) + `fewer_than_3_feet_contact = -0.5` unlocked all 4 motion skills (`VERIFIED`)**: Restoring `fewer_than_3_feet_contact = -0.5` (which does not choke tripod crossover transients) while keeping the fixed `is_cmd_stopped` gate allowed **Probe A** (`1024` envs x `600` iters, `14m 02s`, TB `-0.61 -> 41.25`, `noise_std 1.00 -> 0.41`) to pass **`16/16`** cleanly (`min duty 57.5%`, tripod diff `0.2pp`, single-env `eval_gait_report.py` `corr = -0.8520`).
3. **Why Probe A beat Probe B (`VERIFIED`)**: Widening/boosting yaw to `1.25 @ 0.40` (`Probe B`, `14m 15s`, TB `-0.48 -> 44.70`) degraded `Strafe` (`34.0% FAIL`), `>= 3ft` support (`98.77% FAIL`), and tripod alternation (`corr = -0.4975`), so **Probe C was skipped** (`2/3` probes used) and **Probe A** (`ecd2c66`) ran to full `1500`-iteration Stage 1 completion (`2026-11-12_05-47-14`).

### 2.4 Stage 1 Chunk 1 (`2026-11-12_05-47-14`) Training & Single-Env Gait Report (`VERIFIED`)

- **Training Progression (`1024` envs x `1500` iters, `73,728,000` steps, `35m 26s`)**:
  - `it=0`: reward `-0.61`, `noise_std 1.00`, `time_out 0.0308`
  - `it=250`: reward `22.50`, `noise_std 0.57`, `time_out 0.9990`
  - `it=500`: reward `35.42`, `noise_std 0.45`, `time_out 0.9893` (Forward/Backward/Check #16 already passing; Yaw `17.2%` & Strafe `36.3%` still maturing)
  - `it=750`: reward `45.97`, `noise_std 0.35`, `time_out 1.0000`
  - `it=1000`: reward `51.67`, `noise_std 0.30`, `time_out 1.0000` (**`16/16 PASS`** achieved by iter 1000)
  - `it=1250`: reward `53.38`, `noise_std 0.27`, `time_out 1.0000`
  - `it=1499`: reward **`55.12`**, `noise_std 0.25`, `time_out 1.0000` (`track_lin_vel_xy_exp 1.6944`, `track_ang_vel_z_exp 0.6676`, `base_height_shaped 0.9793`, `action_rate_l2 -0.1559`, `fewer_than_3_feet_contact -0.0153`, `torque_limit -0.2188`).
- **Single-Env Gait Report (`scripts/eval_gait_report.py` on `model_1499.pt`, `VERIFIED`)**:
  - Forward phase (`2.0s–6.0s`) duties: `L1 50.5%`, `L2 63.5%`, `L3 47.5%`, `L4 69.0%`, `L5 59.5%`, `L6 58.5%`.
  - Swing clearances: `L1 3.00`, `L2 0.81`, `L3 1.55`, `L4 0.24`, `L5 0.58`, `L6 0.46 cm` (parallel C2 clearances across all envs: `L1 2.64`, `L2 2.08`, `L3 2.12`, `L4 2.06`, `L5 2.07`, `L6 2.04 cm`).
  - Tripod Group A `(1,3,5)` = `52.5%`, Tripod Group B `(2,4,6)` = `63.7%`, **cross-group correlation = `-0.8422`** (`STRONG ALTERNATING GAIT CONFIRMED`).
  - Step command tracking (`eval_gait_report.py`): Forward `cmd 0.20 -> +0.2298 m/s`; Yaw `cmd 0.50 -> +0.6487 rad/s`; Strafe `cmd 0.15 -> +0.1493 m/s`.

---

## 3. Torque, Contact Distribution, and Static Holding Torque Root Cause

### 3.1 Dynamic Torque & Support Breakdown (`scripts/chunk_check.py`, `VERIFIED`)

*(Hard effort limit = `2.50 N·m`; saturation threshold = `>= 2.45 N·m`; `>80%` threshold = `> 2.00 N·m`; Condition 2 Forward Walk.)*

| Run / Checkpoint | TorqSat `C1 / C2 / C3` (`%`) | C2 Coxa Mean/RMS (`>80%`) | C2 Thigh Mean/RMS (`>80%`) | C2 Foot Mean/RMS (`>80%`) | C2 Overall `>80%` (Joint-Steps / Any-Frame) | C2 Mean Feet Down | Feet Contact Dist `<3 / 3 / 4 / 5 / 6` (`%`) |
|---|---|---|---|---|---|---|---|
| **P1 (`model_599.pt`)** | `7.16 / 14.33 / 14.45` | `0.407 / 0.643 (1.83%)` | `1.210 / 1.516 (29.48%)` | `1.029 / 1.402 (25.59%)` | `18.97%` / `99.04%` | `3.84 / 6` | `5.65 / 29.70 / 45.84 / 11.07 / 7.74` |
| **P4 (`model_599.pt`)** | `9.61 / 15.63 / 15.83` | `0.486 / 0.749 (3.22%)` | `1.243 / 1.551 (30.95%)` | `1.070 / 1.438 (27.03%)` | `20.40%` / `99.31%` | `4.11 / 6` | `1.15 / 26.19 / 40.38 / 23.47 / 8.82` |
| **Probe 3 (`model_599.pt`)** | `12.43 / 13.27 / 13.96` | `0.398 / 0.578 (0.82%)` | `1.080 / 1.360 (20.04%)` | `1.304 / 1.619 (34.53%)` | `18.46%` / `99.30%` | `4.75 / 6` | `0.60 / 1.75 / 25.59 / 64.77 / 7.29` |
| **Probe 4 (`model_599.pt`)** | `7.96 / 9.65 / 10.10` | `0.433 / 0.598 (0.70%)` | `1.102 / 1.347 (17.09%)` | `1.209 / 1.499 (26.60%)` | `14.79%` / `98.85%` | `5.43 / 6` | `0.58 / 0.07 / 6.21 / 40.85 / 52.28` |
| **Probe A (`model_599.pt`)** | `6.87 / 14.05 / 14.11` | `0.372 / 0.588 (1.50%)` | `1.229 / 1.542 (30.47%)` | `1.029 / 1.400 (25.24%)` | `19.07%` / `99.15%` | `4.00 / 6` | `0.72 / 29.86 / 46.37 / 13.76 / 9.30` |
| **Probe B (`model_599.pt`)** | `10.12 / 15.12 / 15.12` | `0.417 / 0.654 (1.94%)` | `1.204 / 1.532 (30.85%)` | `1.015 / 1.392 (25.15%)` | `19.31%` / `99.33%` | `3.87 / 6` | `1.23 / 37.40 / 40.29 / 13.79 / 7.30` |
| **Chunk 1 @ 500** | `8.71 / 14.86 / 15.18` | `0.411 / 0.650 (2.17%)` | `1.232 / 1.554 (31.83%)` | `1.026 / 1.396 (25.05%)` | `19.68%` / `99.13%` | `3.93 / 6` | `0.94 / 36.85 / 38.75 / 13.68 / 9.78` |
| **Chunk 1 @ 1000** | `7.04 / 14.41 / 14.69` | `0.371 / 0.584 (1.25%)` | `1.302 / 1.628 (36.27%)` | `0.931 / 1.300 (20.65%)` | `19.39%` / `99.05%` | `3.85 / 6` | `0.60 / 46.07 / 31.62 / 10.10 / 11.61` |
| **Chunk 1 @ 1499** | **`4.14 / 13.98 / 13.74`** | `0.357 / 0.560 (1.01%)` | `1.288 / 1.610 (34.98%)` | `0.947 / 1.324 (21.74%)` | `19.24%` / `98.78%` | `3.89 / 6` | `0.58 / 39.64 / 38.58 / 11.54 / 9.66` |

Joint-limit proximity (`<= 3 deg`) during Forward Walk was **`0.00%`** for Coxa, Thigh, and Foot across every tested policy. Meanwhile, thermal exposure remains elevated when walking (`34.98%` of Thigh joint-steps and `21.74%` of Foot joint-steps exceed `2.00 N·m` (`80%`), with `13.98%` hitting `>= 2.45 N·m` in simulation), reflecting the real static-mechanics ceiling documented below.

### 3.2 Simulator Static Holding Torques at `~5.0 kg` vs `~4.0 kg` ([`diagnose_static_torques.py`](file:///home/rithwik/SpdrBot-main/scripts/analysis/diagnose_static_torques.py), `4d36642`, `VERIFIED`)

Executed directly in Isaac Lab (`5` envs, `Kp = 16.0` w/ `(0.70, 1.30)` DR, `effort_limit = 2.50 N·m`, stance joints held at default `(0, 0, -pi/2)` while lifted legs clear the floor):

| Support Configuration | `4.9699 kg` (`Mg = 48.75 N`): Height (`m`) | `4.9699 kg`: `max |tau|` (`N·m`) & Offending Joint | `4.9699 kg`: `< 2.50` / `<= 2.00` (`N·m`) | `3.9759 kg` (`Mg = 39.00 N`): Height (`m`) | `3.9759 kg`: `max |tau|` (`N·m`) & Offending Joint | `3.9759 kg`: `< 2.50` / `<= 2.00` (`N·m`) |
|---|---:|---|---:|---:|---|---:|
| **6 feet (all down)** | `0.1420` | `1.813` (`Foot L4`) | `True` / **`True`** | `0.1426` | `1.936` (`Foot L5`) | `True` / **`True`** |
| **5 feet (lift L2)** | `0.1391` | **`2.500`** (`Foot L1 & L3`) | `False` / `False` | `0.1400` | **`2.500`** (`Foot L3`) | `False` / `False` |
| **4 feet symmetrical (lift L2, L5)** | `0.1394` | `2.375` (`Thigh L6`, `Foot L4 +2.237`) | **`True`** / `False` | `0.1399` | `2.186` (`Foot L4`) | **`True`** / `False` |
| **3 feet Tripod A (lift L2, L4, L6)** | `0.1344` | **`2.500`** (`Thigh L1, L3 & Foot L3, L5`) | `False` / `False` | `0.1363` | **`2.500`** (`Foot L3, L5`) | `False` / `False` |
| **3 feet Tripod B (lift L1, L3, L5)** | `0.1360` | **`2.500`** (`Thigh L2, L6`) | `False` / `False` | `0.1382` | **`2.500`** (`Foot L2`) | `False` / `False` |

Why `Foot` saturates even though its vertical gravity moment arm `r_xy(tip - foot)` is only `0.014–0.021 m` (`15 N * 0.017 m ~ 0.25 N·m`, `VERIFIED` values + `INFERRED` closed-chain kinematics):
1. **Closed-chain horizontal friction + tibia moment arm**: Holding default stance targets `(0, 0, -pi/2)` while floor friction pins the foot tips horizontally means vertical body sag (`q_thigh -> -0.15..-0.22 rad`) geometrically pulls `q_foot` inward past `-pi/2` (`delta_q_foot ~ -0.15..-0.19 rad`), producing inward radial ground-reaction friction along the `0.1661 m` tibia (`~15 N * 0.166 m` -> up to `2.500 N·m`).
2. **Asymmetric load concentration**: Lifting `L2` alone dumps `Fz(L1) = 12.63 N` and `Fz(L3) = 15.80 N` (`60%` of body weight) onto the adjacent right-side legs, saturating `Foot L1/L3` even though 5 legs are down.
3. **Why walking policies avoid static saturation (`VERIFIED` behavior + `INFERRED` mechanism)**: Only `6 feet` and **symmetric `4 feet (lift L2, L5)`** avoid `2.50 N·m` saturation with frozen targets at `4–5 kg`. A learned dynamic tripod gait modulates joint targets away from frozen `-pi/2` to relieve horizontal internal constraint forces while still pushing `~35%` of Thigh steps `> 2.00 N·m` during 3-foot stance (`Chunk 1` spends `39.64%` in 3-foot support and `38.58%` in 4-foot support).

---

## 4. Stage 2 Terrain Calibration (`fab4649`, `OLD` vs `NEW`)

Comparing zero-shot terrain traversal across [`scripts/terrain_check.py`](file:///home/rithwik/SpdrBot-main/scripts/terrain_check.py) (`65` envs/terrain, `20.0s`, random headings) under Task 6's fixed **world-frame net-displacement metric (`NEW`)** vs the retired step-integrated metric (`OLD`, `VERIFIED`):

| Terrain | `2026-11-05/model_599.pt` (`NEW / OLD`) | Retracted `2026-11-10/model_2098.pt` (`NEW / OLD`) | `Probe A (2026-11-11_23-36-34/model_599.pt)` Target Surv / Prog (`NEW / OLD`) | `Probe A` Gate Target |
|---|---:|---:|---:|---|
| **`flat`** | `102.5% / 102.0%` | `32.2% / 58.3%` | `100.0%` / **`90.6%` (`90.0%` OLD)** | **PASS** |
| **`rough`** | `53.3% / 60.0%` | `8.7% / 43.9%` | `100.0%` / **`38.9%` (`46.1%` OLD)** (`23.1%` stuck) | **FAIL** |
| **`slope_up`** | `106.4% / 105.7%` | `31.6% / 57.1%` | `100.0%` / **`96.2%` (`91.2%` OLD)** | **PASS** |
| **`slope_down`** | `90.3% / 94.0%` | `28.2% / 56.6%` | `100.0%` / **`72.0%` (`77.9%` OLD)** | **PASS** |
| **`stairs_up`** | `98.1% / 97.6%` | `26.1% / 54.1%` | `100.0%` / **`80.1%` (`78.2%` OLD)** | **PASS** |
| **`stairs_down`** | `37.1% / 51.4%` | `17.0% / 46.2%` | `100.0%` / **`42.1%` (`53.1%` OLD)** (`46.2%` stuck) | **FAIL** |
| **`boxes`** | `63.1% / 69.0%` | `25.4% / 53.4%` | `100.0%` / **`57.0%` (`61.0%` OLD)** | **FAIL** |
| **Overall Deployment Gate** | `FAIL` (`rough`, `stairs_down`, `boxes`) | `100% FAIL` (**RETRACTED**) | **`FAIL`** (`4/7` terrain targets pass zero-shot) | **FAIL** |

- `Probe A` Terrain Check Slope Roll/Pitch (`VERIFIED`): `Roll mean=-0.19 deg (std=7.28 deg)`, `Pitch mean=+5.14 deg (std=6.25 deg)`; zero-shot skill suite (`standing 92.41% >=5ft FAIL`, `yaw +0.313 PASS`, `strafe +0.078 FAIL`, `backward -0.095 PASS`, `forward +0.123 PASS`).
- Stage 2 warm-start readiness: `2026-11-12_05-47-14/model_1499.pt` resolves all Stage 1 locomotion & symmetry blockers and is ready as the `--checkpoint` input for `Zoe-Hexapod-Terrain-v0` when Stage 2 begins; Stage 2 itself remains **`NOT READY`** until trained and gated.

---

## 5. Software-Test Policy Export (`28b83b8`) & Unit Tests

- **Export Command (`VERIFIED`)**:
  ```bash
  python scripts/rsl_rl/export.py --task Zoe-Hexapod-Flat-Deploy-v0 \
    --checkpoint /home/rithwik/SpdrBot-main/logs/rsl_rl/zoe_hexapod_deploy/2026-11-12_05-47-14/model_1499.pt \
    --out deploy/test_export_not_for_hardware --force
  ```
- **Stance Gate During Export (`VERIFIED`)**: `Mean Height = 0.1361 m >= 0.1275 m`, `Non-Foot Contacts = 0.00% < 1.0%` (**PASSED**).
- **Generated Artifacts in `deploy/test_export_not_for_hardware/` (`VERIFIED`)**:
  - [`deploy.yaml`](file:///home/rithwik/SpdrBot-main/deploy/test_export_not_for_hardware/deploy.yaml) (`3,501 B`): `schema_version: 1`, `status: NOT_FOR_HARDWARE_UNCALIBRATED`, `fitted_data_source: DATASHEET_ST3215_12V_ASSUMED_5KG_MASS`, `provenance.git_commit: 6e4a795f34392770db0b3ea341241346468815ca`.
  - `policy.onnx` (`799,254 B`), `policy_mlp.npz` (`800,670 B`), `reference_rollout.npz` (`168,754 B`, 500 steps).
  - Numerical equivalence verified across all `500` rollout steps (`ReferenceEvaluator` ONNX vs `reference_rollout.npz`: `9.54e-7`; NumPy MLP vs `reference_rollout.npz`: `9.54e-7`; ONNX vs NumPy MLP: `1.19e-6`).
- **Unit Tests (`VERIFIED`)**: `pytest tests/ -p no:launch_testing` -> **`34 passed`** across `test_actions.py`, `test_commands.py`, `test_curriculum.py`, `test_export.py`, `test_height_above_ground.py`, `test_params.py`, `test_train_hardening.py`.

---

## 6. Background Reference: Earlier Root-Cause Proof & Mass Tooling (Tasks 1–2, 7, 9)

*(Preserved from 2026-11-11 baseline analysis; details recorded cleanly in [`docs/EXISTING_STATE.md`](file:///home/rithwik/SpdrBot-main/docs/EXISTING_STATE.md) and [`docs/CHUNK_LOG.md`](file:///home/rithwik/SpdrBot-main/docs/CHUNK_LOG.md).)*

1. **Chunk-1 Freeze Root Cause (`f8d4a14`, `33d2e8c`, `06d4486`, `VERIFIED`)**: Tightening `torque_limit` from `-0.5 @ 2.10 N·m` to `-2.0 @ 1.95 N·m` while raising robot mass from `4.17 kg` to `5.00 kg` made 3-foot tripod walking pay `-147.30 /s` torque penalty (`+46.75 /s` Walk vs `+112.22 /s` 6-foot Creep at `0.15 m/s`), freezing `2026-11-09_22-33-44/model_1499.pt` at `0.0065 m/s`. Reverting only `torque_limit` in bisection `P1` jumped speed `24.5×` to `0.1595–0.1632 m/s`.
2. **`penalty_scale_ramp` Removal (`9662a60`, `VERIFIED`)**: Ramping penalty weights `0.1 -> 1.0 @ 800` iters (`P4`, `Probe 3`) stalled `Train/mean_reward` mid-run and held `noise_std` at `0.59–0.62`; disabling the ramp restored smooth convergence (`noise_std -> 0.25` in `Chunk 1`).
3. **Mass Randomization & Recalibration Tooling (`6087218`, `d3da62c`, `b28f803`, `VERIFIED`)**: Mass DR spans `[0.80, 1.10]` (`4.00–5.50 kg` around `5.000 kg`); [`scripts/set_payload_mass.py`](file:///home/rithwik/SpdrBot-main/scripts/set_payload_mass.py) syncs `hexa.urdf`, `deploy/mass_properties.yaml`, `deploy/asset_report.json`, and `hexa_physics.usd` once physical weighing occurs.
4. **Package & Repo Hygiene (`3fbbdf6`, `VERIFIED`)**: `spdrbot3.__file__` resolves to `/home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/__init__.py` (editable install), and `spdrbot3_direct_project` / obsolete USD/URDF assets were removed cleanly.
