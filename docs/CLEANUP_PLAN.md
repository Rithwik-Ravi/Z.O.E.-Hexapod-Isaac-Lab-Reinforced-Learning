# Repository Cleanup Plan (`docs/CLEANUP_PLAN.md`)

**Generated**: `2026-11-05`  
**Pre-Cleanup Git Tag**: `pre-cleanup`  
**Policy**: Two-stage cleanup.
- **Stage A (Executed Now — Zero Risk)**: Remove caches (`__pycache__`, `.pytest_cache`, `*.pyc`), remove empty directories (`deploy/test_run`, `deploy/2026-10-10_13-17-50`), delete untracked root duplicate `01_EXISTING_STATE.md` and `deploy/asset_report_new.json` (identical diff copy of `deploy/asset_report.json`), move one-off analysis/diagnostic scripts to `scripts/analysis/` and update references, update `.gitignore`, and mark `Zoe-Hexapod-Rough-v0` deprecated in `README.md`.
- **Stage B (Pending User `"approve cleanup"`)**: Delete invalidated checkpoints/runs under `logs/` and `outputs/`, legacy root USD/script files (`spdr.usd`, `spdr_stage.usd`, `spdrbot.py`, `spyderbot_test.py`, `check_*.py`), legacy 4-leg CAD directory `spyderbot_minimal URDF/`, and the 5.4 GB embedded duplicate repo `spdrbot3_direct_project/` (after ensuring `pip install -e source/spdrbot3` points to top-level `source/spdrbot3`).

---

## 1. File Inventory Outside `source/` and `tests/`

| Path | Size | Last Modified | Git Status | Referenced By | Verdict | Stage | Reason |
|---|---:|---|---|---|---|---|---|
| `.dockerignore` | 445 B | 2026-10-04 22:36 | Tracked | Docker build | **KEEP** | — | Standard container ignore config |
| `.flake8` | 861 B | 2026-10-04 22:36 | Tracked | Flake8 / pre-commit | **KEEP** | — | Linter configuration |
| `.gitignore` | 59 B → 342 B | 2026-11-05 20:55 | Tracked | Git | **KEEP** | Stage A (Updated) | Updated to ignore `logs/`, `outputs/`, `__pycache__/`, `.pytest_cache/`, `*.pyc`, and generated `deploy/*` artifacts except contract files |
| `.pre-commit-config.yaml` | 2,957 B | 2026-10-04 22:36 | Tracked | Pre-commit | **KEEP** | — | Formatting and lint hooks |
| `.vscode/.gitignore` | 215 B | 2026-10-04 22:36 | Tracked | Git | **KEEP** | — | IDE ignore rules |
| `.vscode/settings.json` | 104 B | 2026-10-07 14:50 | Untracked | VS Code | **KEEP** (Ignored) | — | Local IDE settings (ignored by `.vscode/.gitignore`) |
| `.vscode/tools/setup_vscode.py` | 9,942 B | 2026-10-04 22:36 | Tracked | Isaac Lab template | **KEEP** | — | Workspace IntelliSense helper |
| `01_EXISTING_STATE.md` | 15,264 B | 2026-10-05 12:27 | Untracked | None | **DELETE** | **Stage A** | Untracked stale root duplicate of `docs/EXISTING_STATE.md` (lines 1–167) |
| `README.md` | 15,055 B | 2026-11-05 20:51 | Tracked | GitHub / Users | **KEEP** | Stage A (Updated) | Added deprecation banner for `Zoe-Hexapod-Rough-v0` pointing to `Zoe-Hexapod-Flat-Deploy-v0` |
| `SpdrBot v12.png` | 320,221 B | 2026-10-04 22:35 | Tracked | `README.md` | **KEEP** | — | Project image referenced in `README.md` |
| `check_cam.py` | 1,348 B | 2026-10-05 00:46 | Tracked | None | **DELETE** | Stage B | Dead one-off debug script with hard-coded paths |
| `check_mdp.py` | 515 B | 2026-10-04 22:36 | Tracked | `docs/EXISTING_STATE.md` | **DELETE** | Stage B | Dead one-off debug script from initial commit |
| `check_model.py` | 736 B | 2026-10-05 00:42 | Tracked | `docs/EXISTING_STATE.md` | **DELETE** | Stage B | Dead one-off checkpoint NaN checker from initial commit |
| `check_tb.py` | 852 B | 2026-10-05 19:07 | Tracked | `docs/EXISTING_STATE.md` | **DELETE** | Stage B | Dead one-off TensorBoard reader with hard-coded paths |
| `spdr.usd` | 10.24 MB | 2026-10-04 22:35 | Tracked | `spyderbot_test.py` | **DELETE** | Stage B | Legacy 4-leg spider robot USD; not loaded by any `spdrbot3` task |
| `spdr_stage.usd` | 10.23 MB | 2026-10-04 22:35 | Tracked | None | **DELETE** | Stage B | Legacy 4-leg spider stage USD; unused |
| `spdrbot.py` | 2,681 B | 2026-10-04 22:35 | Tracked | None | **DELETE** | Stage B | Stale root copy of `source/spdrbot3/spdrbot3/assets/spdrbot.py` |
| `spyderbot_test.py` | 14,268 B | 2026-10-05 00:54 | Tracked | None | **DELETE** | Stage B | Standalone script for legacy 4-leg `spdr.usd` |
| `spyderbot_minimal URDF/` (21 files) | 6.90 MB | 2026-10-04 22:35 | Tracked | `README.md` | **DELETE** | Stage B | Legacy 4-leg `SpdrBot_description` xacro/meshes (`arm_a/b/c_1..4.stl`); unrelated to 6-leg Z.O.E. hexapod |
| `spdrbot3_direct_project/` | 5.40 GB | 2026-10-05 19:04 | Tracked (Subproject) | `check_tb.py` | **DELETE** | Stage B | Nested duplicate repo holding 5.4 GB of May 2026 `cartpole_direct` / `zoe_hexapod_rough` logs (re-run `pip install -e source/spdrbot3` before removal) |
| `deploy/actuator_model.yaml` | 883 B | 2026-11-05 19:51 | Tracked | `params.py`, `export.py`, `tests/` | **KEEP** | — | Authoritative ST3215 12V actuator contract |
| `deploy/asset_report.json` | 12,970 B | 2026-11-05 17:11 | Tracked | `params.py`, `audit_asset.py`, `export.py`, `tests/` | **KEEP** | — | Authoritative audited geometry/joint/mass report for `hexapod/hexa/hexa.usd` |
| `deploy/asset_report_new.json` | 12,970 B | 2026-11-05 17:09 | Untracked | None | **DELETE** | **Stage A** | Temporary diff output from Step 1 (byte-for-byte identical to `deploy/asset_report.json`) |
| `deploy/mass_properties.yaml` | 763 B | 2026-11-05 19:51 | Tracked | `tests/test_params.py`, `docs/ASSUMPTIONS.md` | **KEEP** | — | Authoritative mass model metadata and CAD link breakdown |
| `deploy/test_export_not_for_hardware/deploy.yaml` | 3,500 B | 2026-11-05 20:46 | Tracked | `export.py`, `docs/` | **KEEP** | — | Software test export contract template (`status: NOT_FOR_HARDWARE_UNCALIBRATED`) |
| `deploy/test_export_not_for_hardware/*.{onnx,npz}` | 1.77 MB | 2026-11-05 20:47 | Untracked (Ignored) | `export.py` | **KEEP** (Ignored) | — | Software test export binary artifacts (ignored via `.gitignore`) |
| `deploy/2026-10-10_13-17-50/` | 0 B | 2026-10-10 13:45 | Untracked | None | **DELETE** | **Stage A** | Empty directory |
| `deploy/test_run/` | 0 B | 2026-10-10 14:19 | Untracked | None | **DELETE** | **Stage A** | Empty directory |
| `docs/ASSUMPTIONS.md` | 4,857 B | 2026-11-05 19:51 | Tracked | `README.md`, `docs/EXISTING_STATE.md` | **KEEP** | — | Authoritative owner hardware facts & explicit assumptions |
| `docs/EXISTING_STATE.md` | 25,161 B | 2026-11-05 20:51 | Tracked | `README.md`, `docs/` | **KEEP** | — | Living architecture and state record |
| `docs/CHUNK_LOG.md` | 28,763 B | 2026-11-05 20:35 | Tracked | `docs/EXISTING_STATE.md` | **KEEP** | — | Gate evaluation log across all probes |
| `docs/TRAINING_RECIPE.md` | 5,053 B | 2026-10-05 19:06 | Tracked | `README.md`, `docs/EXISTING_STATE.md` | **KEEP** | — | Training, evaluation, and export commands |
| `docs/AUDIT_A_REPORT.md` | 17,110 B | 2026-10-06 00:12 | Tracked | `docs/EXISTING_STATE.md` | **ARCHIVE** | — | Historical initial audit report |
| `docs/DIAG_STANCE_REPORT.md` | 12,706 B | 2026-10-10 14:23 | Tracked | `docs/EXISTING_STATE.md` | **ARCHIVE** | — | Historical Phase 1 stance diagnosis |
| `docs/STAGE1_READINESS_REPORT.md` | 20,522 B | 2026-10-12 17:46 | Tracked | `docs/EXISTING_STATE.md` | **ARCHIVE** | — | Historical Stage 1 readiness report on old asset |
| `docs/CLEANUP_PLAN.md` | — | 2026-11-05 20:55 | Tracked | `docs/EXISTING_STATE.md` | **KEEP** | — | This cleanup inventory and execution plan |
| `hexapod/hexa.urdf` | 30,241 B | 2026-11-04 22:40 | Tracked | `scripts/audit_asset.py` | **KEEP** | — | Active `hexa_ir` URDF (`DCAM` commented out) |
| `hexapod/hexa.sdf` | 34,897 B | 2026-09-23 03:11 | Tracked | `hexapod/model.config` | **KEEP** | — | Companion Gazebo SDF from `hexa_ir` package |
| `hexapod/model.config` | 248 B | 2026-09-23 03:11 | Tracked | Gazebo | **KEEP** | — | Companion model metadata in `hexa_ir` |
| `hexapod/hexa/hexa.usd` | 1,591 B | 2026-11-04 22:43 | Tracked | `zoe_hexapod.py`, `spdrbot.py`, `audit_asset.py` | **KEEP** | — | Active top-level simulation USD loaded by Isaac Lab |
| `hexapod/hexa/configuration/hexa_base.usd` | 23.62 MB | 2026-11-04 22:43 | Tracked | `hexapod/hexa/hexa.usd` | **KEEP** | — | Active USD geometry & rigid body layer |
| `hexapod/hexa/configuration/hexa_physics.usd` | 7,511 B | 2026-11-04 22:43 | Tracked | `hexapod/hexa/hexa.usd` | **KEEP** | — | Active USD PhysX articulation/joint layer |
| `hexapod/hexa/configuration/hexa_sensor.usd` | 644 B | 2026-11-04 22:43 | Tracked | `hexapod/hexa/hexa.usd` | **KEEP** | — | Active USD sensor layer |
| `hexapod/hexa/config.yaml` & `.asset_hash` | 1,674 B | 2026-11-04 22:43 | Tracked | URDF importer | **KEEP** | — | Isaac Lab URDF-to-USD converter provenance |
| `hexapod/meshes/base_link.stl`, `NCCUP1..6.stl`, `NTHIGH1..6.stl`, `NFOOT1..6.stl`, `DCAM.stl` | 19.72 MB | 2026-09-23 03:11 | Tracked | `hexapod/hexa.urdf` | **KEEP** | — | Active STL meshes for `hexa_ir` (`DCAM.stl` kept on disk for URDF reference; wait for Stage B before any mesh deletion) |
| `outputs/` (39 Hydra timestamp dirs) | 2.70 MB | 2026-11-05 19:51 | Untracked (Ignored) | Hydra | **DELETE** | Stage B | Auto-generated Hydra run logs (ignored via `.gitignore`) |
| `scripts/audit_asset.py` | 23,365 B | 2026-11-05 19:51 | Tracked | `params.py` (via JSON), `docs/` | **KEEP** | — | Core M0 USD/URDF asset auditor |
| `scripts/chunk_check.py` | 20,331 B | 2026-11-05 19:51 | Tracked | `docs/CHUNK_LOG.md`, `docs/TRAINING_RECIPE.md` | **KEEP** | — | Core 3-condition parallel gate evaluation suite |
| `scripts/eval_gait_report.py` | 8,673 B | 2026-11-05 19:51 | Tracked | `docs/CHUNK_LOG.md`, `docs/TRAINING_RECIPE.md` | **KEEP** | — | Core single-env tripod alternation & swing clearance verifier |
| `scripts/eval_robustness.py` | 5,535 B | 2026-10-05 19:05 | Tracked | `docs/TRAINING_RECIPE.md` | **KEEP** | — | Core M5 push/friction/mass robustness evaluator |
| `scripts/fit_actuator_sim.py` | 4,330 B | 2026-10-05 14:17 | Tracked | `docs/TRAINING_RECIPE.md` | **KEEP** | — | Core M1 hardware-to-sim actuator system identification script |
| `scripts/verify_reward_budgets.py` | 5,354 B | 2026-11-05 19:14 | Tracked | `docs/EXISTING_STATE.md` | **KEEP** | — | Analytical reward budget verification script |
| `scripts/list_envs.py`, `random_agent.py`, `zero_agent.py` | 6,453 B | 2026-10-05 00:49 | Tracked | Isaac Lab CLI | **KEEP** | — | Standard Isaac Lab environment utilities |
| `scripts/rsl_rl/train.py`, `play.py`, `export.py`, `cli_args.py` | 34,406 B | 2026-11-05 20:45 | Tracked | Primary training/eval/export CLI | **KEEP** | — | Core RSL-RL entry points |
| `scripts/skrl/train.py`, `play.py` | 15,914 B | 2026-10-04 22:36 | Tracked | Template | **KEEP** | — | Standard Isaac Lab SKRL launchers |
| `scripts/sweep_default_pose.py` | 12,958 B | 2026-11-05 19:51 | Tracked | `docs/EXISTING_STATE.md` | **ARCHIVE** (`scripts/analysis/`) | **Stage A** | Moved to `scripts/analysis/sweep_default_pose.py` |
| `scripts/apply_5kg_mass_model.py` | 3,060 B | 2026-10-11 14:15 | Tracked | `docs/EXISTING_STATE.md` | **ARCHIVE** (`scripts/analysis/`) | **Stage A** | Moved to `scripts/analysis/apply_5kg_mass_model.py` |
| `scripts/eval_honest_gate.py` | 9,507 B | 2026-10-10 15:00 | Tracked | `docs/EXISTING_STATE.md` | **ARCHIVE** (`scripts/analysis/`) | **Stage A** | Superseded by `scripts/chunk_check.py`; moved to `scripts/analysis/` |
| `scripts/diagnose_offline.py` | 4,452 B | 2026-10-09 23:59 | Tracked | `docs/EXISTING_STATE.md` | **ARCHIVE** (`scripts/analysis/`) | **Stage A** | One-off Phase 1 diagnostic; moved to `scripts/analysis/` |
| `scripts/diagnose_probe.py` | 3,274 B | 2026-10-10 13:07 | Tracked | None | **ARCHIVE** (`scripts/analysis/`) | **Stage A** | One-off probe diagnostic; moved to `scripts/analysis/` |
| `scripts/diagnose_stance.py` | 8,669 B | 2026-10-10 00:14 | Tracked | `docs/EXISTING_STATE.md` | **ARCHIVE** (`scripts/analysis/`) | **Stage A** | One-off Phase 1 stance diagnostic; moved to `scripts/analysis/` |
| `scripts/inspect_stance_geometry.py` | 4,665 B | 2026-10-10 14:43 | Tracked | `docs/EXISTING_STATE.md` | **ARCHIVE** (`scripts/analysis/`) | **Stage A** | Superseded by `scripts/audit_asset.py`; moved to `scripts/analysis/` |
| `scripts/run_clean_stance_comparison.py` | 3,785 B | 2026-10-10 00:34 | Tracked | `docs/EXISTING_STATE.md` | **ARCHIVE** (`scripts/analysis/`) | **Stage A** | One-off stance comparison script; moved to `scripts/analysis/` |
| `scripts/run_grid_unbuffered.py` | 1,150 B | 2026-10-10 00:24 | Tracked | Calls `run_single_grid_case.py` | **ARCHIVE** (`scripts/analysis/`) | **Stage A** | One-off grid runner; moved to `scripts/analysis/` |
| `scripts/run_single_grid_case.py` | 3,668 B | 2026-10-10 00:19 | Tracked | `run_grid_unbuffered.py` | **ARCHIVE** (`scripts/analysis/`) | **Stage A** | One-off grid helper; moved to `scripts/analysis/` |
| `scripts/rsl_rl/stance_gate.py` | 5,707 B | 2026-10-10 14:18 | Tracked | `docs/DIAG_STANCE_REPORT.md` | **ARCHIVE** (`scripts/analysis/`) | **Stage A** | Superseded by `scripts/chunk_check.py` and inlined gate in `export.py`; moved to `scripts/analysis/` |

---

## 2. Training Run Inventory Under `logs/rsl_rl/` (Total: 2,438.48 MB across 45 runs)

All runs prior to `2026-11-04_22-43` were trained on the old robot USD (`base_link_v1`, leg order `[1, 2, 6, 5, 4, 3]`, default pose `0, 0, 0`) or intermediate uncalibrated configs and are **INVALID** for the current `hexa_ir` robot (`base_link`, leg order `[6, 1, 5, 4, 3, 2]`, default pose `0, 0, -pi/2`).

### 2.1 `logs/rsl_rl/zoe_hexapod_deploy/` (30 runs, 2,296.92 MB)

| Run Directory | Size (MB) | Last Modified | Checkpoints | Asset & Gate Result | Stage B Proposal |
|---|---:|---|---:|---|---|
| `2026-10-05_18-00-49` | 9.95 | 2026-10-05 18:00 | 2 (`model_2.pt`) | Old asset (`base_link_v1`), initial smoke test | **DELETE** (Invalid asset) |
| `2026-10-05_20-21-46` | 5.02 | 2026-10-05 20:21 | 1 (`model_0.pt`) | Old asset, aborted init | **DELETE** (Invalid asset) |
| `2026-10-05_22-09-38` | 5.01 | 2026-10-05 22:09 | 1 (`model_0.pt`) | Old asset, aborted init | **DELETE** (Invalid asset) |
| `2026-10-05_22-48-28` | 5.02 | 2026-10-05 22:48 | 1 (`model_0.pt`) | Old asset, aborted init | **DELETE** (Invalid asset) |
| `2026-10-05_23-14-03` | 5.03 | 2026-10-05 23:14 | 1 (`model_0.pt`) | Old asset, aborted init | **DELETE** (Invalid asset) |
| `2026-10-05_23-27-40` | 5.05 | 2026-10-05 23:27 | 1 (`model_0.pt`) | Old asset, aborted init | **DELETE** (Invalid asset) |
| `2026-10-06_20-23-09` | 9.96 | 2026-10-06 20:23 | 2 (`model_2.pt`) | Old asset, smoke test | **DELETE** (Invalid asset) |
| `2026-10-08_21-40-56` | 146.36 | 2026-10-08 22:26 | 29 (`model_950.pt`) | Old asset, belly-dragging pre-Phase-2 | **DELETE** (Invalid asset) |
| `2026-10-09_00-29-35` | 1014.42 | 2026-10-09 05:09 | 201 (`model_9999.pt`) | Old asset, 10k iter belly-dragging run (1.01 GB) | **DELETE** (Invalid asset) |
| `2026-10-10_11-16-57` | 35.41 | 2026-10-10 11:34 | 7 (`model_50.pt`) | Old asset, early probe | **DELETE** (Invalid asset) |
| `2026-10-10_12-16-54` | 35.41 | 2026-10-10 12:40 | 7 (`model_50.pt`) | Old asset, early probe | **DELETE** (Invalid asset) |
| `2026-10-10_13-17-50` | 35.41 | 2026-10-10 13:41 | 7 (`model_50.pt`) | Old asset, early probe | **DELETE** (Invalid asset) |
| `2026-10-10_19-01-52` | 35.50 | 2026-10-10 19:23 | 7 (`model_50.pt`) | Old asset, Phase 2 300-iter probe | **DELETE** (Invalid asset) |
| `2026-10-10_21-53-14` | 35.62 | 2026-10-10 22:01 | 7 (`model_50.pt`) | Old asset, Phase 2 probe | **DELETE** (Invalid asset) |
| `2026-10-11_19-43-04` | 66.30 | 2026-10-11 19:58 | 13 (`model_599.pt`) | Old asset (5kg, 1.9 N·m) — Probe 1 (FAIL: torque sat) | **DELETE** (Invalid asset) |
| `2026-10-12_10-44-37` | 66.30 | 2026-10-12 11:00 | 13 (`model_599.pt`) | Old asset (5kg, 1.9 N·m) — Probe 2 (FAIL: torque sat) | **DELETE** (Invalid asset) |
| `2026-10-12_11-13-42` | 66.30 | 2026-10-12 11:29 | 13 (`model_599.pt`) | Old asset (5kg, 2.5 N·m) — Probe 3 (FAIL: stability/torque) | **DELETE** (Invalid asset) |
| `2026-10-12_12-21-40` | 66.30 | 2026-10-12 12:35 | 13 (`model_599.pt`) | Old asset (5kg, 2.5 N·m) — Probe 4 (FAIL: stability/torque) | **DELETE** (Invalid asset) |
| `2026-10-12_13-07-38` | 66.30 | 2026-10-12 13:22 | 13 (`model_599.pt`) | Old asset (5kg, 2.5 N·m) — Probe 5 (FAIL: stability `-0.0142m`) | **DELETE** (Invalid asset) |
| `2026-10-12_15-37-46` | 66.30 | 2026-10-12 15:52 | 13 (`model_599.pt`) | Old asset (5kg, 2.5 N·m) — **Probe 6 Reference** (`0.1514 m/s`, `p5 -0.0032m`) | **ARCHIVE / DELETE in Stage B** (Invalid for new asset, historical reference only) |
| `2026-11-04_11-23-22` | 10.16 | 2026-11-04 11:23 | 2 (`model_1.pt`) | Aborted GUI test before URDF fix | **DELETE** |
| `2026-11-04_18-40-35` | 10.16 | 2026-11-04 18:40 | 2 (`model_1.pt`) | Aborted GUI test on `hexa_sld` (flipped orientation) | **DELETE** |
| `2026-11-04_19-16-43` | 193.89 | 2026-11-04 20:13 | 38 (`model_950.pt`) | Trained on `hexa_sld` (flipped orientation, deleted asset) | **DELETE** (Invalid asset) |
| `2026-11-04_21-31-53` | 5.23 | 2026-11-04 21:32 | 1 (`model_0.pt`) | Aborted test | **DELETE** |
| `2026-11-04_21-41-34` | 10.17 | 2026-11-04 21:41 | 2 (`model_1.pt`) | Smoke test on `hexa_ir` with `DCAM` active | **DELETE** (Has `DCAM` link) |
| `2026-11-04_21-43-00` | 193.93 | 2026-11-04 22:38 | 38 (`model_950.pt`) | Trained on `hexa_ir` with `DCAM` active & pre-audit constants | **DELETE** (Has `DCAM` link) |
| `2026-11-04_22-43-38` | 10.17 | 2026-11-04 22:43 | 2 (`model_1.pt`) | 2-iter USD conversion smoke test | **DELETE** |
| `2026-11-04_22-46-40` | 5.30 | 2026-11-04 22:46 | 1 (`model_0.pt`) | 1-iter GUI launch verification | **DELETE** |
| `2026-11-05_19-09-47` | 10.39 | 2026-11-05 19:11 | 2 (`model_49.pt`) | **VALID (`hexa_ir` no-DCAM)** — Step 4(e) 256-env 50-iter smoke test (PASS) | **KEEP** |
| `2026-11-05_19-51-02` | 66.55 | 2026-11-05 20:21 | 13 (`model_599.pt`) | **VALID (`hexa_ir` no-DCAM)** — Step 4(f) 1024-env 600-iter probe (**12/15 PASS**, stability `+0.0472m`, speed `0.1518 m/s`, tripod `r = -0.8551`) | **KEEP** (Active baseline checkpoint) |

### 2.2 `logs/rsl_rl/zoe_hexapod_terrain/`

| Run Directory | Size (MB) | Last Modified | Checkpoints | Asset & Gate Result | Stage B Proposal |
|---|---:|---|---:|---|---|
| `2026-10-05_18-04-12` to `2026-10-13_13-25-02` (all 15 runs) | 141.56 | 2026-10-05 to 2026-10-13 | 1–12 ckpts each | Old asset (`base_link_v1`), early Stage 2 terrain smoke tests | **DELETE** (Invalid asset) |
| `2026-11-06_22-18-04` | 64.20 | 2026-11-06 22:39 | 13 (`model_600.pt`) | **`INVALID - DO NOT USE`** (`hexa_ir`, cold-started with world-Z height reward bug, 100% rightward flip; TB events archived to `logs/_archive_tb/`) | **DELETE in Stage B** |
| `2026-11-07_20-39-49` | 63.80 | 2026-11-07 21:01 | 13 (`model_1199.pt`) | **VALID (`hexa_ir` no-DCAM)** — B3 Probe 1 (100% survival, pre-`ZoeTerrainCommandsCfg`) | **KEEP** (Diagnostic reference) |
| `2026-11-07_22-49-47` | 63.80 | 2026-11-07 23:12 | 13 (`model_1198.pt`) | **VALID (`hexa_ir` no-DCAM)** — B3 Probe 2 (**100% multi-terrain survival across all 7 terrains & L0–9**, `Terrain/Mean_Level = 8.44`, **13/15 Stage 1 flat gate PASS**) | **KEEP** (Active Stage 2 baseline checkpoint) |

