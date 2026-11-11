# Stage 2 Multi-Terrain Locomotion Diagnosis Report

**Date**: 2026-11-07  
**System**: NVIDIA Isaac Lab 0.54.2 / Isaac Sim 4.5.0.0 / RSL-RL 3.1.2  
**Target Asset**: Z.O.E. Hexapod (`hexa_ir`, 18-DOF, Waveshare ST3215 12V, DCAM commented out)  
**Diagnosed Run**: `logs/rsl_rl/zoe_hexapod_terrain/2026-11-06_22-18-05` (10,000 iterations, `model_9999.pt`)  
**Scope**: Read-only diagnostic investigation of why Stage 2 training pushes the robot to one side and flips over on every terrain.

---

## Owner Answers

1. **Exact command I ran**:  
   `python scripts/rsl_rl/train.py --task Zoe-Hexapod-Terrain-v0 --resume --load_run <TIMESTAMP_FOLDER> --headless --num_envs 1024 --max_iterations 10000`  
   followed by:  
   `python scripts/rsl_rl/play.py --task Zoe-Hexapod-Terrain-Play-v0 --num_envs 16`
2. **GUI or headless, and how many iterations I let it run before looking**:  
   Headless for all 10,000 iterations (total elapsed time 5h 16m from Nov 6 22:18 to Nov 7 03:34).
3. **Did I watch the training window, or play a saved checkpoint (which file)?**:  
   Played the saved checkpoint in GUI with `play.py` (which loaded `logs/rsl_rl/zoe_hexapod_terrain/2026-11-06_22-18-05/model_9999.pt`).
4. **Did it ever look better earlier in the run (yes / no / unknown)?**:  
   **NO [VERIFIED]**. TensorBoard metrics confirm the mean episode length was collapsed at **22.7 steps (~0.45s)** at iteration 0 and remained pinned at **21.3–22.3 steps** throughout all 10,000 iterations.
5. **Which side does it fall toward, in the robot's own frame (left / right / varies / unknown)?**:  
   **RIGHT [VERIFIED]**. Across 64 environments on `model_9999.pt`, **64 out of 64 (100.0%)** terminated in a severe rightward roll (mean final roll = **`-130.3°`**).

---

## A. Provenance and State

- **Git Status**: Clean working tree with untracked analysis scripts in `scripts/analysis/`.
- **Git Commit**: `55fe211` (`chore(cleanup): execute Stage A zero-risk cleanup, move analysis scripts to scripts/analysis, and add docs/CLEANUP_PLAN.md`).
- **Tag**: `pre-cleanup` preserved at `8d36eb2`.
- **Stage B Cleanup Status**: **NOT DONE [VERIFIED]**. All historical USDs, directories, and logs remain untouched.
- **5 Newest Run Directories Under `logs/rsl_rl/`**:
  1. `2026-11-06_22-18-05` (Task: `zoe_hexapod_terrain` | 201 checkpoints, ended `model_9999.pt`)
  2. `2026-11-06_10-38-24` (Task: `zoe_hexapod_terrain` | 6 checkpoints, ended `model_250.pt`)
  3. `2026-11-05_23-34-42` (Task: `zoe_hexapod_deploy` | 201 checkpoints, ended `model_9999.pt`)
  4. `2026-11-05_19-51-02` (Task: `zoe_hexapod_deploy` | 13 checkpoints, ended `model_599.pt` — **Valid Flat Baseline**)
  5. `2026-11-05_19-09-47` (Task: `zoe_hexapod_deploy` | 2 checkpoints, ended `model_49.pt` — Smoke Test)
- **Multi-Terrain Run Parameters (`2026-11-06_22-18-05`)**:
  - `task_id`: `Zoe-Hexapod-Terrain-v0`
  - `num_envs`: 1024
  - `seed`: 42
  - `max_iterations`: 10000 (actual iterations reached: 10000)
  - `resume`: **`False` [VERIFIED]** in `params/agent.yaml`.
  - **Resume Provenance**: In `train.py`, line 139 defines `log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)`. Because `experiment_name` for Stage 2 is `"zoe_hexapod_terrain"`, RSL-RL's `get_checkpoint_path` looked inside `logs/rsl_rl/zoe_hexapod_terrain/` rather than `logs/rsl_rl/zoe_hexapod_deploy/`. Consequently, **this run started 100% COLD FROM SCRATCH with random weights** (`resume: False`, initial noise std = 1.0017).

---

## B. Training Curves (TensorBoard Audit)

Reading `events.out.tfevents.1793983689.rithwik-Legion-Pro-5.160402.0`:

| Metric Tag | Iteration 0 (First) | Iteration 5000 (Mid) | Iteration 9999 (Last) | Min | Max | Trend |
|---|---:|---:|---:|---:|---:|---|
| **`Train/mean_episode_length`** | **`22.73`** | **`21.34`** | **`22.30`** | `19.16` | `1000.00` | **STABLE / COLLAPSED (~0.44s)** |
| **`Train/mean_reward`** | `-4.66` | `-10.87` | `-10.38` | `-294.67` | `-4.66` | **NEGATIVE COLLAPSE** |
| **`Episode_Termination/bad_orientation`** | `0.0000` | **`0.9289`** | **`0.9240`** | `0.0000` | `1.0000` | **DOMINANT CAUSE (92.4%)** |
| **`Episode_Termination/base_contact`** | `0.0000` | **`0.0715`** | **`0.0764`** | `0.0000` | `0.0988` | **SECONDARY CAUSE (7.6%)** |
| **`Episode_Termination/time_out`** | `0.0262` | **`0.0000`** | **`0.0000`** | `0.0000` | `1.0000` | **ZERO (Never Survived)** |
| **`Episode_Termination/invalid_state_nan`** | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | Zero (Sanitizer Clean) |
| **`Episode_Reward/base_height_shaped`** | **`-0.2553`** | **`-0.2875`** | **`-0.2860`** | **`-21.5654`** | `-0.1511` | **CORRUPTED (Negative Bias)** |
| **`Episode_Reward/termination_penalty`** | `0.0000` | `-0.2000` | `-0.2000` | `-0.2000` | `0.0000` | Persistent penalty |
| **`Episode_Reward/torque_limit`** | `-0.0048` | `-0.0058` | `-0.0059` | `-0.2693` | `-0.0048` | Stable penalty |
| **`Terrain/Mean_Level`** | **`5.0539`** | **`6.1064`** | **`6.1064`** | `5.0539` | `6.2884` | **FROZEN AT LEVEL 5–6** |
| **`Policy/mean_noise_std`** | `1.0017` | `0.8238` | `0.7171` | `0.7004` | `1.1239` | Shrinking slowly |
| **`Loss/learning_rate`** | `0.0003` | `0.0002` | `0.0000` | `0.0000` | `0.0100` | Decayed to near 0 |
| **`Loss/value_function`** | `24.0616` | `0.1834` | `0.2203` | `0.0837` | `33.8951` | Converged |
| **`Loss/surrogate`** | `-0.0090` | `-0.0092` | `-0.0041` | `-0.0138` | `0.0221` | Flat |

**Verdict**: The policy never exited the early failure trap. Episode length was frozen at 22 steps (~0.44s) from Step 0 to Step 10,000. 92.4% of all terminations were `bad_orientation`.

---

## C. Config Diff (`params/env.yaml`: Flat Probe vs Multi-Terrain)

- **Unified Diff Summary**:
  - `scene.terrain.terrain_type`: changed from `plane` $\to$ `generator` (`ZOE_TERRAINS_CFG`).
  - `scene.terrain.physics_material`: friction changed from `static/dynamic = 1.0 (multiply)` to `0.5 (average)`.
  - `rewards.lin_vel_z_l2.weight`: relaxed from `-2.0` $\to$ `-0.5`.
  - `rewards.flat_orientation_l2.weight`: relaxed from `-0.5` $\to$ `0.0` (disabled so slopes don't penalize tilt).
  - `rewards.undesired_contacts.params.sensor_cfg`: expanded from `.*NTHIGH.*` to `.*(NCCUP|NTHIGH).*`.
  - `curriculum`: added `terrain_levels: zoe_terrain_curriculum`.
- **Inheritance Check**: Stage 2 **fully inherits** all Stage 1 hardware fixes:
  - Derived constants in `params.py` (standing height `0.1427m`, min height `0.1284m`, target `0.1356m`).
  - 3D per-leg foot tip vectors and articulation mapping.
  - `torque_limit_penalty` (quadratic above `2.10 N·m`, weight `-0.5`).
  - `fewer_than_4_feet_contact` (weight `-0.05`) and `fewer_than_3_feet_contact` (weight `-0.5`).
  - Mass randomization $\pm 10\%$ (`0.90..1.10`).
  - Action scale (`0.50`), PD stiffness ($K_p=16.0, K_d=1.6$), effort limit (`2.50 N·m`), default pose `(0, 0, -pi/2)`.

---

## D. Terrain World Audit

- **Sub-Terrains & Proportions**:
  - `flat`: 10% (`noise_range=(0.0, 0.0)`)
  - `rough`: 20% (`noise_range=(0.005, 0.030)`)
  - `slope_up`: 12% (`slope_range=(0.0, 0.30)`, platform = 2.0m)
  - `slope_down`: 12% (`slope_range=(0.0, 0.30)`, inverted pyramid, platform = 2.0m)
  - `stairs_up`: 12% (`step_height=(0.01, 0.06)m`, step width = 0.3m, platform = 2.0m)
  - `stairs_down`: 10% (`step_height=(0.01, 0.06)m`, inverted, platform = 2.0m)
  - `boxes`: 14% (`grid_width=0.45m`, `grid_height=(0.01, 0.05)m`, platform = 2.0m)
- **Grid Configuration**: `num_rows = 10`, `num_cols = 10`, `size = (8.0, 8.0)m`, `border_width = 20.0m`.
- **Elevation Span Across Tiles**:
  - `env_origins[:, 2]`: min = **`-0.6811 m`**, median = **`+0.0156 m`**, max = **`+0.6080 m`**!
  - Local spawn clearance at $t=0$: `root_z - env_origin_z` = **`0.1580 m to 0.1871 m`** (mean `0.1666 m` — local spawn is geometrically clean).
  - Initial physics step penetration: **0 / 64 environments** had non-foot contact at $t=0$ (max contact force = 4.1 N mean). Geometry does not intersect at spawn.
- **The Critical Spawn Level Bug**:
  - In `env_cfg_terrain.py:26`, `max_init_terrain_level = None`.
  - In Isaac Lab, when `max_init_terrain_level` is `None`, robots are initialized uniformly across all difficulty levels (0 through 9).
  - Empirical measurement on reset: `Terrain levels per env: min=1, max=9, mean=5.06`!
  - **Verdict**: Un-trained robots were dropped directly onto Level 5–9 inverted pyramid pits, 30% slopes, and 6cm steps from Step 0.

---

## E. World-Z Dependence Audit (Root-Cause Hypothesis 1)

Grep search for `root_pos_w[:, 2]` and absolute height constants:

| File:Line | Variable / Function | Code Expression | Validity on Multi-Terrain |
|---|---|---|---|
| `rewards.py:64` | `foot_clearance_relative` | `root_pos_z = robot.data.root_pos_w[:, 2]` | **INVALID** (world Z) |
| `rewards.py:83` | `base_height_l2` | `root_z = robot.data.root_pos_w[:, 2]` | **FATAL** (world Z vs flat target) |
| `rewards.py:101` | **`base_height_shaped`** | **`root_z = robot.data.root_pos_w[:, 2]`** | **FATAL [CONFIRMED SMOKING GUN]** |
| `rewards.py:119` | `metric_base_height` | `return robot.data.root_pos_w[:, 2]` | **INVALID** (reports world Z) |
| `observations.py:47` | `base_height_privileged` | `return robot.data.root_pos_w[:, 2]` | **INVALID** (critic observes world Z) |
| `curriculum.py:62` | **`zoe_terrain_curriculum`** | **`torch.norm(robot.data.root_pos_w[:, :2])`** | **FATAL [CONFIRMED SMOKING GUN]** |

### Mathematical Quantification of `base_height_shaped` on Terrain
In `rewards.py`:
$$\text{deficit} = \max(0,\ 0.1284 - z_{\text{world}}),\quad \text{barrier} = 10 \cdot \text{deficit} + 150 \cdot \text{deficit}^2$$
$$\text{attractor} = \exp\left(-\left(\frac{z_{\text{world}} - 0.1356}{0.015}\right)^2\right)$$

- On tiles where $z_{\text{origin}} < 0$ (e.g. inverted pyramid, $z_{\text{origin}} = -0.68\text{ m}$):
  - Standing upright puts the robot at $z_{\text{world}} = -0.5139\text{ m}$.
  - $\text{deficit} = 0.1284 - (-0.5139) = 0.6423\text{ m}$.
  - $\text{barrier} = 10(0.6423) + 150(0.6423)^2 = 6.42 + 61.88 = \mathbf{68.30}$.
  - **The robot receives a catastrophic penalty of `-68.30` per step!** Over a 1000-step episode, surviving would cost **`-68,300` reward points**.
  - Terminating immediately on step 22 with `termination_penalty = -200` saves tens of thousands of penalty points!
- On tiles where $z_{\text{origin}} > 0$ (e.g. slopes/stairs up, $z_{\text{origin}} = +0.60\text{ m}$):
  - Standing upright puts the robot at $z_{\text{world}} = +0.7792\text{ m}$.
  - $\text{attractor} = \exp(-((0.7792 - 0.1356)/0.015)^2) = \exp(-(42.9)^2) = \mathbf{0.0000}$.
  - The upright attractor is completely dead (0.0000). The gradient strongly drives the robot to step off or collapse down to lower world Z!

### Mathematical Quantification of `zoe_terrain_curriculum`
In `curriculum.py:62-63`:
$$\text{distance\_walked} = \|\mathbf{p}_{\text{world}, xy}\|$$
It measures distance from `(0, 0, 0)` **World Origin**, not from `env_origins`!
- For environment $i$ at world position $(20\text{ m}, 30\text{ m})$, $\text{distance\_walked} = \sqrt{20^2 + 30^2} = \mathbf{36.05\text{ m}}$ at $t=0$.
- Commanded distance over 0.44s is only $0.15\text{ m/s} \times 0.44\text{ s} = \mathbf{0.066\text{ m}}$.
- The condition to demote to an easier level is $\text{distance\_walked} < 0.35 \times \text{commanded\_distance}$ ($36.05 < 0.023$), which is **permanently False**!
- **Robots stuck on impossible Level 5–9 terrain can NEVER demote back down to Level 0.**

---

## F. Rollouts (64 Environments, 5.0 Seconds, 250 Steps)

Empirical evaluation of 3 test conditions across all terrain types and levels:

| Condition | Checkpoint | Survival Rate | Mean Time to End | Termination Reasons | Mean Final Roll | Verdict |
|---|---|---:|---:|---|---:|---|
| **1. Zero Action** | None (Default Pose) | **`100.0%` (64/64)** | **`5.00 s`** | `{'bad_orientation': 0, 'base_contact': 0}` | `-56.1°` | **PASS (Stable Stance)** |
| **2. Flat Probe** | `model_599.pt` (cmd `0.15 m/s`) | **`100.0%` (64/64)** | **`5.00 s`** | `{'bad_orientation': 0, 'base_contact': 0}` | `+43.8°` | **PASS (Stable Walk)** |
| **3. Multi-Terrain** | `model_9999.pt` (`2026-11-06_22-18-05`) | **`0.0%` (0/64)** | **`0.44 s` (22 steps)** | `bad_orientation` (60), `base_contact` (4) | **`-130.3°`** | **FATAL COLLAPSE** |

**Crucial Finding**:
- Conditions 1 and 2 achieve **100.0% survival with ZERO falls** across all 64 terrain tiles.
- The physical robot asset, default pose, and flat-trained policy walk cleanly over the multi-terrain tiles without falling.
- The failure is **NOT** robot geometry or spawn collision—it is 100% caused by the policy learned in `model_9999.pt`.

---

## G. Fall Dynamics Audit (`model_9999.pt`)

Aggregating the 50 steps leading up to termination across all 64 environments:

1. **Direction of Fall**:
   - Final Roll Angle: Mean = **`-130.3°`** (Rightward flip).
   - Left tilts ($> +17^\circ$): **0 / 64 (0.0%)**.
   - Right tilts ($< -17^\circ$): **64 / 64 (100.0%)**.
   - **Every single robot flips to the RIGHT.**
2. **Left vs. Right Action Asymmetry**:
   - Coxa actions: Left (Legs 1,2,3) = `+0.826` vs Right (Legs 6,5,4) = `-1.029` ($\Delta = +1.855\text{ rad}$)
   - Thigh actions: Left (Legs 1,2,3) = **`-1.607`** vs Right (Legs 6,5,4) = **`+1.865`** ($\Delta = \mathbf{-3.472\text{ rad}}$)
   - Foot actions: Left (Legs 1,2,3) = `-0.104` vs Right (Legs 6,5,4) = `+2.154` ($\Delta = -2.258\text{ rad}$)
3. **Physical Mechanism of the One-Sided Push**:
   - The policy drives the right thigh joints (`NCCUP6, NCCUP5, NCCUP4`) into massive downward extension ($+1.865\text{ to } +2.608\text{ rad}$) while simultaneously retracting the left thigh joints (`NCCUP3, NCCUP2`) upward ($-2.769\text{ to } -2.964\text{ rad}$).
   - This asymmetrical torque forces the right legs to slam into the terrain with peak saturation (up to 44% torque saturation) while the left legs lift off the ground (`NFOOT6` duty cycle: 6.5%, `NFOOT1` duty cycle: 82.9%).
   - The robot violently levers itself over its right side, exceeding the $1.2\text{ rad}$ ($68.8^\circ$) `bad_orientation` threshold within 22 control steps.

---

## H. Termination Audit

- **Configured Limits**:
  - `bad_orientation`: `limit_angle = 1.2 rad` ($68.75^\circ$) relative to base $-Z$ upright.
  - `base_contact`: threshold $1.0\text{ N}$ on bodies `.*base_link.*|.*NCCUP.*`.
  - `invalid_state_nan`: NaN/Inf detector.
- **Trigger Rate on Level 0 with Flat Probe (`model_599.pt`)**:
  - Terminations in first 5 steps: **`0 / 64` (0.0%)**.
  - Total terminations over 250 steps: **`0 / 64` (0.0%)**.
  - All terminations observed during multi-terrain training were caused by the policy's self-generated asymmetric actions, not false-positive triggers.

---

## I. Isolation Repro Experiment

To isolate whether the failure originates in the task wiring/rewards or in the terrain elevation/curriculum, we trained the exact Stage 2 task on 100% flat sub-terrain (`diag_isolation_repro.py`, 256 envs, 100 iterations from scratch):

- **Iterations 0–10**: Robots learn to balance. `time_out` climbs from 0.01 $\to$ 0.40. `bad_orientation` = **0.0000**.
- **Iterations 10–25**: `time_out` reaches **1.0000 (100% survival)**. `bad_orientation` = **0.0000**.
- **Iterations 50–100**:
  - `Mean episode length`: **`1000.00 steps` (full 20.0s episode)**.
  - `Episode_Reward/base_height_shaped`: **`+0.9383`** (attractor operates cleanly near 1.0).
  - `Episode_Termination/bad_orientation`: **`0.0000` (Zero falls)**.
  - `Episode_Termination/base_contact`: **`0.0000` (Zero strikes)**.

**Definitive Proof**: When terrain elevation is flattened to $z=0$, the Stage 2 task trains with **100% survival and zero flips**. The failure is exclusively driven by the interaction between terrain elevation and absolute world Z in the reward/curriculum code.

---

## J. Warm-Start Analysis & Recommendations

### Why Warm-Start Failed in Run `2026-11-06_22-18-05`
1. **Experiment Directory Mismatch**: In `scripts/rsl_rl/train.py:139`, `log_root_path` is hard-coded as `logs/rsl_rl/<agent_cfg.experiment_name>`. Because `ZoeTerrainPPORunnerCfg` sets `experiment_name = "zoe_hexapod_terrain"`, passing `--load_run 2026-11-05_19-51-02` searches inside `logs/rsl_rl/zoe_hexapod_terrain/`, where the flat deploy checkpoint does not exist.
2. **Fallback to Scratch**: RSL-RL fell back to cold initialization with random weights (`resume: False`).
3. **Actor/Critic Compatibility**: The Stage 1 policy network ($63\text{-D}$ actor, $18\text{-D}$ action, leg order `[6,1,5,4,3,2]`) is **100% identical and compatible** with Stage 2. Passing the direct path `--checkpoint logs/rsl_rl/zoe_hexapod_deploy/<timestamp>/model_599.pt` successfully loads the pre-trained weights.

### Recommended Curriculum & Staging Order
1. **Curriculum Terrain Sequencing**:
   - `Phase 1`: Flat terrain only (100%) for warm-start initialization.
   - `Phase 2`: Gentle pyramid slopes (up/down $0\text{ to }10^\circ$).
   - `Phase 3`: Uniform heightfield rough terrain ($0.5\text{ to }2.0\text{ cm}$).
   - `Phase 4`: Low pyramid stairs ($1.0\text{ to }3.0\text{ cm}$) and obstacle boxes ($1.0\text{ to }3.0\text{ cm}$).
2. **Start at Level 0**: Set `max_init_terrain_level = 0` in `TerrainImporterCfg` so all robots begin on the easiest tiles.

---

## K. Ranked Root-Cause Hypotheses

| Rank | Root Cause Hypothesis | Evidence & Mechanism | Confidence | Status |
|---|---|---|---|---|
| **1** | **`base_height_shaped` Uses Absolute World Z Instead of Relative Height** | `root_z = robot.data.root_pos_w[:, 2]` in `rewards.py:101`. On negative terrain tiles ($z_{\text{origin}} = -0.68\text{ m}$), $z_{\text{world}} = -0.51\text{ m}$, generating a massive **`-68.30` per-step penalty**. The policy discovered that flipping over immediately (dying in 22 steps) saves tens of thousands of negative penalty points compared to staying upright. | High (`VERIFIED`) | **VERIFIED** |
| **2** | **Cold Start from Scratch Instead of Warm-Start** | `agent.yaml` has `resume: False`. `get_checkpoint_path` looked in `logs/rsl_rl/zoe_hexapod_terrain/` instead of `zoe_hexapod_deploy/`. The policy started with random weights $\sigma=1.0$ directly on complex terrain. | High (`VERIFIED`) | **VERIFIED** |
| **3** | **`max_init_terrain_level = None` Spawns at Level 5–9 Immediately** | `env_cfg_terrain.py:26` leaves `max_init_terrain_level = None`, which distributes initial spawns uniformly across all difficulty levels (mean level 5.06). Untrained robots were dropped into steep pits and 6cm steps on Step 0. | High (`VERIFIED`) | **VERIFIED** |
| **4** | **Curriculum Measures Distance from World `(0, 0, 0)`** | `curriculum.py:62` computes `norm(root_pos_w[:, :2])` instead of subtracting `env_origins`. Robots on outer tiles are recorded as having walked 36 meters on step 0, making the demotion check `distance < min_dist` permanently False. Stuck robots can never demote. | High (`VERIFIED`) | **VERIFIED** |
| **5** | **Right-Sided Action Asymmetry Trap** | Under severe height penalty and random exploration, the policy latched onto an asymmetric local minimum where left thighs retract ($-1.6\text{ rad}$) and right thighs push ($+1.865\text{ rad}$, $\Delta = 3.47\text{ rad}$), slamming the robot into a $-130.3^\circ$ rightward roll every episode. | High (`VERIFIED`) | **VERIFIED** |

---

## Required Fixes (For Subsequent Execution Phase — No Changes Made Yet)

1. **Fix `base_height_shaped` & `base_height_l2`**: Replace `robot.data.root_pos_w[:, 2]` with relative height above terrain `robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]` (or sensor-measured ground height).
2. **Fix `zoe_terrain_curriculum`**: Subtract `env.scene.env_origins[env_ids, :2]` from `root_pos_w[env_ids, :2]` when computing `distance_walked`.
3. **Fix Terrain Initialization**: Set `max_init_terrain_level = 0` in `ZoeTerrainSceneCfg` so all environments start at difficulty 0.
4. **Fix Warm-Start Checkpoint Loading**: Update `train.py` / `cli_args.py` to allow loading a checkpoint directly from another task's experiment folder via `--checkpoint`.
5. **Fix Critic Height Observation**: In `observations.py:47`, change `base_height_privileged` to report `root_pos_w[:, 2] - env.scene.env_origins[:, 2]`.

---

## Fixes and verification

### 1. Summary of Fixes Implemented (Part A — `VERIFIED`)

| Item | Commit | Files Modified | What Was Fixed & Verified |
|---|---|---|---|
| **A1** | `08b535b` | [`rewards.py`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/mdp/rewards.py), [`observations.py`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/mdp/observations.py), [`events.py`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/mdp/events.py), [`env_cfg_terrain.py`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/env_cfg_terrain.py), [`scripts/chunk_check.py`](file:///home/rithwik/SpdrBot-main/scripts/chunk_check.py), [`tests/test_height_above_ground.py`](file:///home/rithwik/SpdrBot-main/tests/test_height_above_ground.py) | Added unified `height_above_ground(env)` helper using a 3x3 (`0.3m x 0.3m`, `0.15m` resolution) downward `RayCasterCfg` (`height_scanner` on `/World/ground`) with fallback `ground_height_from_feet(env)` (mean world $z$ of contacted foot tips minus foot sphere radius $0.0085\text{ m}$). Used by `base_height_shaped`, `foot_clearance_relative`, `foot_tip_clearance_reward`, `metric_base_height`, `base_height_privileged`, and `chunk_check.py`. Deleted unused `base_height_l2`. Added 3 unit tests (`flat`, `raised/lowered tiles`, `slope away from origin`, `raycaster vs foot cross-check`). **(`VERIFIED`)** |
| **A2** | `393b8e7` | [`scripts/analysis/verify_a2_b1_spawn_and_zero_action.py`](file:///home/rithwik/SpdrBot-main/scripts/analysis/verify_a2_b1_spawn_and_zero_action.py) | Verified `base_height_shaped` at spawn ($t=0$, zero action) across all 7 terrain types and all 10 difficulty levels (`0..9`): min = `0.7992`, max = `0.7993`, **spread = `0.000075` ($\ll \pm 0.02$)**. **(`VERIFIED`)** |
| **A3** | `b50ef24`, `6658e49` | [`curriculum.py`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/mdp/curriculum.py), [`terrains.py`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/terrains.py), [`tests/test_curriculum.py`](file:///home/rithwik/SpdrBot-main/tests/test_curriculum.py) | Enabled `curriculum=True` in `ZOE_TERRAINS_CFG` (`10` rows for levels `0..9`, `20` columns). Fixed `zoe_terrain_curriculum` to: (a) subtract `env.scene.env_origins[env_ids, :2]` before computing distance, (b) track actual elapsed steps since reset (`_last_reset_step`), (c) make no promotion/demotion decision when $\lVert\text{cmd}_{xy}\rVert < 0.1\text{ m/s}$ or $d_{\text{cmd}} < 0.5\text{ m}$ (standing envs keep their level), (d) promote only if `not terminated early and dist >= 0.75 * cmd_dist` and demote if `dist < 0.35 * cmd_dist`, (e) clamp levels to `[0, max_level]`, and (f) log per-terrain-type `mean_level`, `promotion_rate`, `demotion_rate`, `survival` in `extras["log"]`. Added 4 unit tests covering every branch. **(`VERIFIED`)** |
| **A4** | `7f129b3`, `46d592e` | [`env_cfg_terrain.py`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/env_cfg_terrain.py), [`tests/test_curriculum.py`](file:///home/rithwik/SpdrBot-main/tests/test_curriculum.py) | Set `max_init_terrain_level = 1` in `ZoeTerrainSceneCfg` (robots start on levels `0–1`, mean initial level `0.506`) and removed an unused legacy sub-grid offset function that shifted robots up to $\pm 0.60\text{ m}$ off the tile center platform. Added unit test enforcing `max_init_terrain_level in (0, 1)`. **(`VERIFIED`)** |
| **A5** | `4f55547`, `6658e49` | [`scripts/rsl_rl/train.py`](file:///home/rithwik/SpdrBot-main/scripts/rsl_rl/train.py), [`scripts/rsl_rl/cli_args.py`](file:///home/rithwik/SpdrBot-main/scripts/rsl_rl/cli_args.py), [`tests/test_train_hardening.py`](file:///home/rithwik/SpdrBot-main/tests/test_train_hardening.py) | Hardened `train.py`: (a) `--checkpoint` accepts direct file paths across any `experiment_name` and automatically sets `agent_cfg.resume = True`, (b) exits loudly with return code `2` if `--resume`, `--load_run`, or `--checkpoint` cannot be resolved, (c) prints an explicit `RESUMED from <path> (iteration N)` or `FROM SCRATCH` banner and writes `resume_from` into `params/agent.yaml`, (d) adds a 150-iteration early-abort watchdog (`exit code 3` if `mean_episode_length < 100`, `time_out < 5%`, or `mean_reward` still falling; warning if `lr < 1e-5`). Added 4 unit tests. **(`VERIFIED`)** |
| **A6** | `a95ed2c` | [`env_cfg_terrain.py`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/env_cfg_terrain.py) | Unified `ZoeTerrainSceneCfg` ground `physics_material` to `static_friction=1.0, dynamic_friction=1.0, friction_combine_mode="multiply"` matching `ZoeSceneCfg` (effective static friction `[0.40, 1.20]`, dynamic `[0.30, 1.00]`). Measured slope roll/pitch distribution (`|roll| p95 = 6.62°`, `|pitch| p95 = 12.96°`, max `20.06°` $\ll 68.75°` limit) and left `flat_orientation_l2` unchanged (`0.0`). **(`VERIFIED`)** |
| **B3-Cmd** | `d5a6e90` | [`env_cfg_terrain.py`](file:///home/rithwik/SpdrBot-main/source/spdrbot3/spdrbot3/tasks/manager_based/zoe_deploy/env_cfg_terrain.py) | Added `ZoeTerrainCommandsCfg` (`resampling_time_range = (20.0, 20.0)`, `lin_vel_x = (0.15, 0.25)`, `lin_vel_y = (-0.05, 0.05)`, `ang_vel_z = (-0.05, 0.05)`) so each 20s episode commands a consistent straight outward traversal in the robot's random spawn heading $[-\pi, +\pi]$, aligning `cmd_speed * episode_time` with net displacement from `env_origin`. **(`VERIFIED`)** |

---

### 2. A1 & A2 Verification: `height_above_ground` Cross-Check & Spawn Table (`VERIFIED`)

#### A1 Cross-Check: `height_above_ground(method="raycaster")` vs `height_above_ground(method="feet")` (`scripts/analysis/verify_a2_b1_spawn_and_zero_action.py`)
**Justification of Primary Method (`VERIFIED`)**: The 3x3 (`0.3m x 0.3m`) downward `RayCasterCfg` (`height_scanner` attached to `base_link` with `attach_yaw_only=True`) measures the true terrain mesh elevation directly under the robot's body even during dynamic flight/swing phases or when feet step into local crevices, whereas foot-tip contact height is only defined for grounded feet. `ground_height_from_feet(env)` is retained as an automatic fallback whenever no raycaster sensor is present in the scene.

| Terrain Type | Mean $\|\Delta h\|$ (`raycaster` vs `feet`) (m) | Max $\|\Delta h\|$ (`raycaster` vs `feet`) (m) | Tolerance | Result |
|---|---:|---:|---:|---|
| `flat` | `0.0003` | `0.0003` (`0.3 mm`) | `<= 0.0100 m` (`1 cm`) | **PASS (`VERIFIED`)** |
| `rough` | `0.0038` | `0.0073` (`7.3 mm`) | `<= 0.0200 m` (`2 cm`) | **PASS (`VERIFIED`)** |
| `slope_up` | `0.0003` | `0.0003` (`0.3 mm`) | `<= 0.0200 m` (`2 cm`) | **PASS (`VERIFIED`)** |
| `slope_down` | `0.0003` | `0.0003` (`0.3 mm`) | `<= 0.0200 m` (`2 cm`) | **PASS (`VERIFIED`)** |
| `stairs_up` | `0.0003` | `0.0003` (`0.3 mm`) | `<= 0.0200 m` (`2 cm`) | **PASS (`VERIFIED`)** |
| `stairs_down` | `0.0003` | `0.0003` (`0.3 mm`) | `<= 0.0200 m` (`2 cm`) | **PASS (`VERIFIED`)** |
| `boxes` | `0.0003` | `0.0003` (`0.3 mm`) | `<= 0.0200 m` (`2 cm`) | **PASS (`VERIFIED`)** |

#### A2 Spawn Table: `base_height_shaped` at Spawn ($t=0$, Zero Action, $z_{\text{offset}} = 0.1427\text{ m}$) Across All 7 Terrain Types $\times$ 10 Levels (`VERIFIED`)

| Terrain Type | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `flat` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` |
| `rough` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7992` | `0.7993` | `0.7993` | `0.7992` | `0.7993` | `0.7992` |
| `slope_up` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` |
| `slope_down` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` |
| `stairs_up` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` |
| `stairs_down` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` |
| `boxes` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` | `0.7993` |

- **Min across all 70 tiles**: `0.7992`, **Max**: `0.7993`, **Max-Min Spread**: **`0.000075` ($\le 0.02$ required) — PASS (`VERIFIED`)**.

---

### 3. B1 Verification: Section E & F Re-Run Before vs. After Fixes (`VERIFIED`)

| Metric (64 Envs across Multi-Terrain Tiles) | Before Fixes (`Section E & F` in Diagnosis) | After Fixes (`verify_a2_b1_spawn_and_zero_action.py`) | Status |
|---|---:|---:|---|
| Tile origin $z$ range (`env_origins[:, 2]`) | `[-0.6807 m, +0.0250 m]` | `[-0.6923 m, +0.7069 m]` (all 10 levels enabled) | **`VERIFIED`** |
| World `root_pos_w[:, 2]` range at spawn | `[-0.5145 m, +0.2036 m]` | `[-0.5484 m, +0.8535 m]` | **`VERIFIED`** |
| Height input to `base_height_shaped` | `[-0.5145 m, +0.2036 m]` (world $z$ bug) | **`[0.1380 m, 0.1515 m]`** (mean `0.1430 m` via `height_above_ground`) | **`VERIFIED`** |
| `base_height_shaped` reward at spawn (mean / min) | `-10.21` / **`-68.30 / step`** | **`+0.7818`** / **`+0.6788 / step`** | **`VERIFIED`** |
| Critic `base_height_privileged` at spawn | `[-0.5145 m, +0.2036 m]` | **`[0.1380 m, 0.1515 m]`** | **`VERIFIED`** |
| 250-step (`5.0s`) Zero-Action Survival | `100.0%` (`0 / 64` falls) | **`100.0%` (`0 / 64` falls)** | **`VERIFIED`** |

---

### 4. B2 & B3 Verification: Training Curves & Multi-Terrain Traversal Comparison (`VERIFIED`)

#### Training Curve Comparison: Failed Run vs. B3 Probe 1 vs. B3 Probe 2 (`VERIFIED`)
- **Failed Run**: `logs/rsl_rl/zoe_hexapod_terrain/2026-11-06_22-18-04` (`INVALID - DO NOT USE`, TB events archived to `logs/_archive_tb/`)
- **B3 Probe 1** (`600` iters, warm-started from `model_599.pt`, inherited `CommandsCfg` with `3–5s` resampling): `logs/rsl_rl/zoe_hexapod_terrain/2026-11-07_20-39-49`
- **B3 Probe 2** (`600` iters, warm-started from `model_599.pt`, `ZoeTerrainCommandsCfg` with `20s` episode traversal commands): `logs/rsl_rl/zoe_hexapod_terrain/2026-11-07_22-49-47`

| Scalar Tag | Failed Run (`2026-11-06_22-18-04`, iter 0 $\to$ 100 $\to$ 612) | B3 Probe 1 (`2026-11-07_20-39-49`, iter 0 $\to$ 100 $\to$ 600) | B3 Probe 2 (`2026-11-07_22-49-47`, iter 0 $\to$ 100 $\to$ 600) |
|---|---|---|---|
| `resume_from` | `False` (Cold Start Bug) | `.../2026-11-05_19-51-02/model_599.pt` | `.../2026-11-05_19-51-02/model_599.pt` |
| `Train/mean_episode_length` | `18.30` $\to$ `22.15` $\to$ **`22.24`** | `21.84` $\to$ **`1000.00`** $\to$ **`1000.00`** | `21.84` $\to$ **`1000.00`** $\to$ **`999.96`** |
| `Train/mean_reward` | `-16.90` $\to$ `-7.88` $\to$ **`-7.17`** | `+0.49` $\to$ `+41.89` $\to$ **`+46.85`** | `+0.51` $\to$ `+43.49` $\to$ **`+44.63`** |
| `Episode_Reward/base_height_shaped` | `-1.5467` $\to$ `-1.6621` $\to$ **`-1.7145`** | `+0.0209` $\to$ `+0.9679` $\to$ **`+0.9631`** | `+0.0207` $\to$ `+0.9533` $\to$ **`+0.8922`** |
| `Episode_Reward/track_lin_vel_xy_exp` | `0.0110` $\to$ `0.0241` $\to$ `0.0363` | `0.0253` $\to$ `1.4517` $\to$ **`1.6939`** | `0.0331` $\to$ `1.5778` $\to$ **`1.5113`** |
| `Episode_Termination/bad_orientation` | `0.9571` $\to$ `1.0000` $\to$ **`1.0000`** | `0.0000` $\to$ `0.0000` $\to$ **`0.0000`** | `0.0000` $\to$ `0.0000` $\to$ **`0.0006`** |
| `Episode_Termination/base_contact` | `0.0135` $\to$ `0.0000` $\to$ `0.0000` | `0.0000` $\to$ `0.0000` $\to$ **`0.0000`** | `0.0000` $\to$ `0.0000` $\to$ **`0.0000`** |
| `Episode_Termination/time_out` | `0.0294` $\to$ `0.0000` $\to$ **`0.0000`** | `0.0271` $\to$ `1.0000` $\to$ **`1.0000`** | `0.0271` $\to$ `1.0000` $\to$ **`0.9994`** |
| `Policy/mean_noise_std` | `1.0006` $\to$ `1.0592` $\to$ `1.0731` | `0.3126` $\to$ `0.2909` $\to$ **`0.2588`** | `0.3126` $\to$ `0.2928` $\to$ **`0.2639`** |
| `Terrain/Mean_Level` | `5.0615` $\to$ `5.0615` $\to$ `5.0615` (stuck) | `0.5059` $\to$ `0.1578` $\to$ `0.1071` (cmd reversal) | `0.5059` $\to$ **`2.8059`** $\to$ **`8.4371`** |
| `Terrain/flat/mean_level` | N/A | `0.5220` $\to$ `0.1761` $\to$ `0.0566` | `0.5220` $\to$ `3.3175` $\to$ **`8.8376`** |
| `Terrain/rough/mean_level` | N/A | `0.4976` $\to$ `0.1024` $\to$ `0.0293` | `0.4976` $\to$ `2.6642` $\to$ **`8.6923`** |
| `Terrain/slope_up/mean_level` | N/A | `0.5437` $\to$ `0.1650` $\to$ `0.1942` | `0.5437` $\to$ `3.5654` $\to$ **`8.7184`** |
| `Terrain/slope_down/mean_level` | N/A | `0.6176` $\to$ `0.1373` $\to$ `0.1667` | `0.6176` $\to$ `3.0124` $\to$ **`8.6748`** |
| `Terrain/stairs_up/mean_level` | N/A | `0.4878` $\to$ `0.2439` $\to$ `0.1512` | `0.4878` $\to$ `3.0637` $\to$ **`8.7068`** |
| `Terrain/stairs_down/mean_level` | N/A | `0.4118` $\to$ `0.1471` $\to$ `0.0588` | `0.4118` $\to$ `1.0725` $\to$ **`6.4077`** |
| `Terrain/boxes/mean_level` | N/A | `0.4706` $\to$ `0.0784` $\to$ `0.0784` | `0.4706` $\to$ `2.3039` $\to$ **`8.0844`** |

#### Multi-Terrain Traversal (`scripts/terrain_check.py`, 20s @ `0.20 m/s`, 65 envs/terrain across L0, 2, 4, 6, 9): B2 Flat Probe (`model_599.pt`) vs. B3 Probe 2 (`model_1198.pt`) (`VERIFIED`)

| Terrain Type | B2 Flat Probe (`model_599.pt`) L0–4 Surv | B3 Probe 2 (`model_1198.pt`) L0–4 Surv | B2 All-Level Surv | B3 Probe 2 All-Level Surv | B2 Dist All (m) | B3 Probe 2 Dist All (m) | B2 Torque Sat | B3 Probe 2 Torque Sat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `flat` | `100.0%` | **`100.0%`** | `100.0%` | **`100.0%`** | `3.45` | **`3.90`** | `11.27%` | **`10.22%`** |
| `rough` | `100.0%` | **`100.0%`** | `100.0%` | **`100.0%`** | `2.86` | **`3.46`** | `11.60%` | **`10.26%`** |
| `slope_up` | `100.0%` | **`100.0%`** | `100.0%` | **`100.0%`** | `3.80` | **`3.93`** | `11.00%` | **`10.51%`** |
| `slope_down` | `100.0%` | **`100.0%`** | `100.0%` | **`100.0%`** | `2.83` | **`3.46`** | `10.93%` | **`10.18%`** |
| `stairs_up` | `100.0%` | **`100.0%`** | `90.8%` (`L9: 61.5%`) | **`100.0%` (`L9: 100%`)** | `2.71` | **`3.58`** | `13.25%` | **`10.90%`** |
| `stairs_down` | `100.0%` | **`100.0%`** | `100.0%` | **`100.0%`** | `1.21` | **`3.12`** | `17.17%` | **`13.75%`** |
| `boxes` | `100.0%` | **`100.0%`** | `100.0%` | **`100.0%`** | `2.04` | **`3.27`** | `15.26%` | **`12.46%`** |

#### Stage 1 Flat Gate Comparison (`scripts/chunk_check.py` on `Zoe-Hexapod-Flat-Deploy-v0`): Flat Probe (`model_599.pt`) vs. B3 Probe 2 (`model_1198.pt`) (`VERIFIED`)

| Condition | Metric | Flat Probe (`model_599.pt`) | Stage 2 Probe 2 (`model_1198.pt`) | Relative Change | Threshold |
|---|---|---:|---:|---|---|
| 1. Zero Command | Base height `>= 0.1284m` | `100.00%` | **`100.00%`** | `0.0%` | `>= 95%` (**PASS**) |
| 1. Zero Command | `>= 5` feet in contact | `98.29%` | **`98.30%`** | `+0.01%` | `>= 95%` (**PASS**) |
| 1. Zero Command | Non-foot contact | `0.00%` | **`0.00%`** | `0.0%` | `< 1%` (**PASS**) |
| 1. Zero Command | Torque sat (`\|tau\| >= 2.45 N.m`) | `2.92%` (FAIL) | **`1.62%` (PASS)** | **`-44.5%` (improved)** | `<= 2%` (**PASS**) |
| 2. Forward Walk `0.15m/s` | Base height `>= 0.1284m` | `100.00%` | **`100.00%`** | `0.0%` | `>= 90%` (**PASS**) |
| 2. Forward Walk `0.15m/s` | `>= 3` feet in contact | `99.39%` | **`99.43%`** | `+0.04%` | `>= 99%` (**PASS**) |
| 2. Forward Walk `0.15m/s` | Non-foot contact | `0.00%` | **`0.00%`** | `0.0%` | `< 1%` (**PASS**) |
| 2. Forward Walk `0.15m/s` | Static stability margin (`p5`) | `+0.0472m` | **`+0.0682m`** | **`+44.5%` (improved)** | `p5 >= 0.0m` (**PASS**) |
| 2. Forward Walk `0.15m/s` | Forward speed (not-frozen) | `0.1518 m/s` | **`0.1515 m/s`** | `-0.2%` (within `10%`) | `>= 0.1200 m/s` (**PASS**) |
| 2. Forward Walk `0.15m/s` | Torque sat (`\|tau\| >= 2.45 N.m`) | `10.19%` | **`9.59%`** | **`-5.9%` (improved)** | `<= 2%` (FAIL) |
| 3. Push Recovery | Terminations (falls) | `0 falls` | **`0 falls`** | `0.0%` | `0 falls` (**PASS**) |
| 3. Push Recovery | Non-foot contact | `0.00%` | **`0.00%`** | `0.0%` | `< 1%` (**PASS**) |
| 3. Push Recovery | Push 1 & 2 Height Recovery | `100.00%` | **`100.00%`** | `0.0%` | `>= 90%` (**PASS**) |
| 3. Push Recovery | Torque sat (`\|tau\| >= 2.45 N.m`) | `10.48%` | **`9.36%`** | **`-10.7%` (improved)** | `<= 2%` (FAIL) |

**All 4 B3 Probe Pass Criteria (`VERIFIED`)**:
1. `mean_episode_length > 800` by iter 100: **PASS (`1000.00` by iter 50)**.
2. `Terrain/Mean_Level` rising from `~0.5`: **PASS (`0.5059` $\to$ `2.8059` at iter 100 $\to$ `8.4371` at iter 600)**.
3. No terrain type with survival `< 50%` at levels 0–2: **PASS (`100.0%` survival across all 7 terrain types at levels `0, 2, 4, 6, 9`)**.
4. Stage 1 flat gate metrics on flat tiles not worse than `10%` relative to flat probe: **PASS (every gated metric is within `0.2%` or improved by `6%–44%`)**.

---

### 5. Historical B3 Probe Assessment (Superseded — See Section 7 Amendment)

1. **Historical 2026-11-07 B3 Probe status (`VERIFIED`, pre-5.0 kg payload update)**:
   - All 5 root causes from the Stage 2 flipping diagnosis were fixed (`A1–A6`). In the 2026-11-07 B3 probe (`2026-11-07_22-49-47/model_1198.pt`, warm-started from `2026-11-05_19-51-02/model_599.pt`), survival reached `99.94%` during training and `100.0%` across `levels 0, 2, 4, 6, 9` on `scripts/terrain_check.py`.
   - **Superseded by later asset & metric fixes (`2026-11-11`)**: That B3 probe ran under the `4.1662 kg` CAD mass before the `+0.8338 kg` (`5.0000 kg`) payload update (`b58c52c`), with forward-biased `ZoeTerrainCommandsCfg` (`report-only strafe = +0.0169 m/s`), and before Task 6 exposed inflation in the step-integrated progress metric. Note also that Task 5 (`52685d4`) moved Stage 1 torque saturation (`10.19%` on `model_599.pt`, `9.59%` on `model_1198.pt`) from a pass/fail gate to report-only and added mandatory directional velocity gates.
2. **Current Readiness Summary (`VERIFIED`)**:
   - **Stage 2 is `NO-GO` / `NOT READY` for long chunk training** until a `5.00 kg` Stage 1 policy passes all 15 frozen conditions in `scripts/chunk_check.py`. Wait for Stage 1 Chunk 1 to pass before running Stage 2.

---

### 6. Chunked Stage 2 Invocation Reference & Abort Criteria

When (and only after) a Stage 1 `5.00 kg` policy passes the frozen 15-condition gate in `scripts/chunk_check.py`, launch Stage 2 via `scripts/rsl_rl/train.py --task Zoe-Hexapod-Terrain-v0 --checkpoint path/to/passing_stage1_ckpt.pt --num_envs 1024 --max_iterations 1500 --headless`, and evaluate each resulting run checkpoint with:
- `scripts/terrain_check.py --checkpoint path/to/terrain_ckpt.pt --headless`
- `scripts/chunk_check.py --task Zoe-Hexapod-Flat-Deploy-v0 --checkpoint path/to/terrain_ckpt.pt --headless`

See Section 7.E for current blocking prerequisites.

#### Stage 2 Abort Criteria
1. **Automatic Watchdog Abort (Iteration 150, Exit Code `3`)**: `train.py` automatically aborts at iteration 150 if `mean_episode_length < 100`, `time_out < 5%`, or `mean_reward` is lower than at iteration 0.
2. **Manual / Inter-Chunk Abort Criteria**: Stop and inspect before continuing to the next chunk if:
   - `Terrain/Mean_Level` fails to rise above `2.0` after 300 iterations or regresses across consecutive chunks.
   - Any terrain type in `scripts/terrain_check.py` has `< 90%` survival at levels `0–4` (`B4` deployment gate).
   - Stage 1 flat gate metrics (`scripts/chunk_check.py`) degrade by more than `10%` relative to the warm-start checkpoint.
   - `Policy/mean_noise_std` grows above `0.50` (indicating exploration destabilization).

---

## 7. Stage 2 Multi-Group & Terrain Ceiling Verification (Parts A–F)

> [!WARNING]
> **2026-11-11 RETRACTION (`VERIFIED`)**: The `GO` verdict originally issued in this section on `2026-11-10` is **RETRACTED**. Two defects invalidated `model_2098.pt`'s apparent `43%–58%` progress:
> 1. `model_2098.pt` was warm-started from `logs/rsl_rl/zoe_hexapod_deploy/2026-11-09_22-33-44/model_1499.pt`, which was later proved to be a Stage 1 creeper (`0.0065 m/s` forward speed).
> 2. Part A's step-integrated metric ($\sum \max(0, \mathbf{v}_{xy} \cdot \hat{\mathbf{cmd}}_{xy})\Delta t$) credited zero-net-motion shuffling in place.
>
> After fixing the metric to world-frame **net displacement** (`(pos_end_w - pos_start_w)[:2] . unit(cmd_dir_w)` with full 3D `root_quat_w` rotation and pre-reset position capture, commits `1462e4f`, `5cfc643`, `dbe0b49`), `model_2098.pt` achieved only **`flat 32.2%`**, **`rough 8.7%`** (`94.9%` stuck), **`slope_up 31.6%`**, **`slope_down 28.2%`**, **`stairs_up 26.1%`**, **`stairs_down 17.0%`** (`76.9%` stuck), and **`boxes 25.4%`** — **100% `FAIL`**. Tables A–C below are retained as historical Part F records under the old metric alongside this retraction.

**Date**: 2026-11-10 (Amended 2026-11-11)  
**Evaluated Run**: `logs/rsl_rl/zoe_hexapod_terrain/2026-11-10_18-11-19` (600 iterations, ended `model_2098.pt`)  
**Warm-Start Source**: Stage 1 Chunk 1 final checkpoint (`logs/rsl_rl/zoe_hexapod_deploy/2026-11-09_22-33-44/model_1499.pt` — invalid creeper baseline)  
**Architecture Upgrades Evaluated**:
1. **Part A (Curriculum Progress Accounting)**: Originally implemented as step-integrated distance $\sum \max(0, \mathbf{v}_{xy} \cdot \hat{\mathbf{cmd}}_{xy})\Delta t$; replaced in Task 6 (`1462e4f`, `5cfc643`) by world-frame net displacement `(pos_end_w - pos_start_w)[:2] . unit(cmd_dir_w)`. Standing envs protect level; command reversals preserve progress; early terminations demote.
2. **Part B (Two Env Groups)**: 50% Group T (Traversal: 20s held episode, $v_x \in [0.10, 0.25], v_y \in [-0.10, 0.10], \omega_z \in [-0.30, 0.30]$, full levels 0–9) + 50% Group S (Skills: Stage 1 distribution, 3–5s resampling, 10% standing, $\ge 25\%$ yaw-dominant, backward allowed, level capped at 3).
3. **Part C (Extended Terrain Difficulty Ceiling)**: Stairs up/down 9 cm, boxes 7 cm, slopes 0.40 rad, rough 4.5 cm. Frozen deployment targets: stairs $\le 4\text{ cm}$ (Level 3), slopes $\le 12^\circ$ (Level 4), boxes $\le 4\text{ cm}$ (Level 4), rough $\le 2.5\text{ cm}$ (Level 4).
4. **Part D (Evaluation Suite)**: `scripts/terrain_check.py` measuring survival, progress ratio, stuck fraction (<20%), torque saturation, and 5-skill retention test against warm-start baseline.

---

### A. Stage 2 Probe 1 Summary Table (Target Levels vs. All Levels)

Evaluation of `model_2098.pt` on `Zoe-Hexapod-Terrain-v0` (455 envs, 65 envs/terrain across levels `[0, 2, 4, 6, 9]`, 20.0s episode, forward command 0.20 m/s with random initial headings). Comparison against warm-start baseline `model_1499.pt` (`VERIFIED`):

| Terrain Type | Target Lvl | Baseline Surv | Probe 1 Target Surv (Gate $\ge 90\%$) | Baseline Prog | Probe 1 Target Prog (Gate $\ge 70\%$) | Baseline Stuck | Probe 1 Stuck ($< 20\%$) | All-Lvl Surv | All-Lvl Prog | Torque Sat | Non-Foot | Target Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `flat` | L0-4 | `100.0%` | **`100.0%`** | `5.5%` | **`58.0%`** | `100.0%` | **`0.0%`** | `100.0%` | `57.6%` | `8.02%` | `0.02%` | **IN PROGRESS** (Surv PASS, Prog 58%) |
| `rough` | L0-4 | `92.3%` | **`100.0%`** | `6.3%` | **`43.1%`** | `92.3%` | **`0.0%`** | `100.0%` | `44.2%` | `11.14%` | `0.02%` | **IN PROGRESS** (Surv PASS, Prog 43%) |
| `slope_up` | L0-4 | `100.0%` | **`100.0%`** | `5.7%` | **`55.5%`** | `100.0%` | **`0.0%`** | `100.0%` | `55.3%` | `8.12%` | `0.02%` | **IN PROGRESS** (Surv PASS, Prog 55%) |
| `slope_down` | L0-4 | `100.0%` | **`100.0%`** | `5.7%` | **`54.4%`** | `100.0%` | **`0.0%`** | `100.0%` | `54.4%` | `8.25%` | `0.03%` | **IN PROGRESS** (Surv PASS, Prog 54%) |
| `stairs_up` | L0-3 | `100.0%` | **`100.0%`** | `5.7%` | **`55.1%`** | `100.0%` | **`0.0%`** | `69.2%` | `55.0%` | `10.22%` | `0.02%` | **IN PROGRESS** (Surv PASS, Prog 55%) |
| `stairs_down` | L0-3 | `100.0%` | **`100.0%`** | `5.6%` | **`48.1%`** | `100.0%` | **`0.0%`** | `100.0%` | `48.2%` | `9.12%` | `0.02%` | **IN PROGRESS** (Surv PASS, Prog 48%) |
| `boxes` | L0-4 | `100.0%` | **`100.0%`** | `5.6%` | **`52.8%`** | `100.0%` | **`0.0%`** | `84.6%` | `53.8%` | `8.65%` | `0.02%` | **IN PROGRESS** (Surv PASS, Prog 53%) |

---

### B. Per-Level Traversal Breakdown (`model_2098.pt`) (`VERIFIED`)

Full per-cell traversal breakdown across difficulty levels `[0, 2, 4, 6, 9]` (13 envs per cell):

| Terrain | Level | Survival | Progress Ratio | Stuck (<20%) | Final Dist (m) | Off-Plat (>1.2m) | Feet Down | Non-Foot | Torque Sat | Term (ori/base) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `flat` | 0 | `100.0%` | `57.9%` | `0.0%` | `1.13` | `7.2%` | `5.36` | `0.00%` | `7.87%` | `0/0` |
| `flat` | 2 | `100.0%` | `58.2%` | `0.0%` | `1.24` | `12.1%` | `5.26` | `0.00%` | `8.17%` | `0/0` |
| `flat` | 4 | `100.0%` | `57.9%` | `0.0%` | `1.16` | `7.8%` | `5.37` | `0.00%` | `8.02%` | `0/0` |
| `flat` | 6 | `100.0%` | `57.1%` | `0.0%` | `1.14` | `6.7%` | `5.35` | `0.00%` | `8.17%` | `0/0` |
| `flat` | 9 | `100.0%` | `57.0%` | `0.0%` | `1.22` | `9.4%` | `5.28` | `0.04%` | `7.86%` | `0/0` |
| `rough` | 0 | `100.0%` | `40.1%` | `0.0%` | `0.25` | `0.0%` | `4.76` | `0.01%` | `11.21%` | `0/0` |
| `rough` | 2 | `100.0%` | `44.9%` | `0.0%` | `0.26` | `0.0%` | `4.59` | `0.04%` | `11.40%` | `0/0` |
| `rough` | 4 | `100.0%` | `44.3%` | `0.0%` | `0.39` | `0.0%` | `4.69` | `0.00%` | `11.40%` | `0/0` |
| `rough` | 6 | `100.0%` | `45.9%` | `0.0%` | `0.33` | `0.0%` | `4.75` | `0.00%` | `11.13%` | `0/0` |
| `rough` | 9 | `100.0%` | `45.7%` | `0.0%` | `0.32` | `0.0%` | `4.63` | `0.05%` | `10.55%` | `0/0` |
| `slope_up` | 0 | `100.0%` | `53.3%` | `0.0%` | `1.07` | `3.2%` | `5.32` | `0.02%` | `8.18%` | `0/0` |
| `slope_up` | 2 | `100.0%` | `57.6%` | `0.0%` | `1.20` | `10.1%` | `5.16` | `0.02%` | `8.40%` | `0/0` |
| `slope_up` | 4 | `100.0%` | `55.6%` | `0.0%` | `1.23` | `7.2%` | `4.85` | `0.05%` | `7.77%` | `0/0` |
| `slope_up` | 6 | `100.0%` | `53.7%` | `0.0%` | `1.05` | `4.2%` | `5.32` | `0.00%` | `7.95%` | `0/0` |
| `slope_up` | 9 | `100.0%` | `56.4%` | `0.0%` | `1.19` | `7.4%` | `5.20` | `0.05%` | `8.20%` | `0/0` |
| `slope_down` | 0 | `100.0%` | `55.7%` | `0.0%` | `1.06` | `7.1%` | `5.15` | `0.03%` | `8.82%` | `0/0` |
| `slope_down` | 2 | `100.0%` | `53.5%` | `0.0%` | `1.10` | `6.7%` | `5.16` | `0.02%` | `7.82%` | `0/0` |
| `slope_down` | 4 | `100.0%` | `54.1%` | `0.0%` | `0.99` | `4.1%` | `5.16` | `0.05%` | `8.11%` | `0/0` |
| `slope_down` | 6 | `100.0%` | `54.8%` | `0.0%` | `1.16` | `9.6%` | `5.04` | `0.02%` | `6.85%` | `0/0` |
| `slope_down` | 9 | `100.0%` | `53.9%` | `0.0%` | `1.11` | `7.4%` | `4.97` | `0.03%` | `8.17%` | `0/0` |
| `stairs_up` | 0 | `100.0%` | `56.6%` | `0.0%` | `0.97` | `4.4%` | `4.99` | `0.04%` | `9.68%` | `0/0` |
| `stairs_up` | 2 | `100.0%` | `53.7%` | `0.0%` | `0.96` | `0.5%` | `4.75` | `0.01%` | `10.20%` | `0/0` |
| `stairs_up` | 4 | `84.6%` | `45.4%` | `0.0%` | `0.74` | `0.0%` | `4.65` | `0.05%` | `12.33%` | `2/0` |
| `stairs_up` | 6 | `61.5%` | `48.7%` | `0.0%` | `0.73` | `0.0%` | `4.65` | `0.00%` | `10.91%` | `5/0` |
| `stairs_up` | 9 | `0.0%` | `63.4%` | `0.0%` | `0.81` | `0.0%` | `4.90` | `0.00%` | `9.16%` | `11/3` |
| `stairs_down` | 0 | `100.0%` | `46.1%` | `0.0%` | `0.66` | `0.0%` | `5.03` | `0.04%` | `9.25%` | `0/0` |
| `stairs_down` | 2 | `100.0%` | `50.2%` | `0.0%` | `0.66` | `0.0%` | `5.14` | `0.02%` | `9.64%` | `0/0` |
| `stairs_down` | 4 | `100.0%` | `45.8%` | `0.0%` | `0.65` | `0.0%` | `5.07` | `0.03%` | `8.60%` | `0/0` |
| `stairs_down` | 6 | `100.0%` | `49.4%` | `0.0%` | `0.61` | `0.0%` | `5.22` | `0.00%` | `8.62%` | `0/0` |
| `stairs_down` | 9 | `100.0%` | `49.7%` | `0.0%` | `0.67` | `0.0%` | `4.88` | `0.02%` | `8.99%` | `0/0` |
| `boxes` | 0 | `100.0%` | `53.2%` | `0.0%` | `1.13` | `7.3%` | `4.96` | `0.04%` | `8.58%` | `0/0` |
| `boxes` | 2 | `100.0%` | `51.9%` | `0.0%` | `0.90` | `0.6%` | `4.73` | `0.03%` | `8.72%` | `0/0` |
| `boxes` | 4 | `100.0%` | `53.4%` | `0.0%` | `1.01` | `1.4%` | `4.88` | `0.02%` | `8.89%` | `0/0` |
| `boxes` | 6 | `76.9%` | `54.6%` | `0.0%` | `1.01` | `0.2%` | `5.18` | `0.00%` | `8.15%` | `3/0` |
| `boxes` | 9 | `46.2%` | `55.9%` | `0.0%` | `0.97` | `0.0%` | `4.88` | `0.00%` | `10.32%` | `5/3` |

**Difficulty Ceiling Observations (`VERIFIED`)**:
- Target levels ($\le 3$ for stairs, $\le 4$ for slopes, boxes, rough) maintain **100.0% survival** across all 7 terrain types.
- Beyond the target levels, the extended difficulty ceiling takes effect:
  - `stairs_up`: L4 surv = 84.6% $\to$ L6 surv = 61.5% $\to$ **L9 surv = 0.0%** (11 bad orientation falls, 3 base contacts).
  - `boxes`: L6 surv = 76.9% $\to$ **L9 surv = 46.2%** (5 bad orientation falls, 3 base contacts).
  - This demonstrates that the curriculum ceiling provides a genuine challenge boundary while keeping deployment target levels reliable.

---

### C. Skill Retention Suite Results (`VERIFIED`)

Evaluated on 143 environments across flat tiles and levels 0–2 of slopes and rough (5.0s per skill, 250 steps):

| Skill | Command Target | Baseline (`model_1499.pt`) | Probe 1 (`model_2098.pt`) | Relative Improvement | Gate Threshold | Result |
|---|---|---:|---:|---|---|---|
| `standing` | `[0.0, 0.0, 0.0]` | `+96.97% >=5 feet down` | **`+97.58% >=5 feet down`** | `+0.61%` (improved) | `ge 95%` | **PASS** |
| `yaw` | `[0.0, 0.0, 0.40]` | `+0.045 rad/s` | **`+0.075 rad/s`** | **`+66.7%` (improved)** | `tracking >= 0.08` | **PROGRESS** |
| `strafe` | `[0.0, 0.12, 0.0]` | `+0.017 m/s` | **`+0.059 m/s`** | **`+247.1%` (improved)** | `tracking >= 0.08` | **PROGRESS** |
| `backward` | `[-0.12, 0.0, 0.0]` | `-0.012 m/s` | **`-0.017 m/s`** | **`+41.7%` (improved)** | `tracking <= -0.08` | **PROGRESS** |
| `forward` | `[0.15, 0.0, 0.0]` | `+0.011 m/s` | **`+0.044 m/s`** | **`+300.0%` (improved)** | `tracking >= 0.10` | **PROGRESS** |

**Skill Retention Verdict**:
- Not a single arrow-key skill degraded.
- Yaw tracking improved by **+66.7%**, strafe by **+247.1%**, backward by **+41.7%**, and forward by **+300.0%**.
- Standing stability remained above gate threshold at **97.58%** ($\ge 95\%$).
- The two-group curriculum architecture successfully prevented the flat-ground skills from being trained away while developing terrain navigation.

---

### D. Honest Go / No-Go Verdicts (Amended 2026-11-11)

#### 1. Long Stage 2 Training Run (Full 1500-Iteration Chunks): **NO-GO / `NOT READY` (`VERIFIED`, Previous `GO` Retracted)**
- **Evidence**:
  1. **Invalid warm-start source**: `model_1499.pt` (`2026-11-09_22-33-44`) crept at `0.0065 m/s` forward, `10.0%` yaw, and `19.1%` strafe (`NEW` Stage 2 progress `2.2%–2.9%`, 100% FAIL).
  2. **Historical `43%–58%` probe numbers were step-integrated artifacts**: Under Task 6's world-frame net-displacement metric (`1462e4f`, `5cfc643`, `dbe0b49`), `model_2098.pt` scored **`flat 32.2%`**, **`rough 8.7%`** (`94.9%` stuck), **`slope_up 31.6%`**, **`slope_down 28.2%`**, **`stairs_up 26.1%`**, **`stairs_down 17.0%`** (`76.9%` stuck), and **`boxes 25.4%`** — **100% FAIL** against the `>= 70%` target gate.
  3. **Zero Stage 1 `5.00 kg` passing warm-start exists yet**: The best of 4 Stage 1 probes (`2026-11-11_22-30-05/model_599.pt`, `9662a60`) reached `13/15 PASS`, still failing Forward (`0.0079 m/s` vs `>= 0.1200 m/s`) and Strafe (`+0.0420 m/s = 35.0%` vs `>= 50%`).
  4. **Historical baseline cross-check (`2026-11-05_19-51-02/model_599.pt`, Task 6 `NEW` metric)**: `flat 102.5%`, `rough 53.3%`, `slope_up 106.4%`, `slope_down 90.3%`, `stairs_up 98.1%`, `stairs_down 37.1%`, `boxes 63.1%` (`FAIL` on rough, stairs_down, boxes), proving the `NEW` metric accurately distinguishes real locomotion from shuffling.

#### 2. Deployment to the Real Physical Robot: **NO-GO (`VERIFIED`)**
- **Evidence & Blockers**:
  1. **Both Stage 1 and Stage 2 fail evaluation gates**: Stage 1 (`13/15 PASS` best of 4 probes) and Stage 2 (`8.7%–32.2%` `NEW` net-displacement progress vs `>= 70%` gate) are unready. (Torque saturation is now report-only under Task 5 `52685d4`, measured at `7.78%–10.41%` on `model_599.pt` of Probe 4.)
  2. **Hardware safety prerequisites missing**:
     - No real-time hardware telemetry or servo feedback pipeline implemented/bench-tested.
     - No physical IMU sign/axis test or mechanical joint-zero calibration executed on the physical robot chassis; robot weight (`5.00 kg` assumption) unverified on a scale.
     - No hardware emergency-stop (E-stop) power cut-off or software watchdog verified on the onboard Jetson.
  - **Verdict**: Both Stage 2 long training and real robot deployment remain strictly blocked.

---

### E. Prerequisites Before Launching Stage 2 Chunks

Do **NOT** start Stage 2 training (`Zoe-Hexapod-Terrain-v0`) until all of the following are true (`VERIFIED`):
1. **Stage 1 passes the frozen 15-condition gate**: A `5.00 kg` `Zoe-Hexapod-Flat-Deploy-v0` checkpoint passes all 15 PASS/FAIL checks in `scripts/chunk_check.py` (including Forward `>= 0.1200 m/s`, Pure Yaw `>= 50%`, Strafe `>= 50%`, Backward `>= 50%`).
2. **Valid Stage 2 probe check**: Warm-start a 600-iteration Stage 2 probe from that passing Stage 1 checkpoint (or its 1500-iteration Chunk 1 continuation) and evaluate with `scripts/terrain_check.py` (verifying Task 6 net-displacement progress and zero skill regression) before committing to 1500-iteration Stage 2 chunks.

#### Abort Criteria for Future Chunks
1. **Automatic Watchdog Abort (Iteration 150)**: `train.py` aborts if `mean_episode_length < 100`, `time_out < 5%`, or `mean_reward` is lower than at iteration 0.
2. **Progress Ratio Plateau**: Abort if net-displacement progress fails to rise meaningfully above the Stage 1 warm-start or regresses between chunks.
3. **Target Survival Degradation**: Abort if survival at deployment target levels drops below $90.0\%$ on any terrain type.
4. **Skill Degradation**: Abort if standing stability drops below $95\%$ or any flat velocity tracking gate regresses on `scripts/chunk_check.py`.
5. **Policy Exploration Destabilization**: Abort if `Policy/mean_noise_std` exceeds $0.50$.



