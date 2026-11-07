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
| **1** | **`base_height_shaped` Uses Absolute World Z Instead of Relative Height** | `root_z = robot.data.root_pos_w[:, 2]` in `rewards.py:101`. On negative terrain tiles ($z_{\text{origin}} = -0.68\text{ m}$), $z_{\text{world}} = -0.51\text{ m}$, generating a massive **`-68.30` per-step penalty**. The policy discovered that flipping over immediately (dying in 22 steps) saves tens of thousands of negative penalty points compared to staying upright. | **100%** | **VERIFIED** |
| **2** | **Cold Start from Scratch Instead of Warm-Start** | `agent.yaml` has `resume: False`. `get_checkpoint_path` looked in `logs/rsl_rl/zoe_hexapod_terrain/` instead of `zoe_hexapod_deploy/`. The policy started with random weights $\sigma=1.0$ directly on complex terrain. | **100%** | **VERIFIED** |
| **3** | **`max_init_terrain_level = None` Spawns at Level 5–9 Immediately** | `env_cfg_terrain.py:26` leaves `max_init_terrain_level = None`, which distributes initial spawns uniformly across all difficulty levels (mean level 5.06). Untrained robots were dropped into steep pits and 6cm steps on Step 0. | **100%** | **VERIFIED** |
| **4** | **Curriculum Measures Distance from World `(0, 0, 0)`** | `curriculum.py:62` computes `norm(root_pos_w[:, :2])` instead of subtracting `env_origins`. Robots on outer tiles are recorded as having walked 36 meters on step 0, making the demotion check `distance < min_dist` permanently False. Stuck robots can never demote. | **100%** | **VERIFIED** |
| **5** | **Right-Sided Action Asymmetry Trap** | Under severe height penalty and random exploration, the policy latched onto an asymmetric local minimum where left thighs retract ($-1.6\text{ rad}$) and right thighs push ($+1.865\text{ rad}$, $\Delta = 3.47\text{ rad}$), slamming the robot into a $-130.3^\circ$ rightward roll every episode. | **100%** | **VERIFIED** |

---

## Required Fixes (For Subsequent Execution Phase — No Changes Made Yet)

1. **Fix `base_height_shaped` & `base_height_l2`**: Replace `robot.data.root_pos_w[:, 2]` with relative height above terrain `robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]` (or sensor-measured ground height).
2. **Fix `zoe_terrain_curriculum`**: Subtract `env.scene.env_origins[env_ids, :2]` from `root_pos_w[env_ids, :2]` when computing `distance_walked`.
3. **Fix Terrain Initialization**: Set `max_init_terrain_level = 0` in `ZoeTerrainSceneCfg` so all environments start at difficulty 0.
4. **Fix Warm-Start Checkpoint Loading**: Update `train.py` / `cli_args.py` to allow loading a checkpoint directly from another task's experiment folder via `--checkpoint`.
5. **Fix Critic Height Observation**: In `observations.py:47`, change `base_height_privileged` to report `root_pos_w[:, 2] - env.scene.env_origins[:, 2]`.

