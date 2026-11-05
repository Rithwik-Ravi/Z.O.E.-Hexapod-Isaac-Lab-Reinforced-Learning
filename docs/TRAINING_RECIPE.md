# Z.O.E. Hexapod Isaac Lab Training Recipe

This document provides step-by-step instructions for reproducing the blind, deployable locomotion policy training and export for the Z.O.E. 18-DOF hexapod robot (`spdrbot3`) using Isaac Lab and RSL-RL.

---

## 1. Hardware & System Requirements

- **GPU**: NVIDIA RTX 4070 Laptop GPU (8 GB VRAM) or desktop equivalent (e.g. RTX 3080/4070/4080).
- **Environment**: Conda env `env_isaaclab` (Python 3.10).
- **Installed Stack**:
  - Isaac Lab: `v2.3.2` (`isaaclab 0.54.2`)
  - Isaac Sim: `4.5.0.0`
  - RSL-RL: `rsl-rl-lib 3.1.2`
  - PyTorch: `2.14.0` with CUDA support.

---

## 2. Asset Verification (Milestone M0)

Before running training, execute the automated asset auditor to verify the robot USD geometry, mass, joint order, and standing height:

```bash
/home/rithwik/miniconda3/envs/env_isaaclab/bin/python scripts/audit_asset.py \
    --usd hexapod/hexa/hexa.usd \
    --real-mass-kg 2.0
```

- **Output**: `deploy/asset_report.json`
- **Verification**: Check console output for mass warning (>20% difference between USD mass ~0.63 kg and hardware mass ~2.0 kg is automatically offset via domain randomization in `params.py`).

---

## 3. Stage 1 Training: Flat Ground Locomotion

Stage 1 trains basic omnidirectional velocity tracking, stance stability, and domain randomization resilience on flat terrain.

### Command

```bash
/home/rithwik/miniconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
    --task Zoe-Hexapod-Flat-Deploy-v0 \
    --headless \
    --num_envs 2048 \
    --max_iterations 3000
```

### VRAM & Performance Guidelines (RTX 4070 - 8 GB VRAM)

- **`num_envs=2048`**: ~6.2 GB VRAM usage. Recommended for max throughput (~18,000 steps/sec).
- **`num_envs=1024`**: ~4.5 GB VRAM usage. Use if running alongside GUI or background display applications.
- **Expected Training Time**: ~15-20 minutes for 3000 iterations.

### Stopping Criteria

- **Reward convergence**: Total episode reward plateaus around > 25.0.
- **Command Tracking**: Linear velocity error < 0.08 m/s, angular velocity error < 0.12 rad/s.
- **Tripod Stance Stability**: Zero ground contact violations (<3 feet contact penalty approaches 0).

---

## 4. Stage 2 Training: Multi-Terrain Locomotion

Stage 2 fine-tunes or trains from scratch on a procedurally generated terrain grid (heightfield, slopes, stairs, boxes) with automated difficulty curriculum.

### Command

```bash
/home/rithwik/miniconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
    --task Zoe-Hexapod-Terrain-v0 \
    --headless \
    --num_envs 2048 \
    --max_iterations 5000
```

### Resume from Stage 1 Checkpoint (Optional Warm-Start)

To resume from Stage 1 weights:

```bash
/home/rithwik/miniconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
    --task Zoe-Hexapod-Terrain-v0 \
    --resume \
    --load_run Zoe-Hexapod-Flat-Deploy-v0/<run_dir> \
    --headless
```

---

## 5. Play & Visualization

To visually evaluate the trained checkpoint in Isaac Sim GUI:

```bash
/home/rithwik/miniconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/play.py \
    --task Zoe-Hexapod-Flat-Deploy-Play-v0 \
    --num_envs 16
```

---

## 6. Exporting Policy for Deployment (Milestone M4)

Export the policy ONNX model, pure NumPy weights (`policy_mlp.npz`), and interface contract `deploy.yaml`:

### Method A: Via `play.py` `--export` flag

```bash
/home/rithwik/miniconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/play.py \
    --task Zoe-Hexapod-Flat-Deploy-Play-v0 \
    --export
```

### Method B: Direct Export Script

```bash
/home/rithwik/miniconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/export.py \
    --task Zoe-Hexapod-Flat-Deploy-v0 \
    --checkpoint logs/rsl_rl/zoe_hexapod_flat_deploy/<run>/model_3000.pt \
    --output deploy/export_run
```

### Export Artifacts Generated:

1. `policy.onnx`: Actor ONNX computational graph (input: `obs` [1, 63], output: `actions` [1, 18]).
2. `policy_mlp.npz`: Pure NumPy weights (`W0, b0, W1, b1, W2, b2, W3, b3`), `obs_mean`, `obs_var`, `eps`.
3. `deploy.yaml`: Complete deployment contract metadata (joint order, observation scaling, clip values, joint limits).
4. `reference_rollout.npz`: 500-step golden deterministic simulation rollout for C++ Copter / C++ ROS validation.

---

## 7. Robustness Evaluation (Milestone M5)

Run automated evaluation matrix across nominal and extreme domain scenarios:

```bash
/home/rithwik/miniconda3/envs/env_isaaclab/bin/python scripts/eval_robustness.py \
    --task Zoe-Hexapod-Flat-Deploy-Play-v0 \
    --checkpoint logs/rsl_rl/zoe_hexapod_flat_deploy/<run>/model_3000.pt \
    --num_envs 64 \
    --num_steps 500 \
    --output deploy/robustness_report.json
```

---

## 8. Unit Testing (PyTest)

Verify environment configurations, curriculum math, and ONNX export equivalence without requiring full GPU simulation:

```bash
PYTHONPATH="source/spdrbot3:." /home/rithwik/miniconda3/envs/env_isaaclab/bin/pytest -c /dev/null -p no:launch_testing -p no:launch_testing_ros tests/
```

Expected result: `7 passed`.
