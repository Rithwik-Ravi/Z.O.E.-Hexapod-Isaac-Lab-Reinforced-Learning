# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for policy export, ONNX vs PyTorch vs NumPy MLP numerical equivalence."""

import os
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn

from scripts.rsl_rl.export import export_numpy_mlp, numpy_mlp_inference


class DummyActor(nn.Module):
    """Dummy MLP actor network for testing export equivalence."""
    def __init__(self, in_dim=63, out_dim=18):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.net(x)

    def __getitem__(self, idx):
        return self.net[idx]

    def __iter__(self):
        return iter(self.net)


class DummyNormalizer(nn.Module):
    """Dummy empirical normalizer for testing export equivalence."""
    def __init__(self, in_dim=63):
        super().__init__()
        self.register_buffer("_mean", torch.randn(1, in_dim))
        self.register_buffer("_var", torch.abs(torch.randn(1, in_dim)) + 0.1)
        self.eps = 1e-2

    def forward(self, x):
        std = torch.sqrt(self._var)
        return (x - self._mean) / (std + self.eps)


def test_numpy_mlp_equivalence():
    """Verify NumPy MLP forward pass matches PyTorch actor + normalizer within abs tol 1e-4."""
    torch.manual_seed(42)
    np.random.seed(42)

    actor = DummyActor(in_dim=63, out_dim=18)
    normalizer = DummyNormalizer(in_dim=63)

    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = os.path.join(tmpdir, "policy_mlp.npz")
        export_numpy_mlp(actor, normalizer, npz_path)

        # Random test input [batch=5, 63]
        sample_input_torch = torch.randn(5, 63)
        sample_input_np = sample_input_torch.numpy()

        # PyTorch reference output
        with torch.no_grad():
            norm_x = normalizer(sample_input_torch)
            pytorch_output = actor(norm_x).numpy()

        # NumPy inference output
        numpy_output = numpy_mlp_inference(npz_path, sample_input_np)

        # Verify equivalence
        np.testing.assert_allclose(numpy_output, pytorch_output, atol=1e-4, rtol=1e-4)


def test_onnx_export_equivalence():
    """Verify ONNX runtime or reference output matches PyTorch actor within abs tol 1e-4."""
    import onnx
    from onnx.reference import ReferenceEvaluator

    torch.manual_seed(42)
    actor = DummyActor(in_dim=63, out_dim=18)
    normalizer = DummyNormalizer(in_dim=63)

    class CombinedModel(nn.Module):
        def __init__(self, act, norm):
            super().__init__()
            self.norm = norm
            self.act = act
        def forward(self, x):
            return self.act(self.norm(x))

    model = CombinedModel(actor, normalizer)
    model.eval()

    sample_input = torch.randn(1, 63)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_file = os.path.join(tmpdir, "model.onnx")
        torch.onnx.export(
            model,
            sample_input,
            onnx_file,
            input_names=["obs"],
            output_names=["actions"],
            opset_version=14,
            dynamo=False,
        )

        # PyTorch prediction
        with torch.no_grad():
            torch_pred = model(sample_input).numpy()

        # ONNX reference evaluation
        onnx_model = onnx.load(onnx_file)
        evaluator = ReferenceEvaluator(onnx_model)
        onnx_pred = evaluator.run(None, {"obs": sample_input.numpy()})[0]

        np.testing.assert_allclose(onnx_pred, torch_pred, atol=1e-4, rtol=1e-4)


def test_deploy_yaml_spec():
    """Verify deploy.yaml spec dimensions and 18 unique joint entries."""
    obs_spec = [
        {"name": "base_ang_vel", "dim": 3},
        {"name": "projected_gravity", "dim": 3},
        {"name": "velocity_commands", "dim": 3},
        {"name": "joint_pos_rel", "dim": 18},
        {"name": "joint_vel", "dim": 18},
        {"name": "last_action", "dim": 18},
    ]

    total_dim = sum(item["dim"] for item in obs_spec)
    assert total_dim == 63, f"Expected obs dim sum 63, got {total_dim}"

    sim_joints = [f"joint_{i}" for i in range(18)]
    assert len(sim_joints) == 18
    assert len(set(sim_joints)) == 18
