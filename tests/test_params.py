# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Z.O.E. Hexapod params module, YAML deduplication, and derived asset constants."""

import math
import os
import pytest
import torch
import yaml

from spdrbot3.tasks.manager_based.zoe_deploy.params import (
    ZOE_PARAMS,
    ZoeParams,
    get_foot_tip_offsets_tensor,
)
from spdrbot3.tasks.manager_based.zoe_deploy.mdp.rewards import (
    _compute_foot_tip_state,
    fewer_than_4_feet_contact,
    torque_limit_penalty,
)


class UniqueKeyLoader(yaml.SafeLoader):
    """YAML SafeLoader that raises ValueError if any key is duplicated in a mapping or across the file."""

    def __init__(self, stream):
        super().__init__(stream)
        self.seen_keys_global = set()

    def construct_mapping(self, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"Duplicate YAML key '{key}' within mapping at {key_node.start_mark}")
            if key in self.seen_keys_global:
                raise ValueError(f"Duplicate YAML key '{key}' across file at {key_node.start_mark}")
            self.seen_keys_global.add(key)
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def test_no_duplicate_yaml_keys():
    """Verify deploy/actuator_model.yaml and deploy/mass_properties.yaml have strictly one value per key."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for rel_path in ["deploy/actuator_model.yaml", "deploy/mass_properties.yaml"]:
        full_path = os.path.join(repo_root, rel_path)
        assert os.path.exists(full_path), f"Missing required YAML config: {full_path}"
        with open(full_path, "r") as f:
            data = yaml.load(f, Loader=UniqueKeyLoader)
        assert isinstance(data, dict) and len(data) > 0


def test_params_structure():
    assert isinstance(ZOE_PARAMS, ZoeParams)
    assert ZOE_PARAMS.commands.lin_vel_x_range[0] < ZOE_PARAMS.commands.lin_vel_x_range[1]
    assert ZOE_PARAMS.commands.lin_vel_y_range[0] < ZOE_PARAMS.commands.lin_vel_y_range[1]
    assert ZOE_PARAMS.commands.ang_vel_z_range[0] < ZOE_PARAMS.commands.ang_vel_z_range[1]


def test_params_bounds():
    assert ZOE_PARAMS.obs_scale.base_ang_vel_scale > 0.0
    assert ZOE_PARAMS.obs_scale.joint_vel_scale > 0.0
    assert ZOE_PARAMS.events.max_action_delay_steps >= 0
    assert ZOE_PARAMS.rewards.termination_penalty < 0.0
    assert ZOE_PARAMS.rewards.fewer_than_3_feet_contact < 0.0
    assert ZOE_PARAMS.rewards.fewer_than_4_feet_contact < 0.0
    assert ZOE_PARAMS.rewards.torque_limit < 0.0


def test_mass_randomization_bounds():
    """Verify mass scale ranges match Chunk-1 strict bisection (0.90, 1.10) during Task 3."""
    base_min, base_max = ZOE_PARAMS.events.base_mass_scale_range
    body_min, body_max = ZOE_PARAMS.events.body_mass_scale_range

    assert (base_min, base_max) == (0.90, 1.10)
    assert (body_min, body_max) == (0.90, 1.10)

    # Reference nominal base link mass from audit report (2.5027 kg on new hexa_ir robot)
    ref_base_mass = ZOE_PARAMS.BASE_LINK_MASS_KG
    assert ref_base_mass > 2.0, f"Expected new base_link mass > 2.0 kg, got {ref_base_mass}"
    min_possible_base_mass = ref_base_mass * base_min
    assert min_possible_base_mass > 0.10, f"Min base mass {min_possible_base_mass} kg is dangerously near zero!"


def test_set_payload_mass_reconstruction():
    """Verify scripts/set_payload_mass.py exact mass and inertia reconstruction (Task 7)."""
    import sys
    import os
    sys.path.append(os.path.abspath("scripts"))
    from set_payload_mass import compute_payload_mass_properties, CAD_BASE_MASS_KG, CAD_SIX_LEGS_MASS_KG

    for target_m in [4.50, 5.00, 5.50]:
        props = compute_payload_mass_properties(target_m)
        assert abs(props["total_mass_kg"] - target_m) < 1e-6
        assert abs(props["base_link_mass_kg"] + CAD_SIX_LEGS_MASS_KG - target_m) < 1e-6
        assert abs(props["payload_added_kg"] - (props["base_link_mass_kg"] - CAD_BASE_MASS_KG)) < 1e-6

        # Inertia scales linearly
        scale = props["inertia_scale"]
        assert abs(scale - (props["base_link_mass_kg"] / CAD_BASE_MASS_KG)) < 1e-6
        assert abs(props["scaled_inertia"]["ixx"] - (0.010209323161935659 * scale)) < 1e-9
        assert abs(props["scaled_inertia"]["izz"] - (0.019355228545140914 * scale)) < 1e-9

    # Negative/too low mass raises ValueError
    import pytest
    with pytest.raises(ValueError):
        compute_payload_mass_properties(1.0)


def test_derived_asset_and_actuator_constants():
    """Verify ZOE_PARAMS derived constants match audit report and Waveshare ST3215 12V specs."""
    assert ZOE_PARAMS.BASE_BODY_NAME == "base_link"
    assert abs(ZOE_PARAMS.TOTAL_ASSET_MASS_KG - 5.000) <= 0.05, f"Expected total mass ~5.0 kg (+-0.05), got {ZOE_PARAMS.TOTAL_ASSET_MASS_KG}"
    assert abs(ZOE_PARAMS.STANDING_BASE_HEIGHT - 0.1417) < 0.01
    assert abs(ZOE_PARAMS.GATE_MIN_HEIGHT - round(0.90 * ZOE_PARAMS.STANDING_BASE_HEIGHT, 4)) < 1e-6
    assert abs(ZOE_PARAMS.TARGET_STANDING_HEIGHT - round(0.95 * ZOE_PARAMS.STANDING_BASE_HEIGHT, 4)) < 1e-6

    # Waveshare ST3215 12V specs
    assert abs(ZOE_PARAMS.STALL_TORQUE_12V_NM - 2.942) < 1e-3
    assert abs(ZOE_PARAMS.EFFORT_LIMIT_NM - 2.50) < 1e-3
    assert abs(ZOE_PARAMS.NO_LOAD_SPEED_RAD_S - 4.71) < 1e-3

    # 18 joints and servo IDs 1..18
    assert len(ZOE_PARAMS.joint_info) == 18
    servo_ids = {j["servo_id"] for j in ZOE_PARAMS.joint_info}
    assert servo_ids == set(range(1, 19))
    for j in ZOE_PARAMS.joint_info:
        k = {"coxa": 1, "thigh": 2, "foot": 3}[j["type"]]
        assert j["servo_id"] == 3 * (j["leg"] - 1) + k

    # 3D foot tip vectors in NFOOT1..6 frames
    foot_names = [f"NFOOT{i}" for i in range(1, 7)]
    offsets = get_foot_tip_offsets_tensor(foot_names, device="cpu", dtype=torch.float32)
    assert offsets.shape == (6, 3)
    norms = torch.norm(offsets, dim=-1)
    assert torch.all((norms > 0.160) & (norms < 0.170))


def test_reward_penalties_and_3d_tip_transform():
    """Verify fewer_than_4_feet_contact, torque_limit_penalty, and _compute_foot_tip_state."""

    class DummyContactData:
        def __init__(self, forces_z):
            self.net_forces_w = torch.zeros((forces_z.shape[0], 6, 3))
            self.net_forces_w[:, :, 2] = forces_z

    class DummySensor:
        def __init__(self, forces_z):
            self.data = DummyContactData(forces_z)

    class DummyRobotData:
        def __init__(self):
            self.applied_torque = torch.tensor([[1.5, 2.0, 2.5] + [0.0] * 15])
            self.body_pos_w = torch.zeros((1, 6, 3))
            self.body_quat_w = torch.zeros((1, 6, 4))
            self.body_quat_w[:, :, 0] = 1.0  # identity quaternion
            self.body_lin_vel_w = torch.zeros((1, 6, 3))
            self.body_ang_vel_w = torch.zeros((1, 6, 3))

    class DummyRobot:
        def __init__(self):
            self.data = DummyRobotData()
            self.body_names = [f"NFOOT{i}" for i in range(1, 7)]

    class DummySensorCfg:
        name = "contact_forces"
        body_ids = [0, 1, 2, 3, 4, 5]

    class DummyEnv:
        def __init__(self, forces_z):
            self.scene = {
                "contact_forces": DummySensor(forces_z),
                "robot": DummyRobot(),
            }

    forces_z = torch.tensor([
        [5.0, 5.0, 5.0, 5.0, 0.0, 0.0],  # 4 feet down -> 0 penalty
        [5.0, 5.0, 5.0, 0.0, 0.0, 0.0],  # 3 feet down -> (4-3)^2 = 1.0
        [5.0, 5.0, 0.0, 0.0, 0.0, 0.0],  # 2 feet down -> (4-2)^2 = 4.0
    ])
    env = DummyEnv(forces_z)
    pen4 = fewer_than_4_feet_contact(env, DummySensorCfg())
    assert torch.allclose(pen4, torch.tensor([0.0, 1.0, 4.0]))

    tau_pen = torque_limit_penalty(env, threshold=2.0)
    # Only 2.5 exceeds 2.0 by 0.5 -> 0.5^2 = 0.25
    assert torch.allclose(tau_pen, torch.tensor([0.25]))

    tip_pos_w, tip_vel_w = _compute_foot_tip_state(env.scene["robot"], DummySensorCfg())
    assert tip_pos_w.shape == (1, 6, 3)
    assert tip_vel_w.shape == (1, 6, 3)
    expected_offsets = get_foot_tip_offsets_tensor([f"NFOOT{i}" for i in range(1, 7)], "cpu", torch.float32)
    assert torch.allclose(tip_pos_w[0], expected_offsets, atol=1e-5)


