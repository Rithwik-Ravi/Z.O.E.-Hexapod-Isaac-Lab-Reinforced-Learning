# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import sys
from unittest.mock import MagicMock

class OmniverseMockLoader:
    """Mock import loader for omni and pxr packages when running pure-Python tests."""
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("omni") or fullname.startswith("pxr") or fullname.startswith("isaacsim"):
            from importlib.machinery import ModuleSpec
            return ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        return MagicMock()

    def exec_module(self, module):
        pass

sys.meta_path.insert(0, OmniverseMockLoader())
