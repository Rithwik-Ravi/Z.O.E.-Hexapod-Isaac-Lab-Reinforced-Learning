# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
from unittest.mock import MagicMock

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:] = [p for p in sys.path if not p.startswith("/opt/ros")]
for _p in [os.path.join(_REPO_ROOT, "source/spdrbot3"), _REPO_ROOT]:
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

class MockPackage(MagicMock):
    __name__ = "Mock"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__path__ = []
        self.__name__ = "Mock"

    def __getattr__(self, name):
        if name == "__name__":
            return "Mock"
        return super().__getattr__(name)

class OmniverseMockLoader:
    """Mock import loader for omni and pxr packages when running pure-Python tests."""
    def find_spec(self, fullname, path, target=None):
        if any(fullname.startswith(p) for p in ["omni", "pxr", "isaacsim", "carb"]):
            from importlib.machinery import ModuleSpec
            spec = ModuleSpec(fullname, self, is_package=True)
            return spec
        return None

    def create_module(self, spec):
        mod = MockPackage()
        return mod

    def exec_module(self, module):
        pass

sys.meta_path.insert(0, OmniverseMockLoader())

# Pre-populate root packages with __path__ defined
for root_pkg in ["omni", "pxr", "isaacsim", "carb"]:
    if root_pkg not in sys.modules:
        m = MockPackage()
        sys.modules[root_pkg] = m


def pytest_configure(config):
    """Unregister ROS launch_testing / ament plugins that conflict with standalone Isaac Lab unit tests."""
    for name, plugin in list(config.pluginmanager.list_name_plugin()):
        mod_name = getattr(plugin, "__name__", "") or ""
        if any(k in str(name) or k in mod_name for k in ("launch_testing", "ament")):
            config.pluginmanager.unregister(plugin=plugin, name=name)

