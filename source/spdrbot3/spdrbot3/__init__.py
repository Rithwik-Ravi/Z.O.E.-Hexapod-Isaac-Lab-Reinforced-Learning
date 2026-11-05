# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
try:
    from .tasks import *
except Exception:
    pass

# Register UI extensions.
try:
    from .ui_extension_example import *
except Exception:
    pass

