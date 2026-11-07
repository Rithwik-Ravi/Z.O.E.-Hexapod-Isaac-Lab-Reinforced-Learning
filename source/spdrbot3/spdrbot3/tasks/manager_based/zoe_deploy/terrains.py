# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain Generator Configurations for Z.O.E. Hexapod tasks."""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

# Stage 1: Flat Terrain Generator Config
ZOE_FLAT_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,  # A20 fix: Disable cache to prevent stale terrain reuse
    sub_terrains={
        "flat": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=(0.0, 0.0), noise_step=0.02, border_width=0.25
        ),
    },
)

# Stage 2: Multi-Terrain World Generator Config (Rescaled for 0.15m stance height)
ZOE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    curriculum=True,
    use_cache=False,  # A20 fix
    sub_terrains={
        "flat": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.10, noise_range=(0.0, 0.0), noise_step=0.02, border_width=0.25
        ),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.20, noise_range=(0.005, 0.03), noise_step=0.005, border_width=0.25
        ),
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.12, slope_range=(0.0, 0.30), platform_width=2.0
        ),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.12, slope_range=(0.0, 0.30), platform_width=2.0
        ),
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.12, step_height_range=(0.01, 0.06), step_width=0.30, platform_width=2.0
        ),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.10, step_height_range=(0.01, 0.06), step_width=0.30, platform_width=2.0
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.14, grid_width=0.45, grid_height_range=(0.01, 0.05), platform_width=2.0
        ),
    },
)
