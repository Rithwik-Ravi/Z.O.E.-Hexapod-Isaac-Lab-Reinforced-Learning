import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
from pxr import Usd, UsdGeom

for path in ["hexapod/hexa/hexa.usd", "spdr.usd", "spdr_stage.usd"]:
    abs_p = os.path.abspath(path)
    if os.path.exists(abs_p):
        stage = Usd.Stage.Open(abs_p)
        mpu = UsdGeom.GetStageMetersPerUnit(stage)
        bbox_cache = UsdGeom.BBoxCache(0, ['default', 'render'])
        bbox = bbox_cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
        print(f"[{path}] MetersPerUnit: {mpu}, BBox: {bbox}")

simulation_app.close()
