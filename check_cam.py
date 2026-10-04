import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Zoe-Hexapod-Rough-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks
import omni.usd
from pxr import UsdGeom, Gf
from isaaclab_tasks.utils import parse_env_cfg
import spdrbot3.tasks

env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
env = gym.make(args_cli.task, cfg=env_cfg)
env.reset()

stage = omni.usd.get_context().get_stage()
cam_prim = stage.GetPrimAtPath("/OmniverseKit_Persp")
if cam_prim.IsValid():
    print("[SUCCESS] Found /OmniverseKit_Persp camera!")
    xform = UsdGeom.Xformable(cam_prim)
    ops = {op.GetName(): op for op in xform.GetOrderedXformOps()}
    if "xformOp:translate" in ops:
        ops["xformOp:translate"].Set(Gf.Vec3d(0.8, 0.8, 0.6))
        print("[SUCCESS] Updated xformOp:translate on /OmniverseKit_Persp to (0.8, 0.8, 0.6)")
    if "xformOp:rotateXYZ" in ops:
        ops["xformOp:rotateXYZ"].Set(Gf.Vec3d(-35.0, 45.0, 0.0))
        print("[SUCCESS] Updated xformOp:rotateXYZ on /OmniverseKit_Persp")

env.close()
simulation_app.close()
