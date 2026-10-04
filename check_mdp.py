import isaaclab.envs.mdp as mdp_env
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import inspect

print("Functions in isaaclab.envs.mdp containing 'reset_root':")
for name, func in inspect.getmembers(mdp_env, inspect.isfunction):
    if "reset_root" in name:
        print(name)

print("\nFunctions in isaaclab_tasks.manager_based.locomotion.velocity.mdp containing 'reset_root':")
for name, func in inspect.getmembers(mdp, inspect.isfunction):
    if "reset_root" in name:
        print(name)
