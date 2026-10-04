import gymnasium as gym

gym.register(
    id="Zoe-Hexapod-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Spdrbot3RoughEnvCfg",
        "rsl_rl_cfg_entry_point": "spdrbot3.tasks.direct.spdrbot3.agents.rsl_rl_ppo_cfg:PPORunnerCfg",
        "skrl_amp_cfg_entry_point": "spdrbot3.tasks.direct.spdrbot3.agents:skrl_amp_cfg.yaml",
        "skrl_cfg_entry_point": "spdrbot3.tasks.direct.spdrbot3.agents:skrl_ppo_cfg.yaml",
    },
)
