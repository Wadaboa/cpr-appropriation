from gym.envs.registration import register

register(
    id="CPRGridEnv-v0",
    entry_point="gym_cpr_grid.cpr_grid:CPRGridEnv",
)
