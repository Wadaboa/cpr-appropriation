from gym.envs.registration import register

register(
    id='cpr-grid-v0',
    entry_point='gym_cpr_grid.envs:CPRGridEnv',
)
register(
    id='cpr-grid-extrahard-v0',
    entry_point='gym_cpr_grid.envs:CPRGridExtraHardEnv',
)
