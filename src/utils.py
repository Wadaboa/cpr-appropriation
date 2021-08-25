import random
import os

import gym
import wandb
import torch
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.annotations import override


IGNORE_INDEX = -100


def init_wandb(config):
    """
    Return a new W&B run to be used for logging purposes
    """
    assert isinstance(config, dict), "The given W&B config should be a dictionary"
    assert "api_key" in config, "Missing API key value in W&B config"
    assert "project" in config, "Missing project name in W&B config"
    assert "entity" in config, "Missing entity name in W&B config"
    assert "group" in config, "Missing group name in W&B config"

    os.environ["WANDB_API_KEY"] = config["api_key"]
    exclude_keys = ["api_key", "project", "entity", "group"]
    remaining_config = {k: v for k, v in config.items() if k not in exclude_keys}
    return wandb.init(
        project=config["project"],
        entity=config["entity"],
        group=config["group"],
        config=remaining_config,
    )


def set_seed(seed):
    """
    Fix all possible sources of randomness
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def is_multi_agent_env(env):
    """
    Check if the given environment is multi-agent by calling the reset fuction
    and checking if the returned observations are in a dictionary or not
    """
    assert isinstance(env, gym.Env), "The given environment should be a Gym environment"
    return isinstance(env.reset(), dict)


def make_multi_agent(env, max_steps=1000):
    """
    Convenience wrapper for any single-agent env to be converted into MA
    """

    class MultiEnv(MultiAgentEnv, gym.Env):
        def __init__(self, env):
            self.env = env
            self.n_agents = 1
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space
            self._max_episode_steps = (
                self.env._max_episode_steps
                if hasattr(self.env, "_max_episode_steps")
                else max_steps
            )

        @override(MultiAgentEnv)
        def reset(self):
            return {i: self.env.reset() for i in range(self.n_agents)}

        @override(MultiAgentEnv)
        def step(self, action_dict):
            observations, rewards, dones, infos = {}, {}, {}, {}
            for agent_handle, action in action_dict.items():
                (
                    observations[agent_handle],
                    rewards[agent_handle],
                    dones[agent_handle],
                    infos[agent_handle],
                ) = self.env.step(action)
            dones["__all__"] = all(v for v in dones.values())
            return observations, rewards, dones, infos

        @override(MultiAgentEnv)
        def render(self, mode=None):
            return self.env.render(mode)

    return MultiEnv(env)


def get_torch_device():
    """
    Return a GPU PyTorch device (if one is available), otherwise a CPU
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
