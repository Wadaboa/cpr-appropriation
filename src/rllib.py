import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.agents.callbacks import DefaultCallbacks


class SocialOutcomeMetricsCallbacks(DefaultCallbacks):
    """
    Social outcome metrics callback to be used with RLlib
    """

    def on_episode_start(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        """
        Called after resetting the environment: deals with initializing custom metrics
        """
        # Make sure this episode has just been started
        assert (
            episode.length == 0
        ), "Start callback should only be called after resetting the environment"

        # Initialize metrics
        metrics = ["efficiency", "equality", "sustainability", "peace"]
        for metric in metrics:
            episode.user_data[metric] = []
            episode.hist_data[metric] = []

    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
        """
        Called after one step in the environment: deals with computing custom metrics
        """
        # Make sure this episode is ongoing
        assert (
            episode.length > 0
        ), "Step callback should only be called after stepping in the environment"

        # Update metrics
        metrics = base_env.envs[0].get_social_outcome_metrics()
        for metric, value in metrics.items():
            episode.user_data[metric].append(value)

    def on_episode_end(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        """
        Called after the final step in the environment: deals with storing custom metrics
        """
        # Make sure this episode is really done
        assert episode.batch_builder.policy_collectors["default_policy"].buffers[
            "dones"
        ][-1], "End callback should only be called after the episode is done"

        # Store final episode metrics
        mean_metrics = {k: np.mean(v) for k, v in episode.user_data.items()}
        for metric, value in mean_metrics:
            episode.custom_metrics[metric] = value
            episode.hist_data[metric] = episode.user_data[metric]


class FCNetwork(TorchModelV2, nn.Module):
    """
    Custom RLlib network that delegates to fully-connected layers
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.torch_sub_model = FullyConnectedNetwork(
            obs_space, action_space, num_outputs, model_config, name
        )

    def forward(self, input_dict, state, seq_lens):
        """
        Perform a forward pass by calling the underlying FC model
        """
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        """
        Returns the value function output for the most recent forward pass
        """
        return torch.reshape(self.torch_sub_model.value_function(), [-1])
