import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray import tune
from ray.tune import JupyterNotebookReporter, CLIReporter
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.pg import PGTrainer
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLogger


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
        for metric, value in mean_metrics.items():
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


def rllib_baseline(
    algorithm,
    n_agents,
    grid_width,
    grid_height,
    wandb_project,
    wandb_api_key,
    log_dir,
    max_episodes,
    max_steps=1000,
    tagging_ability=False,
    gifting_mechanism=None,
    num_workers=1,
    jupyter=False,
    seed=42,
):
    """
    Run the DQN model described in the paper using RLlib's implementation
    or RLlib's vanilla PG algorithm
    """
    assert algorithm in ("vpg", "dqn")
    ModelCatalog.register_custom_model("fcn", FCNetwork)
    metric_columns = {
        "episodes_total": "episodes",
        "custom_metrics/efficiency_mean": "U",
        "custom_metrics/equality_mean": "E",
        "custom_metrics/sustainability_mean": "S",
        "custom_metrics/peace_mean": "P",
    }
    reporter = CLIReporter(metric_columns=metric_columns)
    if jupyter:
        reporter = JupyterNotebookReporter(
            overwrite=True,
            metric_columns=metric_columns,
        )
    config = {
        "env": "gym_cpr_grid:CPRGridEnv-v0",
        "env_config": {
            "n_agents": n_agents,
            "grid_width": grid_width,
            "grid_height": grid_height,
            "max_steps": max_steps,
            "tagging_ability": tagging_ability,
            "gifting_mechanism": gifting_mechanism,
            "add_social_outcome_metrics": False,
        },
        "horizon": max_steps,
        "num_workers": num_workers,
        "framework": "torch",
        "seed": seed,
        "model": {
            "custom_model": "fcn",
            "fcnet_hiddens": [32, 32],
            "fcnet_activation": "relu",
        },
        "callbacks": SocialOutcomeMetricsCallbacks,
        "logger_config": {
            "wandb": {
                "project": wandb_project,
                "api_key": wandb_api_key,
                "log_config": True,
                "sync_tensorboard": True,
            }
        },
        "num_gpus": 1 if torch.cuda.is_available() else 0,
    }
    if algorithm == "dqn":
        config = {
            **config,
            "exploration_config": {
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.1,
                "epsilon_timesteps": max_steps,
            },
        }
    return tune.run(
        DQNTrainer if algorithm == "dqn" else PGTrainer,
        config=config,
        progress_reporter=reporter,
        loggers=DEFAULT_LOGGERS + (WandbLogger,),
        checkpoint_at_end=True,
        local_dir=log_dir,
        stop=lambda _, result: result["episodes_total"] >= max_episodes,
    )
