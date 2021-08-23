import os
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import wandb
from torch.distributions import Categorical
from loguru import logger

from . import memory


IGNORE_INDEX = -100


def init_wandb(config):
    """
    Return a new W&B run to be used for logging purposes
    """
    assert isinstance(config, dict), "The given W&B config should be a dictionary"
    assert "api_key" in config, "Missing API key value in W&B config"
    assert "group" in config, "Missing group name in W&B config"
    assert "project" in config, "Missing project name in W&B config"
    assert "entity" in config, "Missing entity name in W&B config"

    os.environ["WANDB_API_KEY"] = config["api_key"]
    exclude_keys = ["api_key", "project", "entity", "group"]
    remaining_config = {k: v for k, v in config.items() if k not in exclude_keys}
    return wandb.init(
        project=config["project"],
        entity=config["entity"],
        group=config["group"],
        config=remaining_config,
    )


class VPGPolicy:
    """
    Vanilla Policy Gradient implementation
    """

    def __init__(self, env, policy_nn, baseline_nn=None):
        # Store parameters
        self.env = env
        self.policy_nn = policy_nn
        self.baseline_nn = baseline_nn

        # Define losses
        self.losses = dict()
        self.losses["policy"] = nn.NLLLoss(ignore_index=IGNORE_INDEX, reduction="mean")
        if self.baseline_nn is not None:
            self.losses["baseline"] = nn.MSELoss(reduction="mean")

    def train(
        self,
        max_epochs,
        lr=1e-3,
        discount=0.99,
        batch_size=32,
        save_every=None,
        checkpoints_path=None,
        wandb=True,
        wandb_config=None,
    ):
        """
        Train VPG by running the specified number of episodes and
        maximum time steps
        """
        # Define optimizer
        params = list(self.policy_nn.parameters())
        if self.baseline_nn is not None:
            params += list(self.baseline_nn.parameters())
        optimizer = optim.Adam(params, lr=lr)

        # Initialize wandb for logging
        if wandb:
            wandb_config = {
                **wandb_config,
                "epochs": max_epochs,
                "lr": lr,
                "discount": discount,
                "batch_size": batch_size,
                "baseline": self.baseline_nn is not None,
            }
            wandb_run = init_wandb(config=wandb_config)

        # Iterate for the specified number of epochs
        metrics = defaultdict(int)
        for epoch in range(max_epochs):
            logger.info(f"Epoch {epoch + 1} / {max_epochs}")

            # Accumulate trajectories to fill-up a batch of examples
            trajectories = memory.TrajectoryPool()
            for _ in range(batch_size // self.env.n_agents):
                episode_trajectories = self.execute_episode()
                social_outcome_metrics = self.env.get_social_outcome_metrics()
                for m in social_outcome_metrics:
                    metrics[m] += social_outcome_metrics[m]
                trajectories.extend(episode_trajectories)

            # Get a batch of (s, a, r) tuples
            logger.info(f"Working with a batch size of {len(trajectories)}")
            states, actions, old_log_probs, returns, _ = trajectories.tensorify(
                self.env.max_steps,
                self.env.observation_space_size(flattened=False),
                self.env.action_space_size(),
                discount=discount,
                ignore_index=IGNORE_INDEX,
            )

            # Compute log-probabilities of actions
            self.policy_nn.train(mode=True)
            log_probs = self.policy_nn(states)

            # Compute baseline
            values = torch.zeros_like(returns)
            if self.baseline_nn is not None:
                values = self.baseline_nn(states).squeeze()

            # Compute loss
            total_loss = self.compute_loss(
                returns,
                actions,
                values,
                log_probs,
                old_log_probs,
            )

            # Log to wandb at end of epoch
            if wandb:
                mean_metrics = {k: v / (epoch + 1) for k, v in metrics.items()}
                wandb_run.log({**mean_metrics, "loss": total_loss}, step=epoch)

            # Backprop
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # Save model checkpoints
        if save_every is not None and epoch % save_every == 0:
            self.save(checkpoints_path)

        # Stop wandb logging
        wandb_run.finish()

    def compute_loss(
        self,
        returns,
        actions,
        values,
        log_probs,
        old_log_probs=None,
    ):
        # Compute loss
        advantage = (returns - values).unsqueeze(-1).repeat(1, 1, log_probs.shape[-1])
        total_loss = self.losses["policy"](
            torch.flatten(log_probs * advantage, start_dim=0, end_dim=1),
            torch.flatten(actions),
        )
        if self.baseline_nn is not None:
            total_loss += self.losses["baseline"](values, returns)
        return total_loss

    def execute_episode(self):
        """
        Run an episode for the maximum number of time-steps defined
        in the environment and return a list of trajectories,
        one for each agent
        """
        # Initialize trajectories and reset environment
        self.policy_nn.eval()
        trajectories = memory.TrajectoryPool(n=self.env.n_agents)
        observations = self.env.reset()

        # Iterate as long as agents are not done or until
        # we reach the maximum number of time-steps
        for _ in range(self.env.max_steps):
            # Compute the best actions based on the current policy
            action_dict, action_probs = dict(), dict()
            for agent_handle in range(self.env.n_agents):
                log_probs = self.policy_nn(
                    torch.tensor(observations[agent_handle], dtype=torch.float32)
                )
                action_probs[agent_handle] = log_probs.detach().numpy()
                action = np.random.choice(
                    np.arange(self.env.action_space_size()),
                    p=np.exp(action_probs[agent_handle]),
                )
                action_dict[agent_handle] = action.item()

            # Perform a step in the environment
            new_observations, rewards, dones, _ = self.env.step(action_dict)

            # Update each agent's trajectory
            for agent_handle in range(self.env.n_agents):
                trajectories.add_to_trajectory(
                    agent_handle,
                    observations[agent_handle],
                    action_dict[agent_handle],
                    action_probs[agent_handle],
                    rewards[agent_handle],
                    new_observations[agent_handle],
                )

            # Update observations and possibly stop
            # if all agents are done
            observations = new_observations
            if dones["__all__"]:
                logger.info(f"Early stopping, all agents done")
                break

        return trajectories

    def save(self, path):
        """
        Save policy network and baseline network (if used) to the given directory
        """
        assert os.path.exists(path), "The given path does not exist"
        assert not os.path.isfile(
            path
        ), "The given path should be a directory, not a file"
        prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.mkdir(os.path.join(path, prefix))
        torch.save(self.policy_nn.state_dict(), os.path.join(path, prefix, "policy"))
        if self.baseline_nn is not None:
            torch.save(
                self.baseline_nn.state_dict(), os.path.join(path, prefix, "baseline")
            )

    def load(self, path):
        """
        Load the policy and baseline network (if used) from the given path
        """
        assert os.path.exists(path), "The given path does not exist"
        self.policy_nn.load_state_dict(torch.load(os.path.join(path, "policy")))
        if self.baseline_nn is not None:
            self.baseline_nn.load_state_dict(torch.load(os.path.join(path, "baseline")))


class TRPOPolicy(VPGPolicy):
    """
    Trust Region Policy Optimization implementation - the KL divergence constraint
    is implemented through regularization (as explained in PPO's paper)
    """

    def __init__(
        self,
        env,
        policy_nn,
        baseline_nn,
        beta=1.0,
        kl_target=0.01,
    ):
        assert isinstance(
            beta, float
        ), "The beta hyperparameter should be given as a float"
        assert kl_target is None or isinstance(
            kl_target, float
        ), "The KL divergence target should be given as a float"
        super().__init__(env, policy_nn, baseline_nn=baseline_nn)
        self.beta = beta
        self.kl_target = kl_target
        self.losses["constraint"] = nn.KLDivLoss(log_target=True, reduction="batchmean")

    def compute_loss(
        self,
        returns,
        actions,
        values,
        log_probs,
        old_log_probs=None,
    ):
        # Compute the advantage
        advantage = (returns - values).unsqueeze(-1).repeat(1, 1, log_probs.shape[-1])

        # Compute the probability ratio
        probs_ratio = torch.exp(log_probs - old_log_probs)

        # Compute total loss as the sum of TRPO loss, baseline loss and
        # the trust region constraint
        total_loss = self.losses["policy"](
            torch.flatten(probs_ratio * advantage, start_dim=0, end_dim=1),
            torch.flatten(actions),
        )
        total_loss += self.losses["baseline"](values, returns)

        # Use KL divergence as regularizer and possibily update beta
        kl_div = self.losses["constraint"](old_log_probs, log_probs)
        total_loss -= self.beta * kl_div
        if self.kl_target is not None:
            if kl_div < (self.kl_target / 1.5):
                self.beta /= 2
            elif kl_div > (self.kl_target * 1.5):
                self.beta *= 2

        return total_loss


class PPOPolicy(VPGPolicy):
    """
    Proximal Policy Optimization implementation
    """

    def __init__(
        self,
        env,
        policy_nn,
        baseline_nn,
        c1=1.0,
        c2=0.0,
        eps=0.2,
    ):
        assert isinstance(c1, float) and isinstance(
            c2, float
        ), "The c1 and c2 hyperparameters should be given as floats"
        super().__init__(env, policy_nn, baseline_nn=baseline_nn)
        self.c1 = c1
        self.c2 = c2
        self.eps = eps

    def compute_loss(
        self,
        returns,
        actions,
        values,
        log_probs,
        old_log_probs=None,
    ):
        # Compute the advantage
        advantage = (returns - values).unsqueeze(-1).repeat(1, 1, log_probs.shape[-1])

        # Compute the probability ratio
        probs_ratio = torch.exp(log_probs - old_log_probs)

        # Compute PPO loss as the minimum of the clipped and unclipped losses
        objective = torch.min(
            probs_ratio * advantage,
            torch.clamp(probs_ratio, 1 - self.eps, 1 + self.eps) * advantage,
        )

        # Compute total loss as the sum of PPO loss, baseline loss and
        # entropy of the categorical action distribution
        total_loss = self.losses["policy"](
            torch.flatten(objective, start_dim=0, end_dim=1),
            torch.flatten(actions),
        )
        total_loss -= self.c1 * self.losses["baseline"](values, returns)
        total_loss += self.c2 * Categorical(probs=torch.exp(log_probs)).entropy().mean()

        return total_loss
