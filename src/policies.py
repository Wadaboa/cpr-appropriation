import os
from datetime import datetime
from collections import defaultdict

import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Categorical
from loguru import logger

from . import memory, utils


class VPGPolicy:
    """
    Vanilla Policy Gradient implementation
    """

    def __init__(self, env, policy_nn, baseline_nn=None, seed=42):
        assert isinstance(
            env, gym.Env
        ), "The given environment should be a Gym environment"
        assert isinstance(
            policy_nn, nn.Module
        ), "The given policy network should be a PyTorch module"
        assert baseline_nn is None or isinstance(
            baseline_nn, nn.Module
        ), "The given baseline network should be None or a PyTorch module"

        # Make the env a mult-agent one to have a single standard
        if not utils.is_multi_agent_env(env):
            env = utils.make_multi_agent(env)

        # Store parameters
        self.env = env
        self.policy_nn = policy_nn
        self.baseline_nn = baseline_nn

        # Fix random seed
        utils.set_seed(seed)

        # Store useful env variables
        self.n_agents = self.env.n_agents if hasattr(self.env, "n_agents") else 1
        self.max_steps = self.env._max_episode_steps
        self.action_space_size = self.env.action_space.n
        self.observation_space_size = int(np.prod(self.env.observation_space.shape))

        # Define losses
        self.losses = dict()
        self.losses["policy"] = nn.NLLLoss(
            ignore_index=utils.IGNORE_INDEX, reduction="mean"
        )
        if self.baseline_nn is not None:
            self.losses["baseline"] = nn.MSELoss(reduction="mean")

    def train(
        self,
        epochs,
        steps_per_epoch,
        policy_lr=1e-3,
        baseline_lr=1e-3,
        discount=0.99,
        save_every=None,
        checkpoints_path=None,
        enable_wandb=True,
        wandb_config=None,
        max_episodes=None,
        std_returns=True,
        episodes_mean_return=100,
    ):
        """
        Train VPG by running the specified number of episodes and
        maximum time steps
        """
        # Define optimizer with different learning rates for
        # policy and value networks
        params = [{"params": list(self.policy_nn.parameters())}]
        if self.baseline_nn is not None:
            params += [
                {"params": list(self.baseline_nn.parameters()), "lr": baseline_lr}
            ]
        optimizer = optim.Adam(params, lr=policy_lr)

        # Initialize wandb for logging
        if enable_wandb:
            wandb_config = {
                **wandb_config,
                "epochs": epochs,
                "policy_lr": policy_lr,
                "baseline_lr": baseline_lr,
                "discount": discount,
                "steps_per_epoch": steps_per_epoch,
                "baseline": self.baseline_nn is not None,
            }
            wandb_run = utils.init_wandb(config=wandb_config)

        # Iterate for the specified number of epochs
        current_episode = 0
        mean_returns = []
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1} / {epochs}")

            # Accumulate trajectories to fill-up a batch of examples
            trajectories = memory.TrajectoryPool()
            epoch_infos = defaultdict(list)
            epoch_episodes = 0
            while trajectories.get_timesteps() < steps_per_epoch:
                logger.info(f"Episode {current_episode + 1}")
                (
                    episode_trajectories,
                    episode_infos,
                    episode_mean_return,
                ) = self.execute_episode()
                if len(episode_infos) > 0:
                    logger.info(f"Episode infos: {episode_infos}")
                    for k, v in episode_infos.items():
                        epoch_infos[k] += [v]
                trajectories.extend(episode_trajectories)
                mean_returns += [episode_mean_return]
                logger.info(f"Mean episode return: {episode_mean_return}")
                logger.info(
                    f"Last {episodes_mean_return} episodes mean return: "
                    f"{np.mean(mean_returns[-episodes_mean_return:])}"
                )
                current_episode += 1
                epoch_episodes += 1

            # Get a batch of (s, a, r) tuples
            actual_batch_size = trajectories.get_timesteps()
            if actual_batch_size > steps_per_epoch:
                logger.warning(
                    f"The actual batch size is {actual_batch_size}, instead of {steps_per_epoch}"
                )
            states, actions, old_log_probs, returns, _ = trajectories.tensorify(
                discount=discount
            )

            # Compute log-probabilities of actions
            self.policy_nn.train(mode=True)
            log_probs = self.policy_nn(states)

            # Compute baseline
            values = torch.zeros_like(returns)
            if self.baseline_nn is not None:
                self.baseline_nn.train(mode=True)
                values = self.baseline_nn(states).squeeze()

            # Compute loss
            total_loss = self.compute_loss(
                returns,
                actions,
                values,
                log_probs,
                old_log_probs,
                std_returns=std_returns,
            )
            logger.info(f"Total loss: {total_loss}")

            # Compute mean epoch metrics
            if len(epoch_infos) > 0:
                epoch_infos = {k: np.mean(v) for k, v in epoch_infos.items()}
                logger.info(f"Epoch infos: {epoch_infos}")

            # Log to wandb at end of epoch
            if enable_wandb:
                wandb_run.log(
                    {
                        **epoch_infos,
                        "loss": total_loss,
                        "mean_return": np.mean(mean_returns[-epoch_episodes:]),
                    },
                    step=epoch,
                )

            # Backprop
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Save model checkpoints
            if save_every is not None and epoch % save_every == 0:
                logger.info(f"Saving model checkpoints to {checkpoints_path}")
                self.save(checkpoints_path)

            # Exit due to maximum episodes
            if max_episodes is not None and current_episode >= max_episodes:
                logger.info(
                    f"Reached the maximum number of episodes {max_episodes}, exiting"
                )
                break

        # Stop wandb logging
        if enable_wandb:
            wandb_run.finish()

    def compute_advantages(self, returns, values, log_probs, std_advs=False):
        """
        Compute advantages and possibly standardize them
        """
        advantages = returns - values
        if std_advs:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        return advantages.unsqueeze(-1).repeat(1, log_probs.shape[-1])

    def compute_loss(
        self,
        returns,
        actions,
        values,
        log_probs,
        old_log_probs=None,
        std_returns=True,
    ):
        # Compute loss
        if std_returns:
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        advantages = self.compute_advantages(returns, values, log_probs)
        total_loss = self.losses["policy"](log_probs * advantages, actions)
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
        trajectories = memory.TrajectoryPool(n=self.n_agents)
        observations = self.env.reset()

        # Iterate as long as agents are not done or until
        # we reach the maximum number of time-steps
        for _ in range(self.max_steps):
            # Compute the best actions based on the current policy
            action_dict, action_probs = dict(), dict()
            for agent_handle in range(self.n_agents):
                log_probs = self.policy_nn(
                    torch.tensor(observations[agent_handle], dtype=torch.float32)
                )
                action_probs[agent_handle] = log_probs.detach().numpy()
                action = np.random.choice(
                    np.arange(self.action_space_size),
                    p=np.exp(action_probs[agent_handle]),
                )
                action_dict[agent_handle] = action.item()

            # Perform a step in the environment
            new_observations, rewards, dones, infos = self.env.step(action_dict)

            # Update each agent's trajectory
            for agent_handle in range(self.n_agents):
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
            if "__all__" in dones and dones["__all__"]:
                logger.debug(f"Early stopping, all agents done")
                break

        # Store episode infos
        all_infos = infos["__all__"] if "__all__" in infos else dict()

        # Compute mean episode return
        agents_returns = [
            t.get_returns(discount=1, to_go=False, as_torch=False) for t in trajectories
        ]
        mean_return = np.mean(agents_returns)

        return trajectories, all_infos, mean_return

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
        seed=42,
        beta=1.0,
        kl_target=None,
    ):
        assert isinstance(
            beta, float
        ), "The beta hyperparameter should be given as a float"
        assert kl_target is None or isinstance(
            kl_target, float
        ), "The KL divergence target should be given as a float"
        super().__init__(env, policy_nn, baseline_nn=baseline_nn, seed=seed)
        self.beta = beta
        self.kl_target = kl_target
        self.losses["constraint"] = nn.KLDivLoss(log_target=True, reduction="batchmean")

    def compute_loss(
        self, returns, actions, values, log_probs, old_log_probs=None, std_returns=True
    ):
        # Compute advantages
        if std_returns:
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        advantages = self.compute_advantages(returns, values, log_probs)

        # Compute the probability ratio
        probs_ratio = torch.exp(log_probs - old_log_probs)

        # Compute total loss as the sum of TRPO loss, baseline loss and
        # the trust region constraint
        total_loss = self.losses["policy"](probs_ratio * advantages, actions)
        total_loss += self.losses["baseline"](values, returns)

        # Use KL divergence as regularizer and possibily update beta
        kl_div = self.losses["constraint"](old_log_probs, log_probs)
        total_loss += self.beta * kl_div
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
        seed=42,
        c1=1.0,
        c2=0.01,
        eps=0.2,
    ):
        assert isinstance(c1, float) and isinstance(
            c2, float
        ), "The c1 and c2 hyperparameters should be given as floats"
        super().__init__(env, policy_nn, baseline_nn=baseline_nn, seed=seed)
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
        std_returns=True,
    ):
        # Compute advantages
        if std_returns:
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        advantages = self.compute_advantages(returns, values, log_probs)

        # Compute the probability ratio
        probs_ratio = torch.exp(log_probs - old_log_probs)

        # Compute PPO loss as the minimum of the clipped and unclipped losses
        objective = torch.min(
            probs_ratio * advantages,
            torch.clamp(probs_ratio, 1 - self.eps, 1 + self.eps) * advantages,
        )

        # Compute total loss as the sum of PPO loss, baseline loss and
        # entropy of the categorical action distribution
        total_loss = self.losses["policy"](objective, actions)
        total_loss += self.c1 * self.losses["baseline"](values, returns)
        total_loss += self.c2 * Categorical(probs=torch.exp(log_probs)).entropy().mean()

        return total_loss
