import os
from datetime import datetime
from collections import defaultdict

import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from loguru import logger
from IPython import display

from . import memory, utils


class VPGPolicy:
    """
    Vanilla Policy Gradient implementation
    """

    def __init__(self, env, policy_nn, baseline_nn=None, alpha=0.5, seed=42):
        assert isinstance(
            env, gym.Env
        ), "The given environment should be a Gym environment"
        assert isinstance(
            policy_nn, nn.Module
        ), "The given policy network should be a PyTorch module"
        assert baseline_nn is None or isinstance(
            baseline_nn, nn.Module
        ), "The given baseline network should be None or a PyTorch module"
        assert isinstance(
            alpha, float
        ), "The alpha hyperparameter should be given as float"

        # Make the env a mult-agent one to have a single standard
        if not utils.is_multi_agent_env(env):
            env = utils.make_multi_agent(env)

        # Store parameters
        self.env = env
        self.policy_nn = policy_nn
        self.baseline_nn = baseline_nn
        self.alpha = alpha

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
        minibatch_size,
        policy_lr=1e-3,
        baseline_lr=1e-3,
        discount=0.99,
        save_every=None,
        checkpoints_path=None,
        enable_wandb=True,
        wandb_config=None,
        max_episodes=None,
        std_returns=True,
        std_advs=False,
        episodes_mean_return=100,
        render_every=None,
        clip_gradient_norm=0.5,
    ):
        """
        Train VPG by running the specified number of episodes and
        maximum time steps
        """
        assert (
            steps_per_epoch % minibatch_size == 0
        ), "The number of steps per epoch should be a multiple of the mini-batch size"
        assert clip_gradient_norm is None or isinstance(
            clip_gradient_norm, float
        ), "The gradient clipping parameter should be None for no clipping or a float"

        # Define optimizer with different learning rates for
        # policy and value networks
        params = [{"params": list(self.policy_nn.parameters()), "lr": policy_lr}]
        if self.baseline_nn is not None:
            params += [
                {"params": list(self.baseline_nn.parameters()), "lr": baseline_lr}
            ]
        optimizer = optim.Adam(params)

        # Initialize wandb for logging
        if enable_wandb:
            wandb_config = {
                **wandb_config,
                "epochs": epochs,
                "steps_per_epoch": steps_per_epoch,
                "minibatch_size": minibatch_size,
                "policy_lr": policy_lr,
                "baseline_lr": baseline_lr,
                "discount": discount,
                "baseline": self.baseline_nn is not None,
                "std_returns": std_returns,
                "std_advs": std_advs,
                "clip_gradient_norm": clip_gradient_norm,
            }
            wandb_run = utils.init_wandb(config=wandb_config)

        # Iterate for the specified number of epochs
        current_episode = 0
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1} / {epochs}")

            # Accumulate trajectories to fill-up a batch of examples
            (
                trajectories,
                current_episode,
                epoch_infos,
                epoch_returns,
            ) = self.collect_trajectories(
                current_episode,
                steps_per_epoch,
                minibatch_size,
                discount=discount,
                episodes_mean_return=episodes_mean_return,
                render_every=render_every,
            )

            # Check if the trajectories collected are more than
            # the selected ones
            actual_batch_size = len(trajectories)
            if actual_batch_size > steps_per_epoch:
                logger.warning(
                    f"The actual batch size is {actual_batch_size}, instead of {steps_per_epoch}"
                )

            # Set networks to training mode
            self.policy_nn.train(mode=True)
            if self.baseline_nn is not None:
                self.baseline_nn.train(mode=True)

            # Perform mini-batch updates
            epoch_loss = 0.0
            num_minibatches = actual_batch_size // minibatch_size
            for minibatch, (states, actions, old_log_probs, returns, _) in enumerate(
                trajectories
            ):
                logger.info(f"Mini-batch {minibatch + 1} / {num_minibatches}")
                minibatch_losses = self.minibatch_update(
                    optimizer,
                    states,
                    actions,
                    old_log_probs,
                    returns,
                    std_returns=std_returns,
                    clip_gradient_norm=clip_gradient_norm,
                )
                epoch_loss += minibatch_losses["total_loss"]

            # Compute mean epoch metrics
            if len(epoch_infos) > 0:
                epoch_infos = {k: np.mean(v) for k, v in epoch_infos.items()}
                logger.info(f"Epoch infos: {epoch_infos}")

            # Log to wandb at end of epoch
            if enable_wandb:
                wandb_run.log(
                    {
                        **epoch_infos,
                        "loss": epoch_loss,
                        "mean_return": np.mean(epoch_returns),
                    },
                    step=epoch,
                )

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

    def collect_trajectories(
        self,
        current_episode,
        steps_per_epoch,
        minibatch_size,
        discount=0.99,
        episodes_mean_return=100,
        render_every=None,
    ):
        """
        Collect a buffer of timesteps of size at least `steps_per_epoch`
        """
        trajectories = memory.TrajectoryPool(
            discount=discount, minibatch_size=minibatch_size
        )
        epoch_infos = defaultdict(list)
        epoch_returns = []
        while len(trajectories) < steps_per_epoch:
            logger.info(f"Episode {current_episode + 1}")
            (
                episode_trajectories,
                episode_infos,
                episode_mean_return,
            ) = self.execute_episode(
                render=(
                    render_every is not None and current_episode % render_every == 0
                )
            )
            if len(episode_infos) > 0:
                logger.info(f"Episode infos: {episode_infos}")
                for k, v in episode_infos.items():
                    epoch_infos[k] += [v]
            trajectories.extend(episode_trajectories)
            epoch_returns += [episode_mean_return]
            logger.info(f"Mean episode return: {episode_mean_return}")
            logger.info(
                f"Last {episodes_mean_return} episodes mean return: "
                f"{np.mean(epoch_returns[-episodes_mean_return:])}"
            )
            current_episode += 1

        return trajectories, current_episode, epoch_infos, epoch_returns

    def minibatch_update(
        self,
        optimizer,
        states,
        actions,
        old_log_probs,
        returns,
        std_returns=True,
        std_advs=False,
        clip_gradient_norm=0.5,
    ):
        """
        Perform a mini-batch step and update network parameters
        """
        # Zero-out gradients
        optimizer.zero_grad()

        # Compute log-probabilities of actions
        log_probs = self.policy_nn(states)

        # Compute baseline
        values = torch.zeros_like(returns, device=utils.get_torch_device())
        if self.baseline_nn is not None:
            values = self.baseline_nn(states).squeeze()

        # Compute loss
        losses = self.compute_loss(
            returns,
            actions,
            values,
            log_probs,
            old_log_probs,
            std_returns=std_returns,
            std_advs=std_advs,
        )
        logger.info(f"Losses: {dict([(k, v.item()) for k, v in losses.items()])}")

        # Backward pass
        losses["total_loss"].backward()

        # Log gradient norms
        logger.info(
            f"Policy network L2 gradient norm: {self.policy_nn.get_gradient_norm()}"
        )
        if self.baseline_nn is not None:
            logger.info(
                f"Baseline network L2 gradient norm: {self.baseline_nn.get_gradient_norm()}"
            )

        # Clip gradient norms
        if clip_gradient_norm is not None:
            nn.utils.clip_grad_norm_(self.policy_nn.parameters(), clip_gradient_norm)
            logger.info(
                f"Policy network L2 gradient norm after clipping: {self.policy_nn.get_gradient_norm()}"
            )
            if self.baseline_nn is not None:
                nn.utils.clip_grad_norm_(
                    self.baseline_nn.parameters(), clip_gradient_norm
                )
                logger.info(
                    f"Baseline network L2 gradient norm after clipping: {self.baseline_nn.get_gradient_norm()}"
                )

        # Update parameters
        optimizer.step()

        return losses

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
        std_advs=False,
    ):
        """
        Compute VPG loss function
        """
        losses = dict()
        if std_returns:
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        advantages = self.compute_advantages(
            returns, values, log_probs, std_advs=std_advs
        )
        losses["policy_loss"] = self.losses["policy"](log_probs * advantages, actions)
        if self.baseline_nn is not None:
            losses["baseline_loss"] = self.losses["baseline"](values, returns)
        losses["total_loss"] = losses["policy_loss"] + (
            self.alpha * losses["baseline_loss"] if self.baseline_nn is not None else 0
        )
        return losses

    def execute_episode(self, render=False):
        """
        Run an episode for the maximum number of time-steps defined
        in the environment and return a list of trajectories,
        one for each agent
        """
        # Initialize trajectories and reset environment
        self.policy_nn.eval()
        trajectories = memory.TrajectoryPool(n=self.n_agents)
        observations = self.env.reset()

        # Initialize plot
        if render:
            _, _, img = self.env.plot(self.env.render("rgb_array"))

        # Iterate as long as agents are not done or until
        # we reach the maximum number of time-steps
        for _ in range(self.max_steps):
            # Plot previous observations
            if render:
                display.display(plt.gcf())

            # Compute the best actions based on the current policy
            action_dict, action_probs = dict(), dict()
            for agent_handle in range(self.n_agents):
                log_probs = self.policy_nn(
                    torch.tensor(
                        observations[agent_handle],
                        dtype=torch.float32,
                        device=utils.get_torch_device(),
                    )
                )
                action_probs[agent_handle] = log_probs.cpu().detach().numpy()
                action = np.random.choice(
                    np.arange(self.action_space_size),
                    p=np.exp(action_probs[agent_handle]),
                )
                action_dict[agent_handle] = action.item()

            # Perform a step in the environment
            new_observations, rewards, dones, infos = self.env.step(action_dict)

            # Update rendering
            if render:
                display.clear_output(wait=True)
                img.set_data(self.env.render(mode="rgb_array"))

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
        agents_returns = trajectories.get_trajectory_returns(
            discount=1, to_go=False, as_torch=False
        )
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
        alpha=0.5,
        beta=1.0,
        kl_target=None,
    ):
        assert isinstance(
            beta, float
        ), "The beta hyperparameter should be given as a float"
        assert kl_target is None or isinstance(
            kl_target, float
        ), "The KL divergence target should be given as None or a float"
        super().__init__(
            env, policy_nn, baseline_nn=baseline_nn, alpha=alpha, seed=seed
        )
        self.beta = beta
        self.kl_target = kl_target
        self.losses["constraint"] = nn.KLDivLoss(log_target=True, reduction="mean")

    def compute_loss(
        self,
        returns,
        actions,
        values,
        log_probs,
        old_log_probs=None,
        std_returns=True,
        std_advs=False,
    ):
        """
        Compute TRPO loss function
        """
        # Compute advantages
        losses = dict()
        if std_returns:
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        advantages = self.compute_advantages(
            returns, values, log_probs, std_advs=std_advs
        )

        # Compute the probability ratio
        probs_ratio = torch.exp(log_probs - old_log_probs)

        # Compute TRPO loss and baseline loss
        losses["policy_loss"] = self.losses["policy"](probs_ratio * advantages, actions)
        losses["baseline_loss"] = self.losses["baseline"](values, returns)

        # Use KL divergence as regularizer
        losses["constraint_loss"] = self.losses["constraint"](old_log_probs, log_probs)

        # Compute total loss
        losses["total_loss"] = (
            losses["policy_loss"]
            + self.alpha * losses["baseline_loss"]
            - self.beta * losses["constraint_loss"]
        )

        # Update beta
        if self.kl_target is not None:
            if losses["constraint_loss"] < (self.kl_target / 1.5):
                self.beta /= 2
            elif losses["constraint_loss"] > (self.kl_target * 1.5):
                self.beta *= 2

        return losses


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
        alpha=1.0,
        beta=0.01,
        eps=0.2,
    ):
        assert isinstance(
            alpha, float
        ), "The alpha hyperparameter should be given as float"
        super().__init__(
            env, policy_nn, baseline_nn=baseline_nn, alpha=alpha, seed=seed
        )
        self.beta = beta
        self.eps = eps

    def compute_loss(
        self,
        returns,
        actions,
        values,
        log_probs,
        old_log_probs=None,
        std_returns=True,
        std_advs=False,
    ):
        """
        Compute PPO loss function
        """
        # Compute advantages
        losses = dict()
        if std_returns:
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        advantages = self.compute_advantages(
            returns, values, log_probs, std_advs=std_advs
        )

        # Compute the probability ratio
        probs_ratio = torch.exp(log_probs - old_log_probs)

        # Compute PPO loss as the minimum of the clipped and unclipped losses
        objective = torch.min(
            probs_ratio * advantages,
            torch.clamp(probs_ratio, 1 - self.eps, 1 + self.eps) * advantages,
        )

        # Compute PPO loss, baseline loss and entropy of the categorical action distribution
        losses["policy_loss"] = self.losses["policy"](objective, actions)
        losses["baseline_loss"] = self.losses["baseline"](values, returns)
        losses["entropy_loss"] = (
            Categorical(probs=torch.exp(log_probs)).entropy().mean()
        )

        # Compute total loss
        losses["total_loss"] = (
            losses["policy_loss"]
            + self.alpha * losses["baseline_loss"]
            - self.beta * losses["entropy_loss"]
        )
        return losses
