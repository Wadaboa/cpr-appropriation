import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from loguru import logger


class VPGPolicy:
    """
    Vanilla Policy Gradient implementation
    """

    def __init__(self, env, policy_nn, baseline_nn=None):
        self.env = env
        self.policy_nn = policy_nn
        self.baseline_nn = baseline_nn

    def train(
        self, max_epochs, lr=1e-3, discount=0.99, batch_size=128, ignore_index=-100
    ):
        """
        Train VPG by running the specified number of episodes and
        maximum time steps
        """
        # Define losses
        policy_loss = nn.NLLLoss(ignore_index=ignore_index, reduction="mean")
        if self.baseline_nn is not None:
            baseline_loss = nn.MSELoss(reduction="mean")

        # Define optimizer
        params = list(self.policy_nn.parameters())
        if self.baseline_nn is not None:
            params += list(self.baseline_nn.parameters())
        optimizer = optim.Adam(params, lr=lr)

        # Iterate for the specified number of epochs
        for epoch in range(max_epochs):
            logger.info(f"Epoch {epoch + 1} / {max_epochs}")

            # Accumulate trajectories to fill-up a batch of examples
            trajectories = TrajectoryPool()
            for _ in range(batch_size // self.env.n_agents):
                episode_trajectories = self.execute_episode()
                trajectories.extend(episode_trajectories)

            # Get a batch of (s, a, r) tuples
            logger.info(f"Working with a batch size of {len(trajectories)}")
            states, actions, returns, _ = trajectories.tensorify(
                self.env.max_steps,
                self.env.observation_space_size(flattened=False),
                discount=discount,
                ignore_index=ignore_index,
            )

            # Compute log-probabilities of actions
            self.policy_nn.train(mode=True)
            log_probs = self.policy_nn(states)

            # Compute baseline
            values = torch.zeros_like(returns)
            if self.baseline_nn is not None:
                values = self.baseline_nn(states).squeeze()

            # Compute loss
            advantage = (
                (returns - values).unsqueeze(-1).repeat(1, 1, log_probs.shape[-1])
            )
            total_loss = policy_loss(
                torch.flatten(log_probs * advantage, start_dim=0, end_dim=1),
                torch.flatten(actions),
            )
            if self.baseline_nn is not None:
                total_loss += baseline_loss(values, returns)

            # Backprop
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    def execute_episode(self):
        """
        Run an episode for the maximum number of time-steps defined
        in the environment and return a list of trajectories,
        one for each agent
        """
        # Initialize trajectories and reset environment
        self.policy_nn.eval()
        trajectories = TrajectoryPool(n=self.env.n_agents)
        observations = self.env.reset()

        # Iterate as long as agents are not done or until
        # we reach the maximum number of time-steps
        for _ in range(self.env.max_steps):
            # Compute the best actions based on the current policy
            action_dict = dict()
            for agent_handle in range(self.env.n_agents):
                log_probs = self.policy_nn(
                    torch.tensor(observations[agent_handle], dtype=torch.float32)
                )
                action = np.random.choice(
                    np.arange(self.env.action_space_size()),
                    p=np.exp(log_probs.detach().numpy()),
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


class TrajectoryPool:
    """
    A trajectory pool is a set of trajectories
    """

    def __init__(self, n=0):
        self.trajectories = [Trajectory() for _ in range(n)]

    def add(self, trajectory):
        """
        Add the given trajectory to the pool
        """
        assert isinstance(
            trajectory, Trajectory
        ), "The given trajectory should be an instance of the Trajectory class"
        self.trajectories.append(trajectory)

    def extend(self, trajectory_pool):
        """
        Extend the current trajectory pool with the given one
        """
        assert isinstance(
            trajectory_pool, TrajectoryPool
        ), "The given trajectory pool should be an instance of the TrajectoryPool class"
        for trajectory in trajectory_pool:
            self.add(trajectory)

    def add_to_trajectory(self, i, state, action, reward, next_state):
        """
        Add a (s, a, r, s') tuple to the i-th trajectory in the pool
        """
        self.trajectories[i].add_timestep(state, action, reward, next_state)

    def tensorify(self, max_steps, state_shape, discount=1, ignore_index=-100):
        """
        Convert the current pool of trajectories to
        a set of PyTorch tensors
        """
        assert isinstance(
            state_shape, tuple
        ), "The given state shape should be a tuple of dimensions"

        # Initialize tensors
        states = torch.full(
            (len(self.trajectories), max_steps, *state_shape),
            ignore_index,
            dtype=torch.float32,
        )
        actions = torch.full(
            (len(self.trajectories), max_steps), ignore_index, dtype=torch.int64
        )
        returns = torch.full_like(actions, ignore_index, dtype=torch.float32)
        next_states = torch.full_like(states, ignore_index, dtype=torch.float32)

        # Update tensors for each trajectory
        for i, trajectory in enumerate(self.trajectories):
            t_states = trajectory.get_states(as_torch=True)
            t_actions = trajectory.get_actions(as_torch=True)
            t_returns = trajectory.get_returns(
                discount=discount, to_go=True, as_torch=True
            )
            t_next_states = trajectory.get_next_states(as_torch=True)
            states[i, : t_states.shape[0]] = t_states
            actions[i, : t_actions.shape[0]] = t_actions
            returns[i, : t_returns.shape[0]] = t_returns
            next_states[i, : t_next_states.shape[0]] = t_next_states

        return states, actions, returns, next_states

    def __getitem__(self, i):
        """
        Return the i-th trajectory in the pool
        """
        return self.trajectories[i]

    def __len__(self):
        """
        Return how many trajectories we have in the pool
        """
        return len(self.trajectories)


class Trajectory:
    """
    A trajectory is a list of (s, a, r, s') tuples, that represents an
    agent's transition from state s to state s', by taking action a and
    observing reward r
    """

    def __init__(self):
        self.states, self.actions, self.rewards, self.next_states = [], [], [], []
        self.current_timestep = 0

    def add_timestep(self, state, action, reward, next_state):
        """
        Add the given (s, a, r, s') tuple to the trajectory
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.current_timestep += 1

    def get_states(self, as_torch=True):
        """
        Return the list of states in the current trajectory
        either as a Numpy array or as a PyTorch tensor
        """
        return np.array(self.states) if not as_torch else torch.tensor(self.states)

    def get_actions(self, as_torch=True):
        """
        Return the list of actions in the current trajectory
        either as a Numpy array or as a PyTorch tensor
        """
        return np.array(self.actions) if not as_torch else torch.tensor(self.actions)

    def get_rewards(self, as_torch=True):
        """
        Return the list of rewards in the current trajectory
        either as a Numpy array or as a PyTorch tensor
        """
        return np.array(self.rewards) if not as_torch else torch.tensor(self.rewards)

    def get_next_states(self, as_torch=True):
        """
        Return the list of next states in the current trajectory
        either as a Numpy array or as a PyTorch tensor
        """
        return (
            np.array(self.next_states)
            if not as_torch
            else torch.tensor(self.next_states)
        )

    def get_returns(self, max_timestep=None, discount=1, to_go=False, as_torch=True):
        """
        Compute returns of the current trajectory. You can compute the following return types:
        - Finite-horizon undiscounted return: set `max_timestep=t` and `discount=1`
        - Infinite-horizon discounted return: set `max_timestep=None` and `discount=d`, with d in (0, 1)
        """
        assert discount > 0 and discount <= 1, "Discount should be in (0, 1]"
        if max_timestep is None:
            max_timestep = self.current_timestep
        max_timestep = np.clip(max_timestep, 0, self.current_timestep)
        discount_per_timestep = discount ** np.arange(max_timestep)
        returns_per_timestep = np.cumsum(
            np.array(self.rewards)[::-1] * discount_per_timestep[::-1]
        )[::-1]
        returns = returns_per_timestep[0] if not to_go else returns_per_timestep
        return returns if not as_torch else torch.tensor(returns.copy())

    def __getitem__(self, t):
        """
        Return the (s, a, r, s') tuple at time-step t
        """
        return (self.states[t], self.actions[t], self.rewards[t], self.next_states[t])

    def __len__(self):
        """
        Return the lenght of the trajectory
        """
        return self.current_timestep
