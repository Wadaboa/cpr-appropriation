import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
from loguru import logger


class VPGPolicy:
    """
    Vanilla Policy Gradient implementation
    """

    def __init__(self, env, estimator, lr=1e-3, discount=0.99, batch_size=128):
        self.env = env
        self.estimator = estimator
        self.discount = discount
        self.batch_size = batch_size
        self.optimizer = optim.Adam(estimator.parameters(), lr=lr)

    def train(self, max_episodes):
        """
        Train VPG by running the specified number of episodes and
        maximum time steps
        """
        # Iterate for the specified number of episodes
        current_episode = 0
        while current_episode < max_episodes:

            # Accumulate trajectories to fill-up a batch of examples
            trajectories = TrajectoryPool()
            for _ in range(self.batch_size // self.env.n_agents):
                logger.info(f"Episode {current_episode + 1} / {max_episodes}")
                episode_trajectories = self.execute_episode()
                trajectories.extend(episode_trajectories)
                current_episode += 1
                if current_episode >= max_episodes:
                    break

            # Get a batch of (s, a, r) tuples
            logger.info(f"Working with a batch size of {len(trajectories)}")
            states, actions, returns, _ = trajectories.tensorify(
                self.env.max_steps,
                self.env.observation_space_size(flattened=False),
                discount=self.discount,
                default_action=self.env.default_action(),
            )

            # Compute loss
            self.estimator.train(mode=True)
            logits = self.estimator(states)
            sampler = Categorical(logits=logits)
            sampled_actions = sampler.sample()
            log_probs = -sampler.log_prob(sampled_actions)
            loss = torch.mean(returns * torch.gather(log_probs, 1, actions))

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def execute_episode(self):
        """
        Run an episode for the maximum number of time-steps defined
        in the environment and return a list of trajectories,
        one for each agent
        """
        # Initialize trajectories and reset environment
        self.estimator.eval()
        trajectories = TrajectoryPool(n=self.env.n_agents)
        observations = self.env.reset()

        # Iterate as long as agents are not done or until
        # we reach the maximum number of time-steps
        for _ in range(self.env.max_steps):
            # Compute the best actions based on the current policy
            action_dict = dict()
            for agent_handle in range(self.env.n_agents):
                logits = self.estimator(
                    torch.tensor(observations[agent_handle], dtype=torch.float32)
                )
                action = Categorical(logits=logits).sample()
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

    def tensorify(self, max_steps, state_shape, discount=1, default_action=0):
        """
        Convert the current pool of trajectories to
        a set of PyTorch tensors
        """
        assert isinstance(
            state_shape, tuple
        ), "The given state shape should be a tuple of dimensions"

        # Initialize tensors
        states = torch.full(
            (len(self.trajectories), max_steps, *state_shape), -1, dtype=torch.float32
        )
        actions = torch.full(
            (len(self.trajectories), max_steps), default_action, dtype=torch.int64
        )
        returns = torch.full_like(actions, 0, dtype=torch.float32)
        next_states = torch.full_like(states, -1, dtype=torch.float32)

        # Update tensors for each trajectory
        for i, trajectory in enumerate(self.trajectories):
            t_states = trajectory.get_states(as_torch=True)
            t_actions = trajectory.get_actions(as_torch=True)
            t_returns = trajectory.get_returns(
                discount=discount, per_timestep=True, as_torch=True
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

    def get_returns(
        self, max_timestep=None, discount=1, per_timestep=False, as_torch=True
    ):
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
        returns_per_timestep = np.cumsum(np.array(self.rewards) * discount_per_timestep)
        returns = returns_per_timestep[-1] if not per_timestep else returns_per_timestep
        return returns if not as_torch else torch.tensor(returns)

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
