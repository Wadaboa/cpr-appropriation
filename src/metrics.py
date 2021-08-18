import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks


class SocialOutcomeMetrics(DefaultCallbacks):
    """
    Define the 4 social outcome metrics as described in the paper
    """

    def _get_returns(self, episode):
        """
        Compute the sum of historical rewards for each agent
        """
        returns = []
        for agent_handle in episode.get_agents():
            returns += [np.sum(episode._agent_reward_history[agent_handle])]
        return returns

    def utilitarian_metric(self, episode):
        """
        The Utilitarian metric (U), also known as Efficiency, measures the sum total
        of all rewards obtained by all agents: it is defined as the average over players
        of sum of rewards
        """
        returns = self._get_returns(episode)
        return np.sum(returns) / episode.length

    def equality_metric(self, episode, eps=1e-6):
        """
        The Equality metric (E) is defined using the Gini coefficient
        """
        returns = self._get_returns(episode)
        numerator = np.sum([abs(ri - rj) for ri in returns for rj in returns])
        return 1 - (numerator / (2 * len(returns) * np.sum(returns) + eps))

    def sustainability_metric(self, episode, base_env):
        """
        The Sustainability metric (S) is defined as the average
        time at which the rewards are collected
        """
        times = []
        max_steps = base_env.envs[0].max_steps
        for agent_handle in episode.get_agents():
            rewards = episode._agent_reward_history[agent_handle]
            ti = np.argwhere(np.array(rewards) > 0)
            times.append(max_steps if len(ti) == 0 else np.mean(ti))
        return np.mean(times)

    def peace_metric(self, episode, base_env):
        """
        The Peace metric (P) is defined as the average number of
        untagged agent steps
        """
        if not base_env.envs[0].tagging_ability:
            return np.nan
        total = 0
        tagging_history = base_env.envs[0].tagging_history
        n_agents = len(episode.get_agents())
        for agent_handle in range(n_agents):
            total += np.sum(
                [
                    int(agent_handle in tagging_checkpoint)
                    for tagging_checkpoint in tagging_history
                ]
            )
        return (n_agents * episode.length - total) / episode.length

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

        # Initialize utilitarian metric
        episode.user_data["efficiencies"] = []
        episode.hist_data["efficiencies"] = []

        # Initialize equality metric
        episode.user_data["equalities"] = []
        episode.hist_data["equalities"] = []

        # Initialize sustainability metric
        episode.user_data["sustainabilities"] = []
        episode.hist_data["sustainabilities"] = []

        # Initialize peace metric
        episode.user_data["peaces"] = []
        episode.hist_data["peaces"] = []

    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
        """
        Called after one step in the environment: deals with computing custom metrics
        """
        # Make sure this episode is ongoing
        assert (
            episode.length > 0
        ), "Step callback should only be called after stepping in the environment"

        # Compute efficiency metric
        efficiency = self.utilitarian_metric(episode)
        episode.user_data["efficiencies"].append(efficiency)

        # Compute equality metric
        equality = self.equality_metric(episode)
        episode.user_data["equalities"].append(equality)

        # Compute sustainability metric
        sustainability = self.sustainability_metric(episode, base_env)
        episode.user_data["sustainabilities"].append(sustainability)

        # Compute peace metric
        peace = self.peace_metric(episode, base_env)
        episode.user_data["peaces"].append(peace)

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

        # Store efficiency metric
        efficiency = np.mean(episode.user_data["efficiencies"])
        episode.custom_metrics["efficiency"] = efficiency
        episode.hist_data["efficiencies"] = episode.user_data["efficiencies"]

        # Store equality metric
        equality = np.mean(episode.user_data["equalities"])
        episode.custom_metrics["equality"] = equality
        episode.hist_data["equalities"] = episode.user_data["equalities"]

        # Store sustainability metric
        sustainability = np.mean(episode.user_data["sustainabilities"])
        episode.custom_metrics["sustainability"] = sustainability
        episode.hist_data["sustainabilities"] = episode.user_data["sustainabilities"]

        # Store peace metric
        peace = np.mean(episode.user_data["peaces"])
        episode.custom_metrics["peace"] = peace
        episode.hist_data["peaces"] = episode.user_data["peaces"]
