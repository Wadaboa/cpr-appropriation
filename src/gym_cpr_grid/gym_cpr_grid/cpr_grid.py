import time
import random
import itertools

import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from matplotlib import colors
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from . import utils


class CPRGridEnv(MultiAgentEnv, gym.Env):
    """
    Defines the CPR appropriation Gym environment
    """

    # Rewards
    NO_RESOURCE_COLLECTION_REWARD = 0
    RESOURCE_COLLECTION_REWARD = 1

    # Colors
    GRID_CELL_COLORS = {
        utils.GridCell.EMPTY: "black",
        utils.GridCell.RESOURCE: "green",
        utils.GridCell.AGENT: "red",
    }
    FOV_OWN_AGENT_COLOR = "blue"
    COLORMAP = colors.ListedColormap(list(GRID_CELL_COLORS.values()))
    COLOR_BOUNDARIES = colors.BoundaryNorm(
        utils.GridCell.values() + [utils.GridCell.size()], utils.GridCell.size()
    )

    # Rendering option
    FIGSIZE = (12, 10)

    # Gym variables
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        n_agents,
        grid_width,
        grid_height,
        fov_squares_front=20,
        fov_squares_side=10,
        tagging_ability=True,
        tagging_steps=25,
        beam_squares_front=20,
        beam_squares_width=5,
        ball_radius=2,
        max_steps=1000,
    ):
        super(CPRGridEnv, self).__init__()

        # Parameters
        self.n_agents = n_agents
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.fov_squares_front = fov_squares_front
        self.fov_squares_side = fov_squares_side
        self.tagging_ability = tagging_ability
        self.tagging_steps = tagging_steps
        self.beam_squares_front = beam_squares_front
        self.beam_squares_side = beam_squares_width // 2
        self.ball_radius = ball_radius
        self.max_steps = max_steps

        # Gym requirements
        self.action_space = utils.CPRGridActionSpace()
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                self.fov_squares_front,
                self.fov_squares_side * 2 + 1,
                3,
            ),
            dtype=np.uint8,
        )

        # Dynamic variables
        (
            self.elapsed_steps,
            self.agent_positions,
            self.grid,
            self.tagged_agents,
            self.tagging_history,
        ) = (None, None, None, None, None)

    def reset(self):
        """
        Spawns a new environment by assigning random positions to the agents
        and initializing a new 2D grid
        """
        # Reset variables
        self.elapsed_steps = 0
        self.agent_positions = [self._random_position() for _ in range(self.n_agents)]
        self.grid = self._get_initial_grid()
        self.tagged_agents = dict()
        self.tagging_history = [dict(self.tagged_agents)]

        # Compute observations for each agent
        observations = {h: None for h in range(self.n_agents)}
        for agent_handle in range(self.n_agents):
            observations[agent_handle] = self._get_observation(agent_handle)

        return observations

    def _random_position(self):
        """
        Returns a random position in the 2D grid
        """
        return utils.AgentPosition(
            x=random.randint(0, self.grid_width - 1),
            y=random.randint(0, self.grid_height - 1),
            o=utils.AgentOrientation.random(),
        )

    def _get_initial_grid(self):
        """
        Initializes the 2D grid by setting agent positions and
        initial random resources
        """
        # Assign agent positions in the grid
        grid = np.full((self.grid_height, self.grid_width), utils.GridCell.EMPTY.value)
        for agent_position in self.agent_positions:
            grid[agent_position.y, agent_position.x] = utils.GridCell.AGENT.value

        # Compute uniformely distributed resources
        resource_mask = np.random.randint(
            low=0, high=2, size=(self.grid_height, self.grid_width), dtype=bool
        )
        ys, xs = resource_mask.nonzero()
        resource_indices = list(zip(list(xs), list(ys)))

        # Assign resources to cells that are not occupied by agents
        for x, y in resource_indices:
            if grid[y, x] == utils.GridCell.EMPTY.value:
                grid[y, x] = utils.GridCell.RESOURCE.value

        return grid

    def step(self, action_dict):
        """
        Perform a step in the environment by moving all the agents
        and return one observation for each agent
        """
        assert (
            isinstance(action_dict, dict) and len(action_dict) == self.n_agents
        ), "Actions should be given as a dictionary with lenght equal to the number of agents"

        # Initiliaze variables
        observations = {h: None for h in range(self.n_agents)}
        rewards = {h: self.NO_RESOURCE_COLLECTION_REWARD for h in range(self.n_agents)}
        dones = {h: False for h in range(self.n_agents)}
        infos = {h: dict() for h in range(self.n_agents)}

        # Move all agents
        tagged_agents = []
        for agent_handle, action in action_dict.items():
            # Perform the action only if not previously tagged
            if (
                self.tagging_ability and agent_handle not in tagged_agents
            ) or not self.tagging_ability:
                # Compute new position
                new_agent_position = self._compute_new_agent_position(
                    agent_handle, action
                )

                # Assign reward for resource collection
                if self._has_resource(new_agent_position):
                    rewards[agent_handle] = self.RESOURCE_COLLECTION_REWARD

                # Move the agent only after checking for resource presence
                self._move_agent(agent_handle, new_agent_position)

                # Tag agents
                if self.tagging_ability and action == utils.AgentAction.TAG:
                    tagged_agents += self._tag(agent_handle)

        # Store the tagged agents and free the ones that were
        # tagged more than the specified timesteps ago
        for agent_handle in range(self.n_agents):
            if agent_handle in self.tagged_agents:
                self.tagged_agents[agent_handle] -= 1
                if self.tagged_agents[agent_handle] == 0:
                    del self.tagged_agents[agent_handle]
            elif agent_handle in tagged_agents:
                self.tagged_agents[agent_handle] = self.tagging_steps

        # Add tagging information to return dictionaries
        self.tagging_history += [dict(self.tagged_agents)]
        for agent_handle in range(self.n_agents):
            tagged = agent_handle in self.tagged_agents
            infos[agent_handle]["tagged"] = tagged
            if tagged:
                infos[agent_handle]["remaining_tagged_steps"] = self.tagged_agents[
                    agent_handle
                ]

        # Check if we reached end of episode
        dones["__all__"] = False
        self.elapsed_steps += 1
        if self._is_resource_depleted() or self.elapsed_steps == self.max_steps:
            dones = {h: True for h in range(self.n_agents)}
            dones["__all__"] = True

        # Compute observations for each agent
        for agent_handle in range(self.n_agents):
            observations[agent_handle] = self._get_observation(agent_handle)

        # Respawn resources
        self._respawn_resources()

        return observations, rewards, dones, infos

    def _compute_new_agent_position(self, agent_handle, action):
        """
        Compute a new position for the given agent, after performing
        the given action
        """
        assert agent_handle in range(
            self.n_agents
        ), "The given agent handle does not exist"

        # Compute new position
        current_position = self.agent_positions[agent_handle]
        new_position = current_position.get_new_position(action)

        # If move is not feasible the agent stands still
        if not self._is_move_feasible(new_position):
            return current_position

        return new_position

    def _move_agent(self, agent_handle, new_position):
        """
        Set the previous position as empty and the new one as occupied
        """
        assert isinstance(
            new_position, utils.AgentPosition
        ), "The given position should be an instance of AgentPosition"
        current_position = self.agent_positions[agent_handle]
        self.grid[current_position.y, current_position.x] = utils.GridCell.EMPTY.value
        self.grid[new_position.y, new_position.x] = utils.GridCell.AGENT.value
        self.agent_positions[agent_handle] = new_position

    def _is_move_feasible(self, position):
        """
        Check if the move leading the agent to the given position
        is a feasible move or an illegal one
        """
        return self._is_position_in_grid(position) and not self._is_position_occupied(
            position
        )

    def _is_position_occupied(self, position):
        """
        Check if the given position is occupied by another agent in the grid
        """
        assert isinstance(
            position, utils.AgentPosition
        ), "The given position should be an instance of AgentPosition"
        return self.grid[position.y, position.x] == utils.GridCell.AGENT.value

    def _is_position_in_grid(self, position):
        """
        Check if the given position is within the boundaries of the grid
        """
        assert isinstance(
            position, utils.AgentPosition
        ), "The given position should be an instance of AgentPosition"
        if position.x < 0 or position.x >= self.grid_width:
            return False
        if position.y < 0 or position.y >= self.grid_height:
            return False
        return True

    def _has_resource(self, position):
        """
        Check if the given position is occupied by a resource in the grid
        """
        assert isinstance(
            position, utils.AgentPosition
        ), "The given position should be an instance of utils.AgentPosition"
        return self.grid[position.y, position.x] == utils.GridCell.RESOURCE.value

    def _is_resource_depleted(self):
        """
        Check if there is at least one resource available in the environment
        or if the resource is depleted
        """
        return len(self.grid[self.grid == utils.GridCell.RESOURCE.value]) == 0

    def _tag(self, agent_handle):
        """
        Perform the tagging action from the point of view of the given agent,
        by marking agents in the beam trajectory for future timesteps
        """
        # Create a grid with the same size as the original one,
        # but only containing agent handles
        agents_grid = np.full((self.grid_height, self.grid_width), -1)
        for h, agent_position in enumerate(self.agent_positions):
            if h != agent_handle:
                agents_grid[agent_position.y, agent_position.x] = h

        # Extract the beam FOV from the agent handles grid
        # and find the tagged agents
        fov = self._extract_fov(agent_handle, grid=agents_grid, beam=True, pad_value=-1)
        tagged_agents = fov[fov != -1]

        return list(tagged_agents)

    def _get_observation(self, agent_handle):
        """
        Extract a rectangular FOV based on the given agent's position
        and convert it into an RGB image
        """
        # Extract the FOV and convert it to 3 channels
        fov = self._extract_fov(agent_handle)
        fov = np.stack((fov,) * 3, axis=-1)

        # Set colors for resources and agents
        fov = np.where(
            fov == utils.GridCell.RESOURCE,
            colors.to_rgb(self.GRID_CELL_COLORS[utils.GridCell.RESOURCE]),
            fov,
        )
        fov = np.where(
            fov == utils.GridCell.AGENT,
            colors.to_rgb(self.GRID_CELL_COLORS[utils.GridCell.AGENT]),
            fov,
        )
        fov[0, self.fov_squares_side] = colors.to_rgb(self.FOV_OWN_AGENT_COLOR)

        return fov

    def _respawn_resources(self):
        """
        Respawn resources based on the number of already-spawned resources
        in a ball centered around each currently empty location
        """
        for x, y in itertools.product(range(self.grid_width), range(self.grid_height)):
            if self.grid[y, x] == utils.GridCell.EMPTY.value:
                l = len(self._extract_ball(x, y))
                p = self._respawn_probability(l)
                if np.random.binomial(1, p):
                    self.grid[y, x] = utils.GridCell.RESOURCE.value

    def _respawn_probability(self, l):
        """
        Compute the respawn probability of a resource in an unspecified
        location based on the number of nearby resources
        """
        if l == 1 or l == 2:
            return 0.01
        elif l == 3 or l == 4:
            return 0.05
        elif l > 4:
            return 0.1
        return 0

    def _pad_grid(self, grid, x, y, xl, yl, pad_value=0):
        """
        Pad the 2D grid by computing pad widths based
        on the given position and span lenghts in both axes
        """
        x_pad_width = (
            abs(np.clip(x - xl, None, 0)),
            np.clip(x + xl - self.grid_width + 1, 0, None),
        )
        y_pad_width = (
            abs(np.clip(y - yl, None, 0)),
            np.clip(y + yl - self.grid_height + 1, 0, None),
        )
        padded_grid = np.pad(
            grid,
            pad_width=[y_pad_width, x_pad_width],
            mode="constant",
            constant_values=pad_value,
        )
        return padded_grid, x_pad_width, y_pad_width

    def _extract_fov(self, agent_handle, grid=None, beam=False, pad_value=None):
        """
        Extract a rectangular local observation from the 2D grid,
        from the point of view of the given agent
        """
        # Get the current agent's position
        agent_position = self.agent_positions[agent_handle]

        # Rotate the grid based on agent's orientation
        grid = grid if grid is not None else self.grid.copy()
        k = (
            1
            if agent_position.o == utils.AgentOrientation.LEFT
            else 2
            if agent_position.o == utils.AgentOrientation.UP
            else 3
            if agent_position.o == utils.AgentOrientation.RIGHT
            else 0
        )
        rotated_grid = np.rot90(grid, k=k)

        # Compute agent's coordinates on the rotated grid
        coords = list(
            itertools.product(range(self.grid_height), range(self.grid_width))
        )
        coords = np.array(coords).reshape(self.grid_height, self.grid_width, 2)
        rotated_coords = np.rot90(coords, k=k)
        rotated_y, rotated_x = np.argwhere(
            (rotated_coords[:, :, 0] == agent_position.y)
            & (rotated_coords[:, :, 1] == agent_position.x)
        )[0]

        # Set front and side squares to extract based on whether we are
        # computing the FOV for the observation or for tagging other agents
        squares_front = self.fov_squares_front if not beam else self.beam_squares_front
        squares_side = self.fov_squares_side if not beam else self.beam_squares_side

        # Pad the 2D grid so as not have indexing errors in FOV extraction
        padded_grid, x_pad_width, y_pad_width = self._pad_grid(
            rotated_grid,
            rotated_x,
            rotated_y,
            squares_side,
            squares_front + k % 2,
            pad_value=pad_value or utils.GridCell.EMPTY.value,
        )

        # Extract the FOV
        sx, ex = (
            x_pad_width[0] + rotated_x - squares_side,
            x_pad_width[0] + rotated_x + squares_side + 1,
        )
        sy, ey = (
            y_pad_width[0] + rotated_y,
            y_pad_width[0] + rotated_y + squares_front,
        )
        fov = padded_grid[sy:ey, sx:ex]
        assert fov.shape == (
            squares_front,
            squares_side * 2 + 1,
        ), "There was an error in FOV extraction, incorrect shape"

        return fov

    def _extract_ball(self, x, y):
        """
        Extract a ball-shaped local patch from the 2D grid,
        centered around the given position
        """
        # Pad the 2D grid so as not have indexing errors in ball extraction
        padded_grid, x_pad_width, y_pad_width = self._pad_grid(
            self.grid,
            x,
            y,
            self.ball_radius,
            self.ball_radius,
            pad_value=utils.GridCell.EMPTY.value,
        )

        # Extract the ball
        sx, ex = (
            x_pad_width[0] + x - self.ball_radius,
            x_pad_width[0] + x + self.ball_radius + 1,
        )
        sy, ey = (
            y_pad_width[0] + y - self.ball_radius,
            y_pad_width[0] + y + self.ball_radius + 1,
        )
        ball = padded_grid[sy:ey, sx:ex]

        # Compute a boolean mask shaped like a ball
        # (see https://stackoverflow.com/questions/8647024/how-to-apply-a-disc-shaped-mask-to-a-numpy-array)
        kernel = np.zeros((2 * self.ball_radius + 1, 2 * self.ball_radius + 1))
        yg, xg = np.ogrid[
            -self.ball_radius : self.ball_radius + 1,
            -self.ball_radius : self.ball_radius + 1,
        ]
        mask = xg ** 2 + yg ** 2 <= self.ball_radius ** 2
        kernel[mask] = 1

        return ball[kernel.astype(bool)]

    def plot(self, img, map_colors=True):
        """
        Plot the given image in a standard way
        """
        fig, ax = plt.subplots(figsize=self.FIGSIZE)
        img = ax.imshow(
            img,
            cmap=self.COLORMAP if map_colors else None,
            norm=self.COLOR_BOUNDARIES if map_colors else None,
            origin="upper",
        )
        ax.axis("off")
        return fig, ax, img

    def plot_observation(self, obs):
        """
        Plot the given observation as an RGB image
        """
        self.plot(obs * 255.0, map_colors=False)
        plt.show()

    def render(self, mode="human"):
        """
        Render the environment as an RGB image
        """
        fig, _, _ = self.plot(self.grid)
        if mode == "rgb_array":
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return data
        plt.show()

    def close(self):
        """
        Close all open rendering figures
        """
        plt.close()
