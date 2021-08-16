import random
from enum import IntEnum

import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from matplotlib import colors


class AgentAction(IntEnum):
    """
    Defines the action of an agent
    """

    STEP_FORWARD = 0
    STEP_BACKWARD = 1
    STEP_LEFT = 2
    STEP_RIGHT = 3
    ROTATE_LEFT = 4
    ROTATE_RIGHT = 5
    STAND_STILL = 6
    TAG = 7

    @classmethod
    def size(cls):
        """
        Return the number of possible actions (i.e. 8)
        """
        return len(cls.__members__)


class AgentOrientation(IntEnum):
    """
    Defines the orientation of an agent in an unspecified position,
    as the 4 cardinal directions (N, E, S, W)
    """

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def rotate_left(self):
        """
        Return a new orientation after the agent rotates itself to its left
        """
        return AgentOrientation((self.value - 1) % self.size())

    def rotate_right(self):
        """
        Return a new orientation after the agent rotates itself to its right
        """
        return AgentOrientation((self.value + 1) % self.size())

    @classmethod
    def random(cls):
        """
        Return a random orientation out of the 4 possible values
        """
        return AgentOrientation(random.randint(0, cls.size()))

    @classmethod
    def size(cls):
        """
        Return the number of possible orientations (i.e. 4)
        """
        return len(cls.__members__)


class AgentPosition:
    """
    Defines the position of an agent as a triplet (x, y, o),
    where (x, y) are coordinates on a 2D grid and (o, ) is
    the orientation of the agent
    """

    def __init__(self, x, y, o):
        assert isinstance(
            o, AgentOrientation
        ), "The given orientation must be an instance of AgentOrientation"
        self.x = x
        self.y = y
        self.o = o

    def step_forward(self):
        """
        Return the new position of the agent after stepping forward
        """
        if self.o == AgentOrientation.UP:
            return AgentPosition(self.x, self.y - 1, self.o)
        elif self.o == AgentOrientation.RIGHT:
            return AgentPosition(self.x + 1, self.y, self.o)
        elif self.o == AgentOrientation.DOWN:
            return AgentPosition(self.x, self.y + 1, self.o)
        elif self.o == AgentOrientation.LEFT:
            return AgentPosition(self.x - 1, self.y, self.o)
        return self

    def step_backward(self):
        """
        Return the new position of the agent after stepping backward
        """
        if self.o == AgentOrientation.UP:
            return AgentPosition(self.x, self.y + 1, self.o)
        elif self.o == AgentOrientation.RIGHT:
            return AgentPosition(self.x - 1, self.y, self.o)
        elif self.o == AgentOrientation.DOWN:
            return AgentPosition(self.x, self.y - 1, self.o)
        elif self.o == AgentOrientation.LEFT:
            return AgentPosition(self.x + 1, self.y, self.o)
        return self

    def step_left(self):
        """
        Return the new position of the agent after stepping to its left
        """
        if self.o == AgentOrientation.UP:
            return AgentPosition(self.x - 1, self.y, self.o)
        if self.o == AgentOrientation.RIGHT:
            return AgentPosition(self.x, self.y - 1, self.o)
        if self.o == AgentOrientation.DOWN:
            return AgentPosition(self.x + 1, self.y, self.o)
        if self.o == AgentOrientation.LEFT:
            return AgentPosition(self.x, self.y + 1, self.o)

    def step_right(self):
        """
        Return the new position of the agent after stepping to its right
        """
        if self.o == AgentOrientation.UP:
            return AgentPosition(self.x + 1, self.y, self.o)
        if self.o == AgentOrientation.RIGHT:
            return AgentPosition(self.x, self.y + 1, self.o)
        if self.o == AgentOrientation.DOWN:
            return AgentPosition(self.x - 1, self.y, self.o)
        if self.o == AgentOrientation.LEFT:
            return AgentPosition(self.x, self.y - 1, self.o)

    def rotate_left(self):
        """
        Return the new position of the agent after rotating left
        """
        return AgentPosition(self.x, self.y, self.o.rotate_left())

    def rotate_right(self):
        """
        Return the new position of the agent after rotating right
        """
        return AgentPosition(self.x, self.y, self.o.rotate_right())

    def stand_still(self):
        """
        Return the new position of the agent after standing still
        """
        return self

    def tag(self):
        """
        Return the new position of the agent after taggin an opponent
        """
        return self

    def get_new_position(self, action):
        """
        Given an action, return the new position of the agent
        """
        assert isinstance(
            action, AgentAction
        ), "The given action must be an instance of AgentAction"
        return getattr(self, action.name.lower())()


class GridCell(IntEnum):
    """
    Defines what could fit in a cell of the 2D grid, i.e.
    either an agent or a resource or an empty cell
    """

    EMPTY = 0
    RESOURCE = 1
    AGENT = 2

    @classmethod
    def values(cls):
        """
        Return all possible values as integers
        """
        return [v for _, v in cls.__members__.items()]

    @classmethod
    def size(cls):
        """
        Return the number of possible cell types (i.e. 3)
        """
        return len(cls.__members__)


class CPRGridEnv(gym.Env):
    """
    Defines the CPR appropriation Gym environment
    """

    RESOURCE_COLLECTION_REWARD = 1
    COLORMAP = colors.ListedColormap(["black", "green", "red"])
    COLOR_BOUNDARIES = colors.BoundaryNorm([-1] + GridCell.values(), GridCell.size())
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        n_agents,
        grid_width,
        grid_height,
        fov_squares_front=20,
        fov_squares_side=10,
        tagging_ability=True,
        beam_squares_front=20,
        beam_squares_width=5,
        max_steps=1000,
    ):
        super(CPRGridEnv, self).__init__()

        self.n_agents = n_agents
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.fov_squares_front = fov_squares_front
        self.fov_squares_side = fov_squares_side
        self.tagging_ability = tagging_ability
        self.beam_squares_front = beam_squares_front
        self.beam_squares_width = beam_squares_width
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(AgentAction.size())
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

        self.elapsed_steps = None
        self.agent_positions = None
        self.grid = None
        self.reset()

    def reset(self):
        """
        Spawns a new environment by assigning random positions to the agents
        and initializing a new 2D grid
        """
        self.elapsed_steps = 0
        self.agent_positions = [self._random_position() for _ in range(self.n_agents)]
        self.grid = self._get_initial_grid()

    def _random_position(self):
        """
        Returns a random position in the 2D grid
        """
        return AgentPosition(
            x=random.randint(0, self.grid_width - 1),
            y=random.randint(0, self.grid_height - 1),
            o=AgentOrientation.random(),
        )

    def _get_initial_grid(self):
        """
        Initializes the 2D grid by setting agent positions and
        initial random resources
        """
        grid = np.full((self.grid_width, self.grid_height), GridCell.EMPTY.value)
        for agent_position in self.agent_positions:
            grid[agent_position.x, agent_position.y] = GridCell.AGENT.value
        return grid

    def step(self, actions):
        """
        Perform a step in the environment by moving all the agents
        and return one observation for each agent
        """
        assert (
            isinstance(actions, list) and len(actions) == self.n_agents
        ), "Actions should be given as a list with lenght equal to the number of agents"

        # Initiliaze variables
        observations = [None] * self.n_agents
        rewards = [-self.RESOURCE_COLLECTION_REWARD] * self.n_agents
        dones = [False] * self.n_agents

        # Move all agents
        for agent_handle, action in enumerate(actions):
            new_agent_position = self._move_agent(agent_handle, action)
            self.agent_positions[agent_handle] = new_agent_position
            if self._has_resource(new_agent_position):
                rewards[agent_handle] = self.RESOURCE_COLLECTION_REWARD

        # Check if the we reached end of episode
        self.elapsed_steps += 1
        if self._is_resource_depleted() or self.elapsed_steps == self.max_steps:
            dones = [True] * self.n_agents

        # Compute observations for each agent
        for agent_handle in range(self.n_agents):
            pass

        return observations, rewards, dones, {}

    def _move_agent(self, agent_handle, action):
        """
        Compute a new position for the given agent, after performing
        the given action
        """
        assert agent_handle in range(
            self.n_agents
        ), "The given agent handle does not exist"

        # Compute new position
        curret_agent_position = self.agent_positions[agent_handle]
        new_agent_position = curret_agent_position.get_new_position(action)

        # If move is not feasible the agent stands still
        if not self._is_move_feasible(new_agent_position):
            return curret_agent_position

        # If move is feasible, set the previous position as empty
        # and the new position as occupied
        self.grid[
            curret_agent_position.x, curret_agent_position.y
        ] = GridCell.EMPTY.value
        self.grid[new_agent_position.x, new_agent_position.y] = GridCell.AGENT.value
        return new_agent_position

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
            position, AgentPosition
        ), "The given position should be an instance of AgentPosition"
        return self.grid[position.x, position.y] == GridCell.AGENT.value

    def _is_position_in_grid(self, position):
        """
        Check if the given position is within the boundaries of the grid
        """
        assert isinstance(
            position, AgentPosition
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
            position, AgentPosition
        ), "The given position should be an instance of AgentPosition"
        return self.grid[position.x, position.y] == GridCell.RESOURCE.value

    def _is_resource_depleted(self):
        """
        Check if there is at least one resource available in the environment
        or if the resource is depleted
        """
        return len(self.grid[self.grid == GridCell.RESOURCE.value]) == 0

    def extract_fov(self, matrix, center_index, window_size, pad=0):
        """
        Extract a patch of size window_size from the given matrix centered around
        the specified position and pad external values with the given fill value
        """
        # Window is entirely contained in the given matrix
        m, n = matrix.shape
        offset = window_size // 2
        yl, yu = center_index[0] - offset, center_index[0] + offset
        xl, xu = center_index[1] - offset, center_index[1] + offset
        if xl >= 0 and xu < n and yl >= 0 and yu < m:
            return np.array(matrix[yl : yu + 1, xl : xu + 1], dtype=matrix.dtype)

        # Window has to be padded
        window = np.full((window_size, window_size), pad, dtype=matrix.dtype)
        c_yl, c_yu = np.clip(yl, 0, m), np.clip(yu, 0, m)
        c_xl, c_xu = np.clip(xl, 0, n), np.clip(xu, 0, n)
        sub = matrix[c_yl : c_yu + 1, c_xl : c_xu + 1]
        w_yl = 0 if yl >= 0 else abs(yl)
        w_yu = window_size if yu < m else window_size - (yu - m) - 1
        w_xl = 0 if xl >= 0 else abs(xl)
        w_xu = window_size if xu < n else window_size - (xu - n) - 1
        window[w_yl:w_yu, w_xl:w_xu] = sub
        return window


    def render(self, mode="human"):
        """
        Render the environment as an RGB image
        """
        _, ax = plt.subplots()
        ax.imshow(self.grid, cmap=self.COLORMAP, norm=self.COLOR_BOUNDARIES)
        ax.grid(which="major", axis="both", linestyle="-", color="k", linewidth=2)
        plt.show()

    def close(self):
        return
