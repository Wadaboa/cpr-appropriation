import random
from enum import IntEnum
from types import ClassMethodDescriptorType

import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class AgentAction(IntEnum):

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
        return len(cls.__members__)


def AgentOrientation(IntEnum):

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def rotate_left(self):
        return AgentOrientation((self.value - 1) % self.size())

    def rotate_right(self):
        return AgentOrientation((self.value + 1) % self.size())

    @classmethod
    def random(cls):
        return AgentOrientation(random.randint(0, cls.size()))

    @classmethod
    def size(cls):
        return len(cls.__members__)


class AgentPosition:
    def __init__(self, x, y, o):
        self.x = x
        self.y = y
        self.o = o

    def get_new_position(self, action):
        if action == AgentAction.STEP_FORWARD:
            if self.o == AgentOrientation.UP:
                return AgentPosition(self.x, self.y - 1, self.o)
            if self.o == AgentOrientation.RIGHT:
                return AgentPosition(self.x + 1, self.y, self.o)
            if self.o == AgentOrientation.DOWN:
                return AgentPosition(self.x, self.y + 1, self.o)
            if self.o == AgentOrientation.LEFT:
                return AgentPosition(self.x - 1, self.y, self.o)

        if action == AgentAction.STEP_BACKWARD:
            if self.o == AgentOrientation.UP:
                return AgentPosition(self.x, self.y + 1, self.o)
            if self.o == AgentOrientation.RIGHT:
                return AgentPosition(self.x - 1, self.y, self.o)
            if self.o == AgentOrientation.DOWN:
                return AgentPosition(self.x, self.y - 1, self.o)
            if self.o == AgentOrientation.LEFT:
                return AgentPosition(self.x + 1, self.y, self.o)

        if action == AgentAction.STEP_LEFT:
            if self.o == AgentOrientation.UP:
                return AgentPosition(self.x - 1, self.y, self.o)
            if self.o == AgentOrientation.RIGHT:
                return AgentPosition(self.x, self.y - 1, self.o)
            if self.o == AgentOrientation.DOWN:
                return AgentPosition(self.x + 1, self.y, self.o)
            if self.o == AgentOrientation.LEFT:
                return AgentPosition(self.x, self.y + 1, self.o)

        if action == AgentAction.STEP_RIGHT:
            if self.o == AgentOrientation.UP:
                return AgentPosition(self.x + 1, self.y, self.o)
            if self.o == AgentOrientation.RIGHT:
                return AgentPosition(self.x, self.y + 1, self.o)
            if self.o == AgentOrientation.DOWN:
                return AgentPosition(self.x - 1, self.y, self.o)
            if self.o == AgentOrientation.LEFT:
                return AgentPosition(self.x, self.y - 1, self.o)


class CPRGridEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    RGB_CHANNELS = 3

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
                self.RGB_CHANNELS,
            ),
            dtype=np.uint8,
        )

        self.grid = None
        self.agent_positions = None

    def _get_initial_grid(self, agent_positions):
        grid = np.zeros((self.grid_width, self.grid_height))
        for agent_handle, agent_position in agent_positions:
            grid[agent_position.x, agent_position.y] = agent_handle
        return grid

    def _random_position(self):
        return AgentPosition(
            x=random.randint(0, self.grid_width - 1),
            y=random.randint(0, self.grid_height - 1),
            o=AgentOrientation.random(),
        )

    def _move_agent(self, agent_handle, action):
        agent_position = self.agent_positions[agent_handle]

    def step(self, action):
        pass

    def _agent_step(self, action):
        pass

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

    def reset(self):
        self.agent_positions = [self._random_position() for _ in range(self.n_agents)]
        self.grid = self._get_initial_grid(self.agent_positions)

    def render(self, mode="human"):
        pass

    def close(self):
        pass
