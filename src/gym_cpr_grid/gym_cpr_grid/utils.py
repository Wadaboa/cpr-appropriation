import random
from enum import IntEnum

import numpy as np
from gym import spaces


class CustomIntEnum(IntEnum):
    """
    IntEnum with utility functions
    """

    @classmethod
    def random(cls):
        """
        Return a random value out of the possible ones
        """
        return cls(random.randint(0, cls.size() - 1))

    @classmethod
    def keys(cls):
        """
        Return all possible keys as strings
        """
        return list(cls.__members__.keys())

    @classmethod
    def values(cls):
        """
        Return all possible values as integers
        """
        return [v.value for v in cls.__members__.values()]

    @classmethod
    def is_valid(cls, x):
        """
        Check if the given value is valid
        """
        if isinstance(x, int) or isinstance(x, np.int64):
            return x in cls.values()
        elif isinstance(x, cls):
            return x in cls.keys()
        return False

    @classmethod
    def size(cls):
        """
        Return the number of possible values
        """
        return len(cls.__members__)

    @classmethod
    def equals(cls, x, y):
        """
        Check if the given values are the same
        """
        assert cls.is_valid(x) and cls.is_valid(
            y
        ), "The given inputs are not valid values"
        if (isinstance(x, int) or isinstance(x, np.int64)) and isinstance(y, cls):
            return x == y.value
        elif (isinstance(y, int) or isinstance(y, np.int64)) and isinstance(x, cls):
            return x.value == y
        return x == y


class AgentAction(CustomIntEnum):
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
    GIFT = 8

    @classmethod
    def still_actions(cls):
        """
        Return a list of actions that do not require moving the agent
        """
        return (cls.ROTATE_LEFT, cls.ROTATE_RIGHT, cls.STAND_STILL, cls.TAG, cls.GIFT)

    @classmethod
    def is_still_action(cls, action):
        """
        Check if the given action is a no-movement action
        """
        assert AgentAction.is_valid(
            action
        ), f"The given action should be compatible with AgentAction, {action} {AgentAction.values()}, {type(action)}"
        if isinstance(action, int) or isinstance(action, np.int64):
            action = AgentAction(action)
        return action in cls.still_actions()


class CPRGridActionSpace(spaces.Discrete):
    """
    The action space spanned by all the possible agent actions
    """

    def __init__(self):
        super(CPRGridActionSpace, self).__init__(AgentAction.size())

    def sample(self):
        """
        Sample a random action from the action space
        """
        return AgentAction.random()

    def contains(self, action):
        """
        Check if the given action is contained in the action space
        """
        return AgentAction.is_valid(action)

    def __repr__(self):
        return "CPRGridActionSpace()"

    def __eq__(self, other):
        return isinstance(other, CPRGridActionSpace)


class AgentOrientation(CustomIntEnum):
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


class AgentPosition:
    """
    Defines the position of an agent as a triplet (x, y, o), where (x, y)
    are coordinates on a 2D grid (with origin on the upper left corner)
    and (o, ) is the orientation of the agent
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
        Return the new position of the agent after tagging an opponent
        """
        return self

    def gift(self):
        """
        Return the new position of the agent after gifting an opponent
        """
        return self

    def get_new_position(self, action):
        """
        Given an action, return the new position of the agent
        """
        assert AgentAction.is_valid(
            action
        ), f"The given action should be compatible with AgentAction, {action} {AgentAction.values()}, {type(action)}"
        if isinstance(action, int) or isinstance(action, np.int64):
            action = AgentAction(action)
        return getattr(self, action.name.lower())()

    def __repr__(self):
        return f"AgentPosition({self.x}, {self.y}, {self.o.name})"

    def __eq__(self, other):
        return (
            isinstance(other, AgentPosition)
            and self.x == other.x
            and self.y == other.y
            and self.o == other.o
        )


class GridCell(CustomIntEnum):
    """
    Defines what could fit in a cell of the 2D grid, i.e.
    either an agent or a resource or an empty cell
    """

    OUTSIDE = -1
    EMPTY = 0
    RESOURCE = 1
    AGENT = 2
    ORIENTATION = 3


class GiftingMechanism(CustomIntEnum):
    """
    Defines the different types of reward gifting mechanisms
    as described in the paper

    Andrei Lupu, Doina Precup,
    "Gifting in Multi-Agent Reinforcement Learning",
    International Foundation for Autonomous Agents and Multiagent Systems, 2020.
    """

    ZERO_SUM = 0
    FIXED_BUDGET = 1
    REPLENISHABLE_BUDGET = 2
