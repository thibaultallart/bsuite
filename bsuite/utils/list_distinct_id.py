import numpy as np
import random
# from copy import deepcopy
# import gym
# from gym import error, Space, spaces, utils
# from gym.utils import seeding


class ListDistinctId: #Space):
    """
    Example usage:
    self.action_space = spaces.ListDistinctId([7, 3, 13, 4, 8], 3)
    """

    def __init__(self, values, length):
        """

        Args:
            values: list of possible values
            length: size of returned list
        """
        self.values = values
        self.length = length
        # super().__init__((), type(values[0]))
        self.rng = random.Random()

    def seed(self, seed=None):
        self.rng.seed(seed)

    def sample(self):
        return self.rng.sample(self.values, self.length)

    def contains(self, x):
        """ True is x is a subset of self.values"""
        if isinstance(x, (np.generic, np.ndarray)):
            x = x.tolist()

        # Do not allow duplicated values
        if len(x) != len(set(x)):
            return False

        if len(x) != self.length:
            return False

        return all(elem in self.values for elem in x)

    def __repr__(self):
        return "Set {}".format(self.values)

    def __eq__(self, other):
        return self.values == other.values and self.length == other.length

    def set_values(self, values):
        self.values = set(values)

    def get_values(self):
        return self.values

    def set_length(self, length):
        self.length = length

    def get_length(self):
        return self.length