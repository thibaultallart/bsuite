# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Simple diagnostic list bandit environment.

Action: a list of k distinct arms to pull.
Observation is a single pixel of 0 - this is an independent arm bandit problem!
Individuals rewards are [0, 0.1, .. 1] assigned randomly to 11 arms and deterministic
Reward is the sum of each individual rewards for the played arms.
"""


# Import all required packages

from bsuite.experiments.bandit_list import sweep
from bsuite.utils import auto_reset_environment
from bsuite.utils.list_distinct_id import ListDistinctId
import dm_env
from dm_env import specs
import numpy as np


class ListBandit(auto_reset_environment.Base):
  """SimpleBandit environment."""

  def __init__(self, seed=None, action_size=2):
    """Builds a simple bandit environment.

    Args:
      seed: Optional integer. Seed for numpy's random number generator (RNG).
    """
    super(ListBandit, self).__init__()
    self._rng = np.random.RandomState(seed)

    self._n_actions = 11
    action_mask = self._rng.choice(
        range(self._n_actions), size=self._n_actions, replace=False)
    self._rewards = np.linspace(0, 1, self._n_actions)[action_mask]

    self._action_size = action_size

    self._total_regret = 0.
    self._optimal_return = self._rewards[-self._action_size:].sum()
    self.bsuite_num_episodes = sweep.NUM_EPISODES

  def _get_observation(self):
    return np.ones(shape=(1, 1), dtype=np.float32)

  def _reset(self):
    observation = self._get_observation()
    return dm_env.restart(observation)

  def _step(self, action):
    # todo: check valid action ?
    reward = self._rewards[action].sum()
    self._total_regret += self._optimal_return - reward
    observation = self._get_observation()
    return dm_env.termination(reward=reward, observation=observation)

  def observation_spec(self):
    return specs.Array(shape=(1, 1), dtype=np.float32)

  def action_spec(self):
    return ListDistinctId([i for i in range(self._n_actions)], self._action_size)
    #return specs.DiscreteArray(self._n_actions, name='action')

  def bsuite_info(self):
    return dict(total_regret=self._total_regret)
