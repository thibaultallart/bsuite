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

from bsuite.experiments.position_based_model import sweep
from bsuite.utils import auto_reset_environment
from bsuite.utils.list_distinct_id import ListDistinctId
import dm_env
from dm_env import specs
import numpy as np


class PositionBasedModel(auto_reset_environment.Base):
  """SimpleBandit environment."""

  def __init__(self, rewards=None, position_bias=None, bernoulli=True, seed=None):
    """Builds a position based model environment.

    Args:
      position_bias: bias for each position.
      seed: Optional integer. Seed for numpy's random number generator (RNG).
    """
    super(PositionBasedModel, self).__init__()
    self._rng = np.random.RandomState(seed)
    self.bernoulli = bernoulli

    if rewards is None:
      self._n_actions = 11
      action_mask = self._rng.choice(
          range(self._n_actions), size=self._n_actions, replace=False)
      self._rewards = np.linspace(0, 1, self._n_actions)[action_mask]
    else:
      self._n_actions = len(rewards)
      self._rewards = rewards

    if position_bias is None:
      self.position_biais = np.array([1, 0.8, 0.5])
    else:
      self.position_biais = position_bias

    self._action_size = len(self.position_biais)

    self._total_regret = np.sort(self._rewards)[:self._action_size].dot(self.position_biais)
    self._optimal_return = np.sort(self._rewards)[::-1][:self._action_size:].dot(self.position_biais)
    self.bsuite_num_episodes = sweep.NUM_EPISODES

  def _get_observation(self):
    return np.ones(shape=(1, 1), dtype=np.float32)

  def _reset(self):
    observation = self._get_observation()
    return dm_env.restart(observation)

  def _step(self, action):
    # todo: check valid action ?

    examine = np.array([self._rng.binomial(p=self.position_biais[i], n=1, size=1)[0]
                        for i in range(self._action_size)], dtype=np.float64)
    interest = np.array([self._rng.binomial(p=self._rewards[a], n=1, size=1)[0]
                         for a in action], dtype=np.float64)
    rewards = examine * interest
    reward = np.sum(rewards)

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
