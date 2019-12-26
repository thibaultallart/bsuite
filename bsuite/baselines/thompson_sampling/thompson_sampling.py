# python3
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
"""Thompson sampling agent."""

from typing import Optional

from bsuite.baselines import base
import dm_env
from dm_env import specs
import numpy as np

from scipy.special import btdtri
from copy import copy


class DiscreteAgent():
  """ Generic class for discrete action agents """

  def __init__(self, env):
    self.env = env
    self.np_random, _ = seeding.np_random()
    self.nb_actions = self.env.nb_arms
    self.seed()

  @property
  def action_space(self):
    return self.env.action_space

  def act(self, observation, reward, done):
    raise NotImplementedError

  def reset(self):
    raise NotImplementedError

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

class BetaBernoulli(object):
    def __init__(self, a=1, b=1, prior=np.ones(2)):
        self.a = a
        self.b = b
        self.n = copy(prior)  # number of {0,1} rewards

    def update(self, reward):
        self.n[int(reward)] += 1

    def sample(self, np_random):
        return np_random.beta(self.a + self.n[1], self.b + self.n[0])


# class ThompsonSampling(DiscreteAgent):
#     def __init__(self, env):
#         super(ThompsonSampling, self).__init__(env)
#         self.posterior = [BetaBernoulli() for i in range(self.nb_actions)]
#         self.t = 1
#         self.action = None
#
#     def reset(self):
#         self.posterior = [BetaBernoulli() for i in range(self.nb_actions)]
#         self.t = 1
#         self.action = None
#
#     def act(self, observation, reward, done):
#         if done:
#             # env have been reset
#             self.reset()
#         else:
#             # update
#             self.posterior[self.action].update(reward)
#
#         # choose action
#         posterior = np.array([self.posterior[i].sample(self.np_random) for i in range(self.nb_actions)])
#         self.action = np.argmax(posterior)
#
#         self.t += 1
#         return self.action


class ThompsonSampling(base.Agent):
  """A Thompson sampling agent for Bernoulli rewards."""

  def __init__(self,
               action_spec: specs.DiscreteArray,
               seed: Optional[int] = None):
    self._num_actions = action_spec.num_values
    self._rng = np.random.RandomState(seed)
    self._posterior = [BetaBernoulli() for i in range(self._num_actions)]

  def policy(self, timestep: dm_env.TimeStep) -> base.Action:
    posterior = np.array([self._posterior[i].sample(self._rng)
                          for i in range(self._num_actions)])
    action = np.argmax(posterior)
    return action

  def update(self,
             timestep: dm_env.TimeStep,
             action: base.Action,
             new_timestep: dm_env.TimeStep) -> None:
    self._posterior[action].update(new_timestep.reward)
    # del timestep
    # del action
    # del new_timestep


def default_agent(obs_spec: specs.Array, action_spec: specs.DiscreteArray,
                  **kwargs) -> ThompsonSampling:
  del obs_spec  # for compatibility
  return ThompsonSampling(action_spec=action_spec, **kwargs)
