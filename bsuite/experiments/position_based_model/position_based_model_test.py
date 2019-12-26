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
"""Tests for bsuite.experiments.bandit."""

# Import all required packages

# Internal dependencies.
from absl.testing import absltest
from bsuite.experiments.position_based_model import position_based_model
from dm_env import test_utils
import numpy as np


class InterfaceTest(test_utils.EnvironmentTestMixin, absltest.TestCase):

  def make_object_under_test(self):
    return position_based_model.PositionBasedModel(rewards=None,
    position_bias=None, bernoulli=True, seed=5)

  def make_action_sequence(self):
    valid_actions = range(11)
    rng = np.random.RandomState(42)

    for _ in range(100):
      yield [1,2,3]#rng.choice(valid_actions, 3, replace=False)

if __name__ == '__main__':
  absltest.main()
