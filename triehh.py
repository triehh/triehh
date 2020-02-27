# coding=utf-8
# Copyright 2020 The Federated Heavy Hitters AISTATS 2020 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An implementation of Trie Heavy Hitters (TrieHH).

This is intended to implement and simulate the Trie Heavy Hitters (TrieHH)
protocol presented in Algorithm 1 of our AISTATS 2020 submission.
"""

import collections
import math
import pickle
import random

from collections import defaultdict
import numpy as np
import scipy
import scipy.stats

class ServerState(object):
  def __init__(self):
    self.quit_sign = False
    self.trie = {}

class SimulateTrieHH(object):
  """Simulation for TrieHH."""

  def __init__(self, max_word_len=10, epsilon=1.0, delta = 2.3e-12, num_runs=5):
    self.MAX_L = max_word_len
    self.delta = delta
    self.epsilon = epsilon
    self.num_runs = num_runs
    self.clients = []
    self.client_num = 0
    self.server_state = ServerState()
    self._init_clients()
    self._set_theta()
    self._set_batch_size()

  def _init_clients(self):
    """Initialization of the dictionary."""
    with open('clients_triehh.txt', 'rb') as fp:
      self.clients = pickle.load(fp)
    self.client_num = len(self.clients)
    print(f'Total number of clients: {self.client_num}')

  def _set_theta(self):
    theta = 5  # initial guess
    delta_inverse = 1 / self.delta
    while ((theta - 3) / (theta - 2)) * math.factorial(theta) < delta_inverse:
      theta += 1
    while theta < np.e ** (self.epsilon/self.MAX_L) - 1:
      theta += 1
    self.theta = theta
    print(f'Theta used by TrieHH: {self.theta}')

  def _set_batch_size(self):
    # check Corollary 1 in our paper.
    # Done in _set_theta: We need to make sure theta >= np.e ** (self.epsilon/self.MAX_L) - 1
    self.batch_size = int( self.client_num * (np.e ** (self.epsilon/self.MAX_L) - 1)/(self.theta * np.e ** (self.epsilon/self.MAX_L)))
    print(f'Batch size used by TrieHH: {self.batch_size}')

  def client_vote(self, word, r):
    if len(word) < r:
      return 0

    pre = word[0:r-1]
    if pre and (pre not in self.server_state.trie):
      return 0

    return 1

  def client_updates(self, r):
    # I encourage you to think about how we could rewrite this function to do
    # one client update (i.e. return 1 vote from 1 chosen client).
    # Then you can have an outer for loop that iterates over chosen clients
    # and calls self.client_update() for each chosen and accumulates the votes.

    votes = defaultdict(int)
    voters = []
    for word in random.sample(self.clients, self.batch_size):
      voters.append(word)

    for word in voters:
      vote_result = self.client_vote(word, r)
      if vote_result > 0:
        votes[word[0:r]] += vote_result
    return votes

  def server_update(self, votes):
    # It might make more sense to define a small class called server_state
    # server_state can track 2 things: 1) updated trie, and 2) quit_sign
    # server_state can be initialized in the constructor of SimulateTrieHH
    # and server_update would just update server_state
    # (i.e, it would update self.server_state.trie & self.server_state.quit_sign)
    self.server_state.quit_sign = True
    for prefix in votes:
      if votes[prefix] >= self.theta:
        self.server_state.trie[prefix] = None
        self.server_state.quit_sign = False

  def start(self, batch_size):
    """Implementation of TrieHH."""
    self.server_state.trie.clear()
    r = 1
    while True:
      votes = self.client_updates(r)
      self.server_update(votes)
      r += 1
      if self.server_state.quit_sign or r > self.MAX_L:
        break

  def get_heavy_hitters(self):
    heavy_hitters = []
    for run in range(self.num_runs):
      self.start(self.batch_size)
      raw_result = self.server_state.trie.keys()
      results = []
      for word in raw_result:
        if word[-1:] == '$':
          results.append(word.rstrip('$'))
      print(f'Discovered {len(results)} heavy hitters in run #{run+1}')
      print(results)
      heavy_hitters.append(results)
    return heavy_hitters
