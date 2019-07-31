# coding=utf-8
# Copyright 2019 The Federated Heavy Hitters Neurips2019 Authors.
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
protocol presented in Algorithm 1 of our NeurIPS19 submission.
"""

import collections
import math
import pickle
import random

from dict_trie import Trie
import numpy as np
import scipy
import scipy.stats


class SimulateTrieHH(object):
  """Simulation for TrieHH."""

  def __init__(self, max_k=300, max_word_len=10, epsilon=4, delta=2.3e-12):
    self.max_word_len = max_word_len
    self.epsilon = epsilon
    self.delta = delta
    # confidence interval for F1 score calculation
    self.confidence = .95
    # maximum value at which F1 score is calculated
    self.max_k = max_k
    self._load_clients_and_ground_truth()
    self._set_theta()
    self._set_batch_size()

  def _set_batch_size(self):
    self.batch_size = int(
        self.client_num * (np.e**(self.epsilon / self.max_word_len) - 1) /
        (self.theta * np.e**(self.epsilon / self.max_word_len)))

  def _set_theta(self):
    theta = 9  # initial guess
    delta_inverse = 1 / self.delta
    while ((theta - 3) / (theta - 2)) * math.factorial(theta) < delta_inverse:
      theta += 1
    self.theta = theta

  def _load_clients_and_ground_truth(self):
    """Load client words and ground truth frequencies."""
    with open('clients_triehh.txt', 'rb') as fp:
      self.clients = pickle.load(fp)
    with open('word_frequencies.txt', 'rb') as fp:
      self.true_frequencies = pickle.load(fp)

    self.client_num = len(self.clients)
    print('client number', 'self.client_num')

  def start(self):
    """Run TrieHH."""
    self.trie = Trie()
    r = 1
    while True:
      votes = {}

      voters = []
      for word in random.sample(self.clients, self.batch_size):
        voters.append(word)

      for word in voters:
        if len(word) < r:
          continue

        curr = word[0:r]
        pre = word[0:r - 1]
        if pre and (pre not in self.trie):
          continue

        if curr not in votes:
          votes[curr] = 1
        else:
          votes[curr] += 1
      quit_sign = True
      for prefix in votes:
        if votes[prefix] >= self.theta:
          self.trie.add(prefix)
          quit_sign = False
      r += 1
      if quit_sign or r > self.max_word_len:
        break

  def get_mean_u_l(self, recall_values):
    """Compute average recall values and confidence intervals."""
    data_mean = []
    ub = []
    lb = []
    for k in range(10, self.max_k):
      curr_mean = np.mean(recall_values[k])
      data_mean.append(curr_mean)
      n = len(recall_values[k])
      std_err = scipy.stats.sem(recall_values[k])
      h = std_err * scipy.stats.t.ppf((1 + self.confidence) / 2, n - 1)
      lb.append(curr_mean - h)
      ub.append(curr_mean + h)
    mean_u_l = [data_mean, ub, lb]
    return mean_u_l

  def get_f1_scores(self):
    """Calculate and return the f1 scores."""
    sorted_all = collections.OrderedDict(
        sorted(self.true_frequencies.items(), key=lambda x: x[1], reverse=True))
    top_words = list(sorted_all.keys())[:self.max_k]

    k_values = np.arange(10, self.max_k, 1)

    f1_scores = {}
    for k in k_values:
      f1_scores[k] = []

    for _ in range(10):
      self.start()
      raw_result = list(self.trie)
      result = []
      for word in raw_result:
        if word[-1:] == '$':
          result.append(word.rstrip('$'))

      for k in k_values:
        recall = 0
        for i in range(k):
          if top_words[i] in result:
            recall += 1
        recall = recall * 1.0 / k
        # precision is always equal to 1 for TrieHH
        f1_scores[k].append(2 * recall / (recall + 1))
    f1_scores_with_confidence = self.get_mean_u_l(f1_scores)
    return f1_scores_with_confidence
