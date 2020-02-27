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

"""Compare TrieHH with SFP."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy
import scipy.stats
from collections import OrderedDict

from sfp import SimulateSFP
from triehh import SimulateTrieHH

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)

class Plot(object):

  def __init__(self, max_k):
    self.confidence = .95
    self.max_k = max_k
    self._load_true_frequencies()

  def _load_true_frequencies(self):
    """Initialization of the dictionary."""
    with open('word_frequencies.txt', 'rb') as fp:
      self.true_frequencies = pickle.load(fp)

  def get_mean_u_l(self, recall_values):
    data_mean = []
    ub = []
    lb = []
    for K in range(10, self.max_k):
      curr_mean = np.mean(recall_values[K])
      data_mean.append(curr_mean)
      n = len(recall_values[K])
      std_err = scipy.stats.sem(recall_values[K])
      h = std_err * scipy.stats.t.ppf((1 + self.confidence) / 2, n - 1)
      lb.append(curr_mean - h)
      ub.append(curr_mean + h)
    mean_u_l = [data_mean, ub, lb]
    return mean_u_l

  def precision(self, result):
    all_words_key = self.true_frequencies.keys()
    precision = 0
    for word in result:
      if word in all_words_key:
        precision += 1
    precision /= len(result)
    return precision

  def plot_f1_scores(self, triehh_all_results, sfp_all_results, epsilon):
    # CHANGE "apple" TO "sfp"
    # CLEAN THIS (REMOVE ANY EXCESS CODE NOT USED ANYMORE

    sorted_all = OrderedDict(sorted(self.true_frequencies.items(), key=lambda x: x[1], reverse = True))
    top_words = list(sorted_all.keys())[:self.max_k]

    all_f1_triehh = []
    all_f1_sfp = []
    k_values = []

    for K in range(10, self.max_k):
      k_values.append(K)

    f1_values_triehh = {}
    f1_values_sfp = {}
    f1_values_inter = {}

    for K in range(10, self.max_k):
      f1_values_triehh[K] = []
      f1_values_sfp[K] = []
      f1_values_inter[K] = []

    for triehh_result in triehh_all_results:
      for K in range(10, self.max_k):
        recall = 0
        for i in range(K):
          if top_words[i] in triehh_result:
            recall += 1
        recall = recall * 1.0/K
        f1_values_triehh[K].append(2*recall/(recall + 1))
    all_f1_triehh = self.get_mean_u_l(f1_values_triehh)

    sfp_precision_list = []
    for sfp_result in sfp_all_results:
      precision_sfp = self.precision(sfp_result)
      sfp_precision_list.append(precision_sfp)
      for K in range(10, self.max_k):
        recall_sfp = 0
        for i in range(K):
          if top_words[i] in sfp_result:
            recall_sfp += 1
        recall_sfp = recall_sfp * 1.0/K
        f1_values_sfp[K].append(2*precision_sfp*recall_sfp/(precision_sfp + recall_sfp))
    all_f1_sfp = self.get_mean_u_l(f1_values_sfp)

    _, ax1 = plt.subplots(figsize=(10, 7))
    ax1.set_xlabel('K', fontsize=16)
    ax1.set_ylabel('F1 Score', fontsize=16)


    ax1.plot(k_values, all_f1_triehh[0], color = 'purple', alpha = 1, label=r'TrieHH, $\varepsilon$ = '+str(epsilon))
    ax1.fill_between(k_values, all_f1_triehh[2], all_f1_triehh[1], color = 'violet', alpha = 0.3)

    ax1.plot(k_values, all_f1_sfp[0], color = 'blue', alpha = 1, label=r'SFP, $\varepsilon$ = '+str(epsilon))
    ax1.fill_between(k_values, all_f1_sfp[2], all_f1_sfp[1], color = 'skyblue', alpha = 0.3)


    plt.legend(loc=4, fontsize=14)

    plt.title('Top K F1 Score vs. K (Single Word)', fontsize=14)
    plt.savefig("f1_single.eps")
    plt.savefig("f1_single.png",  bbox_inches="tight")
    plt.close()

def main():
  # maximum value at which F1 score is calculated
  max_k = 300

  # length of longest word
  max_word_len = 10
  # epsilon for differential privacy
  epsilon = 4
  # delta for differential privacy
  delta = 2.3e-12

  # repeat simulation for num_runs times
  num_runs = 5
  
  simulate_triehh = SimulateTrieHH(
      max_word_len=max_word_len, epsilon=epsilon, delta=delta, num_runs=num_runs)
  triehh_heavy_hitters = simulate_triehh.get_heavy_hitters()

  simulate_sfp = SimulateSFP(
      max_word_len=max_word_len, epsilon=epsilon, delta=delta, num_runs=num_runs)
  sfp_heavy_hitters = simulate_sfp.get_heavy_hitters()

  plot = Plot(max_k)
  plot.plot_f1_scores(triehh_heavy_hitters, sfp_heavy_hitters, epsilon)


if __name__ == '__main__':
  main()
