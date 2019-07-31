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

"""Compare TrieHH with SFP."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sfp import SimulateSFP
from triehh import SimulateTrieHH

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)


def main():
  # maximum value at which F1 score is calculated
  max_k = 300
  k_values = np.arange(10, max_k, 1)

  # length of longest word
  max_word_len = 10
  # epsilon for differential privacy
  epsilon = 4
  # delta for differential privacy
  delta = 2.3e-12

  simulate_triehh = SimulateTrieHH(
      max_k=max_k, max_word_len=max_word_len, epsilon=epsilon, delta=delta)
  triehh_f1_scores = simulate_triehh.get_f1_scores()

  simulate_sfp = SimulateSFP(
      max_k=max_k, max_word_len=max_word_len, epsilon=epsilon, delta=delta)
  sfp_f1_scores = simulate_sfp.get_f1_scores()

  _, ax1 = plt.subplots()
  ax1.set_xlabel('K', fontsize=16)
  ax1.set_ylabel('F1 Score', fontsize=16)

  ax1.plot(
      k_values,
      triehh_f1_scores[0],
      color='purple',
      alpha=1,
      label=r'TrieHH, $\varepsilon = 4$')
  ax1.fill_between(
      k_values,
      triehh_f1_scores[2],
      triehh_f1_scores[1],
      color='violet',
      alpha=0.3)

  ax1.plot(
      k_values,
      sfp_f1_scores[0],
      color='blue',
      alpha=1,
      label=r'SFP, $\varepsilon = 4$')
  ax1.fill_between(
      k_values,
      sfp_f1_scores[2],
      sfp_f1_scores[1],
      color='lightblue',
      alpha=0.3)

  plt.legend(loc=1, fontsize=14)

  plt.title('Top K F1 Score vs. K (Single Word)', fontsize=14)
  plt.savefig('f1_single.eps')
  plt.savefig('f1_single.png', bbox_inches='tight')
  plt.close()


if __name__ == '__main__':
  main()
