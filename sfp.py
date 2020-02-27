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

"""An implementation of Sequence Frequency Puzzle (SFP).

This is intended to implement and simulate the Sequency Frequency Puzzle (SFP)
algorithm, a locally differentially private protocol based on count sketches by
Apple. Our code is based on Algorithm 10 in the following paper:

Learning with Privacy at Scale
    https://machinelearning.apple.com/docs/learning-with-privacy-at-scale/appledifferentialprivacysystem.pdf
"""

import collections
import functools
import hashlib
import itertools
import math
import multiprocessing
import pickle
import random
import re
import string
import time

import numpy as np
import scipy
from scipy import optimize
import scipy.stats


class SimulateSFP(object):
  """Simulation for SFP."""
  def __init__(self, max_word_len=10, epsilon=4.0, delta = 2.3e-12, num_runs=5):
    self.max_word_len = max_word_len
    self.l_range = range(0, 9, 2)
    self.half_l = int(self.max_word_len / 2)
    self.epsilon = epsilon
    self.delta = delta
    self.num_runs = num_runs
    self._load_clients_and_ground_truth()
    # size of hash functions used
    self.m = 1024
    # number of hash functions used
    self.k = 2048
    # pruning threshold
    self.t = 20

    self.leps = self.central_to_local(self.epsilon, self.delta, self.client_num)

    # init salts for hash functions
    self.h = []
    for i in range(self.k):
      self.h.append(str(i))

  def general_blanket_formula(self, leps, n, epsilon):
    if epsilon <= 0:
      return float('inf')
    tmp = ((math.exp(epsilon) - 1) / (math.exp(epsilon) + 1) /
           (math.exp(leps) - math.exp(-leps)))**2
    c = 1 - math.exp(-2)
    delta = 1.0 / tmp * (math.exp(epsilon) - 1) / 4.0 / n * math.exp(
        -c * n * min(math.exp(-leps), tmp))
    return delta

  def central_to_local(self, eps, delta, n):
    """Compute local epsilon given central_epsilon and central_delta."""
    ini = 0.5 * math.log(n / math.log(1 / delta))
    sol = optimize.root(
        lambda leps: self.general_blanket_formula(leps, n, eps) - delta, ini)
    return sol.x[0]

  def _load_clients_and_ground_truth(self):
    """Load client words. """
    with open('clients_sfp.txt', 'rb') as fp:
      self.clients = pickle.load(fp)

    self.client_num = len(self.clients)
    print('client number', self.client_num)

  def hash_func(self, salt, s):
    """Hash function: hash a string s to 10 bits.

    Args:
      salt: Different salt stands for different hash functions
      s: String to be hashed

    Returns:
      A 10-bit hash result.
    """
    h = hashlib.sha256()
    h.update(salt.encode('utf-8') + s.encode('utf-8'))
    return int(int(h.hexdigest(), 16) & 0x3ff)

  def hash_func_fixed(self, s):
    """Hash function: hash a string s to 8 bits.

    Args:
      s: String to be hashed

    Returns:
      A 8-bit hash result.
    """
    h = hashlib.sha256()
    h.update(s.encode('utf-8'))
    return int(int(h.hexdigest(), 16) & 0xff)

  def encode_string(self, s, k, m, h, epsilon):
    """Provides an epsilon locally differentially private encoding of a string.

    Args:
      s: A string to be encoded.
      k: Number of available hash functions.
      m: Number of integers hash functions map to.
      h: A dictionary of salts, stand for different hash functions.
      epsilon: Target differnetial privacy level.

    Returns:
      v1: An epsilon locally differentially private encoding of s.
      hash_function_index: the index of the chosen hash function.
    """
    hash_function_index = np.random.randint(k)
    v = -1.0 * np.ones(m)

    # flip h_j(s) in v
    v[self.hash_func(h[hash_function_index], s)] = 1

    # create vector b
    p = (math.e**(epsilon / 2.0)) / (math.e**(epsilon / 2.0) + 1)
    b = 2 * (np.random.random(m) <= p) - 1

    # return component wise product of v, b
    v1 = v * b

    return v1, hash_function_index

  def encode_strings(self, s, k, m, h, epsilon):
    """Provides an epsilon locally differentially private encoding of a string.

    Args:
      s: A list of strings to be encoded.
      k: Number of available hash functions.
      m: Number of integers hash functions map to.
      h: A dictionary of salts, stand for different hash functions.
      epsilon: Target differnetial privacy level.

    Returns:
      v1: An epsilon locally differentially private encoding of s.
      hash_function_index: the index of the chosen hash function.
    """
    n = len(s)
    hash_function_indexes = np.random.randint(k, size=n)
    v = -1.0 * np.ones((n, m))

    # flip h_j(s) in v
    for i in range(n):
      v[i][self.hash_func(h[hash_function_indexes[i]], s[i])] = 1

    # create vector b
    p = (math.e**(epsilon / 2.0)) / (math.e**(epsilon / 2.0) + 1)
    b = 2 * (np.random.random((n, m)) <= p) - 1

    # return component wise product of v, b
    v1 = v * b

    return v1, hash_function_indexes

  def update_count_sketch(self, hash_indexes, n, k, noisy_data):
    count_sketch = np.zeros((k, 1))
    for i in range(n):
      count_sketch[hash_indexes[i]] += noisy_data[i]
    return count_sketch

  def count_mean_sketch_parallel(self, encoded_vectors, hash_indexes, epsilon,
                                 k, m, n):
    """A Parallel Implementation of Count Mean Sketch.

    Args:
      encoded_vectors: An n by m numpy array containing encoded vectors.
      hash_indexes: A list of indexes of hash functions by all the users.
      epsilon: The local differential privacy level.
      k: Number of available hash functions.
      m: Number of integers hash functions map to.
      n: Total number of users.

    Returns:
      m_matrix: count mean sketch matrix.
    """
    m_matrix = np.zeros((k, m))

    # constant c_epsilon
    ce = (math.e**(epsilon / 2.0) + 1.0) / (math.e**(epsilon / 2.0) - 1.0)

    debiased_encoded_vectors = k * (
        ce / 2.0 * encoded_vectors + 0.5 * np.ones(m))

    num_cores = multiprocessing.cpu_count()
    p = multiprocessing.Pool(num_cores - 10)

    get_count_sketch_column = functools.partial(self.update_count_sketch,
                                                hash_indexes, n, k)
    count_sketch_columns = p.map(get_count_sketch_column,
                                 list(debiased_encoded_vectors.T))
    m_matrix = np.concatenate(count_sketch_columns, axis=1)
    return m_matrix

  def freq_oracle(self, s, m_matrix, k, m, h, n):
    """Estimate frequence of string s given m_matrix, k, m.

    Args:
      s: Input string to estimate its frequence.
      m_matrix: count mean sketch matrix.
      k: number of available hash functions.
      m: number of integers hash functions map to.
      h: A dictionary of hash functions.
      n: number of users.

    Returns:
      f: the estimated frequency of s
    """
    f = 0.0
    for l in range(k):
      f += m_matrix[l][self.hash_func(h[l], s)]
    f /= k
    f = m / (m - 1.0) * (f - 1.0 * n / m)
    return f

  def freq_oracle_parallel(self, m_matrix, k, m, h, n, s):
    """Estimate frequence of string s given m_matrix, k, m.

    Args:
      m_matrix: count mean sketch matrix.
      k: number of available hash functions.
      m: number of integers hash functions map to.
      h: A dictionary of hash functions.
      n: number of users.
      s: Input string to estimate its frequence.

    Returns:
      f: the estimated frequency of s
    """
    f = 0.0
    for l in range(k):
      f += m_matrix[l][self.hash_func(h[l], s)]
    f /= k
    f = m / (m - 1.0) * (f - 1.0 * n / m)
    return f

  def encode_string_2char(self, s, k, m, h, epsilon):
    """Provides an epsilon locally differentially private encoding of a string.

      of length 10. First randomly select an index from 1, 3, 5, 7, 9, call
      encode_string() after some pre-processing.

    Args:
      s: A string to be encoded.
      k: number of available hash functions.
      m: number of integers hash functions map to.
      h: A dictionary of hash functions.
      epsilon: Target differnetial privacy level.

    Returns:
      v1: An epsilon locally differentially private encoding of s.
      hash_function_index: The index of the chosen hash function.
      l: random position chosen from {1, 3 ,5, 7, 9}
    """
    # randomly select l from {1, 3, 5, 7, 9}
    l = random.randrange(0, 9, 2)
    r = str(self.hash_func_fixed(s)) + s[l:l + 2]
    v1, hash_function_index = self.encode_string(r, k, m, h, epsilon)
    return v1, hash_function_index, l

  def encode_strings_2chars(self, s, k, m, h, epsilon):
    """Provides an epsilon locally differentially private encoding.

      String length = 10. First randomly select an index from 1, 3, 5, 7, 9,
      call encode_string() after some pre-processing.

    Args:
      s: A list of strings to be encoded.
      k: number of available hash functions.
      m: number of integers hash functions map to.
      h: A dictionary of hash functions.
      epsilon: Target differnetial privacy level.

    Returns:
      v1: An epsilon locally differentially private encoding of s.
      hash_function_index: The index of the chosen hash function.
      l: random position chosen from {1, 3 ,5, 7, 9}
    """
    # randomly select l from {1, 3, 5, 7, 9}
    n = len(s)
    l = 2 * np.random.randint(self.half_l, size=n)
    r = []
    for i in range(n):
      r.append(str(self.hash_func_fixed(s[i])) + s[i][l[i]:l[i] + 2])
    v1, hash_function_indexes = self.encode_strings(r, k, m, h, epsilon)
    return v1, hash_function_indexes, l

  def get_heavy_hitters_parallel(self, encoded_vectors, hash_indexes, l_array,
                                 threshold, epsilon, k, m, h, n):
    """Estimate heavy hitters by the encodings from all users.

    Args:
      encoded_vectors: A list of encodings returned by all the users.
      hash_indexes: A list of indexes of hash functions by all the users.
      l_array: A list of random indexes l returned by all the usrs.
      threshold: parameter to tune the tradeoff between recall and precision.
      epsilon: The local differential privacy level.
      k: number of available hash functions.
      m: number of integers hash functions map to.
      h: A dictionary of hash functions.
      n: number of users

    Returns:
      all_heavy_hitters: a list of heavy hitters
    """
    m_fix = 256

    s_set = {}
    for l in self.l_range:
      s_set[l] = []

    # s_set[l] is the set of users chose l
    for i in range(n):
      l = l_array[i]
      s_set[l].append(i)

    start_time = time.time()
    m_matrix = {}
    # Calculate m_matrix[l] for all l, create subsets of encoded_vectors,
    # hash_indexes
    for l in self.l_range:
      sub_encoded_vectors = np.asarray(encoded_vectors)[s_set[l]]
      sub_hash_indexes = np.asarray(hash_indexes)[s_set[l]]
      sub_n = len(s_set[l])

      print('CMS for l=', l)
      m_matrix[l] = self.count_mean_sketch_parallel(sub_encoded_vectors,
                                                    sub_hash_indexes, epsilon,
                                                    k, m, sub_n)

    end_time = time.time()
    print('SKETCHING TIME', end_time - start_time)
    # Create all possible strings, total#: 256*26*26 n
    all_strings = []
    extra_symbols = ['@', '#', '-', ';', '(', ')', '*', ':', '.', '\'', '/', '+', '$']
    for i in range(m_fix):
      for c1 in list(string.ascii_lowercase) + extra_symbols:
        for c2 in list(string.ascii_lowercase) + extra_symbols:
          curr_string = str(i) + c1 + c2
          all_strings.append(curr_string)

    # all_freq: get frequence (f_l) for every string,
    # q[l]: sequences with top T frequency
    start_time = time.time()
    q = {}
    num_cores = multiprocessing.cpu_count()
    p = multiprocessing.Pool(num_cores - 10)
    for l in self.l_range:
      get_freq = functools.partial(self.freq_oracle_parallel, m_matrix[l], k, m,
                                   h, len(s_set[l]))
      all_strings_freq = p.map(get_freq, all_strings)
      common_string_indexes = np.argsort(
          np.array(all_strings_freq))[-threshold:]
      q[l] = [all_strings[indx] for indx in common_string_indexes]
    end_time = time.time()
    print('ESTIMATING FREQ TIME', end_time - start_time)

    # heavy_hitters[w]: the heavy hitters we get from w, get w||q_l
    # from every q_l
    heavy_hitters = {}
    for i in range(m_fix):
      w = str(i)
      qw = {}
      for l in self.l_range:
        qw[l] = []
        for curr_string in q[l]:
          if re.search(r'\d+', curr_string).group(0) == w:
            qw[l].append(curr_string.replace(w, ''))

      heavy_hitters[w] = list(
          itertools.product(qw[0], qw[2], qw[4], qw[6], qw[8]))
    all_heavy_hitters = set()
    for i in range(m_fix):
      w = str(i)
      for item in heavy_hitters[w]:
        heavy_hitter = ''.join(item)
        heavy_hitter = heavy_hitter.split('$')[0]
        all_heavy_hitters.add(heavy_hitter)
    return list(all_heavy_hitters)

  def get_heavy_hitters(self):
    heavy_hitters = []
    for run in range(self.num_runs):
      encoded_vectors, hash_function_indexes, l = self.encode_strings_2chars(
          self.clients, self.k, self.m, self.h, self.leps)
      results = self.get_heavy_hitters_parallel(encoded_vectors,
                                                      hash_function_indexes, l,
                                                      self.t, self.epsilon,
                                                      self.k, self.m, self.h,
                                                      self.client_num)
      print(f'Discovered {len(results)} heavy hitters in run #{run+1}')
      print(results)
      heavy_hitters.append(results)
    return heavy_hitters
