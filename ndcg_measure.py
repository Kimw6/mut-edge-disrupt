#!/usr/bin/env python3
from typing import Sequence, Set

import numpy as np

def discounted_cumulative_gain(relevance: np.array) -> np.array:
    """
    :param relevance: array of relevance scores, in intended order
    :return: discounted cumulative gain array
    """
    scores = np.power(2, relevance) - 1
    log_ranks = np.log2(np.arange(2, relevance.shape[0] + 2))
    return np.cumsum(scores / log_ranks)

def normalized_discounted_cumulative_gain(relevance: np.array):
    """
    :param relevance: array of relevance scores, in intended order
    :return: discounted cumulative gain
    """
    # Seems to be the only real way to sort a NumPy array in
    # descending order, oddly enough
    s = np.sort(relevance)[::-1]
    return discounted_cumulative_gain(relevance) / discounted_cumulative_gain(s)

def calculate_gene_list_ndcg(genes: Sequence[str], validation_set: Set[str]) -> float:
    in_validation_set = [gene in validation_set for gene in genes]
    relevance = np.array(in_validation_set).astype(float)
    return normalized_discounted_cumulative_gain(relevance)
