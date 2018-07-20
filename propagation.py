from typing import Union

import networkx as nx
import numpy as np
from scipy import sparse

from utils import DEFAULT_ALPHA

Graph = Union[nx.Graph, nx.DiGraph]

def normalize_matrix(mat: sparse.spmatrix, verbose=False):
    def pv(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    pv('Calculating node degrees')
    mat_bool = mat.astype(bool)
    in_degree = mat_bool.sum(axis=0).A.flatten()
    out_degree = mat_bool.sum(axis=1).A.flatten()
    degree = (in_degree + out_degree) / 2
    pv('Calculating normalization')
    d = sparse.diags(1 / np.sqrt(degree), offsets=0)
    pv('Normalizing adjacency matrix')
    w_prime = d @ mat @ d
    #w_prime = sparse.csr_matrix(network_matrix / d_ij)
    return w_prime

def normalize(network: Graph, verbose=False):
    def pv(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    nodes = sorted(network.nodes())
    pv('Converting network to adjacency matrix')
    network_matrix = nx.to_scipy_sparse_matrix(network, nodes).T
    return normalize_matrix(network_matrix, verbose=verbose)

def normalize_matrix_directed(mat: sparse.spmatrix, verbose=False):
    def pv(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    pv('Calculating node degrees')
    mat_bool = mat.astype(bool)
    out_degree: np.ndarray = mat_bool.sum(axis=0).A.flatten()
    pv('Calculating normalization')
    degree_inv = 1 / out_degree
    # Don't introduce NaNs in the case of nodes with no outgoing edges.
    degree_inv[np.isinf(degree_inv)] = 0
    d = sparse.diags(degree_inv, offsets=0)
    pv('Normalizing adjacency matrix')
    w_prime = mat @ d
    #w_prime = sparse.csr_matrix(network_matrix / d_ij)
    return w_prime

def propagate_step(wp: sparse.spmatrix, previous: np.array, y: np.array, alpha: float) -> np.array:
    """
    :param wp: W' -- normalized weight matrix
    :param previous: F^{t - 1}
    :param y: Y -- prior knowledge
    :param alpha: propagation parameter
    :return: F^t
    """
    return alpha * wp @ previous + (1 - alpha) * y

def propagate(wp: sparse.spmatrix, y: np.array, alpha=DEFAULT_ALPHA, epsilon=1e-10, verbose=True) -> np.array:
    """
    :param wp: W' -- normalized weight matrix
    :param y: Y -- prior knowledge
    :param alpha: propagation parameter
    :return: vector with propagated signal
    """
    # Initial assignment F^1 = Y: use 'y' for previous and y in the first call
    f_prev, f_cur = y, propagate_step(wp, y, y, alpha)
    i = 1
    while np.linalg.norm(f_cur - f_prev) > epsilon:
        f_prev, f_cur = f_cur, propagate_step(wp, f_cur, y, alpha)
        if verbose:
            norm = np.linalg.norm(f_cur - f_prev)
            print('norm:', norm)
        i += 1
    if verbose:
        print('converged in {} iterations'.format(i))
    return f_cur
