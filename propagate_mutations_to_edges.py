#!/usr/bin/env python3
from argparse import ArgumentParser
from multiprocessing import Pool
import pickle

from data_path_utils import (
    create_data_path,
    find_newest_data_path,
)
import numpy as np
import pandas as pd

from gene_mappings import read_hugo_entrez_mapping
from network import insert_dummy_edge_nodes, join_string_keys
from utils import DEFAULT_ALPHA, sorted_intersection
from propagation import propagate, normalize

DEFAULT_SUBPROCESSES = 2

p = ArgumentParser()
p.add_argument('-s', '--subprocesses', type=int, default=DEFAULT_SUBPROCESSES)
p.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)

if __name__ == '__main__':
    args = p.parse_args()
else:
    args = p.parse_args([])

data_path = create_data_path(f'propagate_mutations_edges_alpha_{args.alpha:.2f}')

with (find_newest_data_path('build_hippie_network') / 'network.pickle').open('rb') as f:
    orig_network = pickle.load(f)
print('Loaded network')

self_edge_count = 0
# HACK: remove self edges
for node in orig_network.nodes:
    if orig_network.has_edge(node, node):
        orig_network.remove_edge(node, node)
        self_edge_count += 1
print(f'Removed {self_edge_count} self-edges from original network')

network = insert_dummy_edge_nodes(orig_network, edge_name_func=join_string_keys)

w_prime = normalize(network)
node_set = set(network.nodes())
nodes = sorted(node_set)
node_count = len(nodes)

with pd.HDFStore(find_newest_data_path('parse_tcga_mutations') / 'mutations.hdf5') as store:
    mutations = store['muts']
print('Read mutations')

hugo_entrez_mapping = read_hugo_entrez_mapping()

def propagate_mutations(param_tuple):
    i, sample, label, sample_count, vec = param_tuple
    if not i % 100:
        print('{}: done with {} samples ({:.2f}%)'.format(label, i, (i * 100) / sample_count))
    vector = np.matrix(vec).reshape((node_count, 1))
    propagated = propagate(w_prime, vector, alpha=args.alpha, verbose=False)
    return sample, propagated

data = mutations
label = 'mutations'

sample_count = len(data.index)
data_gene_set = set(data.columns)

common_genes = sorted_intersection(data.columns, node_set)
common_genes_path = data_path / '{}_common_genes.txt'.format(label)
print('{}: saving {} common genes to {}'.format(label, len(common_genes), common_genes_path))
with common_genes_path.open('w') as f:
    for gene in common_genes:
        print(gene, file=f)

only_mut_genes = sorted(data_gene_set - node_set)
only_mut_genes_path = data_path / '{}_only_mut_genes.txt'.format(label)
print('{}: saving {} data-only genes to {}'.format(label, len(only_mut_genes), only_mut_genes_path))
with only_mut_genes_path.open('w') as f:
    for gene in only_mut_genes:
        print(gene, file=f)

only_network_genes = sorted(node_set - data_gene_set)
only_network_genes_path = data_path / '{}_only_network_genes.txt'.format(label)
print('{}: saving {} network-only genes to {}'.format(label, len(only_network_genes), only_network_genes_path))
with only_network_genes_path.open('w') as f:
    for gene in only_network_genes:
        print(gene, file=f)

data_network = pd.DataFrame(0.0, columns=nodes, index=data.index)
data_propagated = pd.DataFrame(0.0, columns=nodes, index=data.index)
data_network.loc[:, common_genes] = data.loc[:, common_genes]

param_generator = (
    (i, sample, label, sample_count, data_network.loc[sample, :])
    for i, sample in enumerate(data_network.index)
)

with Pool(args.subprocesses) as pool:
    for sample, propagated in pool.imap_unordered(
            propagate_mutations,
            param_generator,
    ):
        data_propagated.loc[sample, :] = np.array(propagated).reshape((node_count,))

hdf5_path = data_path / 'data_propagated.hdf5'.format(label)
print('Saving data to', hdf5_path)
with pd.HDFStore(str(hdf5_path)) as store:
    store[label] = data_propagated
