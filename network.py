#!/usr/bin/env python3
from argparse import ArgumentParser
import csv
from math import sqrt
from pathlib import Path
import pickle
from typing import Any, Iterable

from data_path_utils import create_data_path
import networkx as nx

HIPPIE_GENE_INCIDES = ['ID Interactor A', 'ID Interactor B']
HIPPIE_WEIGHT_INDEX = 'Confidence Value'
def get_hippie_proteins_from_line(line):
    data = []
    for index in HIPPIE_GENE_INCIDES:
        value = line[index]
        pieces = value.split(':')
        try:
            data.append(pieces[1])
        except IndexError:
            print('Bad gene data: {}'.format(value))
            return
    if any(d is None for d in data):
        return
    weight = float(line[HIPPIE_WEIGHT_INDEX])
    if weight == 0:
        return
    data.append({'weight': weight})
    return tuple(data)

def get_hippie_proteins(filename: Path):
    with filename.open() as f:
        r = csv.DictReader(f, delimiter='\t')
        for i, line in enumerate(r, 1):
            proteins = get_hippie_proteins_from_line(line)
            if proteins:
                yield proteins
            else:
                print('Line {} bad'.format(i))

def build_hippie_network(filename: Path) -> nx.Graph:
    g = nx.Graph()
    g.add_edges_from(get_hippie_proteins(filename))
    return g

def join_string_keys(args: Iterable[Any]) -> str:
    return '_'.join(str(arg) for arg in args)

def insert_dummy_edge_nodes(input_network: nx.Graph, edge_name_func=tuple) -> nx.Graph:
    """
    :param input_network: undirected, maybe weighted
    :param edge_name_func: Function used to create names for new nodes. Takes one
    iterable argument of the (sorted) node names, and returns a single object used
    for new node labels. By default, this is `tuple`, to preserve as much
    information as possible about the input network, but this can be overridden to
    create string labels or anything else desired. The function `join_string_keys`
    is designed to be used as an alternative to `tuple`.
    :return: A new network, with dummy nodes inserted in the middle of each edge.

    For an original edge (A, B) with weight w(A, B), the new edge weights w((A,), (A, B))
    and w((A, B), (B,)), are equal to √w(A, B), preserving the property that for a
    path P, the reliability r(P) = ∏_{n1, n2 ∈ P} w(n1, n2) is unchanged after this
    transformation.
    """
    network = nx.Graph()

    for node in input_network.nodes:
        network.add_node(edge_name_func([node]))

    for n1, n2, weight in input_network.edges.data('weight'):
        new_n1 = edge_name_func([n1])
        new_n2 = edge_name_func([n2])
        new_node = edge_name_func(sorted([n1, n2]))
        if weight is None:
            new_weight = 1
        else:
            new_weight = sqrt(weight)

        network.add_edge(new_n1, new_node, weight=new_weight)
        network.add_edge(new_node, new_n2, weight=new_weight)

    return network

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('hippie_input_path', type=Path)
    args = p.parse_args()

    data_path = create_data_path('build_hippie_network')
    print('Reading HIPPIE network from', args.hippie_input_path)
    network = build_hippie_network(args.hippie_input_path)
    print('Network contains {} nodes, {} edges'.format(len(network.nodes()), len(network.edges())))
    network_output_path = data_path / 'network.pickle'
    print('Saving network to', network_output_path)
    with network_output_path.open('wb') as f:
        pickle.dump(network, f, protocol=pickle.HIGHEST_PROTOCOL)
