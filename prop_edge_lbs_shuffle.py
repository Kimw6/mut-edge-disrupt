#!/usr/bin/env python3
from argparse import ArgumentParser
from collections import defaultdict
import os
import pickle
from random import sample
import sys
from typing import Dict, List, Set

from data_path_utils import (
    create_data_path,
    create_output_path,
    find_newest_data_path,
)
import matplotlib
if '__file__' in globals() or 'SSH_CONNECTION' in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from gene_mappings import read_hugo_entrez_mapping
from utils import DEFAULT_ALPHA, new_plot

p = ArgumentParser()
p.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)
if __name__ == '__main__' and (sys.argv and 'pydev' not in sys.argv[0]):
    args = p.parse_args()
else:
    args = p.parse_args([])

def main():
    script_label = 'prop_edge_lbs_shuffle'
    data_path = create_data_path(script_label)
    output_path = create_output_path(script_label)

    hem = read_hugo_entrez_mapping()

    lbs_mut_path = find_newest_data_path('intersect_muts_lbs')
    lbs_muts = pd.read_csv(lbs_mut_path / 'brca_lbs_muts.csv')

    prop_edge_path = find_newest_data_path(f'propagate_mutations_edges_alpha_{args.alpha:.2f}')
    with pd.HDFStore(prop_edge_path / 'data_propagated.hdf5') as store:
        mut_edge_prop = store['mutations']

    patients_with_lbs_muts = set(lbs_muts.patient)
    print('Patients with LBS mutations:', len(patients_with_lbs_muts))

    lbs_muts_by_patient = defaultdict(set)
    for i, row in lbs_muts.iterrows():
        if row.gene not in hem:
            print('Skipping gene', row.gene)
            continue
        lbs_muts_by_patient[row.patient].add(hem[row.gene])

    all_edge_set = {i for i in mut_edge_prop.columns if '_' in i}
    all_edges = sorted(all_edge_set)
    all_gene_set = set(mut_edge_prop.columns) - all_edge_set

    shuffle_count = 100
    sorted_patients = sorted(patients_with_lbs_muts)
    patient_count = len(sorted_patients)
    lbs_edges_by_patient = pd.Series(0, index=sorted_patients)

    # Assign label of 1 for an edge if either node has a LBS mutation
    selected_edges_by_patient: Dict[str, Set[str]] = {}
    shuffled_edges_by_patient: Dict[str, List[Set[str]]] = {}

    shuffled_by_patient = {}
    for i, patient in enumerate(patients_with_lbs_muts, 1):
        print(f'Shuffling LBS mutations for patient {patient} ({i}/{patient_count})')
        muts = lbs_muts_by_patient[patient]
        mut_count = len(muts)
        l = []
        for j in range(shuffle_count):
            other_genes = all_gene_set - muts
            new_muts = sample(other_genes, mut_count)
            l.append(new_muts)
        shuffled_by_patient[patient] = l

    # TODO: parallelize this; it's too slow
    for i, patient in enumerate(patients_with_lbs_muts, 1):
        print(f'Computing selected/shuffled edges for patient {i}/{patient_count}')
        lbs_genes = lbs_muts_by_patient[patient]
        selected_edges: Set[str] = set()
        shuffled_edges: List[Set[str]] = [set() for _ in range(shuffle_count)]
        edge_scores = mut_edge_prop.loc[patient, all_edges].copy().sort_values(ascending=False)
        for g1_g2 in edge_scores.index:
            g1, g2 = g1_g2.split('_')
            if g1 in lbs_genes or g2 in lbs_genes:
                selected_edges.add(g1_g2)
            # TODO: clean up iteration
            for j, shuffled_genes in enumerate(shuffled_by_patient[patient]):
                if g1 in shuffled_genes or g2 in shuffled_genes:
                    shuffled_edges[j].add(g1_g2)
        lbs_edges_by_patient.loc[patient] = len(selected_edges)
        selected_edges_by_patient[patient] = selected_edges
        shuffled_edges_by_patient[patient] = shuffled_edges

    selected_edge_count = pd.Series(
        {patient: len(edges) for patient, edges in selected_edges_by_patient.items()}
    ).sort_index()

    with new_plot():
        selected_edge_count.plot.hist(bins=25)
        plt.xlabel('Number of LBS-incident edges')
        plt.ylabel('Patients')

        figure_path = output_path / 'lbs_edge_count.pdf'
        print('Saving LBS edge count histogram to', figure_path)
        plt.savefig(figure_path, bbox_inches='tight')

    shuffled_data_path = data_path / 'shuffled_muts_edges_by_patient.pickle'
    print('Saving shuffled muts by patient to', shuffled_data_path)
    with open(shuffled_data_path, 'wb') as f:
        pickle.dump(
            {
                'shuffled_by_patient': shuffled_by_patient,
                'selected_edges_by_patient': selected_edges_by_patient,
                'shuffled_edges_by_patient': shuffled_edges_by_patient,
            },
            f,
        )

if __name__ == '__main__':
    main()
