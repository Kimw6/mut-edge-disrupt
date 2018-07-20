#!/usr/bin/env python3
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
import os
from pathlib import Path
import pickle
import sys
from textwrap import dedent
from typing import List, Set

from data_path_utils import (
    create_data_path,
    create_output_path,
    find_newest_data_path,
)
import matplotlib
if '__file__' in globals() or 'SSH_CONNECTION' in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import average_precision_score

from gene_mappings import read_hugo_entrez_mapping
from ndcg_measure import normalized_discounted_cumulative_gain
from utils import DEFAULT_ALPHA, PrData, RocData, to_matplotlib_sci_notation, new_plot, plot_cdf

p = ArgumentParser()
p.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)
if __name__ == '__main__' and (sys.argv and 'pydev' not in sys.argv[0]):
    args = p.parse_args()
else:
    args = p.parse_args([])

hist_bin_count = 50

def get_rank_k_edge_values(edge_prop: pd.DataFrame, k: int) -> pd.Series:
    edge_ranking = pd.Series(0.0, index=edge_prop.columns)
    k_new = edge_prop.shape[0] - k - 1
    for edge in edge_ranking.index:
        patient_edge_vector = edge_prop.loc[:, edge]
        top_kth_score = np.partition(patient_edge_vector, k_new)[k_new]
        edge_ranking.loc[edge] = top_kth_score
    return edge_ranking

def main():
    script_label = 'prop_edge_lbs_overlap'
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

    edge_prop = mut_edge_prop.loc[:, all_edges]

    shuffle_count = 100
    sorted_patients = sorted(patients_with_lbs_muts)
    patient_count = len(sorted_patients)
    ndcg = pd.Series(0.0, index=sorted_patients)
    shuffled_ndcg = pd.DataFrame(0.0, index=sorted_patients, columns=range(shuffle_count))
    lbs_edges_by_patient = pd.Series(0, index=sorted_patients)

    print('Loading shuffled data')
    prop_lbs_shuffle_path = find_newest_data_path('prop_edge_lbs_shuffle')

    with open(prop_lbs_shuffle_path / 'shuffled_muts_edges_by_patient.pickle', 'rb') as f:
        d = pickle.load(f)
        shuffled_by_patient = d['shuffled_by_patient']
        selected_edges_by_patient = d['selected_edges_by_patient']
        shuffled_edges_by_patient = d['shuffled_edges_by_patient']

    ## NDCG analysis

    # For each patient, rank edges by propagated mutation scores, assign label of 1 if
    # either node connected to that edge has a LBS mutation

    for i, patient in enumerate(patients_with_lbs_muts, 1):
        print(f'Computing NDCG for patient {i}/{patient_count}')

        edge_scores = mut_edge_prop.loc[patient, all_edges].copy().sort_values(ascending=False)
        selected_edges = selected_edges_by_patient[patient]
        shuffled_edge_list = shuffled_edges_by_patient[patient]

        relevance = np.array([e in selected_edges for e in edge_scores.index]).astype(float)
        ndcg.loc[patient] = normalized_discounted_cumulative_gain(relevance)[-1]

        for j, shuffled_edges in enumerate(shuffled_edge_list):
            shuffled_relevance = np.array(
                [e in shuffled_edges for e in edge_scores.index]
            ).astype(float)
            shuffled_ndcg.loc[patient, j] = normalized_discounted_cumulative_gain(shuffled_relevance)[-1]

    with pd.HDFStore(data_path / 'ndcg_data.hdf5') as store:
        store['ndcg'] = ndcg
        store['shuffled_ndcg'] = shuffled_ndcg
        store['lbs_edges_by_patient'] = lbs_edges_by_patient

    shuffled_ndcg_flat = shuffled_ndcg.unstack()
    #shuffled_ndcg_median = shuffled_ndcg.median(axis=1)

    with new_plot():
        ndcg.plot.hist(bins=hist_bin_count)
        plt.title('NDCG histogram')
        plt.xlabel('Patient NDCG score: selection of LBS edges by propagated edge score')

        figure_path = output_path / 'ndcg_hist.pdf'
        print('Saving NDCG histogram to', figure_path)
        plt.savefig(figure_path, bbox_inches='tight')

    with new_plot():
        shuffled_ndcg_flat.plot.hist(bins=hist_bin_count)
        plt.title('NDCG histogram')
        plt.xlabel('Patient NDCG score: selection of shuffled LBS edges by propagated edge score')

        figure_path = output_path / 'shuffled_ndcg_hist.pdf'
        print('Saving NDCG histogram to', figure_path)
        plt.savefig(figure_path, bbox_inches='tight')

    ndcg_ks = scipy.stats.ks_2samp(ndcg, shuffled_ndcg_flat)
    ndcg_ks_pvalue_str = to_matplotlib_sci_notation(ndcg_ks[1])

    with new_plot():
        ndcg.plot.hist(
            bins=hist_bin_count,
            alpha=0.8,
            label='Real NDCG',
            density=True,
        )
        shuffled_ndcg_flat.plot.hist(
            bins=hist_bin_count,
            alpha=0.8,
            label='Shuffled NDCG, across 100 permutations',
            density=True,
        )
        plt.xlabel('Patient NDCG score: selection of LBS edges by propagated edge score')
        plt.legend()
        plt.figtext(
            0.89,
            0.7,
            f'Kolmogorov-Smirnov $P = {ndcg_ks_pvalue_str}$',
            horizontalalignment='right',
        )

        figure_path = output_path / 'ndcg_both_hist.pdf'
        print('Saving NDCG histogram to', figure_path)
        plt.savefig(figure_path, bbox_inches='tight')

    ## /NDCG analysis

    ## PR and ROC AUC analysis

    roc_auc = pd.Series(0.0, index=sorted_patients)
    average_pr_scores = pd.Series(0.0, index=sorted_patients)
    shuffled_roc_auc = pd.DataFrame(0.0, index=sorted_patients, columns=range(shuffle_count))
    shuffled_average_pr_scores = pd.DataFrame(0.0, index=sorted_patients, columns=range(shuffle_count))
    # Maps patient IDs to performance objects
    roc_data_objects = {}
    pr_data_objects = {}

    for i, patient in enumerate(patients_with_lbs_muts, 1):
        print(f'Computing classifier performance for patient {i}/{patient_count}')
        selected_edges: Set[str] = selected_edges_by_patient[patient]
        edge_scores = mut_edge_prop.loc[patient, all_edges].copy()
        labels = np.array([e in selected_edges for e in edge_scores.index]).astype(float)

        rd = RocData.calculate(labels, edge_scores)
        roc_data_objects[patient] = rd
        roc_auc.loc[patient] = rd.auc

        pr = PrData.calculate(labels, edge_scores)
        pr_data_objects[patient] = pr
        average_pr_scores.loc[patient] = average_precision_score(labels, edge_scores)

        shuffled_edge_list: List[Set[str]] = shuffled_edges_by_patient[patient]

        for j, shuffled_edges in enumerate(shuffled_edge_list):
            shuffled_labels = np.array(
                [e in shuffled_edges for e in edge_scores.index]
            ).astype(float)

            shuffled_rd = RocData.calculate(shuffled_labels, edge_scores)
            shuffled_roc_auc.loc[patient, j] = shuffled_rd.auc

            shuffled_average_pr_scores.loc[patient, j] = average_precision_score(
                shuffled_labels,
                edge_scores,
            )

    with pd.HDFStore(data_path / 'classifier_data.hdf5') as store:
        store['roc_auc'] = roc_auc
        store['average_pr'] = average_pr_scores
        store['shuffled_roc_auc'] = shuffled_roc_auc
        store['shuffled_average_pr'] = shuffled_average_pr_scores

    with new_plot():
        roc_auc.plot.hist(bins=hist_bin_count)
        plt.title('ROC AUC histogram')
        plt.xlabel('Patient ROC AUC: selection of LBS edges by propagated edge score')

        figure_path = output_path / 'roc_auc_hist.pdf'
        print('Saving ROC AUC histogram to', figure_path)
        plt.savefig(figure_path, bbox_inches='tight')

    #shuffled_roc_auc_median = shuffled_roc_auc.median(axis=1)
    shuffled_roc_auc_flat = shuffled_roc_auc.unstack()

    with new_plot():
        shuffled_roc_auc_flat.plot.hist(bins=hist_bin_count)
        plt.title('ROC AUC histogram')
        plt.xlabel('Patient ROC AUC: selection of shuffled LBS edges by propagated edge score')

        figure_path = output_path / 'shuffled_roc_auc_hist.pdf'
        print('Saving ROC AUC histogram to', figure_path)
        plt.savefig(figure_path, bbox_inches='tight')

    roc_auc_ks = scipy.stats.ks_2samp(roc_auc, shuffled_roc_auc_flat)
    roc_auc_ks_pvalue_str = to_matplotlib_sci_notation(roc_auc_ks[1])

    with new_plot():
        roc_auc.plot.hist(
            bins=hist_bin_count,
            alpha=0.8,
            label='Real ROC AUC',
            density=True,
        )
        shuffled_roc_auc_flat.plot.hist(
            bins=50,
            alpha=0.8,
            label='Shuffled ROC AUC, across 100 permutations',
            density=True,
        )
        plt.xlabel('Patient ROC AUC: selection of LBS edges by propagated edge score')
        plt.legend()
        plt.figtext(
            0.14,
            0.7,
            f'Kolmogorov-Smirnov $P = {roc_auc_ks_pvalue_str}$',
            horizontalalignment='left',
        )

        figure_path = output_path / 'roc_auc_both_hist.pdf'
        print('Saving ROC AUC histogram to', figure_path)
        plt.savefig(figure_path, bbox_inches='tight')

    with new_plot():
        average_pr_scores.plot.hist(bins=hist_bin_count)
        plt.title('Average precision histogram')
        plt.xlabel('Average precision: selection of LBS edges by propagated edge score')

        figure_path = output_path / 'avg_prec_hist.pdf'
        print('Saving AP histogram to', figure_path)
        plt.savefig(figure_path, bbox_inches='tight')

    shuffled_average_pr_median = shuffled_average_pr_scores.median(axis=1)
    with new_plot():
        shuffled_average_pr_median.plot.hist(bins=hist_bin_count)
        plt.title('Average precision histogram')
        plt.xlabel('Average precision: selection of shuffled LBS edges by propagated edge score')

        figure_path = output_path / 'shuffled_avg_prec_hist.pdf'
        print('Saving AP histogram to', figure_path)
        plt.savefig(figure_path, bbox_inches='tight')

    top_n = 4
    rest_uniform = 6
    sorted_pr_scores = average_pr_scores.dropna().sort_values()
    usable_patient_count = sorted_pr_scores.shape[0]
    # Top 5, and 5 uniformly distributed from the rest
    patient_indexes = list(
        np.linspace(
            0,
            usable_patient_count - 1 - top_n,
            num=rest_uniform,
        ).astype(int)
    )
    patient_indexes.extend(range(usable_patient_count - top_n, usable_patient_count))
    selected_patients = sorted_pr_scores.index[list(reversed(patient_indexes))]

    with new_plot():
        plt.figure(figsize=(10, 10))
        for patient in selected_patients:
            prd = pr_data_objects[patient]
            plt.plot(prd.rec, prd.prec, label=patient)

        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.axes().set_aspect('equal', 'datalim')
        plt.legend()

        plt.title(f'Precision-recall: top {top_n} patients, uniform spacing of bottom {rest_uniform}')

        figure_path = output_path / 'pr_selected.pdf'
        print('Saving selected PR curves to', figure_path)
        plt.savefig(figure_path, bbox_inches='tight')

    ## /PR and ROC AUC analysis

    ## Spearman correlation P-value analysis
    spearman_pvalues = pd.Series(0.0, index=sorted_patients)
    shuffled_spearman_pvalues = pd.DataFrame(0.0, index=sorted_patients, columns=range(shuffle_count))

    for i, patient in enumerate(patients_with_lbs_muts, 1):
        print(f'Computing Spearman correlation P-value for patient {i}/{patient_count}')
        selected_edges: Set[str] = selected_edges_by_patient[patient]
        edge_scores = mut_edge_prop.loc[patient, all_edges].copy()
        labels = np.array([e in selected_edges for e in edge_scores.index]).astype(float)

        spearman_result = scipy.stats.spearmanr(edge_scores, labels)
        spearman_pvalue = spearman_result[1]

        spearman_pvalues.loc[patient] = spearman_pvalue

        shuffled_edge_list: List[Set[str]] = shuffled_edges_by_patient[patient]

        for j, shuffled_edges in enumerate(shuffled_edge_list):
            shuffled_labels = np.array(
                [e in shuffled_edges for e in edge_scores.index]
            ).astype(float)

            shuffled_spearman_result = scipy.stats.spearmanr(edge_scores, shuffled_labels)
            shuffled_spearman_pvalue = shuffled_spearman_result[1]

            shuffled_spearman_pvalues.loc[patient, j] = shuffled_spearman_pvalue

    sp_dir = Path('data/prop_edge_lbs_overlap_20180606-105746')
    with pd.HDFStore(sp_dir / 'spearman_pvalues.hdf5') as store:
        spearman_pvalues = store['spearman_pvalues']
        shuffled_spearman_pvalues = store['shuffled_spearman_pvalues']

    with pd.HDFStore(data_path / 'spearman_pvalues.hdf5') as store:
        store['spearman_pvalues'] = spearman_pvalues
        store['shuffled_spearman_pvalues'] = shuffled_spearman_pvalues

    nl10_spearman_pvalues_all = -np.log10(spearman_pvalues)
    nl10_spearman_pvalues = nl10_spearman_pvalues_all.loc[
        ~(nl10_spearman_pvalues_all.isnull()) &
        ~(np.isinf(nl10_spearman_pvalues_all))
    ]

    with new_plot():
        nl10_spearman_pvalues.plot.hist(bins=50)
        plt.title('Spearman $P$-value histogram')
        plt.xlabel('Spearman $P$-values ($-\\log_{10}$): LBS edges vs. prop. edge score')

        figure_path = output_path / 'spearman_pvalue_hist.pdf'
        print('Saving Spearman P-value histogram to', figure_path)
        plt.savefig(figure_path, bbox_inches='tight')

    shuffled_spearman_pvalues_flat = shuffled_spearman_pvalues.unstack()
    nl10_shuffled_spearman_pvalues_flat_all = -np.log10(shuffled_spearman_pvalues_flat)
    nl10_shuffled_spearman_pvalues_flat = nl10_shuffled_spearman_pvalues_flat_all.loc[
        ~(nl10_shuffled_spearman_pvalues_flat_all.isnull()) &
        ~(np.isinf(nl10_shuffled_spearman_pvalues_flat_all))
    ]

    with new_plot():
        nl10_shuffled_spearman_pvalues_flat.plot.hist(bins=50)
        plt.title('Spearman $P$-value histogram')
        plt.xlabel('Spearman $P$-values ($-\\log_{10}$): shuffled LBS edges vs. prop. edge score')

        figure_path = output_path / 'shuffled_spearman_pvalue_hist.pdf'
        print('Saving Spearman P-value histogram to', figure_path)
        plt.savefig(figure_path, bbox_inches='tight')

    spearman_ks = scipy.stats.ks_2samp(spearman_pvalues, shuffled_spearman_pvalues_flat)
    spearman_ks_pvalue_str = to_matplotlib_sci_notation(spearman_ks[1])

    with new_plot():
        nl10_spearman_pvalues.plot.hist(
            bins=hist_bin_count,
            alpha=0.8,
            label='Real Spearman $P$-values',
            density=True,
        )
        nl10_shuffled_spearman_pvalues_flat.plot.hist(
            bins=hist_bin_count,
            alpha=0.8,
            label='Shuffled Spearman $P$-values, across 100 permutations',
            density=True,
        )
        plt.xlabel('Spearman $P$-values ($-\\log_{10}$): LBS edges vs. prop. edge score')
        plt.legend()
        plt.figtext(
            0.89,
            0.7,
            f'Kolmogorov-Smirnov $P = {spearman_ks_pvalue_str}$',
            horizontalalignment='right',
        )

        figure_path = output_path / 'spearman_pvalues_both_hist.pdf'
        print('Saving Spearman P-value histogram to', figure_path)
        plt.savefig(figure_path, bbox_inches='tight')

    ## /Spearman correlation P-value analysis

    ## Overall ROC AUC
    print('Creating binary LBS edge matrix')
    lbs_edge_matrix = pd.DataFrame(0, index=edge_prop.index, columns=edge_prop.columns)
    for patient, edges in selected_edges_by_patient.items():
        lbs_edge_matrix.loc[patient, list(edges)] = 1

    lbs_matrix_path = data_path / 'lbs_edge_matrix.hdf5'
    print('Saving LBS edge matrix to', lbs_matrix_path)
    with pd.HDFStore(lbs_matrix_path) as store:
        store['lbs_edge_matrix'] = lbs_edge_matrix

    sorted_flattened_edge_scores = edge_prop.unstack().sort_values(ascending=False)
    flattened_lbs_edges = lbs_edge_matrix.unstack()
    ordered_flattened_lbs_edges = flattened_lbs_edges.loc[sorted_flattened_edge_scores.index]

    flattened_rd = RocData.calculate(ordered_flattened_lbs_edges, sorted_flattened_edge_scores)
    flattened_rd_path = data_path / 'flattened_rd.pickle'
    print('Saving flattened vector RocData to', flattened_rd_path)
    with open(flattened_rd_path, 'wb') as f:
        pickle.dump(flattened_rd, f)
    ## /Overall ROC AUC

    ## Survival analysis

    edge_prop_survival_dir = find_newest_data_path('edge_prop_survival')
    survival_data = pd.read_csv(edge_prop_survival_dir / 'univariate_surv_results.csv', index_col=0)
    # Indexed by gene/edge, across all patients
    surv_edge_sel = [('_' in i) for i in survival_data.index]
    edge_survival_data = survival_data.loc[surv_edge_sel, :]

    lbs_mut_edge_matrix = pd.DataFrame(
        0.0,
        index=sorted(selected_edges_by_patient),
        columns=all_edges,
    )
    for patient, edges in selected_edges_by_patient.items():
        lbs_mut_edge_matrix.loc[patient, list(edges)] = 1

    # Binary vector: is this edge incident on a LBS mut in at least one patient?
    edges_with_lbs_muts = lbs_mut_edge_matrix.sum(axis=0).astype(bool)

    surv_pvalues_with_lbs = edge_survival_data.loc[edges_with_lbs_muts, 'pvalue']
    surv_pvalues_with_lbs.name = 'With LBS'
    surv_pvalues_without_lbs = edge_survival_data.loc[~edges_with_lbs_muts, 'pvalue']
    surv_pvalues_without_lbs.name = 'Without LBS'

    ks_res = scipy.stats.ks_2samp(surv_pvalues_with_lbs, surv_pvalues_without_lbs)

    with new_plot():
        plot_cdf(surv_pvalues_with_lbs)
        plot_cdf(surv_pvalues_without_lbs)

        plt.legend()
        plt.ylabel('CDF')
        plt.xlabel('Univariate Cox Regression $P$-value')

        figure_path = output_path / 'surv_pvalue_cdfs.pdf'
        plt.savefig(figure_path, bbox_inches='tight')

    with new_plot():
        fig = plt.figure()

        surv_pvalues_with_lbs.plot.hist(bins=50, ax=plt.gca(), alpha=0.5)
        surv_pvalues_without_lbs.plot.hist(bins=50, ax=plt.gca(), alpha=0.5)

        plt.legend('topleft')
        plt.xlabel('Univariate Cox Regression $P$-value')

        figure_path = output_path / 'surv_pvalue_hist.pdf'
        plt.savefig(figure_path, bbox_inches='tight')

    ## /Survival analysis

    ## Permuted survival analysis

    pvalues = edge_survival_data.loc[:, 'r_square']

    ks_manual = (np.array([0.1, 0.2, 0.25, 0.3]) * edge_prop.shape[0]).astype(int)
    ks_auto = np.logspace(1, 3, num=15).astype(int)
    ks = sorted(chain(ks_manual, ks_auto))

    edge_count = 1000

    template = dedent('''
    \\begin{{frame}}[plain]
     \\begin{{center}}
      \\includegraphics[width=0.7\\textwidth]{{survival_rsquare_hist_k_{k}}}
     \\end{{center}}
    \\end{{frame}}
    ''')

    with open(data_path / 'figure_include.tex', 'w') as f:
        for k in ks:
            print(template.format(k=k), file=f)

    for k in ks:
        print('Computing edge ranking results for k =', k)
        edge_ranking = get_rank_k_edge_values(edge_prop, k)
        sorted_edge_scores = edge_ranking.sort_values(ascending=False)
        top_edges = sorted_edge_scores.iloc[:edge_count]
        top_edge_pvalues = pvalues.loc[top_edges.index]
        bottom_edges = sorted_edge_scores.iloc[edge_count:]
        permutation_count = 1000
        permutation_pvalues = pd.Series(0.0, index=range(permutation_count))
        for i in range(permutation_count):
            edge_selection = np.random.choice(bottom_edges.index, size=100)
            selected_pvalues = pvalues.loc[edge_selection]
            comparison_result = scipy.stats.mannwhitneyu(
                top_edge_pvalues,
                selected_pvalues,
                alternative='greater',
            )
            permutation_pvalues.iloc[i] = comparison_result.pvalue

        nl10_permutation_pvalues = -np.log10(permutation_pvalues)

        with new_plot():
            plt.figure(figsize=(5, 5))
            nl10_permutation_pvalues.plot.hist(bins=50)
            title = (
                f'Survival $R^2$: top {edge_count} edges ($k = {k}$) vs. '
                f'{permutation_count} random selections'
            )
            plt.title(title)
            plt.xlabel('$- \\log_{10}$($P$-value) from Mann-Whitney $U$ test')

            nl10_0_05 = -np.log10(0.05)
            plt.axvline(x=nl10_0_05, color='#FF0000FF')

            nl10_0_001 = -np.log10(0.001)
            plt.axvline(x=nl10_0_001, color='#000000FF')

            figure_path = output_path / f'survival_rsquare_hist_k_{k}.pdf'
            print('Saving survival R^2 histogram to', figure_path)
            plt.savefig(figure_path, bbox_inches='tight')

    ## /Permuted survival analysis

if __name__ == '__main__':
    main()
