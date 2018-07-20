#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path

from data_path_utils import create_data_path
import pandas as pd

from parse_maf import mafs_to_matrix
from utils import pairs_and_values_to_dataframe

BARCODE_DELIM = '-'
def get_patient_barcode(tumor_sample_barcode: str) -> str:
    pieces = tumor_sample_barcode.split(BARCODE_DELIM)
    return BARCODE_DELIM.join(pieces[:3])

def import_tcga_expr(directory: Path) -> pd.DataFrame:
    expr_values = []
    pairs = []
    nan = float('nan')
    for filename in directory.iterdir():
        with filename.open() as f:
            header_pieces = next(f).strip().split('\t')
            sample_id = get_patient_barcode(header_pieces[1])
            # Next line is "Composite Element REF...", ignore that too
            next(f)
            for line in f:
                if line:
                    gene, expr_str = line.strip().split('\t')
                    try:
                        expr_value = float(expr_str)
                    except ValueError:
                        expr_value = nan
                    expr_values.append(expr_value)
                    pairs.append((sample_id, gene))

    return pairs_and_values_to_dataframe(pairs, expr_values)

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('tcga_mut_path', type=Path, nargs='+')
    args = p.parse_args()

    data_path = create_data_path('parse_tcga_mutations')
    print('Reading TCGA mutation data from', args.tcga_mut_path)
    muts = mafs_to_matrix(args.tcga_mut_path, get_patient_barcode)
    print('Mutation data shape:', muts.shape)
    mut_output_path = data_path / 'mutations.hdf5'
    print('Saving mutations to', mut_output_path)
    with pd.HDFStore(mut_output_path) as store:
        store['muts'] = muts
