import csv
from itertools import chain
from pathlib import Path
from typing import Iterable

import pandas as pd

from utils import pairs_to_zero_one_dataframe

UNKNOWN_GENE_ID = '0'

def read_mutation_file(filename: Path, sample_adj_func):
    """
    Reads a MAF mutation file. Yields 2-tuples:

    [0] Gene name ("hugo symbol")
    [1] Sample ID, run through `sample_adj_func`
    """
    with filename.open() as f:
        r = csv.DictReader(f, delimiter='\t')
        for line in r:
            gene_id = line['Entrez_Gene_Id']
            sample_id = sample_adj_func(line['Tumor_Sample_Barcode'])
            if gene_id != UNKNOWN_GENE_ID:
                yield sample_id, gene_id

def mafs_to_matrix(maf_paths: Iterable[Path], sample_adj_func) -> pd.DataFrame:
    raw_muts = list(
        chain.from_iterable(
            read_mutation_file(maf_path, sample_adj_func)
            for maf_path in maf_paths
        )
    )
    return pairs_to_zero_one_dataframe(raw_muts)
