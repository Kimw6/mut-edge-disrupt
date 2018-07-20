#!/usr/bin/env python3
from pathlib import Path

from data_path_utils import create_data_path
import pandas as pd

from tcga import get_patient_barcode
from utils import strip_prefix

BASE_PATH = Path('~/data').expanduser()

muts = pd.read_table(
    BASE_PATH / 'tcga-brca-muts/TCGA.BRCA.mutect.96983226-d92a-449d-8890-e1b210cee0fe.DR-6.0.somatic.maf',
    skiprows=4,
)

patients = sorted(set(get_patient_barcode(b) for b in muts.Tumor_Sample_Barcode))
genes = sorted(set(muts.Hugo_Symbol))

lbs_dir = BASE_PATH / 'ppi-ligand-mut'
gene_metadata = pd.read_table(lbs_dir / 'mutLBSgene_basic.txt')
lbs_muts = pd.read_table(lbs_dir / 'mutLBSgene_tcga_cosmic_overlapped_mutations.txt')
lbs_genes = sorted(set(lbs_muts.gene))

missing_from_tcga = set(lbs_genes) - set(genes)
print('Genes in LBS data but not TCGA mutation data:', len(missing_from_tcga))

lbs_mut_set = set(zip(lbs_muts.gene, lbs_muts.nsSNV))
# set of tuples of (patient, gene, AA sub)
brca_muts_in_lbs = set()
for i, row in muts.iterrows():
    if isinstance(row.HGVSp_Short, float):
        # null
        continue
    aa_sub = strip_prefix(row.HGVSp_Short, 'p.')
    patient = get_patient_barcode(row.Tumor_Sample_Barcode)
    gene = row.Hugo_Symbol
    if (gene, aa_sub) in lbs_mut_set:
        item = (patient, gene, aa_sub)
        brca_muts_in_lbs.add(item)
        print('Found LBS mut:', item)

data_path = create_data_path('intersect_muts_lbs')

lbs_mut_df = pd.DataFrame(list(brca_muts_in_lbs))
lbs_mut_df.columns = ['patient', 'gene', 'aa_sub']
tcga_mut_path = data_path / 'brca_lbs_muts.csv'
print('Saving BRCA mutations in LBS DB to', tcga_mut_path)
lbs_mut_df.to_csv(tcga_mut_path, index=None)
