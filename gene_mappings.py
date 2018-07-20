import csv
import json
from pathlib import Path
from typing import Dict

from data_path_utils import DATA_PATH, find_newest_data_path
import pandas as pd

HUGO_ENTREZ_MAPPING_PATH = DATA_PATH / 'hugo_entrez_mapping.tab'
ENSEMBL_ENTREZ_MAPPING_PATH = DATA_PATH / 'ensembl_to_entrez.txt'
ENTREZ_UNIPROT_MAPPING_PATH = DATA_PATH / 'HUMAN_9606_idmapping_selected.tab.gz'

def read_hugo_entrez_mapping() -> Dict[str, str]:
    print('Reading Hugo to Entrez mapping')
    hugo_entrez_mapping = {}
    entrez_ids = set()
    with open(HUGO_ENTREZ_MAPPING_PATH) as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            entrez_id = row['Entrez Gene ID(supplied by NCBI)']
            entrez_ids.add(entrez_id)
            hugo_entrez_mapping[row['Approved Symbol']] = entrez_id
            for synonym in row['Synonyms'].split():
                hugo_entrez_mapping[synonym] = entrez_id

    # List of 2-tuples:
    #  [0] key in hugo_entrez_mapping
    #  [1] new key which will map to the same value
    manual_mapping_addition = [
        ('ADGRE5', 'CD97'),
    ]
    for key_existing, key_new in manual_mapping_addition:
        hugo_entrez_mapping[key_new] = hugo_entrez_mapping[key_existing]

    mygene_path = find_newest_data_path('query_mygene')
    with open(mygene_path / 'mapping.json') as f:
        hugo_entrez_mapping.update(json.load(f))

    print(
        'Read Hugo to Entrez mapping: {} gene names to {} Entrez IDs'.format(
            len(hugo_entrez_mapping),
            len(entrez_ids),
        )
    )

    return hugo_entrez_mapping

def read_entrez_hugo_mapping() -> Dict[str, str]:
    t = pd.read_table(HUGO_ENTREZ_MAPPING_PATH)
    sel = ~t.loc[:, 'Entrez Gene ID(supplied by NCBI)'].isnull()
    m = {
        str(int(entrez_id_float)): hugo_symbol
        for entrez_id_float, hugo_symbol in zip(
            t.loc[sel, 'Entrez Gene ID(supplied by NCBI)'],
            t.loc[sel, 'Approved Symbol'],
        )
    }
    # Check that no Entrez IDs were duplicated
    assert len(m) == sel.sum()
    return m

def read_ensembl_entrez_mapping() -> Dict[str, str]:
    ensembl_entrez_mapping = {}
    entrez_ids = set()
    with open(ENSEMBL_ENTREZ_MAPPING_PATH) as f:
        r = csv.DictReader(f)
        for line in r:
            entrez_id = line['EntrezGene ID']
            ensembl_id = line['Ensembl Gene ID']
            if entrez_id:
                entrez_ids.add(entrez_id)
                ensembl_entrez_mapping[ensembl_id] = entrez_id

    print(
        'Read Ensembl to Entrez mapping: {} gene names to {} Entrez IDs'.format(
            len(ensembl_entrez_mapping),
            len(entrez_ids),
        )
    )

    return ensembl_entrez_mapping

def read_entrez_uniprot_mapping() -> Dict[str, str]:
    entrez_uniprot_mapping = {}
    d = pd.read_table(ENTREZ_UNIPROT_MAPPING_PATH)
    entrez_id_present = d.loc[~d.iloc[:, 2].isnull(), :]

    for entrez_ids, uniprot_id in zip(entrez_id_present.iloc[:, 2], entrez_id_present.iloc[:, 0]):
        entrez_id_list = [ei.strip() for ei in entrez_ids.split(';')]
        for entrez_id in entrez_id_list:
            entrez_uniprot_mapping[entrez_id] = uniprot_id

    print(
        'Read Entrez to Uniprot mapping: {} Entrez IDs to {} Uniprot IDs'.format(
            len(entrez_uniprot_mapping),
            len(set(entrez_uniprot_mapping.values())),
        )
    )
    return entrez_uniprot_mapping
