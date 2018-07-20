#!/usr/bin/env python2
"""
Must be used with commercial pymol.
"""
from __future__ import print_function
from os.path import expanduser, splitext
from pprint import pprint
import re
import sys

#pdb_path, aa_sub = sys.argv[-2:]
sys.argv, pdb_path, aa_sub = sys.argv[:-2], sys.argv[-2], sys.argv[-1]
pprint(sys.argv)

## run through pymol, eg.:
## pymol -qc mutate.py 1god A/94/ ASN

from pymol import cmd

verbose = False

def read_amino_acid_mapping():
    path = expanduser('~/data/amino_acid_names.tab')
    dots = re.compile(r'\.+')

    mapping = {}
    with open(path) as f:
        for line in f:
            pieces = dots.split(line.strip())
            letter, short_name = pieces[0], pieces[1].upper()
            mapping[letter] = short_name
            mapping[short_name] = letter

    return mapping

AA_SUB_RE = re.compile(r'([A-Z])(\d+)([A-Z])')

aa_mapping = read_amino_acid_mapping()

if verbose:
    print('Amino acid mapping:')
    pprint(aa_mapping)
    print(sys.argv)

m = AA_SUB_RE.match(aa_sub)
# Doesn't really matter that location is a str; we're just going to use
# it in string formatting very soon anyway
from_aa, location_str, to_aa = m.groups()
location = int(location_str)

mutant = aa_mapping[to_aa]

selection = 'A/{}/'.format(location)

print('PDB path:', pdb_path)
print('Raw AA substitution:', aa_sub)
print('Selection:', selection)
print('New amino acid:', mutant)

print('Starting mutagenesis')
cmd.wizard('mutagenesis')
print('Loading PDB')
cmd.load(pdb_path)

seq_str = cmd.get_fastastr('all')

if verbose:
    print(seq_str)

lines = seq_str.splitlines()
sequence = ''.join(lines[1:])

residue_indexes = []
def append_residue_index(resi, resn, name):
    residue_indexes.append(resi)

namespace = {'append_residue_index': append_residue_index}
cmd.iterate('(all)', 'append_residue_index(resi, resn, name)', space=namespace)

min_residue_index = int(residue_indexes[0])

if verbose:
    for i, aa in enumerate(sequence, min_residue_index):
        print('{:03}'.format(i), aa)

aa_loc = location - min_residue_index
print('Getting amino acid at location', aa_loc)
orig_aa = sequence[aa_loc]
if orig_aa != from_aa:
    raise ValueError('Wrong original amino acid: found {}, need {}'.format(orig_aa, from_aa))

cmd.refresh_wizard()
print('Applying mutation')
cmd.get_wizard().do_select(selection)
cmd.get_wizard().set_mode(mutant)
cmd.get_wizard().apply()
cmd.set_wizard()

stem, ext = splitext(pdb_path)
new_filename = '{}_mut{}{}'.format(stem, aa_sub, ext)
print('Saving to', new_filename)
cmd.save(new_filename)
print('Done')
