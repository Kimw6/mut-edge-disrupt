This code is structured as several standalone scripts, each of which consume
and/or produce some data or plots.

Requirements
============

* Python 3.6 or newer
* R, for survival analysis
* PyMOL 2.0.7 or newer

Python packages
---------------

* NumPy, current version unimportant
* SciPy 1.0.0 or newer
* scikit-learn, current version unimportant
* Pandas 0.22 or newer
* NetworkX 2.0 or newer
* matplotlib 2.2.2 or newer
* data-path-utils, version unimportant

Usage
=====

Extract the data archive linked from
https://www.cs.cmu.edu/~mruffalo/mut-edge-disrupt/ into the directory
containing these scripts.

Most results in the manuscript are produced by the `prop_edge_lbs_overlap.py`
script, using the results of other scripts as input. Permuted ligand binding
site edge labels are computed by `prop_edge_lbs_shuffle.py`. Parsing of LBS
mutations and intersecting with TCGA somatic mutation data is done in
`intersect_muts_lbs.py`.

The `pymol_mutagenesis.py` script automates PyMOL's mutagenesis wizard, and
takes two command-line arguments: the path to a PDB file on disk, and an amino
acid substitution (e.g. F293L).

Edge smoothing is performed by the `propagate_mutations_to_edges.py` script,
and uses the default parameter of alpha = 0.8 unless overridden.

<!---
vim: set tw=79:
-->
