The accompanying codes reproduce all figures and statistics presented in
"Controlling for multiple covariates" by Mark Tygert. This repository also
provides the LaTeX and BibTeX sources required for replicating the paper.

Be sure to ``pip install hilbertcurve`` prior to running any of this software
(the codes depend on [HilbertCurve](https://github.com/galtay/hilbertcurve)).
Also be sure to ``gunzip codes/cup98lrn.txt`` prior to running
``codes/kddcup98.py``.

The main files in the repository are the following:

``tex/multidim.pdf``
PDF version of the paper

``tex/multidim.tex``
LaTeX source for the paper

``tex/multidim.bib``
BibTeX source for the paper

``tex/diffs0.pdf`` ``tex/diffs1.pdf`` ``tex/sums0.pdf`` ``tex/sums1.pdf``
``tex/partition.pdf``
Graphics for Subsection 2.3 of the paper

``codes/acs.py``
Python script for processing the American Community Survey

``codes/psam_h06.csv``
Microdata from the 2019 American Community Survey of the U.S. Census Bureau

``codes/kddcup98.py``
Python script for processing the KDD Cup 1998 data

``codes/cup98lrn.txt.gz``
Data from the 1998 KDD Cup

``codes/synthetic.py``
Python script for generating and processing synthetic examples

``codes/hilbert.pdf``
Plot of an approximation with 255 line segments to the Hilbert curve in 2D

``codes/disjoint.py``
Functions for plotting differences between two subpops. with disjoint scores
(redistributed from the GitHub repo
[fbcddisgraph](https://github.com/facebookresearch/fbcddisgraph))

``codes/disjoint.py``
Functions for plotting differences of a subpop. from the full population
(redistributed from the GitHub repo
[fbcdgraph](https://github.com/facebookresearch/fbcdgraph))

``codes/subpop_weighted.py``
Functions for plotting differences of a subpop. from the full pop. with weights
(redistributed from the GitHub repo
[fbcdgraph](https://github.com/facebookresearch/fbcdgraph))

Regenerating all the figures requires running in the directory ``codes``
``acs.py``, ``kddcup98.py``, and ``synthetic.py``; issue the commands

    cd codes
    pip install hilbertcurve
    gunzip cup98lrn.txt.gz
    python acs.py --var 'MV'
    python acs.py --var 'NOC'
    python acs.py --var 'MV+NOC'
    python acs.py --var 'NOC+MV'
    python kddcup98.py
    python synthetic.py

********************************************************************************

Copyright license

This metamulti software is licensed under the (MIT-type) copyright LICENSE file
in the root directory of this source tree.
