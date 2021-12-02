#!/usr/bin/env python3

"""
Plot the subpopulation deviations for the American Community Survey of USCB.

Copyright (c) Meta Platforms, Inc. and affiliates.

This script creates a directory, "weighted", in the working directory if the
directory does not already exist, then creates a subdirectory there for one
of the four supported combinations of covariates used for conditioning: "MV",
"NOC", "MV+NOC", or "NOC+MV", where "MV" refers to "time since last move",
"NOC" refers to "number of own children", "MV+NOC" refers to both "time since
last move" and "number of own children" (in that order), and "NOC+MV" refers
to "number of own children" and "time since last move", in that order. In all
cases, there is also an additional covariate appended, namely the log of the
adjusted household personal income. The command line flag "--var" specifies
which of the four possibilities to use, defaulting to "MV" if not specified.
The script fills each subdirectory of the main directory, "weighted", with
subdirectories corresponding to counties in California, comparing each county
to all counties in California put together. The script fills each of these
subdirectories, "County_of_[county]-[regressand]" (where [county] is the name
of the county and [regressand] is the variate used for the observed reponses),
with 8 or 10 files (only 8 for "MV+NOC" or "NOC+MV"):
1. metrics.txt -- metrics about the plots
2. cumulative.pdf -- plot of cumulative differences between the county & state
3. equiscores10.pdf -- reliability diagram of the county & state with 10 bins
                       (equispaced in scores)
4. equiscores20.pdf -- reliability diagram of the county & state with 20 bins
                       (equispaced in scores)
5. equiscores100.pdf -- reliability diagram of the county & state with 100 bins
                        (equispaced in scores)
6. equierrs10.pdf -- reliability diagram of the county & state with 10 bins
                     (the error bar is about the same for every bin)
7. equierrs20.pdf -- reliability diagram of the county & state with 20 bins
                     (the error bar is about the same for every bin)
8. equierrs100.pdf -- reliability diagram of the county & state with 100 bins
                      (the error bar is about the same for every bin)
9. inputs.pdf -- PDF scatterplot of the covariates used as controls,
                 overlaying the subpopulation on the full population;
                 shading corresponds to the arc length along the Hilbert curve
10. inputs.jpg -- compressed scatterplot of the covariates used as controls,
                  overlaying the subpopulation on the full population;
                  shading corresponds to the arc length along the Hilbert curve
The script also creates two files, "inputs.pdf" and "inputs.jpg", in the
subdirectories "MV" and "NOC". These files scatterplot the covariates used as
controls, without overlaying any subpopulation over the full population; the
shading in the plots corresponds to the arc length along the Hilbert curve.
The data comes from the American Community Survey of the U.S. Census Bureau,
specifically the household data from the state of California and its counties.
The results/responses are given by the variates specified in the list "exs"
defined below (together with the value of the variate to be considered
"success" in the sense of Bernoulli trials, or else the nonnegative integer
count for the variate, counting people, for instance).

This source code is licensed under the MIT license found in the LICENSE file in
the root directory of this source tree.
"""

import argparse
import math
import numpy as np
import os
import subprocess

from hilbertcurve.hilbertcurve import HilbertCurve
from subpop_weighted import equiscores, equierrs, cumulative

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


# Specify which counties and variates to process, as well as the coded value
# of interest for each variate (or None if the values of interest are
# nonnegative integer counts).
exs = [
    {'county': 'Humboldt', 'var': 'LNGI', 'val': 2},
    {'county': 'Los Angeles', 'var': 'NP', 'val': None},
    {'county': 'Napa', 'var': 'SATELLITE', 'val': 1},
    {'county': 'Orange', 'var': 'HISPEED', 'val': 1},
    {'county': 'San Joaquin', 'var': 'NRC', 'val': None},
    {'county': 'Stanislaus', 'var': 'NRC', 'val': None},
]

# Specify the name of the file of comma-separated values
# for the household data in the American Community Survey.
filename = 'psam_h06.csv'

# Parse the command-line arguments (if any).
parser = argparse.ArgumentParser()
parser.add_argument(
    '--var', default='MV', choices=['MV', 'NOC', 'MV+NOC', 'NOC+MV'])
clargs = parser.parse_args()

# Count the number of lines in the file for filename.
lines = 0
with open(filename, 'r') as f:
    for line in f:
        lines += 1
print(f'reading and filtering all {lines} lines from {filename}....')

# Determine the number of columns in the file for filename.
with open(filename, 'r') as f:
    line = f.readline()
    num_cols = line.count(',') + 1

# Read and store all but the first two columns in the file for filename.
raw = np.zeros((lines, num_cols - 2))
with open(filename, 'r') as f:
    for line_num, line in enumerate(f):
        parsed = line.split(',')[2:]
        if line_num == 0:
            # The initial line is a header ... save its column labels.
            header = parsed.copy()
            # Eliminate the newline character at the end of the line.
            header[-1] = header[-1][:-1]
        else:
            # All but the initial line consist of data ... extract the ints.
            raw[line_num - 1, :] = np.array(
                [int(s if s != '' else -1) for s in parsed])

# Rename especially interesting columns with easier-to-understand phrases.
header[header.index('MV')] = 'duration since the last move'
header[header.index('NOC')] = 'number of householder\'s own children'

# Filter out undesirable observations -- keep only strictly positive weights,
# strictly positive household personal incomes, and strictly positive factors
# for adjusting the income.
keep = np.logical_and.reduce([
    raw[:, header.index('WGTP')] > 0,
    raw[:, header.index('HINCP')] > 0,
    raw[:, header.index('ADJINC')] > 0])
raw = raw[keep, :]
print(f'm = raw.shape[0] = {raw.shape[0]}')

# Form a dictionary of the lower- and upper-bounds on the ranges of numbers
# of the public-use microdata areas (PUMAs) for the counties in California.
puma = {
    'Alameda': (101, 110),
    'Alpine, Amador, Calaveras, Inyo, Mariposa, Mono and Tuolumne': (300, 300),
    'Butte': (701, 702),
    'Colusa, Glenn, Tehama and Trinity': (1100, 1100),
    'Contra Costa': (1301, 1309),
    'Del Norte, Lassen, Modoc, Plumas and Siskiyou': (1500, 1500),
    'El Dorado': (1700, 1700),
    'Fresno': (1901, 1907),
    'Humboldt': (2300, 2300),
    'Imperial': (2500, 2500),
    'Kern': (2901, 2905),
    'Kings': (3100, 3100),
    'Lake and Mendocino': (3300, 3300),
    'Los Angeles': (3701, 3769),
    'Madera': (3900, 3900),
    'Marin': (4101, 4102),
    'Merced': (4701, 4702),
    'Monterey': (5301, 5303),
    'Napa': (5500, 5500),
    'Nevada and Sierra': (5700, 5700),
    'Orange': (5901, 5918),
    'Placer': (6101, 6103),
    'Riverside': (6501, 6515),
    'Sacramento': (6701, 6712),
    'San Bernardino': (7101, 7115),
    'San Diego': (7301, 7322),
    'San Francisco': (7501, 7507),
    'San Joaquin': (7701, 7704),
    'San Luis Obispo': (7901, 7902),
    'San Mateo': (8101, 8106),
    'Santa Barbara': (8301, 8303),
    'Santa Clara': (8501, 8514),
    'Santa Cruz': (8701, 8702),
    'Shasta': (8900, 8900),
    'Solano': (9501, 9503),
    'Sonoma': (9701, 9703),
    'Stanislaus': (9901, 9904),
    'Sutter and Yuba': (10100, 10100),
    'Tulare': (10701, 10703),
    'Ventura': (11101, 11106),
    'Yolo': (11300, 11300),
}

# Read the weights.
w = raw[:, header.index('WGTP')]

# Read the input covariates.
# Adjust the household personal income by the relevant factor.
var0 = '$\\log_{10}$ of the adjusted household personal income'
s0 = raw[:, header.index('HINCP')] * raw[:, header.index('ADJINC')] / 1e6
# Convert the adjusted incomes to a log (base-10) scale.
s0 = np.log(s0) / math.log(10)
# Dither in order to ensure the uniqueness of the scores.
np.random.seed(seed=3820497)
s0 = s0 * (np.ones(s0.shape) + np.random.normal(size=s0.shape) * 1e-8)
# Consider the time until the last move for var 'MV'
# or the number of the household's own children for var 'NOC'.
if clargs.var == 'MV+NOC':
    var1 = 'duration since the last move'
    var2 = 'number of householder\'s own children'
    s1 = raw[:, header.index(var1)].astype(np.float64)
    s2 = raw[:, header.index(var2)].astype(np.float64)
    s2 = np.clip(s2, 0, 8)
    t = np.vstack((s0, s1, s2)).T
elif clargs.var == 'NOC+MV':
    var1 = 'number of householder\'s own children'
    var2 = 'duration since the last move'
    s1 = raw[:, header.index(var1)].astype(np.float64)
    s1 = np.clip(s1, 0, 8)
    s2 = raw[:, header.index(var2)].astype(np.float64)
    t = np.vstack((s0, s1, s2)).T
else:
    if clargs.var == 'MV':
        var1 = 'duration since the last move'
    elif clargs.var == 'NOC':
        var1 = 'number of householder\'s own children'
    else:
        raise NotImplementedError(
            clargs.var + ' is not an implemented option.')
    var2 = None
    s1 = raw[:, header.index(var1)].astype(np.float64)
    if var1 == 'number of householder\'s own children':
        s1 = np.clip(s1, 0, 8)
    t = np.vstack((s0, s1)).T

# Proprocess and order the inputs.
# Set the number of covariates.
p = t.shape[1]
# Set the number of bits in the discretization (mantissa).
precision = 64
# Determine the data type from precision.
if precision == 8:
    dtype = np.uint8
elif precision == 16:
    dtype = np.uint16
elif precision == 32:
    dtype = np.uint32
elif precision == 64:
    dtype = np.uint64
else:
    raise TypeError(f'There is no support for precision = {precision}.')
# Normalize and round the inputs.
it = t.copy()
for k in range(p):
    it[:, k] /= np.max(it[:, k])
it = np.rint((2**precision - 1) * it.astype(np.longdouble)).astype(dtype=dtype)
# Perform the Hilbert mapping from p dimensions to one dimension.
hc = HilbertCurve(precision, p)
ints = hc.distances_from_points(it)
assert np.unique(ints).size == it.shape[0]
# Sort according to the scores.
perm = np.argsort(ints)
t = t[perm, :]
# Construct scores for plotting.
imin = np.min(ints)
imax = np.max(ints)
s = (np.sort(ints) - imin) / (imax - imin)
# Ensure uniqueness even after roundoff errors.
eps = np.finfo(np.float64).eps
s = s + np.arange(0, s.size * eps, eps)
s = s.astype(np.float64)

# Create directories as needed.
dir = 'weighted'
try:
    os.mkdir(dir)
except FileExistsError:
    pass
dir = 'weighted/' + clargs.var
try:
    os.mkdir(dir)
except FileExistsError:
    pass

if var2 is None:
    # Plot all inputs from the full population.
    procs = []
    plt.figure()
    plt.xlabel(var0)
    plt.ylabel(var1)
    colors = .2 + .6 * np.vstack((s, s, s)).T
    plt.scatter(t[:, 0], t[:, 1], s=5, c=colors, marker='D', linewidths=0)
    filename = dir + '/' + 'inputs'
    plt.savefig(filename + '.pdf', bbox_inches='tight')
    args = [
        'convert', '-density', '600', filename + '.pdf', filename + '.jpg']
    procs.append(subprocess.Popen(args))

# Process the examples.
for ex in exs:

    # Form the results.
    np.random.seed(seed=3820497)
    # Read the result (raw integer count if the specified value is None,
    # Bernoulli indicator of success otherwise).
    if ex['val'] is None:
        r = raw[:, header.index(ex['var'])]
    else:
        r = raw[:, header.index(ex['var'])] == ex['val']
    # Sort according to the scores.
    r = r[perm]

    # Set a directory for the county (creating the directory if necessary).
    dir = 'weighted/' + clargs.var + '/County_of_'
    dir += ex['county'].replace(' ', '_').replace(',', '')
    dir += '-'
    dir += ex['var']
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass
    dir += '/'
    print(f'./{dir} is under construction....')

    # Identify the indices of the subset corresponding to the county.
    slice = raw[perm, header.index('PUMA')]
    inds = slice >= (puma[ex['county']][0] * np.ones(raw.shape[0]))
    inds = inds & (slice <= (puma[ex['county']][1] * np.ones(raw.shape[0])))
    inds = np.nonzero(inds)[0]
    inds = np.unique(inds)

    if var2 is None:
        # Plot the inputs.
        plt.figure()
        plt.xlabel(var0)
        plt.ylabel(var1)
        colors = .2 + .6 * np.vstack((s, s, s)).T
        plt.scatter(
            t[inds, 0], t[inds, 1], s=5, c=colors[inds, :], marker='D',
            linewidths=0)
        plt.scatter(
            t[:, 0], t[:, 1], s=.5, c=1 - colors, marker='o', linewidths=0)
        filename = dir + 'inputs'
        plt.savefig(filename + '.pdf', bbox_inches='tight')
        args = [
            'convert', '-density', '600', filename + '.pdf',
            filename + '.jpg']
        procs.append(subprocess.Popen(args))

    # Plot reliability diagrams and the cumulative graph.
    nin = [10, 20, 100]
    nout = {}
    for nbins in nin:
        filename = dir + 'equiscores' + str(nbins) + '.pdf'
        equiscores(r, s, inds, nbins, filename, weights=w, left=0)
        filename = dir + 'equierrs' + str(nbins) + '.pdf'
        nout[str(nbins)] = equierrs(r, s, inds, nbins, filename, weights=w)
    majorticks = 10
    minorticks = 300
    filename = dir + 'cumulative.pdf'
    kuiper, kolmogorov_smirnov, lenscale = cumulative(
        r, s, inds, majorticks, minorticks, ex['val'] is not None,
        filename=filename, weights=w)
    # Save metrics in a text file.
    filename = dir + 'metrics.txt'
    with open(filename, 'w') as f:
        f.write('m:\n')
        f.write(f'{len(s)}\n')
        f.write('n:\n')
        f.write(f'{len(inds)}\n')
        f.write('lenscale:\n')
        f.write(f'{lenscale}\n')
        for nbins in nin:
            f.write("nout['" + str(nbins) + "']:\n")
            f.write(f'{nout[str(nbins)][0]}\n')
            f.write(f'{nout[str(nbins)][1]}\n')
        f.write('Kuiper:\n')
        f.write(f'{kuiper:.4}\n')
        f.write('Kolmogorov-Smirnov:\n')
        f.write(f'{kolmogorov_smirnov:.4}\n')
        f.write('Kuiper / lenscale:\n')
        f.write(f'{(kuiper / lenscale):.4}\n')
        f.write('Kolmogorov-Smirnov / lenscale:\n')
        f.write(f'{(kolmogorov_smirnov / lenscale):.4}\n')
if var2 is None:
    print('waiting for conversion from pdf to jpg to finish....')
    for iproc, proc in enumerate(procs):
        proc.wait()
        print(f'{iproc + 1} of {len(procs)} conversions are done....')
