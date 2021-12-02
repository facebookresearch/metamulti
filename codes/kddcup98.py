#!/usr/bin/env python3

"""
Plot the subpopulation deviations for the veterans org's direct mailing.

Copyright (c) Meta Platforms, Inc. and affiliates.

This script creates a directory, "unweighted", in the working directory if the
directory does not already exist, then creates four subdirectories there,
"folding" comparing the subpop. sent only folding cards to the full population,
"normal" comparing the subpop. sent only normal cards to the full population,
"both" comparing the subpop. sent both kinds of cards to the full population,
and "folding_normal" comparing directly the subpop. sent only folding cards
to the subpop. sent only normal cards. Each of these subdirectories in turn
contains five subdirectories, "02", "12", "20", "21", and "012", corresponding
to controlling for the correspondingly numbered covariates in the specified
order, where "0" labels the covariate for the normalized age of the recipient,
"1" labels the normalized fraction married in the Census block where the
recipient lives, and "2" labels the normalized average household income in the
Census block where the recipient lives. The age and fraction married are both
integer-valued in the data set -- the fraction married is given by the nearest
integer-valued percentage. The script fills each of these subdirectories with
8 or 12 files (only 8 for "012"):
1. metrics.txt -- metrics about the plots
2. cumulative.pdf -- plot of cumulative differences
3. equiscores10.pdf -- reliability diagram with 10 bins, equispaced in scores
4. equiscores20.pdf -- reliability diagram with 20 bins, equispaced in scores
5. equiscores100.pdf -- reliability diagram with 100 bins, equispaced in scores
6. equierrs10.pdf -- reliability diagram with 10 bins, where the error bar is
                     about the same for every bin
7. equierrs20.pdf -- reliability diagram with 20 bins, where the error bar is
                     about the same for every bin
8. equierrs100.pdf -- reliability diagram with 100 bins, where the error bar is
                      about the same for every bin
9. input.pdf -- PDF scatterplot of the covariates used as controls;
                shading corresponds to arc length along the Hilbert curve
10. input.jpg -- compressed scatterplot of the covariates used as controls;
                 shading corresponds to arc length along the Hilbert curve
11. inputs.pdf -- PDF scatterplot of the covariates used as controls,
                  overlaying the subpopulation on the full population;
                  shading corresponds to arc length along the Hilbert curve
12. inputs.jpg -- compressed scatterplot of the covariates used as controls,
                  overlaying the subpopulation on the full population;
                  shading corresponds to arc length along the Hilbert curve
In the subdirectory, "folding_normal", the latter four files are the following:
9. blue_on_red.pdf -- PDF scatterplot of the covariates used as controls,
                      with one subpop. in blue overlaid on the other in red;
                      shading corresponds to arc length along the Hilbert curve
10. blue_on_red.jpg -- compressed scatterplot of the covariates controlled for,
                       with one subpop. in blue overlaid on the other in red;
                       shading corresponds to arc length on the Hilbert curve
11. red_on_blue.pdf -- PDF scatterplot of the covariates used as controls,
                       with one subpop. in red overlaid on the other in blue;
                       shading corresponds to arc length on the Hilbert curve
12. red_on_blue.jpg -- compressed scatterplot of the covariates controlled for,
                       with one subpop. in red overlaid on the other in blue;
                       shading corresponds to arc length on the Hilbert curve
The script also creates 12 files in the directory, "unweighted", namely
"01.pdf", "01.jpg", "02.pdf", "02.jpg", "10.pdf", "10.jpg",
"12.pdf", "12.jpg", "20.pdf", "20.jpg", "21.pdf", "21.jpg".
These files scatterplot the associated covariates against each other
(in the order given by the name of the file).
The data comes from a direct mailing campaign of a national veterans org.

This source code is licensed under the MIT license found in the LICENSE file in
the root directory of this source tree.
"""

import argparse
import math
import numpy as np
import os
import subprocess
from numpy.random import default_rng

from hilbertcurve.hilbertcurve import HilbertCurve
import subpop
import disjoint

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


# Specify the name of the file of comma-separated values
# for the training data about the direct-mail marketing campaign.
filename = 'cup98lrn.txt'

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

# Read and store the data in the file named filename.
with open(filename, 'r') as f:
    for line_num, line in enumerate(f):
        # Eliminate the newline character at the end of line with line[:-1].
        parsed = line[:-1].split(',')
        if line_num == 0:
            # The initial line is a header ... save its column labels.
            header = parsed.copy()
        else:
            # All but the initial line consist of data ... save the data.
            data = []
            for s in parsed:
                if s == '':
                    # Label missing data with None.
                    data.append(None)
                elif s.isdecimal():
                    # Convert a whole number to an int.
                    data.append(int(s))
                else:
                    try:
                        # Convert a floating-point number to a float.
                        data.append(float(s))
                    except ValueError:
                        if(s[-1] == '-'):
                            # The zipcode includes an extra hyphen ... omit it.
                            data.append(int(s[:-1]))
                        else:
                            # Save the entire string.
                            data.append(s)
            # Initialize raw if line_num == 1.
            if line_num == 1:
                raw = []
            # Discard entries whose ages, marital fractions,
            # or average household incomes are missing.
            if data[header.index('AGE')] not in [None, 0]:
                if data[header.index('MARR1')] not in [None, 0]:
                    if data[header.index('IC3')] not in [None, 0]:
                        raw.append(data)

# Rename especially interesting columns with easier-to-understand phrases.
header[header.index('AGE')] = 'normalized age'
header[header.index('MARR1')] = 'normalized fraction married'
header[header.index('IC3')] = 'normalized average household income'

# Tabulate the total numbers of mailings of each type.
count = [0, 0]
for k in range(2, 25):
    count.append(0)
    for rawent in raw:
        if rawent[header.index('ADATE_' + str(k))] is not None:
            count[k] += 1

# Tabulate how many got folding cards in 1994 but not normal cards,
# how many got normal cards but not folding cards, and how many got both.
folding = 0
foldingr = 0
normal = 0
normalr = 0
both = 0
bothr = 0
neither = 0
for rawent in raw:
    if rawent[header.index('ADATE_23')] is not None:
        if rawent[header.index('ADATE_24')] is None:
            folding += 1
            if rawent[header.index('RDATE_23')] is not None:
                foldingr += 1
    if rawent[header.index('ADATE_24')] is not None:
        if rawent[header.index('ADATE_23')] is None:
            normal += 1
            if rawent[header.index('RDATE_24')] is not None:
                normalr += 1
    if rawent[header.index('ADATE_24')] is not None:
        if rawent[header.index('ADATE_23')] is not None:
            both += 1
            if rawent[header.index('RDATE_23')] is not None or \
                    rawent[header.index('RDATE_24')] is not None:
                bothr += 1
    if rawent[header.index('ADATE_24')] is None:
        if rawent[header.index('ADATE_23')] is None:
            neither += 1
print()
print('numbers of mailings:')
print(f'normal = {normal}')
print(f'folding = {folding}')
print(f'both = {both}')
print(f'neither = {neither}')
print()
print('numbers of responses:')
print(f'normalr = {normalr}')
print(f'foldingr = {foldingr}')
print(f'bothr = {bothr}')

# Retain only the full population of those who received mailings in 1994.
newraw = []
for rawent in raw:
    if rawent[header.index('ADATE_23')] is not None:
        newraw.append(rawent)
    elif rawent[header.index('ADATE_24')] is not None:
        newraw.append(rawent)
raw = newraw
print()
print(f'normal + folding + both = len(raw) = {len(raw)}')

# Set up the random number generator.
rng = default_rng(seed=543216789)

# Tabulate all covariates of possible interest.
vars = [
    'normalized age', 'normalized fraction married',
    'normalized average household income']
covars = np.zeros((len(raw), len(vars)))
for k in range(len(raw)):
    for j in range(len(vars)):
        covars[k, j] = raw[k][header.index(vars[j])]
        if vars[j] == 'normalized average household income':
            covars[k, j] *= 1 + 1e-8 * rng.standard_normal()

# Store processes for converting from pdf to jpeg in procs.
procs = []

# Normalize every covariate.
cmin = np.min(covars, axis=0)
cmax = np.max(covars, axis=0)
covars = (covars - cmin) / (cmax - cmin)

# Create a directory, "unweighted", if none exists.
dir = 'unweighted'
try:
    os.mkdir(dir)
except FileExistsError:
    pass
dir += '/'

# Scatterplot pairs of covariates.
for j in range(covars.shape[1]):
    for k in list(range(0, j)) + list(range(j + 1, covars.shape[1])):
        plt.figure()
        plt.scatter(covars[:, j], covars[:, k], s=.1, c='k')
        plt.xlabel(vars[j])
        plt.ylabel(vars[k])
        plt.tight_layout()
        # Save the figure to disk and queue up a process for converting
        # from pdf to jpg.
        filename = dir + str(j) + str(k)
        filepdf = filename + '.pdf'
        filejpg = filename + '.jpg'
        plt.savefig(filepdf, bbox_inches='tight')
        plt.close()
        args = ['convert', '-density', '600', filepdf, filejpg]
        procs.append(subprocess.Popen(args))

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

# Specify tuples of covariates for controls.
controls = [(0, 2), (2, 0), (1, 2), (2, 1), (0, 1, 2)]

# Construct plots for each set of controls.
for control in controls:
    p = len(control)
    t = covars[:, control]
    # Round the inputs.
    it = np.rint((2**precision - 1) * t.astype(np.longdouble)).astype(
        dtype=dtype)
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
    # Flag for each individual whether any response happened in 1994.
    r = []
    for k in range(len(raw)):
        rawent = raw[perm[k]]
        if rawent[header.index('RDATE_23')] is not None or \
                rawent[header.index('RDATE_24')] is not None:
            r.append(1)
        else:
            r.append(0)
    r = np.array(r)

    # Consider a couple different subpopulations to compare with the full pop.
    for sub in ['both', 'folding', 'normal']:
        if sub == 'both':
            # Consider the subpopulation of those who got both normal
            # and folding mailings in 1994.
            inds = []
            for k in range(len(raw)):
                rawent = raw[perm[k]]
                if rawent[header.index('ADATE_24')] is not None:
                    if rawent[header.index('ADATE_23')] is not None:
                        inds.append(k)
            inds = np.array(inds, dtype=np.uint64)
        elif sub == 'folding':
            # Consider the subpopulation of those who got only folding mailings
            # in 1994.
            inds = []
            for k in range(len(raw)):
                rawent = raw[perm[k]]
                if rawent[header.index('ADATE_23')] is not None:
                    if rawent[header.index('ADATE_24')] is None:
                        inds.append(k)
            inds = np.array(inds, dtype=np.uint64)
        elif sub == 'normal':
            # Consider the subpopulation of those who got only normal mailings
            # in 1994.
            inds = []
            for k in range(len(raw)):
                rawent = raw[perm[k]]
                if rawent[header.index('ADATE_24')] is not None:
                    if rawent[header.index('ADATE_23')] is None:
                        inds.append(k)
            inds = np.array(inds, dtype=np.uint64)
        else:
            raise NotImplementedError(
                'There is no subset known as "' + sub + '."')
        # Set a directory for the controls.
        dir = 'unweighted/' + sub
        try:
            os.mkdir(dir)
        except FileExistsError:
            pass
        dir += '/'
        for k in control:
            dir += str(k)
        try:
            os.mkdir(dir)
        except FileExistsError:
            pass
        dir += '/'
        print(f'./{dir} is under construction....')

        if len(control) == 2:

            # Plot the inputs without the subpopulation highlighted.
            plt.figure()
            plt.xlabel(vars[control[0]])
            plt.ylabel(vars[control[1]])
            colors = .2 + .6 * np.vstack((s, s, s)).T
            plt.scatter(
                t[:, 0], t[:, 1], s=5, c=colors, marker='D', linewidths=0)
            # Save the figure to disk and queue up a process for converting
            # from pdf to jpg.
            filename = dir + 'input'
            filepdf = filename + '.pdf'
            filejpg = filename + '.jpg'
            plt.savefig(filepdf, bbox_inches='tight')
            plt.close()
            args = ['convert', '-density', '600', filepdf, filejpg]
            procs.append(subprocess.Popen(args))

            # Plot the inputs with the subpopulation highlighted.
            plt.figure()
            plt.xlabel(vars[control[0]])
            plt.ylabel(vars[control[1]])
            colors = .2 + .6 * np.vstack((s, s, s)).T
            plt.scatter(
                t[inds, 0], t[inds, 1], s=5, c=colors[inds, :], marker='D',
                linewidths=0)
            plt.scatter(
                t[:, 0], t[:, 1], s=.5, c=1 - colors, marker='o', linewidths=0)
            # Save the figure to disk and queue up a process for converting
            # from pdf to jpg.
            filename = dir + 'inputs'
            filepdf = filename + '.pdf'
            filejpg = filename + '.jpg'
            plt.savefig(filepdf, bbox_inches='tight')
            plt.close()
            args = ['convert', '-density', '600', filepdf, filejpg]
            procs.append(subprocess.Popen(args))

        # Plot reliability diagrams and the cumulative graph.
        nin = [10, 20, 100]
        for nbins in nin:
            filename = dir + 'equiscores' + str(nbins) + '.pdf'
            subpop.equiscore(r, s, inds, nbins, filename)
            filename = dir + 'equierrs' + str(nbins) + '.pdf'
            subpop.equisamps(r, s, inds, nbins, filename)
        majorticks = 10
        minorticks = 300
        filename = dir + 'cumulative.pdf'
        kuiper, kolmogorov_smirnov, lenscale = subpop.cumulative(
            r, s, inds, majorticks, minorticks, filename=filename)
        # Save metrics in a text file.
        filename = dir + 'metrics.txt'
        with open(filename, 'w') as f:
            f.write('m:\n')
            f.write(f'{len(s)}\n')
            f.write('n:\n')
            f.write(f'{len(inds)}\n')
            f.write('lenscale:\n')
            f.write(f'{lenscale}\n')
            f.write('Kuiper:\n')
            f.write(f'{kuiper:.4}\n')
            f.write('Kolmogorov-Smirnov:\n')
            f.write(f'{kolmogorov_smirnov:.4}\n')
            f.write('Kuiper / lenscale:\n')
            f.write(f'{(kuiper / lenscale):.4}\n')
            f.write('Kolmogorov-Smirnov / lenscale:\n')
            f.write(f'{(kolmogorov_smirnov / lenscale):.4}\n')

    # Compare two subpopulations directly.
    i0 = []
    s0 = []
    r0 = []
    i1 = []
    s1 = []
    r1 = []
    for k in range(len(raw)):
        rawent = raw[perm[k]]
        # Consider the subpopulation of those who got only folding mailings
        # in 1994.
        if rawent[header.index('ADATE_23')] is not None:
            if rawent[header.index('ADATE_24')] is None:
                i0.append(perm[k])
                s0.append(s[k])
                if rawent[header.index('RDATE_23')] is None:
                    r0.append(0)
                else:
                    r0.append(1)
        # Consider the subpopulation of those who got only normal mailings
        # in 1994.
        if rawent[header.index('ADATE_24')] is not None:
            if rawent[header.index('ADATE_23')] is None:
                i1.append(perm[k])
                s1.append(s[k])
                if rawent[header.index('RDATE_24')] is None:
                    r1.append(0)
                else:
                    r1.append(1)
    s01 = [np.array(s0), np.array(s1)]
    r01 = [np.array(r0), np.array(r1)]
    # Set a directory for the controls.
    dir = 'unweighted/folding_normal'
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass
    dir += '/'
    for k in control:
        dir += str(k)
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass
    dir += '/'
    print(f'./{dir} is under construction....')

    if len(control) == 2:

        # Plot the inputs with the subpopulations colored red and blue.
        plt.figure()
        plt.xlabel(vars[control[0]])
        plt.ylabel(vars[control[1]])
        o0 = np.ones((len(s0)))
        blue = .8 * np.vstack((np.array(s0), np.array(s0), o0)).T
        blue += .2 * np.array([0, 0, 1])
        plt.scatter(
            covars[i0, control[0]], covars[i0, control[1]],
            s=2, c=blue, marker='o', linewidths=0)
        o1 = np.ones((len(s1)))
        red = .8 * np.vstack((o1, np.array(s1), np.array(s1))).T
        red += .2 * np.array([1, 0, 0])
        plt.scatter(
            covars[i1, control[0]], covars[i1, control[1]],
            s=2, c=red, marker='o', linewidths=0)
        # Save the figure to disk and queue up a process for converting
        # from pdf to jpg.
        filename = dir + 'red_on_blue'
        filepdf = filename + '.pdf'
        filejpg = filename + '.jpg'
        plt.savefig(filepdf, bbox_inches='tight')
        plt.close()
        args = ['convert', '-density', '600', filepdf, filejpg]
        procs.append(subprocess.Popen(args))

        # Plot the inputs with the subpopulations colored red and blue,
        # in the opposite order.
        plt.figure()
        plt.xlabel(vars[control[0]])
        plt.ylabel(vars[control[1]])
        o1 = np.ones((len(s1)))
        red = .8 * np.vstack((o1, np.array(s1), np.array(s1))).T
        red += .2 * np.array([1, 0, 0])
        plt.scatter(
            covars[i1, control[0]], covars[i1, control[1]],
            s=2, c=red, marker='o', linewidths=0)
        o0 = np.ones((len(s0)))
        blue = .8 * np.vstack((np.array(s0), np.array(s0), o0)).T
        blue += .2 * np.array([0, 0, 1])
        plt.scatter(
            covars[i0, control[0]], covars[i0, control[1]],
            s=2, c=blue, marker='o', linewidths=0)
        # Save the figure to disk and queue up a process for converting
        # from pdf to jpg.
        filename = dir + 'blue_on_red'
        filepdf = filename + '.pdf'
        filejpg = filename + '.jpg'
        plt.savefig(filepdf, bbox_inches='tight')
        plt.close()
        args = ['convert', '-density', '600', filepdf, filejpg]
        procs.append(subprocess.Popen(args))

    # Plot reliability diagrams and the cumulative graph.
    filename = dir + 'cumulative.pdf'
    majorticks = 10
    minorticks = 300
    kuiper, kolmogorov_smirnov, lenscale, lencums = disjoint.cumulative(
        r01, s01, majorticks, minorticks, False, filename)
    filename = dir + 'metrics.txt'
    with open(filename, 'w') as f:
        f.write('n:\n')
        f.write(f'{lencums}\n')
        f.write('len(s0):\n')
        f.write(f'{len(s0)}\n')
        f.write('len(s1):\n')
        f.write(f'{len(s1)}\n')
        f.write('lenscale:\n')
        f.write(f'{lenscale}\n')
        f.write('Kuiper:\n')
        f.write(f'{kuiper:.4}\n')
        f.write('Kolmogorov-Smirnov:\n')
        f.write(f'{kolmogorov_smirnov:.4}\n')
        f.write('Kuiper / lenscale:\n')
        f.write(f'{(kuiper / lenscale):.4}\n')
        f.write('Kolmogorov-Smirnov / lenscale:\n')
        f.write(f'{(kolmogorov_smirnov / lenscale):.4}\n')
    nin = [10, 20, 100]
    for nbins in nin:
        filename = dir + 'equiscores' + str(nbins) + '.pdf'
        disjoint.equiscore(r01, s01, nbins, filename)
        filename = dir + 'equierrs' + str(nbins) + '.pdf'
        disjoint.equisamps(r01, s01, nbins, filename)
print()
print('waiting for conversion from pdf to jpg to finish....')
for iproc, proc in enumerate(procs):
    proc.wait()
    print(f'{iproc + 1} of {len(procs)} conversions are done....')
