#!/usr/bin/env python3

"""
Plot the subpopulation deviations for the American Community Survey of USCB.

Copyright (c) Meta Platforms, Inc. and affiliates.

This script offers command-line options "--interactive" and "--no-interactive"
for plotting interactively and non-interactively, respectively. Interactive is
the default. The interactive setting does the same as the non-interactive, but
without saving to disk any plots, and without plotting any classical
reliability diagrams or any scatterplots of the covariates used as controls.
When run non-interactively (i.e., with command-line option "--no-interactive"),
this script creates a directory, "weighted", in the working directory if the
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

import matplotlib
from matplotlib.backend_bases import MouseButton
from matplotlib.ticker import FixedFormatter
from matplotlib import get_backend
default_backend = get_backend()
matplotlib.use('agg')
import matplotlib.pyplot as plt

from hilbertcurve.hilbertcurve import HilbertCurve
from subpop_weighted import equiscores, equierrs, cumulative


def icumulative(r, s, t, u, covariates, inds, majorticks, minorticks,
                bernoulli=True,
                title='subpop. deviation is the slope as a function of $A_k$',
                fraction=1, weights=None, window='Figure'):
    """
    Cumulative difference between observations from a subpop. & the full pop.

    Plots the difference between the normalized cumulative weighted sums of r
    for the subpopulation indices inds and the normalized cumulative
    weighted sums of r from the full population interpolated to the subpop.
    indices, with majorticks major ticks and minorticks minor ticks on the
    lower axis, labeling the major ticks with the corresponding values from s.
    This is an interactive version of subpop_weighted.cumulative (probably not
    suitable in general for all data sets, however).

    Parameters
    ----------
    r : array_like
        random outcomes
    s : array_like
        scores (must be unique and in strictly increasing order)
    t : array_like
        normalized values of the covariates
    u : array_like
        unnormalized values of the covariates
    covariates : array_like
        strings labeling the covariates
    inds : array_like
        indices of the subset within s that defines the subpopulation
        (must be unique and in strictly increasing order)
    majorticks : int
        number of major ticks on each of the horizontal axes
    minorticks : int
        number of minor ticks on the lower axis
    bernoulli : bool, optional
        set to True (the default) for Bernoulli variates; set to False
        to use empirical estimates of the variance rather than the formula
        p(1-p) for a Bernoulli variate whose mean is p
    title : string, optional
        title of the plot
    fraction : float, optional
        proportion of the full horizontal axis to display
    weights : array_like, optional
        weights of the observations
        (the default None results in equal weighting)
    window : string, optional
        title of the window displayed in the title bar

    Returns
    -------
    None
    """

    def histcounts(nbins, a):
        # Counts the number of entries of a
        # falling into each of nbins equispaced bins.
        j = 0
        nbin = np.zeros(nbins, dtype=np.int64)
        for k in range(len(a)):
            if a[k] > a[-1] * (j + 1) / nbins:
                j += 1
            if j == nbins:
                break
            nbin[j] += 1
        return nbin

    def aggregate(r, s, inds, w):
        # Determines the total weight and variance of the nonzero entries of r
        # in a bin around each entry of s corresponding to the subset of s
        # specified by the indices inds. The bin ranges from halfway
        # to the nearest entry of s from inds on the left to halfway
        # to the nearest entry of s from inds on the right.
        ss = s[inds]
        q = np.insert(np.append(ss, [1e20]), 0, [-1e20])
        t = np.asarray([(q[k] + q[k + 1]) / 2 for k in range(len(q) - 1)])
        rc = np.zeros((len(inds)))
        rc2 = np.zeros((len(inds)))
        sc = np.zeros((len(inds)))
        j = 0
        for k in range(len(s)):
            if s[k] > t[j + 1]:
                j += 1
                if j == len(inds):
                    break
            if s[k] >= t[0]:
                sc[j] += w[k]
                rc[j] += w[k] * r[k]
                rc2[j] += w[k] * r[k]**2
        means = rc / sc
        return means, rc2 / sc - means**2

    def on_move(event):
        if event.inaxes:
            ax = event.inaxes
            k = round(event.xdata * (len(inds) - 1))
            toptxt = ''
            bottomtxt = ''
            for j in range(len(covariates)):
                toptxt += covariates[j]
                if(np.allclose(
                        np.round(u[inds[k], j]), u[inds[k], j], rtol=1e-5)):
                    toptxt += ' = {}'.format(round(u[inds[k], j]))
                else:
                    toptxt += ' = {:.2f}'.format(u[inds[k], j])
                toptxt += '\n'
                bottomtxt += 'normalized ' + covariates[j]
                bottomtxt += ' = {:.2f}'.format(t[inds[k], j])
                bottomtxt += '\n'
            toptxt += '$S_{i_k}$' + ' = {:.2f}'.format(s[inds[k]])
            bottomtxt += '$S_{i_k}$' + ' = {:.2f}'.format(s[inds[k]])
            toptext.set_text(toptxt)
            bottomtext.set_text(bottomtxt)
            plt.draw()

    def on_click(event):
        if event.button is MouseButton.LEFT:
            plt.disconnect(binding_id)
            plt.close()

    assert all(s[k] < s[k + 1] for k in range(len(s) - 1))
    assert all(inds[k] < inds[k + 1] for k in range(len(inds) - 1))
    # Determine the weighting scheme.
    if weights is None:
        w = np.ones((len(s)))
    else:
        w = weights.copy()
    assert np.all(w > 0)
    w /= w.sum()
    # Create the figure.
    plt.figure(window)
    ax = plt.axes()
    # Aggregate r according to inds, s, and w.
    rt, rtvar = aggregate(r, s, inds, w)
    # Subsample r, s, and w.
    rs = r[inds]
    ss = s[inds]
    ws = w[inds]
    ws /= ws[:int(len(ws) * fraction)].sum()
    # Accumulate the weighted rs and rt, as well as ws.
    f = np.insert(np.cumsum(ws * rs), 0, [0])
    ft = np.insert(np.cumsum(ws * rt), 0, [0])
    x = np.insert(np.cumsum(ws), 0, [0])
    # Plot the difference.
    plt.plot(
        x[:int(len(x) * fraction)], (f - ft)[:int(len(f) * fraction)], 'k')
    # Make sure the plot includes the origin.
    plt.plot(0, 'k')
    # Add an indicator of the scale of 1/sqrt(n) to the vertical axis.
    rtsub = np.insert(rt, 0, [0])[:(int(len(rt) * fraction) + 1)]
    if bernoulli:
        lenscale = np.sqrt(np.sum(ws**2 * rtsub[1:] * (1 - rtsub[1:])))
    else:
        lenscale = np.sqrt(np.sum(ws**2 * rtvar))
    plt.plot(2 * lenscale, 'k')
    plt.plot(-2 * lenscale, 'k')
    kwargs = {
        'head_length': 2 * lenscale, 'head_width': fraction / 20, 'width': 0,
        'linewidth': 0, 'length_includes_head': True, 'color': 'k'}
    plt.arrow(.1e-100, -2 * lenscale, 0, 4 * lenscale, shape='left', **kwargs)
    plt.arrow(.1e-100, 2 * lenscale, 0, -4 * lenscale, shape='right', **kwargs)
    plt.margins(x=0, y=.6)
    # Label the major ticks of the lower axis with the values of ss.
    lenxf = int(len(x) * fraction)
    sl = ['{:.2f}'.format(a) for a in
          np.insert(ss, 0, [0])[:lenxf:(lenxf // majorticks)].tolist()]
    plt.xticks(x[:lenxf:(lenxf // majorticks)], sl)
    if len(rtsub) >= 300 and minorticks >= 50:
        # Indicate the distribution of s via unlabeled minor ticks.
        plt.minorticks_on()
        ax.tick_params(which='minor', axis='x')
        ax.tick_params(which='minor', axis='y', left=False)
        ax.set_xticks(x[np.cumsum(histcounts(minorticks,
                      ss[:int((len(x) - 1) * fraction)]))], minor=True)
    # Label the axes.
    plt.xlabel('$S_{i_k}$ (the subscript on $S$ is $i_k$)')
    plt.ylabel('$F_k - \\tilde{F}_k$')
    ax2 = plt.twiny()
    plt.xlabel(
        '$k/n$ (together with minor ticks at equispaced values of $A_k$)')
    ax2.tick_params(which='minor', axis='x', top=True, direction='in', pad=-16)
    ax2.set_xticks(np.arange(1 / majorticks, 1, 1 / majorticks), minor=True)
    ks = ['{:.2f}'.format(a) for a in
          np.arange(0, 1 + 1 / majorticks, 1 / majorticks).tolist()]
    alist = (lenxf - 1) * np.arange(0, 1 + 1 / majorticks, 1 / majorticks)
    alist = alist.tolist()
    # Jitter minor ticks that overlap with major ticks lest Pyplot omit them.
    alabs = []
    for a in alist:
        multiple = x[int(a)] * majorticks
        if abs(multiple - round(multiple)) > 1e-4:
            alabs.append(x[int(a)])
        else:
            alabs.append(x[int(a)] * (1 - 1e-4))
    plt.xticks(alabs, ks)
    # Include an unbreakable space character (NBSP) as a subscript "_{ }"
    # on the numerical labels to match the baseline offset of the subscript
    # of "k" on "A_k" in order to keep all labels aligned vertically.
    ax2.xaxis.set_minor_formatter(FixedFormatter(
        [r'$A_k\!=\!{:.2f}$'.format(1 / majorticks)]
        + [r'${:.2f}'.format(k / majorticks) + r'_{ }$'
           for k in range(2, majorticks)]))
    # Title the plot.
    plt.title(title)
    # Clean up the whitespace in the plot.
    plt.tight_layout()
    # Set the locations (in the plot) of the covariate values.
    xmid = s[:(len(s) * fraction)][-1] / 2
    toptext = plt.text(
        xmid, max(2 * lenscale, np.max((f - ft)[:(len(f) * fraction)])), '',
        ha='center', va='bottom')
    bottomtext = plt.text(
        xmid, min(-2 * lenscale, np.min((f - ft)[:(len(f) * fraction)])), '',
        ha='center', va='top')
    # Set up interactivity.
    binding_id = plt.connect('motion_notify_event', on_move)
    plt.connect('button_press_event', on_click)
    # Show the plot.
    plt.show()
    plt.close()


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
parser.add_argument('--interactive', dest='interactive', action='store_true')
parser.add_argument(
    '--no-interactive', dest='interactive', action='store_false')
parser.add_argument(
    '--non-interactive', dest='interactive', action='store_false')
parser.set_defaults(interactive=True)
clargs = parser.parse_args()

# Make matplotlib interactive if clargs.interactive is True.
if clargs.interactive:
    plt.switch_backend(default_backend)

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
u = t.copy()
for k in range(p):
    t[:, k] /= np.max(t[:, k])
# Construct scores for plotting.
imin = np.min(ints)
imax = np.max(ints)
s = (np.sort(ints) - imin) / (imax - imin)
# Ensure uniqueness even after roundoff errors.
eps = np.finfo(np.float64).eps
s = s + np.arange(0, s.size * eps, eps)
s = s.astype(np.float64)

if not clargs.interactive:
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

if var2 is None and not clargs.interactive:
    # Plot all inputs from the full population.
    procs = []
    plt.figure()
    plt.xlabel(var0)
    plt.ylabel(var1)
    colors = .2 + .6 * np.vstack((s, s, s)).T
    plt.scatter(u[:, 0], u[:, 1], s=5, c=colors, marker='D', linewidths=0)
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

    if not clargs.interactive:
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

    if var2 is None and not clargs.interactive:
        # Plot the inputs.
        plt.figure()
        plt.xlabel(var0)
        plt.ylabel(var1)
        colors = .2 + .6 * np.vstack((s, s, s)).T
        plt.scatter(
            u[inds, 0], u[inds, 1], s=5, c=colors[inds, :], marker='D',
            linewidths=0)
        plt.scatter(
            u[:, 0], u[:, 1], s=.5, c=1 - colors, marker='o', linewidths=0)
        filename = dir + 'inputs'
        plt.savefig(filename + '.pdf', bbox_inches='tight')
        args = [
            'convert', '-density', '600', filename + '.pdf',
            filename + '.jpg']
        procs.append(subprocess.Popen(args))

    if clargs.interactive:
        # Plot the cumulative differences interactively.
        covariates = [var0, var1]
        if var2 is not None:
            covariates.append(var2)
        majorticks = 5
        minorticks = 300
        window = 'County: ' + ex['county'] + '; Variable: ' + ex['var']
        window += ' (click the plot to continue)'
        icumulative(r, s, t, u, covariates, inds, majorticks, minorticks,
                    ex['val'] is not None, weights=w, window=window)
    else:
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
if var2 is None and not clargs.interactive:
    print('waiting for conversion from pdf to jpg to finish....')
    for iproc, proc in enumerate(procs):
        proc.wait()
        print(f'{iproc + 1} of {len(procs)} conversions are done....')
