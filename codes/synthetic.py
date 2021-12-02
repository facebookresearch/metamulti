#!/usr/bin/env python3

"""
Plot the subpopulation deviations for a range of synthetic toy examples.

Copyright (c) Meta Platforms, Inc. and affiliates.

This script creates a directory, "unweighted", in the working directory if the
directory does not already exist, then creates many files there. The filenames
are "synth####.pdf", "synth####.txt", "reverse####.pdf", "reverse####.jpg",
"randwalk####.pdf", and "randwalk####.txt", where "####" ranges through the
powers of 2 from 0002 to 4096. Each pdf file plots the cumulative differences
between the subpopulation and the full population, controlling for the
specified number of covariates. The corresponding txt files report metrics
about the plots. The files named "reverse####.pdf" and "reverse####.txt"
condition on the covariates in the reverse order from those named
"synth####.pdf" and "synth####.txt". The files named "randwalk####.pdf" and
"randwalk####.txt" use the same distribution of responses for the subpopulation
as for the full population.

The data consists of a full population of 1,000 individual members and a
subpopulation of 100 subselected uniformly at random from the full population.
Each member of the full population consists of p independent and identically
distributed draws from the uniform distribution over the interval (0, 1),
where p is the number of covariates. We condition on all the covariates.

We generate the responses via the following procedure, which consists of only a
single stage for the files whose names begin "randwalk...", but consists of two
separate stages for the files whose names begin "synth..." or "reverse..."): we
collect together the covariates for all the members into a 1000 x p matrix x,
construct the p x 1 vector v whose entries are independent and identically
distributed draws from the standard normal distribution, and finally then apply
the Heaviside function to every entry of "centered" (= x-0.5) applied to v
(the Heaviside function is also known as the unit step function, and takes
the value 0 for negative arguments and the value 1 for positive arguments).
The result is a 1000 x 1 vector of 0s and 1s whose entries are the responses
for the corresponding members of the full population. That concludes the first
stage of the procedure. For the files whose names begin "synth..." or begin
"reverse...", we set the responses for all members of the subpopulation to 1,
as the second stage of the procedure.

This source code is licensed under the MIT license found in the LICENSE file in
the root directory of this source tree.
"""


import math
import numpy as np
from numpy.random import default_rng
import os

from hilbertcurve.hilbertcurve import HilbertCurve
from subpop import cumulative


# Set the number of examples.
m = 1000
# Set the size of the subpopulation.
n = 100
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

# Create a directory as needed.
dir = 'unweighted'
try:
    os.mkdir(dir)
except FileExistsError:
    pass
dir += '/'

# Consider both the original ordering of covariates and the reverse ordering,
# as well as a complete lack of significant deviation in the responses
# for the subpopulation.
for (reverse, diff) in [(False, True), (True, True), (False, False)]:
    print(f'reverse = {reverse}')
    print(f'diff = {diff}')
    pmax = 12
    # Set the number (p) of covariates.
    for p in [2**k for k in range(1, pmax + 1)]:
        print(f'p = {p}')
        # Set up the random number generator.
        rng = default_rng(seed=543216789)
        # Generate a random permutation for the indices of the subpopulation.
        inds = rng.permutation((m))[:n]
        # Generate data at random.
        x = rng.integers(2**precision - 1, size=(m, p), dtype=dtype)
        if reverse:
            x = x[:, ::-1]
        # Perform the Hilbert mapping from p dimensions to one dimension.
        hc = HilbertCurve(precision, p)
        ints = hc.distances_from_points(x)
        assert np.unique(ints).size == x.shape[0]
        # Sort according to the scores.
        perm = np.argsort(ints)
        x = x[perm, :]
        invperm = np.arange(len(perm))
        invperm[perm] = np.arange(len(perm))
        inds = invperm[inds]
        inds = np.sort(inds)
        # Construct scores for plotting.
        imin = np.min(ints)
        imax = np.max(ints)
        s = (np.sort(ints) - imin) / (imax - imin)
        # Ensure uniqueness even after roundoff errors.
        eps = np.finfo(np.float64).eps
        s = s + np.arange(0, s.size * eps, eps)
        s = s.astype(np.float64)
        # Form a random direction.
        w = rng.standard_normal(size=(p))
        w /= np.linalg.norm(w, ord=2)
        if reverse:
            w = w[::-1]
        # Generate responses based on the random direction and membership
        # in the subpopulation.
        centered = x.astype(np.float64) - 2**(precision - 1)
        r = (np.sign(centered @ w) + 1) / 2
        if diff:
            r[inds] = 1
        # Pad with zeros the number in the filename so that every filename
        # has the same number of characters for its length.
        max_digits = math.ceil(pmax * math.log(2) / math.log(10))
        if reverse and diff:
            name = 'reverse'
        elif diff:
            name = 'synth'
        else:
            name = 'randwalk'
        filename = dir + name + str(p).zfill(max_digits) + '.pdf'
        # Construct the graph of cumulative differences.
        majorticks = 10
        minorticks = 100
        kuiper, kolmogorov_smirnov, lenscale = cumulative(
            r, s, inds, majorticks, minorticks, filename=filename)
        # Save metrics in a text file.
        filename = filename[:-4] + '.txt'
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
