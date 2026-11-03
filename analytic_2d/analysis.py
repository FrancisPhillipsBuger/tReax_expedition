#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Post-processing for 2D Monte-Carlo trajectories: bin the sampled points
between the two minima into a mean path with standard deviations, write it
out, and reduce the per-trial sums of a scan into means and standard errors.

Created on: Mon Aug 24 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import os

import numpy as np

#%% 1. Mean path binning
def CompressPath(A, B, X, Y, E, N=10, Axis=None):

    """

    Bin a sampled path between two minima into a mean path with deviations.

    The points are projected on Axis and the projection is scaled so that A
    lands on 0 and B on 1; N bins are centred on 0, 1/(N-1), ..., 1. With the
    x axis this is the binning along the proton-transfer coordinate.

    :param1 A:    reactant minimum.
    :param2 B:    product minimum.
    :param3 X:    sampled x positions.
    :param4 Y:    sampled y positions.
    :param5 E:    sampled energies.
    :param6 N:    number of bins.
    :param7 Axis: direction of the projection, or None for the line A to B.

    :return: s, x, y, e, n: bin centres, (mean, std) pairs of x, y and the
             energy, and the counts, over the non-empty bins.

    """

    A = np.asarray(A, dtype=float)
    d = np.asarray(B, dtype=float) - A if Axis is None else np.asarray(Axis, dtype=float)
    P = np.stack([np.asarray(X, dtype=float), np.asarray(Y, dtype=float)], axis=-1)
    E = np.asarray(E, dtype=float)
    s = (P - A) @ d/np.dot(np.asarray(B, dtype=float) - A, d)

    ds = 1.0/(N - 1)
    Idx = np.floor((s + 0.5*ds)/ds).astype(int)
    Rows = []
    for k in range(N):
        m = Idx == k
        if not m.any():
            continue
        Rows.append((k*ds, P[m, 0].mean(), P[m, 0].std(), P[m, 1].mean(), P[m, 1].std(),
                     E[m].mean(), E[m].std(), int(m.sum())))
    Rows = np.array(Rows).reshape(-1, 8)

    return Rows[:, 0], Rows[:, 1:3], Rows[:, 3:5], Rows[:, 5:7], Rows[:, 7].astype(int)

#%% 2. Mean path output
def SaveMeanPath(s, x, y, e, n, FolderToSave="."):

    """

    Write the binned mean path to ``mean_path.dat``.

    :param1 s:            bin centres, 0 at A and 1 at B.
    :param2 x:            (mean, std) of x per bin.
    :param3 y:            (mean, std) of y per bin.
    :param4 e:            (mean, std) of the energy per bin, in native units.
    :param5 n:            number of samples per bin.
    :param6 FolderToSave: folder for the output file.

    :return: None.

    """

    np.savetxt(os.path.join(FolderToSave, "mean_path.dat"),
               np.stack([x[:, 0], y[:, 0], e[:, 0], y[:, 1], e[:, 1], n, s, x[:, 1]], axis=-1),
               header="x[$1] y[$2] e[$3] std(y)[$4] std(e)[$5] n[$6] s[$7] std(x)[$8]")

    return None

#%% 3. Statistics over trials
def ConvPairMeans(x,x2,n):

    """

    Turn accumulated sums of a quantity and its square into mean and error.

    :param1 x:  sum of the quantity over the trials.
    :param2 x2: sum of its squares over the trials.
    :param3 n:  number of trials.

    :return: x, x2: mean and standard error of the mean, the latter NaN for
             fewer than two trials.

    """

    x /= n
    x2 /= n
    if n < 2:
        return x, np.nan
    # sample variance, n/(n-1) times the second central moment, over n
    x2 = np.sqrt(np.maximum(x2-x**2, 0.0)/(n-1))
    return x,x2
