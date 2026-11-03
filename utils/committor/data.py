#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Training data of the committor gathered over the iterations: blocks of
configurations each sampled under its own frozen bias, pooled with MBAR
weights, the basin configurations of the boundary condition, and the
outcomes of shots, the data of the binomial (AIMMD) likelihood.

Created on: Mon Sep 28 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import copy

import numpy as np

from constants.units import BetaEff

#%% 1. MBAR
def LogSumExp(a, axis=None):

    """

    :return: log(sum(exp(a))) along an axis, without overflow.

    """

    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    s = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m

    return np.squeeze(s, axis=axis) if axis is not None else float(s)

def MBARWeights(Reduced, Counts, MaxIterations=500, Tol=1.0e-10):

    """

    Unbiased weights of samples pooled from several biased ensembles.

    :param1 Reduced:       beta b_k(x_n), shape (K, N): every bias at every sample.
    :param2 Counts:        samples drawn from each ensemble, shape (K,).
    :param3 MaxIterations: self-consistent iterations.
    :param4 Tol:           convergence of the free energies.

    :return: w, f: normalised weights of the samples and the free energies
             of the biased ensembles relative to the unbiased one.

    """

    Reduced = np.asarray(Reduced, dtype=float)
    LogN = np.log(np.asarray(Counts, dtype=float))
    f = np.zeros(len(Counts))
    for _ in range(0, MaxIterations):
        LogDen = LogSumExp(LogN[:, None] + f[:, None] - Reduced, axis=0)
        New = -LogSumExp(-Reduced - LogDen[None, :], axis=1)
        New -= New[0]
        if np.max(np.abs(New - f)) < Tol:
            f = New
            break
        f = New
    LogDen = LogSumExp(LogN[:, None] + f[:, None] - Reduced, axis=0)
    LogW = -LogDen
    w = np.exp(LogW - LogW.max())

    return w/w.sum(), f

def TruncateWeights(Weights, Quantile=0.99):

    """

    Importance weights capped at a quantile of their distribution.

    Weights spanning many orders of magnitude make the variational term an
    average over a handful of configurations, and a model fitted to those few
    points stops separating the two basins. Capping them trades a small bias
    for a large cut in variance; a quantile, unlike a multiple of the mean,
    is set by the bulk of the sample and not by the few huge weights.

    :param1 Weights:  weights of the sample, not necessarily normalised.
    :param2 Quantile: quantile of the weights used as the cap.

    :return: w: capped weights, normalised to sum to one.

    """

    w = np.asarray(Weights, dtype=float).copy()
    if len(w) == 0 or w.sum() <= 0.0:
        return w

    w = np.minimum(w, np.quantile(w, Quantile))
    Total = w.sum()
    if Total <= 0.0:
        return np.full(len(w), 1.0/len(w))

    return w/Total

#%% 2. Store
class LabelledStore:

    """

    Everything the committor is trained on.

    """

    def __init__(self, XA, XB, tau):

        """

        :param1 XA:  basin configurations of the first state (q = 0).
        :param2 XB:  basin configurations of the second state (q = 1).
        :param3 tau: time parameter of the sampling, in fs.

        """

        self.XA = np.atleast_2d(np.asarray(XA, dtype=float))
        self.XB = np.atleast_2d(np.asarray(XB, dtype=float))
        self.Beta = BetaEff(tau)
        self.Blocks = []
        self.Biases = []
        self.Shots = []

    def AddBlock(self, X, Bias=None):

        """

        Append configurations sampled under one frozen bias.

        :param1 X:    configurations, shape (n, NumCoordinates).
        :param2 Bias: the frozen bias they were drawn with (anything with a
                      Values method over flat coordinates), or None. A deep
                      copy is kept, so later refits of the model it wraps do
                      not change it.

        :return: None.

        """

        X = np.atleast_2d(np.asarray(X, dtype=float))
        if len(X) == 0:
            return None
        self.Blocks.append(X)
        self.Biases.append(None if Bias is None else copy.deepcopy(Bias))

        return None

    def AddShots(self, X, NA, NB):

        """

        Append shooting points and how many of their trajectories reached each state.

        :param1 X:  shooting points, shape (m, NumCoordinates).
        :param2 NA: trajectories ending in the first state, shape (m,).
        :param3 NB: trajectories ending in the second state, shape (m,).

        :return: None.

        """

        for x, a, b in zip(np.atleast_2d(X), NA, NB):
            if a + b > 0:
                self.Shots.append((np.asarray(x, dtype=float), int(a), int(b)))

        return None

    @property
    def X(self):

        """

        :return: X: every sampled configuration.

        """

        return np.concatenate(self.Blocks)

    def Weights(self, Quantile=0.99):

        """

        MBAR weights of the pooled blocks, capped at a quantile.

        :param1 Quantile: cap of the weights.

        :return: w: weights summing to one.

        """

        X = self.X
        Reduced = np.zeros((len(self.Blocks), len(X)))
        for k, b in enumerate(self.Biases):
            if b is not None:
                Reduced[k] = self.Beta*np.asarray(b.Values(X), dtype=float)
        w, _ = MBARWeights(Reduced, [len(b) for b in self.Blocks])

        return TruncateWeights(w, Quantile=Quantile)

    def ShotArrays(self):

        """

        :return: X, NA, NB: the shots as arrays, or empty arrays.

        """

        if len(self.Shots) == 0:
            return np.zeros((0, self.XA.shape[1])), np.zeros(0), np.zeros(0)

        return (np.array([s[0] for s in self.Shots]), np.array([s[1] for s in self.Shots], dtype=float),
                np.array([s[2] for s in self.Shots], dtype=float))
