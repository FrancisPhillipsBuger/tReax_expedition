#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Diagnostics for a learned committor on a two-dimensional grid.

Created on: Tue Aug 25 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import numpy as np

from constants.units import BetaEff

#%% 1. Comparison
def AssessCommittor(QOf, x, y, V, QReference, tau, Saddles, SaddleEnergy,
                    Band=(0.05, 0.95), NkT=3.0, WeightFloor=1.0e-6):

    """

    Compare a committor with a numerically exact one on a 2D grid.

    The Boltzmann-weighted error is dominated by the basins, where any sensible
    model is right, so the error over the transition region is reported too:
    the grid points whose exact committor lies inside Band and whose energy is
    at most NkT kT_eff above SaddleEnergy.

    :param1 QOf:          callable mapping points (n, 2) to committor values.
    :param2 x:            grid axis along x.
    :param3 y:            grid axis along y.
    :param4 V:            potential on the grid, in eV.
    :param5 QReference:   the exact committor on the grid, oriented as the labels.
    :param6 tau:          time parameter tau (float), in fs.
    :param7 Saddles:      saddle points, shape (NSaddles, 2), where q is reported.
    :param8 SaddleEnergy: energy of the lowest saddle, in eV.
    :param9 Band:         committor range of the transition region.
    :param10 NkT:         energy window above the lowest saddle, in kT_eff.
    :param11 WeightFloor: ignore grid points below this fraction of the
                          largest Boltzmann weight.

    :return: Report: dict with the learned grid QLearned, TSErr, NTS,
             BoltzErr and the committor at every saddle, qS1, qS2, ...

    """

    Beta = BetaEff(tau)
    XX, YY = np.meshgrid(x, y, indexing="ij")
    QLearned = np.asarray(QOf(np.stack([XX.ravel(), YY.ravel()], axis=-1))).reshape(XX.shape)

    Mask = ((QReference > Band[0]) & (QReference < Band[1]) & (V <= SaddleEnergy + NkT/Beta))
    W = np.exp(-Beta*(V - V.min()))
    Keep = W > W.max()*WeightFloor
    w = W[Keep]/W[Keep].sum()

    Report = {"QLearned": QLearned,
              "TSErr": float(np.mean(np.abs(QLearned[Mask] - QReference[Mask]))) if Mask.any() else np.nan,
              "NTS": int(Mask.sum()),
              "BoltzErr": float(np.sum(w*np.abs(QLearned[Keep] - QReference[Keep])))}
    qS = np.asarray(QOf(np.atleast_2d(Saddles))).ravel()
    for k in range(len(qS)):
        Report["qS%i" % (k + 1)] = float(qS[k])

    return Report
