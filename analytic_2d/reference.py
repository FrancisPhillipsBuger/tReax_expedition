#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Numerically exact committor of a two-dimensional potential, from the backward
Kolmogorov equation on a grid, and its value at the saddles.

Created on: Tue Aug 25 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
from scipy.interpolate import RegularGridInterpolator

from constants.units import BetaEff

from .potential import ProtonTransfer

#%% 1. Grid
def BuildGrid(Potential=None, XLim=None, YLim=None, NX=None, NY=None):

    """

    Build the grid on which the reference problem is solved.

    :param1 Potential: a Potential2D, or None for the proton-transfer model.
    :param2 XLim:      range along x in angstroms, or None for the default.
    :param3 YLim:      range along y in angstroms, or None for the default.
    :param4 NX:        number of grid points along x, or None for the default.
    :param5 NY:        number of grid points along y, or None for the default.

    :return: x, y, V: the two axes and the potential on the grid, in eV.

    """

    Potential = ProtonTransfer() if Potential is None else Potential
    XLim = Potential.GridLimits[0] if XLim is None else XLim
    YLim = Potential.GridLimits[1] if YLim is None else YLim
    NX = Potential.GridShape[0] if NX is None else NX
    NY = Potential.GridShape[1] if NY is None else NY

    x = np.linspace(XLim[0], XLim[1], NX)
    y = np.linspace(YLim[0], YLim[1], NY)
    XX, YY = np.meshgrid(x, y, indexing="ij")

    return x, y, Potential.Energy(np.stack([XX, YY], axis=-1))

#%% 2. Committor
def SolveCommittor(x, y, V, tau, BasinRadius=0.25, Centres=None, MaxExponent=300.0):

    """

    Solve the stationary backward Kolmogorov equation on the grid.

    The self-adjoint form is discretised with harmonic averages of the
    Boltzmann factor on the cell faces, which keeps the operator symmetric and
    negative definite so the linear system is well conditioned.

    :param1 x:           grid axis along x.
    :param2 y:           grid axis along y.
    :param3 V:           potential on the grid, in eV.
    :param4 tau:         time parameter tau (float), in fs, setting beta.
    :param5 BasinRadius: radius of the two absorbing basins, in angstroms.
    :param6 Centres:     the two basin centres, or None for the two minima.
    :param7 MaxExponent: cap on beta (V - Vmin), or None; points above it are
                         all equally unreachable. Past about 370 the product
                         of two face weights underflows and the system is 0/0.

    :return: Q: the committor on the grid, shape (len(x), len(y)).

    """

    Beta = BetaEff(tau)
    NX = len(x)
    NY = len(y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    if Centres is None:
        i = int(np.argmin(V[:NX//2, :].min(axis=1)))
        j = int(np.argmin(V[NX//2:, :].min(axis=1))) + NX//2
        Centres = ((x[i], y[int(np.argmin(V[i, :]))]),
                   (x[j], y[int(np.argmin(V[j, :]))]))

    XX, YY = np.meshgrid(x, y, indexing="ij")
    InA = ((XX - Centres[0][0])**2 + (YY - Centres[0][1])**2) <= BasinRadius**2
    InB = ((XX - Centres[1][0])**2 + (YY - Centres[1][1])**2) <= BasinRadius**2
    Fixed = InA | InB

    # exp(-beta V) shifted so the largest weight is one, which only rescales
    # the whole operator and leaves the solution untouched
    Exponent = Beta*(V - V.min())
    if MaxExponent is not None:
        Exponent = np.minimum(Exponent, MaxExponent)
    W = np.exp(-Exponent)

    Index = -np.ones((NX, NY), dtype=int)
    Free = np.flatnonzero(~Fixed.ravel())
    Index.ravel()[Free] = np.arange(len(Free))

    Rows = []
    Cols = []
    Vals = []
    Rhs = np.zeros(len(Free))

    def Face(a, b):
        return 2.0*W[a]*W[b]/(W[a] + W[b])

    for k, flat in enumerate(Free):
        i, j = divmod(flat, NY)
        Diagonal = 0.0
        for di, dj, h in ((1, 0, dx), (-1, 0, dx), (0, 1, dy), (0, -1, dy)):
            ii, jj = i + di, j + dj
            if ii < 0 or ii >= NX or jj < 0 or jj >= NY:
                continue                      # reflecting outer boundary
            c = Face((i, j), (ii, jj))/h**2
            Diagonal -= c
            if Fixed[ii, jj]:
                Rhs[k] -= c*(1.0 if InB[ii, jj] else 0.0)
            else:
                Rows.append(k)
                Cols.append(Index[ii, jj])
                Vals.append(c)
        Rows.append(k)
        Cols.append(k)
        Vals.append(Diagonal)

    A = sps.csr_matrix((Vals, (Rows, Cols)), shape=(len(Free), len(Free)))
    Solution = spsl.spsolve(A, Rhs)

    Q = np.zeros((NX, NY))
    Q[InB] = 1.0
    Q.ravel()[Free] = Solution

    return np.clip(Q, 0.0, 1.0)

#%% 3. Committor at the saddles
def ReferenceAtSaddles(Potential, x, y, QReference):

    """

    :param1 Potential:  the Potential2D of the grid.
    :param2 x:          grid axis along x.
    :param3 y:          grid axis along y.
    :param4 QReference: the exact committor on the grid.

    :return: q: committor at every saddle, by bilinear interpolation.

    """

    return RegularGridInterpolator((x, y), QReference)(Potential.Saddles())
