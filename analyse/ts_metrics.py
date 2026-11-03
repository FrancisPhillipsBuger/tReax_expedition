#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Structural distances between a located transition state and a reference:
Kabsch RMSD (plain, mass-weighted, on an atom subset), the RMSD minimised
over permutations of like atoms and, for an achiral molecule, the mirror
image, and the dihedral angles a peptide is described by.

Created on: Thu Oct 01 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import numpy as np
from scipy.optimize import linear_sum_assignment

#%% 1. Kabsch alignment
def KabschAlign(Ref, X, Weights=None):

    """

    Rotate and translate X onto Ref, proper rotations only.

    :param1 Ref:     reference positions, shape (N, 3).
    :param2 X:       positions to move, shape (N, 3).
    :param3 Weights: per-atom weights, None for equal weights.

    :return: Aligned: X after the best rigid motion, shape (N, 3).

    """

    Ref = np.asarray(Ref, dtype=float)
    X = np.asarray(X, dtype=float)
    w = np.ones(len(Ref)) if Weights is None else np.asarray(Weights, dtype=float)
    w = w/w.sum()
    CR = w @ Ref
    CX = w @ X
    H = ((X - CX)*w[:, None]).T @ (Ref - CR)
    U, _, Vt = np.linalg.svd(H)
    D = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, D]) @ U.T

    return (X - CX) @ R.T + CR

def RMSD(Ref, X, Weights=None, Subset=None, Align=True):

    """

    :param1 Ref:     reference positions, shape (N, 3).
    :param2 X:       positions, shape (N, 3), same atom order.
    :param3 Weights: per-atom weights (masses for the mass-weighted RMSD).
    :param4 Subset:  atom indices the alignment and the RMSD use, None for all.
    :param5 Align:   superimpose first; False for a fixed frame (2D models).

    :return: RMSD: in the units of the positions.

    """

    Ref = np.asarray(Ref, dtype=float)
    X = np.asarray(X, dtype=float)
    w = np.ones(len(Ref)) if Weights is None else np.asarray(Weights, dtype=float)
    if Subset is not None:
        Ref, X, w = Ref[Subset], X[Subset], w[Subset]
    if Align:
        X = KabschAlign(Ref, X, w)

    return float(np.sqrt(np.sum(w*np.sum((X - Ref)**2, axis=1))/np.sum(w)))

#%% 2. Symmetry-aware RMSD
def _Assign(Ref, X, Symbols):

    """

    :return: Order: for every reference atom, the index of the like atom of X
             nearest to it under one optimal assignment per element.

    """

    Order = np.arange(len(Ref))
    for s in sorted(set(Symbols)):
        Idx = np.array([i for i, t in enumerate(Symbols) if t == s])
        if len(Idx) < 2:
            continue
        Cost = np.sum((Ref[Idx][:, None, :] - X[Idx][None, :, :])**2, axis=2)
        Row, Col = linear_sum_assignment(Cost)
        Order[Idx[Row]] = Idx[Col]

    return Order

def SymmetryRMSD(Ref, X, Symbols, Mirror=True, MaxIter=50):

    """

    RMSD minimised over permutations of like atoms and, optionally, inversion.
    Alignment and assignment alternate until the assignment stops changing,
    starting from the identity, so a nearby structure keeps its own order.

    :param1 Ref:     reference positions, shape (N, 3).
    :param2 X:       positions, shape (N, 3).
    :param3 Symbols: chemical symbols, same order as Ref.
    :param4 Mirror:  also try the mirror image of X (only for achiral systems).
    :param5 MaxIter: cap on the alignment/assignment cycles.

    :return: RMSD, Aligned, Order, Mirrored: the best RMSD, X permuted and
             aligned onto Ref, the permutation applied, and whether X was inverted.

    """

    Ref = np.asarray(Ref, dtype=float)
    Best = None
    for m in ((1.0, -1.0) if Mirror else (1.0,)):
        Y = np.asarray(X, dtype=float)*np.array([1.0, 1.0, m])
        # the identity first, so a nearby structure keeps its own order; then
        # the principal-axes frames, for a structure far from the reference
        for Start in [None] + _AxesStarts(Ref, Y):
            Order = np.arange(len(Ref)) if Start is None else _Assign(Ref, Start, Symbols)
            for _ in range(MaxIter):
                A = KabschAlign(Ref, Y[Order])
                # assignment on the aligned structure, mapped back to Y's indices
                New = Order[_Assign(Ref, A, Symbols)]
                if np.array_equal(New, Order):
                    break
                Order = New
            A = KabschAlign(Ref, Y[Order])
            r = float(np.sqrt(np.mean(np.sum((A - Ref)**2, axis=1))))
            if Best is None or r < Best[0] - 1e-9:
                Best = (r, A, Order, m < 0.0)

    return Best

def _AxesStarts(Ref, Y):

    """

    :return: Starts: Y rotated so its principal axes lie on those of Ref, one
             copy per sign choice of the axes that keeps the rotation proper.

    """

    def Axes(Z):
        C = Z - Z.mean(axis=0)
        return C, np.linalg.eigh(C.T @ C)[1]

    CR, VR = Axes(Ref)
    CY, VY = Axes(Y)
    Starts = []
    for s in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
        R = VR @ np.diag(s) @ VY.T
        if np.linalg.det(R) < 0.0:
            R = VR @ np.diag(-np.array(s)) @ VY.T
        Starts.append(CY @ R.T + Ref.mean(axis=0))

    return Starts

def IsChiral(X, Symbols, Tolerance=0.1):

    """

    :param1 X:         positions of a minimum, shape (N, 3).
    :param2 Symbols:   chemical symbols.
    :param3 Tolerance: RMSD in angstroms below which the mirror image counts as the same.

    :return: Chiral: True when the mirror image cannot be superimposed on X,
             in which case inversion is no symmetry of the reaction.

    """

    Mirror = np.asarray(X, dtype=float)*np.array([1.0, 1.0, -1.0])

    return SymmetryRMSD(X, Mirror, Symbols, Mirror=False)[0] > Tolerance

#%% 3. Dihedral angles
def Dihedral(X, Quad):

    """

    :param1 X:    positions, shape (N, 3).
    :param2 Quad: four 0-based atom indices.

    :return: Angle: IUPAC dihedral angle in degrees, in (-180, 180].

    """

    p = np.asarray(X, dtype=float)[list(Quad)]
    b0, b1, b2 = p[0] - p[1], p[2] - p[1], p[3] - p[2]
    b1 = b1/np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))

def AngleDifference(a, b):

    """

    :return: Delta: a - b wrapped into (-180, 180] degrees.

    """

    return float((a - b + 180.0) % 360.0 - 180.0)
