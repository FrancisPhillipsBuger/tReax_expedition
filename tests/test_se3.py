#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

SE(3) contract of the molecular code. Every quantity entering a Metropolis
test, a coupling, a weight or a progress coordinate is invariant under a
proper rotation and a translation, every construction on configurations is
equivariant, and each frame removal is applied only where the target is
invariant under it: a single walker may be rotated on its own, the walkers of
a pair only together.

Created on: Tue Sep 29 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import random

import numpy as np
import pytest
import torch
from ase import Atoms

from treax_expedition.geometry import (EndpointAxis, calcAlignedRMSDBetweenStructures,
                                       orientPairOnto, rotateStructureRandomly, rotationWidth)
from utils.committor import CommittorNN, KolmogorovBias, PairwiseDistances
from utils.mc.ensemble import Midpoint
from utils.mc.system import AtomsSystem, RotationWidth
from utils.path.coordinate import BondChange, DihedralChange
from utils.path.states import BondGraph, DihedralBox

#%% 1. Helpers
Rng = np.random.default_rng(7)

def RandomProper():

    """

    :return: R, t: a random proper rotation and a random translation.

    """

    Q, _ = np.linalg.qr(Rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0.0:
        Q[:, 0] = -Q[:, 0]

    return Q, Rng.normal(size=3)*3.0

def Moved(X, R, t):

    """

    :return: X rotated by R and translated by t, flat.

    """

    return (np.asarray(X, dtype=float).reshape(-1, 3) @ R.T + t).reshape(-1)

def Molecule():

    """

    :return: a chiral, non planar five-atom molecule with unequal masses.

    """

    A = Atoms("CHFClBr", positions=[[0.0, 0.0, 0.0], [0.63, 0.63, 0.63], [-0.8, -0.7, 0.75],
                                    [-1.0, 1.0, -1.0], [1.1, -1.2, -1.1]])

    return A

class Pseudo(AtomsSystem):

    """

    A molecule on an invariant model energy, sum_ij (r_ij - r0_ij)^2 over all
    pairs, taken directly instead of through a calculator.

    """

    def __init__(self):

        super().__init__(Molecule(), None)
        XYZ = Molecule().positions
        self.R0 = np.linalg.norm(XYZ[:, None] - XYZ[None], axis=-1)

    def Energy(self, X):

        self.NCalls += 1
        XYZ = np.asarray(X, dtype=float).reshape(-1, 3)

        return float(np.sum((np.linalg.norm(XYZ[:, None] - XYZ[None], axis=-1) - self.R0)**2))

X0 = Molecule().positions.reshape(-1)
X1 = X0 + Rng.normal(size=X0.shape)*0.15

#%% 2. Invariants
def test_the_distance_and_the_frame_removal_see_only_the_shape():

    """Distance ignores any motion of either walker; Aligned is equivariant."""

    S = Pseudo()
    R, t = RandomProper()
    R2, t2 = RandomProper()
    d = S.Distance(X0, X1)
    assert S.Distance(Moved(X0, R, t), Moved(X1, R2, t2)) == pytest.approx(d, rel=1e-10)
    # aligning onto a moved reference moves the result with it
    A = S.Aligned(X1, X0).reshape(-1, 3)
    B = S.Aligned(Moved(X1, R2, t2), Moved(X0, R, t)).reshape(-1, 3)
    assert np.allclose(B, A @ R.T, atol=1e-9)
    # the rotation is proper: a mirror image is not aligned onto the original
    Mirror = X0.reshape(-1, 3)*np.array([1.0, 1.0, -1.0])
    assert S.Distance(X0, Mirror.reshape(-1)) > 0.1

def test_the_pair_coupling_is_invariant_only_under_a_common_motion():

    """FrameDistance ignores a common SE(3) motion and separate translations,
    but not a separate rotation, which would add an orbit volume."""

    S = Pseudo()
    R, t = RandomProper()
    _, t2 = RandomProper()
    d = S.FrameDistance(X0, X1)
    assert S.FrameDistance(Moved(X0, R, t), Moved(X1, R, t2)) == pytest.approx(d, rel=1e-10)
    R2, _ = RandomProper()
    assert abs(S.FrameDistance(X0, Moved(X1, R2, 0.0)) - d) > 1.0e-3

def test_the_midpoint_moves_with_the_pair():

    """Midpoint(g X1, g X2) = R Midpoint(X1, X2), centred."""

    S = Pseudo()
    R, t = RandomProper()
    M = Midpoint(S, X0, X1).reshape(-1, 3)
    Mg = Midpoint(S, Moved(X0, R, t), Moved(X1, R, t)).reshape(-1, 3)
    assert np.allclose(Mg, M @ R.T, atol=1e-9)

def test_the_rotation_move_keeps_the_shape_and_its_width():

    """The relative rotation is rigid, so energy and width are those it had."""

    S = Pseudo()
    random.seed(3)
    X = S.Centred(X1)
    Y = S.Rotated(X, 0.7)
    assert S.Energy(Y) == pytest.approx(S.Energy(X), abs=1e-10)
    assert S.Distance(X, Y) == pytest.approx(0.0, abs=1e-7)
    assert S.RotationWidth(Y, 5.0) == pytest.approx(S.RotationWidth(X, 5.0), rel=1e-12)
    R, t = RandomProper()
    assert (RotationWidth(Moved(X, R, t).reshape(-1, 3), S.AtomMasses, 5.0)
            == pytest.approx(S.RotationWidth(X, 5.0), rel=1e-12))

def test_every_progress_coordinate_and_fingerprint_is_invariant():

    """Bond graph, bond change, dihedrals and the endpoint axis."""

    R, t = RandomProper()
    Graph = BondGraph(Molecule().get_atomic_numbers())
    Far = X0.reshape(-1, 3).copy()
    Far[4] = Far[4]*2.0
    Xi = BondChange(Graph, X0, Far.reshape(-1))
    assert Xi.Available
    Box = DihedralBox([(1, 0, 2, 3), (2, 0, 3, 4)])
    Dih = DihedralChange(Box, X0, X1)
    for X in (X1, 0.5*(X0 + X1)):
        Y = Moved(X, R, t)
        assert Graph.Key(Y) == Graph.Key(X)
        assert Xi(Y) == pytest.approx(Xi(X), abs=1e-10)
        assert Box.Key(Y) == Box.Key(X)
        assert Dih(Y) == pytest.approx(Dih(X), abs=1e-9)
    A, B = Molecule(), Molecule()
    B.positions = X1.reshape(-1, 3)
    Axis = EndpointAxis(A, B)
    s = Axis(0.3*X0.reshape(-1, 3) + 0.7*X1.reshape(-1, 3))
    assert Axis(Moved(0.3*X0 + 0.7*X1, R, t).reshape(-1, 3)) == pytest.approx(s, abs=1e-9)

def test_the_committor_and_its_bias_are_invariant():

    """Latent, committor and the Kolmogorov bias, whose mass-weighted
    gradient norm is invariant because the gradient is equivariant."""

    torch.manual_seed(0)
    Model = CommittorNN(PairwiseDistances(5)).double()
    Default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        Masses = np.repeat(Molecule().get_masses(), 3)
        Bias = KolmogorovBias(Model, Masses, 10.0)
        R, t = RandomProper()
        X = np.stack([X1, Moved(X1, R, t)])
        with torch.no_grad():
            z = Model(torch.as_tensor(X)).numpy()
        assert z[1] == pytest.approx(z[0], abs=1e-10)
        V = Bias.Values(X)
        assert V[1] == pytest.approx(V[0], abs=1e-8)
    finally:
        torch.set_default_dtype(Default)

#%% 3. The ase.Atoms side of the pair
def test_the_atoms_pair_helpers_keep_the_pair_target():

    """The random rotation is rigid; the output frame turns both walkers by
    one rotation, which leaves their coupling distance as it was."""

    random.seed(5)
    A, B = Molecule(), Molecule()
    B.positions = X1.reshape(-1, 3)
    for S in (A, B):
        S.positions = S.positions - S.get_center_of_mass()
    Turned = rotateStructureRandomly(B, 0.8)
    assert calcAlignedRMSDBetweenStructures(Turned, B, True) == pytest.approx(0.0, abs=1e-8)
    assert rotationWidth(Turned, 5.0) == pytest.approx(rotationWidth(B, 5.0), rel=1e-12)
    R, _ = RandomProper()
    Ref = 0.5*(A.positions + B.positions) @ R.T
    P, Q = orientPairOnto([A, B], Ref)
    D0 = np.sum(A.get_masses()[:, None]*(A.positions - B.positions)**2)
    D1 = np.sum(A.get_masses()[:, None]*(P.positions - Q.positions)**2)
    assert D1 == pytest.approx(D0, rel=1e-10)
    assert np.allclose(0.5*(P.positions + Q.positions), Ref, atol=1e-8)

#%% 4. The energy back end
def test_the_tight_binding_energy_is_invariant():

    """GFN2-xTB gives the same energy for a moved molecule."""

    pytest.importorskip("tblite")
    from treax_expedition.xtb import getXTBCalculator

    A = Atoms("CH3Cl", positions=[[0.0, 0.0, 0.0], [0.0, 1.03, 0.36], [0.89, -0.51, 0.36],
                                  [-0.89, -0.51, 0.36], [0.0, 0.0, -1.78]])
    A.calc = getXTBCalculator()
    E = A.get_potential_energy()
    R, t = RandomProper()
    A.positions = A.positions @ R.T + t
    assert A.get_potential_energy() == pytest.approx(E, abs=1e-7)
