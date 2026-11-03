#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Tests of the structural distances of analyse/ts_metrics.py: Kabsch RMSD,
the symmetry-aware RMSD over like-atom permutations and inversion, the
chirality test and the dihedral convention.

Created on: Thu Oct 01 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import os
import sys

import numpy as np
from ase.build import molecule
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyse.ts_metrics import AngleDifference, Dihedral, IsChiral, RMSD, SymmetryRMSD

#%% 1. Tests
def Moved(X, Seed=0):

    """

    :return: X rotated and translated at random.

    """

    R = Rotation.random(random_state=Seed).as_matrix()

    return X @ R.T + np.array([0.3, -1.2, 2.0])

def test_rmsd_is_rigid_motion_invariant():

    A = molecule("CH3CHO")
    X = A.get_positions()
    assert RMSD(X, Moved(X)) < 1e-10
    assert RMSD(X, Moved(X), Weights=A.get_masses()) < 1e-10
    assert RMSD(X, Moved(X), Align=False) > 0.5

def test_symmetry_rmsd_undoes_permutation_and_mirror():

    # distorted so that no proper rotation maps the mirror image onto itself
    A = molecule("CH3CHO")
    X = A.get_positions() + np.random.default_rng(7).normal(scale=0.05, size=(len(A), 3))
    Sym = A.get_chemical_symbols()
    H = [i for i, s in enumerate(Sym) if s == "H"]
    Order = np.arange(len(X))
    Order[H] = np.roll(H, 1)
    Y = Moved(X[Order]*np.array([1.0, 1.0, -1.0]), Seed=3)
    assert RMSD(X, Y) > 0.1
    r, Aligned, _, Mirrored = SymmetryRMSD(X, Y, Sym)
    assert r < 1e-8 and Mirrored
    assert np.allclose(Aligned, X, atol=1e-6)
    assert SymmetryRMSD(X, Y, Sym, Mirror=False)[0] > 0.01

def test_symmetry_rmsd_keeps_identity_for_small_displacements():

    A = molecule("CH3CHO")
    X = A.get_positions()
    Y = X + np.random.default_rng(1).normal(scale=0.01, size=X.shape)
    r, _, Order, Mirrored = SymmetryRMSD(X, Y, A.get_chemical_symbols())
    assert np.array_equal(Order, np.arange(len(X))) and not Mirrored
    assert abs(r - RMSD(X, Y)) < 1e-12

def test_chirality():

    assert not IsChiral(molecule("CH3CHO").get_positions(), molecule("CH3CHO").get_chemical_symbols())
    # CHFClBr built from methane: four different substituents
    A = molecule("CH4")
    Sym = ["C", "F", "Cl", "Br", "H"]
    X = A.get_positions()
    assert IsChiral(X, Sym)

def test_dihedral_matches_ase_and_wraps():

    A = molecule("C2H6")
    Quad = [2, 0, 1, 5]
    Ref = AngleDifference(A.get_dihedral(*Quad), 0.0)
    assert abs(AngleDifference(Dihedral(A.get_positions(), Quad), Ref)) < 1e-8
    assert AngleDifference(170.0, -170.0) == -20.0
