#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Tests for the geometry helpers, in particular the decision of whether the
overall frame has to be removed. Removing it from a structure whose frame is
already pinned corrupts the geometry, so this is guarded explicitly.

Created on: Tue Aug 25 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import numpy as np
import pytest
from ase.build import add_adsorbate, fcc111, molecule
from ase.constraints import FixAtoms

from treax_expedition.geometry import (MaybeCenter, MaybeOrient, NeedsFrameRemoval,
                                ProjectOntoEndpointAxis,
                                calcRMSDBetweenStructures, centerMolecule,
                                displaceSWSStructure)

#%% 1. Fixtures
def MakeSlab():

    """A constrained periodic slab with a single mobile adatom."""

    s = fcc111('Al', size=(2, 2, 3), vacuum=10.0)
    add_adsorbate(s, 'Au', 2.0, 'fcc')
    s.set_constraint(FixAtoms(mask=[a.symbol == 'Al' for a in s]))

    return s

#%% 2. Frame detection
def test_free_molecule_needs_frame_removal():

    """A molecule in vacuum carries six spurious degrees of freedom."""

    assert NeedsFrameRemoval(molecule('CH3Cl')) is True

def test_periodic_slab_does_not_need_frame_removal():

    """A cell already pins the frame."""

    assert NeedsFrameRemoval(MakeSlab()) is False

def test_constrained_molecule_does_not_need_frame_removal():

    """Fixed atoms already pin the frame."""

    m = molecule('CH3Cl')
    m.set_constraint(FixAtoms(indices=[0]))
    assert NeedsFrameRemoval(m) is False

#%% 3. The bug the frame detection prevents
def test_centering_a_constrained_slab_would_tear_it_apart():

    """

    ASE refuses to move constrained atoms, so subtracting the centre of mass
    from a slab displaces only the free adatom, by the whole vector. This is
    exactly why the samplers must not centre such a structure.

    """

    s = MakeSlab()
    p0 = s.get_positions().copy()
    broken = centerMolecule(s.copy())
    d = np.linalg.norm(broken.get_positions() - p0, axis=1)

    assert d[:-1].max() == pytest.approx(0.0)   # the slab does not move
    assert d[-1] > 5.0                          # the adatom is thrown away

def test_maybe_center_leaves_a_pinned_structure_alone():

    """The guarded version must be a no-op when the frame is pinned."""

    s = MakeSlab()
    p0 = s.get_positions().copy()
    kept = MaybeCenter(s.copy(), NeedsFrameRemoval(s))

    assert np.allclose(kept.get_positions(), p0)

def test_maybe_center_still_centres_a_free_molecule():

    """The guarded version must keep working for the free molecule case."""

    m = molecule('CH3Cl')
    centred = MaybeCenter(m.copy(), NeedsFrameRemoval(m))

    assert np.allclose(centred.get_center_of_mass(), 0.0, atol=1e-10)

def test_maybe_orient_is_a_no_op_when_pinned():

    """Rotating a slab is meaningless and must be skipped."""

    s = MakeSlab()
    other = MakeSlab()
    p0 = s.get_positions().copy()
    kept = MaybeOrient(s.copy(), other, Align=NeedsFrameRemoval(s))

    assert np.allclose(kept.get_positions(), p0)

def test_maybe_orient_removes_a_rotation_for_a_free_molecule():

    """A rigidly rotated copy must be brought back onto the original."""

    m = molecule('CH3Cl')
    m = centerMolecule(m)
    rotated = m.copy()
    rotated.rotate(37.0, 'z')

    before = calcRMSDBetweenStructures(rotated, m, UseMassWeightedRMSD=True)
    aligned = MaybeOrient(rotated, m, Align=True, UseMassWeightedRMSD=True)
    after = calcRMSDBetweenStructures(aligned, m, UseMassWeightedRMSD=True)

    assert before > 1.0
    assert after == pytest.approx(0.0, abs=1e-8)

#%% 4. Displacement respects constraints
def test_displacement_never_moves_a_constrained_atom():

    """Only the mobile atoms may take a step."""

    s = MakeSlab()
    p0 = s.get_positions().copy()
    d = displaceSWSStructure(s, tau=4.0, SingleAtom=False)
    moved = np.linalg.norm(d.get_positions() - p0, axis=1)

    assert moved[:-1].max() == pytest.approx(0.0)
    assert moved[-1] > 0.0

#%% 5. Progress coordinate
def test_progress_coordinate_spans_the_endpoints():

    """The coordinate must be zero at the initial and one at the final point."""

    a = molecule('CH3Cl').get_positions()
    b = a + 0.5
    m = molecule('CH3Cl').get_masses()

    assert ProjectOntoEndpointAxis(a, a, b, m) == pytest.approx(0.0)
    assert ProjectOntoEndpointAxis(b, a, b, m) == pytest.approx(1.0)
    assert ProjectOntoEndpointAxis(0.5*(a + b), a, b, m) == pytest.approx(0.5)

def test_progress_coordinate_survives_identical_endpoints():

    """Coincident endpoints must give zero rather than divide by zero."""

    a = molecule('CH3Cl').get_positions()
    assert ProjectOntoEndpointAxis(a, a, a) == 0.0
