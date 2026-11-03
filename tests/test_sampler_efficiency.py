#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Tests for the step width, the fail-safe evaluation and the cheaper alignment
of the molecular sampler. The step width may change how fast the walkers move
but never what they sample; a failed energy must reject one trial rather than
end the run; and the alignment done from singular values must agree with the
explicit rotation it replaced.

Created on: Sat Sep 26 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import random

import numpy as np
import pytest
import scipy.spatial.transform as spst
from ase import Atoms
from ase.build import molecule
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read

from constants.units import kTEffeV
from treax_expedition.geometry import (calcAlignedRMSDBetweenStructures,
                                calcRMSDBetweenStructures, centerMolecule,
                                displaceSWSStructure, orientStructureWrtAnother)
from treax_expedition.orca import getORCACalculator
from treax_expedition.sampler import RunStats, doSingleStructureSWSMC, doTwoStructureSWSMC
from utils.surrogate import CalculatorSurrogate

#%% 1. Fixtures
class Harmonic(Calculator):

    """Isotropic harmonic well around the origin, failing beyond a wall."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, k=100.0, Wall=None):

        super().__init__()
        self.k = float(k)
        self.Wall = Wall

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):

        super().calculate(atoms, properties, system_changes)
        x = self.atoms.get_positions()
        if self.Wall is not None and np.any(x[:, 0] > self.Wall):
            raise RuntimeError("SCF not converged")
        self.results["energy"] = 0.5*self.k*float(np.sum(x*x))
        self.results["forces"] = -self.k*x

def OneAtomInABox(**Keywords):

    """A single H atom in a periodic cell, so no frame is removed."""

    a = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[20.0, 20.0, 20.0], pbc=True)
    a.calc = Harmonic(**Keywords)

    return a

def RandomMolecule(Seed):

    """A centred water with a random rotation and a small distortion."""

    rng = np.random.default_rng(Seed)
    m = molecule("H2O")
    m.positions += 0.05*rng.standard_normal(m.positions.shape)
    m.positions = spst.Rotation.random(random_state=Seed).apply(m.positions)

    return centerMolecule(m)

#%% 2. Alignment from singular values
@pytest.mark.parametrize("Seed", range(5))
def test_the_kabsch_rotation_matches_scipy(Seed):

    """The numpy rotation reproduces what align_vectors did before."""

    Ref = RandomMolecule(Seed)
    Mov = RandomMolecule(Seed + 100)
    Expected = spst.Rotation.align_vectors(Ref.positions, Mov.positions,
                                           weights=Ref.get_masses())[0].apply(Mov.positions)
    Got = orientStructureWrtAnother(Mov.copy(), Ref).positions

    assert np.allclose(Got, Expected, atol=1e-10)

@pytest.mark.parametrize("Seed", range(5))
def test_the_aligned_rmsd_equals_rotate_then_measure(Seed):

    """No copy is rotated any more, and the value must not notice."""

    Ref = RandomMolecule(Seed)
    Mov = RandomMolecule(Seed + 100)
    Explicit = calcRMSDBetweenStructures(orientStructureWrtAnother(Mov.copy(), Ref), Ref)

    assert calcAlignedRMSDBetweenStructures(Mov, Ref, True) == pytest.approx(Explicit, rel=1e-10)
    assert calcAlignedRMSDBetweenStructures(Ref, Mov, True) == pytest.approx(Explicit, rel=1e-10)

def test_the_aligned_rmsd_of_a_rotated_copy_is_zero():

    """A pure rotation is not a displacement."""

    Ref = RandomMolecule(1)
    Rot = Ref.copy()
    Rot.positions = spst.Rotation.random(random_state=7).apply(Rot.positions)

    assert calcAlignedRMSDBetweenStructures(Rot, Ref, True) == pytest.approx(0.0, abs=1e-10)

#%% 3. Step width
def test_the_step_scale_stretches_the_same_draws():

    """With the same seed a larger width moves the atom along the same vector."""

    a = OneAtomInABox()
    random.seed(3)
    d1 = displaceSWSStructure(a, 2.0, SingleAtom=False).positions
    random.seed(3)
    d3 = displaceSWSStructure(a, 2.0, SingleAtom=False, StepScale=3.0).positions

    assert np.allclose(d3, 3.0*d1, rtol=1e-12)

@pytest.mark.parametrize("StepScale", [1.0, 3.0])
def test_the_step_scale_leaves_the_sampled_distribution_alone(StepScale, tmp_path):

    """

    The mean energy of a three dimensional harmonic well is 3/2 kT_eff
    whatever the proposal width, since any symmetric proposal cancels in the
    Metropolis ratio.

    """

    a = OneAtomInABox(k=100.0)
    doSingleStructureSWSMC(a, tau=1.0, Nsteps=20000, Neq=2000, Seed=11,
                           LogEvaluations=False, FolderToSave=str(tmp_path),
                           StepScale=StepScale)
    E = np.loadtxt(tmp_path/"sampled_energy.log", usecols=1)

    assert E.mean() == pytest.approx(1.5*kTEffeV(1.0), rel=0.06)

#%% 4. Failed evaluations
def test_a_failed_energy_rejects_the_trial_and_the_run_goes_on(tmp_path):

    """The trial is billed and rejected; the walker never enters the wall."""

    a = OneAtomInABox(k=10.0, Wall=0.02)
    NTrue = doSingleStructureSWSMC(a, tau=1.0, Nsteps=400, Seed=2,
                                   LogEvaluations=False, FolderToSave=str(tmp_path))
    Frames = read(tmp_path/"sampled.trj", index=":", format="xyz")

    assert NTrue == 400
    assert max(f.positions[0, 0] for f in Frames) <= 0.02
    assert min(f.positions[0, 0] for f in Frames) < 0.0

def test_the_run_statistics_count_the_failures():

    """The summary names what the energy calls achieved."""

    s = RunStats()
    s.NEvaluated, s.NAccepted, s.NFailed = 10, 4, 2

    assert "evaluated trials: 10, accepted: 4 (0.40), failed: 2" in s.Summary()

def test_a_failed_cheap_level_screens_the_trial_out():

    """A cheap SCF that does not converge gives +inf, which the first stage rejects."""

    a = OneAtomInABox()
    a.positions[0, 0] = 0.5
    s = CalculatorSurrogate(Harmonic(Wall=0.1))

    assert s(a) == float("inf")
    assert s.NFailed == 1

def test_a_chain_cannot_start_where_the_surrogate_fails(tmp_path):

    """Every move out of such a point would be rejected, so it is refused up front."""

    Start = Atoms("Cu3", positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [1.2, 2.1, 0.0]])
    End = Atoms("Cu3", positions=[[0.0, 0.0, 0.0], [2.6, 0.0, 0.3], [1.0, 2.2, -0.4]])
    from ase.calculators.emt import EMT
    for s in (Start, End):
        s.calc = EMT()

    with pytest.raises(ValueError, match="surrogate is not finite"):
        doTwoStructureSWSMC([Start, End], tau=5.0, tau12=50.0, Nsteps=5,
                            LogEvaluations=False, FolderToSave=str(tmp_path),
                            Surrogate=lambda S: float("inf"))

def test_orca_refuses_gradient_keywords():

    """The sampler lives on single point energies; a gradient or optimiser keyword is an error."""

    Plain = getORCACalculator(LevelOfTheory="hf-3c", PathToOrca="/nonexistent/orca")
    assert Plain.parameters["orcasimpleinput"] == "hf-3c"
    for Level in ("hf-3c engrad", "hf-3c opt", "! hf-3c freq", "hf-3c OptTS"):
        with pytest.raises(ValueError, match="single point"):
            getORCACalculator(LevelOfTheory=Level, PathToOrca="/nonexistent/orca")
