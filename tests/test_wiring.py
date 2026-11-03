#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Tests for the paths that connect the packages to the samplers: the cost the
two-walker trajectory of the 2D model reports, the cheap screen on the
molecular walk, and the committor checkpoint that lets a trained model come
back as a bias.

What is checked here is that the wiring preserves the properties the pieces
were built to have, not that the pieces themselves work.

Created on: Thu Aug 27 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import numpy as np
import pytest
import torch
from ase.build import add_adsorbate, fcc111, molecule
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms

from utils.committor import (CommittorNN, Identity, KolmogorovBias, LoadCommittor,
                             PairwiseDistances, SaveCommittor)
from analytic_2d.potential import X0
from analytic_2d.sampler import RunSingleMCTrajectory
from treax_expedition.sampler import doSingleStructureSWSMC

#%% 1. Fixtures
def MakeSlab():

    """A constrained periodic slab with a single mobile adatom."""

    s = fcc111('Al', size=(2, 2, 2), vacuum=10.0)
    add_adsorbate(s, 'Au', 2.0, 'fcc')
    s.set_constraint(FixAtoms(mask=[a.symbol == 'Al' for a in s]))
    s.calc = EMT()

    return s

def RunTwoWalker(**Keywords):

    """One short two-walker trajectory from the two minima, in a temp folder."""

    Defaults = dict(mass=1.0, tau=5.0, Ninter=10.0, NMax=300,
                    ScaleThreshold=1.0, Seed=11)
    Defaults.update(Keywords)

    return RunSingleMCTrajectory(np.array([X0, 0.0]), np.array([-X0, 0.0]), **Defaults)

#%% 2. The two-walker trajectory
def test_the_cost_is_the_screened_trials(tmp_path):

    """

    The energies of the pre-screened trials are dumped because they are free
    on an analytic potential, but they are diagnostics and must not be counted
    as a cost the same run would have paid against a real potential.

    """

    X, Y, E, Acc, Scr, NTrue, Conv = RunTwoWalker(FolderToSave=str(tmp_path))

    assert len(Acc) == len(Scr)
    assert NTrue == sum(Scr)
    assert all(s for a, s in zip(Acc, Scr) if a)

def test_the_two_walker_run_is_reproducible(tmp_path):

    """A seeded run must not hide an unseeded draw."""

    A = RunTwoWalker(FolderToSave=str(tmp_path))
    B = RunTwoWalker(FolderToSave=str(tmp_path))

    assert A[0] == B[0] and A[5] == B[5]

#%% 3. The molecular walk
def test_the_single_structure_walk_reports_what_it_spent(tmp_path, monkeypatch):

    """

    Without a surrogate every step reaches the calculator, so the cost is the
    number of steps. It is returned rather than left to be counted from the
    dumps, because that is the number a method is judged on.

    """

    monkeypatch.chdir(tmp_path)
    s = MakeSlab()
    NTrue = doSingleStructureSWSMC(Structure=s, tau=30.0, Nsteps=15, Seed=3,
                                   LogEvaluations=False)

    assert NTrue == 15

def test_a_perfect_surrogate_spends_less_on_the_molecular_walk(tmp_path, monkeypatch):

    """The screen rejects cheaply what the residual would have rejected."""

    monkeypatch.chdir(tmp_path)
    Plain = doSingleStructureSWSMC(Structure=MakeSlab(), tau=60.0, Nsteps=60,
                                   Seed=5, LogEvaluations=False)
    Cheap = doSingleStructureSWSMC(Structure=MakeSlab(), tau=60.0, Nsteps=60,
                                   Seed=5, LogEvaluations=False,
                                   Surrogate=lambda S: S.get_potential_energy())

    assert Cheap < Plain

#%% 4. The committor checkpoint
def test_a_checkpoint_round_trips_an_identity_model(tmp_path):

    """

    Rebuilding the bias needs the descriptors and the slope as well as the
    weights, so the checkpoint has to carry them.

    """

    torch.manual_seed(0)
    Model = CommittorNN(Identity(2), Hidden=(8, 8))
    Path = str(tmp_path/"c.pt")
    SaveCommittor(Model, Path)
    Loaded = LoadCommittor(Path)

    X = torch.as_tensor(np.array([[0.3, 0.1], [-0.4, 0.0]]),
                        dtype=torch.get_default_dtype())
    with torch.no_grad():
        assert Loaded.Committor(X).numpy() == pytest.approx(Model.Committor(X).numpy())
    assert Loaded.Slope == Model.Slope

def test_a_checkpoint_round_trips_a_distance_model(tmp_path):

    """The molecular descriptor set has to survive the trip as well."""

    torch.manual_seed(0)
    Model = CommittorNN(PairwiseDistances(4), Hidden=(8,))
    Path = str(tmp_path/"c.pt")
    SaveCommittor(Model, Path)
    Loaded = LoadCommittor(Path)

    assert isinstance(Loaded.Descriptors, PairwiseDistances)
    assert Loaded.Descriptors.NumDescriptors == Model.Descriptors.NumDescriptors
    X = torch.as_tensor(np.random.RandomState(0).normal(size=(3, 12)),
                        dtype=torch.get_default_dtype())
    with torch.no_grad():
        assert Loaded.Committor(X).numpy() == pytest.approx(Model.Committor(X).numpy())

def test_the_checkpoint_preserves_the_layer_widths(tmp_path):

    """A bare state dictionary would leave these to be guessed."""

    torch.manual_seed(0)
    Model = CommittorNN(Identity(2), Hidden=(7, 5, 3))
    Path = str(tmp_path/"c.pt")
    SaveCommittor(Model, Path)
    Loaded = LoadCommittor(Path)

    Widths = [l.out_features for l in Loaded.Net if isinstance(l, torch.nn.Linear)]
    assert Widths == [7, 5, 3, 1]

def test_an_old_checkpoint_is_refused_rather_than_guessed(tmp_path):

    """A file with no architecture cannot be rebuilt, and must say so."""

    Path = str(tmp_path/"old.pt")
    torch.save({"state": {}}, Path)
    with pytest.raises(ValueError, match="no architecture"):
        LoadCommittor(Path)

def test_a_double_precision_checkpoint_loads_into_a_single_precision_session(tmp_path):

    """

    The committor stage trains in double precision; a process in the default
    single precision that loads the checkpoint has to be able to evaluate the
    bias, which fails on its first matrix multiply without the cast.

    """

    Previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(0)
        s = molecule('H2O')
        Model = CommittorNN(PairwiseDistances(len(s)), Hidden=(8,))
        Path = str(tmp_path/"c.pt")
        SaveCommittor(Model, Path)

        torch.set_default_dtype(torch.float32)
        Bias = KolmogorovBias(LoadCommittor(Path), np.repeat(s.get_masses(), 3), 20.0)

        assert np.isfinite(Bias(s.get_positions().reshape(-1)))
    finally:
        torch.set_default_dtype(Previous)
