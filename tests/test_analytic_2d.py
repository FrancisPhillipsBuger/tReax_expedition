#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Tests for the 2D model potentials behind run_2d, run_muller and run_two_paths:
agreement with their definitions, the analytic gradients, the stationary
points and frequencies, the exact committor, the samplers and the scan.

Created on: Sun Sep 27 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import random

import numpy as np
import pytest

from constants.units import BetaEff, ConvCoeffForce, wvn2eV
from analytic_2d import (BuildGrid, CompressPath, GetPotential, PotentialEnergy,
                         Potentials, ReferenceAtSaddles, RunParameterScan,
                         RunSingleMCTrajectory, SolveCommittor, vbath)
from utils.mc.system import AnalyticSystem
from utils.mc.walk import Walk

Names = sorted(Potentials)

#%% 1. Potentials
def test_the_class_reproduces_the_proton_transfer_function():

    """The class wraps PotentialEnergy, which the samplers used before."""

    P = GetPotential("2d")
    for R in ([0.3, -0.2], [-0.61, -0.42], [0.0, 0.0]):
        assert P(R) == pytest.approx(PotentialEnergy(R), rel=1e-14, abs=1e-10)

def test_the_committor20_potentials_match_the_plumed_colvar():

    """First lines of the committor_2.0 unbiased COLVAR files, energy column."""

    Muller = GetPotential("muller", Walls=False)
    assert Muller.Native([-0.25, 1.75]) == pytest.approx(6.431815, abs=1e-5)
    assert Muller.Native([-0.502789, 1.512029]) == pytest.approx(0.283625, abs=1e-5)

    TwoPaths = GetPotential("two_paths", Walls=False)
    assert TwoPaths.Native([-1.0, 0.0]) == pytest.approx(-0.559344, abs=1e-5)
    assert TwoPaths.Native([-1.03087, -0.354777]) == pytest.approx(-2.36083, abs=1e-4)

@pytest.mark.parametrize("Name", Names)
def test_the_gradient_matches_finite_differences(Name):

    """Walls included, at points on both sides of them."""

    P = GetPotential(Name)
    (x0, x1), (y0, y1) = P.GridLimits
    Rng = np.random.default_rng(0)
    R = np.stack([Rng.uniform(x0, x1, 50), Rng.uniform(y0, y1, 50)], axis=-1)
    h = 1.0e-6
    FD = np.stack([(P.Total(R + [h, 0]) - P.Total(R - [h, 0]))/(2*h),
                   (P.Total(R + [0, h]) - P.Total(R - [0, h]))/(2*h)], axis=-1)

    assert np.allclose(P.TotalGradient(R), FD, rtol=1e-6, atol=1e-4*np.abs(FD).max())

@pytest.mark.parametrize("Name", Names)
def test_one_call_returns_cm1_and_tau_maps_onto_kt(Name):

    """The samplers multiply a call by wvn2eV, so it must return cm-1."""

    P = GetPotential(Name)
    R = np.array([0.1, 0.4])

    assert P(R)*wvn2eV == pytest.approx(float(P.Energy(R)), rel=1e-12)
    assert P.kTAt(P.TauAtkT(2.0)) == pytest.approx(2.0, rel=1e-12)
    assert BetaEff(P.DefaultTau)*P.kTAt(P.DefaultTau)*P.Unit*wvn2eV == pytest.approx(1.0, rel=1e-12)

#%% 2. Stationary points
def test_proton_transfer_stationary_points_and_bath_frequency():

    """Symmetric minima, the saddle at the origin, the bath mode at vbath."""

    P = GetPotential("2d")
    A, B = P.Minima()
    S, = P.Saddles()

    assert np.allclose(A, -B, atol=1e-8)
    assert np.allclose(S, 0.0, atol=1e-10)
    # without the coupling the bath curvature alone gives vbath
    H = P.Hessian(S)*P.Unit
    assert np.sqrt(H[1, 1]/(ConvCoeffForce*P.DefaultMass)) == pytest.approx(vbath, rel=1e-6)
    Nu, _ = P.NormalModes(S, P.DefaultMass)
    assert Nu[0] < 0.0 < Nu[1]

def test_mueller_brown_stationary_points():

    """The textbook points, scaled by 0.15 and shifted to put A at zero."""

    P = GetPotential("muller")
    A, B, I = P.Minima()
    S1, S2 = P.Saddles()

    assert np.allclose(A, [-0.5582, 1.4417], atol=1e-3)
    assert np.allclose(B, [0.6235, 0.0280], atol=1e-3)
    assert np.allclose(I, [-0.0500, 0.4667], atol=1e-3)
    assert P.Total(S1) - P.Total(A) == pytest.approx(15.905, abs=1e-2)
    assert P.Total(S2) - P.Total(A) == pytest.approx(11.168, abs=1e-2)

def test_two_paths_has_one_saddle_per_channel():

    """The lower channel is the easier one, with an almost flat saddle."""

    P = GetPotential("two_paths")
    A = P.Minima()[0]
    S1, S2 = P.Saddles()

    assert S1[1] > 0.0 > S2[1]
    assert P.Total(S1) - P.Total(A) == pytest.approx(16.445, abs=1e-2)
    assert P.Total(S2) - P.Total(A) == pytest.approx(12.359, abs=1e-2)

#%% 3. Exact committor
@pytest.mark.parametrize("Name, Expected", [("2d", (0.5,)),
                                            ("muller", (0.3675, 0.9956)),
                                            ("two_paths", (0.4380, 0.5112))])
def test_exact_committor_at_the_saddles(Name, Expected):

    """One half by symmetry on the proton-transfer model; grid converged otherwise."""

    P = GetPotential(Name)
    x, y, V = BuildGrid(P, NX=121, NY=121)
    A, B = P.Minima()[:2]
    Q = SolveCommittor(x, y, V, P.DefaultTau, BasinRadius=P.BasinRadius,
                       Centres=(tuple(A), tuple(B)))

    assert not np.isnan(Q).any()
    assert np.allclose(ReferenceAtSaddles(P, x, y, Q), Expected, atol=2e-3)

#%% 4. Samplers
@pytest.mark.parametrize("Name", Names)
def test_the_walker_stays_in_the_reactant_basin(Name):

    """A barrier of 8 to 16 kT_eff is not crossed in a short unbiased walk."""

    P = GetPotential(Name)
    random.seed(3)
    R = Walk(AnalyticSystem(P, P.DefaultMass), P.Minima()[0], P.DefaultTau, 3000)

    assert 0.05 < R["Acceptance"] < 0.8
    assert R["ELast"] == pytest.approx(P(R["Last"])*wvn2eV, rel=1e-12)
    assert np.all(P.Total(R["X"]) < P.Total(P.Saddles()).min())

def test_the_two_walker_trajectory_records_true_energies(tmp_path):

    """Every recorded energy is the potential at the recorded point."""

    P = GetPotential("two_paths")
    A, B = P.Minima()
    X, Y, E, Acc, Scr, NTrue, Converged = RunSingleMCTrajectory(
        A, B, 4.0, P.TauAtkT(5.0), 0.1, NMax=3000, FolderToSave=str(tmp_path),
        ScaleThreshold=0.5, Seed=2, Potential=P)

    assert Converged
    assert np.allclose(E[::50], [P([a, b]) for a, b in zip(X[::50], Y[::50])], rtol=1e-12)

def test_path_binning_along_x_is_the_proton_transfer_binning():

    """With the x axis the bins are centred on the N points from A_x to B_x."""

    A, B = np.array([-0.6, -0.4]), np.array([0.6, 0.4])
    X = np.linspace(-0.65, 0.65, 27)
    s, x, y, e, n = CompressPath(A, B, X, 0.3*X, X**2, N=5, Axis=(1.0, 0.0))

    assert np.allclose(s, np.linspace(0.0, 1.0, 5))
    assert n.sum() == len(X)
    assert np.allclose(x[:, 0], np.linspace(-0.6, 0.6, 5), atol=0.05)

@pytest.mark.parametrize("Name", Names)
def test_the_scan_writes_every_column(Name, tmp_path):

    """Thirteen trajectory columns, then kT and one column per stationary point."""

    P = GetPotential(Name)
    RunParameterScan(P, P.DefaultMass, [P.DefaultTau/4], [1.0], NumTrials=2, NMax=300,
                     FolderToSave=str(tmp_path), Seed=1)
    Rows = np.loadtxt(tmp_path/"statistics.res", ndmin=2)
    Labels, _ = P.StationaryPoints()

    assert Rows.shape == (1, 14 + len(Labels))
    assert Rows[0, -len(Labels):].sum() == pytest.approx(100.0)
    assert (tmp_path/"trials.dat").exists()

#%% 5. committor_2.0 data
