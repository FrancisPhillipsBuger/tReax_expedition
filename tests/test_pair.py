#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Tests for the pair ensemble: the midpoint of a coupled pair represents twice
the bead tau, the midpoint weights make it exact for any coupling, the side
constraint gives the windowed Boltzmann distribution on the surface, and the
midpoint dumps change neither the trajectory nor the unweighted analysis.

Created on: Mon Sep 28 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import random

import numpy as np
import pytest
from scipy.special import ndtr

from analyse.barrier import BinnedProfile, LoadMidpoints
from analytic_2d.potential import ProtonTransfer
from analytic_2d.reference import BuildGrid
from analytic_2d.sampler import RunSingleMCTrajectory
from constants.units import BetaEff, KECoeff, PECoeff
from utils.mc.ensemble import EffectiveSampleFraction, NormalisedWeights, PairEnsemble
from utils.mc.pair import PairWalk
from utils.mc.system import AnalyticSystem, AtomsSystem
from utils.path.tse import DividingSurface, SampleTSE, TSESettings

#%% 1. Helpers
P = ProtonTransfer()

def Reference(tau, Window=None, XLim=(-1.5, 1.5), YLim=(-1.5, 1.5), N=401):

    """

    :return: <V>, P(|x| < 0.15), <x^2>, <y^2> of exp(-beta V) Window on a grid.

    """

    x, y, V = BuildGrid(P, XLim=XLim, YLim=YLim, NX=N, NY=N)
    XX, YY = np.meshgrid(x, y, indexing="ij")
    w = np.exp(-BetaEff(tau)*(V - V.min()))
    if Window is not None:
        w = w*Window(XX)
    w /= w.sum()

    return (float(np.sum(w*V)), float(np.sum(w*(np.abs(XX) < 0.15))),
            float(np.sum(w*XX**2)), float(np.sum(w*YY**2)))

def Run(Ensemble, Seed, Steps=40000, Start=((-0.5, 0.0), (-0.45, 0.0)), Side=None, StepTau=None):

    """

    :return: the PairWalk result on the proton transfer model.

    """

    random.seed(Seed)

    return PairWalk(AnalyticSystem(P, 1.0), np.array(Start[0]), np.array(Start[1]), Ensemble, Steps,
                    Ensemble.TauBead if StepTau is None else StepTau, Side=Side, Stride=4, Tune=2000)

TauMid = 1.0/(PECoeff*0.15)

#%% 2. The ensemble the midpoint represents
def test_the_log_weight_is_the_curvature_along_the_difference():

    """On a quadratic potential the log-weight is exactly beta_b D.H.D/4."""

    E = PairEnsemble(3.0, 1.0)
    H = np.array([[2.0, 0.3], [0.3, 0.5]])
    V = lambda X: 0.5*X @ H @ X + 0.1*X[0]
    R1 = np.array([0.2, -0.1])
    R2 = np.array([-0.05, 0.3])
    D = R1 - R2
    assert float(E.LogWeight(V(0.5*(R1 + R2)), V(R1), V(R2))) == \
        pytest.approx(E.BeadBeta*D @ H @ D/4.0, rel=1e-10)
    assert PairEnsemble.AtMidpoint(8.0, 0.5) == PairEnsemble(4.0, 2.0)
    assert E.MidpointBeta == pytest.approx(2.0*E.BeadBeta, rel=1e-12)

def test_the_midpoint_of_a_pair_is_at_twice_the_bead_tau():

    """A pair whose walkers both sample tau puts its midpoint at 2 tau, not tau."""

    R = Run(PairEnsemble(TauMid, TauMid), Seed=3)
    AtMid = Reference(2.0*TauMid)
    AtBead = Reference(TauMid)
    assert R["VM"].mean() == pytest.approx(AtMid[0], abs=0.01)
    assert abs(R["VM"].mean() - AtBead[0]) > 0.05

def test_beads_at_half_beta_put_the_midpoint_at_the_target():

    """Split beads with a stiff coupling sample exp(-beta V(M)) before and after weighting."""

    R = Run(PairEnsemble.AtMidpoint(TauMid, 1.0), Seed=3)
    Ref = Reference(TauMid)
    w = NormalisedWeights(R["LogW"])
    assert R["ESS"] > 0.95
    assert R["VM"].mean() == pytest.approx(Ref[0], abs=0.01)
    assert np.sum(w*R["VM"]) == pytest.approx(Ref[0], abs=0.01)
    assert np.sum(w*(np.abs(R["M"][:, 0]) < 0.15)) == pytest.approx(Ref[1], abs=0.006)

def test_the_weights_repair_a_soft_coupling():

    """With a soft coupling the raw midpoint is biased and the weighted one is not."""

    R = Run(PairEnsemble.AtMidpoint(TauMid, 8.0), Seed=3)
    Ref = Reference(TauMid)
    w = NormalisedWeights(R["LogW"])
    assert abs(np.mean(np.abs(R["M"][:, 0]) < 0.15) - Ref[1]) > 0.008
    assert np.sum(w*(np.abs(R["M"][:, 0]) < 0.15)) == pytest.approx(Ref[1], abs=0.006)
    assert np.sum(w*R["VM"]) == pytest.approx(Ref[0], abs=0.012)

#%% 3. The pair held across a surface
def test_a_pair_across_the_surface_samples_the_windowed_boltzmann():

    """
    The midpoint marginal is exp(-beta V(M)) Phi(-2|x|/sigma_D), sigma_D the
    width of D along x under the coupling, whatever the potential.
    """

    for T12 in (0.1, 1.0):
        E = PairEnsemble.AtMidpoint(P.DefaultTau, T12)
        sD = np.sqrt(E.Tau12/(2.0*KECoeff*1.0))
        Ref = Reference(E.TauMidpoint, Window=lambda X: ndtr(-2.0*np.abs(X)/sD),
                        XLim=(-0.6, 0.6), YLim=(-0.8, 0.8), N=501)
        R = Run(E, Seed=5, Start=((-0.05, 0.0), (0.05, 0.0)), Side=lambda X: X[0], StepTau=E.Tau12)
        w = NormalisedWeights(R["LogW"])
        assert np.all(R["M"][:, 0]**2 < 1.0)
        assert np.sum(w*R["VM"]) == pytest.approx(Ref[0], abs=0.004)
        assert np.sum(w*R["M"][:, 0]**2) == pytest.approx(Ref[2], rel=0.1)
        assert np.sum(w*R["M"][:, 1]**2) == pytest.approx(Ref[3], rel=0.1)

def test_the_tse_stage_starts_across_the_crossing_and_reports():

    """SampleTSE picks one node per side and returns the weighted ensemble."""

    Nodes = [np.array([x, 0.8*x]) for x in np.linspace(-0.6, 0.6, 13)]
    Side = DividingSurface(lambda X: float(X[0]), 0.0, Nodes[0])
    assert Side(Nodes[0]) < 0.0 < Side(Nodes[-1])
    random.seed(7)
    R = SampleTSE(AnalyticSystem(P, 1.0), lambda X: float(X[0]), np.zeros(2), Nodes, Nodes[0],
                  P.DefaultTau, TSESettings(T12=0.1, Steps=8000, Tune=1000, Stride=4))
    assert R["ESS"] > 0.9
    assert R["XiStar"] == 0.0
    assert np.all(np.abs(R["Xi"]) < 0.3)
    assert R["EMean"] == pytest.approx(0.28, abs=0.01)
    assert SampleTSE(AnalyticSystem(P, 1.0), lambda X: float(X[0]), np.zeros(2), Nodes[:5], Nodes[0],
                     P.DefaultTau) is None

#%% 4. Midpoint dumps and their analysis
def test_the_midpoint_dump_leaves_the_trajectory_unchanged(tmp_path):

    """The midpoint draws no random number, so the walkers do exactly the same."""

    Runs = []
    for k, Stride in enumerate((0, 5)):
        Folder = tmp_path / ("run%i" % (k,))
        Folder.mkdir()
        Runs.append(RunSingleMCTrajectory(np.array([-0.6, -0.42]), np.array([0.6, 0.42]), 1.0, 2.0, 1.0,
                                          NMax=400, FolderToSave=str(Folder), ScaleThreshold=0.0,
                                          Seed=11, Quiet=True, MidpointStride=Stride))
    assert Runs[0][0] == Runs[1][0] and Runs[0][1] == Runs[1][1] and Runs[0][2] == Runs[1][2]
    Rows = np.loadtxt(tmp_path / "run1" / "midpoint.dat")
    assert len(Rows) == 80
    assert Runs[1][5] - Runs[0][5] == len(Rows)
    assert not (tmp_path / "run0" / "midpoint.dat").exists()

def test_equal_weights_leave_the_profile_unchanged_and_the_log_parses(tmp_path):

    """A weighted profile with equal weights is the unweighted one; the header gives the ensemble."""

    rng = np.random.default_rng(0)
    IC = rng.random(300)
    En = np.sin(3.0*IC) + 0.1*rng.standard_normal(300)
    A = BinnedProfile(IC, En, N=7)
    B = BinnedProfile(IC, En, N=7, Weights=np.full(300, 1.0/300))
    for a, b in zip(A, B):
        assert np.allclose(a, b, rtol=1e-12, atol=1e-14)

    E = PairEnsemble(2.0, 6.0)
    with open(tmp_path / "midpoint_energy.log", "w") as f:
        f.write("# tau_bead,fs = 2 tau12,fs = 6\n# t,fs[$1] s[$2] V(M),eV[$3] V1,eV[$4] V2,eV[$5] logw[$6]\n")
        f.write(" 0.00 0.5 -1.0 -0.9 -0.95 0.0\n 20.00 0.6 -1.1 -1.0 -1.05 0.0\n")
    IC, VM, LogW, Ens = LoadMidpoints(str(tmp_path))
    assert Ens == E
    assert np.allclose(VM, [-1.0, -1.1])
    assert np.allclose(LogW, E.LogWeight(VM, [-0.9, -1.0], [-0.95, -1.05]))
    assert EffectiveSampleFraction(np.zeros(4)) == pytest.approx(1.0)

#%% 5. The frame of a molecular pair
class Dimer(AtomsSystem):

    """

    Two atoms of 1 amu on V = k (r - r0)^2, with the energy taken directly
    instead of through a calculator.

    """

    k, r0 = 0.5, 1.5

    def __init__(self):

        from ase import Atoms

        A = Atoms("H2", positions=[[0.0, 0.0, 0.0], [self.r0, 0.0, 0.0]])
        A.set_masses([1.0, 1.0])
        super().__init__(A, None)

    def Energy(self, X):

        self.NCalls += 1
        XYZ = np.asarray(X, dtype=float).reshape(-1, 3)

        return self.k*(np.linalg.norm(XYZ[0] - XYZ[1]) - self.r0)**2

def test_the_coupling_and_the_midpoint_live_in_one_frame():

    """The coupling distance ignores a rotation of the pair only as a whole."""

    S = Dimer()
    random.seed(1)
    X1 = np.array([0.1, 0.2, 0.0, 1.4, -0.1, 0.3])
    X2 = S.Rotated(S.Centred(X1), 0.4)
    assert S.Distance(X1, X2) == pytest.approx(0.0, abs=1e-8)
    assert S.FrameDistance(X1, X2) > 0.1
    Turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    Both = [(S.Centred(X).reshape(-1, 3) @ Turn).reshape(-1) for X in (X1, X2)]
    assert S.FrameDistance(*Both) == pytest.approx(S.FrameDistance(X1, X2))

def test_the_midpoint_of_a_molecular_pair_carries_no_orbit_volume():

    """
    <r> of the midpoint against the exact r^2 exp(-beta V) of a free dimer; a
    rotation-minimised coupling gives r^4 exp(-beta V), about 0.08 too long.
    """

    TauMid = 10.0
    g = np.linspace(1.0e-4, 6.0, 200001)
    p = g**2*np.exp(-BetaEff(TauMid)*Dimer.k*(g - Dimer.r0)**2)
    Exact = float(np.sum(g*p)/np.sum(p))

    S = Dimer()
    Got = []
    for Seed in (1, 2):
        random.seed(Seed)
        X0 = np.array([0.0, 0.0, 0.0, Dimer.r0, 0.0, 0.0])
        R = PairWalk(S, X0, X0 + 1.0e-3, PairEnsemble.AtMidpoint(TauMid, 0.2), 60000, 2.0,
                     Stride=10, Tune=2000)
        w = NormalisedWeights(R["LogW"])
        Got.append(np.sum(w*[np.linalg.norm(M.reshape(-1, 3)[0] - M.reshape(-1, 3)[1]) for M in R["M"]]))
        assert 0.2 < R["RotationAcceptance"] < 0.95

    assert np.mean(Got) == pytest.approx(Exact, abs=0.015)
