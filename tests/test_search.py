#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Tests for the staged search: the principles it rests on (energies only, no
reference structure read, GFN2-xTB only), the kernel, the quench, the states,
the label bisection, the continuation chain and its certificate, MBAR and the
binomial likelihood of shots.

Created on: Mon Sep 28 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import builtins
import random

import numpy as np
import pytest
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT

from analytic_2d import GetPotential
from analytic_2d.pipeline import AnalyticSpec
from constants.units import wvn2eV
from utils.mc.kernel import DelayedAcceptanceStep, WalkerState
from utils.mc.quench import Quench, QuenchSchedule
from utils.mc.system import AnalyticSystem, AtomsSystem
from utils.path.continuation import BuildChain, ChainSettings
from utils.path.label import LabelPath
from utils.path.search import Bottleneck, RunSearch, SearchSettings
from utils.path.states import BondGraph, DihedralBox, Region, StateBook

#%% 1. Principles
class EnergyOnly(Calculator):

    """EMT that refuses to hand out forces."""

    implemented_properties = ["energy", "forces"]

    def __init__(self):
        super().__init__()
        self.Inner = EMT()

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        if "forces" in properties:
            raise AssertionError("a force was requested")
        super().calculate(atoms, properties, system_changes)
        a = self.atoms.copy()
        a.calc = self.Inner
        self.results["energy"] = a.get_potential_energy()

def test_the_molecular_system_never_asks_for_forces():

    """Energy, proposal and quench of a molecule use energies only."""

    from ase import Atoms
    A = Atoms("Cu3", positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [1.2, 2.1, 0.0]])
    S = AtomsSystem(A, EnergyOnly())
    random.seed(1)
    R = Quench(S, A.get_positions().reshape(-1), 2.0, Schedule=QuenchSchedule(Steps=40))

    assert np.isfinite(R.Energy) and R.Energy <= S.Energy(A.get_positions().reshape(-1)) + 1e-9

def test_the_2d_search_needs_no_gradient_and_no_stationary_points(monkeypatch, tmp_path):

    """A whole search on a 2D potential with every derivative and saddle routine disabled."""

    P = GetPotential("2d")

    def Refuse(*Args, **Kwargs):
        raise AssertionError("a derivative or a stationary point was used")

    for Name in ("TotalGradient", "NativeGradient", "Hessian", "Minima", "Saddles"):
        monkeypatch.setattr(type(P), Name, Refuse)
    Spec = AnalyticSpec("2d")
    Settings = SearchSettings(NSeeds=2, MaxRungs=2, ExploreSteps=400, LabelRuns=2, MaxChains=1,
                              Chain=ChainSettings(NSlices=6, Schedule=QuenchSchedule(Steps=40)),
                              EndSchedule=QuenchSchedule(Steps=100), CrossingVotes=1, CrossingRefine=1)
    Report = RunSearch(Spec, Settings, Folder=str(tmp_path), NWorkers=1, Log=lambda s: None)

    assert Report["Stages"]["S1"]["TauExplore"] is not None

@pytest.mark.reaction("hcn")
def test_the_search_never_opens_a_reference_structure(monkeypatch, tmp_path):

    """ts_guess.xyz is for grading; the search must not read it."""

    from treax_expedition.pipeline import MoleculeSpec
    Opened = []
    Open = builtins.open

    def Watch(File, *Args, **Kwargs):
        Opened.append(str(File))
        return Open(File, *Args, **Kwargs)

    monkeypatch.setattr(builtins, "open", Watch)
    Spec = MoleculeSpec("hcn")
    Spec.Endpoints()
    Spec.BuildSystem(Worker="t")

    assert not any("ts_guess" in f for f in Opened)

def test_the_search_uses_gfn2_only():

    """Only GFN2-xTB is accepted as the tight binding method."""

    from treax_expedition.xtb import normaliseMethod
    for Name in ("gfn1", "ipea1", "gfn-ff"):
        with pytest.raises(ValueError):
            normaliseMethod(Name)

#%% 2. Kernel and quench
def test_the_kernel_keeps_the_invariant_distribution():

    """A 1D harmonic well sampled with the kernel has the Boltzmann variance of tau."""

    from constants.units import BetaEff
    random.seed(3)
    tau = 20.0
    k = 1.0
    State = WalkerState(np.zeros(1), 0.0)
    X = []
    for _ in range(0, 60000):
        R = DelayedAcceptanceStep(State, lambda x: x + np.array([random.gauss(0.0, 0.3)]),
                                  lambda x: 0.5*k*float(x[0]**2), tau)
        State = R.State
        X.append(State.Config[0])

    assert np.var(X[5000:]) == pytest.approx(1.0/(BetaEff(tau)*k), rel=0.1)

def test_a_quench_ends_in_the_basin_it_starts_in():

    """Started on the reactant side of the proton transfer, the quench ends at the reactant minimum."""

    P = GetPotential("2d")
    S = AnalyticSystem(P, P.DefaultMass)
    random.seed(2)
    R = Quench(S, np.array([-0.3, -0.1]), 1.0)

    assert R.X[0] < -0.5 and R.Energy < S.Energy(np.array([-0.3, -0.1]))

#%% 3. States and labels
def test_the_bond_graph_sees_h2():

    """The additive tolerance has to count the H-H bond of H2 as a bond."""

    B = BondGraph([1, 1])
    assert B.Bonds(np.array([0.0, 0.0, 0.0, 0.80, 0.0, 0.0])) == ((0, 1),)
    assert B.Bonds(np.array([0.0, 0.0, 0.0, 1.20, 0.0, 0.0])) == ()

def test_the_dihedral_convention_is_iupac():

    """A dihedral of +60 degrees must come out positive."""

    X = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                  [np.cos(np.radians(60.0)), np.sin(np.radians(60.0)), 1.0]])
    Box = DihedralBox([(0, 1, 2, 3)])

    assert Box.Key(X.reshape(-1))[0] == pytest.approx(60.0, abs=0.1)

def test_the_book_merges_a_state_and_names_a_new_one():

    """Within the tolerance a state is found again, beyond it a new one is registered."""

    Book = StateBook(Region(), Tolerance=0.1)
    Book.Register(np.array([0.0, 0.0]), 1.0, Name="A")

    assert Book.Register(np.array([0.05, 0.0]), 0.5) == "A"
    assert Book.Get("A")["Energy"] == 0.5
    assert Book.Register(np.array([1.0, 0.0]), 0.0) == "I1"

def test_the_bisection_finds_every_change_of_label():

    """A path A...A I...I B...B gives two changes, each between adjacent frames."""

    Labels = ["A"]*7 + ["I"]*5 + ["B"]*9
    Path = [np.array([float(i)]) for i in range(len(Labels))]
    Found, _ = LabelPath(Path, lambda x: Labels[int(x[0])], "A", "B")

    assert [(t.Lo, t.Hi, t.From, t.To) for t in Found] == [(6, 7, "A", "I"), (11, 12, "I", "B")]

def test_the_bottleneck_route_takes_the_lower_maximum():

    """Through I the highest bound is 5; directly it is 7."""

    Bound, Route = Bottleneck({("A", "I"): 5.0, ("I", "B"): 3.0, ("A", "B"): 7.0}, "A", "B")

    assert Bound == 5.0 and Route == ["A", "I", "B"]

#%% 4. Continuation
def test_a_chain_on_the_proton_transfer_bounds_the_saddle_from_above():

    """The certified bound of a chain along x lies at or just above the saddle."""

    P = GetPotential("2d")
    S = AnalyticSystem(P, P.DefaultMass)
    Book = StateBook(Region(), Tolerance=P.BasinRadius)
    random.seed(4)
    XA = Quench(S, np.array([-0.6, -0.4]), 1.0).X
    XB = Quench(S, np.array([0.6, 0.4]), 1.0).X
    Book.Register(XA, S.Energy(XA), Name="A")
    Book.Register(XB, S.Energy(XB), Name="B")
    C = BuildChain(S, lambda X: float(X[0]), Book, np.array([0.05, 0.02]), XA, XB, ("A", "B"), 16.0,
                   Settings=ChainSettings(NSlices=12))
    Saddle = P.Total(np.zeros(2))*P.Unit*wvn2eV

    assert C.Certified
    assert C.EUp >= Saddle - 1.0e-4
    assert C.EUp - Saddle < 0.005

#%% 5. Committor data
def test_mbar_recovers_the_unbiased_weights_of_two_windows():

    """Two harmonic windows on a flat line pool into uniform weights."""

    from utils.committor.data import MBARWeights
    rng = np.random.default_rng(0)
    x0 = rng.normal(-1.0, 0.5, 4000)
    x1 = rng.normal(1.0, 0.5, 4000)
    x = np.concatenate([x0, x1])
    Reduced = np.array([0.5*(x + 1.0)**2/0.25, 0.5*(x - 1.0)**2/0.25])
    w, f = MBARWeights(Reduced, [4000, 4000])
    Hist, _ = np.histogram(x, bins=6, range=(-1.2, 1.2), weights=w)

    # the unbiased distribution on a flat line is uniform, and the two
    # symmetric windows have the same free energy
    assert Hist.max()/Hist.min() < 1.3
    assert f[1] == pytest.approx(0.0, abs=0.1)

def test_the_binomial_loss_is_lowest_at_the_true_committor():

    """For shots drawn with probability p the likelihood peaks at q = p."""

    import torch
    from utils.committor import BinomialShotLoss, CommittorNN, Identity
    torch.manual_seed(0)
    Model = CommittorNN(Identity(1))
    X = torch.zeros((1, 1), dtype=torch.float64)
    Model = Model.double()
    Losses = []
    for Bias in (-1.0, 0.0, 1.0):
        with torch.no_grad():
            for p in Model.parameters():
                p.zero_()
            Model.Net[-1].bias.fill_(Bias)
        Losses.append(float(BinomialShotLoss(Model, X, torch.tensor([30.0], dtype=torch.float64),
                                             torch.tensor([30.0], dtype=torch.float64))))

    assert Losses[1] < Losses[0] and Losses[1] < Losses[2]

def test_each_segment_is_bounded_by_the_lower_of_its_two_paths():

    """A segment's bound is min(trajectory, straight links); the chain never exceeds the straight bound."""

    P = GetPotential("muller")
    S = AnalyticSystem(P, P.DefaultMass)
    Book = StateBook(Region(), Tolerance=P.BasinRadius)
    random.seed(5)
    XA = Quench(S, np.array([-0.56, 1.44]), 1.0).X
    XI = Quench(S, np.array([-0.05, 0.47]), 1.0).X
    Book.Register(XA, S.Energy(XA), Name="A")
    Book.Register(XI, S.Energy(XI), Name="I")
    C = BuildChain(S, lambda X: float(X[1]), Book, np.array([-0.8, 0.9]), XA, XI, ("A", "I"), 16.0,
                   Settings=ChainSettings(NSlices=10))
    Straight = max(max(C.Energies), max(C.LinkMax))

    assert len(C.Segments) == sum(1 for n in C.Nodes if n.Main) - 1
    for a, b, Line, Trajectory in C.Segments:
        assert a < b and np.isfinite(Line)
    assert C.EUp <= max(Straight, max(C.DescentMax)) + 1.0e-12
