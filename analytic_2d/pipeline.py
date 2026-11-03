#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

The staged search on the analytic 2D potentials: the spec the shared driver
needs (energies, MCTSS runs, states, output) and the grading of its result
against the stationary points and the exact committor, which the search
itself never sees.

Created on: Mon Sep 28 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import os
from dataclasses import dataclass

import numpy as np

from constants.units import wvn2eV
from treax_expedition.io_xyz import OUTPUT_ROOT
from utils.mc.ladder import ExploreRun
from utils.mc.system import AnalyticSystem
from utils.path.search import SearchSettings
from utils.path.states import Region

from .potential import GetPotential
from .sampler import RunSingleMCTrajectory

#%% 1. Spec
@dataclass
class AnalyticSpec:

    """

    One analytic potential, as the staged search sees it.

    :param1 Name:        '2d', 'muller' or 'two_paths'.
    :param2 Temperature: temperature in K whose kT is one reduced unit.
    :param3 Walls:       add the harmonic walls of the PLUMED input.
    :param4 Mass:        mass of the particle in amu, None for the default.
    :param5 Threshold:   meeting threshold of the MCTSS runs.
    :param6 Ninter:      tau12/tau of the MCTSS runs.

    """

    Name: str
    Temperature: float = 300.0
    Walls: bool = True
    Mass: float = None
    Threshold: float = 0.5
    Ninter: float = 1.0

    @property
    def Potential(self):

        """

        :return: Potential: the Potential2D, built once per process.

        """

        if not hasattr(self, "_Potential"):
            self._Potential = GetPotential(self.Name, Temperature=self.Temperature, Walls=self.Walls)

        return self._Potential

    def __getstate__(self):

        """

        :return: State: the fields only; a worker rebuilds the potential.

        """

        return {k: v for k, v in self.__dict__.items() if k != "_Potential"}

    @property
    def ParticleMass(self):

        """

        :return: Mass: the mass in amu.

        """

        return self.Potential.DefaultMass if self.Mass is None else float(self.Mass)

    @property
    def Unit(self):

        """

        :return: Unit: eV per native energy unit of the potential.

        """

        return self.Potential.Unit*wvn2eV

    @property
    def UnitName(self):

        """

        :return: Name: the native energy unit.

        """

        return self.Potential.UnitName

    @property
    def Tolerance(self):

        """

        :return: Radius: largest distance, in angstroms, within one basin.

        """

        return self.Potential.BasinRadius

    @property
    def TauPhys(self):

        """

        :return: tau: the tau of the physical temperature, in fs; kT = 1 for
                 the committor_2.0 potentials, 10 fs for the proton transfer.

        """

        return float(self.Potential.DefaultTau)

    def kTAt(self, tau):

        """

        :return: kT: thermal energy of one walker sampling at tau, in native
                 units; a pair's midpoint at bead tau is at kTAt(2 tau).

        """

        return self.Potential.kTAt(tau)

    def DefaultSettings(self):

        """

        :return: Settings: SearchSettings for a 2D particle; energies cost
                 nothing here, so more paths are labelled and more chains
                 grown, which is what finds a second channel.

        """

        return SearchSettings(LabelRuns=12, MaxChains=8)

    def DefaultCommittorSettings(self):

        """

        :return: Settings: CommittorSettings at the scale of the 2D trainer.

        """

        from utils.committor.stage import CommittorSettings

        return CommittorSettings(Iterations=6, BasinSteps=20000, ExploreSteps=5000, Steps=20000,
                                 Pace=50, Stride=10, BatchSize=512)

    def BuildSystem(self, Worker=None):

        """

        :return: System: the AnalyticSystem.

        """

        return AnalyticSystem(self.Potential, self.ParticleMass)

    def Endpoints(self):

        """

        :return: X0, X1: the reactant and product basin guesses of the potential.

        """

        return (np.array(self.Potential.BasinGuesses[0], dtype=float),
                np.array(self.Potential.BasinGuesses[1], dtype=float))

    def Fingerprint(self):

        """

        :return: Fingerprint: Region.

        """

        return Region()

    def Descriptors(self):

        """

        :return: Descriptors: the coordinates themselves.

        """

        from utils.committor import Identity

        return Identity(2)

    def Explore(self, X0, X1, tau, Seed, Folder, Steps, Overlap):

        """

        One two-walker MCTSS run on the potential.

        :return: Run: ExploreRun, the path running from X0 to X1.

        """

        RunFolder = os.path.join(Folder, "tau_%.3f" % (tau,), "seed_%i" % (Seed,))
        os.makedirs(RunFolder, exist_ok=True)
        Report = {}
        X, Y, E, _, _, NTrue, _ = RunSingleMCTrajectory(
            np.array(X0, dtype=float), np.array(X1, dtype=float), self.ParticleMass, tau,
            self.Ninter, NMax=Steps, FolderToSave=RunFolder, ScaleThreshold=self.Threshold,
            Seed=Seed, Potential=self.Potential, StopOnOverlap=Overlap, Report=Report, Quiet=True)
        W1 = [np.array([x, y]) for x, y in zip(X[0::2], Y[0::2])]
        W2 = [np.array([x, y]) for x, y in zip(X[1::2], Y[1::2])]
        E1 = [e*wvn2eV for e in E[0::2]]
        E2 = [e*wvn2eV for e in E[1::2]]

        return ExploreRun(tau, Seed, Report["Converged"], Report["Reason"], [np.array(X0, dtype=float)] + W1 + W2[::-1] + [np.array(X1, dtype=float)],
                          [None] + E1 + E2[::-1] + [None], NTrue, RunFolder)

    def WriteStructure(self, FileName, X, Comment=""):

        """

        :return: None; writes FileName.dat holding one point.

        """

        with open(FileName + ".dat", "w") as f:
            f.write("# %s\n# x[$1] y[$2] E,%s[$3]\n" % (Comment, self.UnitName))
            f.write(" %10.5f %10.5f %14.6f\n" % (X[0], X[1], self.Potential.Total(np.asarray(X))))

        return None

    def WriteChain(self, FileName, C):

        """

        :return: None; writes FileName.dat with one node per row.

        """

        with open(FileName + ".dat", "w") as f:
            f.write("# chain %s along %s, ends %s-%s, certified %s, E_up %.8f eV\n"
                    % (C.Seed, C.Coordinate, C.Ends[0], C.Ends[1], C.Certified, C.EUp))
            f.write("# x[$1] y[$2] E,%s[$3] xi[$4] target[$5] link_max_to_next,%s[$6]\n"
                    % (self.UnitName, self.UnitName))
            for i, n in enumerate(C.Nodes):
                Link = C.LinkMax[i]/self.Unit if i < len(C.LinkMax) else np.nan
                f.write(" %10.5f %10.5f %14.6f %10.4f %10.4f %14.6f\n"
                        % (n.X[0], n.X[1], n.Energy/self.Unit, n.Xi, n.Target, Link))

        return None

#%% 2. Grading against the exact reference
def GradeSearch(Spec, Report):

    """

    Compare the certified bounds of a search with the saddles of the potential.

    The stationary points are found here with the analytic derivatives, which
    is why this lives outside the search: it only grades what the search found.

    :param1 Spec:   AnalyticSpec.
    :param2 Report: the dict RunSearch returned.

    :return: Lines: text lines of the comparison.

    """

    P = Spec.Potential
    Labels, Points = P.StationaryPoints()
    E = P.Total(Points)
    EA = P.Total(P.Minima()[0])
    Lines = ["[Grade] stationary points: " + ", ".join(
        "%s (%+.3f, %+.3f) %.3f %s" % (l, x[0], x[1], e - EA, P.UnitName)
        for l, x, e in zip(Labels, Points, E))]
    Saddles = {l: e for l, e in zip(Labels, E) if l.startswith("S")}
    for Pair in Report["Pairs"]:
        for Ch in Pair["Channels"]:
            EUp = Ch["EUp"]/Spec.Unit
            Nearest = min(Saddles, key=lambda l: abs(Saddles[l] - EUp))
            Line = ("[Grade] %s->%s %s: E_up %.4f vs %s %.4f (%+.4f %s)"
                    % (Pair["From"], Pair["To"], Ch["File"], EUp - EA, Nearest,
                       Saddles[Nearest] - EA, EUp - Saddles[Nearest], P.UnitName))
            if "Crossing" in Ch:
                Ec = Ch["Crossing"]["Forward"] + _StateOffset(Report, Pair["From"], Spec)
                Line += "; crossing %.4f (%+.4f)" % (Ec, Ec - (Saddles[Nearest] - EA))
            Lines.append(Line)

    return Lines

def _StateOffset(Report, Name, Spec):

    """

    :return: dE: energy of a state above the exact reactant minimum, native units.

    """

    EA = Spec.Potential.Total(Spec.Potential.Minima()[0])
    for s in Report["States"]:
        if s["Name"] == Name:
            return s["Energy"]/Spec.Unit - EA

    return 0.0

def GradeCommittor(Spec, Report, Folder):

    """

    Compare the committor of every pair that went through S4 with the exact
    committor between the two exact minima nearest to the pair's states.

    :param1 Spec:   AnalyticSpec.
    :param2 Report: the dict RunSearch returned.
    :param3 Folder: folder of the search, holding committor_<Sa>_<Sb>.

    :return: Lines: text lines of the comparison.

    """

    from utils.committor import AssessCommittor, CommittorOf, LoadCommittor

    from .reference import BuildGrid, ReferenceAtSaddles, SolveCommittor

    P = Spec.Potential
    Minima = P.Minima()
    Saddles = P.Saddles()
    x, y, V = BuildGrid(P)
    Lines = []
    for Pair in Report["Pairs"]:
        File = os.path.join(Folder, "committor_%s_%s" % (Pair["From"], Pair["To"]), "committor.pt")
        if not os.path.isfile(File):
            continue
        Centres = []
        for Name in (Pair["From"], Pair["To"]):
            X = np.asarray(_StateX(Folder, Name), dtype=float)
            Centres.append(tuple(Minima[int(np.argmin(np.sum((Minima - X)**2, axis=1)))]))
        Q = SolveCommittor(x, y, V, Spec.TauPhys, BasinRadius=P.BasinRadius, Centres=tuple(Centres))
        Model = LoadCommittor(File)
        EFloor = float(np.min(P.Energy(Saddles)))
        R = AssessCommittor(lambda X: CommittorOf(Model, X), x, y, V, Q, Spec.TauPhys, Saddles, EFloor)
        QS = ReferenceAtSaddles(P, x, y, Q)
        Lines.append("[Grade] committor %s->%s at kT %.3g %s: TSErr %.4f, BoltzErr %.2e; %s"
                     % (Pair["From"], Pair["To"], P.kTAt(Spec.TauPhys), P.UnitName, R["TSErr"], R["BoltzErr"],
                        ", ".join("q(S%i) %.3f exact %.3f" % (k + 1, R["qS%i" % (k + 1)], QS[k])
                                  for k in range(len(Saddles)))))

    return Lines

def _StateX(Folder, Name):

    """

    :return: X: the minimum the search recorded for a state.

    """

    with open(os.path.join(Folder, "state_%s.dat" % (Name,))) as f:
        Row = [l for l in f if not l.startswith("#")][0].split()

    return [float(Row[0]), float(Row[1])]

#%% 3. Command line
def RunSearchCLI(Name):

    """

    Parse the command line and run the staged search on one potential.

    :param1 Name: '2d', 'muller' or 'two_paths'.

    :return: None.

    """

    import argparse
    from dataclasses import replace

    from utils.path.search import RunSearch

    p = argparse.ArgumentParser(description="Staged gradient-free transition state search on the %s potential"
                                % (Name,))
    p.add_argument('--search', action='store_true', help='Run the staged search (the default of this entry).')
    p.add_argument('--outdir', type=str, default=os.path.join(OUTPUT_ROOT, Name, "search"),
                   help='Output folder. Default: reaction/output/<potential>/search.')
    p.add_argument('--nworkers', type=int, default=4, help='Worker processes. Default: 4.')
    p.add_argument('--seed', type=int, default=1, help='Base seed. Default: 1.')
    p.add_argument('--seeds_per_rung', type=int, default=4, help='MCTSS runs per tau rung. Default: 4.')
    p.add_argument('--explore_steps', type=int, default=5000, help='Steps per MCTSS run. Default: 5000.')
    p.add_argument('--max_chains', type=int, default=8, help='Continuation chains per pair of states. Default: 8.')
    p.add_argument('--committor', action='store_true',
                   help='Also learn and validate the committor of every pair at the physical tau (S4).')
    p.add_argument('--committor_iterations', type=int, default=None, help='S4 iterations. Default: 6.')
    p.add_argument('--tse', action='store_true',
                   help='Also sample the transition state ensemble at the physical tau by a pair held across the surface through each crossing (S6).')
    p.add_argument('--temperature', type=float, default=300.0,
                   help='muller and two_paths: temperature in K whose kT is one reduced unit. Default: 300.')
    p.add_argument('--no_walls', dest='walls', action='store_false',
                   help='muller and two_paths: leave out the harmonic walls of the PLUMED input.')
    p.add_argument('--mass', type=float, default=None, help='Mass of the particle in amu.')
    Args = p.parse_args()

    import torch
    torch.set_num_threads(1)
    Spec = AnalyticSpec(Name, Temperature=Args.temperature, Walls=Args.walls, Mass=Args.mass)
    Settings = Spec.DefaultSettings()
    Settings.Seed = Args.seed
    Settings.NSeeds = Args.seeds_per_rung
    Settings.ExploreSteps = Args.explore_steps
    Settings.MaxChains = Args.max_chains
    Settings.Committor = Args.committor
    Settings.TSE = Args.tse
    Settings.CommittorStage = replace(Spec.DefaultCommittorSettings(), Seed=Args.seed,
                                      **({} if Args.committor_iterations is None
                                         else {"Iterations": Args.committor_iterations}))
    Report = RunSearch(Spec, Settings, Folder=Args.outdir, NWorkers=Args.nworkers)
    Lines = GradeSearch(Spec, Report) + (GradeCommittor(Spec, Report, Args.outdir) if Args.committor else [])
    for Line in Lines:
        print(Line)
    with open(os.path.join(Args.outdir, "grade.txt"), "w") as f:
        f.write("\n".join(Lines) + "\n")

    return None
