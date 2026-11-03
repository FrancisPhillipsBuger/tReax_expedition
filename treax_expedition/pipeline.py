#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

The staged search on a reaction under reaction/input: the spec the shared
driver needs. The energy back end, the state fingerprint and the physical
temperature come from the reaction's meta.json, the command line may
override them. Every calculator answers energies only.

Created on: Mon Sep 28 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import os
from dataclasses import dataclass

import ase.io as asio
import numpy as np

import scipy.constants as spc

from constants.units import PECoeff, eV2kJmol, kTEffeV
from utils.mc.ladder import ExploreRun
from utils.mc.system import AtomsSystem
from utils.path.search import SearchSettings
from utils.path.states import BondGraph, DihedralBox

from .io_xyz import INPUT_ROOT, readMeta, readXYZFile, writeXYZFile
from .sampler import doTwoStructureSWSMC

#%% 1. Spec
@dataclass
class MoleculeSpec:

    """

    One reaction, as the staged search sees it.

    :param1 Reaction:   reaction name under reaction/input, or a folder.
    :param2 Calculator: 'xtb', 'orca', 'amber' or 'emt'; None takes meta.json.
    :param3 Level:      ORCA level of theory; None takes meta.json or hf-3c.
    :param4 Basis:      ORCA basis set.
    :param5 Charge:     charge; None takes meta.json.
    :param6 Mult:       multiplicity; None takes meta.json.
    :param7 OrcaPath:   ORCA executable.
    :param8 NProcs:     threads of the tight binding back end.
    :param9 Screen:     screen trial moves on GFN2-xTB before the true energy.
    :param10 WorkDir:   folder for the ORCA working directories.
    :param11 T12:       tau12/tau of the MCTSS runs.
    :param12 Threshold: meeting threshold of the MCTSS runs.
    :param13 StepScale: multiplier of the free-particle step width of MCTSS.

    """

    Reaction: str
    Calculator: str = None
    Level: str = None
    Basis: str = ""
    Charge: int = None
    Mult: int = None
    OrcaPath: str = None
    NProcs: int = 1
    Screen: bool = False
    WorkDir: str = "."
    T12: float = 3.0
    Threshold: float = 1.0
    StepScale: float = 1.0

    def __post_init__(self):

        """

        Fill what the command line left open from the reaction's meta.json.

        """

        Meta = readMeta(self.Reaction)
        self.Meta = Meta
        if self.Calculator is None:
            self.Calculator = Meta.get("calculator", "xtb")
        if self.Level is None:
            self.Level = Meta.get("level", "hf-3c")
        if self.Charge is None:
            self.Charge = int(Meta.get("charge", 0))
        if self.Mult is None:
            self.Mult = int(Meta.get("mult", 1))
        if self.Calculator not in ("xtb", "orca", "amber", "emt"):
            raise ValueError("Unknown calculator %s" % (self.Calculator,))
        if self.Screen and self.Calculator != "orca":
            raise ValueError("The GFN2-xTB screen is meant for the ORCA back end.")

    @property
    def Name(self):

        """

        :return: Name: the reaction name.

        """

        return os.path.basename(os.path.normpath(self.Reaction))

    @property
    def Template(self):

        """

        :return: Atoms: the reactant, without calculator.

        """

        if not hasattr(self, "_Template"):
            self._Template = readXYZFile(self.Reaction)[0]

        return self._Template

    def __getstate__(self):

        """

        :return: State: the fields only; a worker rereads the input.

        """

        return {k: v for k, v in self.__dict__.items() if k != "_Template"}

    # energies are reported in kJ/mol, as in the MCTSS reference work
    Unit = 1.0/eV2kJmol
    UnitName = "kJ/mol"

    @property
    def Tolerance(self):

        """

        :return: Tolerance: largest fingerprint distance within one state.

        """

        FP = self.Meta.get("fingerprint", {})

        return float(FP.get("tolerance", 0.0))

    @property
    def TauPhys(self):

        """

        :return: tau: hbar/kT at the T_phys of meta.json (298.15 K if absent), in fs.

        """

        kT = spc.physical_constants["Boltzmann constant in eV/K"][0]*float(self.Meta.get("T_phys", 298.15))

        return 1.0/(PECoeff*kT)

    @property
    def NewStates(self):

        """

        :return: Allowed: whether states beyond A and B may be registered;
                 meta.json fingerprint "new_states", True if absent.

        """

        return bool(self.Meta.get("fingerprint", {}).get("new_states", True))

    def kTAt(self, tau):

        """

        :return: kT: thermal energy of one walker sampling at tau, in kJ/mol;
                 a pair's midpoint at bead tau is at kTAt(2 tau).

        """

        return kTEffeV(tau)*eV2kJmol

    def DefaultSettings(self):

        """

        :return: Settings: SearchSettings for a molecule; meta.json may set
                 "quench_scale", a multiplier of every quench budget, and
                 "quench_patience", the patience of every quench.

        """

        Patience = self.Meta.get("quench_patience")

        return SearchSettings(QuenchScale=float(self.Meta.get("quench_scale", 1.0)),
                              QuenchPatience=None if Patience is None else int(Patience))

    def DefaultCommittorSettings(self):

        """

        :return: Settings: CommittorSettings at the scale of the molecular trainer.

        """

        from utils.committor.stage import CommittorSettings

        # a single-atom move relaxes one atom at a time, so a shot needs more
        # steps to commit the more atoms there are
        Ratio = max(1.0, 3.0*len(self.Template)/18.0)

        # a molecular barrier is tens of kT at T_phys, so the biased walks
        # rarely reach the ridge and the shots are the only data there; with
        # them in the fit (Gamma 1, relative) q at the HCOH crossing moved
        # from 0.14 to 0.43 against pB 0.60 +- 0.11
        return CommittorSettings(Iterations=6, BasinSteps=4000, ExploreSteps=1500, Steps=12000,
                                 Pace=20, Stride=5, BatchSize=256, Gamma=1.0,
                                 ShotMaxSteps=int(round(2000*Ratio)))

    def MakeCalculator(self, Tag):

        """

        :param1 Tag: name of the working directory of an ORCA calculator.

        :return: Calculator: an ASE calculator answering energies only.

        """

        if self.Calculator == "xtb":
            from .xtb import getXTBCalculator
            return getXTBCalculator(charge=self.Charge, mult=self.Mult, nProcs=self.NProcs)
        if self.Calculator == "orca":
            from .orca import getORCACalculator
            return getORCACalculator(LevelOfTheory=(self.Level + " " + self.Basis).strip(),
                                     charge=self.Charge, mult=self.Mult, nProcs=1,
                                     PathToOrca=self.OrcaPath,
                                     Directory=os.path.join(self.WorkDir, "orca_%s" % (Tag,)))
        if self.Calculator == "amber":
            from .amber import getAmberCalculator
            return getAmberCalculator()
        from .calculators import getEMTCalculator

        return getEMTCalculator()

    def ScreenCalculator(self):

        """

        :return: Calculator: GFN2-xTB for the delayed-acceptance screen, or None.

        """

        if not self.Screen:
            return None
        from .xtb import getXTBCalculator

        return getXTBCalculator(charge=self.Charge, mult=self.Mult, nProcs=self.NProcs)

    def BuildSystem(self, Worker=None):

        """

        :return: System: the AtomsSystem of a process.

        """

        return AtomsSystem(self.Template, self.MakeCalculator("sys_%s" % (Worker,)),
                           Screen=self.ScreenCalculator())

    def Endpoints(self):

        """

        :return: X0, X1: flat coordinates of the reactant and the product.

        """

        S = readXYZFile(self.Reaction)
        if len(S) != 2:
            raise ValueError("%s holds %i structure(s); the search needs two endpoints."
                             % (self.Reaction, len(S)))

        return S[0].get_positions().reshape(-1), S[1].get_positions().reshape(-1)

    def Fingerprint(self):

        """

        :return: Fingerprint: the bond graph, or the dihedral box of meta.json.

        """

        FP = self.Meta.get("fingerprint", {"kind": "bonds"})
        if FP.get("kind", "bonds") == "dihedrals":
            return DihedralBox(FP["quads"])

        return BondGraph(self.Template.get_atomic_numbers(), Tolerance=float(FP.get("bond_tolerance", 0.4)))

    def Descriptors(self):

        """

        :return: Descriptors: all interatomic distances.

        """

        from utils.committor import PairwiseDistances

        return PairwiseDistances(len(self.Template))

    def Explore(self, X0, X1, tau, Seed, Folder, Steps, Overlap):

        """

        One two-walker MCTSS run of the reaction.

        :return: Run: ExploreRun, the path running from X0 to X1.

        """

        RunFolder = os.path.join(Folder, "tau_%.3f" % (tau,), "seed_%i" % (Seed,))
        os.makedirs(RunFolder, exist_ok=True)
        Walkers = []
        for n, X in enumerate((X0, X1)):
            W = self.Template.copy()
            W.set_positions(np.asarray(X, dtype=float).reshape(-1, 3))
            W.calc = self.MakeCalculator("explore_%.3f_%i_w%i" % (tau, Seed, n + 1))
            Walkers.append(W)
        Surrogate = None
        if self.Screen:
            from utils.surrogate import CalculatorSurrogate
            Surrogate = CalculatorSurrogate(self.ScreenCalculator())
        Report = {}
        NTrue = doTwoStructureSWSMC(Walkers, tau=tau, tau12=self.T12*tau, Nsteps=Steps,
                                    ScaleThreshold=self.Threshold, Seed=Seed, LogEvaluations=False,
                                    Surrogate=Surrogate, FolderToSave=RunFolder,
                                    StepScale=self.StepScale, StopOnOverlap=Overlap,
                                    Report=Report, Quiet=True)
        Paths = []
        Energies = []
        for n in (1, 2):
            Frames = asio.read(os.path.join(RunFolder, "sampled%i.trj" % (n,)), index=":", format="xyz")
            E = np.atleast_2d(np.loadtxt(os.path.join(RunFolder, "sampled_energy%i.log" % (n,))))[:, 1]
            Paths.append([F.get_positions().reshape(-1) for F in Frames])
            Energies.append(list(E[:len(Frames)]))

        return ExploreRun(tau, Seed, Report["Converged"], Report["Reason"],
                          Paths[0] + Paths[1][::-1], Energies[0] + Energies[1][::-1], NTrue, RunFolder)

    def WriteStructure(self, FileName, X, Comment=""):

        """

        :return: None; writes FileName.xyz.

        """

        A = self.Template.copy()
        A.set_positions(np.asarray(X, dtype=float).reshape(-1, 3))
        asio.write(FileName + ".xyz", A, format="xyz", comment=Comment)

        return None

    def WriteChain(self, FileName, C):

        """

        :return: None; writes FileName.xyz with one node per frame and
                 FileName.dat with the node energies.

        """

        if os.path.isfile(FileName + ".xyz"):
            os.remove(FileName + ".xyz")
        E0 = C.Nodes[0].Energy
        with open(FileName + ".dat", "w") as f:
            f.write("# chain %s along %s, ends %s-%s, certified %s, E_up %.8f eV\n"
                    % (C.Seed, C.Coordinate, C.Ends[0], C.Ends[1], C.Certified, C.EUp))
            f.write("# node[$1] xi[$2] target[$3] E,eV[$4] E-E(node 0),kJ/mol[$5] link_max_to_next,kJ/mol[$6]\n")
            for i, n in enumerate(C.Nodes):
                Link = (C.LinkMax[i] - E0)*eV2kJmol if i < len(C.LinkMax) else np.nan
                f.write(" %4i %10.5f %10.5f %16.8f %12.4f %12.4f\n"
                        % (i, n.Xi, n.Target, n.Energy, (n.Energy - E0)*eV2kJmol, Link))
                A = self.Template.copy()
                A.set_positions(n.X.reshape(-1, 3))
                writeXYZFile(A, FileName + ".xyz", Append=True)

        return None

#%% 2. Grading against the reference of meta.json
def GradeMolecule(Spec, Report):

    """

    Compare the certified bound of a search with the reference of meta.json:
    the energy of a reference transition state structure on the same back
    end, and published barriers when the back end is the one they were
    computed with. Only this function ever reads the reference.

    :param1 Spec:   MoleculeSpec.
    :param2 Report: the dict RunSearch returned.

    :return: Lines: text lines of the comparison.

    """

    Ref = Spec.Meta.get("reference", {})
    States = {s["Name"]: s["Energy"] for s in Report["States"]}
    Fwd, Bwd = Report["AtoB"]["Forward"], Report["AtoB"]["Backward"]
    Lines = []
    if Fwd is None:
        return ["[Grade] %s: no certified chain joins A and B" % (Spec.Name,)]
    Lines.append("[Grade] %s: certified E_up fwd %.2f bwd %.2f kJ/mol via %s"
                 % (Spec.Name, Fwd, Bwd, "-".join(Report["AtoB"]["Route"])))
    if "ts_structure" in Ref:
        TS = asio.read(os.path.join(INPUT_ROOT, Spec.Name, Ref["ts_structure"]))
        E = Spec.BuildSystem(Worker="grade").Energy(TS.get_positions().reshape(-1))
        if E is not None:
            Rel = (E - States["A"])*eV2kJmol
            Lines.append("[Grade] %s: reference structure %s: E - E(A) = %.2f kJ/mol on %s; E_up - E_ref = %+.2f kJ/mol"
                         % (Spec.Name, Ref["ts_structure"], Rel, Spec.Calculator, Fwd - Rel))
    if "forward_kJmol" in Ref and Spec.Calculator == Spec.Meta.get("calculator"):
        Lines.append("[Grade] %s: published fwd %.2f bwd %s kJ/mol (%s); E_up fwd %+.2f"
                     % (Spec.Name, Ref["forward_kJmol"], Ref.get("backward_kJmol", "-"),
                        Ref.get("source", ""), Fwd - Ref["forward_kJmol"]))
    for Pair in Report["Pairs"]:
        for Ch in Pair["Channels"]:
            if "Crossing" in Ch and "pB" in Ch["Crossing"]:
                Lines.append("[Grade] %s %s->%s %s: crossing q = %.3f, pB = %.3f +- %.3f"
                             % (Spec.Name, Pair["From"], Pair["To"], Ch["File"], Ch["Crossing"]["q"],
                                Ch["Crossing"]["pB"], Ch["Crossing"]["pBErr"]))

    return Lines
