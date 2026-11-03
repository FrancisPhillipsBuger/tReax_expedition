#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

The staged search shared by every system. S0 quenches the endpoints, S1 runs
MCTSS on a tau ladder, S2 labels the MCTSS paths by quenches, S3 grows and
certifies continuation chains through every change of label, S5 locates the
zero-temperature separatrix on the best chain of every channel, and S6, on
request, samples the transition state ensemble at the physical temperature by
a pair held across the surface through that crossing. A system only supplies
a picklable spec that builds its energy function and runs MCTSS.

Created on: Mon Sep 28 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import functools
import json
import multiprocessing
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field

import numpy as np

from utils.mc.ladder import ExploreLadder
from utils.mc.quench import Quench, QuenchSchedule

from .bisect import ChainCurvature, LocateCrossing
from .continuation import BuildChain, ChainSettings, Channels, MoveLength
from .coordinate import BondChange, DihedralChange, Latent, TrainLabelClassifier
from .label import DistinctFrames, IntervalLabels, LabelPath, LabelSchedule, Labeller
from .states import BondGraph, DihedralBox, StateBook
from .tse import SampleTSE, TSESettings

#%% 1. Settings
@dataclass
class SearchSettings:

    """

    :param1 Tau0:          first tau of the exploration ladder, in fs.
    :param2 Factor:        ratio of two rungs.
    :param3 NSeeds:        MCTSS runs per rung.
    :param4 MaxRungs:      largest number of rungs.
    :param5 ExploreSteps:  Monte-Carlo steps per MCTSS run.
    :param6 Overlap:       stop an MCTSS run once the walkers' progress overlaps.
    :param7 LabelRuns:     MCTSS paths labelled by quenches.
    :param8 MaxChains:     continuation chains per pair of states.
    :param9 Coordinate:    'auto', 'bond' or 'label'.
    :param10 Chain:        ChainSettings.
    :param11 EndSchedule:  quench of the endpoints.
    :param11b LabelSchedule: quench naming the state of a configuration.
    :param12 CrossingVotes: quenches per label of the separatrix bisection.
    :param13 CrossingRefine: bisections of the link holding the crossing.
    :param14 Seed:         base seed.
    :param15 Committor:    run the committor stage S4 on every pair.
    :param16 CommittorStage: CommittorSettings of S4.
    :param17 QuenchScale:  multiplier of every quench budget on top of the
                           scaling with the degrees of freedom.
    :param18 QuenchPatience: patience of every quench, or None to keep the
                           schedules' own.
    :param19 TSE:          run S6, the pair TSE at the physical tau, on every channel.
    :param20 TSEStage:     TSESettings of S6.

    """

    Tau0: float = 1.0
    Factor: float = 2.0
    NSeeds: int = 4
    MaxRungs: int = 8
    ExploreSteps: int = 5000
    Overlap: bool = True
    LabelRuns: int = 8
    MaxChains: int = 4
    Coordinate: str = "auto"
    Chain: ChainSettings = field(default_factory=ChainSettings)
    EndSchedule: QuenchSchedule = field(default_factory=lambda: QuenchSchedule(
        kT=(0.02, 0.008, 0.003, 0.001, 0.0005), Steps=400, Patience=4))
    LabelSchedule: QuenchSchedule = field(default_factory=lambda: LabelSchedule)
    CrossingVotes: int = 3
    CrossingRefine: int = 4
    Seed: int = 1
    Committor: bool = False
    CommittorStage: object = None
    QuenchScale: float = 1.0
    QuenchPatience: int = None
    TSE: bool = False
    TSEStage: TSESettings = field(default_factory=TSESettings)

#%% 2. Workers
# the system a worker process evaluates energies with, built once per process
_WORKER = {}

def InitWorker(Spec):

    """

    Build the system of a worker process.

    :param1 Spec: the picklable system spec.

    :return: None.

    """

    _WORKER["Spec"] = Spec
    _WORKER["System"] = Spec.BuildSystem(Worker=os.getpid())
    # one process per task already; threads inside a worker only contend, and
    # the committor models travel in double precision
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float64)
    except ImportError:
        pass

    return None

def _Apply(Packed):

    """

    :param1 Packed: (function, arguments).

    :return: the function's result.

    """

    Function, Args = Packed

    return Function(*Args)

class Pool:

    """

    Serial or process-parallel map over tasks that need the worker's system.

    """

    def __init__(self, Spec, NWorkers=1):

        """

        :param1 Spec:     the picklable system spec.
        :param2 NWorkers: processes; one runs everything in this process.

        """

        self.NWorkers = int(NWorkers)
        self.Executor = None
        if self.NWorkers > 1:
            # spawned, not forked: the tight binding library is threaded
            Context = multiprocessing.get_context("spawn")
            self.Executor = ProcessPoolExecutor(self.NWorkers, mp_context=Context,
                                                initializer=InitWorker, initargs=(Spec,))
        else:
            InitWorker(Spec)

    def Map(self, Function, Args):

        """

        :param1 Function: module-level callable.
        :param2 Args:     list of argument tuples.

        :return: Results: list of results, in order.

        """

        if self.Executor is None:
            return [Function(*a) for a in Args]

        return list(self.Executor.map(_Apply, [(Function, a) for a in Args]))

    def Close(self):

        """

        :return: None.

        """

        if self.Executor is not None:
            self.Executor.shutdown()

        return None

def KnownStates(Book):

    """

    :return: States: the records of a book, for a worker's own copy.

    """

    return [dict(s) for s in Book.States]

def LocalBook(Spec, Known):

    """

    :return: Book: a worker's copy of the book.

    """

    Book = StateBook(Spec.Fingerprint(), Spec.Tolerance, NewStates=getattr(Spec, "NewStates", True))
    Book.States = [dict(s) for s in Known]

    return Book

def TaskExplore(X0, X1, Folder, ExploreSteps, Overlap, tau, seed):

    """

    One MCTSS run in the worker.

    :return: Run: ExploreRun.

    """

    return _WORKER["Spec"].Explore(X0, X1, tau, seed, Folder, ExploreSteps, Overlap)

def TaskLabel(Path, First, Last, Known, StepTau, Schedule, Seed):

    """

    Label bisection of one MCTSS path in the worker.

    :return: Quenched: dict index -> (XMin, EMin) of every quenched frame, and
             the energy calls spent.

    """

    random.seed(Seed)
    System = _WORKER["System"]
    Calls0 = System.NCalls
    Book = LocalBook(_WORKER["Spec"], Known)
    L = Labeller(System, Book, StepTau, Schedule=Schedule)
    _, Labels = LabelPath(Path, L, First, Last)
    # the labels hold the two ends first, then every quenched index in the
    # order of its quench, which is the order of the one-vote records
    Order = [i for i in Labels if i not in (0, len(Path) - 1)]
    Quenched = {}
    for i, Rec in zip(Order, L.Records):
        Quenched[i] = (Rec["XMin"], Rec["EMin"], Rec["Converged"])

    return Quenched, System.NCalls - Calls0

def TaskChain(SeedX, XA, XB, States, Coordinate, StepTau0, Settings, Known, Seed, Name):

    """

    One continuation chain in the worker.

    :return: Chain, NCalls.

    """

    random.seed(Seed)
    System = _WORKER["System"]
    Calls0 = System.NCalls
    Book = LocalBook(_WORKER["Spec"], Known)
    C = BuildChain(System, Coordinate, Book, SeedX, XA, XB, States, StepTau0,
                   Settings=Settings, SeedName=Name)

    return C, System.NCalls - Calls0

def TaskTSE(Coordinate, Crossing, Nodes, XA, TauPhys, Settings, Seed):

    """

    Pair sampling of the TSE of one channel in the worker.

    :return: Result of SampleTSE (None when the chain never crosses the
             surface), and the energy calls spent.

    """

    random.seed(Seed)
    System = _WORKER["System"]
    Calls0 = System.NCalls
    R = SampleTSE(System, Coordinate, Crossing, Nodes, XA, TauPhys, Settings)

    return R, System.NCalls - Calls0

#%% 3. Helpers
def PickRuns(Rungs, TauExplore, N):

    """

    Up to N converged runs, coldest rung first and alternating between rungs,
    so the paths labelled come from more than one temperature.

    :return: Runs: list of ExploreRun.

    """

    ByRung = [[r for r in Runs if r.Converged] for tau, Runs in
              sorted(Rungs, key=lambda t: -t[0]) if tau <= TauExplore + 1.0e-12]
    Picked = []
    k = 0
    while len(Picked) < N and any(k < len(R) for R in ByRung):
        for R in ByRung:
            if k < len(R) and len(Picked) < N:
                Picked.append(R[k])
        k += 1

    return Picked

def FarthestPoints(System, Points, N):

    """

    Up to N mutually distant points, the first one kept.

    :return: Chosen: list of indices into Points.

    """

    if len(Points) == 0:
        return []
    Chosen = [0]
    Dist = np.array([System.Distance(P, Points[0]) for P in Points])
    while len(Chosen) < min(N, len(Points)):
        k = int(np.argmax(Dist))
        if Dist[k] <= 0.0:
            break
        Chosen.append(k)
        Dist = np.minimum(Dist, [System.Distance(P, Points[k]) for P in Points])

    return Chosen

def Bottleneck(Edges, Start, End):

    """

    Minimax path between two states: the route whose highest barrier bound is lowest.

    :param1 Edges: dict (Sa, Sb) -> upper bound, taken as undirected.
    :param2 Start: first state.
    :param3 End:   last state.

    :return: Bound, Route: the bound and the list of states, or (inf, []).

    """

    Nodes = set([Start, End])
    for a, b in Edges:
        Nodes.update([a, b])
    Best = {n: np.inf for n in Nodes}
    Prev = {}
    Best[Start] = -np.inf
    Open = set(Nodes)
    while len(Open) > 0:
        u = min(Open, key=lambda n: Best[n])
        Open.remove(u)
        if not np.isfinite(Best[u]) and Best[u] > 0:
            break
        for (a, b), w in Edges.items():
            for x, y in ((a, b), (b, a)):
                if x == u and y in Open and max(Best[u], w) < Best[y]:
                    Best[y] = max(Best[u], w)
                    Prev[y] = u
    if End not in Prev and End != Start:
        return np.inf, []
    Route = [End]
    while Route[-1] != Start:
        Route.append(Prev[Route[-1]])

    return float(Best[End]), Route[::-1]

#%% 4. The search
def RunSearch(Spec, Settings=None, Folder=".", NWorkers=1, Log=print):

    """

    Run S0-S3 and S5, and on request S4 and S6, on one system and write what
    was found.

    :param1 Spec:     system spec: BuildSystem, Endpoints, Explore,
                      Fingerprint, Tolerance, Descriptors, WriteStructure,
                      Unit and UnitName.
    :param2 Settings: SearchSettings, or None for Spec.DefaultSettings().
    :param3 Folder:   output folder.
    :param4 NWorkers: worker processes.
    :param5 Log:      callable printing one line.

    :return: Report: dict of everything reported, also written to search.json.

    """

    Settings = ScaleSchedules(Spec.DefaultSettings() if Settings is None else Settings,
                              len(Spec.BuildSystem(Worker="probe").Masses))
    os.makedirs(Folder, exist_ok=True)
    random.seed(Settings.Seed)
    Start = time.time()
    Workers = Pool(Spec, NWorkers)
    System = Spec.BuildSystem(Worker="main")
    Book = StateBook(Spec.Fingerprint(), Spec.Tolerance, NewStates=getattr(Spec, "NewStates", True))
    Unit, UnitName = Spec.Unit, Spec.UnitName
    Cost = {}
    Report = {"System": Spec.Name, "Settings": _Plain(asdict(Settings)), "Stages": {}}

    # S0: the endpoints drain into the two states the search connects
    Calls0 = System.NCalls
    X0, X1 = Spec.Endpoints()
    Ends = []
    for X, Name in ((X0, "A"), (X1, "B")):
        R = Quench(System, X, Settings.Tau0, Schedule=Settings.EndSchedule)
        Registered = Book.Register(R.X, R.Energy, Name=Name)
        Ends.append(R)
        Log("[S0] endpoint %s: E = %.6f eV, %s, %i energies"
            % (Registered, R.Energy, Book.Fingerprint.Describe(Book.Get(Registered)["Key"]), R.NCalls))
    if Book.Names() != ["A", "B"]:
        raise RuntimeError("Both endpoints drain into the same state; there is no reaction to search.")
    Cost["S0"] = System.NCalls - Calls0
    Report["Stages"]["S0"] = {"EA": Ends[0].Energy, "EB": Ends[1].Energy}

    # S1: MCTSS on a tau ladder, the coldest rung that still meets sets tau_explore;
    # tau is the bead tau of the pair, whose midpoint represents 2 tau
    ExploreFolder = os.path.join(Folder, "explore")
    RunOne = functools.partial(TaskExplore, Ends[0].X, Ends[1].X, ExploreFolder,
                               Settings.ExploreSteps, Settings.Overlap)
    TauX, Rungs = ExploreLadder(RunOne, Tau0=Settings.Tau0, Factor=Settings.Factor,
                                NSeeds=Settings.NSeeds, MaxRungs=Settings.MaxRungs,
                                BaseSeed=Settings.Seed*1000, Map=Workers.Map, Log=Log)
    Cost["S1"] = sum(r.NTrue for _, Runs in Rungs for r in Runs)
    Report["Stages"]["S1"] = {"TauExplore": TauX,
                              "TauMidpoint": None if TauX is None else 2.0*TauX,
                              "Rungs": [{"Tau": t, "Met": [r.Converged for r in R],
                                         "Reason": [r.Reason for r in R],
                                         "NTrue": [r.NTrue for r in R]} for t, R in Rungs]}
    if TauX is None:
        Workers.Close()
        raise RuntimeError("No MCTSS run joined the endpoints.")
    Log("[S1] tau_explore = %.3f fs per walker (midpoint kT = %.4g %s), %i energies"
        % (TauX, Spec.kTAt(2.0*TauX), UnitName, Cost["S1"]))

    # S2: zero-temperature labels along the paths, by bisection
    Runs = PickRuns(Rungs, TauX, Settings.LabelRuns)
    Paths = [DistinctFrames(r.Path)[0] for r in Runs]
    Out = Workers.Map(TaskLabel, [(P, "A", "B", KnownStates(Book), TauX/16.0, Settings.LabelSchedule,
                                   Settings.Seed*2000 + k) for k, P in enumerate(Paths)])
    Cost["S2"] = sum(n for _, n in Out)
    Transitions = []
    Labelled = []
    for r, P, (Quenched, _) in zip(Runs, Paths, Out):
        Labels = {0: "A", len(P) - 1: "B"}
        for i, (XMin, EMin, Conv) in Quenched.items():
            Labels[i] = Book.Register(XMin, EMin, Converged=Conv)
        Known = sorted(Labels)
        Found = [(a, b, Labels[a], Labels[b]) for a, b in zip(Known[:-1], Known[1:])
                 if b - a == 1 and Labels[a] != Labels[b]]
        Transitions.extend([(r, P, a, b, La, Lb) for a, b, La, Lb in Found])
        Labelled.append((P, IntervalLabels(len(P), Labels)))
        Log("[S2] tau %7.3f seed %6i: %4i frames, %2i quenches, %s"
            % (r.Tau, r.Seed, len(P), len(Quenched),
               " ".join("%s->%s@%i" % (La, Lb, a) for a, _, La, Lb in Found) or "no change"))
    Log("[S2] states: %s; %i energies" % (", ".join(Book.Names()), Cost["S2"]))

    # S3: chains through every pair of states that the labels connect
    Pairs = []
    for _, _, _, _, La, Lb in Transitions:
        if (La, Lb) not in Pairs and (Lb, La) not in Pairs:
            Pairs.append((La, Lb))
    Cost["S3"] = 0
    Edges = {}
    Objects = {}
    Report["Pairs"] = []
    ChainFolder = os.path.join(Folder, "chains")
    os.makedirs(ChainFolder, exist_ok=True)
    for Sa, Sb in Pairs:
        XA = Book.Get(Sa)["X"]
        XB = Book.Get(Sb)["X"]
        Seeds = [P[a] if La == Sa else P[b] for r, P, a, b, La, Lb in Transitions
                 if {La, Lb} == {Sa, Sb}]
        Chosen = FarthestPoints(System, Seeds, Settings.MaxChains)
        Coordinate = ChooseCoordinate(Spec, Book, Settings, Sa, Sb, Labelled, Log)
        Chains, n = GrowChains(Workers, Book, [Seeds[k] for k in Chosen], XA, XB, (Sa, Sb), Coordinate,
                               TauX, Settings, Unit, UnitName, Log)
        Cost["S3"] += n
        Committor = None
        if not any(C.Certified for C in Chains) and Settings.Committor:
            # no certified chain along xi: learn the committor from the seeds
            # and sweep along its latent instead
            from utils.committor.stage import RunCommittorStage
            Log("[S3] %s->%s: no certified chain along %s; S4 first, then S3 along z"
                % (Sa, Sb, getattr(Coordinate, "Name", "")))
            Committor = RunCommittorStage(Spec, System, Book, (Sa, Sb), Coordinate, Seeds[:4], Seeds, [],
                                          Spec.TauPhys, Workers, Settings.CommittorStage,
                                          os.path.join(Folder, "committor_%s_%s" % (Sa, Sb)), Log)
            Cost["S4"] = Cost.get("S4", 0) + Committor["Calls"]
            Coordinate = Latent(Committor["Model"], "z")
            More, n = GrowChains(Workers, Book, [Seeds[k] for k in Chosen], XA, XB, (Sa, Sb), Coordinate,
                                 TauX, Settings, Unit, UnitName, Log)
            Cost["S3"] += n
            Chains += More
        Best = Channels(System, Chains, 3.0*MoveLength(TauX))
        PairReport = {"From": Sa, "To": Sb, "Coordinate": getattr(Coordinate, "Name", ""),
                      "Chains": [ChainSummary(C, Book, Unit) for C in Chains], "Channels": []}
        Objects[(Sa, Sb)] = {"Best": Best, "Coordinate": Coordinate, "Committor": Committor}
        for k, C in enumerate(Best):
            Tag = "%s_%s_ch%i" % (Sa, Sb, k)
            Spec.WriteChain(os.path.join(ChainFolder, Tag), C)
            PairReport["Channels"].append(dict(ChainSummary(C, Book, Unit), File=Tag))
        if len(Best) > 0:
            Edges[(Sa, Sb)] = Best[0].EUp
        Report["Pairs"].append(PairReport)
    Log("[S3] %i energies" % (Cost["S3"],))

    # S5: the zero-temperature separatrix on the best chain of every channel
    Calls0 = System.NCalls
    Crossing = Labeller(System, Book, TauX/16.0, Schedule=Settings.LabelSchedule,
                        Votes=Settings.CrossingVotes)
    for PairReport in Report["Pairs"]:
        Best = Objects[(PairReport["From"], PairReport["To"])]["Best"]
        Objects[(PairReport["From"], PairReport["To"])]["Crossings"] = []
        for C, Summary in zip(Best, PairReport["Channels"]):
            X = LocateCrossing(System, C, Crossing, Refine=Settings.CrossingRefine)
            Objects[(PairReport["From"], PairReport["To"])]["Crossings"].append(X["X"])
            Curv = ChainCurvature(System, C)
            Sa, Sb = PairReport["From"], PairReport["To"]
            Summary["Crossing"] = {"Energy": X["Energy"],
                                   "Forward": (X["Energy"] - Book.Get(Sa)["Energy"])/Unit,
                                   "Backward": (X["Energy"] - Book.Get(Sb)["Energy"])/Unit,
                                   "Node": X["Node"], "t": X["t"]}
            Summary["Curvature"] = Curv
            Spec.WriteStructure(os.path.join(ChainFolder, Summary["File"] + "_crossing"), X["X"],
                                "q=1/2 crossing, E = %.8f eV" % (X["Energy"],))
            Log("[S5] %s->%s channel %s: E_up fwd %.4f bwd %.4f %s; crossing fwd %.4f bwd %.4f %s%s"
                % (Sa, Sb, Summary["File"], Summary["Forward"], Summary["Backward"], UnitName,
                   Summary["Crossing"]["Forward"], Summary["Crossing"]["Backward"], UnitName,
                   "" if Curv is None else "; nu %.0f cm-1, T_c %.0f K" % (Curv["nu"], Curv["Tc"])))
    Cost["S5"] = System.NCalls - Calls0

    # S4: the committor at the physical tau, validated by shots from the crossings
    if Settings.Committor:
        from utils.committor.stage import RunCommittorStage
        for PairReport in Report["Pairs"]:
            Sa, Sb = PairReport["From"], PairReport["To"]
            O = Objects[(Sa, Sb)]
            if len(O["Best"]) == 0:
                continue
            Tops = [C.Nodes[int(np.argmax(C.Energies))].X for C in O["Best"]]
            Nodes = [n.X for C in O["Best"] for n in C.Nodes]
            R = RunCommittorStage(Spec, System, Book, (Sa, Sb), O["Coordinate"], Tops, Nodes,
                                  O["Crossings"], Spec.TauPhys, Workers, Settings.CommittorStage,
                                  os.path.join(Folder, "committor_%s_%s" % (Sa, Sb)), Log)
            Cost["S4"] = Cost.get("S4", 0) + R["Calls"]
            PairReport["Committor"] = {"TauPhys": Spec.TauPhys, "History": R["History"],
                                       "Probes": R["Validation"]}
            for Summary, V in zip(PairReport["Channels"], R["Validation"]):
                Summary["Crossing"].update(q=V["q"], pB=V["pB"], pBErr=V["pBErr"])

    # S6: the transition state ensemble at the physical tau, by a pair held
    # across the surface of the chain coordinate through each crossing
    if Settings.TSE:
        Jobs = []
        for PairReport in Report["Pairs"]:
            Sa, Sb = PairReport["From"], PairReport["To"]
            O = Objects[(Sa, Sb)]
            for C, X, Summary in zip(O["Best"], O.get("Crossings", []), PairReport["Channels"]):
                Jobs.append(((O["Coordinate"], X, [n.X for n in C.Nodes], Book.Get(Sa)["X"],
                              Spec.TauPhys, Settings.TSEStage, Settings.Seed*3000 + len(Jobs)),
                             Summary, Sa, Sb))
        Out = Workers.Map(TaskTSE, [j[0] for j in Jobs])
        Cost["S6"] = sum(n for _, n in Out)
        for (Args, Summary, Sa, Sb), (R, _) in zip(Jobs, Out):
            if R is None:
                Log("[S6] %s->%s channel %s: no chain node on one side of the surface"
                    % (Sa, Sb, Summary["File"]))
                continue
            np.savez(os.path.join(ChainFolder, Summary["File"] + "_tse.npz"), M=R["M"], VM=R["VM"],
                     V1=R["V1"], V2=R["V2"], LogW=R["LogW"], Xi=R["Xi"], XiStar=R["XiStar"],
                     TauBead=R["Ensemble"].TauBead, Tau12=R["Ensemble"].Tau12)
            dXi = R["Xi"] - R["XiStar"]
            Summary["TSE"] = {"TauPhys": Spec.TauPhys, "T12": Settings.TSEStage.T12,
                              "N": int(len(R["VM"])), "ESS": R["ESS"],
                              "Acceptance": R["Acceptance"], "Screened": R["Screened"],
                              "EMean": R["EMean"], "XiStar": R["XiStar"],
                              "XiSpread": float(np.sqrt(np.sum(R["Weights"]*dXi**2))),
                              "NFailed": R["NFailed"]}
            Log("[S6] %s->%s channel %s: kT %.4g %s, <V(M)> - E(%s) %.4f %s, xi spread %.3g, ESS/N %.3f, "
                "acceptance %.2f, %i midpoints"
                % (Sa, Sb, Summary["File"], Spec.kTAt(Spec.TauPhys), UnitName, Sa,
                   (R["EMean"] - Book.Get(Sa)["Energy"])/Unit, UnitName, Summary["TSE"]["XiSpread"],
                   R["ESS"], R["Acceptance"], len(R["VM"])))

    # a state's minimum may still have dropped after S3 reported; every
    # barrier is quoted against the final minima
    for PairReport in Report["Pairs"]:
        Sa, Sb = PairReport["From"], PairReport["To"]
        for Summary in PairReport["Chains"] + PairReport["Channels"]:
            for Key, Name in (("Forward", Summary["Ends"][0]), ("Backward", Summary["Ends"][1])):
                if Name in Book.Names():
                    Summary[Key] = (Summary["EUp"] - Book.Get(Name)["Energy"])/Unit
            if "Crossing" in Summary:
                X = Summary["Crossing"]
                X["Forward"] = (X["Energy"] - Book.Get(Sa)["Energy"])/Unit
                X["Backward"] = (X["Energy"] - Book.Get(Sb)["Energy"])/Unit
            if "TSE" in Summary:
                Summary["TSE"]["Forward"] = (Summary["TSE"]["EMean"] - Book.Get(Sa)["Energy"])/Unit

    Bound, Route = Bottleneck(Edges, "A", "B")
    Report["AtoB"] = {"EUp": Bound, "Route": Route,
                      "Forward": (Bound - Book.Get("A")["Energy"])/Unit if np.isfinite(Bound) else None,
                      "Backward": (Bound - Book.Get("B")["Energy"])/Unit if np.isfinite(Bound) else None}
    Report["States"] = [{"Name": s["Name"], "Energy": s["Energy"],
                         "Fingerprint": Book.Fingerprint.Describe(s["Key"])} for s in Book.States]
    for s in Book.States:
        Spec.WriteStructure(os.path.join(Folder, "state_%s" % (s["Name"],)), s["X"],
                            "state %s, E = %.8f eV" % (s["Name"], s["Energy"]))
    Report["Cost"] = dict(Cost, Total=sum(Cost.values()))
    Report["WallSeconds"] = time.time() - Start
    Workers.Close()
    if np.isfinite(Bound):
        Log("[Result] A -> B via %s: certified E_up fwd %.4f bwd %.4f %s; energies %s = %i; %.0f s"
            % ("-".join(Route), Report["AtoB"]["Forward"], Report["AtoB"]["Backward"], UnitName,
               " + ".join("%s %i" % kv for kv in Cost.items()), Report["Cost"]["Total"],
               Report["WallSeconds"]))
    else:
        Log("[Result] no certified chain joins A and B")
    with open(os.path.join(Folder, "search.json"), "w") as f:
        json.dump(_Plain(Report), f, indent=1)

    return Report

def GrowChains(Workers, Book, Seeds, XA, XB, States, Coordinate, TauX, Settings, Unit, UnitName, Log):

    """

    Continuation chains from several seeds, in parallel, their ends named in
    the main book.

    :return: Chains, NCalls.

    """

    Sa, Sb = States
    Out = Workers.Map(TaskChain, [(X, XA, XB, States, Coordinate, TauX, Settings.Chain,
                                   KnownStates(Book), Settings.Seed*3000 + k, "seed%i" % (k,))
                                  for k, X in enumerate(Seeds)])
    Chains = []
    NCalls = 0
    for C, n in Out:
        NCalls += n
        # the worker named the ends in its own copy of the book
        C.Ends = tuple(Book.Register(X, E, Converged=c) for X, E, c in zip(C.EndX, C.EndEnergy, C.EndConverged))
        C.Certified = C.Ends == (Sa, Sb) and np.isfinite(C.EUp)
        Chains.append(C)
        Log("[S3] %s->%s %-6s along %s: %3i nodes, ends %s-%s, %s, E_up - E(%s) = %.4f %s, top at %s %i, %i energies"
            % (Sa, Sb, C.Seed, C.Coordinate, len(C.Nodes), C.Ends[0], C.Ends[1],
               "certified" if C.Certified else "NOT certified", Sa,
               (C.EUp - Book.Get(Sa)["Energy"])/Unit, UnitName, C.Top[0], C.Top[1], n))

    return Chains, NCalls

def ScaleSchedules(Settings, NumCoordinates):

    """

    Copy of the settings whose quench schedules grow with the degrees of freedom.

    :param1 Settings:       SearchSettings.
    :param2 NumCoordinates: degrees of freedom of the system.

    :return: Settings: the scaled copy.

    """

    from dataclasses import replace

    Kw = dict(Factor=Settings.QuenchScale, Patience=Settings.QuenchPatience)
    Chain = replace(Settings.Chain, Schedule=Settings.Chain.Schedule.Scaled(NumCoordinates, Cool=True, **Kw),
                    EndSchedule=Settings.Chain.EndSchedule.Scaled(NumCoordinates, **Kw))

    return replace(Settings, Chain=Chain,
                   EndSchedule=Settings.EndSchedule.Scaled(NumCoordinates, Cool=True, **Kw),
                   LabelSchedule=Settings.LabelSchedule.Scaled(NumCoordinates, **Kw))

def ChooseCoordinate(Spec, Book, Settings, Sa, Sb, Labelled, Log):

    """

    Progress coordinate of one pair of states: the change of the fingerprint
    between them (bond change for bond graphs, dihedral change for dihedral
    boxes) when the settings allow it, else the latent of a classifier of the
    zero-temperature labels of the two states.

    :return: Coordinate: callable X -> xi with a Name.

    """

    XA = Book.Get(Sa)["X"]
    XB = Book.Get(Sb)["X"]
    if Settings.Coordinate in ("auto", "bond") and isinstance(Book.Fingerprint, BondGraph):
        C = BondChange(Book.Fingerprint, XA, XB)
        if C.Available:
            Log("[S3] %s->%s: xi_bond, %s" % (Sa, Sb, C.Describe()))
            return C
        if Settings.Coordinate == "bond":
            raise RuntimeError("No bond changes between %s and %s." % (Sa, Sb))
    if Settings.Coordinate in ("auto", "bond") and isinstance(Book.Fingerprint, DihedralBox):
        C = DihedralChange(Book.Fingerprint, XA, XB)
        if C.Available:
            Log("[S3] %s->%s: xi_dihedral, %s" % (Sa, Sb, C.Describe()))
            return C

    import torch
    torch.set_default_dtype(torch.float64)
    X = []
    y = []
    for P, Names in Labelled:
        for x, n in zip(P, Names):
            if n in (Sa, Sb):
                X.append(np.asarray(x, dtype=float))
                y.append(1.0 if n == Sb else 0.0)
    X.extend([XA, XB])
    y.extend([0.0, 1.0])
    Model = TrainLabelClassifier(np.array(X), np.array(y), Spec.Descriptors(), Seed=Settings.Seed)
    C = Latent(Model, "xi_label")
    Log("[S3] %s->%s: xi_label from %i labelled frames, xi(%s) = %.2f, xi(%s) = %.2f"
        % (Sa, Sb, len(y), Sa, C(XA), Sb, C(XB)))

    return C

def ChainSummary(C, Book, Unit):

    """

    :return: Summary: plain numbers describing a chain.

    """

    Sa, Sb = C.Ends

    def Rel(Name):
        try:
            return (C.EUp - Book.Get(Name)["Energy"])/Unit
        except KeyError:
            return None

    return {"Seed": C.Seed, "Nodes": len(C.Nodes), "Certified": bool(C.Certified),
            "Ends": list(C.Ends), "EUp": C.EUp, "Forward": Rel(Sa), "Backward": Rel(Sb),
            "Top": list(C.Top), "NCalls": C.NCalls}

def _Plain(Obj):

    """

    :return: Obj: with numpy scalars and arrays turned into JSON types.

    """

    if isinstance(Obj, dict):
        return {str(k): _Plain(v) for k, v in Obj.items()}
    if isinstance(Obj, (list, tuple)):
        return [_Plain(v) for v in Obj]
    if isinstance(Obj, np.ndarray):
        return Obj.tolist()
    if isinstance(Obj, (np.floating, np.integer, np.bool_)):
        return Obj.item()
    if isinstance(Obj, float) and not np.isfinite(Obj):
        return None

    return Obj
