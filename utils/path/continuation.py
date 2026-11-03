#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Gradient-free continuation chain between two states. A progress coordinate
is cut into slices; each node is a quench restrained to one slice, started
from the previous node, so the nodes follow one valley. Neighbouring nodes
are kept within one MCTSS move of each other, the links between them are
checked point by point, and the two ends are quenched freely to name the
states they belong to. The highest energy met is an upper bound of the
barrier along a connected path, certified when the ends are the two states. Between two slices the chain
is certified along whichever is lower of two connected paths: the straight
links through the nodes, or the quench trajectory itself, a sequence of
accepted single-atom moves from the node before to the node after.

Created on: Mon Sep 28 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
from dataclasses import dataclass, field

import numpy as np

from constants.units import SxCoeff
from utils.mc.quench import Quench, QuenchSchedule

#%% 1. Settings and result
@dataclass
class ChainSettings:

    """

    :param1 NSlices:    slices between the two states.
    :param2 Schedule:   quench schedule of one node.
    :param3 EndSchedule: schedule of the free quench at either end; it names
                        the end states, so it is as cold as the label quench.
    :param4 KFactor:    initial restraint, in eV per squared slice spacing.
    :param5 Tolerance:  largest miss of a slice, in slice spacings, before the
                        restraint is stiffened.
    :param6 MaxStiffen: how many times the restraint may be stiffened.
    :param7 MaxGap:     largest mass-weighted distance between two nodes, in
                        sqrt(amu) angstroms; None takes one MCTSS move.
    :param8 MaxDensify: how many times a link may be halved.
    :param9 LinkPoints: energies checked per MaxGap of link length.
    :param10 Screen:    screen quench proposals on the cheaper calculator.
    :param11 Ramp:      sub-steps over which the restraint centre moves from
                        the starting xi to the slice before the full quench,
                        so the walker follows the slices instead of being
                        yanked by a restraint half an eV away, which strains
                        bonds when only single atoms move.

    """

    NSlices: int = 24
    Schedule: QuenchSchedule = field(default_factory=lambda: QuenchSchedule(Patience=4))
    EndSchedule: QuenchSchedule = field(default_factory=lambda: QuenchSchedule(
        kT=(0.005, 0.002, 0.001, 0.0005), Steps=300, Patience=4))
    KFactor: float = 1.0
    Tolerance: float = 0.25
    MaxStiffen: int = 3
    MaxGap: float = None
    MaxDensify: int = 3
    LinkPoints: int = 4
    Screen: bool = False
    Ramp: int = 6

def MoveLength(StepTau):

    """

    Mass-weighted length of one single-atom MCTSS move of width parameter StepTau.

    :param1 StepTau: time parameter of the proposal width, in fs.

    :return: d: sqrt(3) sqrt(hbar tau/2), in sqrt(amu) angstroms.

    """

    return float(SxCoeff*np.sqrt(3.0*StepTau))

@dataclass
class Node:

    """

    One node of a chain.

    """

    X: np.ndarray
    Energy: float
    Target: float
    Xi: float
    PathMax: float = np.inf
    Main: bool = True

@dataclass
class Chain:

    """

    A continuation chain and its certificate.

    """

    Nodes: list
    Seed: str = ""
    Coordinate: str = ""
    LinkMax: list = field(default_factory=list)
    LinkLength: list = field(default_factory=list)
    Ends: tuple = (None, None)
    EndX: tuple = (None, None)
    EndEnergy: tuple = (np.nan, np.nan)
    EndConverged: tuple = (True, True)
    DescentMax: tuple = (-np.inf, -np.inf)
    Certified: bool = False
    EUp: float = np.inf
    Top: tuple = ("node", -1)
    Segments: list = field(default_factory=list)
    NCalls: int = 0

    @property
    def Energies(self):

        """

        :return: E: node energies, in eV.

        """

        return np.array([n.Energy for n in self.Nodes])

#%% 2. One node
def SliceNode(System, Coordinate, X, Target, Spacing, K, Settings, StepTau):

    """

    Quench restrained to one slice, stiffening the restraint until the node
    sits within the tolerance of its slice.

    :param1 System:     AnalyticSystem or AtomsSystem.
    :param2 Coordinate: callable X -> xi.
    :param3 X:          starting coordinates.
    :param4 Target:     centre of the slice.
    :param5 Spacing:    slice spacing.
    :param6 K:          restraint to start from, in eV per squared xi unit.
    :param7 Settings:   ChainSettings.
    :param8 StepTau:    initial time parameter of the proposal width, in fs.

    :return: Node, K, StepTau: the node, holding the highest energy along its
             quench trajectory from X, the restraint it needed and the
             proposal width the quench ended with.

    """

    PathMax = -np.inf
    # the restraint centre is carried over from where the walker stands
    Xi0 = Coordinate(X)
    Follow = QuenchSchedule(kT=(Settings.Schedule.kT[-2],), Steps=max(1, Settings.Schedule.Steps//2),
                            Block=Settings.Schedule.Block, Acceptance=Settings.Schedule.Acceptance)
    for j in range(1, Settings.Ramp):
        c = Xi0 + (Target - Xi0)*j/float(Settings.Ramp)

        def Restraint(Y, K=K, c=c):
            return 0.5*K*(Coordinate(Y) - c)**2

        R = Quench(System, X, StepTau, Schedule=Follow, Bias=Restraint, Screen=Settings.Screen)
        PathMax = max(PathMax, R.EMax)
        X = R.X
        StepTau = max(R.StepTau, 0.25*StepTau)
    for Attempt in range(0, Settings.MaxStiffen + 1):
        def Restraint(Y, K=K):
            return 0.5*K*(Coordinate(Y) - Target)**2
        R = Quench(System, X, StepTau, Schedule=Settings.Schedule, Bias=Restraint,
                   Screen=Settings.Screen)
        PathMax = max(PathMax, R.EMax)
        Xi = Coordinate(R.X)
        # the width a cold quench ends with is too small to start the next one
        StepTau = max(R.StepTau, 0.25*StepTau)
        if abs(Xi - Target) <= Settings.Tolerance*abs(Spacing) or Attempt == Settings.MaxStiffen:
            break
        K *= 4.0
        X = R.X

    return Node(R.X, R.Energy, Target, Xi, PathMax=PathMax), K, StepTau

#%% 3. Chain
def Sweep(System, Coordinate, Start, Targets, Spacing, K, Settings, StepTau, MaxGap):

    """

    Nodes from a starting node through a sequence of slices, halving a link
    whenever two nodes end up more than MaxGap apart.

    :return: Nodes: the new nodes, in sweep order, the start excluded.

    """

    Nodes = []
    Previous = Start
    for Target in Targets:
        New, K, StepTau = SliceNode(System, Coordinate, Previous.X, Target, Spacing, K,
                                    Settings, StepTau)

        def Densify(P, Q, Depth):
            if Depth >= Settings.MaxDensify or System.Distance(P.X, Q.X) <= MaxGap:
                return []
            M, _, _ = SliceNode(System, Coordinate, P.X, 0.5*(P.Target + Q.Target),
                                0.5*Spacing, K, Settings, StepTau)
            M.Main = False
            return Densify(P, M, Depth + 1) + [M] + Densify(M, Q, Depth + 1)

        Nodes.extend(Densify(Previous, New, 0))
        Nodes.append(New)
        Previous = New

    return Nodes

def LinkEnergies(System, P, Q, MaxGap, PointsPerGap):

    """

    Energies along the straight link between two nodes, in the frame of the first.

    :return: EMax, Length: the highest energy of the interior points and the
             mass-weighted length of the link.

    """

    Q = System.Aligned(Q, P)
    Length = System.Distance(P, Q)
    NPoints = max(PointsPerGap, int(np.ceil(PointsPerGap*Length/MaxGap)))
    EMax = -np.inf
    for t in np.arange(1, NPoints + 1)/(NPoints + 1.0):
        E = System.Energy(P + t*(Q - P))
        EMax = max(EMax, np.inf if E is None else E)

    return EMax, Length

def BuildChain(System, Coordinate, Book, SeedX, XA, XB, States, StepTau0, Settings=None,
               SeedName="seed"):

    """

    Continuation chain through a seed configuration, from the first state to
    the second, with its certificate.

    :param1 System:     AnalyticSystem or AtomsSystem.
    :param2 Coordinate: callable X -> xi.
    :param3 Book:       StateBook naming the end states.
    :param4 SeedX:      configuration the chain is grown from.
    :param5 XA:         minimum of the first state.
    :param6 XB:         minimum of the second state.
    :param7 States:     names of the first and second state.
    :param8 StepTau0:   time parameter of the initial proposal width, in fs,
                        the bead tau of the exploration; it also sets the
                        default link spacing. A width, not a temperature.
    :param9 Settings:   ChainSettings, or None for the defaults.
    :param10 SeedName:  label of the seed, for the output.

    :return: Chain: the chain, its ends, links and certificate.

    """

    Settings = ChainSettings() if Settings is None else Settings
    MaxGap = MoveLength(StepTau0) if Settings.MaxGap is None else Settings.MaxGap
    Calls0 = System.NCalls
    XiA = Coordinate(XA)
    XiB = Coordinate(XB)
    Grid = np.linspace(XiA, XiB, Settings.NSlices + 1)
    Spacing = (XiB - XiA)/Settings.NSlices
    Sign = 1.0 if XiB >= XiA else -1.0
    K = Settings.KFactor/Spacing**2
    StepTau = float(StepTau0)

    Xi0 = Coordinate(SeedX)
    Xi0 = float(np.clip(Xi0, min(XiA, XiB), max(XiA, XiB)))
    Start, K, StepTau = SliceNode(System, Coordinate, SeedX, Xi0, Spacing, K, Settings, StepTau)

    Forward = [g for g in Grid if Sign*(g - Start.Xi) > 0.25*abs(Spacing)]
    Backward = [g for g in Grid[::-1] if Sign*(g - Start.Xi) < -0.25*abs(Spacing)]
    Ahead = Sweep(System, Coordinate, Start, Forward, Spacing, K, Settings, StepTau, MaxGap)
    Behind = Sweep(System, Coordinate, Start, Backward, Spacing, K, Settings, StepTau, MaxGap)
    Nodes = Behind[::-1] + [Start] + Ahead

    Result = Chain(Nodes=Nodes, Seed=SeedName, Coordinate=getattr(Coordinate, "Name", ""))

    # both ends drain freely into the state they belong to
    Ends = []
    for Node_ in (Nodes[0], Nodes[-1]):
        R = Quench(System, Node_.X, StepTau0/16.0, Schedule=Settings.EndSchedule,
                   Energy=Node_.Energy, Screen=Settings.Screen)
        Ends.append((Book.Register(R.X, R.Energy, Converged=R.Converged), R.X, R.Energy, R.EMax,
                     R.Converged))
    Result.Ends = (Ends[0][0], Ends[1][0])
    Result.EndX = (Ends[0][1], Ends[1][1])
    Result.EndEnergy = (Ends[0][2], Ends[1][2])
    Result.DescentMax = (Ends[0][3], Ends[1][3])
    Result.EndConverged = (Ends[0][4], Ends[1][4])

    for P, Q in zip(Nodes[:-1], Nodes[1:]):
        EMax, Length = LinkEnergies(System, P.X, Q.X, MaxGap, Settings.LinkPoints)
        Result.LinkMax.append(EMax)
        Result.LinkLength.append(Length)

    # every node grown from the one before it along its sweep is joined to it
    # by its quench trajectory; each such segment is bounded by the lower of
    # that trajectory and the straight links through the nodes between them
    s = len(Behind)
    Main = [i for i, n in enumerate(Nodes) if n.Main]
    Candidates = [("node", s, Start.Energy)]
    for k, m in enumerate(Main):
        if m == s:
            continue
        Parent = Main[k - 1] if m > s else Main[k + 1]
        a, b = min(m, Parent), max(m, Parent)
        Straight = max([n.Energy for n in Nodes[a:b + 1]] + Result.LinkMax[a:b])
        Bound = min(Straight, Nodes[m].PathMax)
        Result.Segments.append((a, b, Straight, Nodes[m].PathMax))
        Candidates.append(("trajectory" if Nodes[m].PathMax < Straight else "straight", m, Bound))
    Candidates += [("descent", i, e) for i, e in enumerate(Result.DescentMax)]
    Where, Index, EUp = max(Candidates, key=lambda c: c[2])
    Result.EUp = float(EUp)
    Result.Top = (Where, Index)
    Result.Certified = (Result.Ends[0] == States[0] and Result.Ends[1] == States[1]
                        and np.isfinite(Result.EUp))
    Result.NCalls = System.NCalls - Calls0

    return Result

#%% 4. Channels
def SameChannel(System, C1, C2, Radius):

    """

    Whether two chains cross the barrier region at the same place: the top node
    of each lies within Radius of some node of the other.

    :return: Same: bool.

    """

    def TopX(C):
        return C.Nodes[int(np.argmax(C.Energies))].X

    def Near(X, C):
        return min(System.Distance(X, n.X) for n in C.Nodes) <= Radius

    return Near(TopX(C1), C2) and Near(TopX(C2), C1)

def Channels(System, Chains, Radius):

    """

    Group certified chains into channels and keep the lowest bound of each.

    :param1 System: AnalyticSystem or AtomsSystem.
    :param2 Chains: list of Chain.
    :param3 Radius: mass-weighted distance below which two tops coincide.

    :return: Best: list of the lowest certified Chain of every channel,
             lowest first.

    """

    Best = []
    for C in sorted([c for c in Chains if c.Certified], key=lambda c: c.EUp):
        if not any(SameChannel(System, C, B, Radius) for B in Best):
            Best.append(C)

    return Best
