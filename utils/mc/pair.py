#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Equilibrium sampling of a coupled pair on any system: the two walkers are
updated in turn on the joint target of a PairEnsemble, an optional side
constraint keeping them on opposite sides of a dividing surface, and the
midpoint energy is paid for on every Stride-th sweep to weight the midpoints.
The walkers share one frame; their relative orientation is sampled by a rigid
rotation that costs no energy. The recorded midpoints are rotated onto the
first one, which changes nothing but the frame they are written in.

Created on: Mon Sep 28 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import random

import numpy as np

from .ensemble import EffectiveSampleFraction, Midpoint
from .kernel import DelayedAcceptanceStep, WalkerState

#%% 1. Pair walk
def PairWalk(System, X1, X2, Ensemble, NSteps, StepTau, Side=None, Stride=10, Tune=0,
             TargetAcceptance=0.3, E1=None, E2=None):

    """

    Metropolis sweeps of a pair on its joint target, times 1[Side(R1) < 0 < Side(R2)].

    The side constraint and the coupling are both cheap geometric terms of the
    target, so they share the pre-screen of the kernel and a move that breaks
    them costs no energy. During the first Tune sweeps the proposal width is
    adapted towards TargetAcceptance and nothing is recorded; it is frozen
    afterwards, which keeps the recorded chain exact.

    :param1 System:   AnalyticSystem or AtomsSystem.
    :param2 X1:       start of the first walker; Side(X1) < 0 when Side is given.
    :param3 X2:       start of the second walker; Side(X2) > 0 when Side is given.
    :param4 Ensemble: PairEnsemble.
    :param5 NSteps:   recorded sweeps, one move of each walker per sweep.
    :param6 StepTau:  initial time parameter of the proposal width, in fs.
    :param7 Side:     callable X -> signed distance to the dividing surface, or None.
    :param8 Stride:   evaluate the midpoint every this many recorded sweeps.
    :param9 Tune:     sweeps of width adaptation before recording.
    :param10 TargetAcceptance: accepted share of moves the adaptation aims at.
    :param11 E1:      energy at X1 if already known, in eV.
    :param12 E2:      energy at X2 if already known, in eV.

    :return: Result: dict with M, VM, V1, V2, LogW (per recorded midpoint),
             ESS (fraction), Acceptance, Screened, RotationAcceptance (nan
             without a frame), StepTau, NCalls, NFailed,
             Last (the two walkers) and ELast.

    """

    # the walkers share one frame: the second starts aligned onto the first,
    # and from then on no walker is rotated onto anything
    Walkers = [System.Centred(X1), System.Aligned(X2, X1)]
    Frame = Midpoint(System, Walkers[0], Walkers[1])
    if Side is not None and not (Side(Walkers[0]) < 0.0 < Side(Walkers[1])):
        raise ValueError("The walkers do not start on the two sides of the dividing surface.")
    Calls0 = System.NCalls
    Energies = [System.Energy(Walkers[0]) if E1 is None else float(E1),
                System.Energy(Walkers[1]) if E2 is None else float(E2)]
    if Energies[0] is None or Energies[1] is None:
        raise ValueError("The energy is not available at a starting configuration.")
    Tau = float(StepTau)
    Sign = (-1.0, 1.0)

    def Coupling(n):
        Other = Walkers[1 - n]
        LogOld = Ensemble.CouplingLog(System.FrameDistance(Walkers[n], Other)**2)

        def Log(Trial):
            if Side is not None and not Sign[n]*Side(Trial) > 0.0:
                return -np.inf, LogOld, None
            return Ensemble.CouplingLog(System.FrameDistance(Trial, Other)**2), LogOld, None

        return Log

    M = []
    VM = []
    V1 = []
    V2 = []
    NAcc = NScreen = NMoves = NFailed = NRot = 0
    Block = [0, 0]
    for i in range(0, Tune + NSteps):
        Recording = i >= Tune
        for n in (0, 1):
            Result = DelayedAcceptanceStep(WalkerState(Walkers[n], Energies[n]),
                                           lambda Y: System.Displace(Y, Tau), System.Energy,
                                           Ensemble.TauBead, Coupling=Coupling(n))
            if Result.Accepted:
                Walkers[n] = Result.State.Config
                Energies[n] = Result.State.Energy
            Block[0] += int(Result.Accepted)
            Block[1] += 1
            if Recording:
                NMoves += 1
                NAcc += int(Result.Accepted)
                NScreen += int(Result.Screened)
        if System.Rotational:
            # the relative orientation, by a rigid rotation of the second
            # walker: its energy and its side are those it had, so only the
            # coupling enters the test and no energy is paid for
            Sigma = System.RotationWidth(Walkers[1], Ensemble.Tau12)
            Trial = System.Rotated(Walkers[1], Sigma)
            LogNew, LogOld, _ = Coupling(1)(Trial)
            if np.log(random.random()) < LogNew - LogOld:
                Walkers[1] = Trial
                NRot += 1
        if not Recording and Block[1] >= 100:
            # multiplicative width update, between blocks only
            Tau *= float(np.exp(2.0*(Block[0]/Block[1] - TargetAcceptance)))
            Block = [0, 0]
        if Recording and (i - Tune) % Stride == 0:
            X = System.Aligned(Midpoint(System, Walkers[0], Walkers[1]), Frame)
            E = System.Energy(X)
            if E is None:
                NFailed += 1
                continue
            M.append(X)
            VM.append(E)
            V1.append(Energies[0])
            V2.append(Energies[1])

    LogW = Ensemble.LogWeight(VM, V1, V2) if len(VM) > 0 else np.zeros(0)

    return {"M": np.array(M), "VM": np.array(VM), "V1": np.array(V1), "V2": np.array(V2),
            "LogW": LogW, "ESS": EffectiveSampleFraction(LogW),
            "Acceptance": NAcc/max(1, NMoves), "Screened": NScreen/max(1, NMoves),
            "RotationAcceptance": NRot/max(1, Tune + NSteps) if System.Rotational else np.nan,
            "StepTau": Tau, "NCalls": System.NCalls - Calls0, "NFailed": NFailed,
            "Last": [w.copy() for w in Walkers], "ELast": list(Energies)}
