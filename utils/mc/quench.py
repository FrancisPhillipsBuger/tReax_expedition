#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Gradient-free quench: the Monte-Carlo kernel with one walker, run down a
ladder of effective temperatures, i.e. the tau -> infinity end of the SWS
ensemble. Used to identify the basin a configuration drains into, and, with a
restraint as the bias, to relax a configuration on a slice.

Created on: Mon Sep 28 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
from dataclasses import dataclass, field

import numpy as np

from constants.units import PECoeff

from .kernel import DelayedAcceptanceStep, WalkerState

#%% 1. Schedule and result
@dataclass
class QuenchSchedule:

    """

    Effective temperatures of the quench and the steps spent at each.

    :param1 kT:          thermal energies in eV, hottest first.
    :param2 Steps:       Monte-Carlo steps per temperature.
    :param3 Block:       steps between two adjustments of the proposal width.
    :param4 Acceptance:  target band of the acceptance rate.
    :param5 Patience:    blocks in a row without a drop of the lowest energy
                         by more than Tol that end a temperature early; zero
                         always runs the full Steps.
    :param6 Tol:         the drop, in eV, that counts as progress.

    """

    kT: tuple = (0.02, 0.008, 0.003, 0.001)
    Steps: int = 250
    Block: int = 25
    Acceptance: tuple = (0.2, 0.5)
    Patience: int = 0
    Tol: float = 1.0e-4

    def Scaled(self, NumCoordinates, Reference=18, Factor=1.0, Patience=None, Cool=False):

        """

        :param1 NumCoordinates: degrees of freedom of the system.
        :param2 Reference:      degrees of freedom the Steps are meant for.
        :param3 Factor:         further multiplier, for a system whose soft
                                modes relax slowly under single-atom moves.
        :param4 Patience:       replaces the patience when given; zero spends
                                the whole budget, which a slowly relaxing
                                system needs, since its energy keeps drifting
                                down below any fixed tolerance per block.
        :param5 Cool:           also scale the temperatures by Reference over
                                the degrees of freedom when there are more, so
                                the excess energy (d/2) kT of the hot stage,
                                which a chain trajectory has to pass through,
                                stays that of the reference system.

        :return: Schedule: a copy whose Steps grow with the degrees of
                 freedom, a single-atom move relaxing one atom at a time.

        """

        from dataclasses import replace

        Ratio = max(1.0, NumCoordinates/float(Reference))

        return replace(self, Steps=int(round(Factor*self.Steps*Ratio)),
                       kT=tuple(t/Ratio for t in self.kT) if Cool else self.kT,
                       Patience=self.Patience if Patience is None else int(Patience))

@dataclass
class QuenchResult:

    """

    Where the quench ended and what it cost.

    """

    X: np.ndarray
    Energy: float
    Bias: float = 0.0
    EMax: float = -np.inf
    NCalls: int = 0
    StepTau: float = 0.0
    Acceptance: float = 0.0
    Converged: bool = True
    History: list = field(default_factory=list)

def TauAtkT(kT):

    """

    :param1 kT: thermal energy, in eV.

    :return: tau: the time parameter whose SWS ensemble has that kT, in fs.

    """

    return 1.0/(PECoeff*kT)

#%% 2. Quench
def Quench(System, X, StepTau, Schedule=None, Bias=None, Energy=None, Screen=False,
           KeepHistory=False):

    """

    Anneal one walker down the schedule with the MCTSS kernel.

    The proposal is the free-particle Gaussian of the sampler, its width set
    by StepTau. The width is adjusted between blocks to keep the acceptance in
    the target band, since at the cold end of a quench the width that belongs
    to tau itself would reject every move. The chain is an optimiser here, not
    a sampler of a distribution, so the adaptation costs nothing in rigour.

    :param1 System:      AnalyticSystem or AtomsSystem.
    :param2 X:           starting coordinates.
    :param3 StepTau:     initial time parameter of the proposal width, in fs.
    :param4 Schedule:    QuenchSchedule, or None for the default.
    :param5 Bias:        callable X -> bias in eV (e.g. a restraint), or None.
    :param6 Energy:      energy at X if already known, or None.
    :param7 Screen:      screen proposals on the system's cheaper calculator.
    :param8 KeepHistory: keep every accepted configuration (bool).

    :return: Result: a QuenchResult; EMax is the highest potential energy the
             walker held along the way, starting point included, and Converged
             says whether the coldest temperature ended on a plateau (always
             True when the schedule has no patience).

    """

    Schedule = QuenchSchedule() if Schedule is None else Schedule
    Calls0 = System.NCalls
    X = np.asarray(X, dtype=float).copy()
    E0 = System.Energy(X) if Energy is None else float(Energy)
    if E0 is None:
        raise RuntimeError("The quench cannot start: the energy failed at the starting point.")
    Surrogate = System.ScreenEnergy if (Screen and getattr(System, "Probe", None) is not None) else None
    State = WalkerState(X, E0, 0.0 if Bias is None else float(Bias(X)),
                        0.0 if Surrogate is None else Surrogate(X))
    EMax = E0
    History = [X.copy()] if KeepHistory else []
    Tau = float(StepTau)
    Low, High = Schedule.Acceptance
    NAcc = 0
    NTot = 0
    Converged = False

    def Propose(Y):
        return System.Propose(Y, Tau)

    for kT in Schedule.kT:
        tau = TauAtkT(kT)
        BlockAcc = 0
        Lowest = State.Energy + State.Bias
        Stalled = 0
        Converged = False
        for i in range(0, Schedule.Steps):
            Result = DelayedAcceptanceStep(State, Propose, System.Energy, tau, Bias=Bias,
                                           Surrogate=Surrogate)
            NTot += 1
            if Result.Accepted:
                State = Result.State
                NAcc += 1
                BlockAcc += 1
                EMax = max(EMax, State.Energy)
                if KeepHistory:
                    History.append(State.Config.copy())
            if (i + 1) % Schedule.Block == 0:
                Rate = BlockAcc/Schedule.Block
                if Rate < Low:
                    Tau *= 0.5
                elif Rate > High:
                    Tau *= 1.5
                BlockAcc = 0
                Now = State.Energy + State.Bias
                Stalled = Stalled + 1 if Now > Lowest - Schedule.Tol else 0
                Lowest = min(Lowest, Now)
                if Schedule.Patience > 0 and Stalled >= Schedule.Patience:
                    Converged = True
                    break

    return QuenchResult(X=np.asarray(State.Config, dtype=float), Energy=State.Energy,
                        Bias=State.Bias, EMax=EMax, NCalls=System.NCalls - Calls0,
                        StepTau=Tau, Acceptance=NAcc/max(1, NTot),
                        Converged=Converged or Schedule.Patience == 0, History=History)
