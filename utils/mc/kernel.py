#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

The one Monte-Carlo move every stage of the sampler is built from: a symmetric
proposal, an optional geometric pre-screen on the coupling to another walker,
an optional cheap screen on a surrogate, and the Metropolis test on the true
energy. Only energies and bias values enter, never a gradient.

Created on: Mon Sep 28 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import random
from dataclasses import dataclass
from typing import Any

from .probability import (calcBiasedPELogProbability, calcPELogProbability,
                          calcResidualPELogProbability, probabilityFromLog)

#%% 1. State of one walker
@dataclass
class WalkerState:

    """

    Configuration a walker holds and the values already paid for there.

    :param1 Config:    configuration (ase.Atoms, or a flat numpy array).
    :param2 Energy:    its potential energy, in the unit of the Energy callable.
    :param3 Bias:      bias at the configuration, in eV.
    :param4 Surrogate: surrogate energy at the configuration, in eV.

    """

    Config: Any
    Energy: float
    Bias: float = 0.0
    Surrogate: float = 0.0

@dataclass
class StepResult:

    """

    What one move did, for the dumps and the counters of the callers.

    """

    State: WalkerState
    Trial: Any
    Accepted: bool = False
    Screened: bool = True
    PassedSurrogate: bool = True
    Evaluated: bool = False
    Failed: bool = False
    TrialEnergy: Any = None
    TrialBias: float = 0.0
    TrialSurrogate: float = 0.0
    CouplingInfo: Any = None

#%% 2. The move
def DelayedAcceptanceStep(Current, Propose, Energy, tau, Coupling=None, Finalise=None,
                          Bias=None, Surrogate=None, EnergyScale=1.0, Uniform=random.random):

    """

    One delayed-acceptance Metropolis move on exp(-tau (V + Vbias)/hbar) times
    an optional coupling term.

    The stages run in a fixed order, each drawing at most one uniform number:
    the proposal, the coupling pre-screen, the surrogate screen, the true
    energy. A stage that rejects ends the move without paying for the next
    ones. The coupling term is part of the target and cancels against the
    pre-screen; the surrogate is not, so the last stage tests the residual.

    :param1 Current:     WalkerState of the walker.
    :param2 Propose:     callable Config -> trial configuration; the proposal
                         has to be symmetric.
    :param3 Energy:      callable Config -> energy, or None when the back end
                         fails, which rejects the trial.
    :param4 tau:         time parameter tau (float), in fs, setting beta.
    :param5 Coupling:    callable Trial -> (LogNew, LogOld, Info), the log of
                         the coupling term at the trial and at the current
                         configuration, or None.
    :param6 Finalise:    callable (Trial, CurrentConfig) -> Trial applied once
                         the pre-screen has passed, e.g. a rotation onto the
                         current frame, or None.
    :param7 Bias:        callable Config -> bias in eV, or None.
    :param8 Surrogate:   callable Config -> surrogate energy in eV, or None.
    :param9 EnergyScale: eV per unit of the Energy callable.
    :param10 Uniform:    source of uniform numbers in [0, 1).

    :return: Result: a StepResult holding the new state and the diagnostics.

    """

    Trial = Propose(Current.Config)
    Result = StepResult(State=Current, Trial=Trial, TrialEnergy=Current.Energy,
                        TrialBias=Current.Bias, TrialSurrogate=Current.Surrogate)

    # stage 1: geometric pre-screen towards the other walker, tried once; the
    # walker stays put when it fails, so no proposal normalisation is left over
    if Coupling is not None:
        LogNew, LogOld, Result.CouplingInfo = Coupling(Trial)
        Result.Screened = probabilityFromLog(LogNew - LogOld) >= Uniform()
        if not Result.Screened:
            return Result

    if Finalise is not None:
        Trial = Finalise(Trial, Current.Config)
        Result.Trial = Trial

    # a Metropolis test needs the value of the bias, never its gradient
    Result.TrialBias = 0.0 if Bias is None else float(Bias(Trial))

    # stage 2: screen on the surrogate and the bias, both cheap
    if Surrogate is not None:
        Result.TrialSurrogate = float(Surrogate(Trial))
        logScreen = calcBiasedPELogProbability(Result.TrialSurrogate, Current.Surrogate,
                                               Result.TrialBias, Current.Bias, tau)
        Result.PassedSurrogate = probabilityFromLog(logScreen) >= Uniform()
        if not Result.PassedSurrogate:
            return Result

    # stage 3: the true energy
    TrialProb = Uniform()
    TrialEnergy = Energy(Trial)
    Result.Evaluated = True
    Result.Failed = TrialEnergy is None
    if Result.Failed:
        return Result
    Result.TrialEnergy = TrialEnergy

    E1 = TrialEnergy*EnergyScale
    E0 = Current.Energy*EnergyScale
    if Surrogate is not None:
        # the residual, which is what keeps the chain exact
        logPEProb = calcResidualPELogProbability(E1, E0, Result.TrialSurrogate,
                                                 Current.Surrogate, tau)
    elif Bias is None:
        logPEProb = calcPELogProbability(E1, E0, tau)
    else:
        logPEProb = calcBiasedPELogProbability(E1, E0, Result.TrialBias, Current.Bias, tau)

    Result.Accepted = probabilityFromLog(logPEProb) >= TrialProb
    if Result.Accepted:
        Result.State = WalkerState(Config=Trial, Energy=TrialEnergy,
                                   Bias=Result.TrialBias, Surrogate=Result.TrialSurrogate)

    return Result
