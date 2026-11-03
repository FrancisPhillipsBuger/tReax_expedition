#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Single-walker Monte-Carlo chain on exp(-tau (V + Vbias)/hbar) over any system,
with the free-particle proposal of width sqrt(hbar tau/2m): the sampling step
of the committor stage and the engine of its shots.

Created on: Mon Sep 28 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import numpy as np

from .kernel import DelayedAcceptanceStep, WalkerState

#%% 1. Walk
def Walk(System, X0, tau, NSteps, Bias=None, Pace=0, Stride=1, Energy=None, Stop=None):

    """

    Metropolis walk at tau with an optional bias.

    :param1 System: AnalyticSystem or AtomsSystem.
    :param2 X0:     starting coordinates.
    :param3 tau:    time parameter tau (float), in fs; it sets both beta and
                    the free-particle width of the proposal.
    :param4 NSteps: Monte-Carlo steps.
    :param5 Bias:   callable X -> bias in eV, or None; a Deposit method is
                    called every Pace steps and the cached value follows it.
    :param6 Pace:   deposit into an adaptive bias every this many steps.
    :param7 Stride: keep every this many configurations.
    :param8 Energy: energy at X0 if already known.
    :param9 Stop:   callable X -> value, the walk ending at the first accepted
                    configuration where it is not None; or None.

    :return: Result: dict with X (kept configurations), V (their biases),
             Last, ELast, Acceptance, NSteps (taken), Outcome (of Stop).

    """

    X0 = np.asarray(X0, dtype=float)
    E0 = System.Energy(X0) if Energy is None else float(Energy)
    State = WalkerState(X0, E0, 0.0 if Bias is None else float(Bias(X0)))

    def Propose(Y):
        return System.Propose(Y, tau)

    Kept = []
    Values = []
    NAcc = 0
    Outcome = None
    Taken = 0
    for i in range(0, NSteps):
        Result = DelayedAcceptanceStep(State, Propose, System.Energy, tau, Bias=Bias)
        Taken = i + 1
        if Result.Accepted:
            State = Result.State
            NAcc += 1
            if Stop is not None:
                Outcome = Stop(State.Config)
                if Outcome is not None:
                    break
        if Pace > 0 and (i % Pace) == 0 and hasattr(Bias, "Deposit"):
            Bias.Deposit(State.Config)
            State = WalkerState(State.Config, State.Energy, float(Bias(State.Config)))
        if (i % Stride) == 0:
            Kept.append(np.asarray(State.Config, dtype=float).copy())
            Values.append(State.Bias)

    return {"X": np.array(Kept), "V": np.array(Values), "Last": np.asarray(State.Config, dtype=float),
            "ELast": State.Energy, "Acceptance": NAcc/max(1, Taken), "NSteps": Taken,
            "Outcome": Outcome}
