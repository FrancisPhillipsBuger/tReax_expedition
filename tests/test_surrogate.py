#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Tests for the delayed acceptance against a surrogate, which must leave the
sampled distribution untouched, and for the surrogate values the molecular
two-walker run caches.

Created on: Tue Aug 25 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import numpy as np
import pytest

from constants.units import kTEffeV
from utils.mc.probability import (calcPELogProbability,
                                  calcResidualPELogProbability,
                                  probabilityFromLog)

#%% 1. The residual acceptance
def test_a_perfect_surrogate_makes_the_residual_always_accept():

    """

    When the surrogate equals the potential the residual vanishes, the second
    stage accepts unconditionally and the first stage carries the whole test.

    """

    for e in (-0.3, 0.0, 0.7):
        logP = calcResidualPELogProbability(e, 0.0, e, 0.0, 4.0)
        assert probabilityFromLog(logP) == 1.0

def test_a_constant_offset_in_the_surrogate_cancels():

    """Only differences of the residual can matter."""

    logP = calcResidualPELogProbability(0.4, 0.1, 5.0, 5.0, 4.0)

    assert logP == pytest.approx(calcPELogProbability(0.4, 0.1, 4.0))

def test_the_two_stages_compose_into_the_true_metropolis_ratio():

    """

    The product of the two acceptance probabilities, taken in both directions,
    must reproduce the ratio of the target at the two points. This is the
    identity that makes the scheme exact and it is the thing that would break
    if the second stage tested the energy rather than the residual.

    """

    tau = 4.0
    kT = kTEffeV(tau)
    Vx, Vy = 0.10, 0.35
    Sx, Sy = 0.08, 0.30

    def Forward():
        return (probabilityFromLog(calcPELogProbability(Sy, Sx, tau)) *
                probabilityFromLog(calcResidualPELogProbability(Vy, Vx, Sy, Sx, tau)))

    def Backward():
        return (probabilityFromLog(calcPELogProbability(Sx, Sy, tau)) *
                probabilityFromLog(calcResidualPELogProbability(Vx, Vy, Sx, Sy, tau)))

    assert Forward()/Backward() == pytest.approx(np.exp(-(Vy - Vx)/kT), rel=1e-10)

#%% 2. The molecular two-walker run
def test_cached_values_describe_the_state_the_walkers_hold(monkeypatch, tmp_path):

    """

    The two-walker run caches the surrogate energy of each walker. It need
    not be rotation invariant, so nothing between two steps may rotate a
    walker in place without the cache following; the convergence check used
    to do exactly that to the second walker.

    """

    from ase import Atoms
    from ase.calculators.emt import EMT
    import treax_expedition.sampler as Sampler

    def Tilted(S):
        # deliberately not rotation invariant
        return 0.01*float(S.get_positions()[0, 0])

    Start = Atoms("Cu3", positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [1.2, 2.1, 0.0]])
    End = Atoms("Cu3", positions=[[0.0, 0.0, 0.0], [2.6, 0.0, 0.3], [1.0, 2.2, -0.4]])
    Structures = []
    for a in (Start, End):
        a.calc = EMT()
        Structures.append(a)

    Original = Sampler.mcTwoStructuresStep
    Checked = []

    def CheckingStep(Structures, CurrentEnergies, *Args, **Keywords):
        for n in range(0, 2):
            assert Keywords["CurrentSurrogates"][n] == pytest.approx(Tilted(Structures[n]), abs=1e-12)
        Checked.append(True)
        return Original(Structures, CurrentEnergies, *Args, **Keywords)

    monkeypatch.setattr(Sampler, "mcTwoStructuresStep", CheckingStep)
    Sampler.doTwoStructureSWSMC(Structures, tau=5.0, tau12=50.0, Nsteps=40,
                                ScaleThreshold=1.0e-9, Seed=4, LogEvaluations=False,
                                Surrogate=Tilted,
                                FolderToSave=str(tmp_path))

    assert len(Checked) == 39
