#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Tests for the bias entering the acceptance probability of the samplers.

A Metropolis chain only ever needs the value of the bias, so adding one is a
change to the acceptance ratio and nothing else. These tests use deterministic
synthetic biases rather than a network, so the expected acceptance can be
written down in closed form.

Created on: Tue Aug 25 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import random

import numpy as np
import pytest

from constants.units import kTEffeV
from analytic_2d.potential import X0, ProtonTransfer
from utils.mc.probability import (calcBiasedPELogProbability,
                                  calcPELogProbability, probabilityFromLog)
from utils.mc.system import AnalyticSystem
from utils.mc.walk import Walk

#%% 1. The acceptance ratio
def test_a_zero_bias_reproduces_the_unbiased_ratio():

    """Passing zeros must be indistinguishable from not biasing at all."""

    for dE in (-0.2, 0.0, 0.35):
        assert calcBiasedPELogProbability(dE, 0.0, 0.0, 0.0, 4.0) == \
               pytest.approx(calcPELogProbability(dE, 0.0, 4.0))

def test_the_bias_shifts_the_ratio_by_its_difference():

    """Only the difference of the bias between the two points can matter."""

    tau = 4.0
    a = calcBiasedPELogProbability(0.1, 0.0, 0.7, 0.4, tau)
    b = calcPELogProbability(0.1 + 0.3, 0.0, tau)

    assert a == pytest.approx(b)

def test_a_constant_bias_cancels():

    """A bias that is the same everywhere changes nothing."""

    tau = 4.0
    for c in (-5.0, 0.0, 12.0):
        assert calcBiasedPELogProbability(0.2, 0.0, c, c, tau) == \
               pytest.approx(calcPELogProbability(0.2, 0.0, tau))

def test_the_invariant_distribution_is_the_biased_boltzmann_factor():

    """The acceptance weight must be exp(-(dV + dVbias)/kT_eff)."""

    tau = 4.0
    kT = kTEffeV(tau)
    p = probabilityFromLog(calcBiasedPELogProbability(0.10, 0.0, 0.05, 0.0, tau))

    assert p == pytest.approx(np.exp(-0.15/kT), rel=1e-10)

#%% 2. The hook inside the walk
def RunBiasedWalk(R0, NSteps, Bias=None, Seed=1, Pace=0):

    """Seeded walk on the proton-transfer model, one amu, tau 20 fs."""

    random.seed(Seed)

    return Walk(AnalyticSystem(ProtonTransfer(), 1.0), R0, 20.0, NSteps, Bias=Bias, Pace=Pace)

def test_a_flat_bias_leaves_the_walk_unchanged():

    """

    A bias that does not depend on position must reproduce the unbiased walk
    step for step, which shows the hook adds nothing of its own.

    """

    Plain = RunBiasedWalk([0.5, 0.0], 600, Bias=None, Seed=5)
    Flat = RunBiasedWalk([0.5, 0.0], 600, Bias=lambda R: 3.0, Seed=5)

    assert np.allclose(Plain["X"], Flat["X"])
    assert Plain["Acceptance"] == pytest.approx(Flat["Acceptance"])

def test_a_tilt_shifts_the_invariant_distribution():

    """

    A bias linear in the reaction coordinate tilts the landscape, so the mean
    coordinate must move against the tilt and do so monotonically. This is the
    statement that the bias really enters the sampled distribution rather than
    merely being evaluated, and unlike a test built on the barrier itself it
    does not depend on the shape of the model potential.

    Two bias shapes that look reasonable are worth avoiding. A narrow well
    sitting on the barrier cannot attract a walker that never reaches it,
    which is why the Kolmogorov bias raises the basins as well as lowering the
    barrier. Cancelling the whole quartic removes the outer confinement along
    with the barrier and the walker diffuses away to infinity.

    """

    Means = []
    for Tilt in (-0.05, 0.0, 0.05):
        R = RunBiasedWalk([X0, 0.0], 6000, Bias=lambda P, c=Tilt: c*P[0], Seed=17)["X"][500:]
        Means.append(float(R[:, 0].mean()))

    # a positive tilt penalises large x, so the mean must decrease
    assert Means[0] > Means[1] > Means[2]

def test_deposition_only_happens_when_a_pace_is_given():

    """The walker must not touch an adaptive bias unless asked to."""

    class Counter:
        def __init__(self):
            self.N = 0

        def __call__(self, R):
            return 0.0

        def Deposit(self, R):
            self.N += 1

    Silent = Counter()
    RunBiasedWalk([0.5, 0.0], 200, Bias=Silent, Seed=1, Pace=0)
    assert Silent.N == 0

    Loud = Counter()
    RunBiasedWalk([0.5, 0.0], 200, Bias=Loud, Seed=1, Pace=20)
    assert Loud.N == 10

def test_the_cached_bias_follows_a_deposit():

    """

    A deposit changes the bias under the walker. The value the next move is
    compared against, and the value recorded, must be the bias after the
    deposit, not the one the walker carried before it.

    """

    class Hills:
        def __init__(self):
            self.Centres = []
            self.AfterDeposit = []

        def __call__(self, R):
            return float(sum(0.05*np.exp(-np.sum((np.asarray(R) - c)**2)/0.02)
                             for c in self.Centres))

        def Deposit(self, R):
            self.Centres.append(np.array(R, dtype=float))
            self.AfterDeposit.append(self(R))

    Bias = Hills()
    VB = RunBiasedWalk([0.5, 0.0], 50, Bias=Bias, Seed=3, Pace=1)["V"]

    assert np.allclose(VB, Bias.AfterDeposit)
