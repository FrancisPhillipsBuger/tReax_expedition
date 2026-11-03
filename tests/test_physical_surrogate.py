#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Tests for screening an expensive level of theory on a cheaper one. The
exactness of the two stage acceptance is established in test_surrogate.py and
does not depend on where the cheap potential comes from; what is checked here
is that this surrogate leaves the run it is accelerating alone.

Created on: Thu Aug 27 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import numpy as np
import pytest

from treax_expedition.calculators import setCalculatorToStructure
from treax_expedition.io_xyz import readXYZFile
from treax_expedition.xtb import getXTBCalculator
from utils.surrogate import CalculatorSurrogate, ForXTB

#%% 1. Behaviour
@pytest.mark.reaction("sn2")
def test_the_true_calculator_stays_attached():

    """Screening must not steal the structure's own calculator."""

    Structure = readXYZFile("sn2")[0]
    Truth = getXTBCalculator(Method="gfn2-xtb", charge=-1, mult=1)
    Structure = setCalculatorToStructure(Structure, Truth)
    Surrogate = ForXTB(charge=-1, mult=1)
    Surrogate(Structure)

    assert Structure.calc is Truth

@pytest.mark.reaction("sn2")
def test_the_positions_are_not_disturbed():

    """A screening call is a question, not a move."""

    Structure = readXYZFile("sn2")[0]
    Before = Structure.get_positions().copy()
    ForXTB(charge=-1, mult=1)(Structure)

    assert np.array_equal(Structure.get_positions(), Before)

@pytest.mark.reaction("sn2")
def test_it_answers_the_cheap_energy():

    """The value handed back is the cheap potential, not the true one."""

    Structure = readXYZFile("sn2")[0]
    Surrogate = ForXTB(charge=-1, mult=1)
    Probe = Structure.copy()
    Probe.calc = getXTBCalculator(charge=-1, mult=1)

    # two SCF runs agree only to the convergence threshold, and a threaded
    # reduction changes the last bits from one run to the next
    assert Surrogate(Structure) == pytest.approx(Probe.get_potential_energy(), abs=1e-6)

@pytest.mark.reaction("sn2")
def test_calls_are_counted():

    """The screening cost has to be reportable against what it saved."""

    Structure = readXYZFile("sn2")[0]
    Surrogate = ForXTB(charge=-1, mult=1)
    for _ in range(3):
        Surrogate(Structure)

    assert Surrogate.NCalls == 3

def test_there_is_nothing_to_train():

    """The sampler's training hooks must skip a surrogate that cannot learn."""

    Surrogate = CalculatorSurrogate(getXTBCalculator(charge=-1, mult=1))

    assert not hasattr(Surrogate, "Observe")
    assert not hasattr(Surrogate, "MaybeRefit")
