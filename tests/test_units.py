#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Tests for the unit conversions, in particular the identity that ties the time
parameter tau to the effective temperature of the sampled ensemble.

Created on: Tue Aug 25 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import numpy as np
import pytest
import scipy.constants as spc

from constants.units import (BetaEff, KECoeff, PECoeff, SxCoeff, eV2kJmol,
                             kTEffeV)
from utils.mc.probability import (calcKELogProbabilityFromRMSD,
                                  calcPELogProbability, probabilityFromLog)

#%% 1. Effective temperature
def test_beta_eff_is_tau_over_hbar():

    """The acceptance weight exp(-PECoeff*tau*dE) must equal exp(-tau*dE/hbar)."""

    hbar_eV = spc.hbar/spc.eV
    for tau in [0.5, 1.0, 2.0, 4.0, 10.0]:
        assert BetaEff(tau) == pytest.approx(tau*1e-15/hbar_eV, rel=1e-12)

def test_kT_eff_table():

    """Reference values of kT_eff used throughout the analysis, in kJ/mol."""

    expected = {2: 31.75, 3: 21.17, 4: 15.88, 5: 12.70, 6: 10.58}
    for tau, kT in expected.items():
        assert kTEffeV(tau)*eV2kJmol == pytest.approx(kT, abs=0.01)

def test_T_eff_scales_inversely_with_tau():

    """T_eff must be 7638 K at 1 fs and fall like 1/tau."""

    kB = spc.physical_constants["Boltzmann constant in eV/K"][0]
    assert kTEffeV(1.0)/kB == pytest.approx(7638.0, rel=1e-3)
    for tau in [2.0, 5.0, 10.0]:
        assert kTEffeV(tau) == pytest.approx(kTEffeV(1.0)/tau, rel=1e-12)

#%% 2. Acceptance probability
def test_acceptance_reproduces_boltzmann_at_kT_eff():

    """The potential term must be a Boltzmann factor at the effective temperature."""

    tau = 4.0
    kT = kTEffeV(tau)
    for dE in [0.0, 0.01, 0.05, 0.2, 1.0]:
        p = probabilityFromLog(calcPELogProbability(dE, 0.0, tau))
        assert p == pytest.approx(np.exp(-dE/kT), rel=1e-10)

def test_downhill_moves_are_always_accepted():

    """A move that lowers the energy must have probability one."""

    assert probabilityFromLog(calcPELogProbability(-0.5, 0.0, 3.0)) == 1.0

#%% 3. The proposal is the kinetic kernel
def test_displacement_width_matches_kinetic_kernel():

    """

    The SWS displacement has variance hbar*tau/(2m), so a Gaussian step of that
    width has exponent -m*dR^2/(hbar*tau), which is exactly the kinetic term.
    This is why the proposal cancels in the Metropolis ratio.

    """

    for tau in [0.5, 2.0, 7.0]:
        for mass in [1.0, 12.0, 79.9]:
            sigma = SxCoeff*np.sqrt(tau/mass)
            dR = 0.37
            gaussian_exponent = -dR**2/(2.0*sigma**2)
            kinetic = calcKELogProbabilityFromRMSD(mass*dR**2, tau)
            assert kinetic == pytest.approx(gaussian_exponent, rel=1e-12)

def test_ke_coefficient_consistency():

    """KECoeff and SxCoeff must derive from the same hbar*tau/(2m) convention."""

    assert KECoeff*SxCoeff**2 == pytest.approx(0.5, rel=1e-12)
    assert PECoeff > 0.0
