#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Unit-conversion coefficients shared by the SWS Monte-Carlo samplers. They map
the fs / amu / Angstrom working units onto the displacement width and onto the
kinetic- and potential-energy terms of the acceptance probability.

Created on: Mon Aug 24 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import numpy as np
import scipy.constants as spc

#%% 1. SWS displacement and acceptance-probability coefficients
SxCoeff = 1.0e10 * np.sqrt(0.5 * spc.hbar * 1e-15 / (spc.physical_constants["atomic mass unit-kilogram relationship"][0]) )
KECoeff = spc.physical_constants["atomic mass unit-kilogram relationship"][0] * (1.0e-20)/(spc.hbar * 1.e-15)
PECoeff = spc.eV * 1.e-15/spc.hbar

#%% 2. Model-potential conversion factors
# cm-1 force constant from an amu mass and a cm-1 frequency
ConvCoeffForce = spc.physical_constants["atomic mass unit-kilogram relationship"][0] * (1.0e-20) * spc.c * (1e2) * 4 * np.pi**2  / (spc.h)
wvn2eV = 0.00012398425731484318
wvn2Ha = 0.00000455633528121
amu2au = 1822.888486
Bohr2Ang = 0.529177210544
eV2kJmol = spc.eV * spc.N_A * 1.e-3

#%% 3. Effective temperature of the SWS ensemble
def BetaEff(tau):

    """

    Inverse effective temperature of the SWS ensemble sampled at tau.

    This is the ensemble of one walker. The midpoint of a coupled pair whose
    walkers sample at tau represents BetaEff(2 tau), see PairEnsemble.

    :param1 tau: time parameter tau (float), in fs.

    :return: beta: inverse effective temperature, in 1/eV.

    """

    return PECoeff * tau

def kTEffeV(tau):

    """

    Effective thermal energy of the SWS ensemble sampled at tau.

    :param1 tau: time parameter tau (float), in fs.

    :return: kT: effective thermal energy, in eV.

    """

    return 1.0 / BetaEff(tau)
