#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Acceptance-probability core shared by both samplers: the kinetic term from a
mass-weighted RMSD, the potential term from an energy difference, its biased
and residual variants, and the conversion of a log-probability back to a
probability.

Created on: Mon Aug 24 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import numpy as np

from constants.units import KECoeff, PECoeff

#%% 1. Kinetic energy term
def calcKELogProbabilityFromRMSD(mwRMSD, tau):

    """

    Kinetic-energy part of the MC acceptance probability, as a logarithm.

    :param1 mwRMSD: mass-weighted sum of squared displacements between two
                    structures, sum_i m_i |dr_i|^2, in amu*angstroem**2; not a
                    root mean square, despite the name.
    :param2 tau:    time parameter tau (float), in fs.

    :return: logP: logarithm of the probability.

    """

    logP = -KECoeff*mwRMSD/tau

    return logP

#%% 2. Potential energy term
def calcPELogProbability(TrialEnergy, CurrentEnergy, tau):

    """

    Potential-energy part of the MC acceptance probability, as a logarithm.

    :param1 TrialEnergy:   energy of trial structure (in eV).
    :param2 CurrentEnergy: energy of current structure (in eV).
    :param3 tau:           time parameter tau (float), in fs.

    :return: logP: logarithm of the probability.

    """

    logP = -PECoeff * tau * (TrialEnergy - CurrentEnergy)

    return logP

#%% 3. Log-probability to probability
def probabilityFromLog(logP):

    """

    Acceptance probability from its logarithm.

    :param1 logP: logarithm of the probabilty (any float).

    :return: probability: probability (between 0 and 1).

    """

    probability = 0.
    if logP >= 0.:
        probability = 1.
    else:
        probability = np.exp(logP)
    return probability

#%% 4. Biased potential energy term
def calcBiasedPELogProbability(TrialEnergy, CurrentEnergy, TrialBias, CurrentBias, tau):

    """

    Potential-energy part of the acceptance probability including a bias.

    Adding a bias to a Metropolis chain only requires the value of the bias at
    the two points; unlike molecular dynamics it never needs its gradient. The
    invariant distribution becomes exp(-tau (V + Vbias)/hbar).

    :param1 TrialEnergy:   energy of trial structure (in eV).
    :param2 CurrentEnergy: energy of current structure (in eV).
    :param3 TrialBias:     bias at the trial structure (in eV).
    :param4 CurrentBias:   bias at the current structure (in eV).
    :param5 tau:           time parameter tau (float), in fs.

    :return: logP: logarithm of the probability.

    """

    return calcPELogProbability(TrialEnergy + TrialBias,
                                CurrentEnergy + CurrentBias, tau)

#%% 5. Residual potential energy term
def calcResidualPELogProbability(TrialEnergy, CurrentEnergy,
                                 TrialSurrogate, CurrentSurrogate, tau):

    """

    Second stage of a delayed-acceptance move against a surrogate.

    The existing two-stage move needs no correction term because its cheap
    first stage tests a factor that is itself part of the target distribution,
    so it cancels. A surrogate is not part of the target, and skipping the
    correction would silently sample the wrong distribution. The correct second
    stage tests the residual between the true potential and the surrogate,

        min(1, exp(-beta ([V(y) - Vtilde(y)] - [V(x) - Vtilde(x)])))

    whose product with the first stage is exactly the Metropolis ratio of the
    true target. Any bias applied to the chain belongs in the first stage and
    cancels out of this residual.

    The better the surrogate, the closer this is to one, and the fewer true
    evaluations are wasted on moves that are then rejected.

    :param1 TrialEnergy:      true energy of the trial structure (in eV).
    :param2 CurrentEnergy:    true energy of the current structure (in eV).
    :param3 TrialSurrogate:   surrogate energy of the trial structure (in eV).
    :param4 CurrentSurrogate: surrogate energy of the current structure (in eV).
    :param5 tau:              time parameter tau (float), in fs.

    :return: logP: logarithm of the probability.

    """

    return calcPELogProbability(TrialEnergy - TrialSurrogate,
                                CurrentEnergy - CurrentSurrogate, tau)
