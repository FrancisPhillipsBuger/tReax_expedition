#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

The ensemble a coupled pair of walkers represents. Each walker (bead) samples
exp(-tau_b V/hbar) and the coupling exp(-kappa mwRMSD/tau12) ties the two, so
the midpoint M samples the ensemble of tau_mid = 2 tau_b up to a factor that
the midpoint weights remove exactly, whatever the coupling.

Created on: Mon Sep 28 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
from dataclasses import dataclass

import numpy as np

from constants.units import BetaEff

from .probability import calcKELogProbabilityFromRMSD

#%% 1. The pair ensemble
@dataclass(frozen=True)
class PairEnsemble:

    """

    Joint target of a pair, exp(-beta_b [V(R1) + V(R2)] - kappa |R1 - R2|^2/tau12).

    With M = (R1 + R2)/2 and D = R1 - R2 the two potentials add up to
    2 V(M) + D.H.D/4 + O(D^4), so the midpoint samples exp(-2 beta_b V(M))
    times det(kappa/tau12 + beta_b H(M)/4)^(-1/2): the ensemble of
    tau_mid = 2 tau_b, exactly so once the weights of LogWeight are applied.

    |R1 - R2| is taken in the common frame of the two walkers: centred, never
    rotated one onto the other (System.FrameDistance). The rotation-minimised
    distance is the Laplace approximation of the integral over their relative
    orientation without its Gaussian width, and that width, sqrt(det I) of the
    rotational orbit, would be left in the midpoint as an extra weight.

    :param1 TauBead: tau of the Boltzmann factor of each walker, in fs.
    :param2 Tau12:   tau of the coupling, in fs.

    """

    TauBead: float
    Tau12: float

    @classmethod
    def AtMidpoint(cls, TauMidpoint, T12):

        """

        :param1 TauMidpoint: tau of the ensemble the midpoint is to sample, in fs.
        :param2 T12:         tau12 in units of the bead tau.

        :return: Ensemble: the pair whose midpoint samples exp(-tau_mid V/hbar).

        """

        TauBead = 0.5*float(TauMidpoint)

        return cls(TauBead, float(T12)*TauBead)

    @property
    def TauMidpoint(self):

        """

        :return: tau: tau of the ensemble the midpoint samples, 2 tau_b, in fs.

        """

        return 2.0*self.TauBead

    @property
    def BeadBeta(self):

        """

        :return: beta: inverse temperature of each walker, in 1/eV.

        """

        return BetaEff(self.TauBead)

    @property
    def MidpointBeta(self):

        """

        :return: beta: inverse temperature of the midpoint ensemble, in 1/eV.

        """

        return BetaEff(self.TauMidpoint)

    def CouplingLog(self, mwRMSD):

        """

        :param1 mwRMSD: sum_i m_i |dr_i|^2 between the walkers, in amu*angstroem**2.

        :return: logP: logarithm of the coupling term.

        """

        return calcKELogProbabilityFromRMSD(mwRMSD, self.Tau12)

    def LogWeight(self, VM, V1, V2):

        """

        Log-weight taking the pair onto exp(-beta_mid V(M) - kappa |D|^2/tau12).

        That target keeps the coupling of the pair and replaces the two walker
        potentials by the potential at the midpoint, so integrating out D
        leaves exactly exp(-beta_mid V(M)). The log-weight is beta_b/4 D.H.D to
        leading order, small when the coupling is stiff.

        :param1 VM: potential at the midpoint, in eV.
        :param2 V1: potential of the first walker, in eV.
        :param3 V2: potential of the second walker, in eV.

        :return: logw: unnormalised log-weight.

        """

        return (-self.MidpointBeta*np.asarray(VM, dtype=float)
                + self.BeadBeta*(np.asarray(V1, dtype=float) + np.asarray(V2, dtype=float)))

#%% 2. Weights
def NormalisedWeights(LogW):

    """

    :param1 LogW: unnormalised log-weights; non-finite entries weigh nothing.

    :return: w: weights summing to one.

    """

    LogW = np.asarray(LogW, dtype=float)
    w = np.zeros(len(LogW))
    Good = np.isfinite(LogW)
    if Good.any():
        w[Good] = np.exp(LogW[Good] - LogW[Good].max())
        w /= w.sum()

    return w

def EffectiveSampleFraction(LogW):

    """

    :param1 LogW: unnormalised log-weights.

    :return: ESS/N: Kish effective sample size over the number of samples.

    """

    w = NormalisedWeights(LogW)
    if len(w) == 0 or w.sum() <= 0.0:
        return 0.0

    return float(1.0/np.sum(w*w)/len(w))

#%% 3. Midpoint
def Midpoint(System, X1, X2):

    """

    :param1 System: AnalyticSystem or AtomsSystem.
    :param2 X1:     coordinates of the first walker.
    :param3 X2:     coordinates of the second walker.

    :return: M: midpoint of the two centred walkers in their common frame.
             The weights of LogWeight are exact for this linear map of the
             coordinates only; aligning one walker onto the other first would
             change the measure of M by a rotational orbit volume.

    """

    return 0.5*(System.Centred(X1) + System.Centred(X2))
