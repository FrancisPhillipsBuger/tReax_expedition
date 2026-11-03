#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

The transition state ensemble at the physical temperature, sampled by a pair
held on the two sides of the dividing surface xi = xi* through the S5
crossing: each walker at beta*/2, so the weighted midpoint samples
exp(-beta* V) on a slab of the surface whose width the coupling sets.

Created on: Mon Sep 28 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
from dataclasses import dataclass

import numpy as np

from utils.mc.ensemble import NormalisedWeights, PairEnsemble
from utils.mc.pair import PairWalk

#%% 1. Settings and dividing surface
@dataclass
class TSESettings:

    """

    :param1 T12:    tau12 in units of the bead tau; small is stiff, and the
                    midpoint weights stay near one while (omega tau_b)^2 T12
                    is well below 8 for the stiffest mode.
    :param2 Steps:  recorded sweeps.
    :param3 Tune:   sweeps of proposal-width adaptation before recording.
    :param4 Stride: sweeps between two midpoint energies.

    """

    T12: float = 0.05
    Steps: int = 20000
    Tune: int = 2000
    Stride: int = 10

class DividingSurface:

    """

    Signed distance Sign (xi(X) - xi*), negative on the side of the first state.

    """

    def __init__(self, Coordinate, XiStar, XA):

        """

        :param1 Coordinate: callable X -> xi.
        :param2 XiStar:     xi on the surface.
        :param3 XA:         a configuration of the first state.

        """

        self.Coordinate = Coordinate
        self.XiStar = float(XiStar)
        self.Sign = -1.0 if Coordinate(XA) > self.XiStar else 1.0

    def __call__(self, X):

        """

        :return: s: signed distance to the surface along xi.

        """

        return self.Sign*(float(self.Coordinate(X)) - self.XiStar)

#%% 2. The ensemble
def StartingPair(Nodes, Side):

    """

    :param1 Nodes: configurations along a chain.
    :param2 Side:  DividingSurface.

    :return: X1, X2: the node nearest the surface on each side, or None.

    """

    s = np.array([Side(X) for X in Nodes])
    Below = np.flatnonzero(s < 0.0)
    Above = np.flatnonzero(s > 0.0)
    if len(Below) == 0 or len(Above) == 0:
        return None

    return Nodes[Below[np.argmax(s[Below])]], Nodes[Above[np.argmin(s[Above])]]

def SampleTSE(System, Coordinate, Crossing, Nodes, XA, TauPhys, Settings=None):

    """

    Pair sampling of the transition state ensemble at tau_phys.

    The pair starts from the chain nodes nearest the surface on either side;
    its midpoint samples exp(-beta* V(M)) times the probability that the two
    walkers straddle the surface, a window only of the distance of M to it,
    so conditional averages at fixed xi(M) are those of the Boltzmann
    distribution on the surface.

    :param1 System:     AnalyticSystem or AtomsSystem.
    :param2 Coordinate: callable X -> xi, the coordinate of the chain.
    :param3 Crossing:   the S5 crossing, which fixes xi*.
    :param4 Nodes:      configurations along the chain.
    :param5 XA:         a configuration of the first state.
    :param6 TauPhys:    tau of the physical temperature, in fs.
    :param7 Settings:   TSESettings, or None for the defaults.

    :return: Result: dict of PairWalk plus Xi (of every midpoint), XiStar,
             Ensemble, Weights and EMean (weighted mean midpoint energy, eV);
             None when no node lies on one of the sides.

    """

    Settings = TSESettings() if Settings is None else Settings
    Side = DividingSurface(Coordinate, Coordinate(Crossing), XA)
    Start = StartingPair(Nodes, Side)
    if Start is None:
        return None
    Ensemble = PairEnsemble.AtMidpoint(TauPhys, Settings.T12)
    R = PairWalk(System, Start[0], Start[1], Ensemble, Settings.Steps, Ensemble.Tau12, Side=Side,
                 Stride=Settings.Stride, Tune=Settings.Tune)
    R["Xi"] = np.array([float(Coordinate(X)) for X in R["M"]])
    R["XiStar"] = Side.XiStar
    R["Ensemble"] = Ensemble
    R["Weights"] = NormalisedWeights(R["LogW"])
    R["EMean"] = float(np.sum(R["Weights"]*R["VM"])) if len(R["VM"]) > 0 else np.nan

    return R
