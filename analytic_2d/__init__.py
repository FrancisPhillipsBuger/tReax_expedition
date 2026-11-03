#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Public API for the analytic_2d package: the analytic 2D potentials (proton
transfer, Mueller-Brown, two channels), their two-walker SWS Monte-Carlo
sampler, the exact committor, the (tau, Ninter) scan, the PES writers and the
command-line driver.

Created on: Mon Aug 24 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Public re-exports
from .potential import (EnergyUnitAt, GetPotential, MuellerBrown, Potential2D,
                        PotentialEnergy, PotentialGradient, Potentials,
                        ProtonTransfer, ReducedPotential, TwoPaths, Vpt, X0,
                        alpha, mass, vbath)
from .sampler import GenDisplacement, RunSingleMCTrajectory, calcKELogProbability
from .analysis import CompressPath, ConvPairMeans, SaveMeanPath
from .pes_io import SaveCuts, SavePES, SaveStationaryPoints
from .scan import ClosestLabel, RunParameterScan
from .reference import BuildGrid, ReferenceAtSaddles, SolveCommittor

__all__ = [
    'Potential2D',
    'ProtonTransfer',
    'ReducedPotential',
    'MuellerBrown',
    'TwoPaths',
    'Potentials',
    'GetPotential',
    'EnergyUnitAt',
    'PotentialEnergy',
    'PotentialGradient',
    'Vpt',
    'X0',
    'vbath',
    'alpha',
    'mass',
    'GenDisplacement',
    'calcKELogProbability',
    'RunSingleMCTrajectory',
    'CompressPath',
    'SaveMeanPath',
    'ConvPairMeans',
    'SavePES',
    'SaveCuts',
    'SaveStationaryPoints',
    'ClosestLabel',
    'RunParameterScan',
    'BuildGrid',
    'SolveCommittor',
    'ReferenceAtSaddles',
]
