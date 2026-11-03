#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Public API for the analyse package: estimators for the reaction barrier of a
sampled MCTSS run, the bin-averaged transition state structure, and the
extrapolation of the barriers to zero kT, used by analyse_barrier.py; the
walkers' samples or the weighted midpoints of a pair can be binned.

Created on: Tue Aug 25 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Public re-exports
from .barrier import (BarriersFromProfile, BinnedProfile, BinnedStructures,
                      BootstrapBarriers, ExtrapolateToZeroKT, LoadMidpoints,
                      LoadRun, LoadRunStructures, SpreadErrors, TauFromTimeColumn)

__all__ = [
    'LoadRun',
    'LoadRunStructures',
    'LoadMidpoints',
    'TauFromTimeColumn',
    'BinnedProfile',
    'BinnedStructures',
    'BarriersFromProfile',
    'SpreadErrors',
    'BootstrapBarriers',
    'ExtrapolateToZeroKT',
]
