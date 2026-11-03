#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Public API for the path package: states and their fingerprints, the
zero-temperature labels along MCTSS paths, the progress coordinates, the
certified continuation chains, the separatrix crossing, and the staged
search that strings them together.

Created on: Mon Sep 28 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Public re-exports
from .states import BondGraph, DihedralBox, Region, StateBook
from .label import Labeller, LabelPath, LabelSchedule
from .coordinate import BondChange, DihedralChange, Latent, TrainLabelClassifier
from .continuation import BuildChain, Chain, ChainSettings, Channels, MoveLength
from .bisect import ChainCurvature, LocateCrossing
from .search import RunSearch, SearchSettings

__all__ = [
    'BondGraph',
    'DihedralBox',
    'Region',
    'StateBook',
    'Labeller',
    'LabelPath',
    'LabelSchedule',
    'BondChange',
    'DihedralChange',
    'Latent',
    'TrainLabelClassifier',
    'BuildChain',
    'Chain',
    'ChainSettings',
    'Channels',
    'MoveLength',
    'ChainCurvature',
    'LocateCrossing',
    'RunSearch',
    'SearchSettings',
]
