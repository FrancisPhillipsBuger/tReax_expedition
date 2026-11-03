#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Public API for the committor package: differentiable descriptors, the neural
parametrisation of the committor, the variational and shot losses, the
Kolmogorov and adaptive biases, and the MBAR pooling of biased blocks.

Created on: Tue Aug 25 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Public re-exports
from .bias import KolmogorovBias
from .zbias import AdaptiveZBias, SumOfBiases
from .descriptors import Identity, PairwiseDistances
from .data import LabelledStore, MBARWeights, TruncateWeights
from .loss import BinomialShotLoss, CommittorLoss, MassWeightedGradientSquared
from .model import (BuildDescriptors, CommittorNN, CommittorOf, DescriptorSpecification,
                    LoadCommittor, SaveCommittor)
from .train import CommittorDataset, Fit
from .validate import AssessCommittor

__all__ = [
    'Identity',
    'PairwiseDistances',
    'CommittorNN',
    'CommittorOf',
    'SaveCommittor',
    'LoadCommittor',
    'DescriptorSpecification',
    'BuildDescriptors',
    'CommittorLoss',
    'BinomialShotLoss',
    'LabelledStore',
    'MBARWeights',
    'TruncateWeights',
    'MassWeightedGradientSquared',
    'CommittorDataset',
    'Fit',
    'AssessCommittor',
    'KolmogorovBias',
    'AdaptiveZBias',
    'SumOfBiases',
]
