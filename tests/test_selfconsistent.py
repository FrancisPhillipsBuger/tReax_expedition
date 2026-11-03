#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Tests for the pieces of one self-consistent committor iteration: the bias
the walkers run under, whose deposits and freeze have to reach its adaptive
term, and the array layouts the model and the biases are handed.

Created on: Sat Sep 26 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import numpy as np
import torch

from utils.committor import (AdaptiveZBias, CommittorNN, CommittorOf, Identity,
                             KolmogorovBias, PairwiseDistances)
from utils.committor.stage import BuildIterationBias
from utils.committor.train import CommittorDataset

#%% 1. Bias of one iteration
def test_deposits_and_freezes_reach_the_adaptive_term():

    """The iteration bias grows along z while it explores and stops once frozen."""

    torch.manual_seed(0)
    Model = CommittorNN(PairwiseDistances(3))
    X = np.random.default_rng(0).normal(size=(20, 9))
    Bias = BuildIterationBias(Model, np.ones(9), 20.0, X)
    Z = Bias.Terms[1]
    assert isinstance(Bias.Terms[0], KolmogorovBias) and isinstance(Z, AdaptiveZBias)

    Before = Bias(X[0])
    Bias.Deposit(X[0])
    Grown = Z.Bias.copy()
    assert Grown.max() > 0.0
    assert Bias(X[0]) > Before

    Bias.Freeze()
    Bias.Deposit(X[0])
    assert np.array_equal(Z.Bias, Grown)

def test_the_latent_term_is_left_out_on_request():

    """ZBias=False runs on the Kolmogorov bias alone."""

    Model = CommittorNN(Identity(2))
    Bias = BuildIterationBias(Model, np.ones(2), 10.0, np.zeros((4, 2)), ZBias=False)

    assert len(Bias.Terms) == 1

#%% 2. Array layout
def test_reversed_arrays_are_accepted_everywhere():

    """A path reversed with [::-1] has negative strides, which torch refuses unless copied."""

    torch.manual_seed(0)
    Model = CommittorNN(Identity(2))
    X = np.random.default_rng(0).random((5, 2))[::-1]
    assert np.allclose(CommittorOf(Model, X), CommittorOf(Model, X.copy()))
    Bias = KolmogorovBias(Model, np.ones(2), 10.0)
    assert np.allclose(Bias.Values(X), Bias.Values(X.copy()))
    assert AdaptiveZBias(Model, 10.0, ZLim=(-5.0, 5.0)).Values(X).shape == (5,)
    assert len(CommittorDataset(X, np.ones(5)[::-1], X, X)) == 5
