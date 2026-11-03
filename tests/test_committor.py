#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Tests for the committor package: the differentiable descriptors, the
mass-weighted gradient that the variational principle needs, the numerical
safety of the Kolmogorov bias, the capping of weights and the adaptive bias.

Created on: Tue Aug 25 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import math

import numpy as np
import pytest
import torch

from constants.units import BetaEff
from utils.committor import (AdaptiveZBias, CommittorNN, Identity, KolmogorovBias,
                             MassWeightedGradientSquared, PairwiseDistances,
                             SumOfBiases, TruncateWeights)

def Bound(Bias):

    """Upper bound 2 lambda |log Delta|/beta of the Kolmogorov bias, in eV."""

    return abs(2.0*Bias.Lam*np.log(Bias.Delta))/Bias.Beta

def Kish(Weights):

    """Kish effective sample size of a set of weights."""

    w = np.asarray(Weights, dtype=float)

    return float(w.sum()**2/np.sum(w*w))

#%% 1. Descriptors
def test_pairwise_distance_count():

    """Six atoms give the fifteen distances used for the SN2 system."""

    assert PairwiseDistances(6).NumDescriptors == 15
    assert PairwiseDistances(10).NumDescriptors == 45   # the alanine descriptor set

def test_pairwise_distances_are_translation_and_rotation_invariant():

    """This is why they replace the centre of mass removal and the alignment."""

    torch.manual_seed(0)
    d = PairwiseDistances(6)
    X = torch.randn(1, 18, dtype=torch.float64)*1.5
    R = X.view(6, 3)

    Shifted = (R + torch.tensor([1.3, -0.7, 2.0], dtype=torch.float64)).reshape(1, 18)
    a = 0.7
    Rot = torch.tensor([[math.cos(a), -math.sin(a), 0.0],
                        [math.sin(a), math.cos(a), 0.0],
                        [0.0, 0.0, 1.0]], dtype=torch.float64)
    Rotated = (R @ Rot.T).reshape(1, 18)

    assert torch.allclose(d(X), d(Shifted), atol=1e-12)
    assert torch.allclose(d(X), d(Rotated), atol=1e-12)

def test_pairwise_distances_are_permutation_covariant():

    """Swapping two identical atoms reorders the set instead of changing it."""

    torch.manual_seed(1)
    d = PairwiseDistances(4)
    X = torch.randn(1, 12, dtype=torch.float64)
    R = X.view(4, 3)
    Swapped = R[[1, 0, 2, 3]].reshape(1, 12)

    assert torch.allclose(torch.sort(d(X))[0], torch.sort(d(Swapped))[0], atol=1e-12)

#%% 2. The gradient the variational principle needs
def test_mass_weighted_gradient_matches_finite_differences():

    """

    The gradient is taken with respect to the mass-weighted coordinates and
    the chain rule has to run back through the descriptor layer. Getting this
    wrong is silent, so it is checked against finite differences.

    """

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(3)

    Masses = torch.tensor([12.011, 1.008, 1.008, 1.008, 35.45, 79.904]).repeat_interleave(3)
    Model = CommittorNN(PairwiseDistances(6))
    X = torch.randn(1, 18)*1.5

    G2, _ = MassWeightedGradientSquared(Model, X, Masses, CreateGraph=False)

    h = 1.0e-6
    fd = torch.zeros(18, dtype=torch.float64)
    for k in range(0, 18):
        Xp = X.clone()
        Xp[0, k] += h
        Xm = X.clone()
        Xm[0, k] -= h
        fd[k] = (Model.Committor(Xp) - Model.Committor(Xm)).item()/(2.0*h)

    assert float(G2) == pytest.approx(float(torch.sum(fd*fd/Masses)), rel=1e-6)

#%% 3. Numerical safety of the bias
def test_bias_is_finite_and_bounded_everywhere():

    """

    The two regularisations exist so the bias never diverges where the
    committor is flat, which is most of configuration space.

    """

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    Model = CommittorNN(Identity(2))
    Bias = KolmogorovBias(Model, np.ones(2), tau=4.0, Lam=1.0)

    X = np.random.RandomState(0).uniform(-50.0, 50.0, size=(5000, 2))
    V = Bias.Values(X)

    assert np.all(np.isfinite(V))
    assert np.all(np.abs(V) <= Bound(Bias) + 1e-9)

def test_bias_bound_scales_with_the_regularisation():

    """The cap is set by the regularisation inside the committor logarithms."""

    torch.manual_seed(0)
    Model = CommittorNN(Identity(2))
    Loose = KolmogorovBias(Model, np.ones(2), tau=4.0, Delta=1e-3)
    Tight = KolmogorovBias(Model, np.ones(2), tau=4.0, Delta=1e-7)

    assert Bound(Tight) > Bound(Loose)

def test_bias_in_thermal_units_does_not_depend_on_tau():

    """

    The bracket the bias is built from is dimensionless and set only by the
    regularisation, so beta times the bias is the same number at every tau.
    The bias measured as an energy therefore scales as one over tau, and the
    spread of the reweighting factors it produces is the same at every
    effective temperature.

    """

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    Model = CommittorNN(Identity(2))
    X = np.array([[0.3, -0.2]])

    Cold = float(KolmogorovBias(Model, np.ones(2), tau=8.0).Values(X)[0])
    Warm = float(KolmogorovBias(Model, np.ones(2), tau=4.0).Values(X)[0])

    # twice the tau, half the energy
    assert Cold == pytest.approx(0.5*Warm, rel=1e-10)
    # and the same number of thermal units
    assert BetaEff(8.0)*Cold == pytest.approx(BetaEff(4.0)*Warm, rel=1e-10)

def test_bias_is_minus_lambda_over_beta_log_grad_q_squared():

    """

    V_K = -(lambda/beta) log|grad q|^2 up to a constant, wherever the two
    regularisations are negligible. Taking the gradient of q rather than of z
    in the log|grad z|^2 + 2 log q(1-q) form would leave a spurious
    -2 log(s q(1-q)) that varies from point to point.

    """

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    Model = CommittorNN(Identity(2))
    Masses = np.array([1.0, 2.0])
    X = np.random.RandomState(1).uniform(-0.5, 0.5, size=(8, 2))
    G2q, q = MassWeightedGradientSquared(Model, torch.as_tensor(X), torch.as_tensor(Masses),
                                         CreateGraph=False)
    assert np.all((q.detach().numpy() > 1e-3) & (q.detach().numpy() < 1.0 - 1e-3))

    for Lam in (0.5, 1.0, 2.0):
        Bias = KolmogorovBias(Model, Masses, tau=4.0, Lam=Lam)
        Rest = BetaEff(4.0)*Bias.Values(X)/Lam + np.log(G2q.detach().numpy())
        # the constant is 2 log s + log(epsilon)
        assert np.allclose(Rest, 2.0*np.log(Model.Slope) + np.log(Bias.Epsilon), atol=1e-4)

def test_bias_bound_is_reached_in_a_flat_basin_for_any_lambda():

    """

    Where z is flat the gradient term vanishes and q sits at zero, so the bias
    reaches its upper bound 2 lambda |log Delta|/beta, which has to hold for
    every lambda, the small ones included.

    """

    torch.set_default_dtype(torch.float64)
    Model = CommittorNN(Identity(2))
    with torch.no_grad():
        Model.Net[-1].weight.zero_()
        Model.Net[-1].bias.fill_(-20.0)
    X = np.array([[0.3, -0.2], [5.0, 1.0]])

    for Lam in (0.25, 1.0, 2.0):
        Bias = KolmogorovBias(Model, np.ones(2), tau=4.0, Lam=Lam)
        V = Bias.Values(X)
        assert np.all(V <= Bound(Bias) + 1e-12)
        assert np.allclose(V, Bound(Bias), rtol=1e-6)

#%% 4. Capped weights
def test_truncation_rescues_a_degenerate_weight_set():

    """

    A handful of astronomically large weights would otherwise carry the whole
    variational term and the fitted model degenerates. A cap on a quantile is
    unaffected by those outliers, unlike a cap on the mean.

    """

    w = np.exp(np.concatenate([np.zeros(995), [60.0, 58.0, 55.0, 50.0, 45.0]]))

    assert Kish(w) < 5.0
    Capped = TruncateWeights(w)
    assert Kish(Capped) > 900.0
    assert Capped.sum() == pytest.approx(1.0)

def test_truncation_leaves_uniform_weights_alone():

    """Capping must do nothing when there is nothing to cap."""

    w = TruncateWeights(np.ones(400))

    assert Kish(w) == pytest.approx(400.0)

#%% 5. Adaptive bias
def test_adaptive_grid_is_sized_from_the_sample():

    """

    The scale of the latent variable is not fixed by the loss and grows at
    every refit, so a grid fixed once eventually stops covering the basins,
    deposition there stops having any effect and the walker is trapped.

    """

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    Model = CommittorNN(Identity(2))
    X = np.random.RandomState(0).uniform(-3.0, 3.0, size=(500, 2))

    def Covers(Bias):
        z = Bias.Latent(X)
        return z.min() >= Bias.Grid[0] and z.max() <= Bias.Grid[-1]

    assert Covers(AdaptiveZBias.FromSample(Model, 20.0, X, Margin=2.0))
    assert not Covers(AdaptiveZBias(Model, 20.0, ZLim=(0.05, 0.06)))

def test_freezing_stops_deposition():

    """A frozen bias is static, which is what makes the reweighting exact."""

    torch.manual_seed(0)
    Model = CommittorNN(Identity(2))
    Bias = AdaptiveZBias(Model, 20.0, ZLim=(-5.0, 5.0))

    Bias.Deposit(np.array([0.1, 0.2]))
    Filled = Bias.Bias.copy()
    assert Filled.max() > 0.0

    Bias.Freeze()
    Bias.Deposit(np.array([0.1, 0.2]))
    assert np.array_equal(Bias.Bias, Filled)

def test_sum_of_biases_adds_values_and_forwards_deposition():

    """The composite must behave as the sum of its terms."""

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    Model = CommittorNN(Identity(2))
    a = KolmogorovBias(Model, np.ones(2), tau=20.0)
    b = AdaptiveZBias(Model, 20.0, ZLim=(-5.0, 5.0))
    Both = SumOfBiases([a, b])

    X = np.array([0.3, -0.1])
    assert Both(X) == pytest.approx(a(X) + b(X))

    Both.Deposit(X)
    assert b.Bias.max() > 0.0
    Both.Freeze()
    assert b.Frozen
