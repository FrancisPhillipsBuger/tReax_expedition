#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Tests for the numerically exact committor of the two-dimensional model, which
is what the learned one is judged against. Its correctness is checked against
a symmetry of the model rather than against another numerical solution.

Created on: Tue Aug 25 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import numpy as np

from constants.units import BetaEff
from analytic_2d.reference import BuildGrid, SolveCommittor

#%% 1. Grid
def test_the_potential_has_the_expected_symmetry():

    """

    The model is invariant under sending both coordinates to their negatives,
    which is the symmetry every other check here rests on. It is not symmetric
    in x alone, because of the bilinear coupling to the bath.

    """

    x, y, V = BuildGrid(XLim=(-1.0, 1.0), YLim=(-0.5, 0.5), NX=41, NY=31)

    assert np.allclose(x, -x[::-1])
    assert np.allclose(y, -y[::-1])
    assert np.abs(V - V[::-1, ::-1]).max() < 1e-12
    assert np.abs(V - V[::-1, :]).max() > 1e-3

#%% 2. Committor
def test_the_committor_obeys_the_symmetry_of_the_model():

    """

    Under the symmetry the two basins exchange, so the committor must map onto
    one minus itself. Nothing in the solver enforces this, so it is a genuine
    check on the discretisation.

    """

    x, y, V = BuildGrid(XLim=(-1.0, 1.0), YLim=(-0.5, 0.5), NX=81, NY=61)
    Q = SolveCommittor(x, y, V, tau=5.0, BasinRadius=0.22)

    Residual = Q + Q[::-1, ::-1] - 1.0
    W = np.exp(-BetaEff(5.0)*(V - V.min()))
    W = W/W.sum()

    assert np.abs(Residual).max() < 5.0e-2
    assert float(np.sum(W*np.abs(Residual))) < 1.0e-3

def test_the_committor_is_bounded_and_pinned_in_the_basins():

    """Boundary conditions and the range of a probability."""

    x, y, V = BuildGrid(XLim=(-1.0, 1.0), YLim=(-0.5, 0.5), NX=81, NY=61)
    Q = SolveCommittor(x, y, V, tau=5.0, BasinRadius=0.22)

    assert Q.min() >= 0.0
    assert Q.max() <= 1.0
    Middle = len(y)//2
    assert Q[2, Middle] < 0.1        # deep on the reactant side
    assert Q[-3, Middle] > 0.9       # deep on the product side

def test_a_colder_ensemble_sharpens_the_committor():

    """

    As the effective temperature falls the committor turns over more abruptly,
    which is exactly what makes it awkward to use directly as a collective
    variable and why the latent variable is used instead.

    """

    x, y, V = BuildGrid(XLim=(-1.0, 1.0), YLim=(-0.5, 0.5), NX=81, NY=61)
    Middle = len(y)//2

    Widths = []
    for tau in (2.0, 10.0):
        Q = SolveCommittor(x, y, V, tau=tau, BasinRadius=0.22)[:, Middle]
        Widths.append(float(np.sum((Q > 0.1) & (Q < 0.9))))

    assert Widths[1] < Widths[0]
