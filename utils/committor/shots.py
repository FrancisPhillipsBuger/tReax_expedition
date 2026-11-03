#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Shooting in the AIMMD sense with Monte-Carlo chains: from a shooting point,
independent unbiased chains at the physical tau run until they enter the core
of one of the two states. The outcomes are binomial draws of the committor.
The shooting points are picked around q = 1/2 of the current model with a
Lorentzian in the latent, flattened against their density.

Created on: Mon Sep 28 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import numpy as np

from utils.mc.walk import Walk

#%% 1. Core sets
class CoreSets:

    """

    The two states as a shot must see them: the right fingerprint and a
    progress coordinate within a fraction of the way from the state's own
    minimum, so that a configuration on the ridge never counts as committed.

    """

    def __init__(self, Book, States, Coordinate, Fraction=0.2):

        """

        :param1 Book:       StateBook of the search.
        :param2 States:     names of the first and second state.
        :param3 Coordinate: progress coordinate of the pair, callable X -> xi.
        :param4 Fraction:   largest share of the xi distance between the two
                            minima that still counts as inside a core.

        """

        self.Book = Book
        self.States = tuple(States)
        self.Coordinate = Coordinate
        self.Xi = [Coordinate(Book.Get(s)["X"]) for s in self.States]
        self.Fraction = float(Fraction)

    def __call__(self, X):

        """

        :param1 X: coordinates.

        :return: Name: the state whose core holds X, or None.

        """

        s = (self.Coordinate(X) - self.Xi[0])/(self.Xi[1] - self.Xi[0])
        if s <= self.Fraction:
            Name = self.States[0]
        elif s >= 1.0 - self.Fraction:
            Name = self.States[1]
        else:
            return None

        return Name if self.Book.Identify(X) == Name else None

#%% 2. Shots
def Shoot(System, X, tau, Cores, MaxSteps):

    """

    One unbiased chain from a shooting point until it enters a core.

    :param1 System:   AnalyticSystem or AtomsSystem.
    :param2 X:        shooting point.
    :param3 tau:      time parameter tau (float), in fs.
    :param4 Cores:    CoreSets.
    :param5 MaxSteps: budget of the chain.

    :return: Outcome, NSteps: the state reached, None when undecided.

    """

    R = Walk(System, X, tau, MaxSteps, Stride=MaxSteps + 1, Stop=Cores)

    return R["Outcome"], R["NSteps"]

#%% 3. Selection
def SelectShootingPoints(Latent, Candidates, N, Width=0.5, Rng=None):

    """

    Draw shooting points around z = 0 (q = 1/2), AIMMD style.

    Each candidate is weighted by a Lorentzian in the latent divided by the
    density of candidates at its latent value, so a region the sampling
    visited often is not picked more for that reason alone.

    :param1 Latent:     callable returning z of a batch, shape (n,).
    :param2 Candidates: configurations, shape (n, NumCoordinates).
    :param3 N:          points to draw, without replacement.
    :param4 Width:      half width of the Lorentzian in z.
    :param5 Rng:        numpy Generator, or None.

    :return: Index: indices of the chosen candidates.

    """

    Rng = np.random.default_rng() if Rng is None else Rng
    z = np.asarray(Latent(Candidates), dtype=float)
    Hist, Edges = np.histogram(z, bins=min(30, max(2, len(z)//10)))
    Bin = np.clip(np.digitize(z, Edges[1:-1]), 0, len(Hist) - 1)
    w = 1.0/(1.0 + (z/Width)**2)/np.maximum(Hist[Bin], 1)
    w = w/w.sum()
    N = min(N, int(np.sum(w > 0)))

    return Rng.choice(len(z), size=N, replace=False, p=w)
