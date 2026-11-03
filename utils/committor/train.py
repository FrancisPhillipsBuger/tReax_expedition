#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Optimisation of the committor network on a dataset of sampled configurations.

One call to Fit is one training cycle of the self-consistent procedure: the
model is fitted on the configurations gathered so far, and the caller then
runs new biased trajectories with the frozen result.

Created on: Tue Aug 25 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import numpy as np
import torch

from .loss import BinomialShotLoss, CommittorLoss

#%% 1. Dataset
class CommittorDataset:

    """

    Sampled configurations with their weights and the two labelled basins.

    """

    def __init__(self, X, Weights, XA, XB, Shots=None):

        """

        :param1 X:       sampled configurations, shape (n, NumCoordinates).
        :param2 Weights: their Boltzmann weights, shape (n,).
        :param3 XA:      configurations labelled as the reactant basin.
        :param4 XB:      configurations labelled as the product basin.
        :param5 Shots:   (XS, NA, NB) shooting points and outcomes, or None.

        """

        dt = torch.get_default_dtype()
        self.X = torch.as_tensor(np.ascontiguousarray(np.atleast_2d(X)), dtype=dt)
        self.Weights = torch.as_tensor(np.ascontiguousarray(np.atleast_1d(Weights)), dtype=dt)
        self.XA = torch.as_tensor(np.ascontiguousarray(np.atleast_2d(XA)), dtype=dt)
        self.XB = torch.as_tensor(np.ascontiguousarray(np.atleast_2d(XB)), dtype=dt)
        if len(self.Weights) != len(self.X):
            raise ValueError("One weight per configuration is required")
        self.Shots = None
        if Shots is not None and len(Shots[0]) > 0:
            self.Shots = tuple(torch.as_tensor(np.ascontiguousarray(np.atleast_1d(a) if k else np.atleast_2d(a)), dtype=dt)
                               for k, a in enumerate(Shots))

    def __len__(self):

        """

        :return: N: number of sampled configurations.

        """

        return len(self.X)

#%% 2. Fit
def Fit(Model, Dataset, Masses, Alpha=1.0, Epochs=500, BatchSize=256,
        LearningRate=1.0e-3, Seed=None, Verbose=False, Gamma=1.0, RelativeGamma=True):

    """

    Minimise the variational loss on a dataset.

    :param1 Model:        a CommittorNN to be optimised in place.
    :param2 Dataset:      a CommittorDataset.
    :param3 Masses:       per coordinate masses.
    :param4 Alpha:        weight of the boundary condition term.
    :param5 Epochs:       number of passes over the dataset.
    :param6 BatchSize:    size of the minibatches drawn from the sample.
    :param7 LearningRate: step size of the Adam optimiser.
    :param8 Seed:         seed of the shuffling, or None.
    :param9 Verbose:      print the loss every tenth of the run.
    :param10 Gamma:       weight of the binomial likelihood of the shots, when
                          the dataset carries any; the total loss holds it.
    :param11 RelativeGamma: measure Gamma in units of the ratio of the Ritz
                          term to the shot likelihood at the start of the fit.
                          The Ritz term is a Boltzmann average of |grad q|^2,
                          orders of magnitude below a likelihood per shot, so
                          an absolute weight lets a few dozen shots decide
                          the model everywhere.

    :return: History: per epoch list of (Loss, Lv, Lb).

    """

    if Seed is not None:
        torch.manual_seed(Seed)

    Loss = CommittorLoss(Masses, Alpha=Alpha)
    Optimiser = torch.optim.Adam(Model.parameters(), lr=LearningRate)
    Generator = np.random.default_rng(Seed)

    N = len(Dataset)
    History = []
    Weight = Gamma
    if Dataset.Shots is not None and Gamma > 0.0 and RelativeGamma:
        _, Lv0, _ = Loss(Model, Dataset.X, Dataset.Weights, Dataset.XA, Dataset.XB)
        NLL0 = BinomialShotLoss(Model, *Dataset.Shots)
        Weight = Gamma*float(Lv0.detach())/max(float(NLL0.detach()), 1.0e-12)
    for e in range(0, Epochs):
        Order = Generator.permutation(N)
        Totals = np.zeros(3)
        NBatch = 0
        for s in range(0, N, BatchSize):
            idx = Order[s:s+BatchSize]
            if len(idx) < 2:
                continue
            Optimiser.zero_grad()
            L, Lv, Lb = Loss(Model, Dataset.X[idx], Dataset.Weights[idx],
                             Dataset.XA, Dataset.XB)
            if Dataset.Shots is not None and Weight > 0.0:
                L = L + Weight*BinomialShotLoss(Model, *Dataset.Shots)
            L.backward()
            Optimiser.step()
            Totals += np.array([L.item(), Lv.item(), Lb.item()])
            NBatch += 1
        if NBatch > 0:
            History.append(tuple(Totals/NBatch))
            if Verbose and (e % max(1, Epochs//10) == 0):
                print("    epoch %5i  loss %12.6g  Lv %12.6g  Lb %12.6g" %
                      ((e,) + History[-1]))

    return History
