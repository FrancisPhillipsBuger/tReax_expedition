#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

The Kolmogorov bias potential, evaluated as a value only.

Created on: Tue Aug 25 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import numpy as np
import torch

from constants.units import BetaEff
from .loss import MassWeightedGradientSquared

#%% 1. Default regularisation
# the reference plumed.dat uses EPSILON=1e-7 in the gradient logarithm and
# 1e-7 inside the two committor logarithms
DefaultEpsilon = 1.0e-7
DefaultDelta = 1.0e-7

#%% 2. Bias
class KolmogorovBias:

    """

    Callable returning the Kolmogorov bias of a configuration, in eV.

    The bias is V_K = -(lambda/beta) log|grad q|^2, with the gradient taken in
    mass-weighted coordinates. Since q = sigmoid(s z), log|grad q|^2 equals
    log|grad z|^2 + 2 log q(1-q) up to a constant, and it is evaluated in that
    form, each logarithm regularised on its own as in the reference plumed.dat.
    The gradient is therefore the one of z: taking it of q as well would count
    the factor q(1-q) twice.

    """

    def __init__(self, Model, Masses, tau, Lam=1.0,
                 Epsilon=DefaultEpsilon, Delta=DefaultDelta):

        """

        :param1 Model:   a trained CommittorNN.
        :param2 Masses:  per coordinate masses, shape (NumCoordinates,).
        :param3 tau:     time parameter tau (float), in fs, setting beta_eff.
        :param4 Lam:     prefactor of the bias; one in the reference, the
                         alanine input uses two. The gradient term is never
                         negative and each committor logarithm is at least
                         log(Delta), so V_K <= 2 lambda |log Delta|/beta.
        :param5 Epsilon: regularisation inside the gradient logarithm.
        :param6 Delta:   regularisation inside the committor logarithms.

        """

        self.Model = Model
        self.Masses = torch.as_tensor(Masses, dtype=torch.get_default_dtype())
        self.Beta = BetaEff(tau)
        self.Lam = float(Lam)
        self.Epsilon = float(Epsilon)
        self.Delta = float(Delta)

    def Values(self, X):

        """

        Bias of a batch of configurations.

        :param1 X: flat coordinates, shape (batch, NumCoordinates).

        :return: V: bias of every configuration, in eV.

        """

        if isinstance(X, np.ndarray):
            # torch refuses the negative strides of a reversed or sliced view
            X = np.ascontiguousarray(X)
        X = torch.as_tensor(X, dtype=torch.get_default_dtype()).reshape(1, -1) \
            if np.ndim(X) == 1 else torch.as_tensor(X, dtype=torch.get_default_dtype())

        G2, q = MassWeightedGradientSquared(self.Model, X, self.Masses,
                                            CreateGraph=False, OfLatent=True)
        G2 = G2.detach()
        q = q.detach()

        # the two parts of log|grad q|^2; lambda scales the whole of it
        ZBias = torch.log(G2 + self.Epsilon) - np.log(self.Epsilon)
        LogQ = torch.log(q + self.Delta) + torch.log(1.0 - q + self.Delta)
        V = -self.Lam*(ZBias + 2.0*LogQ)/self.Beta

        return V.numpy()

    def __call__(self, X):

        """

        Bias of a single configuration.

        :param1 X: flat coordinates of one configuration.

        :return: V: the bias, in eV.

        """

        return float(self.Values(np.reshape(np.asarray(X, dtype=float), (1, -1)))[0])
