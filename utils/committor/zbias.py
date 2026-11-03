#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Adaptive bias along the committor-related collective variable.


Created on: Tue Aug 25 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import numpy as np
import torch

from constants.units import kTEffeV

#%% 1. Adaptive bias
class AdaptiveZBias:

    """

    Well-tempered bias on a one-dimensional grid along the latent variable.

    """

    def __init__(self, Model, tau, ZLim=(-8.0, 8.0), NBins=201, Height=None,
                 Width=0.25, BiasFactor=10.0):

        """

        :param1 Model:      a CommittorNN providing the latent variable.
        :param2 tau:        time parameter tau (float), in fs.
        :param3 ZLim:       range of the latent variable covered by the grid.
        :param4 NBins:      number of grid points.
        :param5 Height:     initial deposition height in eV, default kT_eff/2.
        :param6 Width:      width of the deposited kernel, in units of z.
        :param7 BiasFactor: well-tempered bias factor; larger explores further.

        """

        self.Model = Model
        self.kT = kTEffeV(tau)
        self.Grid = np.linspace(ZLim[0], ZLim[1], NBins)
        self.Bias = np.zeros(NBins)
        self.Height = 0.5*self.kT if Height is None else float(Height)
        self.Width = float(Width)
        self.BiasFactor = float(BiasFactor)
        self.Frozen = False

    @classmethod
    def FromSample(cls, Model, tau, X, Margin=2.0, **Kwargs):

        """

        Build a bias whose grid is sized from the latent range of a sample.

        The scale of the latent variable is not fixed by the loss: only the
        product of z and the sigmoid slope enters the committor, so every
        refit is free to stretch z further. A grid fixed once therefore stops
        covering the basins after a few iterations, the interpolation flattens
        outside it, deposition stops having any effect there and the walker
        is trapped in whichever basin it started from. Sizing the grid from
        the data each time removes that failure mode.

        :param1 Model:  a CommittorNN providing the latent variable.
        :param2 tau:    time parameter tau (float), in fs.
        :param3 X:      configurations used to measure the latent range.
        :param4 Margin: padding added on both sides, in units of z.
        :param5 Kwargs: forwarded to the constructor.

        """

        Probe = cls(Model, tau, **Kwargs)
        z = Probe.Latent(X)
        Kwargs["ZLim"] = (float(z.min()) - Margin, float(z.max()) + Margin)

        return cls(Model, tau, **Kwargs)

    def Latent(self, X):

        """

        Latent variable of a batch of configurations.

        :param1 X: flat coordinates, shape (batch, NumCoordinates).

        :return: z: latent variable, shape (batch,).

        """

        with torch.no_grad():
            t = torch.as_tensor(np.ascontiguousarray(np.atleast_2d(X)), dtype=torch.get_default_dtype())

            return self.Model(t).numpy()

    def Values(self, X):

        """

        Bias of a batch of configurations, in eV.

        :param1 X: flat coordinates, shape (batch, NumCoordinates).

        :return: V: the bias of every configuration, in eV.

        """

        return np.interp(self.Latent(X), self.Grid, self.Bias)

    def __call__(self, X):

        """

        Bias of a single configuration, in eV.

        :param1 X: flat coordinates of one configuration.

        :return: V: the bias, in eV.

        """

        return float(self.Values(np.reshape(np.asarray(X, dtype=float), (1, -1)))[0])

    def Deposit(self, X):

        """

        Add one well-tempered kernel at the current configuration.

        :param1 X: flat coordinates of the current configuration.

        :return: None.

        """

        if self.Frozen:
            return None

        z = float(self.Latent(np.reshape(np.asarray(X, dtype=float), (1, -1)))[0])
        Here = np.interp(z, self.Grid, self.Bias)
        # well-tempered scaling: deposition slows down where the bias is deep,
        # so the bias converges instead of growing without bound
        h = self.Height*np.exp(-Here/(self.kT*(self.BiasFactor - 1.0)))
        self.Bias += h*np.exp(-0.5*((self.Grid - z)/self.Width)**2)

        return None

    def Freeze(self):

        """

        Stop depositing, so that reweighting by this bias is exact.

        :return: None.

        """

        self.Frozen = True

        return None

#%% 2. Composition
class SumOfBiases:

    """

    Sum of several bias terms, forwarding deposition to the adaptive ones.

    """

    def __init__(self, Terms):

        """

        :param1 Terms: list of bias callables.

        """

        self.Terms = list(Terms)

    def Values(self, X):

        """

        :param1 X: flat coordinates, shape (batch, NumCoordinates).

        :return: V: total bias of every configuration, in eV.

        """

        return sum(t.Values(X) for t in self.Terms)

    def __call__(self, X):

        """

        :param1 X: flat coordinates of one configuration.

        :return: V: the total bias, in eV.

        """

        return float(sum(t(X) for t in self.Terms))

    def Deposit(self, X):

        """

        :param1 X: flat coordinates of the current configuration.

        :return: None.

        """

        for t in self.Terms:
            if hasattr(t, "Deposit"):
                t.Deposit(X)

        return None

    def Freeze(self):

        """

        :return: None.

        """

        for t in self.Terms:
            if hasattr(t, "Freeze"):
                t.Freeze()

        return None
