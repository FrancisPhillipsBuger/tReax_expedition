#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Objective function of the variational determination of the committor.


Created on: Tue Aug 25 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import torch

#%% 1. Mass-weighted gradient
def MassWeightedGradientSquared(Model, X, Masses, CreateGraph=True, OfLatent=False):

    """

    Squared gradient of the committor, or of its latent, in mass-weighted coordinates.

    With u_j = sqrt(m_j) x_j the chain rule gives dq/du_j = (dq/dx_j)/sqrt(m_j),
    so the squared norm is a sum of squared Cartesian derivatives divided by
    the corresponding masses.

    :param1 Model:       a CommittorNN.
    :param2 X:           flat coordinates, shape (batch, NumCoordinates).
    :param3 Masses:      per coordinate masses, shape (NumCoordinates,).
    :param4 CreateGraph: keep the graph so the result can be differentiated.
    :param5 OfLatent:    differentiate the latent z instead of q = sigmoid(s z);
                         the two are related by |grad q| = s q (1-q) |grad z|.

    :return: G2, q: the squared mass-weighted gradient and the committor.

    """

    if not X.requires_grad:
        X = X.detach().requires_grad_(True)

    z = Model(X)
    q = Model.CommittorFromLatent(z)
    Target = z if OfLatent else q
    Grad = torch.autograd.grad(Target.sum(), X, create_graph=CreateGraph)[0]

    return torch.sum(Grad*Grad/Masses, dim=-1), q

#%% 2. Loss
class CommittorLoss(torch.nn.Module):

    """

    Variational term plus the boundary condition penalty.

    """

    def __init__(self, Masses, Alpha=1.0):

        """

        :param1 Masses: per coordinate masses, shape (NumCoordinates,).
        :param2 Alpha:  weight of the boundary condition term.

        """

        super().__init__()
        self.register_buffer("Masses", torch.as_tensor(Masses, dtype=torch.get_default_dtype()))
        self.Alpha = float(Alpha)

    def forward(self, Model, X, Weights, XA, XB):

        """

        Evaluate the loss on a batch.

        :param1 Model:   a CommittorNN.
        :param2 X:       coordinates of the sampled configurations.
        :param3 Weights: Boltzmann weights of those configurations.
        :param4 XA:      coordinates labelled as the reactant basin.
        :param5 XB:      coordinates labelled as the product basin.

        :return: Loss, Lv, Lb: total loss and its two terms.

        """

        G2, _ = MassWeightedGradientSquared(Model, X, self.Masses)
        w = Weights/Weights.sum()
        Lv = torch.sum(w*G2)

        qA = Model.Committor(XA)
        qB = Model.Committor(XB)
        Lb = torch.mean(qA*qA) + torch.mean((qB - 1.0)*(qB - 1.0))

        return Lv + self.Alpha*Lb, Lv, Lb

#%% 3. Binomial likelihood of shots
def BinomialShotLoss(Model, X, NA, NB):

    """

    Negative log likelihood of shooting outcomes, per trajectory.

    A shooting point x sends NA + NB trajectories, NB of which reach the
    second state; under the committor q(x) that is a binomial draw, the
    likelihood the AIMMD scheme trains on. The value is computed from the
    latent, log q = log sigmoid(s z), so no probability underflows.

    :param1 Model: a CommittorNN.
    :param2 X:     shooting points, shape (m, NumCoordinates).
    :param3 NA:    trajectories ending in the first state, shape (m,).
    :param4 NB:    trajectories ending in the second state, shape (m,).

    :return: NLL: the mean negative log likelihood per trajectory.

    """

    s = Model.Slope*Model(X)
    LogQ = torch.nn.functional.logsigmoid(s)
    Log1mQ = torch.nn.functional.logsigmoid(-s)

    return -torch.sum(NB*LogQ + NA*Log1mQ)/torch.clamp(torch.sum(NA + NB), min=1.0)
