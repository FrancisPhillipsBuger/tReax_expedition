#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Progress coordinates the continuation chain is swept along. Each is a value of
the configuration only: the bond-change coordinate built from the bond graphs
of the two states, the latent of a classifier of the zero-temperature labels,
and the latent of a committor trained at the physical temperature.

Created on: Mon Sep 28 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import numpy as np

#%% 1. Bond change
class BondChange:

    """

    xi = sum of the breaking bond lengths - sum of the forming ones, from the
    bond graphs of the two states; the reaction coordinate of the MCTSS SN2
    analysis, r(C-Cl) - r(C-Br) up to sign, generalised.

    """

    Name = "xi_bond"

    def __init__(self, Fingerprint, XA, XB):

        """

        :param1 Fingerprint: BondGraph.
        :param2 XA:          minimum of the first state.
        :param3 XB:          minimum of the second state.

        """

        A = set(Fingerprint.Bonds(XA))
        B = set(Fingerprint.Bonds(XB))
        self.Breaking = sorted(A - B)
        self.Forming = sorted(B - A)
        self.Available = len(self.Breaking) + len(self.Forming) > 0

    def __call__(self, X):

        """

        :param1 X: flat coordinates.

        :return: xi: the coordinate, in angstroms.

        """

        XYZ = np.asarray(X, dtype=float).reshape(-1, 3)

        def Length(Pairs):
            return sum(float(np.linalg.norm(XYZ[i] - XYZ[j])) for i, j in Pairs)

        return Length(self.Breaking) - Length(self.Forming)

    def Describe(self):

        """

        :return: Text: the bonds that break and form.

        """

        return ("breaking %s, forming %s"
                % (" ".join("%i-%i" % p for p in self.Breaking) or "-",
                   " ".join("%i-%i" % p for p in self.Forming) or "-"))

#%% 2. Dihedral change
class DihedralChange:

    """

    xi = projection of the change of the fingerprint dihedrals, from the first
    state towards the second along the shorter arc of each, normalised so the
    first state sits at 0 and the second at 1: the conformational counterpart
    of the bond change, built from the two states' fingerprints alone.

    """

    Name = "xi_dihedral"

    def __init__(self, Fingerprint, XA, XB):

        """

        :param1 Fingerprint: DihedralBox.
        :param2 XA:          minimum of the first state.
        :param3 XB:          minimum of the second state.

        """

        self.Fingerprint = Fingerprint
        self.ThetaA = np.array(Fingerprint.Key(XA), dtype=float)
        self.Delta = Wrap(np.array(Fingerprint.Key(XB), dtype=float) - self.ThetaA)
        self.Norm = float(np.dot(self.Delta, self.Delta))
        self.Available = self.Norm > 0.0

    def __call__(self, X):

        """

        :param1 X: flat coordinates.

        :return: xi: the coordinate, dimensionless.

        """

        from .states import Dihedral

        XYZ = np.asarray(X, dtype=float).reshape(-1, 3)
        Theta = np.array([Dihedral(XYZ, q) for q in self.Fingerprint.Quads])

        return float(np.dot(Wrap(Theta - self.ThetaA), self.Delta)/self.Norm)

    def Describe(self):

        """

        :return: Text: the change of every dihedral.

        """

        return "dihedral change (" + ", ".join("%+.1f" % d for d in self.Delta) + ") deg"

def Wrap(Angle):

    """

    :return: Angle: wrapped into [-180, 180) degrees.

    """

    return (np.asarray(Angle) + 180.0) % 360.0 - 180.0

#%% 3. Latent of a learned model
class Latent:

    """

    The latent variable z of a CommittorNN as a coordinate.

    """

    def __init__(self, Model, Name="z"):

        """

        :param1 Model: a CommittorNN.
        :param2 Name:  what the coordinate is called in the output.

        """

        self.Model = Model
        self.Name = Name
        self.Available = True

    def __call__(self, X):

        """

        :param1 X: flat coordinates.

        :return: z: the latent value.

        """

        import torch
        # a worker process may run in another default precision than the fit
        Type = next(self.Model.parameters()).dtype
        with torch.no_grad():
            t = torch.as_tensor(np.ascontiguousarray(np.reshape(X, (1, -1))), dtype=Type)
            return float(self.Model(t)[0])

    def Describe(self):

        """

        :return: Text: the kind of model.

        """

        return "latent of %s" % (self.Name,)

def TrainLabelClassifier(X, y, Descriptors, Epochs=400, WeightDecay=1.0e-3,
                         LearningRate=3.0e-3, Seed=1):

    """

    Classifier of the zero-temperature labels, whose latent serves as xi_label.

    The binary cross entropy of q = sigmoid(s z) against the labels is
    minimised with a weight decay, which keeps z finite on separable data so
    that its level sets stay usable as slices.

    :param1 X:            labelled configurations, shape (n, NumCoordinates).
    :param2 y:            labels, 0 for the first state and 1 for the second.
    :param3 Descriptors:  descriptor layer of the model.
    :param4 Epochs:       full-batch epochs.
    :param5 WeightDecay:  L2 penalty of Adam.
    :param6 LearningRate: Adam learning rate.
    :param7 Seed:         seed of the initialisation.

    :return: Model: the trained CommittorNN.

    """

    import torch
    from utils.committor.model import CommittorNN

    torch.manual_seed(Seed)
    Model = CommittorNN(Descriptors)
    dt = torch.get_default_dtype()
    Xt = torch.as_tensor(np.ascontiguousarray(X), dtype=dt)
    yt = torch.as_tensor(np.asarray(y, dtype=float), dtype=dt)
    # the two classes count equally, however many frames each has
    w = torch.where(yt > 0.5, 0.5/max(1.0, float(yt.sum())), 0.5/max(1.0, float((1.0 - yt).sum())))
    Optimiser = torch.optim.Adam(Model.parameters(), lr=LearningRate, weight_decay=WeightDecay)
    for _ in range(0, Epochs):
        Optimiser.zero_grad()
        Loss = torch.sum(w*torch.nn.functional.binary_cross_entropy_with_logits(
            Model.Slope*Model(Xt), yt, reduction="none"))
        Loss.backward()
        Optimiser.step()
    Model.eval()

    return Model
