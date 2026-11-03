#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Neural parametrisation of the committor function.


Created on: Tue Aug 25 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import numpy as np
import torch

from .descriptors import Identity, PairwiseDistances

#%% 1. Default architecture
# the reference models are 2-32-32-1 for the two dimensional potentials,
# 45-32-32-1 for alanine and 210-64-32-1 for chignolin, all with Tanh
DefaultHidden = (32, 32)
DefaultSlope = 3.0

#%% 2. Feed forward stack
def BuildFeedForward(NumInputs, Hidden=DefaultHidden, Activation=torch.nn.Tanh):

    """

    Build the descriptor to latent variable stack.

    :param1 NumInputs:  number of descriptors.
    :param2 Hidden:     widths of the hidden layers.
    :param3 Activation: activation module inserted between the linear layers.

    :return: Net: a torch.nn.Sequential mapping descriptors onto one output.

    """

    Layers = []
    Previous = int(NumInputs)
    for h in Hidden:
        Layers.append(torch.nn.Linear(Previous, int(h)))
        Layers.append(Activation())
        Previous = int(h)
    Layers.append(torch.nn.Linear(Previous, 1))

    return torch.nn.Sequential(*Layers)

#%% 3. Committor model
class CommittorNN(torch.nn.Module):

    """

    Descriptors, a feed forward stack and the sigmoid that produces q.

    """

    def __init__(self, Descriptors, Hidden=DefaultHidden, Slope=DefaultSlope):

        """

        :param1 Descriptors: descriptor layer mapping coordinates onto inputs.
        :param2 Hidden:      widths of the hidden layers.
        :param3 Slope:       slope of the sigmoid, three in the reference.

        """

        super().__init__()
        self.Descriptors = Descriptors
        self.Net = BuildFeedForward(Descriptors.NumDescriptors, Hidden=Hidden)
        self.Slope = float(Slope)

    def LatentFromDescriptors(self, D):

        """

        :param1 D: descriptors, shape (batch, NumDescriptors).

        :return: z: latent variable, shape (batch,).

        """

        return self.Net(D).squeeze(-1)

    def forward(self, X):

        """

        :param1 X: flat coordinates, shape (batch, NumCoordinates).

        :return: z: latent variable, shape (batch,).

        """

        return self.LatentFromDescriptors(self.Descriptors(X))

    def Committor(self, X):

        """

        :param1 X: flat coordinates, shape (batch, NumCoordinates).

        :return: q: committor, shape (batch,).

        """

        return torch.sigmoid(self.Slope*self.forward(X))

    def CommittorFromLatent(self, Z):

        """

        :param1 Z: latent variable.

        :return: q: committor.

        """

        return torch.sigmoid(self.Slope*Z)

#%% 4. Evaluation
def CommittorOf(Model, X):

    """

    :param1 Model: a CommittorNN.
    :param2 X:     coordinates, shape (n, NumCoordinates).

    :return: q: committor of every configuration, as a numpy array.

    """

    with torch.no_grad():
        # torch refuses the negative strides of a reversed or sliced view
        t = torch.as_tensor(np.ascontiguousarray(np.atleast_2d(X)), dtype=torch.get_default_dtype())

        return Model.Committor(t).numpy()

#%% 5. Checkpoints
def DescriptorSpecification(Descriptors):

    """

    Describe a descriptor layer well enough to rebuild it.

    :param1 Descriptors: the descriptor layer of a CommittorNN.

    :return: Spec: a dictionary of plain values.

    """

    if isinstance(Descriptors, PairwiseDistances):
        return {"Kind": "PairwiseDistances",
                "NumAtoms": int(Descriptors.NumAtoms),
                "Pairs": [[int(i), int(j)] for i, j in Descriptors.PairIndex.tolist()]}

    if isinstance(Descriptors, Identity):
        return {"Kind": "Identity",
                "NumCoordinates": int(Descriptors.NumCoordinates)}

    raise ValueError("no checkpoint format for descriptors of type %s"
                     % (type(Descriptors).__name__,))

def BuildDescriptors(Spec):

    """

    Rebuild a descriptor layer from its specification.

    :param1 Spec: the dictionary written by DescriptorSpecification.

    :return: Descriptors: the descriptor layer.

    """

    if Spec["Kind"] == "PairwiseDistances":
        return PairwiseDistances(Spec["NumAtoms"], Pairs=Spec["Pairs"])

    if Spec["Kind"] == "Identity":
        return Identity(Spec["NumCoordinates"])

    raise ValueError("unknown descriptor kind %r" % (Spec["Kind"],))

def SaveCommittor(Model, FileName):

    """

    Write a trained model together with the architecture that rebuilds it.

    The Kolmogorov bias differentiates the committor, so rebuilding it needs
    the descriptor layer and the sigmoid slope as well as the weights. A bare
    state dictionary leaves the layer widths to be guessed, which is why the
    specification travels with the weights.

    :param1 Model:    a trained CommittorNN.
    :param2 FileName: destination of the checkpoint.

    :return: None.

    """

    Hidden = [int(Layer.out_features) for Layer in Model.Net
              if isinstance(Layer, torch.nn.Linear)][:-1]
    torch.save({"Descriptors": DescriptorSpecification(Model.Descriptors),
                "Hidden": Hidden,
                "Slope": float(Model.Slope),
                "State": Model.state_dict()}, FileName)

    return None

def LoadCommittor(FileName):

    """

    Rebuild a trained model from a checkpoint written by SaveCommittor.

    The weights are cast to the dtype of the session doing the loading. A
    committor is trained in double precision, whereas a sampler that only
    evaluates the bias normally runs in the default single precision, and
    without the cast the first matrix multiply of the bias would fail.

    :param1 FileName: the checkpoint to read.

    :return: Model: the CommittorNN, in evaluation mode and in the current
             default dtype.

    """

    try:
        Checkpoint = torch.load(FileName, map_location="cpu", weights_only=False)
    except TypeError:
        # weights_only predates torch 2.0 and is not accepted there
        Checkpoint = torch.load(FileName, map_location="cpu")

    if "Descriptors" not in Checkpoint:
        raise ValueError("%s holds no architecture; it predates SaveCommittor "
                         "and cannot be rebuilt without knowing the widths"
                         % (FileName,))

    Model = CommittorNN(BuildDescriptors(Checkpoint["Descriptors"]),
                        Hidden=tuple(Checkpoint["Hidden"]),
                        Slope=Checkpoint["Slope"])
    Model.load_state_dict(Checkpoint["State"])
    Model.eval()

    return Model.to(torch.get_default_dtype())
