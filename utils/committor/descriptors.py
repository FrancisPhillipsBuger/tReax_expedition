#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Differentiable descriptor layers feeding the committor network.

Created on: Tue Aug 25 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import itertools

import torch

#%% 1. Identity
class Identity(torch.nn.Module):

    """

    Pass the coordinates through unchanged.

    This is the descriptor set of the two-dimensional model potential, whose
    coordinates are already the collective variables of interest.

    """

    def __init__(self, NumCoordinates):

        """

        :param1 NumCoordinates: number of coordinates, which is also the number
                                of descriptors.

        """

        super().__init__()
        self.NumCoordinates = int(NumCoordinates)
        self.NumDescriptors = int(NumCoordinates)

    def forward(self, X):

        """

        :param1 X: coordinates, shape (batch, NumCoordinates).

        :return: X: the same tensor.

        """

        return X

#%% 2. Pairwise distances
class PairwiseDistances(torch.nn.Module):

    """

    All, or a chosen subset of, the interatomic distances.

    Distances are invariant under translation and rotation, so no centre of
    mass removal and no Kabsch alignment are needed, and a permutation of
    identical atoms only reorders the set instead of inflating an RMSD. This
    is the descriptor set used for the alanine model of the reference
    implementation.

    """

    def __init__(self, NumAtoms, Pairs=None):

        """

        :param1 NumAtoms: number of atoms in the structure.
        :param2 Pairs:    list of atom index pairs, or None for all of them.

        """

        super().__init__()
        self.NumAtoms = int(NumAtoms)
        if Pairs is None:
            Pairs = list(itertools.combinations(range(0, self.NumAtoms), 2))
        self.register_buffer("PairIndex", torch.tensor(Pairs, dtype=torch.long))
        self.NumCoordinates = 3*self.NumAtoms
        self.NumDescriptors = len(Pairs)

    def forward(self, X):

        """

        :param1 X: flat coordinates, shape (batch, 3*NumAtoms).

        :return: D: distances, shape (batch, NumDescriptors).

        """

        R = X.view(-1, self.NumAtoms, 3)
        i = self.PairIndex[:, 0]
        j = self.PairIndex[:, 1]
        Delta = R[:, i, :] - R[:, j, :]

        # the square root is not differentiable at zero, which never happens
        # for distinct atoms but would silently produce NaN if it did
        return torch.sqrt(torch.sum(Delta*Delta, dim=-1) + 1.0e-12)
