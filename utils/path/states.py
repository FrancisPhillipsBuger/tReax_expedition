#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

States as the search sees them: the basin a configuration drains into under a
quench, told apart by a fingerprint. A molecule is fingerprinted by its bond
graph, a conformer by the box of its backbone dihedrals, a 2D particle by the
region of its minimum.

Created on: Mon Sep 28 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import numpy as np

#%% 1. Fingerprints
class BondGraph:

    """

    Set of bonded atom pairs, a bond being shorter than the sum of the covalent
    radii plus a fixed tolerance. An additive tolerance, as in the common
    cheminformatics rule, separates H-H (0.74-0.80 A) as well as a stretched
    ring C-O from an open one; a multiple of the radii cannot do both.

    """

    Name = "bonds"

    def __init__(self, Numbers, Tolerance=0.4):

        """

        :param1 Numbers:   atomic numbers.
        :param2 Tolerance: added to the sum of covalent radii, in angstroms.

        """

        from ase.data import covalent_radii
        self.Numbers = np.asarray(Numbers, dtype=int)
        Radii = covalent_radii[self.Numbers]
        self.Cutoff = Radii[:, None] + Radii[None, :] + Tolerance
        self.Upper = np.triu(np.ones((len(Radii), len(Radii)), dtype=bool), k=1)

    def Bonds(self, X):

        """

        :param1 X: flat coordinates.

        :return: Pairs: sorted tuple of the bonded (i, j), i < j.

        """

        XYZ = np.asarray(X, dtype=float).reshape(-1, 3)
        D = np.linalg.norm(XYZ[:, None, :] - XYZ[None, :, :], axis=-1)
        i, j = np.nonzero((D < self.Cutoff) & self.Upper)

        return tuple(zip(i.tolist(), j.tolist()))

    def Key(self, X):

        """

        :return: Key: the bond graph, hashable.

        """

        return self.Bonds(X)

    def Distance(self, K1, K2):

        """

        :return: d: zero for the same graph, infinity otherwise.

        """

        return 0.0 if K1 == K2 else np.inf

    def Describe(self, Key):

        """

        :return: Text: number of bonds.

        """

        return "%i bonds" % (len(Key),)

def Dihedral(XYZ, Quad):

    """

    :param1 XYZ:  coordinates, shape (natoms, 3).
    :param2 Quad: four atom indices.

    :return: phi: the dihedral angle in degrees, IUPAC sign convention.

    """

    b1 = XYZ[Quad[1]] - XYZ[Quad[0]]
    b2 = XYZ[Quad[2]] - XYZ[Quad[1]]
    b3 = XYZ[Quad[3]] - XYZ[Quad[2]]
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2/np.linalg.norm(b2))

    return float(-np.degrees(np.arctan2(np.dot(m1, n2), np.dot(n1, n2))))

class DihedralBox:

    """

    Backbone dihedrals of a conformer; two configurations are the same state
    when every dihedral agrees within the tolerance of the book.

    """

    Name = "dihedrals"

    def __init__(self, Quads):

        """

        :param1 Quads: list of four-atom index tuples, zero based.

        """

        self.Quads = [tuple(int(i) for i in q) for q in Quads]

    def Key(self, X):

        """

        :return: Key: tuple of the dihedrals, in degrees.

        """

        XYZ = np.asarray(X, dtype=float).reshape(-1, 3)

        return tuple(round(Dihedral(XYZ, q), 1) for q in self.Quads)

    def Distance(self, K1, K2):

        """

        :return: d: the largest periodic difference of the dihedrals, degrees.

        """

        d = np.abs((np.asarray(K1) - np.asarray(K2) + 180.0) % 360.0 - 180.0)

        return float(d.max())

    def Describe(self, Key):

        """

        :return: Text: the dihedrals.

        """

        return "(" + ", ".join("%.1f" % a for a in Key) + ") deg"

class Region:

    """

    Position of a 2D particle; the same state within the tolerance of the book.

    """

    Name = "region"

    def Key(self, X):

        """

        :return: Key: the rounded position.

        """

        return tuple(round(float(v), 4) for v in np.asarray(X, dtype=float))

    def Distance(self, K1, K2):

        """

        :return: d: Euclidean distance, in angstroms.

        """

        return float(np.linalg.norm(np.asarray(K1) - np.asarray(K2)))

    def Describe(self, Key):

        """

        :return: Text: the position.

        """

        return "(%+.3f, %+.3f)" % tuple(Key)

#%% 2. Book of known states
class StateBook:

    """

    The states met so far, each with the minimum a quench found for it.

    """

    def __init__(self, Fingerprint, Tolerance=0.0, NewStates=True):

        """

        :param1 Fingerprint: BondGraph, DihedralBox or Region.
        :param2 Tolerance:   largest fingerprint distance of one state.
        :param3 NewStates:   register states met beyond the named ones; when
                             False every configuration belongs to the nearest
                             named state, as when the states of a
                             conformational change are defined beforehand.

        """

        self.Fingerprint = Fingerprint
        self.Tolerance = float(Tolerance)
        self.NewStates = bool(NewStates)
        self.States = []

    def Names(self):

        """

        :return: Names: the names of the known states, in order of discovery.

        """

        return [s["Name"] for s in self.States]

    def Get(self, Name):

        """

        :return: State: the record of a named state.

        """

        for s in self.States:
            if s["Name"] == Name:
                return s
        raise KeyError(Name)

    def Identify(self, X):

        """

        Name of the known state a configuration belongs to, without registering.

        :param1 X: coordinates.

        :return: Name: the state, or None when no known state matches.

        """

        Key = self.Fingerprint.Key(X)
        Best = None
        BestD = np.inf
        for s in self.States:
            d = self.Fingerprint.Distance(Key, s["Key"])
            if d <= self.Tolerance and d < BestD:
                Best = s["Name"]
                BestD = d

        return Best

    def Nearest(self, X):

        """

        :return: Name: the known state closest in fingerprint, or None when
                 the fingerprint has no distance between different keys.

        """

        Key = self.Fingerprint.Key(X)
        D = [self.Fingerprint.Distance(Key, s["Key"]) for s in self.States]
        if len(D) == 0 or not np.isfinite(min(D)):
            return None

        return self.States[int(np.argmin(D))]["Name"]

    def Register(self, X, Energy, Name=None, Converged=True):

        """

        Name of the state of a quenched configuration, adding it when new.

        A new state keeps the given name, or is called I1, I2, ... A known
        state keeps the lowest minimum found for it. A quench that was still
        descending when its budget ran out is never taken for a new state:
        where the fingerprint has a distance it is assigned the nearest known
        state, the one it was draining towards.

        :param1 X:         quenched coordinates.
        :param2 Energy:    their energy, in eV.
        :param3 Name:      name to give a new state, or None.
        :param4 Converged: whether the quench ended on a plateau.

        :return: Name: the name of the state.

        """

        if (not Converged or not self.NewStates) and Name is None and self.Identify(X) is None:
            Nearest = self.Nearest(X)
            if Nearest is not None:
                return Nearest

        Known = self.Identify(X)
        if Known is not None:
            s = self.Get(Known)
            if Energy < s["Energy"]:
                s["X"] = np.asarray(X, dtype=float).copy()
                s["Energy"] = float(Energy)
            return Known

        if Name is None:
            Name = "I%i" % (sum(1 for s in self.States if s["Name"].startswith("I")) + 1)
        self.States.append({"Name": Name, "Key": self.Fingerprint.Key(X),
                            "X": np.asarray(X, dtype=float).copy(), "Energy": float(Energy)})

        return Name
