#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

What a stage of the search needs from a system: energies in eV, the symmetric
free-particle proposal, the mass-weighted distance between two configurations
after frame removal, and nothing else. There is deliberately no gradient.

A frame may be removed from one configuration on its own only where the target
is invariant under a rotation of that configuration alone: a single walker.
The target of a coupled pair is invariant only under a rotation of both, so
its walkers share one frame (Displace, FrameDistance) and their relative
orientation is sampled by an explicit, energy-free rotation (Rotated).

Created on: Mon Sep 28 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import random

import numpy as np

from constants.units import KECoeff, SxCoeff, wvn2eV

#%% 1. Frame removal on flat coordinates
def Centred(XYZ, Masses):

    """

    :param1 XYZ:    coordinates, shape (natoms, 3).
    :param2 Masses: per atom masses.

    :return: XYZ: the coordinates with the centre of mass at the origin.

    """

    return XYZ - (Masses[:, None]*XYZ).sum(axis=0)/Masses.sum()

def RotatedOnto(XYZ, RefXYZ, Masses):

    """

    Mass-weighted Kabsch rotation of centred coordinates onto a centred reference.

    :param1 XYZ:    coordinates to rotate, shape (natoms, 3).
    :param2 RefXYZ: reference coordinates.
    :param3 Masses: per atom masses.

    :return: XYZ: the rotated coordinates.

    """

    B = (RefXYZ*Masses[:, None]).T @ XYZ
    U, _, Vt = np.linalg.svd(B)
    d = 1.0 if np.linalg.det(U @ Vt) >= 0.0 else -1.0

    return XYZ @ ((U*np.array([1.0, 1.0, d])) @ Vt).T

def RandomRotation(Sigma):

    """

    Rotation about a uniformly drawn axis by a Gaussian angle. The angle and its
    negative are equally likely, so the move and its inverse have the same
    density and the proposal is symmetric.

    :param1 Sigma: standard deviation of the angle, in radians.

    :return: R: 3x3 rotation matrix.

    """

    u = np.array([random.gauss(mu=0.0, sigma=1.0) for _ in range(0, 3)])
    u /= np.linalg.norm(u)
    t = random.gauss(mu=0.0, sigma=Sigma)
    K = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])

    return np.eye(3) + np.sin(t)*K + (1.0 - np.cos(t))*(K @ K)

def RotationWidth(XYZ, Masses, Tau12):

    """

    Angle over which a rigid rotation of one walker changes the coupling
    kappa d^2/tau12 of a pair by about one, from the mean principal moment of
    inertia; at most 0.5 rad. The moment does not change under the rotation,
    so the width is the same before and after the move.

    :param1 XYZ:    coordinates, shape (natoms, 3).
    :param2 Masses: per atom masses.
    :param3 Tau12:  tau of the coupling, in fs.

    :return: Sigma: the width of the rotation angle, in radians.

    """

    C = Centred(XYZ, Masses)
    Mean = 2.0*float(np.sum(Masses[:, None]*C*C))/3.0
    if Mean <= 0.0:
        return 0.5

    return float(min(0.5, np.sqrt(Tau12/(KECoeff*Mean))))

def JointRotation(A, B, RefXYZ, Masses):

    """

    Rotate both walkers of a pair by the one rotation that takes their
    midpoint onto a reference, to write them in a fixed frame; the walkers
    held by the chain are never rotated this way.

    :param1 A:      centred coordinates of the first walker, shape (natoms, 3).
    :param2 B:      centred coordinates of the second walker.
    :param3 RefXYZ: centred reference coordinates.
    :param4 Masses: per atom masses.

    :return: A, B: both rotated.

    """

    H = (RefXYZ*Masses[:, None]).T @ (0.5*(A + B))
    U, _, Vt = np.linalg.svd(H)
    d = 1.0 if np.linalg.det(U @ Vt) >= 0.0 else -1.0
    Rot = ((U*np.array([1.0, 1.0, d])) @ Vt).T

    return A @ Rot, B @ Rot

#%% 2. Analytic two-dimensional system
class AnalyticSystem:

    """

    A particle on an analytic 2D potential.

    """

    def __init__(self, Potential, Mass):

        """

        :param1 Potential: a Potential2D; a call on one point returns cm-1.
        :param2 Mass:      mass of the particle in a.m.u.s.

        """

        self.Potential = Potential
        self.Mass = float(Mass)
        self.Masses = np.array([self.Mass, self.Mass])
        self.Rotational = False
        self.NCalls = 0
        self.NFailed = 0

    def Energy(self, X):

        """

        :param1 X: 2D coordinates, in angstroms.

        :return: E: potential energy, in eV.

        """

        self.NCalls += 1

        return float(self.Potential(np.asarray(X, dtype=float)))*wvn2eV

    def Propose(self, X, StepTau, SingleAtom=True):

        """

        Free-particle Gaussian displacement of width sqrt(hbar StepTau/2m).

        :param1 X:          2D coordinates.
        :param2 StepTau:    time parameter setting the width, in fs.
        :param3 SingleAtom: unused, a 2D particle always moves as a whole.

        :return: Trial: the displaced coordinates.

        """

        Sigma = SxCoeff*np.sqrt(StepTau/self.Mass)
        Step = np.array([random.gauss(mu=0.0, sigma=Sigma) for _ in range(0, 2)])

        return np.asarray(X, dtype=float) + Step

    def Displace(self, X, StepTau):

        """

        :return: Trial: the free-particle displacement of Propose; a 2D particle
                 has no frame, so the two are the same.

        """

        return self.Propose(X, StepTau)

    def Aligned(self, X, Ref):

        """

        :return: X: unchanged, a 2D particle has no frame to remove.

        """

        return np.asarray(X, dtype=float)

    def Centred(self, X):

        """

        :return: X: unchanged.

        """

        return np.asarray(X, dtype=float)

    def FrameDistance(self, X1, X2):

        """

        :return: d: the mass-weighted distance, the same as Distance.

        """

        return self.Distance(X1, X2)

    def Distance(self, X1, X2):

        """

        :return: d: mass-weighted distance, in sqrt(amu) angstroms.

        """

        D = np.asarray(X1, dtype=float) - np.asarray(X2, dtype=float)

        return float(np.sqrt(self.Mass*np.dot(D, D)))

#%% 3. Molecular system
class AtomsSystem:

    """

    A molecule on an ASE calculator, handled in flat Cartesian coordinates.

    """

    def __init__(self, Template, Calculator, Screen=None, SingleAtom=True):

        """

        :param1 Template:   ase.Atoms fixing the elements, masses and charge.
        :param2 Calculator: ASE calculator answering energies only.
        :param3 Screen:     cheaper ASE calculator for the delayed-acceptance
                            screen, or None.
        :param4 SingleAtom: move one randomly chosen atom per step (bool).

        """

        self.Atoms = Template.copy()
        self.Atoms.calc = Calculator
        self.Probe = None
        if Screen is not None:
            self.Probe = Template.copy()
            self.Probe.calc = Screen
        self.Numbers = self.Atoms.get_atomic_numbers()
        self.AtomMasses = np.asarray(self.Atoms.get_masses(), dtype=float)
        self.Masses = np.repeat(self.AtomMasses, 3)
        self.Align = not any(Template.pbc) and len(Template.constraints) == 0
        self.Rotational = self.Align
        self.SingleAtom = bool(SingleAtom)
        self.NCalls = 0
        self.NFailed = 0

    def Energy(self, X):

        """

        :param1 X: flat coordinates, in angstroms.

        :return: E: potential energy in eV, or None when the back end fails.

        """

        self.NCalls += 1
        self.Atoms.set_positions(np.asarray(X, dtype=float).reshape(-1, 3))
        try:
            return float(self.Atoms.get_potential_energy())
        except Exception as Error:
            self.NFailed += 1
            print("[System] energy evaluation failed, trial rejected: %s" % (Error,))

            return None

    def ScreenEnergy(self, X):

        """

        :param1 X: flat coordinates.

        :return: E: energy of the screening calculator in eV, +inf on failure.

        """

        self.Probe.set_positions(np.asarray(X, dtype=float).reshape(-1, 3))
        try:
            return float(self.Probe.get_potential_energy())
        except Exception:
            return float("inf")

    def Propose(self, X, StepTau, SingleAtom=None):

        """

        Free-particle Gaussian displacement of one atom, or of all of them, of
        width sqrt(hbar StepTau/2m), followed by frame removal onto X.

        :param1 X:          flat coordinates.
        :param2 StepTau:    time parameter setting the width, in fs.
        :param3 SingleAtom: override of the system setting, or None.

        :return: Trial: the displaced flat coordinates.

        """

        return self.Aligned(self.Displaced(X, StepTau, SingleAtom), X)

    def Displaced(self, X, StepTau, SingleAtom=None):

        """

        :return: XYZ: X with the free-particle Gaussian displacement of one
                 atom, or of all of them, and no frame removed.

        """

        SingleAtom = self.SingleAtom if SingleAtom is None else SingleAtom
        XYZ = np.asarray(X, dtype=float).reshape(-1, 3).copy()
        Sigma = SxCoeff*np.sqrt(StepTau/self.AtomMasses)
        Moved = [random.randrange(0, len(XYZ))] if SingleAtom else range(0, len(XYZ))
        for n in Moved:
            for i in range(0, 3):
                XYZ[n][i] += random.gauss(mu=0.0, sigma=Sigma[n])

        return XYZ.reshape(-1)

    def Displace(self, X, StepTau, SingleAtom=None):

        """

        The free-particle displacement of Propose with the centre of mass
        removed but no rotation, for a walker that shares its frame with
        another one.

        :return: Trial: the displaced, centred flat coordinates.

        """

        return self.Centred(self.Displaced(X, StepTau, SingleAtom))

    def Aligned(self, X, Ref):

        """

        :param1 X:   flat coordinates to align.
        :param2 Ref: flat reference coordinates.

        :return: X: centred and rotated onto the centred reference.

        """

        XYZ = np.asarray(X, dtype=float).reshape(-1, 3)
        if not self.Align:
            return XYZ.reshape(-1).copy()
        RefXYZ = Centred(np.asarray(Ref, dtype=float).reshape(-1, 3), self.AtomMasses)

        return RotatedOnto(Centred(XYZ, self.AtomMasses), RefXYZ, self.AtomMasses).reshape(-1)

    def Centred(self, X):

        """

        :return: X: flat coordinates with the centre of mass at the origin,
                 unchanged when the frame is pinned.

        """

        XYZ = np.asarray(X, dtype=float).reshape(-1, 3)
        if not self.Align:
            return XYZ.reshape(-1).copy()

        return Centred(XYZ, self.AtomMasses).reshape(-1)

    def FrameDistance(self, X1, X2):

        """

        Mass-weighted distance in the common frame: centres removed, no rotation.
        Minimising over the rotation instead, as Distance does, would make a
        target built on it invariant under a rotation of either configuration
        alone and give each an extra weight of its rotational orbit volume.

        :return: d: the distance, in sqrt(amu) angstroms.

        """

        D = self.Centred(X1) - self.Centred(X2)

        return float(np.sqrt(np.sum(self.Masses*D*D)))

    def RotationWidth(self, X, Tau12):

        """

        :return: Sigma: RotationWidth of X, in radians.

        """

        return RotationWidth(np.asarray(X, dtype=float).reshape(-1, 3), self.AtomMasses, Tau12)

    def Rotated(self, X, Sigma):

        """

        :param1 X:     flat coordinates, centred.
        :param2 Sigma: width of the rotation angle, in radians.

        :return: Trial: X rigidly rotated about its centre of mass by a
                 symmetric random rotation; its energy is that of X.

        """

        XYZ = self.Centred(X).reshape(-1, 3)

        return (XYZ @ RandomRotation(Sigma).T).reshape(-1)

    def Distance(self, X1, X2):

        """

        :return: d: mass-weighted distance after alignment, in sqrt(amu) angstroms.

        """

        D = self.Aligned(X1, X2) - self.Aligned(X2, X2)

        return float(np.sqrt(np.sum(self.Masses*D*D)))
