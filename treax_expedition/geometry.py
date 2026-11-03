#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Geometry helpers for the SWS sampler: mass-weighted RMSD, Kabsch alignment of
one structure onto another, centre-of-mass removal, the random SWS
displacement of a structure, the decision of whether the overall frame has
to be removed at all, and the progress of a structure along the axis joining
the two endpoints.

Created on: Mon Aug 24 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import random

import numpy as np

from constants.units import SxCoeff
from utils.mc.system import JointRotation, RandomRotation, RotationWidth

#%% 1. Mass-weighted RMSD
def calcRMSDBetweenStructures(Structure1, Structure2, UseMassWeightedRMSD=True):

    """

    Mass-weighted RMSD between two structures, in amu*angstroem**2.

    Despite the name, which the whole package keeps, this is the mass-weighted
    sum of squared displacements, sum_i m_i |r1_i - r2_i|^2, with neither a
    mean over the atoms nor a square root. That is the quantity the kinetic
    term m dx^2/(hbar tau) of the acceptance probability needs, and it grows
    with the number of atoms, which is why the meeting threshold of the
    two-structure run is scaled by it.

    :param1 Structure1:          ase.Atom instance, first structure.
    :param2 Structure2:          ase.Atom instance, second structure.
    :param3 UseMassWeightedRMSD: flag to use mass-weighted RMSD (bool).

    :return: RMSD: the RMSD value for two structures.

    """

    XYZ1 = Structure1.get_positions()
    XYZ2 = Structure2.get_positions()
    NumAt = len(XYZ1)
    if UseMassWeightedRMSD:
        m = Structure1.get_masses()
    else:
        m = np.ones(NumAt)

    DR = XYZ1 - XYZ2
    RMSD = float(np.sum(m[:, None]*DR*DR))

    return RMSD

#%% 2. Kabsch alignment
def KabschCovariance(RefXYZ, XYZ, Weights):

    """

    Weighted covariance of two coordinate sets and its singular decomposition.

    Both the optimal rotation and the RMSD it achieves follow from this one
    3x3 decomposition, so neither needs a rotated copy of the structure. The
    coordinates are used as they are, like scipy's align_vectors, since the
    callers have already moved the centre of mass to the origin.

    :param1 RefXYZ:  reference coordinates, shape (natoms, 3).
    :param2 XYZ:     coordinates to be rotated onto the reference.
    :param3 Weights: per atom weights, shape (natoms,).

    :return: U, S, Vt, d: the decomposition of sum_i w_i a_i b_i^T and the sign
             that keeps the rotation proper.

    """

    B = (RefXYZ*Weights[:, None]).T @ XYZ
    U, S, Vt = np.linalg.svd(B)
    d = 1.0 if np.linalg.det(U @ Vt) >= 0.0 else -1.0

    return U, S, Vt, d

def orientStructureWrtAnother(StructureToOrient, ReferenceStructure, UseMassWeightedRMSD=True):

    """

    Orient one structure with respect to another using the Kabsch algorithm.

    :param1 StructureToOrient:   ase.Atom instance, whose geometry will be oriented.
    :param2 ReferenceStructure:  ase.Atom instance used as a reference.
    :param3 UseMassWeightedRMSD: flag to use mass-weighted RMSD (bool).

    :return: OrientedStructure: ase.Atom instance with updated structure.

    """

    RefXYZ = ReferenceStructure.get_positions()
    XYZ    = StructureToOrient.get_positions()
    if UseMassWeightedRMSD:
        m = ReferenceStructure.get_masses()
    else:
        m = np.ones(len(XYZ))

    U, _, Vt, d = KabschCovariance(RefXYZ, XYZ, m)
    RotMat = (U*np.array([1.0, 1.0, d])) @ Vt
    StructureToOrient.set_positions(XYZ @ RotMat.T)
    return StructureToOrient

def calcAlignedRMSDBetweenStructures(Structure1, Structure2, Align, UseMassWeightedRMSD=True):

    """

    Mass-weighted RMSD after the optimal rotation of one structure onto the other.

    Equal to orienting a copy of Structure1 onto Structure2 and calling
    calcRMSDBetweenStructures, but taken from the singular values directly,
    min_R sum w|a - R b|^2 = sum w|a|^2 + sum w|b|^2 - 2 (s1 + s2 + d s3),
    so no structure is copied or rotated. The value is symmetric in the two
    structures.

    :param1 Structure1:          ase.Atom instance, first structure.
    :param2 Structure2:          ase.Atom instance, the reference.
    :param3 Align:               whether frame removal applies; without it the
                                 plain RMSD is returned.
    :param4 UseMassWeightedRMSD: flag to use mass-weighted RMSD (bool).

    :return: RMSD: the aligned RMSD, in amu*angstroem**2.

    """

    if not Align:
        return calcRMSDBetweenStructures(Structure1, Structure2, UseMassWeightedRMSD=UseMassWeightedRMSD)

    RefXYZ = Structure2.get_positions()
    XYZ = Structure1.get_positions()
    m = Structure2.get_masses() if UseMassWeightedRMSD else np.ones(len(XYZ))

    _, S, _, d = KabschCovariance(RefXYZ, XYZ, m)
    RMSD = (np.sum(m[:, None]*RefXYZ*RefXYZ) + np.sum(m[:, None]*XYZ*XYZ)
            - 2.0*(S[0] + S[1] + d*S[2]))

    # the difference of two large sums can dip below zero by rounding
    return max(float(RMSD), 0.0)

#%% 3. Centre of mass
def centerMolecule(Structure):

    """

    Set the center of molecule to its center of mass.

    :param1 Structure: ase.Atom instance to be centered.

    :return: CenteredStructure: ase.Atom instance with updated structure.

    """

    CoM = Structure.get_center_of_mass()
    Structure.set_positions(Structure.get_positions() - CoM)
    return Structure

#%% 4. SWS displacement
def displaceSWSStructure(Structure, tau, SingleAtom=True, SingleCoordinate=False, Debug=False, StepScale=1.0):

    """

    Displace the structure with a randomly generated SWS displacement.

    The width is the free-particle one, sqrt(hbar tau/2m), times StepScale. Any
    width gives a symmetric proposal, so the sampled distribution does not
    depend on it; only how fast the walkers move does.

    :param1 Structure:        ase.Atom instance to be displaced.
    :param2 tau:              time parameter tau (float), in fs.
    :param3 SingleAtom:       flag (bool) to move only one randomly chosen atom.
    :param4 SingleCoordinate: flag (bool) to vary only one coordinate (x, y, or z).
    :param5 Debug:            also return the varied atom and coordinate indexes.
    :param6 StepScale:        multiplier of the free-particle width (float).

    :return: DisplacedStructure: ase.Atom instance with updated structure.

    """

    DisplacedStructure = Structure.copy()
    DisplacedStructure.calc = Structure.calc
    M = Structure.get_masses()
    S = StepScale * SxCoeff * np.sqrt(tau / M )
    # index of an atom
    IdxAtom = list()
    if SingleAtom:
        IdxAtom.append(random.randrange(0,len(M)))
    else:
        IdxAtom = [i for i in range(0,len(M))]

    # index of an x/y/z coordinate
    IdxCoord = list()
    if SingleCoordinate:
        IdxCoord.append(random.randrange(0,3))
    else:
        IdxCoord = [i for i in range(0,3)]

    XYZ = DisplacedStructure.get_positions()
    for n in IdxAtom:
        for i in IdxCoord:
            disp = random.gauss(mu=0.0, sigma=S[n])
            XYZ[n][i] += disp

    DisplacedStructure.set_positions(XYZ)
    if Debug:
        return DisplacedStructure, IdxAtom, IdxCoord
    else:
        return DisplacedStructure

#%% 5. Frame handling
def NeedsFrameRemoval(Structure):

    """

    Decide whether the overall translation and rotation have to be removed.

    A free molecule carries six spurious degrees of freedom that have to be
    projected out before two structures can be compared by RMSD. A periodic
    cell or a set of fixed atoms already pins the frame, and removing it again
    is not merely redundant but wrong: ASE refuses to move constrained atoms,
    so the shift or rotation is absorbed entirely by the free atoms and tears
    the structure apart.

    :param1 Structure: ase.Atom instance to inspect.

    :return: Needed: True for a free molecule, False otherwise.

    """

    if any(Structure.pbc):
        return False
    if len(Structure.constraints) > 0:
        return False

    return True

def MaybeCenter(Structure, Align):

    """

    Remove the centre of mass only when the frame is not already pinned.

    :param1 Structure: ase.Atom instance to centre.
    :param2 Align:     whether frame removal applies.

    :return: Structure: the centred, or untouched, structure.

    """

    if not Align:
        return Structure

    return centerMolecule(Structure)

def MaybeOrient(StructureToOrient, ReferenceStructure, Align, UseMassWeightedRMSD=True):

    """

    Orient one structure onto another only when the frame is not pinned.

    :param1 StructureToOrient:   ase.Atom instance to orient.
    :param2 ReferenceStructure:  ase.Atom instance used as a reference.
    :param3 Align:               whether frame removal applies.
    :param4 UseMassWeightedRMSD: flag to use mass-weighted RMSD (bool).

    :return: Structure: the oriented, or untouched, structure.

    """

    if not Align:
        return StructureToOrient

    return orientStructureWrtAnother(StructureToOrient=StructureToOrient,
                                     ReferenceStructure=ReferenceStructure,
                                     UseMassWeightedRMSD=UseMassWeightedRMSD)

#%% 6. Progress along the endpoint axis
def alignedCoordinates(XYZ, RefXYZ, Masses, Align):

    """

    Coordinates centred and rotated onto a centred reference.

    :param1 XYZ:    coordinates to align, shape (natoms, 3).
    :param2 RefXYZ: reference coordinates, already centred.
    :param3 Masses: per atom masses.
    :param4 Align:  whether frame removal applies at all.

    :return: X: the aligned coordinates.

    """

    if not Align:
        return XYZ

    X = XYZ - (Masses[:, None]*XYZ).sum(axis=0)/Masses.sum()
    U, _, Vt, d = KabschCovariance(RefXYZ, X, Masses)

    return X @ ((U*np.array([1.0, 1.0, d])) @ Vt).T

def ProjectOntoEndpointAxis(Positions, Initial, Final, Masses=None):

    """

    Progress coordinate of a structure along the initial to final direction.

    The mass-weighted displacement from the initial structure is projected on
    the mass-weighted vector joining the two endpoints, and normalised so that
    the initial structure sits at zero and the final one at one. This gives a
    reaction coordinate for any system without having to pick internal
    coordinates by hand.

    :param1 Positions: coordinates to project, shape (natoms, 3).
    :param2 Initial:   coordinates of the initial structure.
    :param3 Final:     coordinates of the final structure.
    :param4 Masses:    per atom masses, or None for an unweighted projection.

    :return: s: progress coordinate, 0 at the initial and 1 at the final point.

    """

    x = np.asarray(Positions, dtype=float)
    a = np.asarray(Initial, dtype=float)
    b = np.asarray(Final, dtype=float)
    if Masses is None:
        w = np.ones(len(a))
    else:
        w = np.asarray(Masses, dtype=float)
    w = np.sqrt(w)[:, None]

    d = (b - a)*w
    n2 = float(np.sum(d*d))
    if n2 <= 0.0:
        return 0.0

    return float(np.sum((x - a)*w*d)/n2)

class EndpointAxis:

    """

    Progress of a structure along the mass-weighted endpoint axis.

    The initial endpoint sits at zero and the final one at one, whatever the
    frame a structure was recorded in.

    """

    def __init__(self, Initial, Final):

        """

        :param1 Initial: ase.Atom instance, the first endpoint.
        :param2 Final:   ase.Atom instance, the second endpoint.

        """

        self.Align = NeedsFrameRemoval(Initial)
        self.Masses = np.asarray(Initial.get_masses(), dtype=float)
        Ref = Initial.get_positions()
        if self.Align:
            Ref = Ref - (self.Masses[:, None]*Ref).sum(axis=0)/self.Masses.sum()
        self.Ref = Ref
        self.Fin = alignedCoordinates(Final.get_positions(), Ref, self.Masses, self.Align)

    def __call__(self, XYZ):

        """

        :param1 XYZ: coordinates, shape (natoms, 3).

        :return: s: progress coordinate.

        """

        return ProjectOntoEndpointAxis(alignedCoordinates(XYZ, self.Ref, self.Masses, self.Align),
                                       self.Ref, self.Fin, self.Masses)

#%% 7. Common frame of a coupled pair
def rotationWidth(Structure, tau12):

    """

    :param1 Structure: ase.Atom instance.
    :param2 tau12:     tau of the coupling, in fs.

    :return: Sigma: width of the rotation angle of the relative-orientation
             move of a pair, in radians (utils.mc.system.RotationWidth).

    """

    return RotationWidth(Structure.get_positions(), np.asarray(Structure.get_masses(), dtype=float), tau12)

def rotateStructureRandomly(Structure, Sigma):

    """

    :param1 Structure: ase.Atom instance, centred.
    :param2 Sigma:     width of the rotation angle, in radians.

    :return: Rotated: a copy rigidly rotated about its centre of mass by a
             symmetric random rotation, with the calculator kept.

    """

    Rotated = Structure.copy()
    Rotated.calc = Structure.calc
    CoM = Structure.get_center_of_mass()
    Rotated.set_positions((Structure.get_positions() - CoM) @ RandomRotation(Sigma).T + CoM)

    return Rotated

def orientPairOnto(Structures, Reference):

    """

    Rotate both structures of a pair by the one rotation taking their midpoint
    onto a reference, which fixes the frame without changing the pair target.

    :param1 Structures: list of two centred ase.Atom instances.
    :param2 Reference:  centred reference coordinates, shape (natoms, 3).

    :return: Structures: the two rotated copies, calculators kept.

    """

    m = np.asarray(Structures[0].get_masses(), dtype=float)
    A, B = JointRotation(Structures[0].get_positions(), Structures[1].get_positions(),
                         np.asarray(Reference, dtype=float), m)
    Out = []
    for S, XYZ in zip(Structures, (A, B)):
        T = S.copy()
        T.calc = S.calc
        T.set_positions(XYZ)
        Out.append(T)

    return Out[0], Out[1]
