#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

AMBER ff99SB energy of alanine dipeptide in vacuum, from the parameters exported out of the
committor_2.0 run input (tools/export_ff99sb.py, removed after c1689db). Flexible X-H
bonds, no cutoff, plain Coulomb. Energies only: the calculator offers no forces.

Created on: Mon Sep 28 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import os

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from .io_xyz import INPUT_ROOT

# Coulomb constant of GROMACS, kJ mol^-1 nm e^-2, and kJ/mol per eV
COULOMB = 138.935458
KJMOL_PER_EV = 96.48533212
DEFAULT_PARAMETERS = os.path.join(INPUT_ROOT, "alanine_dipeptide", "ff99sb.npz")

#%% 1. Force field
def Dihedrals(X, Index):

    """

    :param1 X:     coordinates, shape (natoms, 3).
    :param2 Index: four atom indices per row.

    :return: phi: dihedral angles in radians, GROMACS sign convention.

    """

    b1 = X[Index[:, 1]] - X[Index[:, 0]]
    b2 = X[Index[:, 2]] - X[Index[:, 1]]
    b3 = X[Index[:, 3]] - X[Index[:, 2]]
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2/np.linalg.norm(b2, axis=1)[:, None])

    return -np.arctan2(np.sum(m1*n2, axis=1), np.sum(n1*n2, axis=1))

class FF99SB:

    """

    The energy terms of ff99SB over flat arrays.

    """

    def __init__(self, FileName=DEFAULT_PARAMETERS):

        """

        :param1 FileName: reaction/input/alanine_dipeptide/ff99sb.npz.

        """

        P = np.load(FileName)
        self.P = {k: P[k] for k in P.files}
        self.NAtoms = len(self.P["Charges"])
        q = self.P["Charges"]
        self.Q14 = self.P["FudgeQQ"]*COULOMB*q[self.P["PairIndex"][:, 0]]*q[self.P["PairIndex"][:, 1]]
        NB = self.P["NBIndex"]
        self.QNB = COULOMB*q[NB[:, 0]]*q[NB[:, 1]]
        T = self.P["TypeIndex"]
        C = self.P["LJ"][T[NB[:, 0]], T[NB[:, 1]]]
        self.C6 = C[:, 0]
        self.C12 = C[:, 1]

    def Energy(self, Positions):

        """

        :param1 Positions: coordinates in angstroms, shape (natoms, 3).

        :return: E: the potential energy in kJ/mol.

        """

        P = self.P
        X = np.asarray(Positions, dtype=float).reshape(-1, 3)*0.1

        b = P["BondIndex"]
        r = np.linalg.norm(X[b[:, 0]] - X[b[:, 1]], axis=1)
        E = 0.5*np.sum(P["BondK"]*(r - P["BondR0"])**2)

        a = P["AngleIndex"]
        v1 = X[a[:, 0]] - X[a[:, 1]]
        v2 = X[a[:, 2]] - X[a[:, 1]]
        c = np.sum(v1*v2, axis=1)/(np.linalg.norm(v1, axis=1)*np.linalg.norm(v2, axis=1))
        E += 0.5*np.sum(P["AngleK"]*(np.arccos(np.clip(c, -1.0, 1.0)) - P["AngleTheta0"])**2)

        E += np.sum(P["DihK"]*(1.0 + np.cos(P["DihMult"]*Dihedrals(X, P["DihIndex"]) - P["DihPhase"])))
        E += np.sum(P["ImpK"]*(1.0 + np.cos(P["ImpMult"]*Dihedrals(X, P["ImpIndex"]) - P["ImpPhase"])))

        p = P["PairIndex"]
        r = np.linalg.norm(X[p[:, 0]] - X[p[:, 1]], axis=1)
        E += np.sum(P["PairC12"]/r**12 - P["PairC6"]/r**6 + self.Q14/r)

        n = P["NBIndex"]
        r = np.linalg.norm(X[n[:, 0]] - X[n[:, 1]], axis=1)
        E += np.sum(self.C12/r**12 - self.C6/r**6 + self.QNB/r)

        return float(E)

#%% 2. ASE calculator
class AmberCalculator(Calculator):

    """

    ASE calculator returning the ff99SB energy in eV, and nothing else.

    """

    implemented_properties = ["energy"]

    def __init__(self, FileName=DEFAULT_PARAMETERS, **Kwargs):

        """

        :param1 FileName: reaction/input/alanine_dipeptide/ff99sb.npz.

        """

        super().__init__(**Kwargs)
        self.FF = FF99SB(FileName)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):

        """

        Evaluate the energy of the attached atoms.

        """

        super().calculate(atoms, properties, system_changes)
        if len(self.atoms) != self.FF.NAtoms:
            raise ValueError("ff99SB parameters are for %i atoms, the structure has %i"
                             % (self.FF.NAtoms, len(self.atoms)))
        self.results["energy"] = self.FF.Energy(self.atoms.get_positions())/KJMOL_PER_EV

def getAmberCalculator(FileName=DEFAULT_PARAMETERS):

    """

    :param1 FileName: reaction/input/alanine_dipeptide/ff99sb.npz.

    :return: Calculator: the AmberCalculator.

    """

    return AmberCalculator(FileName)
