#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Grading-only reference saddles: ts_guess re-optimised to a first-order saddle
on the back end a search ran on, with Sella and a Hessian check, so located
TSs can be compared with a true stationary point. Not part of the search,
which never uses gradients; ts_compare.py picks the saddles up when present.

Created on: Thu Oct 01 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import argparse
import json
import os
import sys
import time

import ase.io as asio
import numpy as np
from ase import units
from ase.calculators.calculator import Calculator, all_changes

# the repository root, one level above this folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyse.ts_metrics import Dihedral, SymmetryRMSD
from treax_expedition.io_xyz import INPUT_ROOT, OUTPUT_ROOT, listReactions, readMeta

SADDLE_ROOT = os.path.join(OUTPUT_ROOT, "reference_saddle")

#%% 1. CLI argument parser
def parse_args():

    """

    Build the argparse parser for the reference saddles.

    :return: Parsed argparse.Namespace.

    """

    p = argparse.ArgumentParser(description='Re-optimise ts_guess.xyz to a first-order saddle on a back end, for grading only, and check it with a Hessian. Writes reaction/output/reference_saddle/<reaction>_<calculator>/.')
    p.add_argument('reactions', nargs='*',
                   help='Reaction names under reaction/input. Default: every reaction with a ts_guess.xyz.')
    p.add_argument('--calculator', type=str, default=None, choices=['xtb', 'orca', 'amber', 'emt'],
                   help="Back end; it must be the one the searches ran on. Default: the reaction's meta.json.")
    p.add_argument('--orcapath', type=str, default=None,
                   help='ORCA executable. Default: the ASE configuration.')
    p.add_argument('--fmax', type=float, default=1.0e-3,
                   help='Largest force component at convergence, in eV/A. Default: 1e-3.')
    p.add_argument('--steps', type=int, default=1000,
                   help='Cap on the Sella steps. Default: 1000.')
    p.add_argument('--fd_step', type=float, default=1.0e-3,
                   help='Displacement of the finite differences, in A, for back ends without forces and for the Hessian. Default: 1e-3.')
    p.add_argument('--imag_cut', type=float, default=20.0,
                   help='Imaginary frequencies smaller than this, in cm-1, count as numerical noise. Default: 20.')

    return p.parse_args()

#%% 2. Forces of an energy-only back end
class FiniteDifferenceForces(Calculator):

    """

    Energies from a wrapped calculator, forces from central differences of
    them. Grading only: the search itself never sees these forces.

    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, Inner, Step=1.0e-3, **Kwargs):

        """

        :param1 Inner: ASE calculator answering energies.
        :param2 Step:  displacement in angstroms.

        """

        super().__init__(**Kwargs)
        self.Inner = Inner
        self.Step = float(Step)

    def _Energy(self, Positions):

        """

        :return: E: the energy of the attached atoms at Positions, in eV.

        """

        A = self.atoms.copy()
        A.set_positions(Positions)
        A.calc = self.Inner

        return A.get_potential_energy()

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):

        """

        Evaluate the energy and, when asked, the forces.

        """

        super().calculate(atoms, properties, system_changes)
        X = self.atoms.get_positions()
        self.results["energy"] = self._Energy(X)
        if "forces" in properties:
            F = np.zeros_like(X)
            for i in range(len(X)):
                for k in range(3):
                    D = np.zeros_like(X)
                    D[i, k] = self.Step
                    F[i, k] = -(self._Energy(X + D) - self._Energy(X - D))/(2.0*self.Step)
            self.results["forces"] = F

def ForceCalculator(Spec, Step):

    """

    :param1 Spec: MoleculeSpec of the reaction.
    :param2 Step: finite-difference displacement in angstroms.

    :return: Calculator, Analytic: a calculator with forces, and whether they are analytic.

    """

    if Spec.Calculator == "xtb":
        from treax_expedition.xtb import getXTBCalculator
        return getXTBCalculator(charge=Spec.Charge, mult=Spec.Mult, nProcs=Spec.NProcs), True

    return FiniteDifferenceForces(Spec.MakeCalculator("reference_saddle"), Step=Step), False

#%% 3. Hessian check
def Frequencies(Atoms, Step):

    """

    Harmonic frequencies from central differences of the forces, with the
    translations and rotations projected out.

    :param1 Atoms: structure with a force calculator attached.
    :param2 Step:  displacement in angstroms.

    :return: Nu: frequencies in cm-1, imaginary ones negative, ascending.

    """

    X0 = Atoms.get_positions().copy()
    N = len(X0)
    H = np.zeros((3*N, 3*N))
    for j in range(3*N):
        D = np.zeros(3*N)
        D[j] = Step
        Atoms.set_positions(X0 + D.reshape(N, 3))
        Fp = Atoms.get_forces().reshape(-1)
        Atoms.set_positions(X0 - D.reshape(N, 3))
        Fm = Atoms.get_forces().reshape(-1)
        H[:, j] = -(Fp - Fm)/(2.0*Step)
    Atoms.set_positions(X0)
    H = 0.5*(H + H.T)
    m = np.repeat(Atoms.get_masses(), 3)
    Hm = H/np.sqrt(np.outer(m, m))
    # mass-weighted translations and rotations about the centre of mass
    Sm = np.sqrt(Atoms.get_masses())
    C = X0 - Atoms.get_center_of_mass()
    V = []
    for k in range(3):
        T = np.zeros((N, 3))
        T[:, k] = Sm
        V.append(T.reshape(-1))
        R = np.cross(np.eye(3)[k], C)*Sm[:, None]
        V.append(R.reshape(-1))
    Q, S, _ = np.linalg.svd(np.array(V).T, full_matrices=False)
    Q = Q[:, S > 1e-6*S.max()]
    P = np.eye(3*N) - Q @ Q.T
    w = np.linalg.eigvalsh(P @ Hm @ P)
    # eV/(A^2 amu) -> cm-1
    Conv = np.sqrt(units._e/(1e-20*units._amu))/(2.0*np.pi*units._c*100.0)
    Nu = np.sign(w)*np.sqrt(np.abs(w))*Conv

    return _DropRigid(Nu, Q.shape[1])

def _DropRigid(Nu, NRigid):

    """

    :return: Nu: the frequencies without the NRigid smallest in magnitude, ascending.

    """

    Keep = np.argsort(np.abs(Nu))[NRigid:]

    return np.sort(Nu[Keep])

#%% 4. One reaction
def PreRelax(Atoms, Quads, FMax, Steps, Log):

    """

    Relax everything but the fingerprint dihedrals, so an unrelaxed guess
    (the alanine ts_guess is a rotated product) starts Sella near the ridge.

    :return: None; Atoms is relaxed in place.

    """

    from ase.constraints import FixInternals
    from ase.optimize import BFGS

    Atoms.set_constraint(FixInternals(dihedrals_deg=[[Atoms.get_dihedral(*q), list(q)] for q in Quads]))
    BFGS(Atoms, logfile=Log).run(fmax=FMax, steps=Steps)
    Atoms.set_constraint()

    return None

def ReferenceSaddle(Name, Args):

    """

    :param1 Name: reaction name under reaction/input.
    :param2 Args: parsed command line.

    :return: Summary: dict written to saddle.json.

    """

    from sella import Sella
    from treax_expedition.pipeline import MoleculeSpec

    Spec = MoleculeSpec(Name, Calculator=Args.calculator, OrcaPath=Args.orcapath)
    Folder = os.path.join(SADDLE_ROOT, "%s_%s" % (Name, Spec.Calculator))
    os.makedirs(Folder, exist_ok=True)
    Spec.WorkDir = Folder
    Calc, Analytic = ForceCalculator(Spec, Args.fd_step)
    Guess = asio.read(os.path.join(INPUT_ROOT, Name, "ts_guess.xyz"))
    Reactant = asio.read(os.path.join(INPUT_ROOT, Name, "reactant.xyz"))
    Reactant.calc = Calc
    EReactant = Reactant.get_potential_energy()
    Atoms = Guess.copy()
    Atoms.calc = Calc
    EGuess = Atoms.get_potential_energy()
    FP = readMeta(Name).get("fingerprint", {})
    Quads = FP.get("quads", []) if FP.get("kind") == "dihedrals" else []
    Log = os.path.join(Folder, "optimise.log")
    if os.path.isfile(Log):
        os.remove(Log)
    Start = time.time()
    if Quads:
        PreRelax(Atoms, Quads, 10.0*Args.fmax, Args.steps, Log)
    try:
        Opt = Sella(Atoms, order=1, internal=True, logfile=Log, trajectory=os.path.join(Folder, "sella.traj"))
        Converged = bool(Opt.run(fmax=Args.fmax, steps=Args.steps))
    except Exception as Err:
        # internal coordinates fail for some linear or near-linear guesses
        print("[Saddle] %s: internal coordinates failed (%s), Cartesian Sella" % (Name, Err))
        Atoms = Guess.copy()
        Atoms.calc = Calc
        Opt = Sella(Atoms, order=1, internal=False, logfile=Log, trajectory=os.path.join(Folder, "sella.traj"))
        Converged = bool(Opt.run(fmax=Args.fmax, steps=Args.steps))
    E = Atoms.get_potential_energy()
    Nu = Frequencies(Atoms, Args.fd_step)
    NImag = int(np.sum(Nu < -Args.imag_cut))
    r = SymmetryRMSD(Guess.get_positions(), Atoms.get_positions(), Guess.get_chemical_symbols())[0]
    Summary = {"Reaction": Name, "Calculator": Spec.Calculator, "AnalyticForces": Analytic,
               "Converged": Converged, "Steps": int(Opt.nsteps), "FMax": float(np.abs(Atoms.get_forces()).max()),
               "Energy_eV": E, "EMinusReactant_eV": E - EReactant, "EGuessMinusReactant_eV": EGuess - EReactant,
               "NImaginary": NImag, "Imaginary_cm1": [float(v) for v in Nu[Nu < -Args.imag_cut]],
               "LowestReal_cm1": [float(v) for v in Nu[Nu >= -Args.imag_cut][:3]],
               "RMSDToGuess_A": r, "Seconds": time.time() - Start, "Start": "ts_guess.xyz"
               + (" relaxed at fixed dihedrals" if Quads else "")}
    if Quads:
        Summary["Dihedrals_deg"] = [Dihedral(Atoms.get_positions(), q) for q in Quads]
    Atoms.calc = None
    asio.write(os.path.join(Folder, "ts_saddle.xyz"), Atoms, format="xyz",
               comment="reference saddle of %s on %s, E = %.8f eV, %i imaginary, grading only"
               % (Name, Spec.Calculator, E, NImag))
    with open(os.path.join(Folder, "saddle.json"), "w") as f:
        json.dump(Summary, f, indent=1)
    Flag = "" if Converged and NImag == 1 else "  <-- check: %s, %i imaginary" % (
        "converged" if Converged else "NOT converged", NImag)
    print("[Saddle] %-18s %-5s E - E(reactant) %.4f eV (ts_guess %.4f), nu %s cm-1, RMSD to ts_guess %.3f A, %i steps, %.0f s%s"
          % (Name, Spec.Calculator, E - EReactant, EGuess - EReactant,
             ", ".join("%.0fi" % (-v,) for v in Summary["Imaginary_cm1"]) or "none",
             r, Summary["Steps"], Summary["Seconds"], Flag))

    return Summary

#%% 5. Main
def main():

    """

    Locate the reference saddle of every reaction given.

    :return: None.

    """

    Args = parse_args()
    Names = Args.reactions or [n for n in listReactions()
                               if os.path.isfile(os.path.join(INPUT_ROOT, n, "ts_guess.xyz"))]
    for Name in Names:
        ReferenceSaddle(Name, Args)

    return None

if __name__ == "__main__":
    main()
