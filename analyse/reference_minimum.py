#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Grading-only reference minima: reactant.xyz and product.xyz re-optimised to
minima on the back end a search ran on, with Sella and a Hessian check, so the
reaction energy of a search can be compared with true stationary points. Not
part of the search, which never uses gradients.

Created on: Thu Oct 29 2026

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

# the repository root, one level above this folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyse.reference_saddle import ForceCalculator, Frequencies
from analyse.ts_metrics import SymmetryRMSD
from treax_expedition.io_xyz import INPUT_ROOT, OUTPUT_ROOT, listReactions

MINIMUM_ROOT = os.path.join(OUTPUT_ROOT, "reference_minimum")

#%% 1. CLI argument parser
def parse_args():

    """

    Build the argparse parser for the reference minima.

    :return: Parsed argparse.Namespace.

    """

    p = argparse.ArgumentParser(description='Re-optimise reactant.xyz and product.xyz to minima on a back end, for grading only, and check them with a Hessian. Writes reaction/output/reference_minimum/<reaction>_<calculator>/.')
    p.add_argument('reactions', nargs='*',
                   help='Reaction names under reaction/input. Default: every reaction with a ts_guess.xyz.')
    p.add_argument('--calculator', type=str, default=None, choices=['xtb', 'orca', 'amber', 'emt'],
                   help="Back end; it must be the one the searches ran on. Default: the reaction's meta.json.")
    p.add_argument('--orcapath', type=str, default=None,
                   help='ORCA executable. Default: the ASE configuration.')
    p.add_argument('--fmax', type=float, default=1.0e-3,
                   help='Largest force component at convergence, in eV/A. Default: 1e-3.')
    p.add_argument('--steps', type=int, default=2000,
                   help='Cap on the Sella steps. Default: 2000.')
    p.add_argument('--fd_step', type=float, default=1.0e-3,
                   help='Displacement of the finite differences, in A, for back ends without forces and for the Hessian. Default: 1e-3.')
    p.add_argument('--imag_cut', type=float, default=20.0,
                   help='Imaginary frequencies smaller than this, in cm-1, count as numerical noise. Default: 20.')

    return p.parse_args()

#%% 2. One endpoint
def RelaxEnd(Start, Calc, Folder, End, Args):

    """

    :param1 Start:  ASE Atoms of the input endpoint.
    :param2 Calc:   calculator with forces.
    :param3 Folder: output folder of the reaction.
    :param4 End:    'reactant' or 'product'.
    :param5 Args:   parsed command line.

    :return: Atoms, Converged, Steps: the relaxed structure, whether Sella converged, its steps.

    """

    from sella import Sella

    Log = os.path.join(Folder, "%s_optimise.log" % (End,))
    if os.path.isfile(Log):
        os.remove(Log)
    Traj = os.path.join(Folder, "%s_sella.traj" % (End,))
    Atoms = Start.copy()
    Atoms.calc = Calc
    try:
        Opt = Sella(Atoms, order=0, internal=True, logfile=Log, trajectory=Traj)
        Converged = bool(Opt.run(fmax=Args.fmax, steps=Args.steps))
    except Exception as Err:
        # internal coordinates fail for some linear or near-linear structures
        print("[Minimum] %s: internal coordinates failed (%s), Cartesian Sella" % (End, Err))
        Atoms = Start.copy()
        Atoms.calc = Calc
        Opt = Sella(Atoms, order=0, internal=False, logfile=Log, trajectory=Traj)
        Converged = bool(Opt.run(fmax=Args.fmax, steps=Args.steps))

    return Atoms, Converged, int(Opt.nsteps)

#%% 3. One reaction
def ReferenceMinimum(Name, Args):

    """

    :param1 Name: reaction name under reaction/input.
    :param2 Args: parsed command line.

    :return: Summary: dict written to minimum.json.

    """

    from treax_expedition.pipeline import MoleculeSpec

    Spec = MoleculeSpec(Name, Calculator=Args.calculator, OrcaPath=Args.orcapath)
    Folder = os.path.join(MINIMUM_ROOT, "%s_%s" % (Name, Spec.Calculator))
    os.makedirs(Folder, exist_ok=True)
    Spec.WorkDir = Folder
    Calc, Analytic = ForceCalculator(Spec, Args.fd_step)
    FP = Spec.Fingerprint()
    Summary = {"Reaction": Name, "Calculator": Spec.Calculator, "AnalyticForces": Analytic,
               "Fingerprint": FP.Name}
    Start = time.time()
    for End in ("reactant", "product"):
        Input = asio.read(os.path.join(INPUT_ROOT, Name, "%s.xyz" % (End,)))
        Probe = Input.copy()
        Probe.calc = Calc
        EInput = Probe.get_potential_energy()
        Atoms, Converged, Steps = RelaxEnd(Input, Calc, Folder, End, Args)
        E = Atoms.get_potential_energy()
        Nu = Frequencies(Atoms, Args.fd_step)
        NImag = int(np.sum(Nu < -Args.imag_cut))
        KIn, KOut = FP.Key(Input.get_positions().reshape(-1)), FP.Key(Atoms.get_positions().reshape(-1))
        Summary[End.capitalize()] = {
            "Converged": Converged, "Steps": Steps, "FMax": float(np.abs(Atoms.get_forces()).max()),
            "Energy_eV": E, "EInput_eV": EInput, "NImaginary": NImag,
            "Imaginary_cm1": [float(v) for v in Nu[Nu < -Args.imag_cut]],
            "LowestReal_cm1": [float(v) for v in Nu[Nu >= -Args.imag_cut][:3]],
            "RMSDToInput_A": SymmetryRMSD(Input.get_positions(), Atoms.get_positions(),
                                          Input.get_chemical_symbols())[0],
            "SameStateAsInput": bool(FP.Distance(KIn, KOut) <= Spec.Tolerance),
            "State": FP.Describe(KOut)}
        Atoms.calc = None
        asio.write(os.path.join(Folder, "%s_minimum.xyz" % (End,)), Atoms, format="xyz",
                   comment="reference %s minimum of %s on %s, E = %.8f eV, %i imaginary, grading only"
                   % (End, Name, Spec.Calculator, E, NImag))
    Summary["DeltaE_eV"] = Summary["Product"]["Energy_eV"] - Summary["Reactant"]["Energy_eV"]
    Summary["Seconds"] = time.time() - Start
    with open(os.path.join(Folder, "minimum.json"), "w") as f:
        json.dump(Summary, f, indent=1)
    Checks = ["%s %s" % (End, "NOT converged" if not S["Converged"] else "%i imaginary" % S["NImaginary"])
              for End, S in (("reactant", Summary["Reactant"]), ("product", Summary["Product"]))
              if not S["Converged"] or S["NImaginary"] > 0]
    Checks += ["%s left its input state" % (End,)
               for End, S in (("reactant", Summary["Reactant"]), ("product", Summary["Product"]))
               if not S["SameStateAsInput"]]
    print("[Minimum] %-18s %-5s E(P) - E(R) %+.4f eV; relaxation R %.4f P %.4f eV; %i + %i steps, %.0f s%s"
          % (Name, Spec.Calculator, Summary["DeltaE_eV"],
             Summary["Reactant"]["Energy_eV"] - Summary["Reactant"]["EInput_eV"],
             Summary["Product"]["Energy_eV"] - Summary["Product"]["EInput_eV"],
             Summary["Reactant"]["Steps"], Summary["Product"]["Steps"], Summary["Seconds"],
             "  <-- check: " + "; ".join(Checks) if Checks else ""))

    return Summary

#%% 4. Main
def main():

    """

    Locate the reference minima of every reaction given.

    :return: None.

    """

    Args = parse_args()
    Names = Args.reactions or [n for n in listReactions()
                               if os.path.isfile(os.path.join(INPUT_ROOT, n, "ts_guess.xyz"))]
    for Name in Names:
        ReferenceMinimum(Name, Args)

    return None

if __name__ == "__main__":
    main()
