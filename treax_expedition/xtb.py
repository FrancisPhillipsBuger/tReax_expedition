#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

GFN2-xTB interface for the SWS sampler: builds the tblite calculator for one
of the semi-empirical tight binding parametrisations.

The semi-empirical tight binding parametrisations are evaluated in process,
so a single point costs milliseconds and no scratch files are written. That
suits a sampler whose cost is a very large number of very cheap energies far
better than a file driven quantum chemistry code does.

Created on: Thu Aug 27 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import os

#%% 1. Parametrisations
# GFN2-xTB is the only tight binding parametrisation the project uses, keyed
# on what a user may reasonably type
METHODS = {
    "gfn2": "GFN2-xTB",
    "gfn2-xtb": "GFN2-xTB",
}

DEFAULT_METHOD = "GFN2-xTB"

def normaliseMethod(Method):

    """

    Turn what the user typed into a parametrisation tblite knows.

    :param1 Method: name of the parametrisation (str), or None for the default.

    :return: the canonical name (str).

    """

    if Method is None or str(Method).strip() == "":
        return DEFAULT_METHOD

    Key = str(Method).strip().lower()
    if Key not in METHODS:
        raise ValueError(
            "Unknown method '%s'. The tight binding back end is restricted "
            "to %s."
            % (Method, ", ".join(sorted(set(METHODS.values())))))

    return METHODS[Key]

#%% 2. Calculator setup
def getXTBCalculator(Method=DEFAULT_METHOD, charge=0, mult=1, nProcs=1,
                     Accuracy=1.0, ElectronicTemperature=300.0,
                     MaxIterations=250):

    """

    Set up the tight binding calculator using tblite through ASE.

    :param1 Method:      parametrisation (str). Default: GFN2-xTB.
    :param2 charge:      charge of the system (int).
    :param3 mult:        multiplicity of the system (int).
    :param4 nProcs:      number of OpenMP threads to allow (int).
    :param5 Accuracy:    numerical thresholds, smaller is tighter (float).
    :param6 ElectronicTemperature: Fermi smearing temperature in K (float).
    :param7 MaxIterations: cap on the number of SCF iterations (int).

    :return: Calculator: an ASE calculator returning energies in eV.

    """

    Name = normaliseMethod(Method)

    # the back end is threaded with OpenMP rather than run as separate
    # processes, so the process count of the command line becomes a thread
    # count. It has to be in place before the library reads it.
    if nProcs is not None and int(nProcs) > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(int(nProcs)))

    # imported here so the env var above is set first, and so that a run on the
    # free back end needs no tblite at all
    from tblite.ase import TBLite

    Calculator = TBLite(method=Name,
                        charge=int(charge),
                        multiplicity=int(mult),
                        accuracy=float(Accuracy),
                        electronic_temperature=float(ElectronicTemperature),
                        max_iterations=int(MaxIterations),
                        verbosity=0)

    return Calculator
