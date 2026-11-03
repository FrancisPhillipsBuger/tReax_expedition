#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

ORCA interface for the SWS sampler: builds the ASE ORCA calculator, optionally
with a CASSCF or TDDFT block, for the runs whose numbers have to match a
published level of theory rather than a semi-empirical one.

ORCA is driven through files and a separate process, so a single point costs
orders of magnitude more than the tight binding back end does. Reach for it
when the accuracy is the point; sample and debug on GFN2-xTB.

Created on: Mon Aug 24 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import os
from pathlib import Path

DEFAULT_METHOD = "hf-3c"

# ORCA keywords that would compute a gradient, a Hessian or run an optimiser;
# the sampler works on single point energies only
GRADIENT_KEYWORDS = ("engrad", "numgrad", "opt", "copt", "zopt", "tightopt",
                     "looseopt", "verytightopt", "optts", "freq", "numfreq",
                     "anfreq", "irc", "neb", "neb-ts", "neb-ci", "scants")

#%% 1. Calculator setup
def getORCACalculator(LevelOfTheory="hf-3c", charge=0, mult=1, nProcs=1, PathToOrca=None, numElectrons=None, numOrbital=None, actState=0, Directory=None):

    """

    Set up the ORCA calculator using ASE.

    :param1 LevelOfTheory: level of theory for the single point energy (str).
    :param2 charge:        charge of the system (int).
    :param3 mult:          multiplicity of the system (int).
    :param4 nProcs:        number of processes for parallel execution (int).
    :param5 PathToOrca:    path to ORCA executable (str).
    :param6 numElectrons:  number of electrons for CASSCF (int). Default: None (no CASSCF).
    :param7 numOrbital:    number of active orbitals for CASSCF (int). Default: None (no CASSCF).
    :param8 actState:      active state (from 1 for CASSCF and from 0 for TDDFT).
    :param9 Directory:     working directory of this calculator (str). Two
                           walkers must not share one: ORCA restarts its SCF
                           from the orbitals left in the directory, and a
                           guess taken from the other walker's geometry costs
                           several times the iterations of the walker's own.

    :return: orcaCalculator: ASE ORCA calculator.

    """

    Asked = set(LevelOfTheory.lower().replace("!", " ").split())
    Forbidden = sorted(Asked.intersection(GRADIENT_KEYWORDS))
    if len(Forbidden) > 0:
        raise ValueError("ORCA keyword(s) %s ask for gradients or an optimiser; "
                         "the sampler uses single point energies only" % (", ".join(Forbidden),))

    # imported here so that a run on another back end needs no ORCA support
    # in ASE, and so that importing this package never depends on it
    from ase.calculators.orca import ORCA, OrcaProfile

    OrcaBlocks = '%pal nprocs '+str(nProcs)+' end'

    # this is the CASSCF block
    if (not numElectrons is None) and (not numOrbital is None):
        OrcaBlocks += '\n %casscf \n'
        OrcaBlocks += 'nel  %i\n' % numElectrons
        OrcaBlocks += 'norb        %i\n' % numOrbital
        OrcaBlocks += 'mult        %i\n' % mult
        if actState > 0:
            OrcaBlocks += "nroots %i\n" %  (actState+1)
            OrcaBlocks += "iroot  %i\n" %  (actState+1)
        OrcaBlocks += "end\n"
    else:
        if actState>0:
            OrcaBlocks += '\n %tddft\n'
            OrcaBlocks += 'nroots %i\n' % actState
            OrcaBlocks += 'iroot  %i\n' % actState
            OrcaBlocks += 'end\n\n'

    Settings = dict(orcasimpleinput=LevelOfTheory, orcablocks=OrcaBlocks,
                    charge=charge, mult=mult)

    if Directory is not None:
        os.makedirs(Directory, exist_ok=True)
        Settings["directory"] = Path(Directory)

    if PathToOrca is not None:
        return ORCA(profile=OrcaProfile(command=PathToOrca), **Settings)

    # without a path ASE falls back to its own configuration file, which is
    # not written by default; say what to do rather than let it raise a
    # message about configuration that names neither this run nor the fix
    try:
        return ORCA(**Settings)
    except Exception as Error:
        raise RuntimeError(
            "ORCA was not found. Either pass the executable with "
            "--orcapath=/path/to/orca, or add an [orca] section to the ASE "
            "configuration file. The original error was: %s" % (Error,))
