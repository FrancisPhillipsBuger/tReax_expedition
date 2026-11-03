#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Back end agnostic calculator handling: attaching a calculator to a structure,
evaluating an energy robustly whichever back end answers it, and the free
EMT potential used for benchmarking the sampler without quantum chemistry.


Created on: Tue Aug 25 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
from ase.calculators.emt import EMT

#%% 1. Attachment and robust evaluation
def setCalculatorToStructure(Structure,Calculator):

    """

    Attach the ASE calculator to a structure.

    :param1 Structure:  ase.Atom instance to attach the calculator to.
    :param2 Calculator: ASE calculator.

    :return: Structure: updated ase.Atom instance.

    """

    Structure.calc = Calculator
    return Structure

def wrapGetPotentialEnergy(AtomInstance, NumIter=4):

    """

    Get the potential energy using ASE, retrying on a failed evaluation.

    A repeat helps where the failure is external to the geometry, such as the
    sporadic MPI trouble of a file driven back end. It cannot help an SCF that
    genuinely does not converge, which fails the same way every time.

    :param1 AtomInstance: ase.Atom instance with an attached calculator.
    :param2 NumIter:      number of iterations to resubmit the calculations.

    :return: En: potential energy in eV.

    """

    LastError = None
    for i in range(0,NumIter):
        try:
            En = AtomInstance.get_potential_energy()
            return En
        except Exception as Error:
            # keep the last one: a file driven back end fails for reasons a
            # bare message cannot convey, and the caller has no other way in
            LastError = Error

    raise RuntimeError("Failed to evaluate the potential energy after %i "
                       "attempt(s). The last error was: %s"
                       % (NumIter, LastError)) from LastError

#%% 2. Calculator setup
def getEMTCalculator():

    """

    Set up the Effective Medium Theory calculator.

    EMT covers Al, Cu, Ag, Au, Ni, Pd and Pt, which is enough for the surface
    diffusion benchmarks; it is not meaningful for the SN2 system.

    :return: Calculator: an ASE calculator.

    """

    return EMT()
