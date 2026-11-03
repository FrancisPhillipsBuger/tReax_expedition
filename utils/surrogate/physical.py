#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

A cheaper level of theory used as the surrogate of an expensive one.

The delayed acceptance the sampler already implements does not care where the
cheap potential comes from, only that it is cheap and correlated with the true
one. A semi-empirical method is both, and unlike a fitted surrogate it needs
no training data, is valid the moment the run starts, and does not degrade
when the walkers reach a region the run has not visited before.

The second stage tests the residual between the two potentials as a double
difference, so the large constant offset between two levels of theory cancels
exactly and the sampled distribution stays that of the true potential.

Created on: Thu Aug 27 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 1. Cheap level of theory as a surrogate
class CalculatorSurrogate:

    """

    Surrogate that evaluates a second, cheaper ASE calculator.

    """

    def __init__(self, Calculator):

        """

        :param1 Calculator: the cheap ASE calculator to screen with.

        """

        self.Calculator = Calculator
        self.NCalls = 0
        self.NFailed = 0

    def __call__(self, Structure):

        """

        Evaluate the cheap potential on a structure.

        A failed evaluation returns +inf, so the first stage rejects the trial
        at no true cost. The screen is then zero on the geometries the cheap
        method cannot describe, which keeps the two-stage move reversible; the
        sampler refuses to start from such a geometry, since it could not
        leave it.

        :param1 Structure: ase.Atom instance to evaluate.

        :return: the cheap potential energy, in eV, or +inf on failure.

        """

        # the true calculator must stay attached to the structure the sampler
        # owns, so the screening runs on a copy
        Probe = Structure.copy()
        Probe.calc = self.Calculator
        self.NCalls += 1

        try:
            return float(Probe.get_potential_energy())
        except Exception:
            self.NFailed += 1

            return float("inf")

def ForXTB(charge=0, mult=1, nProcs=1):

    """

    Build a GFN2-xTB surrogate for an expensive back end.

    :param1 charge: charge of the system (int).
    :param2 mult:   multiplicity of the system (int).
    :param3 nProcs: threads the tight binding back end may use (int).

    :return: a CalculatorSurrogate.

    """

    from treax_expedition.xtb import getXTBCalculator

    return CalculatorSurrogate(getXTBCalculator(charge=charge,
                                                mult=mult, nProcs=nProcs))
