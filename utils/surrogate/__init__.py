#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Public API for the surrogate package: a cheaper level of theory screening the
trial moves of an expensive one in the delayed-acceptance kernel.

Created on: Tue Aug 25 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Public re-exports
from .physical import CalculatorSurrogate, ForXTB

__all__ = [
    'CalculatorSurrogate',
    'ForXTB',
]
