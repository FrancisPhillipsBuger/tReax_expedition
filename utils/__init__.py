#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Helpers shared by the 2D and the molecular samplers: the Monte-Carlo kernel,
systems, quench and tau ladder (mc), the staged transition state search
built on them (path), the surrogates that screen proposals (surrogate) and
the committor with its Kolmogorov bias, MBAR data and shots (committor).
The samplers themselves sit at the top level: analytic_2d and treax_expedition.

Created on: Mon Aug 24 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""
