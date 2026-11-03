#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

CLI entry point for the two-channel potential of committor_2.0, whose basins are
joined by an upper and a lower channel, in units of kT at 300 K. Runs the staged search;
with --scan, writes the PES files and scans tau and Ninter as in the reference
work. See analytic_2d.driver for the options.

Created on: Sun Sep 27 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analytic_2d.driver import RunScanCLI

#%% 1. Main
def main():

    """

    Run the staged search, or with --scan the (tau, Ninter) scan, on the two_paths potential.

    :return: None.

    """

    RunScanCLI("two_paths")

    return None


if __name__ == '__main__':
    main()
