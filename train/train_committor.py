#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

CLI entry point of the committor: the staged search followed by its
committor stage (S4) on any system, a 2D potential or a reaction under
reaction/input. The committor is learned at the physical temperature on the
states and channels the search certified, as in committor_2.0, and validated
with shots as in AIMMD. All options after --system go to the search entry of
that system.

Created on: Mon Sep 28 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import os
import sys

# the repository root, one level above this folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

POTENTIALS = ("2d", "muller", "two_paths")

#%% 1. Main
def main():

    """

    Dispatch to the search entry of the system with S4 switched on.

    :return: None.

    """

    if "--system" not in sys.argv[1:] or sys.argv.index("--system") + 1 >= len(sys.argv):
        raise SystemExit("usage: train_committor.py --system {2d,muller,two_paths,<reaction>} [search options]")
    i = sys.argv.index("--system")
    System = sys.argv[i + 1]
    Rest = sys.argv[1:i] + sys.argv[i + 2:]
    if System in POTENTIALS:
        from analytic_2d.pipeline import RunSearchCLI
        sys.argv = [sys.argv[0], "--committor"] + Rest
        return RunSearchCLI(System)

    import run_treax_expedition
    sys.argv = [sys.argv[0], "--stages", "search+committor", "-f", System] + Rest

    return run_treax_expedition.main()

if __name__ == '__main__':
    main()
