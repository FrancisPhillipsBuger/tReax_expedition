#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

The tau ladder of the exploration stage. MCTSS is run hot first and made
colder, tau doubling per rung, for as long as at least half of the seeds of a
rung still meet; the coldest rung that does sets tau_explore. Nothing outside
the MCTSS runs themselves enters the choice. Every tau here is the bead tau of
each walker; the pair's midpoint represents 2 tau (see PairEnsemble).

Created on: Mon Sep 28 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
from dataclasses import dataclass, field

#%% 1. One exploration run
@dataclass
class ExploreRun:

    """

    One two-walker MCTSS run.

    :param1 Tau:       bead tau of each walker, in fs.
    :param2 Seed:      seed of the run.
    :param3 Converged: whether the walkers met, or their paths overlapped.
    :param4 Reason:    'meet', 'overlap' or 'budget'.
    :param5 Path:      configurations from the first endpoint to the second:
                       walker 1 forwards, then walker 2 backwards.
    :param6 Energies:  their energies, in eV.
    :param7 NTrue:     true energy evaluations spent.
    :param8 Folder:    folder of the dumps, or None.

    """

    Tau: float
    Seed: int
    Converged: bool
    Reason: str
    Path: list = field(default_factory=list)
    Energies: list = field(default_factory=list)
    NTrue: int = 0
    Folder: str = None

#%% 2. Ladder
def ExploreLadder(RunOne, Tau0=1.0, Factor=2.0, NSeeds=4, MaxRungs=8, SuccessFraction=0.5,
                  MaxHotter=3, BaseSeed=1, Map=None, Log=print):

    """

    Run MCTSS on a ladder of taus and keep the coldest rung that still meets.

    :param1 RunOne:          callable (tau, seed) -> ExploreRun.
    :param2 Tau0:            first tau of the ladder, in fs.
    :param3 Factor:          ratio of two consecutive taus.
    :param4 NSeeds:          runs per rung.
    :param5 MaxRungs:        largest number of rungs towards the cold side.
    :param6 SuccessFraction: share of the runs of a rung that have to meet.
    :param7 MaxHotter:       how many times to halve Tau0 if the first rung fails.
    :param8 BaseSeed:        seeds of the rungs are counted up from it.
    :param9 Map:             callable (function, list of argument tuples) ->
                             list of results, to run the seeds in parallel;
                             None runs them one after the other.
    :param10 Log:            callable printing one line.

    :return: TauExplore, Rungs: the coldest successful tau (None if none met)
             and the list of (tau, runs) of every rung tried.

    """

    Map = (lambda f, Args: [f(*a) for a in Args]) if Map is None else Map
    Rungs = []
    Counter = [BaseSeed]

    def Rung(tau):
        Args = [(tau, Counter[0] + k) for k in range(0, NSeeds)]
        Counter[0] += NSeeds
        Runs = Map(RunOne, Args)
        Met = sum(1 for r in Runs if r.Converged)
        Rungs.append((tau, Runs))
        Log("[Explore] tau %8.3f fs: %i/%i met, %s energies"
            % (tau, Met, NSeeds, "/".join("%i" % r.NTrue for r in Runs)))
        return Met >= SuccessFraction*NSeeds

    tau = float(Tau0)
    Success = Rung(tau)
    Hotter = 0
    while not Success and Hotter < MaxHotter:
        tau /= Factor
        Hotter += 1
        Success = Rung(tau)
    if not Success:
        Log("[Explore] no rung met; the endpoints could not be joined")
        return None, Rungs

    TauExplore = tau
    for _ in range(1, MaxRungs):
        tau *= Factor
        if not Rung(tau):
            break
        TauExplore = tau

    return TauExplore, Rungs
