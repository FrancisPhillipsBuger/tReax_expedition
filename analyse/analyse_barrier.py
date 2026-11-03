#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

CLI entry point for the barrier analysis of finished two-structure runs, done
as in the ESI of the MCTSS paper. Bins the energies of each run along the
reaction coordinate the sampler dumped, quotes the barriers in kJ/mol with the
ESI's spread errors and bootstrap errors, writes the binned profile and the
bin-averaged structures, the highest bin being the transition state, and,
given runs at three or more values of tau, extrapolates to zero kT. The
walkers' samples (the ESI's estimator) or the weighted midpoints can be binned.

Created on: Sat Sep 26 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import argparse
import os
import sys

import ase.io as asio
import numpy as np

# the repository root, one level above this folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants.units import eV2kJmol, kTEffeV
from analyse import (BinnedProfile, BinnedStructures, BootstrapBarriers,
                            ExtrapolateToZeroKT, LoadMidpoints, LoadRun,
                            LoadRunStructures, SpreadErrors)
from utils.mc.ensemble import EffectiveSampleFraction, NormalisedWeights

ESTIMATORS = ("mean", "min")

#%% 1. CLI argument parser
def parse_args():

    """

    Build the argparse parser for the barrier analysis.

    :return: Parsed argparse.Namespace.

    """

    p = argparse.ArgumentParser(description='Barriers and transition state structures of finished two-structure runs, as in the ESI, and their extrapolation to zero kT_eff.')
    p.add_argument('runs', nargs='+',
                   help='Output folders of two-structure runs, each holding ic1.trj, ic2.trj, sampled_energy1.log, sampled_energy2.log, sampled1.trj and sampled2.trj')
    p.add_argument('--bins', type=int, default=15,
                   help='Number of bins along the reaction coordinate. Default: 15.')
    p.add_argument('--grid', type=str, default='tile', choices=['tile', 'esi'],
                   help="Bin grid: 'tile' tiles [min, max] with the bins, so the end bins average over the tails; 'esi' uses the ESI's bins of width (max - min)/(bins - 1), whose last bin holds only the unmoved initial endpoint, and reproduces the published barriers. Default: tile.")
    p.add_argument('--nboot', type=int, default=200,
                   help='Number of bootstrap replicas. Default: 200.')
    p.add_argument('--seed', type=int, default=0,
                   help='Seed of the bootstrap generator. Default: 0.')
    p.add_argument('--estimator', type=str, default='walkers', choices=['walkers', 'midpoint'],
                   help="Samples to bin: 'walkers' pools the two walkers' samples, the ESI's estimator, which belongs to no single temperature; 'midpoint' bins the weighted midpoints of a run explored with --midpoint_stride, an exact sample of exp(-tau_mid V/hbar), tau_mid = 2 tau, over the stationary part of the chain. Default: walkers.")
    p.add_argument('--no_write', action='store_true',
                   help='Do not write binned_pes.dat, binned.trj and mc_ts.xyz into the run folders. Default: write them.')

    return p.parse_args()

#%% 2. Per-run output
def writeBinned(Folder, Centres, Mean, Std, Counts, Averages):

    """

    Write the binned profile, the bin-averaged structures and the TS structure.

    :param1 Folder:   output folder of the run.
    :param2 Centres:  bin centres along the reaction coordinate.
    :param3 Mean:     mean energy of each bin, in eV.
    :param4 Std:      standard deviation of the energies of each bin, in eV.
    :param5 Counts:   population of each bin.
    :param6 Averages: bin-averaged structures, or None when they are unavailable.

    :return: None.

    """

    np.savetxt(os.path.join(Folder, "binned_pes.dat"), np.column_stack([Centres, Mean, Std, Counts]),
               fmt=["%15.10f", "%18.10f", "%15.10f", "%8i"],
               header="s[$1] <E>,eV[$2] sd(E),eV[$3] N[$4]: 1 at the initial endpoint")
    if Averages is None:
        return None

    # one frame per row of binned_pes.dat, in the same order
    asio.write(os.path.join(Folder, "binned.trj"), Averages, format="xyz")
    asio.write(os.path.join(Folder, "mc_ts.xyz"), Averages[int(np.nanargmax(Mean))], format="xyz")

    return None

#%% 3. Main
def main():

    """

    Analyse every run given and extrapolate when the taus allow it.

    :return: None.

    """

    args = parse_args()

    Midpoint = args.estimator == "midpoint"
    Rows = []
    for Folder in args.runs:
        try:
            if Midpoint:
                IC, En, LogW, Ensemble = LoadMidpoints(Folder)
                Weights = NormalisedWeights(LogW)
                tau = Ensemble.TauMidpoint
            else:
                IC, En, tau = LoadRun(Folder)
                Weights = None
        except (RuntimeError, OSError, ValueError) as Error:
            print("[Barrier] skipped %s: %s" % (Folder, Error))
            continue
        Centres, Mean, Std, _, Counts = BinnedProfile(IC, En, N=args.bins, Grid=args.grid, Weights=Weights)
        if len(Mean) < 2:
            print("[Barrier] skipped %s: the reaction coordinate never moved" % (Folder,))
            continue
        Row = {"Name": os.path.basename(os.path.normpath(Folder)), "tau": tau, "N": len(En),
               "Spread": SpreadErrors(Mean, Std)}
        for Estimator in ESTIMATORS:
            Row[Estimator] = BootstrapBarriers(IC, En, N=args.bins, Estimator=Estimator,
                                               NBoot=args.nboot, Seed=args.seed, Grid=args.grid,
                                               Weights=Weights)
        Rows.append(Row)
        if Midpoint:
            print("[Barrier] %s: %i midpoints, ESS/N %.3f" % (Folder, len(En), EffectiveSampleFraction(LogW)))

        # bin-averaged structures are made from the walkers' frames only
        if args.no_write or Midpoint:
            continue
        Averages = None
        try:
            Averages = BinnedStructures(IC, LoadRunStructures(Folder), N=args.bins, Grid=args.grid)
        except (RuntimeError, OSError, ValueError) as Error:
            print("[Barrier] %s: no structures averaged, %s" % (Folder, Error))
        writeBinned(Folder, Centres, Mean, Std, Counts, Averages)
        iTS = int(np.nanargmax(Mean))
        print("[Barrier] %s: binned_pes.dat%s, TS bin at s = %.3f with %i structures"
              % (Folder, ", binned.trj, mc_ts.xyz" if Averages is not None else "",
                 Centres[iTS], Counts[iTS]))

    if len(Rows) == 0:
        print("[Barrier] no usable run")
        return None

    print("\nbarriers in kJ/mol; forward is from the initial endpoint")
    print("ESI: sqrt(sd_TS^2 + sd_end^2) of the bin energies, as in the ESI; boot: bootstrap standard deviation")
    if Midpoint:
        print("samples: weighted midpoints; tau and kT are those of the midpoint ensemble, tau_mid = 2 tau_bead")
    else:
        print("samples: the walkers', the ESI's estimator; tau and kT are those of one walker, the pair's midpoint being at tau_mid = 2 tau")
    print("%-24s %7s %9s %6s  %-9s %9s %8s %8s  %9s %8s %8s"
          % ("run", "tau,fs", "kT,kJ/mol", "N", "estimator",
             "forward", "ESI", "boot", "backward", "ESI", "boot"))
    for Row in Rows:
        for Estimator in ESTIMATORS:
            dEf, dEfErr, dEb, dEbErr = (v*eV2kJmol for v in Row[Estimator])
            # the spread describes the bin means, so the minimum has none
            Spread = [v*eV2kJmol for v in Row["Spread"]] if Estimator == "mean" else [np.nan, np.nan]
            print("%-24s %7.3f %9.2f %6i  %-9s %9.2f %8.2f %8.2f  %9.2f %8.2f %8.2f"
                  % (Row["Name"][-24:], Row["tau"], kTEffeV(Row["tau"])*eV2kJmol, Row["N"],
                     Estimator, dEf, Spread[0], dEfErr, dEb, Spread[1], dEbErr))

    # a line in 1/tau needs at least three distinct points to say anything
    Taus = np.array([Row["tau"] for Row in Rows])
    if len(np.unique(np.round(Taus, 6))) < 3:
        print("\n[Barrier] extrapolation to zero kT_eff needs runs at three or more values of tau")
        return None

    print("\nextrapolated linearly in 1/tau (of the %s) to zero kT, weighted by the bootstrap errors, in kJ/mol"
          % ("midpoint ensemble" if Midpoint else "walkers",))
    print("%-9s %-9s %10s %10s %10s %8s" % ("estimator", "barrier", "intercept", "error", "slope", "R2"))
    for Estimator in ESTIMATORS:
        for Name, k in (("forward", 0), ("backward", 2)):
            Values = [Row[Estimator][k]*eV2kJmol for Row in Rows]
            Errors = [Row[Estimator][k+1]*eV2kJmol for Row in Rows]
            Intercept, Slope, R2, InterceptErr = ExtrapolateToZeroKT(Taus, Values, Errors)
            print("%-9s %-9s %10.2f %10.2f %10.2f %8.3f"
                  % (Estimator, Name, Intercept, InterceptErr, Slope, R2))

    return None


if __name__ == '__main__':
    main()
