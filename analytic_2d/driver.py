#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Command-line driver shared by run_2d.py, run_muller.py and run_two_paths.py:
the staged search by default; with --scan, characterise the potential, write
the PES files, then scan tau and Ninter as in the reference work.

Created on: Sun Sep 27 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import argparse
import os
import sys

from .pes_io import SaveCuts, SavePES, SaveStationaryPoints
from .potential import GetPotential, Potentials, ReducedPotential
from .scan import RunParameterScan

#%% 1. CLI argument parser
def BuildScanParser(Name):

    """

    Build the argparse parser of the scan for one potential.

    :param1 Name: '2d', 'muller' or 'two_paths'.

    :return: Parser: argparse.ArgumentParser with the defaults of that potential.

    """

    Class = Potentials[Name]
    Reduced = issubclass(Class, ReducedPotential)
    p = argparse.ArgumentParser(description='SWS Monte-Carlo on the %s potential' % (Name,))
    p.add_argument('--outdir', type=str, default='.',
                   help='Root folder for the PES files and the scan tree')
    p.add_argument('--num_trials', type=int, default=10,
                   help='Number of trajectories per (tau, Ninter) pair')
    p.add_argument('--n_max', type=int, default=10000,
                   help='Maximal number of Monte-Carlo steps per trajectory')
    p.add_argument('--threshold', type=float, default=0.5,
                   help='Convergence threshold on the dimensionless RMSD')
    p.add_argument('--n_compress', type=int, default=10,
                   help='Number of bins for the mean path')
    p.add_argument('--seed', type=int, default=None,
                   help='Base seed for the Monte-Carlo generator (default: unseeded)')
    p.add_argument('--mass', type=float, default=Class.DefaultMass,
                   help='Mass of the particle in amu. Default: %g.' % (Class.DefaultMass,))
    Scan = p.add_mutually_exclusive_group()
    Scan.add_argument('--tau', type=float, nargs='+', default=None,
                      help='Taus to scan, in fs')
    Scan.add_argument('--kt', type=float, nargs='+', default=None,
                      help='Temperatures of one walker to scan, in %s; each sets tau = hbar/kT, the pair midpoint being at kT/2'
                      % (Class.UnitName,))
    p.add_argument('--ninter', type=float, nargs='+', default=[0.01, 0.1, 1.0, 10.0, 100.0],
                   help='Tau scalings of the coupling between the walkers')
    if Reduced:
        p.add_argument('--temperature', type=float, default=300.0,
                       help='Temperature in K whose kT is one reduced unit. Default: 300.')
        p.add_argument('--no_walls', dest='walls', action='store_false',
                       help='Leave out the harmonic walls of the committor_2.0 PLUMED input')

    return p

#%% 2. Driver
def RunScanCLI(Name):

    """

    Parse the command line and run the scan on one potential.

    :param1 Name: '2d', 'muller' or 'two_paths'.

    :return: None.

    """

    # the staged search is the default; the (tau, Ninter) scan of the
    # reference work stays available behind --scan
    if "--scan" not in sys.argv[1:]:
        from .pipeline import RunSearchCLI
        return RunSearchCLI(Name)
    sys.argv.remove("--scan")

    args = BuildScanParser(Name).parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    Potential = GetPotential(Name, Temperature=getattr(args, "temperature", 300.0),
                             Walls=getattr(args, "walls", True))
    Potential.Describe(args.mass)
    SavePES(Potential, os.path.join(args.outdir, "pes.dat"))
    SaveCuts(Potential, FolderToSave=args.outdir)
    SaveStationaryPoints(Potential, os.path.join(args.outdir, "stationary.dat"), Mass=args.mass)

    if args.tau is not None:
        Taus = list(args.tau)
    elif args.kt is not None:
        Taus = [Potential.TauAtkT(kT) for kT in args.kt]
    else:
        Taus = Potential.DefaultScanTaus()
    print("[Scan] tau = %s fs" % (", ".join("%.2f" % t for t in Taus),))
    print("[Scan] kT_eff = %s %s" % (", ".join("%.4g" % Potential.kTAt(t) for t in Taus), Potential.UnitName))

    RunParameterScan(Potential, args.mass, Taus, args.ninter, NumTrials=args.num_trials,
                     NMax=args.n_max, ScaleThreshold=args.threshold, NCompress=args.n_compress,
                     FolderToSave=args.outdir, Seed=args.seed)
    print("[Output] %s: pes.dat, pes_x.dat, pes_y.dat, stationary.dat, statistics.res, trials.dat,"
          " tau_*/ninter_*/*" % (args.outdir,))

    return None
