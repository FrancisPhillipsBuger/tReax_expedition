#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Parameter scan over the time step tau and the biasing factor Ninter on a 2D
potential. Every (tau, Ninter) pair is sampled with several trajectories and
reduced into one line of ``statistics.res``; ``trials.dat`` keeps every trial,
with the stationary point nearest to where the two walkers met.

Created on: Mon Aug 24 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import os

import numpy as np

from .analysis import CompressPath, ConvPairMeans, SaveMeanPath
from .sampler import RunSingleMCTrajectory

#%% 1. Meeting point
def ClosestLabel(Potential, R):

    """

    :param1 Potential: a Potential2D.
    :param2 R:         2D position.

    :return: Label: name of the stationary point closest to R, 'A', 'S1', ...

    """

    Labels, Points = Potential.StationaryPoints()

    return Labels[int(np.argmin(np.sum((Points - np.asarray(R))**2, axis=1)))]

def QuenchedBasins(Potential, mass):

    """

    :param1 Potential: a Potential2D.
    :param2 mass:      mass of the particle in a.m.u.s.

    :return: A, B: the reactant and product basin guesses after a quench.

    """

    import random

    from utils.mc.quench import Quench, QuenchSchedule
    from utils.mc.system import AnalyticSystem

    State = random.getstate()
    random.seed(0)
    System = AnalyticSystem(Potential, mass)
    Schedule = QuenchSchedule(kT=(0.005, 0.002, 0.001, 0.0005, 0.0001), Steps=400)
    Ends = [Quench(System, np.array(g, dtype=float), 1.0, Schedule=Schedule).X
            for g in Potential.BasinGuesses[:2]]
    random.setstate(State)

    return Ends[0], Ends[1]

#%% 2. Scan driver
def RunParameterScan(Potential, mass, Taus, Ninter, NumTrials=10, NMax=10000,
                     ScaleThreshold=0.5, NCompress=10, FolderToSave=".", Seed=None):

    """

    Scan tau and Ninter, running NumTrials trajectories per pair.

    The walkers start in the reactant and product minima. The first thirteen
    columns of ``statistics.res`` are the trajectory statistics; the kT of
    one walker (the pair's midpoint is at half of it) and the share of trials
    meeting nearest each stationary point follow.

    :param1 Potential:      a Potential2D.
    :param2 mass:           mass of the particle in a.m.u.s.
    :param3 Taus:           list of time steps tau, in fs.
    :param4 Ninter:         list of tau scalings for the biasing procedure.
    :param5 NumTrials:      number of trajectories per (tau, Ninter) pair.
    :param6 NMax:           maximal number of Monte-Carlo steps per trajectory.
    :param7 ScaleThreshold: convergence threshold on the dimensionless RMSD.
    :param8 NCompress:      number of bins for the mean path.
    :param9 FolderToSave:   root folder for the per-run output tree.
    :param10 Seed:          base seed; each trajectory gets a distinct offset of
                            it, so a scan is reproducible as a whole. None
                            leaves the generator untouched.

    :return: None.

    """

    # the walkers start from the basin guesses relaxed by the energy-only
    # quench; the stationary points below only name where the walkers met
    A, B = QuenchedBasins(Potential, mass)
    Labels, _ = Potential.StationaryPoints()
    Unit = Potential.UnitName

    Counter = 0
    Stats = open(os.path.join(FolderToSave, "statistics.res"), "w")
    Stats.write("# tau,fs[$1]  Ninter[$2]  BH,%s[$3,$4]  AccRate,%%[$5,$6]  ScrRate,%%[$7,$8]  Conv,%%[$9]"
                " TrjLen[$10,$11] NTrue[$12,$13] kT_walker,%s[$14] %s\n"
                % (Unit, Unit, " ".join("meet@%s,%%[$%i]" % (l, 15 + i) for i, l in enumerate(Labels))))
    Trials = open(os.path.join(FolderToSave, "trials.dat"), "w")
    Trials.write("# tau,fs[$1] Ninter[$2] trial[$3] conv[$4] BH,%s[$5] xtop[$6] ytop[$7]"
                 " xmeet[$8] ymeet[$9] meet[$10] NTrue[$11]\n" % (Unit,))

    for t in Taus:
        tFN = os.path.join(FolderToSave, "tau_%.2f" % (t))
        os.makedirs(tFN, exist_ok=True)

        for ni in Ninter:
            nFN = os.path.join(tFN, "ninter_%.2f" % (ni))
            os.makedirs(nFN, exist_ok=True)
            Rows = []
            Meet = dict.fromkeys(Labels, 0)
            for it in range(0, NumTrials):
                iFN = os.path.join(nFN, "%05i" % (it))
                os.makedirs(iFN, exist_ok=True)
                TrialSeed = None if Seed is None else Seed + Counter
                Counter += 1
                X, Y, E, AccOrNot, ScreenOK, NTrue, Converged = RunSingleMCTrajectory(
                    A, B, mass=mass, tau=t, Ninter=ni, NMax=NMax, FolderToSave=iFN,
                    ScaleThreshold=ScaleThreshold, Seed=TrialSeed, Potential=Potential)

                s, x, y, e, n = CompressPath(A, B, X, Y, np.asarray(E)/Potential.Unit,
                                             N=NCompress, Axis=Potential.PathAxis)
                SaveMeanPath(s, x, y, e, n, FolderToSave=iFN)
                Top = int(np.argmax(e[:, 0]))
                # the last two samples are the two walkers after the last step
                Mid = 0.5*(np.array([X[-1], Y[-1]]) + np.array([X[-2], Y[-2]]))
                Label = ClosestLabel(Potential, Mid)
                Meet[Label] += 1

                Rows.append((np.max(e[:, 0]) - np.min(e[:, 0]), np.mean(AccOrNot), np.mean(ScreenOK),
                             float(Converged), len(X), float(NTrue)))
                Trials.write(" %9.3f %9.3f %5i %2i %12.3f %8.4f %8.4f %8.4f %8.4f %4s %8i\n"
                             % (t, ni, it, int(Converged), Rows[-1][0], x[Top, 0], y[Top, 0],
                                Mid[0], Mid[1], Label, NTrue))

            Rows = np.array(Rows)
            Stats.write(" %10.3f %10.3f " % (t, ni))
            for c, Scale in ((0, 1.0), (1, 100.0), (2, 100.0)):
                m, err = ConvPairMeans(Rows[:, c].sum(), (Rows[:, c]**2).sum(), NumTrials)
                Stats.write(" %10.2f %10.2f " % (m*Scale, err*Scale))
            Stats.write(" %10.2f  " % (100.0*Rows[:, 3].mean(),))
            for c, Scale in ((4, 1.0), (5, 1.0)):
                m, err = ConvPairMeans(Rows[:, c].sum(), (Rows[:, c]**2).sum(), NumTrials)
                Stats.write(" %10.2f %10.2f " % (m*Scale, err*Scale))
            Stats.write(" %10.4g " % (Potential.kTAt(t),))
            Stats.write(" ".join("%8.1f" % (100.0*Meet[l]/NumTrials) for l in Labels) + "\n")
            Stats.flush()
            Trials.flush()
        Stats.write("\n")

    Stats.close()
    Trials.close()

    return None
