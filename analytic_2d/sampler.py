#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

SWS Monte-Carlo trajectory on the 2D model potential. Two walkers start from
the reactant and product minima and are biased towards each other until their
mass-weighted RMSD drops below the convergence threshold.

Created on: Mon Aug 24 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import random

import numpy as np

from constants.units import SxCoeff, wvn2eV
from utils.mc.ensemble import PairEnsemble
from utils.mc.kernel import DelayedAcceptanceStep, WalkerState
from utils.mc.probability import calcKELogProbabilityFromRMSD

from .potential import PotentialEnergy, Vpt, X0, alpha, vbath

#%% 1. Trial displacement
def GenDisplacement(tau,mass):

    """

    Draw a Gaussian SWS displacement for a 2D walker.

    :param1 tau:  time parameter tau (float), in fs.
    :param2 mass: mass of the particle in a.m.u.s.

    :return: r: 2D displacement vector in angstroms.

    """

    r = np.zeros(2,dtype=float)
    sigma = SxCoeff*np.sqrt(tau/mass)
    for i in range(0,len(r)):
        r[i] = random.gauss(mu=0.0, sigma=sigma)
    return r

#%% 2. Kinetic energy probability for 2D points
def calcKELogProbability(R1, R2, mass, tau):

    """

    Kinetic-energy part of the acceptance probability for two 2D points.

    :param1 R1:   2D coordinates of the first point, in angstroms.
    :param2 R2:   2D coordinates of the second point, in angstroms.
    :param3 mass: mass of the particle in a.m.u.s.
    :param4 tau:  time parameter tau (float), in fs.

    :return: logP: logarithm of the probability.

    """

    # mass-weighted RMSD in amu*angstroem**2
    mwRMSD = mass * np.dot(R1-R2, R1-R2)

    return calcKELogProbabilityFromRMSD(mwRMSD, tau)

#%% 3. Monte-Carlo trajectory
def RunSingleMCTrajectory(Rreact, Rprod, mass, tau, Ninter, NMax=10000, FolderToSave=".", ScaleThreshold=1., Seed=None, Potential=None, StopOnOverlap=False, Report=None, Quiet=False, MidpointStride=0):

    """

    Run one two-walker SWS Monte-Carlo trajectory on the 2D model potential.

    :param1 Rreact:         2D coordinates of the initial structure, in angstroms.
    :param2 Rprod:          2D coordinates of the final structure, in angstroms.
    :param3 mass:           mass of the particle in a.m.u.s.
    :param4 tau:            time step tau (float), in fs.
    :param5 Ninter:         scaling of tau for the biasing procedure.
    :param6 NMax:           maximal number of Monte-Carlo steps.
    :param7 FolderToSave:   folder for the trajectory dumps.
    :param8 ScaleThreshold: convergence threshold on the dimensionless RMSD.
    :param9 Seed:           seed of the random generator, or None to leave it alone.
    :param10 Potential:     callable returning the energy in cm-1 at a 2D point,
                            or None for the proton-transfer model.
    :param11 StopOnOverlap: also stop once the walkers' progress along the line
                            joining the two starts overlaps.
    :param12 Report:        dict filled with Converged, Reason and Steps, or None.
    :param13 Quiet:         do not print the stopping message.
    :param14 MidpointStride: every this many steps write the midpoint, its
                            energy and its weight onto exp(-tau_mid V(M)/hbar)
                            to midpoint.dat; 0 leaves it out.

    :return: X, Y, E, AccOrNot, ScreenOK, NTrue, Converged: sampled path,
             energies, acceptance flags, geometric pre-screen pass flags, the
             number of cost-bearing energy evaluations and the convergence
             flag.

    Each step is a delayed-acceptance move on the coupled target

        Pi(R1,R2) ~ exp(-tau V(R1)/hbar) exp(-tau V(R2)/hbar)
                    exp(-kappa rho(R1,R2)/tau12),

    i.e. the geometric bias towards the other walker is part of the sampled
    distribution rather than a proposal trick. The pre-screen is tried once per
    step and the walker stays put when it fails, so the proposal normalisation
    cancels exactly. Roughly 1/<screen pass rate> more steps are needed than
    with a re-drawing loop, but the number of energy evaluations per accepted
    move is unchanged.

    The true energy of every trial is computed, including the trials the
    geometric pre-screen rejected, because on an analytic potential it is
    free and it makes the dumps far more informative. Those extra evaluations
    are diagnostics and are not counted in NTrue, which reports what the same
    trajectory would have cost against a real potential.

    Each walker samples exp(-tau V/hbar), so the midpoint of the pair
    represents tau_mid = 2 tau (see PairEnsemble). The midpoint energies are
    counted in NTrue; they draw no random number, so the trajectory is the
    same with or without them.

    """

    if Seed is not None:
        random.seed(Seed)

    if Potential is None:
        def Potential(R):
            return PotentialEnergy(R, Vpt=Vpt, x0=X0, v=vbath, alpha=alpha, mass=mass)

    tau12 = tau * Ninter
    Ensemble = PairEnsemble(tau, tau12)
    count = 0
    Converged = False

    Rvecs = np.array([Rreact,Rprod])
    CurrentEnergy = np.array([Potential(R) for R in Rvecs])
    if Report is None:
        Report = {}
    Report.update(Converged=False, Reason="budget", Steps=NMax)
    # progress along the line joining the starts: walker 1 at 0, walker 2 at 1
    Axis = Rvecs[1] - Rvecs[0]
    Axis = Axis/np.dot(Axis, Axis)
    Reach = [0.0, 1.0]

    outf1 = [open(FolderToSave+"/pretrialed1.trj", "w"),open(FolderToSave+"/pretrialed2.trj", "w")]
    outf2 = [open(FolderToSave+"/rejected1.trj", "w"), open(FolderToSave+"/rejected2.trj", "w")]
    outf3 = [open(FolderToSave+"/sampled1.trj", "w"), open(FolderToSave+"/sampled2.trj", "w")]
    # every point whose energy was computed, whether the move was taken or not.
    # On the analytic potential this is every trial; the xTB sampler only
    # evaluates the ones that pass the pre-screen, hence the screened flag.
    outf4 = [open(FolderToSave+"/evaluated1.dat", "w"), open(FolderToSave+"/evaluated2.dat", "w")]
    for f in outf4:
        f.write("# step[$1] x[$2] y[$3] E,cm-1[$4] screened[$5] accepted[$6]\n")
    MidDump = None
    if MidpointStride > 0:
        MidDump = open(FolderToSave+"/midpoint.dat", "w")
        MidDump.write("# tau_bead,fs = %.10g tau12,fs = %.10g\n" % (Ensemble.TauBead, Ensemble.Tau12))
        MidDump.write("# step[$1] x[$2] y[$3] V(M),cm-1[$4] V1,cm-1[$5] V2,cm-1[$6] logw[$7]\n")

    ConvThreshold = ScaleThreshold

    ScreenOK = []
    AccOrNot = []
    NTrue = 0
    X = []
    Y = []
    E = []
    Converged = False
    def Propose(R):
        return R + GenDisplacement(tau,mass)

    def Energy(R):
        return Potential(R)

    while count < NMax and (not Converged):
        for n in range(0,2):
            OtherIdx = (n + 1) % 2

            # stage 1 is the geometric pre-screen towards the other walker; on
            # the coupled target the bias exp(-kappa*rho/tau12) cancels between
            # target and pre-screen, and the free-particle term is the proposal
            # itself, so only the potential is left in the acceptance test
            def Coupling(trialR):
                return (calcKELogProbability(trialR, Rvecs[OtherIdx], mass, tau12),
                        calcKELogProbability(Rvecs[n], Rvecs[OtherIdx], mass, tau12), None)

            Result = DelayedAcceptanceStep(WalkerState(Rvecs[n], CurrentEnergy[n]),
                                           Propose, Energy, tau, Coupling=Coupling,
                                           EnergyScale=wvn2eV)
            trialR = Result.Trial
            Screened = Result.Screened
            Evaluated = Result.Evaluated
            Accepted = Result.Accepted
            TrialEnergy = Result.TrialEnergy
            ScreenOK.append(int(Screened))
            NTrue += int(Evaluated)

            if not Screened:
                # free on an analytic potential, and the dumps are worth far
                # more with it; it is not counted as a cost
                TrialEnergy = Potential(trialR)
                outf1[n].write(" %7.4f %7.4f %15.3f\n" % (Rvecs[n][0],Rvecs[n][1],CurrentEnergy[n]))
                outf1[n].write(" %7.4f %7.4f %15.3f\n" % (trialR[0],trialR[1],TrialEnergy))
                outf1[n].write(" %7.4f %7.4f %15.3f\n" % (Rvecs[n][0],Rvecs[n][1],CurrentEnergy[n]))

            outf4[n].write(" %7i %7.4f %7.4f %15.3f %3i %3i\n" %
                           (count, trialR[0], trialR[1], TrialEnergy, int(Screened), int(Accepted)))

            if Accepted:
                Rvecs[n] = trialR
                CurrentEnergy[n] = TrialEnergy
                AccOrNot.append(1)
            else:
                if Screened:
                    outf2[n].write(" %7.4f %7.4f %15.3f\n" % (Rvecs[n][0],Rvecs[n][1],CurrentEnergy[n]))
                    outf2[n].write(" %7.4f %7.4f %15.3f\n" % (trialR[0],trialR[1],TrialEnergy))
                    outf2[n].write(" %7.4f %7.4f %15.3f\n" % (Rvecs[n][0],Rvecs[n][1],CurrentEnergy[n]))
                AccOrNot.append(0)
            outf3[n].write(" %7.4f %7.4f %15.3f\n" % (Rvecs[n][0],Rvecs[n][1],CurrentEnergy[n]))

            X.append(Rvecs[n][0])
            Y.append(Rvecs[n][1])
            E.append(CurrentEnergy[n])

        if MidDump is not None and count % MidpointStride == 0:
            RM = 0.5*(Rvecs[0] + Rvecs[1])
            VM = Potential(RM)
            NTrue += 1
            MidDump.write(" %7i %9.5f %9.5f %15.5f %15.5f %15.5f %12.6f\n"
                          % (count, RM[0], RM[1], VM, CurrentEnergy[0], CurrentEnergy[1],
                             float(Ensemble.LogWeight(VM*wvn2eV, CurrentEnergy[0]*wvn2eV,
                                                      CurrentEnergy[1]*wvn2eV))))

        DimLessRMSD = -calcKELogProbability(Rvecs[0], Rvecs[1], mass,  tau)
        if DimLessRMSD <= ConvThreshold:
            if not Quiet:
                print("Criterion reached!")
                print("%f <= %f" % (DimLessRMSD, ConvThreshold) )
            Converged = True
            outfC=open(FolderToSave+"/converged", "w")
            outfC.close()
            Report.update(Converged=True, Reason="meet", Steps=count)

            break
        Reach[0] = max(Reach[0], float(np.dot(Rvecs[0] - Rreact, Axis)))
        Reach[1] = min(Reach[1], float(np.dot(Rvecs[1] - Rreact, Axis)))
        if StopOnOverlap and Reach[0] >= Reach[1]:
            Converged = True
            Report.update(Converged=True, Reason="overlap", Steps=count)

            break

        count += 1

    for i in range(0,2):
        outf1[i].close()
        outf2[i].close()
        outf3[i].close()
        outf4[i].close()
    if MidDump is not None:
        MidDump.close()

    return X,Y,E,AccOrNot,ScreenOK,NTrue,Converged
