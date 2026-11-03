#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

SWS Monte-Carlo sampling of molecular structures on any ASE calculator.
Provides a single-structure random walk and a two-structure run in which the
endpoints are biased towards each other until they meet. Both accept a
cheaper level of theory that screens the trial moves before the true energy.

Created on: Mon Aug 24 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import random

import math

from .calculators import wrapGetPotentialEnergy
from .geometry import (EndpointAxis, MaybeCenter, MaybeOrient, NeedsFrameRemoval,
                       calcAlignedRMSDBetweenStructures, calcRMSDBetweenStructures,
                       displaceSWSStructure, orientPairOnto, rotateStructureRandomly,
                       rotationWidth)
from .io_xyz import EvaluationLog, writeXYZFile
from utils.mc.ensemble import PairEnsemble
from utils.mc.kernel import DelayedAcceptanceStep, WalkerState
from utils.mc.probability import calcKELogProbabilityFromRMSD

#%% 1. Run statistics and fail-safe evaluation
class RunStats:

    """

    Counters of what the energy calls of a run achieved.

    The acceptance rate of the evaluated trials is the number that says
    whether the step width is well chosen: close to one means most of the
    evaluations were spent on moves too small to matter.

    """

    def __init__(self):

        """

        Start every counter at zero.

        """

        self.NEvaluated = 0
        self.NAccepted = 0
        self.NFailed = 0

    def Summary(self):

        """

        One line describing the counters.

        :return: Line: the summary (str).

        """

        Rate = self.NAccepted/self.NEvaluated if self.NEvaluated > 0 else float("nan")
        Line = ("[Sampler] evaluated trials: %i, accepted: %i (%.2f), failed: %i"
                % (self.NEvaluated, self.NAccepted, Rate, self.NFailed))

        return Line

def evaluateTrialEnergy(Structure):

    """

    True energy of a trial structure, or None when the back end cannot give one.

    A trial whose SCF does not converge is rejected rather than allowed to end
    the run. This is the Metropolis test with the energy taken as infinite
    there, which restricts the sampled distribution to the geometries the
    back end can describe; the call is still paid for and still counted.

    :param1 Structure: ase.Atom instance with an attached calculator.

    :return: Energy: potential energy in eV, or None on failure.

    """

    try:
        return wrapGetPotentialEnergy(Structure)
    except RuntimeError as Error:
        print("[Sampler] trial rejected, energy evaluation failed: %s" % (Error.__cause__ or Error,))

        return None

def checkFiniteStart(Name, Values):

    """

    Refuse to start a chain from a structure where a cheap term is not finite.

    A screen that fails at a trial only rejects that trial, but one that
    fails at the starting point would reject every move out of it.

    :param1 Name:   what the values are, for the message (str).
    :param2 Values: the values at the starting structures.

    :return: None.

    """

    for v in Values:
        if not math.isfinite(v):
            raise ValueError("The %s is not finite at a starting structure; "
                             "the chain could never leave it." % (Name,))

    return None

#%% 2. Single-structure step
def mcSingleStructureStep(Structure, CurrentEnergy, tau, SingleAtom=True, EvalSink=None, AlignFrames=None, Surrogate=None, CurrentSurrogate=0.0, StepScale=1.0, Stats=None):

    """

    Perform one Monte-Carlo step from a single structure.

    With a surrogate the move becomes a delayed acceptance one, exactly as in
    the two-structure step: a cheap first stage on the surrogate, and a second
    stage on the residual between the true potential and the surrogate, which
    is what keeps the invariant distribution untouched.

    :param1 Structure:        ase.Atom instance with an attached calculator.
    :param2 CurrentEnergy:    current potential energy of the structure.
    :param3 tau:              time parameter tau (float), in fs.
    :param4 SingleAtom:       flag (bool) to move only one randomly chosen atom.
    :param5 EvalSink:         optional EvaluationLog recording every energy call.
    :param6 AlignFrames:      remove the overall translation and rotation; None
                              decides from the cell and the constraints.
    :param7 Surrogate:        cheap approximation of the potential used to screen
                              proposals before paying for a true evaluation, or
                              None. It must be held fixed while the chain runs.
    :param8 CurrentSurrogate: surrogate energy of the current structure.
    :param9 StepScale:        multiplier of the free-particle step width.
    :param10 Stats:           optional RunStats collecting the counters.

    :return: Structure, CurrentEnergy, Updated, CurrentSurrogate, Evaluated:
             updated (or the same) structure, its energy, whether the step was
             accepted, the cached surrogate value, and whether the move
             reached the calculator at all.

    """

    if AlignFrames is None:
        AlignFrames = NeedsFrameRemoval(Structure)

    def Propose(S):
        Trial = displaceSWSStructure(S, tau, SingleAtom=SingleAtom, SingleCoordinate=False, Debug=False, StepScale=StepScale)
        Trial = MaybeCenter(Trial, AlignFrames)
        return MaybeOrient(StructureToOrient=Trial, ReferenceStructure=S,
                           Align=AlignFrames, UseMassWeightedRMSD=True)

    # the trial is drawn from a symmetric Gaussian, the free-particle kernel
    # scaled by StepScale, so the proposal density cancels in the Metropolis
    # ratio; the invariant distribution is exp(-tau*V/hbar)
    Result = DelayedAcceptanceStep(WalkerState(Structure, CurrentEnergy, Surrogate=CurrentSurrogate),
                                   Propose, evaluateTrialEnergy, tau, Surrogate=Surrogate)
    Updated = Result.Accepted
    Evaluated = Result.Evaluated
    Failed = Result.Failed
    Structure = Result.State.Config
    CurrentEnergy = Result.State.Energy
    CurrentSurrogate = Result.State.Surrogate

    # only the trials that reached the calculator cost an energy call; a failed
    # one has no energy to record
    if Evaluated and not Failed and EvalSink is not None:
        EvalSink.Record(Result.Trial, Result.TrialEnergy, Updated)
    if Stats is not None:
        Stats.NEvaluated += int(Evaluated)
        Stats.NAccepted += int(Updated)
        Stats.NFailed += int(Failed)

    return Structure, CurrentEnergy, Updated, CurrentSurrogate, Evaluated

#%% 3. Two-structure step
def mcTwoStructuresStep(Structures, CurrentEnergies, tau, tau12, SingleAtom=True, EvalSinks=None, AlignFrames=None, Surrogate=None, CurrentSurrogates=None, StepScale=1.0, Stats=None, CurrentPairRMSD=None):

    """

    Perform one Monte-Carlo step for two structures moving towards each other.

    The step is a delayed-acceptance move on the coupled target

        Pi(R1,R2) ~ exp(-tau*V(R1)/hbar) exp(-tau*V(R2)/hbar)
                    exp(-kappa*mwRMSD(R1,R2)/tau12),

    which is symmetric in the two structures, so updating them one after the
    other is a valid two-block sweep. The mwRMSD of the coupling is taken in
    the common frame of the two structures, centred but never rotated one onto
    the other: a rotation-minimised one would weight each structure by its
    rotational orbit volume. Their relative orientation is sampled by a rigid
    rotation of the second structure, which costs no energy. The bias towards the other structure is
    part of Pi, hence it is not undone by the acceptance test; the pre-screen
    is tried once and the structure stays put when it fails, which keeps the
    proposal normalisation out of the acceptance probability.

    The mwRMSD between the two walkers is carried from one update to the next:
    it only changes when a move is accepted, and then it is the one already
    computed for the trial.

    :param1 Structures:      list of two ase.Atom instances with attached calculators.
    :param2 CurrentEnergies: current potential energies of the structures.
    :param3 tau:             time parameter tau (float), in fs.
    :param4 tau12:           time parameter for the interconversion (float), in fs.
    :param5 SingleAtom:      flag (bool) to move only one randomly chosen atom.
    :param6 EvalSinks:       optional list of two EvaluationLog instances, one
                             per walker, recording every energy call.
    :param7 AlignFrames:     remove the overall translation and rotation; None
                             decides from the cell and the constraints.
    :param8 Surrogate:       cheap approximation of the potential used to screen
                             proposals before paying for a true evaluation, or
                             None. It must be held fixed while the chain runs.
    :param9 CurrentSurrogates: surrogate energies of the two structures.
    :param10 StepScale:      multiplier of the free-particle step width.
    :param11 Stats:          optional RunStats collecting the counters.
    :param12 CurrentPairRMSD: mwRMSD between the two structures, or
                             None to compute it here.

    :return: Structures, CurrentEnergies, Updated, Screened, mwRMSDs,
             CurrentSurrogates, SurrogatePassed: updated structures and
             energies, the acceptance flags, the geometric pre-screen pass
             flags, the mwRMSDs towards the other structure, the cached
             surrogate values, and the surrogate screen pass flags. The two
             sets of pass flags are kept apart so the diagnostics say which
             screen rejected a move.

    """

    if AlignFrames is None:
        AlignFrames = NeedsFrameRemoval(Structures[0])
    if CurrentSurrogates is None:
        CurrentSurrogates = [0.0 if Surrogate is None else float(Surrogate(S))
                             for S in Structures]
    PairRMSD = CurrentPairRMSD
    if PairRMSD is None:
        PairRMSD = calcRMSDBetweenStructures(Structures[0], Structures[1])

    def Propose(S):
        Trial = displaceSWSStructure(S, tau, SingleAtom=SingleAtom, SingleCoordinate=False, Debug=False, StepScale=StepScale)
        return MaybeCenter(Trial, AlignFrames)

    Screened = []
    SurrogatePassed = []
    Updated = []
    mwRMSDs = []
    for n in range(0,2):
        Other = Structures[(n + 1) % 2]

        # stage 1: cheap geometric pre-screen towards the other structure, on
        # the mwRMSD in the common frame; the pre-screen bias cancels against
        # the coupling term of the target
        def Coupling(Trial):
            TrialRMSD = calcRMSDBetweenStructures(Trial, Other)
            return (calcKELogProbabilityFromRMSD(TrialRMSD, tau12),
                    calcKELogProbabilityFromRMSD(PairRMSD, tau12), TrialRMSD)

        Result = DelayedAcceptanceStep(WalkerState(Structures[n], CurrentEnergies[n],
                                                   Surrogate=CurrentSurrogates[n]),
                                       Propose, evaluateTrialEnergy, tau, Coupling=Coupling,
                                       Surrogate=Surrogate)
        Screened.append(Result.Screened)
        SurrogatePassed.append(Result.PassedSurrogate)
        Updated.append(Result.Accepted)

        if Result.Accepted:
            Structures[n] = Result.State.Config
            CurrentEnergies[n] = Result.State.Energy
            CurrentSurrogates[n] = Result.State.Surrogate
            PairRMSD = Result.CouplingInfo

        # only the trials that reached the calculator cost an energy call; a
        # failed one has no energy to record
        if Result.Evaluated and not Result.Failed and EvalSinks is not None:
            EvalSinks[n].Record(Result.Trial, Result.TrialEnergy, Result.Accepted)
        if Stats is not None:
            Stats.NEvaluated += int(Result.Evaluated)
            Stats.NAccepted += int(Result.Accepted)
            Stats.NFailed += int(Result.Failed)

        # mwRMSD of the structure the walker actually holds after the step
        mwRMSDs.append(PairRMSD)

    if AlignFrames:
        # the relative orientation: a rigid rotation leaves the energy of the
        # second structure as it was, so only the coupling is tested and no
        # energy is paid for; a surrogate need not be invariant, so its cached
        # value follows the structure
        Trial = rotateStructureRandomly(Structures[1], rotationWidth(Structures[1], tau12))
        TrialRMSD = calcRMSDBetweenStructures(Structures[0], Trial)
        if math.log(random.random()) < (calcKELogProbabilityFromRMSD(TrialRMSD, tau12)
                                        - calcKELogProbabilityFromRMSD(PairRMSD, tau12)):
            Structures[1] = Trial
            PairRMSD = TrialRMSD
            mwRMSDs[1] = PairRMSD
            if Surrogate is not None:
                CurrentSurrogates[1] = float(Surrogate(Trial))

    return (Structures, CurrentEnergies, Updated, Screened, mwRMSDs,
            CurrentSurrogates, SurrogatePassed)

#%% 4. Single-structure run
def doSingleStructureSWSMC(Structure, tau, Nsteps, Neq=0, SingleAtom=True, Seed=None, LogEvaluations=True, AlignFrames=None, Surrogate=None, FolderToSave=".", StepScale=1.0):

    """

    Run the Monte-Carlo sampling from a single structure.

    :param1 Structure:  ase.Atom instance with an attached calculator.
    :param2 tau:        time parameter tau (float), in fs.
    :param3 Nsteps:     total number of steps.
    :param4 Neq:        number of initial steps ignored as an equilibration phase.
    :param5 SingleAtom: flag (bool) to move only one randomly chosen atom.
    :param6 Seed:       seed of the random generator, or None to leave it alone.
    :param7 LogEvaluations: also dump every structure whose energy was computed.
    :param8 AlignFrames: remove the overall translation and rotation; None
                        decides from the cell and the constraints.
    :param9 Surrogate:  cheap approximation of the potential used to screen
                        proposals, or None.
    :param10 FolderToSave: folder for the trajectory and energy dumps.
    :param11 StepScale: multiplier of the free-particle step width.

    :return: NTrue: the number of true energy evaluations the run spent, which
             is the cost that has to be quoted against any other method.

    """

    if Seed is not None:
        random.seed(Seed)

    TrjDumpFile=FolderToSave+"/sampled.trj"
    if Neq>0:
        TrjDumpFile=FolderToSave+"/equilibration.trj"
        EnDump=open(FolderToSave+"/equilibration_energy.log", "w")
    else:
        EnDump=open(FolderToSave+"/sampled_energy.log", "w")

    # remove CoM
    if AlignFrames is None:
        AlignFrames = NeedsFrameRemoval(Structure)

    EvalSink = EvaluationLog(0, FolderToSave=FolderToSave, Enabled=LogEvaluations)
    Stats = RunStats()

    Structure = MaybeCenter(Structure, AlignFrames)
    Energy = wrapGetPotentialEnergy(Structure) #  energy in eV
    NTrue = 1
    EvalSink.Record(Structure, Energy, True)
    CurrentSurrogate = 0.0 if Surrogate is None else float(Surrogate(Structure))
    checkFiniteStart("surrogate", [CurrentSurrogate])
    EnDump.write(" %8.2f  %15.10f   %2i\n" % (0, Energy, int(True)))
    writeXYZFile(Structure, TrjDumpFile, Append=True, PrintStuff=False)
    for i in range(1,Nsteps):
        # if equilibration part ended, switch output files
        if i == Neq:
            TrjDumpFile=FolderToSave+"/sampled.trj"
            EnDump.close()
            EnDump=open(FolderToSave+"/sampled_energy.log", "w")

        EvalSink.Step = i
        (Structure, Energy, Updated, CurrentSurrogate,
         Evaluated) = mcSingleStructureStep(Structure, Energy, tau, SingleAtom=SingleAtom,
                                            EvalSink=EvalSink, AlignFrames=AlignFrames,
                                            Surrogate=Surrogate,
                                            CurrentSurrogate=CurrentSurrogate,
                                            StepScale=StepScale, Stats=Stats)
        NTrue += int(Evaluated)
        EnDump.write(" %8.2f  %15.10f   %2i\n" % (i*tau, Energy, int(Updated)))
        EnDump.flush()
        writeXYZFile(Structure, TrjDumpFile, Append=True, PrintStuff=False)

    EnDump.close()
    EvalSink.close()
    print(Stats.Summary())

    return NTrue

#%% 5. Two-structure run
def doTwoStructureSWSMC(Structures, tau, tau12, Nsteps, ScaleThreshold = 1.0, SingleAtom=True, Seed=None, LogEvaluations=True, AlignFrames=None, Surrogate=None, FolderToSave=".", StepScale=1.0, StopOnOverlap=False, Report=None, Quiet=False, MidpointCalculator=None, MidpointStride=10):

    """

    Run the Monte-Carlo sampling from two structures until they meet.

    Each walker samples exp(-tau V/hbar), so the pair is the PairEnsemble of
    bead tau tau: its midpoint represents tau_mid = 2 tau, not tau. With a
    midpoint calculator the midpoint energy is paid for every MidpointStride
    steps and written with its weight onto exp(-tau_mid V(M)/hbar); it draws
    no random number, so the trajectory is the same with or without it.

    :param1 Structures:     list of two ase.Atom instances with attached calculators.
    :param2 tau:            time parameter tau (float), in fs.
    :param3 tau12:          time parameter for the interconversion (float), in fs.
    :param4 Nsteps:         total number of steps. Steps whose pre-screen fails
                            cost no energy evaluation but do not move the
                            structures, so Nsteps has to be larger than with a
                            re-drawing biasing loop by about 1/(pass rate).
    :param5 ScaleThreshold: scaling of the threshold for stopping the simulation;
                            the default is the one of the command line.
    :param6 SingleAtom:     flag (bool) to move only one randomly chosen atom.
    :param7 Seed:           seed of the random generator, or None to leave it alone.
    :param8 LogEvaluations: also dump every structure whose energy was computed.
    :param9 AlignFrames:    remove the overall translation and rotation; None
                            decides from the cell and the constraints.
    :param10 Surrogate:     cheap approximation of the potential used to screen
                            proposals, or None.
    :param11 FolderToSave:  folder for the trajectory and energy dumps.
    :param12 StepScale:     multiplier of the free-particle step width.
    :param13 StopOnOverlap: also stop once the two walkers' progress along the
                            endpoint axis overlaps: walker 1 has been as far
                            towards the second endpoint as walker 2 has been
                            towards the first.
    :param14 Report:        dict filled with Converged, Reason and Steps, or None.
    :param15 Quiet:         do not print the stopping message and the counters.
    :param16 MidpointCalculator: ASE calculator for the midpoint energies, or
                            None to leave the midpoint out.
    :param17 MidpointStride: evaluate the midpoint every this many steps.

    :return: NTrue: the number of true energy evaluations the run spent, which
             is the cost that has to be quoted against any other method.

    """

    if Seed is not None:
        random.seed(Seed)
    if Report is None:
        Report = {}
    Report.update(Converged=False, Reason="budget", Steps=Nsteps)
    Ensemble = PairEnsemble(tau, tau12)

    TrjDumpFiles=[FolderToSave+"/sampled1.trj", FolderToSave+"/sampled2.trj"]
    EnDumps=[open(FolderToSave+"/sampled_energy1.log", "w"), open(FolderToSave+"/sampled_energy2.log", "w")]
    # the reaction coordinate of every row of the energy logs, which is what
    # analyse.LoadRun bins the energies along. It is measured from the
    # final endpoint, so the initial one falls in the last bin, the convention
    # BarriersFromProfile shares with r(C-Cl) - r(C-Br) on the SN2 benchmark
    Axis = EndpointAxis(Structures[1], Structures[0])
    ICDumps=[open(FolderToSave+"/ic%i.trj" % (n+1), "w") for n in range(0,2)]
    for f in ICDumps:
        f.write("# t,fs[$1] s[$2]: progress along the mass-weighted endpoint axis, 1 at the initial endpoint and 0 at the final one\n")
    EvalSinks=[EvaluationLog(n+1, FolderToSave=FolderToSave, Enabled=LogEvaluations)
               for n in range(0,2)]
    Stats = RunStats()
    NumOfAtoms = len(Structures[0].get_masses())
    ConvThreshold = ScaleThreshold * 1.5 * NumOfAtoms
    if AlignFrames is None:
        AlignFrames = NeedsFrameRemoval(Structures[0])

    # remove CoM, unless the cell or the constraints already pin the frame
    for n in range(0,2):
        Structures[n] = MaybeCenter(Structures[n], AlignFrames)
    Structures[1] = MaybeOrient(StructureToOrient=Structures[1], ReferenceStructure=Structures[0],
                                Align=AlignFrames, UseMassWeightedRMSD=True)
    mwRMSD = calcRMSDBetweenStructures(Structures[1], Structures[0])
    PairFrame = 0.5*(Structures[0].get_positions() + Structures[1].get_positions())
    Initial = -calcKELogProbabilityFromRMSD(mwRMSD, tau)/ConvThreshold
    Energies = [wrapGetPotentialEnergy(S) for S in Structures] # energy in eV
    NTrue = 2
    Surrogates = [0.0, 0.0] if Surrogate is None else [float(Surrogate(S)) for S in Structures]
    checkFiniteStart("surrogate", Surrogates)
    for n in range(0,2):
        EvalSinks[n].Record(Structures[n], Energies[n], True)
        # the same six columns as every later row, so the file parses as a table
        EnDumps[n].write(" %8.2f  %15.10f   %2i %5i %10.5e %10.5e\n" %
                         (0, Energies[n], int(True), 0, mwRMSD, Initial))
        ICDumps[n].write(" %8.2f  %12.6f\n" % (0, Axis(Structures[n].get_positions())))
        writeXYZFile(Structures[n], TrjDumpFiles[n], Append=True, PrintStuff=False)
    # walker 1 starts at progress 1 and walker 2 at 0
    Reach = [1.0, 0.0]
    MidDump = None
    if MidpointCalculator is not None:
        Mid = Structures[0].copy()
        Mid.calc = MidpointCalculator
        MidDump = open(FolderToSave+"/midpoint_energy.log", "w")
        MidDump.write("# tau_bead,fs = %.10g tau12,fs = %.10g\n" % (Ensemble.TauBead, Ensemble.Tau12))
        MidDump.write("# t,fs[$1] s[$2] V(M),eV[$3] V1,eV[$4] V2,eV[$5] logw[$6]: weight onto exp(-tau_mid V(M)/hbar), tau_mid = 2 tau_bead\n")

    for i in range(1,Nsteps):
        for n in range(0,2):
            EvalSinks[n].Step = i
        (Structures, Energies, Updated, Screened, mwRMSDs, Surrogates,
         SurrogatePassed) = mcTwoStructuresStep(Structures, Energies, tau, tau12,
                                                SingleAtom=SingleAtom, EvalSinks=EvalSinks,
                                                AlignFrames=AlignFrames, Surrogate=Surrogate,
                                                CurrentSurrogates=Surrogates,
                                                StepScale=StepScale, Stats=Stats,
                                                CurrentPairRMSD=mwRMSD)
        NTrue += sum(1 for n in range(0, 2) if Screened[n] and SurrogatePassed[n])

        mwRMSD = mwRMSDs[1]
        # the walkers meet when their shapes do, whatever their orientation
        DimLessRMSD = -calcKELogProbabilityFromRMSD(
            calcAlignedRMSDBetweenStructures(Structures[1], Structures[0], AlignFrames), tau)
        # both walkers are dumped in their common frame, turned as one onto the
        # starting midpoint; the chain itself is never rotated this way
        S0, S1 = Structures
        if AlignFrames:
            S0, S1 = orientPairOnto(Structures, PairFrame)
        for n, Frame in enumerate((S0, S1)):
            EnDumps[n].write(" %8.2f  %15.10f   %2i %5i %10.5e %10.5e\n" % (i*tau, Energies[n], int(Updated[n]), int(Screened[n]), mwRMSDs[n], DimLessRMSD/ConvThreshold))
            EnDumps[n].flush()
            s = Axis(Frame.get_positions())
            Reach[n] = min(Reach[n], s) if n == 0 else max(Reach[n], s)
            ICDumps[n].write(" %8.2f  %12.6f\n" % (i*tau, s))
            ICDumps[n].flush()
            writeXYZFile(Frame, TrjDumpFiles[n], Append=True, PrintStuff=False)
        if MidDump is not None and i % MidpointStride == 0:
            Mid.set_positions(0.5*(S0.get_positions() + S1.get_positions()))
            VM = evaluateTrialEnergy(Mid)
            NTrue += 1
            if VM is not None:
                MidDump.write(" %8.2f  %12.6f  %15.10f  %15.10f  %15.10f  %12.6f\n"
                              % (i*tau, Axis(Mid.get_positions()), VM, Energies[0], Energies[1],
                                 float(Ensemble.LogWeight(VM, Energies[0], Energies[1]))))
                MidDump.flush()
                writeXYZFile(Mid, FolderToSave+"/midpoint.trj", Append=True, PrintStuff=False)

        if DimLessRMSD <= ConvThreshold:
            if not Quiet:
                print("Criterion reached!")
                print("%f <= %f" % (DimLessRMSD, ConvThreshold) )
            Report.update(Converged=True, Reason="meet", Steps=i)
            break
        if StopOnOverlap and Reach[0] <= Reach[1]:
            Report.update(Converged=True, Reason="overlap", Steps=i)
            break

    for n in range(0,2):
        EnDumps[n].close()
        ICDumps[n].close()
        EvalSinks[n].close()
    if MidDump is not None:
        MidDump.close()
    if not Quiet:
        print(Stats.Summary())

    return NTrue
