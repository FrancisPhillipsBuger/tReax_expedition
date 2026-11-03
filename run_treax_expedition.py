#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

CLI entry point of the gradient-free transition state search on a reaction
under reaction/input (SN2 on ORCA HF-3c, alanine dipeptide on AMBER ff99SB,
the SACTS set on GFN2-xTB; the back end comes from meta.json). The default
runs the staged search built on MCTSS: endpoint quench, MCTSS on a tau
ladder, zero-temperature labels, certified continuation chains, the
separatrix crossing, and with search+committor the committor at the physical
temperature. --stages explore runs one MCTSS sampling exactly as before.

Created on: Mon Aug 24 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from treax_expedition.calculators import getEMTCalculator
from treax_expedition.io_xyz import (OUTPUT_ROOT, prepareOutputFolder, reactionName,
                              readMeta, readXYZFile)
from treax_expedition.calculators import setCalculatorToStructure
from treax_expedition.orca import getORCACalculator
from treax_expedition.orca import DEFAULT_METHOD as ORCA_DEFAULT_METHOD
from treax_expedition.xtb import getXTBCalculator, normaliseMethod
from treax_expedition.sampler import doSingleStructureSWSMC, doTwoStructureSWSMC

#%% 1. CLI argument parser
def parse_args():

    """

    Build the argparse parser for the SWS Monte-Carlo sampler.

    :return: Parsed arguments and the unknown ones.

    """

    parser = argparse.ArgumentParser(description='Gradient-free transition state search built on SWS Monte-Carlo transition state sampling (MCTSS). The default runs the staged search on a reaction under reaction/input; --stages explore runs one MCTSS sampling as before.')

    parser.add_argument('--stages', type=str, default='search', choices=['search', 'search+committor', 'explore'],
                    help="'search': S0 endpoint quench, S1 MCTSS tau ladder, S2 quench labels, S3 certified continuation chains, S5 separatrix crossing. 'search+committor' adds S4, the committor at the physical temperature with shots. 'explore': one MCTSS run with the options below, as in the reference work. Default: search.")

    parser.add_argument('-f', '--file', type=str, default='sn2',
                    help="XYZ file with 1 or 2 structures, a folder holding reactant.xyz and product.xyz, or the name of a reaction under reaction/input. Default is 'sn2'.")

    parser.add_argument('-t', '--tau', type=float, default=1.,
                    help="Time parameter tau in fs (float): each walker samples exp(-tau V/hbar), kT = hbar/tau, and tau sets the proposal width; the midpoint of a two-structure pair represents 2 tau, i.e. kT/2. Default is 1.")

    parser.add_argument('-s', '--steps', type=int, default=10,
                    help="Total number of steps (int). Default is 10.")

    parser.add_argument('-e', '--eq', type=int, default=0,
                    help="Total number of initial equilibration steps to ignore (int). Default is 0.")

    parser.add_argument('-p', '--nprocs', type=int, default=1,
                    help="Number of OpenMP threads for the tight binding back end. Default is 1.")

    parser.add_argument('-o', '--orcapath', type=str, default=None,
                    help="Path to the ORCA executable. Default is None (take the ASE configuration). Ignored unless the back end is 'orca'.")

    parser.add_argument('-m', '--method', type=str, default=None,
                    help="For 'xtb', the parametrisation: 'gfn2-xtb', 'gfn1-xtb' or 'ipea1-xtb'. For 'orca', the level of theory, e.g. 'hf-3c'. Default: gfn2-xtb on xtb, hf-3c on orca.")

    parser.add_argument('-b', '--basis', type=str, default=None,
                    help="Basis set to be used (str). Default is None. Ignored unless the back end is 'orca'.")

    parser.add_argument('--t12', type=float, default=None,
                    help="By how much tau12, the tau of the coupling between the two walkers, is larger than tau (float). Default: 10 for --stages explore, 3 for the search.")

    parser.add_argument('--charge', type=int, default=None,
                    help="Charge of the system (int). Default: the reaction's meta.json, or 0.")

    parser.add_argument('--mult', type=int, default=None,
                    help="Multiplicity of the system (int). Default: the reaction's meta.json, or 1.")

    # the flag asks for all atoms, so it clears the single atom switch it feeds
    parser.add_argument('--all', dest='single_atom', action='store_false',
                    help="Flag to move all atoms during the MC. Default: False.")

    parser.add_argument('--threshold', type=float, default=1.,
                    help="Matching threshold for structures to meet (float). Default is 1.")

    parser.add_argument('--seed', type=int, default=None,
                    help="Seed for the Monte-Carlo generator (int). Default: None (unseeded).")

    parser.add_argument('--calculator', type=str, default=None, choices=['xtb', 'orca', 'amber', 'emt'],
                    help="Energy back end: GFN2-xTB 'xtb', the file driven 'orca', AMBER ff99SB 'amber' (alanine dipeptide), or the free ASE 'emt'. Default: the reaction's meta.json for the search, xtb for --stages explore.")

    parser.add_argument('--nactel', type=int, default=2,
                    help="Number of active electrons for CASSCF (int). Default: 2. Not used unless the method is 'CASSCF'.")

    parser.add_argument('--nactorb', type=int, default=2,
                    help="Number of active orbitals for CASSCF (int). Default: 2. Not used unless the method is 'CASSCF'.")

    parser.add_argument('--state', type=int, default=0,
                    help="Active excited state for TDDFT/CASSCF (int). Default: 0 (ground). Only the ORCA back end has excited states.")

    parser.add_argument('--accuracy', type=float, default=1.,
                    help="Numerical thresholds of the tight binding back end, smaller is tighter (float). Default: 1. Ignored unless the back end is 'xtb'.")

    parser.add_argument('--etemp', type=float, default=300.,
                    help="Fermi smearing temperature of the electronic structure, in K (float). Default: 300. Ignored unless the back end is 'xtb'.")

    parser.add_argument('--no_eval_log', action='store_true',
                    help="Do not dump the structures whose energy was evaluated. Default: dump them.")

    parser.add_argument('--surrogate_xtb', action='store_true',
                    help="Explore: screen proposals on GFN2-xTB before the true energy. Meant for the 'orca' back end, whose single points cost hundreds of times a tight binding one. The residual second stage keeps the sampled distribution exact. Default: off.")

    parser.add_argument('--outdir', type=str, default=None,
                    help="Folder for the trajectory and energy dumps. Default: reaction/output/<reaction>.")

    parser.add_argument('--nworkers', type=int, default=4,
                    help="Search: worker processes; ORCA runs serially inside each. Default: 4.")

    parser.add_argument('--seeds_per_rung', type=int, default=4,
                    help="Search: MCTSS runs per rung of the tau ladder. Default: 4.")

    parser.add_argument('--explore_steps', type=int, default=5000,
                    help="Search: Monte-Carlo steps per MCTSS run. Default: 5000.")

    parser.add_argument('--max_chains', type=int, default=4,
                    help="Search: continuation chains grown per pair of states. Default: 4.")

    parser.add_argument('--screen', action='store_true',
                    help="Search on ORCA: screen every trial move on GFN2-xTB first (delayed acceptance; the sampled distribution stays the ORCA one). Default: off.")

    parser.add_argument('--crossing_votes', type=int, default=3,
                    help="Search: quenches per label when the separatrix crossing is bisected (S5). Default: 3.")

    parser.add_argument('--committor_iterations', type=int, default=None,
                    help="Search with S4: biased iterations of the committor. Default: 6.")

    parser.add_argument('--step_scale', type=float, default=1.,
                    help="Multiplier of the free-particle step width sqrt(hbar*tau/2m) (float). The proposal stays symmetric, so the sampled distribution is unchanged; values of 2 to 4 cut the energy evaluations of the SN2 benchmark by 2 to 4 times. Default: 1.")

    parser.add_argument('--tse', action='store_true',
                    help="Search: also run S6, which samples the transition state ensemble at T_phys by a pair held across the dividing surface through each crossing, every walker at 2 T_phys so that the weighted midpoint is at T_phys. Default: off.")

    parser.add_argument('--tse_steps', type=int, default=None,
                    help="S6: recorded sweeps of the pair. Default: 20000.")

    parser.add_argument('--tse_t12', type=float, default=None,
                    help="S6: tau12 of the pair in units of its bead tau; smaller is stiffer and keeps the midpoint weights near one. Default: 0.05.")

    parser.add_argument('--midpoint_stride', type=int, default=0,
                    help="Explore with two structures: every this many steps also pay for the energy at the midpoint of the pair and write it with its weight to midpoint_energy.log and midpoint.trj, the input of 'analyse_barrier.py --estimator midpoint'. The pair's midpoint represents tau_mid = 2 tau. Default: 0 (off).")

    args,unknown = parser.parse_known_args() # parse the arguments

    # saying so beats silently ignoring a flag meant to change the physics
    if len(unknown) > 0:
        print("warning: ignored unrecognised argument(s): %s" % (" ".join(unknown),))

    return args

#%% 2. Main
def main():

    """

    Set up the requested calculator and run the sampling.

    :return: None.

    """

    args = parse_args()

    if args.stages != "explore":
        return RunSearchMain(args)
    if args.calculator is None:
        args.calculator = "xtb"
    if args.t12 is None:
        args.t12 = 10.0

    # a charge and a multiplicity are properties of the system, so a reaction
    # may state them once in its meta.json; the command line still wins
    Meta = readMeta(args.file)
    Defaults = {"charge": 0, "mult": 1}
    for Key, Fallback in Defaults.items():
        if getattr(args, Key) is None:
            setattr(args, Key, int(Meta.get(Key, Fallback)))
            if Key in Meta:
                print("%s: %i, from the meta.json of %s" % (Key, getattr(args, Key), args.file))

    # the two back ends name their methods differently, so the default and the
    # validation both depend on which one is going to answer. Fail here, before
    # anything is read or written, rather than after the first geometry.
    if args.calculator == "xtb":
        args.method = normaliseMethod(args.method)
        if args.state > 0:
            raise ValueError(
                "The tight binding back end is ground state only. Ask for "
                "excited state %i on --calculator=orca." % (args.state,))
    elif args.calculator == "orca":
        if args.method is None:
            args.method = ORCA_DEFAULT_METHOD
        if args.basis is None:
            args.basis = ""
        if args.method.lower() == "casscf":
            args.method = ""
        else:
            args.nactel = None
            args.nactorb = None

    if not args.step_scale > 0.0:
        raise ValueError("--step_scale has to be positive, got %g." % (args.step_scale,))

    # step 1: read XYZ file
    Structures = readXYZFile(args.file, PrintStuff=True)
    # the midpoint of a pair gets a calculator of its own, so it never hands a
    # walker an SCF guess from another geometry
    NCalc = len(Structures) + int(len(Structures) == 2 and args.midpoint_stride > 0)

    # dumps go to a folder of their own per reaction, not to wherever the run started
    if args.outdir is None:
        args.outdir = os.path.join(OUTPUT_ROOT, reactionName(args.file))
    prepareOutputFolder(args.outdir, PrintStuff=True)
    print("output: %s" % (args.outdir,))
    if args.calculator == "emt":
        calc = getEMTCalculator()
    elif args.calculator == "amber":
        from treax_expedition.amber import getAmberCalculator
        calc = [getAmberCalculator() for n in range(0, NCalc)]
        print("method: AMBER ff99SB")
    elif args.calculator == "orca":
        LevelOfTheory=args.method+"  "+args.basis
        # one working directory per walker: ORCA restarts its SCF from the
        # orbitals left in the directory, and two walkers sharing one hand each
        # other a guess from the wrong geometry, which costs several times the
        # SCF iterations that the walker's own previous orbitals do
        calc = [getORCACalculator(LevelOfTheory=LevelOfTheory, charge=args.charge, mult=args.mult, nProcs=args.nprocs, PathToOrca=args.orcapath, numElectrons=args.nactel, numOrbital=args.nactorb, actState=args.state, Directory=os.path.join(args.outdir, "orca%i" % (n+1)))
                for n in range(0, NCalc)]
        print("method: ORCA %s, charge %+i, multiplicity %i, %i process(es)"
              % (LevelOfTheory.strip(), args.charge, args.mult, args.nprocs))
    else:
        # one calculator per walker, for the reason given for ORCA above: the
        # tight binding back end starts each SCF from its previous result
        calc = [getXTBCalculator(Method=args.method, charge=args.charge,
                                 mult=args.mult, nProcs=args.nprocs,
                                 Accuracy=args.accuracy,
                                 ElectronicTemperature=args.etemp)
                for n in range(0, NCalc)]
        print("method: %s, charge %+i, multiplicity %i, %i thread(s)"
              % (args.method, args.charge, args.mult, args.nprocs))
    # step 2: optional screen on a cheaper level of theory
    Surrogate = None
    if args.surrogate_xtb:
        from utils.surrogate import ForXTB
        Surrogate = ForXTB(charge=args.charge, mult=args.mult,
                           nProcs=args.nprocs)
        print("surrogate: GFN2-xTB screening the %s back end" % (args.calculator,))

    if args.step_scale != 1.0:
        print("step width: %.2f times the free-particle width" % (args.step_scale,))

    if len(Structures) == 1:
        Structure = Structures[0]
        Structure = setCalculatorToStructure(Structure, calc[0] if isinstance(calc, list) else calc)
        NTrue = doSingleStructureSWSMC(Structure=Structure, tau=args.tau, Nsteps=args.steps, Neq=args.eq, SingleAtom=args.single_atom,
                                       Seed=args.seed, LogEvaluations=not args.no_eval_log,
                                       Surrogate=Surrogate,
                                       FolderToSave=args.outdir, StepScale=args.step_scale)
    else:
        # a run that stops when the walkers meet has no equilibration phase
        if args.eq > 0:
            print("warning: --eq is ignored when two structures are sampled")
        for n in range(0,2):
            Structures[n] = setCalculatorToStructure(Structures[n], calc[n] if isinstance(calc, list) else calc)
        NTrue = doTwoStructureSWSMC(Structures=Structures, tau=args.tau, tau12=args.tau*args.t12,  Nsteps=args.steps, ScaleThreshold = args.threshold, SingleAtom=args.single_atom,
                                    Seed=args.seed, LogEvaluations=not args.no_eval_log,
                                    Surrogate=Surrogate,
                                    FolderToSave=args.outdir, StepScale=args.step_scale,
                                    MidpointCalculator=((calc[2] if isinstance(calc, list) else calc)
                                                        if args.midpoint_stride > 0 else None),
                                    MidpointStride=max(1, args.midpoint_stride))

    # the cost of a transition state search is quoted in energy evaluations,
    # so it is reported here rather than left to be counted from the dumps
    print("\ntrue energy evaluations spent: %i" % (NTrue,))
    if Surrogate is not None:
        print("surrogate: %i screening evaluations, %i failed and rejected"
              % (Surrogate.NCalls, Surrogate.NFailed))

    return None


#%% 3. Staged search
def RunSearchMain(args):

    """

    Run the staged search on a reaction and report its certified bounds.

    :param1 args: parsed command line.

    :return: None.

    """

    from dataclasses import replace

    from treax_expedition.pipeline import GradeMolecule, MoleculeSpec
    from utils.path.search import RunSearch

    OrcaPath = args.orcapath or os.environ.get("TREAX_ORCA")
    Outdir = args.outdir or os.path.join(OUTPUT_ROOT, reactionName(args.file), "search")
    Spec = MoleculeSpec(args.file, Calculator=args.calculator, Level=args.method,
                        Basis=args.basis or "", Charge=args.charge, Mult=args.mult, OrcaPath=OrcaPath,
                        NProcs=args.nprocs, Screen=args.screen, WorkDir=Outdir,
                        T12=3.0 if args.t12 is None else args.t12,
                        Threshold=args.threshold, StepScale=args.step_scale)
    print("[Setup] %s: %s%s, charge %+i, multiplicity %i, T_phys tau* %.2f fs, %i worker(s)"
          % (Spec.Name, Spec.Calculator, " " + Spec.Level if Spec.Calculator == "orca" else "",
             Spec.Charge, Spec.Mult, Spec.TauPhys, args.nworkers))
    Seed = args.seed if args.seed is not None else 1
    Settings = replace(Spec.DefaultSettings(), Seed=Seed,
                       NSeeds=args.seeds_per_rung, ExploreSteps=args.explore_steps,
                       MaxChains=args.max_chains, CrossingVotes=args.crossing_votes,
                       Committor=args.stages == "search+committor", TSE=args.tse,
                       TSEStage=replace(Spec.DefaultSettings().TSEStage,
                                        **({} if args.tse_steps is None else {"Steps": args.tse_steps}),
                                        **({} if args.tse_t12 is None else {"T12": args.tse_t12})),
                       CommittorStage=replace(Spec.DefaultCommittorSettings(), Seed=Seed,
                                              **({} if args.committor_iterations is None
                                                 else {"Iterations": args.committor_iterations})))
    Report = RunSearch(Spec, Settings, Folder=Outdir, NWorkers=args.nworkers)
    Lines = GradeMolecule(Spec, Report)
    for Line in Lines:
        print(Line)
    with open(os.path.join(Outdir, "grade.txt"), "w") as f:
        f.write("\n".join(Lines) + "\n")

    return None

if __name__ == '__main__':
    main()
