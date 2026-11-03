#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

The committor stage of the search (S4): the committor of one pair of states
at the physical tau, learned self-consistently as in committor_2.0 (Ritz loss,
Kolmogorov bias plus a well-tempered bias along its latent, blocks pooled by
MBAR) and validated as in AIMMD (shooting points picked around q = 1/2,
binomial likelihood of their outcomes in the loss and as the test score).
Every chain, biased walk and shot, runs on the MCTSS kernel with energies only.

Created on: Mon Sep 28 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import os
import random
from dataclasses import dataclass

import numpy as np

from utils.mc.walk import Walk

from .bias import KolmogorovBias
from .data import LabelledStore
from .model import CommittorNN, CommittorOf, SaveCommittor
from .shots import CoreSets, SelectShootingPoints, Shoot
from .train import CommittorDataset, Fit
from .zbias import AdaptiveZBias, SumOfBiases

#%% 1. Settings
@dataclass
class CommittorSettings:

    """

    :param1 Iterations:        biased iterations after the first fit.
    :param2 BasinSteps:        unbiased steps from each minimum.
    :param3 ExploreSteps:      steps per walker growing the latent bias.
    :param4 Steps:             steps per walker with the bias frozen.
    :param5 Pace:              deposit every this many exploration steps.
    :param6 Stride:            keep every this many configurations.
    :param7 Lam:               prefactor of the Kolmogorov bias.
    :param8 ZBias:             add the well-tempered bias along z.
    :param9 Epochs:            training epochs per fit.
    :param10 BatchSize:        minibatch of the Ritz term.
    :param11 LearningRate:     Adam learning rate.
    :param12 Alpha:            weight of the boundary condition.
    :param13 Gamma:            weight of the binomial likelihood of the shots in
                               the fit, relative to the Ritz term. Zero keeps the
                               shots for picking points and for validation only,
                               which on the 2D references fitted better (two-path
                               upper channel q(S1) 0.36 vs 0.91 at weight one).
    :param14 ShotsPerIteration: shooting points per iteration.
    :param15 ShotsPerPoint:    chains per shooting point.
    :param16 ShotMaxSteps:     budget of one chain.
    :param17 CoreFraction:     share of the way between the minima that counts
                               as a state's core.
    :param18 ValidationShots:  chains from each crossing point for pB.
    :param19 Seed:             base seed.

    """

    Iterations: int = 4
    BasinSteps: int = 4000
    ExploreSteps: int = 1500
    Steps: int = 6000
    Pace: int = 20
    Stride: int = 5
    Lam: float = 1.0
    ZBias: bool = True
    Epochs: int = 300
    BatchSize: int = 256
    LearningRate: float = 1.0e-3
    Alpha: float = 1.0
    Gamma: float = 0.0
    ShotsPerIteration: int = 16
    ShotsPerPoint: int = 2
    ShotMaxSteps: int = 2000
    CoreFraction: float = 0.2
    ValidationShots: int = 20
    Seed: int = 1

#%% 2. Bias of one iteration
def BuildIterationBias(Model, Masses, tau, X, Lam=1.0, ZBias=True):

    """

    Kolmogorov bias of the current model, plus an adaptive bias along its latent.

    The Kolmogorov bias pulls the walkers onto the transition region the model
    currently predicts. The adaptive term along z helps them cross towards the
    other basin, which the Kolmogorov bias alone does not favour; its grid is
    sized from the data, since every refit may stretch z.

    :param1 Model:  the current CommittorNN.
    :param2 Masses: per coordinate masses.
    :param3 tau:    time parameter tau (float), in fs.
    :param4 X:      configurations used to size the grid of the adaptive term.
    :param5 Lam:    prefactor of the Kolmogorov bias.
    :param6 ZBias:  add the adaptive term along z (bool).

    :return: Bias: a SumOfBiases whose Deposit and Freeze reach the adaptive term.

    """

    Terms = [KolmogorovBias(Model, Masses, tau, Lam=Lam)]
    if ZBias:
        Terms.append(AdaptiveZBias.FromSample(Model, tau, X))

    return SumOfBiases(Terms)

#%% 3. Worker tasks
def TaskWalk(X0, tau, NSteps, Bias, Stride, Seed):

    """

    One walk in the worker, with a frozen bias or none.

    :return: X, V, Last, NCalls: kept configurations, their bias values, the
             last configuration and the energy calls spent.

    """

    from utils.path.search import _WORKER

    random.seed(Seed)
    System = _WORKER["System"]
    Calls0 = System.NCalls
    R = Walk(System, X0, tau, NSteps, Bias=Bias, Stride=Stride)

    return R["X"], R["V"], R["Last"], System.NCalls - Calls0

def TaskShots(X, tau, Known, States, Coordinate, Fraction, MaxSteps, NShots, Seed):

    """

    Independent unbiased chains from one shooting point, in the worker.

    :return: NA, NB, NUndecided, NCalls.

    """

    from utils.path.search import _WORKER, LocalBook

    random.seed(Seed)
    System = _WORKER["System"]
    Calls0 = System.NCalls
    Cores = CoreSets(LocalBook(_WORKER["Spec"], Known), States, Coordinate, Fraction)
    Count = {States[0]: 0, States[1]: 0, None: 0}
    for _ in range(0, NShots):
        Outcome, _ = Shoot(System, X, tau, Cores, MaxSteps)
        Count[Outcome] += 1

    return Count[States[0]], Count[States[1]], Count[None], System.NCalls - Calls0

#%% 4. Diagnostics
def ShotScores(Model, X, NA, NB):

    """

    How well a model predicts shots it was not trained on.

    :return: NLL, TPRatio: the binomial negative log likelihood per chain and
             the observed over the expected share of split pairs, i.e. of
             points whose two chains reached different states (AIMMD's
             transition-path ratio).

    """

    if len(X) == 0:
        return np.nan, np.nan
    q = np.clip(CommittorOf(Model, X), 1.0e-12, 1.0 - 1.0e-12)
    n = NA + NB
    NLL = -np.sum(NB*np.log(q) + NA*np.log(1.0 - q))/max(1.0, np.sum(n))
    Pairs = n >= 2
    if not np.any(Pairs):
        return float(NLL), np.nan
    Observed = np.mean((NA[Pairs] > 0) & (NB[Pairs] > 0))
    Expected = np.mean(2.0*q[Pairs]*(1.0 - q[Pairs]))

    return float(NLL), float(Observed/Expected) if Expected > 0 else np.nan

#%% 5. The stage
def RunCommittorStage(Spec, System, Book, States, Coordinate, Starts, Candidates, Probes, tau,
                      Workers, Settings=None, Folder=".", Log=print):

    """

    Learn and validate the committor between two states.

    :param1 Spec:       system spec, for the descriptors.
    :param2 System:     the main process's system.
    :param3 Book:       StateBook of the search.
    :param4 States:     names of the first and second state.
    :param5 Coordinate: progress coordinate of the pair, for the state cores.
    :param6 Starts:     configurations the biased walkers start from besides
                        the two minima, e.g. the tops of the chains.
    :param7 Candidates: configurations shooting points may be picked from
                        besides the sampled ones, e.g. the chain nodes.
    :param8 Probes:     configurations whose pB is measured at the end, e.g.
                        the separatrix crossings of the chains.
    :param9 tau:        the physical tau, in fs.
    :param10 Workers:   Pool of the search.
    :param11 Settings:  CommittorSettings, or None for the defaults.
    :param12 Folder:    output folder.
    :param13 Log:       callable printing one line.

    :return: Result: dict with the model, the history, q and pB at the probes.

    """

    import torch
    torch.set_default_dtype(torch.float64)
    Settings = CommittorSettings() if Settings is None else Settings
    torch.manual_seed(Settings.Seed)
    os.makedirs(Folder, exist_ok=True)
    Rng = np.random.default_rng(Settings.Seed)
    Masses = np.asarray(System.Masses, dtype=float)
    Sa, Sb = States
    XA = Book.Get(Sa)["X"]
    XB = Book.Get(Sb)["X"]
    Cores = CoreSets(Book, States, Coordinate, Settings.CoreFraction)
    Known = [dict(s) for s in Book.States]
    Calls = 0

    def Seed():
        return int(Rng.integers(0, 2**31 - 1))

    # the two basins, unbiased; their cores are the boundary condition
    Out = Workers.Map(TaskWalk, [(X, tau, Settings.BasinSteps, None, Settings.Stride, Seed())
                                 for X in (XA, XB)])
    Calls += sum(o[3] for o in Out)
    Basin = np.concatenate([o[0] for o in Out])
    Names = [Cores(x) for x in Basin]
    InA = np.array([n == Sa for n in Names])
    InB = np.array([n == Sb for n in Names])
    Log("[S4] %s->%s tau* %.2f fs: basin walks %i frames, %i in the core of %s, %i in %s"
        % (Sa, Sb, tau, len(Basin), InA.sum(), Sa, InB.sum(), Sb))
    if InA.sum() == 0 or InB.sum() == 0:
        raise RuntimeError("A basin walk never stayed in its core; lower CoreFraction.")
    Store = LabelledStore(Basin[InA], Basin[InB], tau)
    Store.AddBlock(Basin)
    # walker of every stored frame: 0 and 1 from the minima, then the starts
    Walker = [np.repeat([0, 1], [len(o[0]) for o in Out])]

    Model = CommittorNN(Spec.Descriptors())
    FitKwargs = dict(Alpha=Settings.Alpha, Epochs=Settings.Epochs, BatchSize=Settings.BatchSize,
                     LearningRate=Settings.LearningRate, Seed=Settings.Seed, Gamma=Settings.Gamma)

    def Refit():
        Data = CommittorDataset(Store.X, Store.Weights(), Store.XA, Store.XB, Shots=Store.ShotArrays())
        Fit(Model, Data, Masses, **FitKwargs)

    Refit()
    Walkers = [XA, XB] + [np.asarray(s, dtype=float) for s in Starts]
    Pool = np.atleast_2d(np.asarray(Candidates, dtype=float)) if len(Candidates) > 0 else None
    History = []
    for k in range(1, Settings.Iterations + 1):
        Bias = BuildIterationBias(Model, Masses, tau, Store.X, Lam=Settings.Lam, ZBias=Settings.ZBias)
        # the latent bias grows from every walker in turn, then is frozen
        for i, X in enumerate(Walkers):
            Calls0 = System.NCalls
            R = Walk(System, X, tau, Settings.ExploreSteps, Bias=Bias, Pace=Settings.Pace,
                     Stride=Settings.ExploreSteps + 1)
            Walkers[i] = R["Last"]
            Calls += System.NCalls - Calls0
        Bias.Freeze()
        Out = Workers.Map(TaskWalk, [(X, tau, Settings.Steps, Bias, Settings.Stride, Seed())
                                     for X in Walkers])
        Calls += sum(o[3] for o in Out)
        Block = np.concatenate([o[0] for o in Out])
        Walkers = [o[2] for o in Out]
        Store.AddBlock(Block, Bias)
        Walker.append(np.repeat(np.arange(len(Out)), [len(o[0]) for o in Out]))

        # shots from around q = 1/2 of the current model, scored before it sees them
        Cand = Block if Pool is None else np.concatenate([Pool, Block])
        Chosen = SelectShootingPoints(lambda X: _Latent(Model, X), Cand, Settings.ShotsPerIteration, Rng=Rng)
        Out = Workers.Map(TaskShots, [(Cand[i], tau, Known, States, Coordinate, Settings.CoreFraction,
                                       Settings.ShotMaxSteps, Settings.ShotsPerPoint, Seed())
                                      for i in Chosen])
        Calls += sum(o[3] for o in Out)
        NA = np.array([o[0] for o in Out], dtype=float)
        NB = np.array([o[1] for o in Out], dtype=float)
        NLL, TP = ShotScores(Model, Cand[Chosen], NA, NB)
        Store.AddShots(Cand[Chosen], NA, NB)
        Refit()
        Row = {"Iteration": k, "NData": len(Store.X), "NShots": len(Store.Shots),
               "TestNLL": NLL, "TPRatio": TP, "Undecided": int(sum(o[2] for o in Out)),
               "qA": float(np.mean(CommittorOf(Model, Store.XA))),
               "qB": float(np.mean(CommittorOf(Model, Store.XB))), "Calls": Calls}
        History.append(Row)
        SaveCommittor(Model, os.path.join(Folder, "committor_iter%02i.pt" % (k,)))
        Log("[S4] iter %i: %6i frames, %3i shots, predictive NLL %.3f (0.693 = coin), TP obs/exp %.2f, "
            "<q>A %.3f <q>B %.3f, %i energies"
            % (k, Row["NData"], Row["NShots"], NLL, TP, Row["qA"], Row["qB"], Calls))
    SaveCommittor(Model, os.path.join(Folder, "committor.pt"))
    # the pooled sample, for the figures; a free energy needs the MBAR weights uncapped
    ShotX, ShotNA, ShotNB = Store.ShotArrays()
    np.savez(os.path.join(Folder, "data.npz"), X=Store.X, Weights=Store.Weights(Quantile=1.0),
             Block=np.repeat(np.arange(len(Store.Blocks)), [len(b) for b in Store.Blocks]),
             Walker=np.concatenate(Walker), XA=Store.XA, XB=Store.XB, ShotX=ShotX, ShotNA=ShotNA,
             ShotNB=ShotNB, Tau=tau)

    # pB of the probes against the model's q
    Out = Workers.Map(TaskShots, [(X, tau, Known, States, Coordinate, Settings.CoreFraction,
                                   Settings.ShotMaxSteps, Settings.ValidationShots, Seed())
                                  for X in Probes])
    Calls += sum(o[3] for o in Out)
    Validation = []
    for X, o in zip(Probes, Out):
        n = o[0] + o[1]
        pB = o[1]/n if n > 0 else np.nan
        Validation.append({"q": float(CommittorOf(Model, np.atleast_2d(X))[0]), "pB": pB,
                           "pBErr": float(np.sqrt(pB*(1.0 - pB)/n)) if n > 0 else np.nan,
                           "NA": o[0], "NB": o[1], "Undecided": o[2]})
        Log("[S4] probe: q = %.3f, pB = %.3f +- %.3f (%i/%i chains, %i undecided)"
            % (Validation[-1]["q"], pB, Validation[-1]["pBErr"], o[1], n, o[2]))
    with open(os.path.join(Folder, "history.dat"), "w") as f:
        Keys = ["Iteration", "NData", "NShots", "TestNLL", "TPRatio", "Undecided", "qA", "qB", "Calls"]
        f.write("# " + " ".join("%s[$%i]" % (c, i + 1) for i, c in enumerate(Keys)) + "\n")
        for Row in History:
            f.write(" ".join("%12.6g" % Row[c] for c in Keys) + "\n")

    return {"Model": Model, "History": History, "Validation": Validation, "Calls": Calls}

def _Latent(Model, X):

    """

    :return: z: the latent of a batch.

    """

    import torch

    with torch.no_grad():
        t = torch.as_tensor(np.ascontiguousarray(np.atleast_2d(X)), dtype=next(Model.parameters()).dtype)
        return Model(t).numpy()
