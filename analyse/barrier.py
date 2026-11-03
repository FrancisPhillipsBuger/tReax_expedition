#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Barrier estimators for a sampled MCTSS run and their extrapolation to zero
effective temperature, following the ESI of the MCTSS paper: the barriers,
their errors and the transition state structure all come from the sampled
structures themselves, with no saddle optimisation.

A run is binned along a one-dimensional reaction coordinate and reduced to a
profile. Two sample sets can be binned. The walkers' own samples are the ESI's
estimator; they belong to no single temperature, since each walker samples
exp(-tau V/hbar) while the pair is apart and the coupled pair samples its
midpoint at tau_mid = 2 tau once it is together, and the approach to the
meeting is not an equilibrium sample at all. The weighted midpoints of a run
dumped with a midpoint calculator are an exact estimator of exp(-tau_mid V/hbar)
for the stationary part of the chain. Either way the bin mean is biased
upwards by roughly (d-1)/2 kT; extrapolating linearly in 1/tau of the
ensemble removes the leading part of that drift.

Created on: Tue Aug 25 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import os

import ase.io as asio
import numpy as np

from treax_expedition.geometry import NeedsFrameRemoval, alignedCoordinates
from utils.mc.ensemble import PairEnsemble

#%% 1. Run loading
def TauFromTimeColumn(FileName):

    """

    Recover the time step of a run from the first column of an energy log.

    The samplers write i*tau in that column, so the spacing between two
    consecutive rows is tau itself. This is more robust than parsing the
    folder name, which differs between the HF-3c and the DLPNO trees.

    :param1 FileName: path of a ``sampled_energy*.log`` file.

    :return: tau: time parameter tau (float), in fs.

    """

    t = np.atleast_1d(np.loadtxt(FileName, usecols=0))
    if len(t) < 2:
        raise RuntimeError("Cannot recover tau from "+FileName)
    dt = np.diff(t)

    return float(np.median(dt))

def walkerRows(ICPath, EnergyPath):

    """

    Reaction coordinate and energies of one walker, cut to their common rows.

    :param1 ICPath:     path of the walker's reaction coordinate file.
    :param2 EnergyPath: path of the walker's energy log.

    :return: ic, en: the two columns, of equal length.

    """

    # usecols keeps the reader working on logs whose row length varies,
    # which older trajectories on disk do
    ic = np.atleast_1d(np.loadtxt(ICPath, usecols=1))
    en = np.atleast_1d(np.loadtxt(EnergyPath, usecols=1))
    # the walkers stop on different steps, so keep the common prefix
    n = min(len(ic), len(en))

    return ic[:n], en[:n]

def LoadRun(Folder, ICFiles=("ic1.trj", "ic2.trj"),
            EnergyFiles=("sampled_energy1.log", "sampled_energy2.log")):

    """

    Load the reaction coordinate and the energies of one MCTSS run.

    The two walkers are concatenated, since both sample the same coupled
    target and the profile is built from all sampled points.

    :param1 Folder:      folder holding the trajectory dumps of one run.
    :param2 ICFiles:     names of the per-walker reaction coordinate files.
    :param3 EnergyFiles: names of the per-walker energy logs.

    :return: IC, En, tau: reaction coordinate, energies in eV and the time
             parameter tau in fs.

    """

    IC = []
    En = []
    tau = None
    for icf, enf in zip(ICFiles, EnergyFiles):
        icp = os.path.join(Folder, icf)
        enp = os.path.join(Folder, enf)
        if not (os.path.exists(icp) and os.path.exists(enp)):
            continue
        ic, en = walkerRows(icp, enp)
        if tau is None:
            tau = TauFromTimeColumn(enp)
        IC.append(ic)
        En.append(en)

    if len(IC) == 0:
        raise RuntimeError("No usable walker data in "+Folder)

    return np.concatenate(IC), np.concatenate(En), tau

def LoadRunStructures(Folder, ICFiles=("ic1.trj", "ic2.trj"),
                      EnergyFiles=("sampled_energy1.log", "sampled_energy2.log"),
                      TrjFiles=("sampled1.trj", "sampled2.trj")):

    """

    Load the sampled structures of one run, row for row with LoadRun.

    :param1 Folder:      folder holding the trajectory dumps of one run.
    :param2 ICFiles:     names of the per-walker reaction coordinate files.
    :param3 EnergyFiles: names of the per-walker energy logs.
    :param4 TrjFiles:    names of the per-walker sampled trajectories.

    :return: Frames: list of ase.Atom instances, the i-th one belonging to the
             i-th point that LoadRun returns.

    """

    Frames = []
    for icf, enf, trf in zip(ICFiles, EnergyFiles, TrjFiles):
        icp = os.path.join(Folder, icf)
        enp = os.path.join(Folder, enf)
        if not (os.path.exists(icp) and os.path.exists(enp)):
            continue
        n = len(walkerRows(icp, enp)[0])
        trp = os.path.join(Folder, trf)
        if not os.path.exists(trp):
            raise RuntimeError("%s is missing, so the structures of %s cannot be averaged" % (trf, Folder))
        Trj = asio.read(trp, index=":", format="xyz")
        if len(Trj) < n:
            raise RuntimeError("%s holds %i frames for %i energies" % (trp, len(Trj), n))
        Frames.extend(Trj[:n])

    if len(Frames) == 0:
        raise RuntimeError("No usable walker data in "+Folder)

    return Frames

def LoadMidpoints(Folder, FileName="midpoint_energy.log"):

    """

    Load the weighted midpoints of one run dumped with a midpoint calculator.

    :param1 Folder:   folder holding the trajectory dumps of one run.
    :param2 FileName: name of the midpoint log.

    :return: IC, VM, LogW, Ensemble: progress of the midpoints, their energies
             in eV, their log-weights onto exp(-tau_mid V/hbar) and the
             PairEnsemble of the run.

    """

    Path = os.path.join(Folder, FileName)
    with open(Path) as f:
        Words = f.readline().replace("=", " ").split()
    try:
        Ensemble = PairEnsemble(float(Words[Words.index("tau_bead,fs") + 1]),
                                float(Words[Words.index("tau12,fs") + 1]))
    except (ValueError, IndexError):
        raise RuntimeError("No tau_bead and tau12 in the header of "+Path)
    Data = np.atleast_2d(np.loadtxt(Path))
    if Data.shape[1] < 5 or len(Data) == 0:
        raise RuntimeError("No midpoints in "+Path)

    return Data[:, 1], Data[:, 2], Ensemble.LogWeight(Data[:, 2], Data[:, 3], Data[:, 4]), Ensemble

#%% 2. Binned profile
def binIndex(IC, N, Grid="tile"):

    """

    Equal-width bins along the reaction coordinate.

    With Grid 'tile' the N bins tile [min, max] exactly. With 'esi' they are
    those of the ESI's process.sh, of width (max - min)/(N - 1) from min, so
    the last bin holds only the points at the maximum, the unmoved initial
    endpoint; this reproduces the published barriers.

    :param1 IC:   reaction coordinate of every sampled point.
    :param2 N:    number of bins.
    :param3 Grid: 'tile' or 'esi'.

    :return: Centres, Index: the N bin centres and the bin of every point, -1
             for none, or None for the index when the coordinate never moved.

    """

    if Grid not in ("tile", "esi"):
        raise ValueError("Grid has to be 'tile' or 'esi', got %r." % (Grid,))
    lo = IC.min()
    hi = IC.max()
    dx = (hi - lo)/(N - 1 if Grid == "esi" else N)
    Centres = lo + dx*(np.arange(N) + 0.5)
    if not dx > 0.0:
        return Centres, None

    if Grid == "tile":
        # the right edge belongs to the last bin rather than to a bin of its own
        return Centres, np.minimum(np.floor((IC - lo)/dx).astype(int), N - 1)

    # the ESI's own arithmetic, so points on a bin edge land where they did there
    Centres = dx*np.arange(0, N, 1) + lo + 0.5*dx
    Index = np.full(len(IC), -1)
    for i, x in enumerate(Centres):
        Index[(IC >= (x - 0.5*dx)) & (IC < (x + 0.5*dx))] = i

    return Centres, Index

def BinnedProfile(IC, En, N=15, Grid="tile", Weights=None):

    """

    Bin the sampled points along the reaction coordinate.

    :param1 IC:      reaction coordinate of every sampled point.
    :param2 En:      energy of every sampled point, in eV.
    :param3 N:       number of bins.
    :param4 Grid:    'tile' or 'esi', see binIndex.
    :param5 Weights: importance weight of every point, or None for equal ones;
                     the means and deviations are then weighted.

    With the default grid the N bins have equal width and tile [min, max]
    exactly, the largest value falling in the last bin. Every sample lands in
    exactly one bin, so the two end bins, which serve as the reactant and
    product references, are averages over the tails rather than single points.

    :return: Centres, Mean, Std, Minimum, Counts: bin centres and the mean,
             standard deviation, minimum and population of each occupied bin.

    """

    Centres, Index = binIndex(IC, N, Grid)
    Mean = np.full(N, np.nan)
    Std = np.full(N, np.nan)
    Minimum = np.full(N, np.nan)
    Counts = np.zeros(N, dtype=int)
    if Index is None:
        # a coordinate that never moved has no profile to speak of
        return Centres[:0], Mean[:0], Std[:0], Minimum[:0], Counts[:0]

    for i in range(0, N):
        m = Index == i
        Counts[i] = int(m.sum())
        if Counts[i] == 0:
            continue
        Minimum[i] = En[m].min()
        if Weights is None:
            Mean[i] = En[m].mean()
            Std[i] = En[m].std()
        elif np.sum(Weights[m]) > 0.0:
            w = Weights[m]/np.sum(Weights[m])
            Mean[i] = float(np.sum(w*En[m]))
            Std[i] = float(np.sqrt(np.sum(w*(En[m] - Mean[i])**2)))

    occ = Counts > 0

    return Centres[occ], Mean[occ], Std[occ], Minimum[occ], Counts[occ]

def BinnedStructures(IC, Frames, N=15, Grid="tile"):

    """

    Average structure of every occupied bin, the ESI's averaged.trj.

    Each structure of a bin is overlaid on the running average of the ones
    before it, mass-weighted, and added to it, as the ESI's process.sh does.
    A structure pinned by its cell or constraints is averaged as it is.

    :param1 IC:     reaction coordinate of every sampled point.
    :param2 Frames: ase.Atom instance of every sampled point, see LoadRunStructures.
    :param3 N:      number of bins.
    :param4 Grid:   'tile' or 'esi', see binIndex.

    :return: Averages: one ase.Atom instance per occupied bin, in the order of
             the arrays BinnedProfile returns.

    """

    _, Index = binIndex(IC, N, Grid)
    if Index is None:
        return []

    Masses = np.asarray(Frames[0].get_masses(), dtype=float)
    Align = NeedsFrameRemoval(Frames[0])
    Averages = []
    for i in range(0, N):
        Members = np.flatnonzero(Index == i)
        if len(Members) == 0:
            continue
        Average = Frames[Members[0]].get_positions()
        if Align:
            Average = Average - (Masses[:, None]*Average).sum(axis=0)/Masses.sum()
        for k, j in enumerate(Members[1:], start=1):
            X = alignedCoordinates(Frames[j].get_positions(), Average, Masses, Align)
            Average = (k*Average + X)/(k + 1)
        Structure = Frames[Members[0]].copy()
        Structure.calc = None
        Structure.set_positions(Average)
        Averages.append(Structure)

    return Averages

#%% 3. Barriers
def BarriersFromProfile(Profile):

    """

    Forward and backward barrier of a binned profile, in eV.

    The reaction coordinate used for the SN2 benchmark is r(C-Cl) - r(C-Br),
    so the reactant sits in the last bin and the product in the first one.
    This is the convention of the reference ``calc_barriers.sh``.

    :param1 Profile: per-bin energies, in eV.

    :return: dEf, dEb: forward and backward barrier, in eV.

    """

    p = np.asarray(Profile, dtype=float)
    if np.all(np.isnan(p)):
        return np.nan, np.nan
    i = int(np.nanargmax(p))

    return p[i] - p[-1], p[i] - p[0]

def SpreadErrors(Mean, Std):

    """

    Errors of the two barriers from the energy spread in the bins, as in the ESI.

    The ESI's calc_barriers.sh adds the standard deviations of the energies in
    the top bin and in the end bin in quadrature. It measures how widely the
    hot ensemble scatters, not how well the mean is known; for the latter see
    BootstrapBarriers.

    :param1 Mean: per-bin mean energies, in eV, the profile of BarriersFromProfile.
    :param2 Std:  per-bin standard deviations of the energies, in eV.

    :return: dEfErr, dEbErr: errors of the forward and backward barrier, in eV.

    """

    p = np.asarray(Mean, dtype=float)
    s = np.asarray(Std, dtype=float)
    if np.all(np.isnan(p)):
        return np.nan, np.nan
    i = int(np.nanargmax(p))

    return float(np.hypot(s[i], s[-1])), float(np.hypot(s[i], s[0]))

def BootstrapBarriers(IC, En, N=15, Estimator="mean", NBoot=200, Seed=0, Grid="tile", Weights=None):

    """

    Bootstrap the barriers of one run by resampling the sampled points.

    :param1 IC:        reaction coordinate of every sampled point.
    :param2 En:        energy of every sampled point, in eV.
    :param3 N:         number of bins.
    :param4 Estimator: ``mean`` or ``min``, the per-bin reduction to use.
    :param5 NBoot:     number of bootstrap replicas.
    :param6 Seed:      seed of the bootstrap generator.
    :param7 Grid:      'tile' or 'esi', see binIndex.
    :param8 Weights:   importance weight of every point, or None.

    :return: dEf, dEfErr, dEb, dEbErr: barriers and their bootstrap standard
             deviations, in eV.

    """

    idx = {"mean": 1, "min": 3}[Estimator]
    Ref = BarriersFromProfile(BinnedProfile(IC, En, N=N, Grid=Grid, Weights=Weights)[idx])

    rng = np.random.default_rng(Seed)
    n = len(IC)
    f = []
    b = []
    for _ in range(NBoot):
        k = rng.integers(0, n, size=n)
        try:
            df, db = BarriersFromProfile(BinnedProfile(IC[k], En[k], N=N, Grid=Grid,
                                                       Weights=None if Weights is None else Weights[k])[idx])
        except (ValueError, IndexError):
            continue
        if np.isfinite(df) and np.isfinite(db):
            f.append(df)
            b.append(db)

    fErr = float(np.std(f)) if len(f) > 1 else np.nan
    bErr = float(np.std(b)) if len(b) > 1 else np.nan

    return Ref[0], fErr, Ref[1], bErr

#%% 4. Effective-temperature analysis
def ExtrapolateToZeroKT(Taus, Values, Errors=None):

    """

    Extrapolate an estimator linearly in 1/tau to the zero temperature limit.

    Since kB T = hbar/tau for the ensemble the estimator describes, a bias
    proportional to the temperature is linear in 1/tau and its intercept is the
    zero temperature value. Scaling every tau by one factor, as between the
    bead and the midpoint tau of a pair, changes the slope and not the intercept.

    :param1 Taus:   tau of the ensemble of each run, in fs.
    :param2 Values: estimator of each run.
    :param3 Errors: optional per-run uncertainties used as inverse weights.

    :return: Intercept, Slope, R2, InterceptErr: fitted intercept, slope,
             coefficient of determination and standard error of the intercept.

    """

    x = 1.0/np.asarray(Taus, dtype=float)
    y = np.asarray(Values, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if len(x) < 3:
        return np.nan, np.nan, np.nan, np.nan

    if Errors is None:
        w = np.ones_like(x)
    else:
        e = np.asarray(Errors, dtype=float)[good]
        w = np.where(np.isfinite(e) & (e > 0), 1.0/np.maximum(e, 1e-12), 1.0)

    A = np.vstack([x, np.ones_like(x)]).T
    Aw = A * w[:, None]
    yw = y * w
    coef, _, _, _ = np.linalg.lstsq(Aw, yw, rcond=None)
    Slope, Intercept = float(coef[0]), float(coef[1])

    # the statistics use the same weights as the fit, so that R2 and the
    # error describe the line actually fitted; with unit weights they reduce
    # to the ordinary least squares expressions
    pred = A @ coef
    w2 = w*w
    yMean = float(np.sum(w2*y)/np.sum(w2))
    ssRes = float(np.sum(w2*(y - pred)**2))
    ssTot = float(np.sum(w2*(y - yMean)**2))
    R2 = 1.0 - ssRes/ssTot if ssTot > 0 else np.nan

    # standard error of the intercept from the residual variance
    dof = len(x) - 2
    InterceptErr = np.nan
    if dof > 0:
        s2 = ssRes/dof
        cov = s2 * np.linalg.inv(Aw.T @ Aw)
        InterceptErr = float(np.sqrt(cov[1, 1]))

    return Intercept, Slope, R2, InterceptErr
