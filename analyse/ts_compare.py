#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Grading of finished staged searches against reference transition states:
RMSD (in angstroms) and energy difference (in eV) of every located TS (the
top node of each channel and its q = 1/2 crossing) against ts_guess, the
reference saddle of reference_saddle.py when present, or the exact saddles
of a 2D model. Writes numbers and structures only; plot_ts_compare.py draws
the figures from them, so the figures can be redone without this step.

Created on: Thu Oct 01 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import argparse
import csv
import json
import os
import re
import sys
import time

import ase.io as asio
import numpy as np

# the repository root, one level above this folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyse.ts_metrics import AngleDifference, Dihedral, IsChiral, RMSD, SymmetryRMSD
from treax_expedition.io_xyz import INPUT_ROOT, OUTPUT_ROOT

ANALYTIC = ("2d", "muller", "two_paths")
DEFAULT_OUTDIR = os.path.join(OUTPUT_ROOT, "ts_compare")
SADDLE_ROOT = os.path.join(OUTPUT_ROOT, "reference_saddle")

# columns of ts_compare.csv, one row per located TS and reference
COLUMNS = ["Class", "System", "Run", "Pair", "Channel", "OnRoute", "Primary", "Candidate",
           "Reference", "RefLabel", "Backend", "NAtoms", "Xi", "RMSD", "RMSD_mw", "RMSD_sym",
           "RMSD_heavy", "RMSD_backbone", "Permuted", "Mirrored", "E_eV", "ERef_eV", "dE_eV",
           "EUp_eV", "EUpMinusRef_eV", "Certified", "X", "Y", "XRef", "YRef", "Phi", "Psi",
           "PhiRef", "PsiRef", "dPhi", "dPsi", "Energies", "WallSeconds", "Structure", "Folder"]

# columns of chains.csv, one row per chain node of every channel
CHAIN_COLUMNS = ["Class", "System", "Run", "Pair", "Channel", "Node", "Xi", "E_eV", "ERelA_eV",
                 "IsTop", "X", "Y", "Phi", "Psi"]

#%% 1. CLI argument parser
def parse_args():

    """

    Build the argparse parser for the TS comparison.

    :return: Parsed argparse.Namespace.

    """

    p = argparse.ArgumentParser(description='RMSD (A) and energy difference (eV) of the transition states located by finished staged searches against ts_guess, the reference saddle and the exact 2D saddles. Writes CSV, NPZ and XYZ files for plot_ts_compare.py.')
    p.add_argument('searches', nargs='*',
                   help='Search folders (holding search.json) or folders searched recursively for them. Default: everything under reaction/output.')
    p.add_argument('--outdir', type=str, default=DEFAULT_OUTDIR,
                   help='Folder receiving the tables and structures. Default: reaction/output/ts_compare.')
    p.add_argument('--labels', type=str, nargs='*', default=None,
                   help='Run label of each search folder, in order. Default: the last two path components.')
    p.add_argument('--orcapath', type=str, default=None,
                   help='ORCA executable, for reactions whose back end is ORCA. Default: the ASE configuration.')
    p.add_argument('--grid', type=int, default=None,
                   help='Points per axis of the 2D potential grid stored for the figures. Default: the grid of the exact committor.')
    p.add_argument('--keep_others', action='store_true',
                   help='Keep the rows of the systems not graded in this call from the tables already in --outdir, for regrading some systems once the folders of the others are gone.')

    return p.parse_args()

#%% 2. Inputs
def FindSearches(Paths):

    """

    :param1 Paths: folders or search.json files; empty for reaction/output.

    :return: Folders: every folder holding a search.json, sorted.

    """

    Paths = Paths or [OUTPUT_ROOT]
    Skip = (os.path.abspath(DEFAULT_OUTDIR), os.path.abspath(SADDLE_ROOT))
    Found = []
    for P in Paths:
        P = os.path.abspath(P)
        if P.endswith(".json"):
            P = os.path.dirname(P)
        if os.path.isfile(os.path.join(P, "search.json")):
            Found.append(P)
            continue
        for Root, Dirs, Files in os.walk(P):
            Dirs[:] = [d for d in Dirs if os.path.join(Root, d) not in Skip and d not in ("explore", "chains")]
            if "search.json" in Files:
                Found.append(Root)

    return sorted(set(Found))

def RunLabel(Folder):

    """

    :return: Label: the last two components of the folder path.

    """

    Parts = os.path.normpath(Folder).split(os.sep)

    return "/".join(Parts[-2:])

def SystemClass(Name, Meta):

    """

    :return: Class: 'two_dimensional', 'peptide' (dihedral fingerprint) or 'gas_phase'.

    """

    if Name in ANALYTIC:
        return "two_dimensional"
    if Meta.get("fingerprint", {}).get("kind") == "dihedrals":
        return "peptide"

    return "gas_phase"

def CrossingEnergy(File):

    """

    :return: E: the energy in eV written in the comment line of a crossing file.

    """

    with open(File) as f:
        Text = f.read()
    M = re.search(r"E = ([-+0-9.eE]+) eV", Text)

    return float(M.group(1)) if M else np.nan

def CrossingXi(Channel, Xi):

    """

    :param1 Channel: channel entry of search.json.
    :param2 Xi:      coordinate of every chain node.

    :return: Xi: coordinate of the crossing, interpolated between the two
             nodes it lies between, NaN without a crossing.

    """

    X = Channel.get("Crossing")
    if X is None or X.get("Node") is None:
        return np.nan
    n = int(X["Node"])
    t = float(X.get("t") or 0.0)
    if n + 1 >= len(Xi):
        return float(Xi[n])

    return float(Xi[n] + t*(Xi[n + 1] - Xi[n]))

def RouteFlags(Report):

    """

    :return: OnRoute, Bottleneck: the pairs along the certified A -> B route,
             and the pair whose best channel sets the bound.

    """

    Route = Report["AtoB"].get("Route") or []
    Pairs = set(zip(Route[:-1], Route[1:]))
    Edges = {}
    for P in Report["Pairs"]:
        if (P["From"], P["To"]) in Pairs and P["Channels"]:
            Edges[(P["From"], P["To"])] = P["Channels"][0]["EUp"]
    Bottleneck = max(Edges, key=Edges.get) if Edges else None

    return Pairs, Bottleneck

def ChannelMeta(Report):

    """

    :return: Items: (Pair dict, channel index, channel dict, OnRoute, Primary)
             for every channel; the primary channel is the lowest one of the
             bottleneck pair, the TS the search itself would report.

    """

    OnRoute, Bottleneck = RouteFlags(Report)
    Items = []
    for P in Report["Pairs"]:
        Key = (P["From"], P["To"])
        for k, Ch in enumerate(P["Channels"]):
            Items.append((P, k, Ch, Key in OnRoute, Key == Bottleneck and k == 0))

    return Items

#%% 3. Molecules
class MoleculeGrader:

    """

    Reference structures, energies and atom subsets of one reaction, with the
    energy back end of each run recovered from its state energies.

    """

    def __init__(self, Name, WorkDir, OrcaPath=None):

        """

        :param1 Name:     reaction name under reaction/input.
        :param2 WorkDir:  folder for calculator scratch files.
        :param3 OrcaPath: ORCA executable, None for the ASE configuration.

        """

        from treax_expedition.io_xyz import readMeta

        self.Name = Name
        self.WorkDir = WorkDir
        self.OrcaPath = OrcaPath
        self.Meta = readMeta(Name)
        self.Class = SystemClass(Name, self.Meta)
        self.Guess = asio.read(os.path.join(INPUT_ROOT, Name, "ts_guess.xyz"))
        self.Reactant = asio.read(os.path.join(INPUT_ROOT, Name, "reactant.xyz"))
        self.Symbols = self.Reactant.get_chemical_symbols()
        self.Masses = self.Reactant.get_masses()
        self.Heavy = np.array([i for i, s in enumerate(self.Symbols) if s != "H"])
        FP = self.Meta.get("fingerprint", {})
        self.Quads = FP.get("quads", []) if FP.get("kind") == "dihedrals" else []
        self.Backbone = np.array(sorted(set(sum(self.Quads, [])))) if self.Quads else None
        # inversion is a symmetry of the reaction only when the reactant is achiral
        self.Chiral = IsChiral(self.Reactant.get_positions(), self.Symbols)
        self.Systems = {}
        self.RefEnergies = {}

    def System(self, Calculator):

        """

        :return: System: the AtomsSystem of a back end, built once.

        """

        if Calculator not in self.Systems:
            from treax_expedition.pipeline import MoleculeSpec
            Spec = MoleculeSpec(self.Name, Calculator=Calculator, WorkDir=self.WorkDir,
                                OrcaPath=self.OrcaPath)
            self.Systems[Calculator] = Spec.BuildSystem(Worker="ts_compare")

        return self.Systems[Calculator]

    def Backend(self, Folder, Report, Tol=1e-3):

        """

        :param1 Folder: search folder, holding state_A.xyz.
        :param2 Report: its search.json.
        :param3 Tol:    largest energy mismatch in eV accepted.

        :return: Calculator: the back end that reproduces the energy of state A.

        """

        EA = [s["Energy"] for s in Report["States"] if s["Name"] == "A"][0]
        X = asio.read(os.path.join(Folder, "state_A.xyz")).get_positions().reshape(-1)
        Tried = []
        for c in dict.fromkeys([self.Meta.get("calculator", "xtb"), "xtb"]):
            try:
                E = self.System(c).Energy(X)
            except RuntimeError as Err:
                Tried.append("%s unavailable (%s)" % (c, str(Err).split(".")[0]))
                continue
            Tried.append("%s %s" % (c, "failed" if E is None else "%.6f" % (E,)))
            if E is not None and abs(E - EA) < Tol:
                return c
        raise RuntimeError("%s: no back end reproduces E(A) = %.6f eV of %s (%s)"
                           % (self.Name, EA, Folder, ", ".join(Tried)))

    def References(self, Calculator):

        """

        :return: Refs: (label, positions, energy in eV) of ts_guess and, when
                 reference_saddle.py wrote one for this back end, the saddle.

        """

        if Calculator not in self.RefEnergies:
            Refs = [("ts_guess", self.Guess.get_positions())]
            File = os.path.join(SADDLE_ROOT, "%s_%s" % (self.Name, Calculator), "ts_saddle.xyz")
            if os.path.isfile(File):
                Refs.append(("saddle", asio.read(File).get_positions()))
            Out = []
            for Label, X in Refs:
                E = self.System(Calculator).Energy(X.reshape(-1))
                Out.append((Label, X, np.nan if E is None else E))
            self.RefEnergies[Calculator] = Out

        return self.RefEnergies[Calculator]

    def Angles(self, X):

        """

        :return: Phi, Psi: the first two fingerprint dihedrals, or NaN.

        """

        if len(self.Quads) < 2:
            return np.nan, np.nan

        return Dihedral(X, self.Quads[0]), Dihedral(X, self.Quads[1])

    def Compare(self, X, Ref):

        """

        :param1 X:   located TS, shape (N, 3).
        :param2 Ref: reference, shape (N, 3).

        :return: Row, Aligned: the distances, and X permuted and aligned onto Ref.

        """

        r, Aligned, Order, Mirrored = SymmetryRMSD(Ref, X, self.Symbols, Mirror=not self.Chiral)
        Row = {"RMSD": RMSD(Ref, X), "RMSD_mw": RMSD(Ref, X, Weights=self.Masses), "RMSD_sym": r,
               "RMSD_heavy": RMSD(Ref, Aligned, Subset=self.Heavy),
               "RMSD_backbone": RMSD(Ref, Aligned, Subset=self.Backbone) if self.Backbone is not None else np.nan,
               "Permuted": bool(np.any(Order != np.arange(len(Order)))), "Mirrored": bool(Mirrored)}
        if self.Quads:
            Phi, Psi = self.Angles(X)
            PhiRef, PsiRef = self.Angles(Ref)
            Row.update(Phi=Phi, Psi=Psi, PhiRef=PhiRef, PsiRef=PsiRef,
                       dPhi=AngleDifference(Phi, PhiRef), dPsi=AngleDifference(Psi, PsiRef))

        return Row, Aligned

def GradeMoleculeSearch(G, Folder, Report, Run, OutDir):

    """

    :param1 G:      MoleculeGrader of the reaction.
    :param2 Folder: search folder.
    :param3 Report: its search.json.
    :param4 Run:    run label.
    :param5 OutDir: output folder, receiving structures/.

    :return: Rows, ChainRows, Backend.

    """

    Calc = G.Backend(Folder, Report)
    Refs = G.References(Calc)
    EA = [s["Energy"] for s in Report["States"] if s["Name"] == "A"][0]
    Rows, ChainRows = [], []
    for P, k, Ch, OnRoute, Primary in ChannelMeta(Report):
        Stem = os.path.join(Folder, "chains", Ch["File"])
        Dat = np.atleast_2d(np.loadtxt(Stem + ".dat"))
        Frames = asio.read(Stem + ".xyz", index=":")
        Top = int(np.argmax(Dat[:, 3]))
        for i, (F, d) in enumerate(zip(Frames, Dat)):
            Phi, Psi = G.Angles(F.get_positions())
            ChainRows.append({"Class": G.Class, "System": G.Name, "Run": Run, "Pair": "%s_%s" % (P["From"], P["To"]),
                              "Channel": k, "Node": i, "Xi": d[1], "E_eV": d[3], "ERelA_eV": d[3] - EA,
                              "IsTop": i == Top, "Phi": Phi, "Psi": Psi})
        Candidates = [("top", Frames[Top].get_positions(), Dat[Top, 3])]
        Xis = {"top": Dat[Top, 1], "crossing": CrossingXi(Ch, Dat[:, 1])}
        if os.path.isfile(Stem + "_crossing.xyz"):
            Candidates.append(("crossing", asio.read(Stem + "_crossing.xyz").get_positions(),
                               CrossingEnergy(Stem + "_crossing.xyz")))
        for Cand, X, E in Candidates:
            for Label, XRef, ERef in Refs:
                Row, Aligned = G.Compare(X, XRef)
                Name = "%s__%s__%s__%s__vs_%s.xyz" % (G.Name, Run.replace("/", "-"), Ch["File"], Cand, Label)
                WritePair(os.path.join(OutDir, "structures", Name), G.Symbols, XRef, Aligned,
                          "%s %s %s" % (G.Name, Label, Calc), "%s %s %s, RMSD_sym %.4f A, dE %+.5f eV"
                          % (Run, Ch["File"], Cand, Row["RMSD_sym"], E - ERef))
                Row.update(Class=G.Class, System=G.Name, Run=Run, Pair="%s_%s" % (P["From"], P["To"]),
                           Channel=k, OnRoute=OnRoute, Primary=Primary, Candidate=Cand, Reference=Label,
                           RefLabel=Label, Backend=Calc, NAtoms=len(G.Symbols), Xi=Xis[Cand], E_eV=E, ERef_eV=ERef,
                           dE_eV=E - ERef, EUp_eV=Ch["EUp"], EUpMinusRef_eV=Ch["EUp"] - ERef,
                           Certified=Ch["Certified"], Energies=Report["Cost"]["Total"],
                           WallSeconds=Report["WallSeconds"], Structure=os.path.join("structures", Name),
                           Folder=Folder)
                Rows.append(Row)

    return Rows, ChainRows, Calc

def WritePair(File, Symbols, Ref, X, CommentRef, CommentX):

    """

    :return: None; writes the reference and the aligned TS as two xyz frames.

    """

    from ase import Atoms

    os.makedirs(os.path.dirname(File), exist_ok=True)
    asio.write(File, [Atoms(Symbols, positions=Ref, info={"comment": CommentRef}),
                      Atoms(Symbols, positions=X, info={"comment": CommentX})], format="extxyz")

    return None

#%% 4. Analytic 2D models
def GradeAnalyticSearch(Name, Folder, Report, Run):

    """

    :param1 Name:   '2d', 'muller' or 'two_paths'.
    :param2 Folder: search folder.
    :param3 Report: its search.json.
    :param4 Run:    run label.

    :return: Rows, ChainRows: every channel against its nearest exact saddle;
             no alignment, the frame of a 2D model is fixed.

    """

    from analytic_2d.pipeline import AnalyticSpec

    Spec = AnalyticSpec(Name)
    P = Spec.Potential
    Saddles = P.Saddles()
    ESaddle = P.Total(Saddles)*Spec.Unit
    EA = [s["Energy"] for s in Report["States"] if s["Name"] == "A"][0]
    Rows, ChainRows = [], []
    for Pair, k, Ch, OnRoute, Primary in ChannelMeta(Report):
        Stem = os.path.join(Folder, "chains", Ch["File"])
        Dat = np.atleast_2d(np.loadtxt(Stem + ".dat"))
        Top = int(np.argmax(Dat[:, 2]))
        for i, d in enumerate(Dat):
            ChainRows.append({"Class": "two_dimensional", "System": Name, "Run": Run,
                              "Pair": "%s_%s" % (Pair["From"], Pair["To"]), "Channel": k, "Node": i,
                              "Xi": d[3], "E_eV": d[2]*Spec.Unit, "ERelA_eV": d[2]*Spec.Unit - EA,
                              "IsTop": i == Top, "X": d[0], "Y": d[1]})
        Candidates = [("top", Dat[Top, :2], Dat[Top, 2]*Spec.Unit)]
        Xis = {"top": Dat[Top, 3], "crossing": CrossingXi(Ch, Dat[:, 3])}
        if os.path.isfile(Stem + "_crossing.dat"):
            Cross = np.loadtxt(Stem + "_crossing.dat")
            Candidates.append(("crossing", Cross[:2], P.Total(Cross[:2])*Spec.Unit))
        # the saddle of a channel is the one nearest its crossing (or top node)
        Anchor = Candidates[-1][1]
        s = int(np.argmin(np.sum((Saddles - Anchor)**2, axis=1)))
        for Cand, X, E in Candidates:
            Rows.append({"Class": "two_dimensional", "System": Name, "Run": Run,
                         "Pair": "%s_%s" % (Pair["From"], Pair["To"]), "Channel": k, "OnRoute": OnRoute,
                         "Primary": Primary, "Candidate": Cand, "Reference": "exact", "RefLabel": "S%i" % (s + 1),
                         "Backend": "analytic", "NAtoms": 1, "Xi": Xis[Cand], "RMSD": float(np.linalg.norm(X - Saddles[s])),
                         "RMSD_mw": float(np.linalg.norm(X - Saddles[s])), "RMSD_sym": float(np.linalg.norm(X - Saddles[s])),
                         "E_eV": E, "ERef_eV": ESaddle[s], "dE_eV": E - ESaddle[s], "EUp_eV": Ch["EUp"],
                         "EUpMinusRef_eV": Ch["EUp"] - ESaddle[s], "Certified": Ch["Certified"],
                         "X": X[0], "Y": X[1], "XRef": Saddles[s][0], "YRef": Saddles[s][1],
                         "Energies": Report["Cost"]["Total"], "WallSeconds": Report["WallSeconds"],
                         "Folder": Folder})

    return Rows, ChainRows

def WritePotentialGrid(Name, Searches, OutDir, NGrid=None):

    """

    Store what the 2D figures need: the potential on a grid in eV, the
    stationary points and the exact committor of every pair any run visited.

    :param1 Name:     '2d', 'muller' or 'two_paths'.
    :param2 Searches: (Report, Folder) of every run on this potential.
    :param3 OutDir:   output folder.
    :param4 NGrid:    points per axis, None for the default grid.

    :return: File: the pes_<name>.npz written.

    """

    from analytic_2d.pipeline import AnalyticSpec
    from analytic_2d.reference import BuildGrid, SolveCommittor

    Spec = AnalyticSpec(Name)
    P = Spec.Potential
    x, y, V = BuildGrid(P, NX=NGrid, NY=NGrid)
    Labels, Points = P.StationaryPoints()
    Minima = P.Minima()
    Data = {"x": x, "y": y, "V": V, "Labels": np.array(Labels), "Points": Points,
            "E": P.Total(Points)*Spec.Unit, "EA": P.Total(Minima[0])*Spec.Unit,
            "kT": P.kTAt(Spec.TauPhys)*Spec.Unit}
    for Report, Folder in Searches:
        for Pair in Report["Pairs"]:
            Key, Centres = ExactPair(P, Folder, Pair)
            # keyed by the exact minima, so runs naming them differently share one solve
            if Key not in Data:
                Data[Key] = SolveCommittor(x, y, V, Spec.TauPhys, BasinRadius=P.BasinRadius,
                                           Centres=tuple(tuple(Minima[c]) for c in Centres))
                Data["pK" + Key[1:]] = KolmogorovDensity(x, y, V, Data[Key], Data["kT"])
    File = os.path.join(OutDir, "pes_%s.npz" % (Name,))
    np.savez_compressed(File, **Data)

    return File

def ExactPair(P, Folder, Pair):

    """

    :param1 P:      Potential2D.
    :param2 Folder: search folder, holding state_<name>.dat.
    :param3 Pair:   pair entry of search.json.

    :return: Key, Centres: 'q_<a>_<b>' named by the exact minima nearest the
             two states of the pair, and the indices of those minima.

    """

    Minima = P.Minima()
    Centres = []
    for S in (Pair["From"], Pair["To"]):
        with open(os.path.join(Folder, "state_%s.dat" % (S,))) as f:
            XS = np.array([float(v) for v in [l for l in f if not l.startswith("#")][0].split()[:2]])
        Centres.append(int(np.argmin(np.sum((Minima - XS)**2, axis=1))))

    return "q_%s_%s" % (P.BasinNames[Centres[0]], P.BasinNames[Centres[1]]), Centres

def KolmogorovDensity(x, y, V, Q, kT):

    """

    :param1 x, y: grid axes in angstroms.
    :param2 V:    potential on the grid in eV.
    :param3 Q:    exact committor on the grid.
    :param4 kT:   thermal energy in eV.

    :return: pK: |grad q|^2 exp(-V/kT), the transition state ensemble the
             committor defines, normalised to unit integral over the grid.

    """

    gx, gy = np.gradient(Q, x, y)
    p = (gx**2 + gy**2)*np.exp(-(V - V.min())/kT)

    return p/(np.sum(p)*(x[1] - x[0])*(y[1] - y[0]))

def LatentOf(Model, X):

    """

    :return: z, q: latent and committor of a CommittorNN at every configuration.

    """

    import torch

    with torch.no_grad():
        t = torch.as_tensor(np.ascontiguousarray(np.atleast_2d(X)), dtype=next(Model.parameters()).dtype)
        z = Model(t)

        return z.numpy(), Model.CommittorFromLatent(z).numpy()

def WriteSampling(Name, Searches, OutDir):

    """

    Store what the first run on a 2D potential sampled, for the figures: for
    every pair that went through S4, its pooled sample with MBAR weights and
    the learned latent z and committor at that sample, on the grid of
    pes_<name>.npz and at the stationary points; for every channel, the S6
    transition state ensemble. One run only: the latents of two runs differ.

    :param1 Name:     '2d', 'muller' or 'two_paths'.
    :param2 Searches: (Report, Folder) of every run on this potential.
    :param3 OutDir:   output folder, holding pes_<name>.npz.

    :return: File: the sampling_<name>.npz written, None when the run has no S4 or S6 output.

    """

    from scipy.interpolate import RegularGridInterpolator

    from analytic_2d.pipeline import AnalyticSpec
    from utils.committor import LoadCommittor
    from utils.mc.ensemble import NormalisedWeights

    Report, Folder = Searches[0]
    P = AnalyticSpec(Name).Potential
    G = np.load(os.path.join(OutDir, "pes_%s.npz" % (Name,)))
    Grid = np.stack(np.meshgrid(G["x"], G["y"], indexing="ij"), axis=-1).reshape(-1, 2)
    Data = {}
    Pairs = []
    for Pair in Report["Pairs"]:
        Tag = "%s_%s" % (Pair["From"], Pair["To"])
        Key, _ = ExactPair(P, Folder, Pair)
        Stage = os.path.join(Folder, "committor_%s" % (Tag,))
        Model = None
        if os.path.isfile(os.path.join(Stage, "data.npz")):
            Model = LoadCommittor(os.path.join(Stage, "committor.pt"))
            S = np.load(os.path.join(Stage, "data.npz"))
            Exact = RegularGridInterpolator((G["x"], G["y"]), G[Key], bounds_error=False, fill_value=np.nan)
            z, q = LatentOf(Model, S["X"])
            zGrid, qGrid = LatentOf(Model, Grid)
            Data.update({"Ref_" + Tag: Key, "X_" + Tag: S["X"], "W_" + Tag: S["Weights"],
                         "Block_" + Tag: S["Block"], "Walker_" + Tag: S["Walker"], "zFit_" + Tag: z,
                         "qFit_" + Tag: q, "qExact_" + Tag: Exact(S["X"]),
                         "zFitGrid_" + Tag: zGrid.reshape(len(G["x"]), len(G["y"])),
                         "qFitGrid_" + Tag: qGrid.reshape(len(G["x"]), len(G["y"])),
                         "zFitPoints_" + Tag: LatentOf(Model, G["Points"])[0]})
        for k, Ch in enumerate(Pair["Channels"]):
            File = os.path.join(Folder, "chains", Ch["File"] + "_tse.npz")
            if not os.path.isfile(File):
                continue
            T = np.load(File)
            Data["TSEX_%s_ch%i" % (Tag, k)] = T["M"]
            Data["TSEW_%s_ch%i" % (Tag, k)] = NormalisedWeights(T["LogW"])
            if Model is not None:
                Data["TSEzFit_%s_ch%i" % (Tag, k)] = LatentOf(Model, T["M"])[0]
        if Model is not None or any(k.startswith("TSEX_" + Tag) for k in Data):
            Pairs.append(Tag)
    if not Pairs:
        return None
    if len(Searches) > 1:
        print("[ts_compare] %s: sampling stored from %s only" % (Name, Folder))
    File = os.path.join(OutDir, "sampling_%s.npz" % (Name,))
    np.savez_compressed(File, Folder=Folder, Pairs=np.array(Pairs), **Data)

    return File

#%% 5. Output tables
def WriteTable(File, Rows, Columns):

    """

    :return: None; writes Rows as CSV with the given columns, blanks for missing values.

    """

    with open(File, "w", newline="") as f:
        W = csv.DictWriter(f, fieldnames=Columns, extrasaction="ignore")
        W.writeheader()
        for R in Rows:
            W.writerow({c: _Cell(R.get(c)) for c in Columns})

    return None

def _Cell(v):

    """

    :return: Text: a CSV cell, NaN and None as blanks, floats at full precision.

    """

    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    if isinstance(v, (float, np.floating)):
        return "%.10g" % (v,)

    return v

def ReadTable(File):

    """

    :return: Rows: the rows of a CSV as dicts of text, empty when it is absent.

    """

    if not os.path.isfile(File):
        return []
    with open(File, newline="") as f:
        return list(csv.DictReader(f))

def MergeRows(Old, New, Graded):

    """

    :param1 Old:    rows (or sources) already in the output folder.
    :param2 New:    those of the systems graded now.
    :param3 Graded: names of those systems.

    :return: Rows: Old with the rows of every graded system replaced by its
             new ones, where its first old row stood; new systems last.

    """

    Out, Placed = [], set()
    for r in Old:
        s = r["System"]
        if s not in Graded:
            Out.append(r)
        elif s not in Placed:
            Out += [n for n in New if n["System"] == s]
            Placed.add(s)

    return Out + [n for n in New if n["System"] not in Placed]

#%% 6. Main
def main():

    """

    Grade every search folder and write the tables, grids and structures.

    :return: None.

    """

    Args = parse_args()
    Folders = FindSearches(Args.searches)
    if not Folders:
        raise SystemExit("[ts_compare] no search.json found under %s" % (", ".join(Args.searches) or OUTPUT_ROOT,))
    if Args.labels is not None and len(Args.labels) != len(Folders):
        raise SystemExit("[ts_compare] %i labels for %i search folders" % (len(Args.labels), len(Folders)))
    OutDir = os.path.abspath(Args.outdir)
    os.makedirs(os.path.join(OutDir, "structures"), exist_ok=True)
    Graders, Grids, Rows, ChainRows, Sources = {}, {}, [], [], []
    for n, Folder in enumerate(Folders):
        with open(os.path.join(Folder, "search.json")) as f:
            Report = json.load(f)
        Name = Report["System"]
        Run = Args.labels[n] if Args.labels is not None else RunLabel(Folder)
        Start = time.time()
        if Name in ANALYTIC:
            R, C = GradeAnalyticSearch(Name, Folder, Report, Run)
            Grids.setdefault(Name, []).append((Report, Folder))
            Source = {"Folder": Folder, "Run": Run, "System": Name, "Backend": "analytic",
                      "Grid": "pes_%s.npz" % (Name,)}
        else:
            if Name not in Graders:
                Graders[Name] = MoleculeGrader(Name, os.path.join(OutDir, "work"), Args.orcapath)
            R, C, Calc = GradeMoleculeSearch(Graders[Name], Folder, Report, Run, OutDir)
            Source = {"Folder": Folder, "Run": Run, "System": Name, "Backend": Calc,
                      "Chiral": Graders[Name].Chiral,
                      "References": [l for l, _, _ in Graders[Name].References(Calc)]}
        Rows += R
        ChainRows += C
        Sources.append(Source)
        Best = [r for r in R if r["Candidate"] == "crossing" and r["Primary"]]
        print("[ts_compare] %-10s %-24s %s: %i rows%s (%.0f s)"
              % (Name, Run, Source["Backend"], len(R),
                 "".join("; %s RMSD_sym %.3f A dE %+.4f eV" % (r["Reference"], r["RMSD_sym"], r["dE_eV"])
                         for r in Best), time.time() - Start))
    for Name, Searches in Grids.items():
        WritePotentialGrid(Name, Searches, OutDir, Args.grid)
        WriteSampling(Name, Searches, OutDir)
    if Args.keep_others:
        Graded = {s["System"] for s in Sources}
        Rows = MergeRows(ReadTable(os.path.join(OutDir, "ts_compare.csv")), Rows, Graded)
        ChainRows = MergeRows(ReadTable(os.path.join(OutDir, "chains.csv")), ChainRows, Graded)
        Old = os.path.join(OutDir, "sources.json")
        if os.path.isfile(Old):
            with open(Old) as f:
                Sources = MergeRows(json.load(f)["Sources"], Sources, Graded)
    WriteTable(os.path.join(OutDir, "ts_compare.csv"), Rows, COLUMNS)
    WriteTable(os.path.join(OutDir, "chains.csv"), ChainRows, CHAIN_COLUMNS)
    with open(os.path.join(OutDir, "sources.json"), "w") as f:
        json.dump({"Created": time.strftime("%Y-%m-%d %H:%M:%S"), "Sources": Sources,
                   "Units": {"RMSD": "angstrom", "E": "eV"}}, f, indent=1)
    Work = os.path.join(OutDir, "work")
    if os.path.isdir(Work) and not os.listdir(Work):
        os.rmdir(Work)
    print("[ts_compare] %i rows from %i searches -> %s" % (len(Rows), len(Folders), OutDir))

    return None

if __name__ == "__main__":
    main()
