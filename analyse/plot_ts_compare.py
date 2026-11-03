#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Figures and summary tables of the TS comparison, drawn only from the files
ts_compare.py wrote (ts_compare.csv, chains.csv, pes_<name>.npz), so the
presentation can be changed and redrawn without grading again. Every style
choice lives in figures/style.json, written with the defaults on first use.

Created on: Thu Oct 01 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import argparse
import copy
import json
import os
import sys

import ase.io as asio
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.data import atomic_numbers, covalent_radii
from matplotlib.patches import Circle, Polygon

# the repository root, one level above this folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyse.ts_metrics import Dihedral
from treax_expedition.io_xyz import INPUT_ROOT, OUTPUT_ROOT

DEFAULT_INDIR = os.path.join(OUTPUT_ROOT, "ts_compare")

# defaults of figures/style.json; edit that file, not these, to restyle
STYLE = {
    "formats": ["png", "pdf"],
    "dpi": 300,
    "font_family": "DejaVu Sans",
    "font_size": 8,
    "width_single_in": 3.5,
    "width_double_in": 7.2,
    "colors": {"crossing": "#2a78d6", "top": "#eb6834", "two_dimensional": "#2a78d6",
               "gas_phase": "#eb6834", "peptide": "#1baf7a", "ink": "#0b0b0b",
               "ink_secondary": "#52514e", "grid": "#e1e0dc", "band": "#f0efec",
               "chain": "#9a9893", "pes_cmap": "Blues_r", "s4": "#1baf7a", "tse": "#2a78d6",
               "reference": "#6e6d68", "density_cmap": "Oranges"},
    "markers": {"crossing": "o", "top": "D", "reference": "x", "ts_guess": "s"},
    "marker_size": 4.5,
    "line_width": 1.0,
    "candidates": ["crossing", "top"],
    "candidate_labels": {"crossing": "q = 1/2 crossing", "top": "chain top node"},
    "reference": "auto",
    "only_primary": True,
    "rmsd_column": "RMSD_sym",
    "threshold_dE_eV": 0.0434,
    "threshold_dE_label": "1 kcal/mol",
    "threshold_rmsd_A": 0.1,
    "kT_eV": 0.0257,
    "show_kT": True,
    "pes_emax_eV": None,
    "pes_levels": 24,
    "pes_alpha": 0.6,
    "dE_linthresh_eV": 0.1,
    "show_committor": True,
    "two_dimensional_errors": False,
    "pes_view_factor": 1.3,
    "two_dimensional_layout": {"left": 0.5, "right": 0.1, "col_gap": 0.4, "profile_height": 0.95,
                               "title": 0.24, "gap": 0.1, "tick_room": 0.2,
                               "axis_room": 0.42, "legend": 0.22},
    "density_levels": [0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "density_bins": 70,
    "density_smooth": 1.0,
    "z_bins": 60,
    "tse_points": 400,
    "parity_points": 2000,
    "parity_inset_in": 0.75,
    "tse_range": [0.05, 0.95],
    "sampling_labels": {"reference": "Reference", "s4": "Biased", "tse": "TSE", "reference_q": "reference q",
                        "learned_q": "learned q", "sampling": "Sampling",
                        "kolmogorov": "Kolmogorov distribution p$_\\mathrm{K}$", "latent": "learned latent z"},
    "class_labels": {"two_dimensional": "2D models", "gas_phase": "Baker / SACTS gas phase",
                     "peptide": "Alanine dipeptide"},
    "system_labels": {"2d": "Proton transfer 2D", "muller": "Müller–Brown",
                      "two_paths": "Two-path potential",
                      "hcn": "HCN → HNC", "h2co": "H$_2$CO → H$_2$ + CO",
                      "hcho": "HCOH → H$_2$CO", "hcch": "HCCH → H$_2$CC",
                      "ch3o": "CH$_3$O· → ·CH$_2$OH", "ch3cho": "CH$_3$CHO → CH$_2$CHOH",
                      "ch3ch2f": "C$_2$H$_4$ + HF → CH$_3$CH$_2$F",
                      "migration": "C$_3$H$_5$O$_2$· ring opening",
                      "sn2": "Cl$^-$ + CH$_3$Br → CH$_3$Cl + Br$^-$",
                      "alanine_dipeptide": "Alanine dipeptide C7eq → C7ax"},
    "system_order": ["hcn", "h2co", "hcho", "hcch", "ch3o", "ch3cho", "ch3ch2f", "migration", "sn2"],
    "profile_columns": 3,
    "schematic_candidate": "crossing",
    "schematic_columns": 5,
    "schematic_common_scale": True,
    "schematic_row_labels": ["Reference", "tReax"],
    "schematic_atom_colors": {"H": "#ffffff", "C": "#5c5c5c", "N": "#2f4fd8", "O": "#e8130c", "F": "#7fd04a",
                              "Cl": "#2fc22f", "Br": "#a52a2a", "other": "#d98cd9"},
    "schematic_atom_radius": [0.195, 0.178],
    "schematic_bond_radius_A": 0.10,
    "schematic_bond_factor": 1.15,
    "schematic_bond_orders": True,
    "schematic_change_factor": 1.3,
    "schematic_geometry": {},
    "schematic_view": {},
    "schematic_label_color": "#e3120b",
    "schematic_margin_A": 0.6,
}

#%% 1. CLI argument parser
def parse_args():

    """

    Build the argparse parser for the figures.

    :return: Parsed argparse.Namespace.

    """

    p = argparse.ArgumentParser(description='Figures and tables of the TS comparison, from the files of ts_compare.py only.')
    p.add_argument('--indir', type=str, default=DEFAULT_INDIR,
                   help='Folder written by ts_compare.py. Default: reaction/output/ts_compare.')
    p.add_argument('--style', type=str, default=None,
                   help='Style file. Default: <indir>/figures/style.json, created with the defaults when absent.')
    p.add_argument('--only', type=str, nargs='*', default=None,
                   choices=['two_dimensional', 'gas_phase', 'schematic', 'peptide', 'scatter', 'profiles', 'tables'],
                   help='Draw only these items. Default: all.')
    p.add_argument('--reset_style', action='store_true',
                   help='Overwrite the style file with the defaults first.')

    return p.parse_args()

#%% 2. Style and data
def LoadStyle(File, Reset=False):

    """

    :param1 File:  style.json path.
    :param2 Reset: rewrite it with the defaults.

    :return: Style: the defaults updated by the file (nested dicts merged).

    """

    Style = copy.deepcopy(STYLE)
    if Reset or not os.path.isfile(File):
        os.makedirs(os.path.dirname(File), exist_ok=True)
        with open(File, "w") as f:
            json.dump(STYLE, f, indent=1, ensure_ascii=False)
        print("[plot_ts_compare] style written to %s" % (File,))
    with open(File) as f:
        User = json.load(f)
    for k, v in User.items():
        if isinstance(v, dict) and isinstance(Style.get(k), dict):
            Style[k].update(v)
        else:
            Style[k] = v
    plt.rcParams.update({"font.family": Style["font_family"], "font.size": Style["font_size"],
                         "axes.edgecolor": Style["colors"]["ink_secondary"],
                         "axes.labelcolor": Style["colors"]["ink"], "xtick.color": Style["colors"]["ink_secondary"],
                         "ytick.color": Style["colors"]["ink_secondary"], "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "savefig.bbox": "tight"})

    return Style

def LoadTables(InDir):

    """

    :return: Rows, Chains: ts_compare.csv and chains.csv as DataFrames.

    """

    Rows = pd.read_csv(os.path.join(InDir, "ts_compare.csv"))
    Chains = pd.read_csv(os.path.join(InDir, "chains.csv"))

    return Rows, Chains

def ChooseReference(Rows, Style):

    """

    :return: Rows: one reference per system; 'auto' takes the saddle when
             reference_saddle.py made one, else ts_guess (exact for 2D).

    """

    if Style["reference"] != "auto":
        return Rows[Rows["Reference"].isin([Style["reference"], "exact"])]
    Keep = []
    for _, G in Rows.groupby("System"):
        Refs = set(G["Reference"])
        Pick = "exact" if "exact" in Refs else ("saddle" if "saddle" in Refs else "ts_guess")
        Keep.append(G[G["Reference"] == Pick])

    return pd.concat(Keep)

def Selected(Rows, Style, Class=None):

    """

    :return: Rows: of one class, the chosen candidates, on-route (and primary) channels.

    """

    R = Rows[Rows["Candidate"].isin(Style["candidates"])]
    if Class is not None:
        R = R[R["Class"] == Class]
    if Class != "two_dimensional":
        R = R[R["OnRoute"].astype(bool)]
        if Style["only_primary"]:
            R = R[R["Primary"].astype(bool)]

    return R

def Label(Style, System):

    """

    :return: Text: display name of a system.

    """

    return Style["system_labels"].get(System, System)

def PlainLabel(Style, System):

    """

    :return: Text: display name without the mathtext markup, for tables.

    """

    Text = Label(Style, System)
    for a, b in (("$_", ""), ("$^", ""), ("$", ""), ("{", ""), ("}", "")):
        Text = Text.replace(a, b)

    return Text

def Save(Fig, Style, OutDir, Name):

    """

    :return: None; writes Name in every format of the style.

    """

    for Ext in Style["formats"]:
        Fig.savefig(os.path.join(OutDir, "%s.%s" % (Name, Ext)), dpi=Style["dpi"])
    plt.close(Fig)
    print("[plot_ts_compare] %s.{%s}" % (os.path.join(OutDir, Name), ",".join(Style["formats"])))

    return None

def Grid(Ax, Style, Axis="x"):

    """

    :return: None; recessive grid lines behind the data.

    """

    Ax.grid(True, axis=Axis, color=Style["colors"]["grid"], linewidth=0.5)
    Ax.set_axisbelow(True)

    return None

#%% 3. Two-dimensional models
def PlotTwoDimensional(Rows, Chains, InDir, Style, OutDir):

    """

    One panel per potential, x and y on one scale and cut to where the
    potential stays below pes_view_factor times its highest saddle, every
    panel as tall as the others: the potential, every chain, the exact
    q = 1/2 isolines, and the located TSs over the exact saddles; with
    two_dimensional_errors, also RMSD and dE per channel as text.

    :return: None.

    """

    R = Selected(Rows, Style, "two_dimensional")
    Systems = [s for s in ("2d", "muller", "two_paths") if s in set(R["System"])]
    if not Systems:
        return None
    C = Style["colors"]
    G = Style["two_dimensional_layout"]
    Grids = {s: np.load(os.path.join(InDir, "pes_%s.npz" % (s,))) for s in Systems}
    Views = [PotentialView(Grids[s], Style) for s in Systems]
    Ratios = [(v[1] - v[0])/(v[3] - v[2]) for v in Views]
    Width = Style["width_double_in"]
    Ph = (Width - G["left"] - G["right"] - (len(Systems) - 1)*G["col_gap"])/sum(Ratios)
    Bottom = G["axis_room"] + G["legend"]
    Height = G["title"] + Ph + Bottom
    Fig = plt.figure(figsize=(Width, Height))
    x0 = G["left"]
    for n, (Sys, View, Ratio) in enumerate(zip(Systems, Views, Ratios)):
        Ax = Fig.add_axes([x0/Width, Bottom/Height, Ratio*Ph/Width, Ph/Height])
        x0 += Ratio*Ph + G["col_gap"]
        D = Grids[Sys]
        V = D["V"] - float(D["EA"])
        EMax = Style["pes_emax_eV"] or float(np.max(D["E"]) - D["EA"])*1.6
        Levels = np.linspace(min(0.0, float(V.min())), EMax, Style["pes_levels"])
        Ax.contourf(D["x"], D["y"], np.clip(V, None, EMax).T, levels=Levels, cmap=C["pes_cmap"], alpha=Style["pes_alpha"])
        Ax.contour(D["x"], D["y"], np.clip(V, None, EMax).T, levels=Levels[::3], colors=C["ink_secondary"],
                   linewidths=0.3, alpha=0.6)
        if Style["show_committor"]:
            # only over the coloured potential; beyond it the isoline runs over empty high ground
            Far = V > EMax
            for k in [k for k in D.files if k.startswith("q_")]:
                Ax.contour(D["x"], D["y"], np.ma.masked_where(Far, D[k]).T, levels=[0.5], colors=C["ink"],
                           linewidths=0.8, linestyles="--")
        CS = Chains[(Chains["System"] == Sys)]
        for _, Ch in CS.groupby(["Run", "Pair", "Channel"]):
            Ax.plot(Ch["X"], Ch["Y"], color=C["chain"], linewidth=Style["line_width"], zorder=2)
        # the saddle cross sits under the located TSs, its arms showing around them
        IsSaddle = np.char.startswith(D["Labels"].astype(str), "S")
        Ax.plot(D["Points"][IsSaddle, 0], D["Points"][IsSaddle, 1], linestyle="none", marker=Style["markers"]["reference"],
                color=C["ink"], markersize=Style["marker_size"] + 3, markeredgewidth=1.0, zorder=3)
        S = R[R["System"] == Sys]
        for Cand in Style["candidates"]:
            P = S[S["Candidate"] == Cand]
            Ax.plot(P["X"], P["Y"], linestyle="none", marker=Style["markers"][Cand], color=C[Cand],
                    markersize=Style["marker_size"], markeredgecolor="white", markeredgewidth=0.6, zorder=4)
        if Style["two_dimensional_errors"]:
            Lines = []
            for _, Row in S[S["Candidate"] == Style["candidates"][0]].sort_values(["Pair", "Channel"]).iterrows():
                Lines.append("%s: %.3f Å, %+.4f eV" % (Row["RefLabel"], Row[Style["rmsd_column"]], Row["dE_eV"]))
            Ax.text(0.02, 0.98, "\n".join(Lines), transform=Ax.transAxes, va="top", ha="left",
                    fontsize=Style["font_size"] - 1, color=C["ink"],
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.8))
        Ax.set_xlim(View[0], View[1])
        Ax.set_ylim(View[2], View[3])
        Ax.set_aspect("equal")
        Ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        Ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        Ax.set_title(Label(Style, Sys), fontsize=Style["font_size"] + 1)
        if n == 0:
            Ax.set_ylabel("y (Å)")
    # one x label for the row, the legend under it
    Middle = (G["left"] + Width - G["right"])/2.0/Width
    Fig.text(Middle, (Bottom - G["axis_room"] + 0.04)/Height, "x (Å)", ha="center", va="bottom",
             fontsize=Style["font_size"], color=C["ink"])
    Mark = dict(linestyle="none", markersize=Style["marker_size"], markeredgecolor="white", markeredgewidth=0.6)
    Handles = [plt.Line2D([], [], marker=Style["markers"][c], color=C[c], **Mark) for c in Style["candidates"]] + \
              [plt.Line2D([], [], linestyle="none", marker=Style["markers"]["reference"], color=C["ink"],
                          markersize=Style["marker_size"] + 1, markeredgewidth=1.0),
               plt.Line2D([], [], color=C["ink"], linewidth=0.8, linestyle="--"),
               plt.Line2D([], [], color=C["chain"], linewidth=Style["line_width"])]
    Labels = [Style["candidate_labels"][c] for c in Style["candidates"]] + ["exact saddle", "exact q = 1/2", "chain"]
    Fig.legend(Handles, Labels, loc="center", bbox_to_anchor=(Middle, G["legend"]/2.0/Height), ncol=len(Labels),
               fontsize=Style["font_size"] - 1, handlelength=1.6, handletextpad=0.4, columnspacing=1.6,
               borderaxespad=0.0, borderpad=0.0)
    Save(Fig, Style, OutDir, "fig_two_dimensional")

    return None

def PotentialView(D, Style):

    """

    :return: x0, x1, y0, y1: the box where the potential stays below
             pes_view_factor times the highest stationary point above E(A),
             padded by 5 % of its larger side and kept inside the grid.

    """

    In = D["V"] - float(D["EA"]) <= Style["pes_view_factor"]*float(np.max(D["E"]) - D["EA"])
    x, y = D["x"][In.any(axis=1)], D["y"][In.any(axis=0)]
    Pad = 0.05*max(x[-1] - x[0], y[-1] - y[0])

    return (max(D["x"][0], x[0] - Pad), min(D["x"][-1], x[-1] + Pad),
            max(D["y"][0], y[0] - Pad), min(D["y"][-1], y[-1] + Pad))

def PlotTwoDimensionalSampling(Rows, InDir, Style, OutDir):

    """

    What S4 and S6 sampled on the 2D potentials, one column per potential, in
    the manner of the Kolmogorov bias figures of committor_2.0: a, the density
    of the S4 walks, with the learned against the exact committor at those
    walks inset in the emptiest corner of each map; b, the exact Kolmogorov distribution
    |grad q|^2 exp(-V/kT) with the exact and learned q = 1/2 isolines and the
    S6 transition state ensemble; c, the densities along the learned latent z
    of the primary pair; d, the free energy along z, exact and from the S4
    walks reweighted by MBAR, with the stretch of z the S6 ensemble covers.

    :return: None; no figure when ts_compare.py stored no sampling_<name>.npz.

    """

    R = Selected(Rows, Style, "two_dimensional")
    Samples = {s: LoadSampling(InDir, s) for s in ("2d", "muller", "two_paths") if s in set(R["System"])}
    Systems = [s for s, v in Samples.items() if v is not None and S4Pairs(v)]
    if not Systems:
        return None
    C = Style["colors"]
    L = Style["sampling_labels"]
    G = Style["two_dimensional_layout"]
    Grids = {s: np.load(os.path.join(InDir, "pes_%s.npz" % (s,))) for s in Systems}
    # the maps share the view of fig_two_dimensional, x and y on one scale
    Views = [PotentialView(Grids[s], Style) for s in Systems]
    Ratios = [(v[1] - v[0])/(v[3] - v[2]) for v in Views]
    Kinds = ["sampling", "tse", "density", "free_energy"]
    Width, Height, Boxes = TwoDimensionalLayout(Style, Ratios, Kinds)
    Fig = plt.figure(figsize=(Width, Height))
    Axes = {k: [Fig.add_axes([b[0]/Width, b[1]/Height, b[2]/Width, b[3]/Height]) for b in Boxes[k]] for k in Kinds}
    Cmap = plt.get_cmap(C["density_cmap"])
    Norm = mcolors.BoundaryNorm(Style["density_levels"], Cmap.N)
    Maps = {}
    for i, Sys in enumerate(Systems):
        D, Smp = Grids[Sys], Samples[Sys]
        Axes["sampling"][i].set_title(Label(Style, Sys), fontsize=Style["font_size"] + 1)
        Maps[("sampling", i)] = SamplingPanel(Axes["sampling"][i], D, Smp, Style, Cmap, Norm)
        # a square inset of parity_inset_in inches, as fractions of this map
        w, h = Style["parity_inset_in"]/Boxes["sampling"][i][2], Style["parity_inset_in"]/Boxes["sampling"][i][3]
        Inset = EmptiestCorner(*Maps[("sampling", i)], Views[i], w, h)
        ParityInset(Axes["sampling"][i], Smp, Style, Inset, w, h)
        Maps[("tse", i)] = TSEPanel(Axes["tse"][i], D, Smp, Style, Cmap, Norm)
        if i == 0:
            InsetFirst = Inset
        Pair = PrimaryPair(R[R["System"] == Sys], Smp)
        Bins = ZBins(Smp, Pair, Style)
        ZDensityPanel(Axes["density"][i], Smp, Pair, Bins, Style)
        FreeEnergyPanel(Axes["free_energy"][i], D, Smp, Pair, Bins, Style)
        for k in ("density", "free_energy"):
            StationaryLines(Axes[k][i], D, Smp, Pair, Bins, Style, Names=(k == "density"))
        if len(S4Pairs(Smp)) > 1:
            Axes["density"][i].text(0.02, 0.95, PairName(Smp, Pair), transform=Axes["density"][i].transAxes,
                                    ha="left", va="top", fontsize=Style["font_size"] - 1, color=C["ink_secondary"])
    for k in Kinds:
        for i, Ax in enumerate(Axes[k]):
            if k in ("sampling", "tse"):
                View = Views[i]
                Ax.set_xlim(View[0], View[1])
                Ax.set_ylim(View[2], View[3])
                Ax.set_aspect("equal")
                Ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
                Ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
                Ax.tick_params(labelbottom=(k == "tse"))
                if i == 0:
                    Ax.set_ylabel("y (Å)")
            if k == "free_energy" and i == 0:
                Ax.set_ylabel("Free energy (k$_\\mathrm{B}$T)")
            if k == "density" and i == 0:
                Ax.set_ylabel("Sampling density")
    # one x label per row that needs it, centred under the row; the latent
    # names the axis of c and d, which share it
    Middle = (G["left"] + Width - G["right"])/2.0/Width
    for k, Text in (("tse", "x (Å)"), ("free_energy", L["latent"])):
        Fig.text(Middle, (Boxes[k][0][1] - G["axis_room"] + 0.04)/Height, Text, ha="center", va="bottom",
                 fontsize=Style["font_size"], color=C["ink"])
    # the panel letters at the top left of each row; a and b carry a short
    # caption inside their first map, c and d are named by their axes
    # (left of the y labels of the first column; the tight bounding box keeps them)
    for n, k in enumerate(Kinds):
        Fig.text(-0.15/Width, (Boxes[k][0][1] + Boxes[k][0][3])/Height, "abcd"[n], fontsize=Style["font_size"] + 2,
                 fontweight="bold", va="top")
    CornerCaption(Axes["sampling"][0], L["sampling"], *Maps[("sampling", 0)], Views[0], Style, Skip=InsetFirst)
    CornerCaption(Axes["tse"][0], L["kolmogorov"], *Maps[("tse", 0)], Views[0], Style)
    # one strip under the figure: the three sources by colour, then the density scale of a and b
    Fig.legend([plt.Line2D([], [], color=C[c], linewidth=1.5) for c in ("reference", "s4", "tse")],
               [L[c] for c in ("reference", "s4", "tse")], loc="center left",
               bbox_to_anchor=(G["left"]/Width, G["legend"]/2.0/Height), ncol=3, fontsize=Style["font_size"] - 1,
               handlelength=1.6, handletextpad=0.4, columnspacing=1.6, borderaxespad=0.0, borderpad=0.0)
    Bar = Fig.add_axes([(Width - G["right"] - 1.3)/Width, (G["legend"]/2.0 - 0.035)/Height, 1.3/Width, 0.07/Height])
    CB = Fig.colorbar(plt.cm.ScalarMappable(norm=Norm, cmap=Cmap), cax=Bar, orientation="horizontal")
    CB.set_ticks([])
    CB.outline.set_linewidth(0.4)
    Bar.text(-0.03, 0.5, "normalized density  0", transform=Bar.transAxes, ha="right", va="center",
             fontsize=Style["font_size"] - 1)
    Bar.text(1.03, 0.5, "1", transform=Bar.transAxes, ha="left", va="center", fontsize=Style["font_size"] - 1)
    Save(Fig, Style, OutDir, "fig_two_dimensional_sampling")

    return None

def TwoDimensionalLayout(Style, Ratios, Kinds):

    """

    Fixed geometry in inches, so every row keeps its size whatever its labels:
    the column titles, then per row the panels and the room their x axis
    needs (a small gap when the row below shares it), and the strip of the
    legend. Each column is as wide as its map, Ratios its width over its
    height, and all maps are equally tall.

    :return: Width, Height, Boxes: the figure size and, per row kind, the
             [left, bottom, width, height] of every panel in inches from the
             bottom left.

    """

    G = Style["two_dimensional_layout"]
    Width = Style["width_double_in"]
    Unit = (Width - G["left"] - G["right"] - (len(Ratios) - 1)*G["col_gap"])/sum(Ratios)
    Lefts = G["left"] + np.r_[0.0, np.cumsum([r*Unit + G["col_gap"] for r in Ratios])][:-1]
    Size = {"sampling": Unit, "tse": Unit, "density": G["profile_height"], "free_energy": G["profile_height"]}
    Below = {}
    for n, k in enumerate(Kinds):
        Next = Kinds[n + 1] if n + 1 < len(Kinds) else None
        if k == "sampling" and Next == "tse":
            Below[k] = G["gap"]
        elif k == "density" and Next == "free_energy":
            Below[k] = G["tick_room"]
        else:
            Below[k] = G["axis_room"]
    # the legend of the figure sits in the strip under the last row
    Height = G["title"] + sum(Size[k] + Below[k] for k in Kinds) + G["legend"]
    Boxes = {}
    y = Height - G["title"]
    for k in Kinds:
        y -= Size[k]
        Boxes[k] = [[Lefts[i], y, r*Unit, Size[k]] for i, r in enumerate(Ratios)]
        y -= Below[k]

    return Width, Height, Boxes

def LoadSampling(InDir, System):

    """

    :return: Data: sampling_<system>.npz of ts_compare.py, None when absent.

    """

    File = os.path.join(InDir, "sampling_%s.npz" % (System,))

    return np.load(File) if os.path.isfile(File) else None

def S4Pairs(Smp):

    """

    :return: Pairs: the pairs whose S4 sample was stored.

    """

    return [str(p) for p in Smp["Pairs"] if "X_%s" % (p,) in Smp.files]

def PrimaryPair(S, Smp):

    """

    :return: Pair: the pair of the primary channel when S4 ran on it, else the first S4 pair.

    """

    Pairs = S4Pairs(Smp)
    P = S[S["Primary"].astype(bool)]["Pair"]

    return P.iloc[0] if len(P) > 0 and P.iloc[0] in Pairs else Pairs[0]

def PairName(Smp, Pair):

    """

    :return: Text: the pair named by its exact minima, e.g. 'A → I'.

    """

    return " → ".join(str(Smp["Ref_" + Pair]).split("_")[1:])

def PESLines(Ax, D, Style):

    """

    :return: None; the potential above E(A) as contour lines, over a density.

    """

    V = D["V"] - float(D["EA"])
    EMax = Style["pes_emax_eV"] or float(np.max(D["E"]) - D["EA"])*1.6
    Levels = np.linspace(min(0.0, float(V.min())), EMax, Style["pes_levels"])
    Ax.contour(D["x"], D["y"], np.clip(V, None, EMax).T, levels=Levels[::3], colors=Style["colors"]["ink_secondary"],
               linewidths=0.3, alpha=0.5, zorder=3)

    return None

def ExactIsolines(Ax, D, Style):

    """

    :return: None; the exact q = 1/2 isoline of every pair, over the drawn potential only.

    """

    if Style["show_committor"]:
        EMax = Style["pes_emax_eV"] or float(np.max(D["E"]) - D["EA"])*1.6
        Far = D["V"] - float(D["EA"]) > EMax
        for k in [k for k in D.files if k.startswith("q_")]:
            Ax.contour(D["x"], D["y"], np.ma.masked_where(Far, D[k]).T, levels=[0.5], colors=Style["colors"]["reference"],
                       linewidths=0.8, linestyles="--", zorder=4)

    return None

def ExactSaddles(Ax, D, Style):

    """

    :return: None; the exact saddles as crosses, ringed in white so the grey
             stays visible on the dark top of a density.

    """

    IsSaddle = np.char.startswith(D["Labels"].astype(str), "S")
    Ax.plot(D["Points"][IsSaddle, 0], D["Points"][IsSaddle, 1], linestyle="none", marker=Style["markers"]["reference"],
            color=Style["colors"]["reference"], markersize=Style["marker_size"] + 1, markeredgewidth=1.2, zorder=6,
            path_effects=[pe.Stroke(linewidth=3.0, foreground="white"), pe.Normal()])

    return None

def SamplingPanel(Ax, D, Smp, Style, Cmap, Norm):

    """

    :return: xc, yc, H: bin centres and the density of every S4 walk,
             normalised to its maximum, drawn under the contour lines of the potential.

    """

    from scipy.ndimage import gaussian_filter

    X = np.concatenate([Smp["X_" + p] for p in S4Pairs(Smp)])
    H, xe, ye = np.histogram2d(X[:, 0], X[:, 1], bins=Style["density_bins"],
                               range=[[D["x"][0], D["x"][-1]], [D["y"][0], D["y"][-1]]])
    H = gaussian_filter(H, Style["density_smooth"])
    H = H/H.max()
    xc, yc = 0.5*(xe[1:] + xe[:-1]), 0.5*(ye[1:] + ye[:-1])
    Ax.contourf(xc, yc, H.T, levels=Style["density_levels"], cmap=Cmap, norm=Norm, zorder=1)
    PESLines(Ax, D, Style)

    return xc, yc, H

def EmptiestCorner(xc, yc, H, View, w, h, Skip=None):

    """

    :param1 xc, yc, H: bin centres and density drawn on the map.
    :param2 View:      x0, x1, y0, y1 of the map.
    :param3 w, h:      size of the box as fractions of the map.
    :param4 Skip:      a corner already taken, or None.

    :return: Corner: the corner whose box holds the least density; upper
             right on a tie, as in the committor_2.0 figures.

    """

    x0, x1, y0, y1 = View
    Best, Least = None, np.inf
    for Corner in [c for c in ("upper right", "upper left", "lower right", "lower left") if c != Skip]:
        xa = (x1 - w*(x1 - x0), x1) if "right" in Corner else (x0, x0 + w*(x1 - x0))
        ya = (y1 - h*(y1 - y0), y1) if "upper" in Corner else (y0, y0 + h*(y1 - y0))
        Mass = H[np.ix_((xc >= xa[0]) & (xc <= xa[1]), (yc >= ya[0]) & (yc <= ya[1]))].sum()
        if Mass < Least - 1.0e-9:
            Best, Least = Corner, Mass

    return Best

def CornerCaption(Ax, Text, xc, yc, H, View, Style, Skip=None):

    """

    :param1 Text: the caption, a few words.
    :param2 Skip: a corner already taken, or None.

    :return: None; the caption inside the map, in the corner whose box,
             measured on the drawn text, covers the least density.

    """

    Pad = 0.03
    T = Ax.text(0.0, 0.0, Text, transform=Ax.transAxes, fontsize=Style["font_size"], color=Style["colors"]["ink"],
                zorder=8, bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8))
    Box = T.get_window_extent(Ax.figure.canvas.get_renderer()).transformed(Ax.transAxes.inverted())
    Corner = EmptiestCorner(xc, yc, H, View, Box.width + 2.0*Pad, Box.height + 2.0*Pad, Skip=Skip)
    T.set_position((1.0 - Pad if "right" in Corner else Pad, 1.0 - Pad if "upper" in Corner else Pad))
    T.set_horizontalalignment("right" if "right" in Corner else "left")
    T.set_verticalalignment("top" if "upper" in Corner else "bottom")

    return None

def ParityInset(Ax, Smp, Style, Corner, w, h):

    """

    :param1 Corner: 'upper right', 'upper left', 'lower right' or 'lower left'.
    :param2 w, h:   size of the inset as fractions of the map.

    :return: None; the learned against the exact committor at the S4 walks of
             every pair, framed, flush in one corner of the map, with the line
             x = y over the points and the labels inside the frame.

    """

    C = Style["colors"]
    L = Style["sampling_labels"]
    Pairs = S4Pairs(Smp)
    qe = np.concatenate([Smp["qExact_" + p] for p in Pairs])
    qf = np.concatenate([Smp["qFit_" + p] for p in Pairs])
    Ok = np.flatnonzero(np.isfinite(qe))
    Pick = np.random.default_rng(0).choice(Ok, size=min(Style["parity_points"], len(Ok)), replace=False)
    In = Ax.inset_axes([0.0 if "left" in Corner else 1.0 - w, 0.0 if "lower" in Corner else 1.0 - h, w, h], zorder=7)
    In.set_facecolor("white")
    In.scatter(qe[Pick], qf[Pick], s=1.5, color=C["s4"], alpha=0.4, linewidths=0, rasterized=True, zorder=2)
    In.plot([0, 1], [0, 1], color=C["ink"], linewidth=0.9, linestyle="--", zorder=3)
    In.set_xlim(-0.02, 1.02)
    In.set_ylim(-0.02, 1.02)
    In.set_xticks([])
    In.set_yticks([])
    for s in In.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color(C["ink_secondary"])
    Small = Style["font_size"] - 2
    In.text(0.6, 0.04, L["reference_q"], transform=In.transAxes, ha="center", va="bottom", fontsize=Small,
            color=C["ink"])
    In.text(0.04, 0.6, L["learned_q"], transform=In.transAxes, rotation=90, ha="left", va="center",
            fontsize=Small, color=C["ink"])

    return None

def TSEPanel(Ax, D, Smp, Style, Cmap, Norm):

    """

    :return: x, y, Used: the grid and where the map is drawn on, the
             Kolmogorov density or a q = 1/2 isoline, for placing a caption;
             draws the exact Kolmogorov distribution of every pair, each
             normalised to its maximum, the exact and the learned q = 1/2
             isolines, the exact saddles and the S6 midpoints, resampled by
             their weights.

    """

    C = Style["colors"]
    pK = [D[k]/D[k].max() for k in D.files if k.startswith("pK_")]
    Used = np.max(pK, axis=0) if pK else np.zeros_like(D["V"])
    if pK:
        Ax.contourf(D["x"], D["y"], Used.T, levels=Style["density_levels"], cmap=Cmap, norm=Norm, zorder=1)
    PESLines(Ax, D, Style)
    ExactIsolines(Ax, D, Style)
    # the learned committor only where the potential is drawn; beyond, no walk informed it
    EMax = Style["pes_emax_eV"] or float(np.max(D["E"]) - D["EA"])*1.6
    Far = D["V"] - float(D["EA"]) >= EMax
    for k in [k for k in D.files if k.startswith("q_")]:
        Used = np.maximum(Used, (np.abs(D[k] - 0.5) < 0.2) & ~Far)
    for p in S4Pairs(Smp):
        Ax.contour(D["x"], D["y"], np.ma.masked_where(Far, Smp["qFitGrid_" + p]).T, levels=[0.5], colors=C["s4"],
                   linewidths=1.0, zorder=5)
    Rng = np.random.default_rng(0)
    for k in sorted(k for k in Smp.files if k.startswith("TSEX_")):
        X, W = Smp[k], Smp["TSEW_" + k[5:]]
        Pick = Rng.choice(len(X), size=min(Style["tse_points"], len(X)), replace=False, p=W)
        Ax.scatter(X[Pick, 0], X[Pick, 1], s=2.0, color=C["tse"], alpha=0.6, linewidths=0, zorder=6, rasterized=True)
    ExactSaddles(Ax, D, Style)

    return D["x"], D["y"], Used

def ZBins(Smp, Pair, Style):

    """

    :return: Bins: edges along the learned latent z covering the S4 walks of a pair.

    """

    z = Smp["zFit_" + Pair]
    Lo, Hi = np.percentile(z, [0.5, 99.5])
    Pad = 0.05*(Hi - Lo)

    return np.linspace(Lo - Pad, Hi + Pad, Style["z_bins"] + 1)

def ZDensityPanel(Ax, Smp, Pair, Bins, Style):

    """

    :return: None; along z of one pair, the densities of the unbiased basin
             walks, of the biased walks and of the S6 midpoints (weighted,
             channels pooled), each normalised to unit area.

    """

    C = Style["colors"]
    z, Block = Smp["zFit_" + Pair], Smp["Block_" + Pair]
    Series = [("reference", z[Block == 0], None), ("s4", z[Block > 0], None)]
    Keys = sorted(k for k in Smp.files if k.startswith("TSEzFit_%s_ch" % (Pair,)))
    if Keys:
        Series.append(("tse", np.concatenate([Smp[k] for k in Keys]),
                       np.concatenate([Smp["TSEW_" + k[8:]]/len(Keys) for k in Keys])))
    Centres = 0.5*(Bins[1:] + Bins[:-1])
    for Key, v, w in Series:
        h, _ = np.histogram(v, bins=Bins, weights=w, density=True)
        Ax.fill_between(Centres, h, color=C[Key], alpha=0.45, linewidth=0)
        Ax.plot(Centres, h, color=C[Key], linewidth=0.8)
    Ax.set_xlim(Bins[0], Bins[-1])
    Ax.set_ylim(0, None)
    Ax.tick_params(labelbottom=True)

    return None

def FreeEnergy(h):

    """

    :return: F: -ln h in kT, zero at its minimum, NaN where h is empty.

    """

    F = np.full(len(h), np.nan)
    F[h > 0] = -np.log(h[h > 0])

    return F - np.nanmin(F)

def FreeEnergyPanel(Ax, D, Smp, Pair, Bins, Style):

    """

    :return: None; the free energy along z of one pair: exact, from exp(-V/kT)
             on the grid, and from the S4 walks with their MBAR weights; the
             stretch of z each S6 ensemble covers is laid over the exact curve,
             since a slab across the surface gives no profile of its own.

    """

    C = Style["colors"]
    Centres = 0.5*(Bins[1:] + Bins[:-1])
    Boltzmann = np.exp(-(D["V"] - D["V"].min())/float(D["kT"]))
    FExact = FreeEnergy(np.histogram(Smp["zFitGrid_" + Pair].ravel(), bins=Bins, weights=Boltzmann.ravel())[0])
    FS4 = FreeEnergy(np.histogram(Smp["zFit_" + Pair], bins=Bins, weights=Smp["W_" + Pair])[0])
    Ax.plot(Centres, FExact, color=C["reference"], linewidth=1.2, linestyle=":", zorder=3)
    Ax.plot(Centres, FS4, color=C["s4"], linewidth=1.2, zorder=2)
    Ok = np.isfinite(FExact)
    for k in sorted(k for k in Smp.files if k.startswith("TSEzFit_%s_ch" % (Pair,))):
        z = np.linspace(*WeightedRange(Smp[k], Smp["TSEW_" + k[8:]], *Style["tse_range"]), 50)
        Ax.plot(z, np.interp(z, Centres[Ok], FExact[Ok]), color=C["tse"], linewidth=5, alpha=0.4,
                solid_capstyle="butt", zorder=1)
    Ax.set_xlim(Bins[0], Bins[-1])
    # scaled to the bins both cover; an edge bin of the grid alone can sit far up
    Both = np.isfinite(FExact) & np.isfinite(FS4)
    Ax.set_ylim(0, 1.2*np.max(np.r_[FExact[Both], FS4[Both]]))

    return None

def WeightedRange(v, w, Lo, Hi):

    """

    :return: a, b: the Lo and Hi quantiles of v under the weights w.

    """

    Order = np.argsort(v)
    Cum = np.cumsum(w[Order])/np.sum(w)

    return v[Order][np.searchsorted(Cum, Lo)], v[Order][min(np.searchsorted(Cum, Hi), len(v) - 1)]

def StationaryLines(Ax, D, Smp, Pair, Bins, Style, Names=False):

    """

    :param1 Names: name the lines on the x axis instead of the z values;
                   points closer than 6 % of the range share one name.

    :return: None; a dashed line at the learned z of every stationary point inside the range.

    """

    z = Smp["zFitPoints_" + Pair]
    Labels = D["Labels"].astype(str)
    In = np.flatnonzero((z >= Bins[0]) & (z <= Bins[-1]))
    for i in In:
        Ax.axvline(z[i], color=Style["colors"]["ink_secondary"], linewidth=0.6, linestyle="--", zorder=1)
    if Names:
        Groups = []
        for i in In[np.argsort(z[In])]:
            if Groups and z[i] - Groups[-1][-1][0] < 0.06*(Bins[-1] - Bins[0]):
                Groups[-1].append((z[i], Labels[i]))
            else:
                Groups.append([(z[i], Labels[i])])
        Ax.set_xticks([np.mean([g[0] for g in G]) for G in Groups])
        Ax.set_xticklabels(["/".join(g[1] for g in G) for G in Groups])
        Ax.tick_params(axis="x", length=0)

    return None

#%% 4. Gas-phase reactions
def PlotGasPhase(Rows, Style, OutDir, Class="gas_phase", Name="fig_gas_phase"):

    """

    Dot plot per reaction, as Fig. 4 of the SACTS paper: RMSD (log scale) and
    signed dE against the reference, every run and candidate one marker.

    :return: None.

    """

    R = Selected(Rows, Style, Class)
    if R.empty:
        return None
    C = Style["colors"]
    Order = [s for s in Style["system_order"] if s in set(R["System"])] + \
            sorted(set(R["System"]) - set(Style["system_order"]))
    y = {s: i for i, s in enumerate(Order[::-1])}
    Cands = Style["candidates"]
    Off = {c: (k - (len(Cands) - 1)/2.0)*0.22 for k, c in enumerate(Cands)}
    Fig, (A1, A2) = plt.subplots(1, 2, sharey=True, figsize=(Style["width_double_in"], 0.32*len(Order) + 1.0))
    if Style["show_kT"]:
        A2.axvspan(-Style["kT_eV"], Style["kT_eV"], color=C["band"], zorder=0, label="± kT (298 K)")
    for v in (-Style["threshold_dE_eV"], Style["threshold_dE_eV"]):
        A2.axvline(v, color=C["ink_secondary"], linewidth=0.6, linestyle=":")
    A2.axvline(0.0, color=C["ink_secondary"], linewidth=0.6)
    A1.axvline(Style["threshold_rmsd_A"], color=C["ink_secondary"], linewidth=0.6, linestyle=":")
    for Cand in Cands:
        P = R[R["Candidate"] == Cand]
        Y = np.array([y[s] for s in P["System"]]) + Off[Cand]
        Kw = dict(linestyle="none", marker=Style["markers"][Cand], color=C[Cand], markersize=Style["marker_size"],
                  markeredgecolor="white", markeredgewidth=0.5, label=Style["candidate_labels"][Cand])
        A1.plot(np.maximum(P[Style["rmsd_column"]], 1e-4), Y, **Kw)
        A2.plot(P["dE_eV"], Y, **Kw)
    A1.set_xscale("log")
    A1.set_yticks(list(y.values()))
    A1.set_yticklabels([Label(Style, s) for s in y])
    A1.set_xlabel("RMSD (Å)")
    A2.set_xlabel("ΔE = E(tReax) − E(reference) (eV)")
    Refs = ", ".join(sorted(set(R["Reference"])))
    A1.set_title("Structure vs %s" % (Refs,), fontsize=Style["font_size"] + 1, loc="left")
    A2.set_title("Energy vs %s (dotted: ±%s)" % (Refs, Style["threshold_dE_label"]),
                 fontsize=Style["font_size"] + 1, loc="left")
    Lim = max(1.2*Style["threshold_dE_eV"], 1.15*float(np.nanmax(np.abs(R["dE_eV"]))))
    A2.set_xlim(-Lim, Lim)
    for Ax in (A1, A2):
        Grid(Ax, Style)
    Handles, Labels = A2.get_legend_handles_labels()
    Fig.tight_layout()
    Fig.legend(Handles, Labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(Labels),
               fontsize=Style["font_size"] - 1)
    Save(Fig, Style, OutDir, Name)

    return None

def PlotProfiles(Rows, Chains, Style, OutDir):

    """

    Small multiples of the chain energy profiles of the gas-phase reactions,
    the reference saddle as a dashed level and the crossing as a marker.

    :return: None.

    """

    R = Selected(Rows, Style, "gas_phase")
    if R.empty:
        return None
    C = Style["colors"]
    Order = [s for s in Style["system_order"] if s in set(R["System"])] + \
            sorted(set(R["System"]) - set(Style["system_order"]))
    NC = Style["profile_columns"]
    NR = int(np.ceil(len(Order)/NC))
    Fig, Axes = plt.subplots(NR, NC, figsize=(Style["width_double_in"], 1.7*NR), squeeze=False)
    for Ax, Sys in zip(Axes.flat, Order):
        S = R[R["System"] == Sys]
        for _, Row in S.drop_duplicates(["Run", "Pair", "Channel"]).iterrows():
            G = Chains[(Chains["System"] == Sys) & (Chains["Run"] == Row["Run"]) &
                       (Chains["Pair"] == Row["Pair"]) & (Chains["Channel"] == Row["Channel"])]
            EA = float((G["E_eV"] - G["ERelA_eV"]).iloc[0])
            Ax.plot(G["Xi"], G["ERelA_eV"], color=C["chain"], linewidth=Style["line_width"], marker=".", markersize=2)
            Ax.axhline(Row["ERef_eV"] - EA, color=C["ink"], linewidth=0.7, linestyle="--")
            for Cand in Style["candidates"]:
                P = S[(S["Candidate"] == Cand) & (S["Run"] == Row["Run"]) & (S["Pair"] == Row["Pair"]) &
                      (S["Channel"] == Row["Channel"])]
                if not P.empty:
                    Ax.plot(P["Xi"].iloc[0], P["E_eV"].iloc[0] - EA, linestyle="none",
                            marker=Style["markers"][Cand], color=C[Cand], markersize=Style["marker_size"],
                            markeredgecolor="white", markeredgewidth=0.5, zorder=4,
                            label=Style["candidate_labels"][Cand])
        Ax.set_title(Label(Style, Sys), fontsize=Style["font_size"])
        Grid(Ax, Style, "y")
    for Ax in Axes.flat[len(Order):]:
        Ax.axis("off")
    for Ax in Axes[:, 0]:
        Ax.set_ylabel("E − E(A) (eV)")
    for Ax in Axes[-1, :]:
        Ax.set_xlabel("ξ")
    Handles, Labels = Axes.flat[0].get_legend_handles_labels()
    n = len(Style["candidates"])
    Handles = Handles[:n] + [plt.Line2D([], [], color=C["ink"], linewidth=0.7, linestyle="--")]
    Labels = Labels[:n] + ["reference"]
    Fig.tight_layout()
    Fig.legend(Handles, Labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=n + 1,
               fontsize=Style["font_size"] - 1)
    Save(Fig, Style, OutDir, "fig_gas_phase_profiles")

    return None

#%% 5. Gas-phase structures
def KeyGeometry(System, Style):

    """

    Internal coordinates to annotate: schematic_geometry[System] when given
    (atom-index tuples: 2 distance, 3 angle, 4 dihedral), else every pair whose
    bonding differs between reactant.xyz and product.xyz of the reaction input.

    :return: Key: list of index tuples, in the reference atom order.

    """

    if System in Style["schematic_geometry"]:
        return [tuple(int(i) for i in k) for k in Style["schematic_geometry"][System]]
    Files = [os.path.join(INPUT_ROOT, System, f) for f in ("reactant.xyz", "product.xyz")]
    if not all(os.path.isfile(f) for f in Files):
        return []
    Bonded = []
    for f in Files:
        A = asio.read(f)
        r = covalent_radii[A.numbers]
        Bonded.append(A.get_all_distances() < Style["schematic_change_factor"]*(r[:, None] + r[None, :]))
    i, j = np.nonzero(np.triu(Bonded[0] != Bonded[1], 1))

    return [(int(a), int(b)) for a, b in zip(i, j)]

def Measure(X, Key):

    """

    :return: Text: the distance (Å), angle or dihedral (deg) of Key in X.

    """

    if len(Key) == 2:
        return "%.3f" % (np.linalg.norm(X[Key[0]] - X[Key[1]]),)
    if len(Key) == 3:
        u, v = X[Key[0]] - X[Key[1]], X[Key[2]] - X[Key[1]]
        return "%.2f°" % (np.degrees(np.arccos(np.clip(np.dot(u, v)/np.linalg.norm(u)/np.linalg.norm(v), -1.0, 1.0))),)

    return "%.2f°" % (Dihedral(X, Key),)

def Rotation(Angles):

    """

    :return: Rot: turns about the view x, y and z by Angles (deg), in that
             order, as a right factor of the view matrix.

    """

    Rot = np.eye(3)
    for (p, q), a in zip([(1, 2), (2, 0), (0, 1)], np.radians(Angles)):
        R = np.eye(3)
        R[p, p], R[p, q], R[q, p], R[q, q] = np.cos(a), -np.sin(a), np.sin(a), np.cos(a)
        Rot = Rot @ R.T

    return Rot

def ViewFrame(X, Rad, Key, Angles=None):

    """

    Principal axes of X as the view, largest spread along x and smallest along
    the line of sight z. Without Angles, the turn about x and y (up to 60 deg)
    that least hides atoms and least foreshortens the key distances is taken,
    then the view is turned in its plane to lie widest along x.

    :param1 X:      positions, shape (N, 3).
    :param2 Rad:    drawn sphere radii.
    :param3 Key:    index tuples from KeyGeometry.
    :param4 Angles: turns about x, y, z (deg) after the principal axes, or None.

    :return: Center, M: Y = (X - Center) @ M has z toward the viewer.

    """

    Center = X.mean(axis=0)
    Y0 = X - Center
    M = np.linalg.eigh(Y0.T @ Y0)[1][:, ::-1]
    # the sign of each axis follows the skew of the projections; z keeps it right-handed
    M = M*np.where(np.sum((Y0 @ M)**3, axis=0) < 0, -1.0, 1.0)
    M[:, 2] = np.cross(M[:, 0], M[:, 1])
    if Angles is not None:
        return Center, M @ Rotation(Angles)
    Upper = np.triu_indices(len(X), 1)
    Pairs = [k for k in Key if len(k) == 2]
    Best = (np.inf, M)
    for ax in range(-60, 61, 10):
        for ay in range(-60, 61, 10):
            Mt = M @ Rotation([ax, ay, 0.0])
            Y = Y0 @ Mt
            D2 = np.linalg.norm(Y[:, None, :2] - Y[None, :, :2], axis=2)
            Hide = np.sum(np.maximum(Rad[:, None] + Rad[None, :] - D2, 0.0)[Upper]**2)
            Short = sum(1.0 - D2[i, j]/np.linalg.norm(Y0[i] - Y0[j]) for i, j in Pairs)
            Score = Hide + 0.1*Short + 0.01*(ax**2 + ay**2)/3600.0
            Best = min(Best, (Score, Mt), key=lambda b: b[0])
    Mt = Best[1]
    Y = (Y0 @ Mt)[:, :2]
    v = np.linalg.eigh(Y.T @ Y)[1][:, -1]

    return Center, Mt @ Rotation([0.0, 0.0, -np.degrees(np.arctan2(v[1], v[0]))])

def AtomRadii(Symbols, Style):

    """

    :return: Rad, Cov: drawn sphere radii a + b*r_cov and covalent radii (Å).

    """

    a, b = Style["schematic_atom_radius"]
    Cov = covalent_radii[[atomic_numbers[s] for s in Symbols]]

    return a + b*Cov, Cov

def DrawSphere(Ax, xy, r, Color, z):

    """

    :return: None; a shaded sphere as concentric discs lit from the upper left.

    """

    Base = np.array(mcolors.to_rgb(Color))
    Dark, Light = 0.62*Base, Base + 0.75*(1.0 - Base)
    for t in np.linspace(0.0, 1.0, 24):
        Col = Dark + (Base - Dark)*min(t/0.5, 1.0) + (Light - Base)*max((t - 0.5)/0.5, 0.0)**2
        Ax.add_patch(Circle(xy + 0.28*r*t*np.array([-1.0, 1.0]), r*(1.0 - 0.93*t), facecolor=Col,
                            edgecolor="none", zorder=z))

    return None

def DrawBond(Ax, Pi, Pj, ri, rj, Ci, Cj, w, Order=1):

    """

    :return: None; Order parallel sticks between the sphere surfaces, each half
             in the colour of its atom, drawn at the mean depth of the two atoms.

    """

    u = (Pj - Pi)/np.linalg.norm(Pj - Pi)
    a, m, b = (Pi + ri*u)[:2], ((Pi + Pj)/2.0)[:2], (Pj - rj*u)[:2]
    d = b - a
    if np.dot(d, u[:2]) <= 0.0 or np.linalg.norm(d) < 1e-6:
        return None
    n = np.array([-d[1], d[0]])/np.linalg.norm(d)
    # the highlight sits on the side facing the light (upper left)
    n = n if n[1] - n[0] >= 0.0 else -n
    z = 10.0 + (Pi[2] + Pj[2])/2.0
    h, Shifts = {1: (w, [0.0]), 2: (0.55*w, [-0.9*w, 0.9*w]), 3: (0.45*w, [-1.2*w, 0.0, 1.2*w])}[int(Order)]
    for s in Shifts:
        for p, q, Color in ((a, m, Ci), (m, b, Cj)):
            Base = np.array(mcolors.to_rgb(Color))
            # a shaded cylinder: nested strips, each lighter and nearer the light
            for f, o, Col in ((1.0, 0.0, 0.62*Base), (0.75, 0.12, Base), (0.4, 0.25, Base + 0.6*(1.0 - Base))):
                c = (s + o*h)*n
                Ax.add_patch(Polygon([p + c + f*h*n, q + c + f*h*n, q + c - f*h*n, p + c - f*h*n], closed=True,
                                     facecolor=Col, edgecolor="none", zorder=z))

    return None

def LabelSpot(Y, K, Spot):

    """

    :param1 Y:    view-frame positions, shape (N, 3).
    :param2 K:    key index tuple.
    :param3 Spot: (t, s, pt): fraction along the bond, side (+1/-1), offset (pt).

    :return: Anchor, Dir: label anchor (Å) and unit offset direction; an angle
             is anchored at its vertex, offset along its bisector.

    """

    t, s, _ = Spot
    if len(K) == 3:
        Anchor = Y[K[1], :2]
        Dir = sum((Y[k, :2] - Anchor)/np.linalg.norm(Y[k, :2] - Anchor) for k in (K[0], K[2]))
    else:
        i, j = (K[0], K[1]) if len(K) == 2 else (K[1], K[2])
        Anchor = Y[i, :2] + t*(Y[j, :2] - Y[i, :2])
        Dir = np.array([Y[i, 1] - Y[j, 1], Y[j, 0] - Y[i, 0]])
    Dir = s*Dir/np.linalg.norm(Dir) if np.linalg.norm(Dir) > 1e-9 else np.array([0.0, float(s)])

    return Anchor, Dir

def AddLabel(Ax, Y, K, Spot, Style):

    """

    :return: Text: the value of K as an annotation at Spot.

    """

    Anchor, Dir = LabelSpot(Y, K, Spot)

    return Ax.annotate(Measure(Y, K).replace("-", "−"), Anchor, xytext=Spot[2]*Dir, textcoords="offset points",
                       color=Style["schematic_label_color"], fontsize=Style["font_size"] - 1, zorder=50,
                       ha="left" if Dir[0] > 0.35 else ("right" if Dir[0] < -0.35 else "center"),
                       va="bottom" if Dir[1] > 0.35 else ("top" if Dir[1] < -0.35 else "center"))

def LabelPenalty(Box, Disks, Segments, Boxes, Frame):

    """

    :param1 Box:      extent of the label, display units.
    :param2 Disks:    atom centres and radii, shape (N, 3), display units.
    :param3 Segments: sticks and dashed lines, shape (M, 2, 2), display units.
    :param4 Boxes:    extents of the labels already placed.
    :param5 Frame:    extent of the axes.

    :return: Penalty: overlap with atoms, sticks, other labels and the frame edge.

    """

    x0, y0, x1, y1 = Box.x0, Box.y0, Box.x1, Box.y1
    Zero = np.zeros(len(Disks))
    d = np.hypot(np.maximum.reduce([x0 - Disks[:, 0], Zero, Disks[:, 0] - x1]),
                 np.maximum.reduce([y0 - Disks[:, 1], Zero, Disks[:, 1] - y1]))
    Pen = float(np.sum(np.clip(1.0 - d/Disks[:, 2], 0.0, None)))
    if len(Segments):
        P = Segments[:, :1] + np.linspace(0.0, 1.0, 16)[None, :, None]*(Segments[:, 1:] - Segments[:, :1])
        Pen += 2.0*float(np.sum(np.mean((P[..., 0] > x0) & (P[..., 0] < x1) & (P[..., 1] > y0) &
                                        (P[..., 1] < y1), axis=1)))
    Area = max(Box.width*Box.height, 1e-9)
    Cut = lambda B: max(min(x1, B.x1) - max(x0, B.x0), 0.0)*max(min(y1, B.y1) - max(y0, B.y0), 0.0)
    Pen += 3.0*sum(Cut(B) for B in Boxes)/Area + 2.0*(1.0 - Cut(Frame)/Area)

    return Pen

def BondTable(Y, Symbols, Style):

    """

    :return: Bonds: per atom pair 0 (no bond) or the drawn bond order; between
             heavy atoms the Pauling order exp((r_i + r_j - d)/0.3) rounded to
             1-3 when schematic_bond_orders, else 1.

    """

    Cov = AtomRadii(Symbols, Style)[1]
    Single = Cov[:, None] + Cov[None, :]
    D = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2)
    Bonds = (D < Style["schematic_bond_factor"]*Single).astype(int)
    np.fill_diagonal(Bonds, 0)
    if Style["schematic_bond_orders"]:
        n = np.exp((Single - D)/0.3)
        Heavy = np.array([s != "H" for s in Symbols])
        Pair = (Bonds > 0) & Heavy[:, None] & Heavy[None, :]
        Bonds[Pair] = np.where(n >= 2.5, 3, np.where(n >= 1.5, 2, 1))[Pair]

    return Bonds

def DrawMolecule(Ax, Y, Symbols, Key, Style, Layout=None):

    """

    Ball-and-stick drawing in the view frame: the bonds of the structure as
    sticks (multiple bonds as parallel sticks), the key coordinates that are not
    bonds as dashed lines between the atom centres, each key labelled with its
    value where it least covers atoms, sticks and the other labels.

    :param1 Ax:      axes in Å with equal aspect, limits already set.
    :param2 Y:       view-frame positions, shape (N, 3), z toward the viewer.
    :param3 Symbols: chemical symbols.
    :param4 Key:     index tuples from KeyGeometry.
    :param5 Style:   style dict.
    :param6 Layout:  (Bonds, Spots) of an earlier call, so both rows share the
                     drawn bonds and the label spots; None to choose them here.

    :return: Layout: the bond table and the label spot of every key.

    """

    Palette = Style["schematic_atom_colors"]
    Colors = [Palette.get(s, Palette["other"]) for s in Symbols]
    Rad = AtomRadii(Symbols, Style)[0]
    Bonds, Spots = Layout or (BondTable(Y, Symbols, Style), None)
    Red = Style["schematic_label_color"]
    Segments = []
    for i, j in zip(*np.nonzero(np.triu(Bonds, 1))):
        DrawBond(Ax, Y[i], Y[j], Rad[i], Rad[j], Colors[i], Colors[j], Style["schematic_bond_radius_A"], Bonds[i, j])
        Segments.append([Y[i, :2], Y[j, :2]])
    for i in range(len(Y)):
        DrawSphere(Ax, Y[i, :2], Rad[i], Colors[i], 10.0 + Y[i, 2])
    for K in Key:
        if len(K) == 2 and not Bonds[K[0], K[1]]:
            p, q = Y[K[0], :2], Y[K[1], :2]
            Ax.plot([p[0], q[0]], [p[1], q[1]], color=Red, linewidth=0.7, linestyle=(0, (2.2, 1.6)), zorder=40)
            Segments.append([p, q])
        elif len(K) == 3:
            Th = [np.degrees(np.arctan2(*(Y[k, 1::-1] - Y[K[1], 1::-1]))) for k in (K[0], K[2])]
            Lo, Hi = (Th[0], Th[1]) if (Th[1] - Th[0]) % 360.0 <= 180.0 else (Th[1], Th[0])
            Ax.add_patch(matplotlib.patches.Arc(Y[K[1], :2], 0.6, 0.6, theta1=Lo, theta2=Hi, color=Red,
                                                linewidth=0.8, zorder=40))
    if Spots is None:
        Ax.apply_aspect()
        Renderer = Ax.figure.canvas.get_renderer()
        T = Ax.transData
        Px = T.transform([[1.0, 0.0]])[0, 0] - T.transform([[0.0, 0.0]])[0, 0]
        Disks = np.column_stack([T.transform(Y[:, :2]), Rad*Px])
        Segs = T.transform(np.reshape(Segments, (-1, 2))).reshape(-1, 2, 2) if Segments else np.zeros((0, 2, 2))
        Center = Y[:, :2].mean(axis=0)
        Spots, Boxes = [], []
        for K in Key:
            Options = [(t, s, pt) for t in ((0.5,) if len(K) == 3 else (0.5, 0.35, 0.65))
                       for s in (1, -1) for pt in ((9.0, 13.0) if len(K) == 3 else (2.5, 6.0))]
            Scores = []
            for Spot in Options:
                Text = AddLabel(Ax, Y, K, Spot, Style)
                Box = Text.get_window_extent(Renderer)
                Text.remove()
                Anchor, Dir = LabelSpot(Y, K, Spot)
                # prefer the bond middle, the short offset and the side away from the molecule
                Prior = 0.1*abs(Spot[0] - 0.5)/0.15 + 0.05*(Spot[2] > 6.0 if len(K) != 3 else Spot[2] > 9.0) + \
                        0.02*(np.dot(Anchor - Center, Dir) < 0.0)
                Scores.append(LabelPenalty(Box, Disks, Segs, Boxes, Ax.bbox) + Prior)
            Spots.append(Options[int(np.argmin(Scores))])
            Boxes.append(AddLabel(Ax, Y, K, Spots[-1], Style).get_window_extent(Renderer))
    else:
        for K, Spot in zip(Key, Spots):
            AddLabel(Ax, Y, K, Spot, Style)

    return Bonds, Spots

def PlotGasPhaseSchematic(Rows, InDir, Style, OutDir):

    """

    Ball-and-stick panels of the gas-phase reactions: the reference saddle on
    top, the located TS below in the same view, the forming and breaking bonds
    labelled with their lengths (Å).

    :return: None.

    """

    R = Selected(Rows, Style, "gas_phase")
    Cand = Style["schematic_candidate"]
    R = R[R["Candidate"] == Cand]
    if R.empty:
        return None
    C = Style["colors"]
    Order = [s for s in Style["system_order"] if s in set(R["System"])] + \
            sorted(set(R["System"]) - set(Style["system_order"]))
    Panels = []
    for Sys in Order:
        S = R[R["System"] == Sys].sort_values(Style["rmsd_column"])
        Row = S.iloc[len(S)//2]  # the run of median RMSD
        Frames = asio.read(os.path.join(InDir, Row["Structure"]), index=":")
        Symbols = Frames[0].get_chemical_symbols()
        Key = KeyGeometry(Sys, Style)
        Rad = AtomRadii(Symbols, Style)[0]
        Center, M = ViewFrame(Frames[0].get_positions(), Rad, Key, Style["schematic_view"].get(Sys))
        Ys = [(F.get_positions() - Center) @ M for F in Frames[:2]]
        Lo = np.min([Y[:, :2] - Rad[:, None] for Y in Ys], axis=(0, 1))
        Hi = np.max([Y[:, :2] + Rad[:, None] for Y in Ys], axis=(0, 1))
        Panels.append({"System": Sys, "Symbols": Symbols, "Ys": Ys, "Key": Key,
                       "Mid": (Lo + Hi)/2.0, "Size": Hi - Lo + 2.0*Style["schematic_margin_A"]})
    Sizes = np.array([P["Size"] for P in Panels])
    NC = min(Style["schematic_columns"], len(Panels))
    NB = int(np.ceil(len(Panels)/NC))
    W = Style["width_double_in"]
    Left, Title, Gap = 0.75, 0.34, 0.10
    cw = (W - Left)/NC
    # cell height/width per block; on a common scale every cell spans the widest
    # molecule and a block is as tall as its own tallest one
    if Style["schematic_common_scale"]:
        Ratios = [float(np.max(Sizes[k*NC:(k + 1)*NC, 1])/np.max(Sizes[:, 0])) for k in range(NB)]
    else:
        Ratios = [float(np.clip(np.median(Sizes[:, 1]/Sizes[:, 0]), 0.6, 1.0))]*NB
    Heights = [Title + 2.0*cw*Rt for Rt in Ratios]
    H = sum(Heights) + Gap*(NB - 1)
    Fig = plt.figure(figsize=(W, H))
    Renderer = Fig.canvas.get_renderer()
    for n, P in enumerate(Panels):
        b, c = divmod(n, NC)
        Top = H - sum(Heights[:b]) - Gap*b
        # the name on one line when it fits the column, else broken after the arrow
        Name = Label(Style, P["System"])
        T = Fig.text((Left + (c + 0.5)*cw)/W, Top/H, Name, ha="center", va="top", fontsize=Style["font_size"],
                     color=C["ink"])
        if T.get_window_extent(Renderer).width > 0.95*cw*Fig.dpi:
            T.set_text(Name.replace(" → ", " →\n", 1))
        Top -= Title
        Ratio = Ratios[b]
        ch = cw*Ratio
        Wx = np.max(Sizes[:, 0]) if Style["schematic_common_scale"] else max(P["Size"][0], P["Size"][1]/Ratio)
        Layout = None
        for r, Y in enumerate(P["Ys"]):
            Ax = Fig.add_axes([(Left + c*cw)/W, (Top - (r + 1)*ch)/H, cw/W, ch/H])
            Ax.set_xlim(P["Mid"][0] - Wx/2.0, P["Mid"][0] + Wx/2.0)
            Ax.set_ylim(P["Mid"][1] - Wx*Ratio/2.0, P["Mid"][1] + Wx*Ratio/2.0)
            Ax.set_aspect("equal", adjustable="datalim")
            Ax.axis("off")
            Layout = DrawMolecule(Ax, Y, P["Symbols"], P["Key"], Style, Layout)
            if c == 0:
                Ax.text(-0.04, 0.5, Style["schematic_row_labels"][r], transform=Ax.transAxes, ha="right",
                        va="center", fontsize=Style["font_size"], color=C["ink"])
    Save(Fig, Style, OutDir, "fig_gas_phase_schematic")

    return None

#%% 6. Alanine dipeptide
def PlotPeptide(Rows, Chains, Style, OutDir):

    """

    Three panels: the (phi, psi) plane with every chain, the located TSs, the
    reference saddle and ts_guess; RMSD by atom set; dE against each reference.

    :return: None.

    """

    R = Rows[(Rows["Class"] == "peptide") & Rows["Candidate"].isin(Style["candidates"])]
    if R.empty:
        return None
    C = Style["colors"]
    for Sys, S in R.groupby("System"):
        Fig, (A1, A2, A3) = plt.subplots(1, 3, figsize=(Style["width_double_in"], Style["width_double_in"]/3.0*1.05))
        G = Chains[Chains["System"] == Sys]
        for _, H in G.groupby(["Run", "Pair", "Channel"]):
            A1.plot(H["Phi"], H["Psi"], color=C["chain"], linewidth=Style["line_width"], zorder=1)
        Ref = S.drop_duplicates("Reference")
        for _, Row in Ref.iterrows():
            Mk = Style["markers"]["reference"] if Row["Reference"] == "saddle" else Style["markers"]["ts_guess"]
            A1.plot(Row["PhiRef"], Row["PsiRef"], linestyle="none", marker=Mk, color=C["ink"],
                    markersize=Style["marker_size"] + 1.5, markerfacecolor="none" if Mk != "x" else C["ink"],
                    markeredgewidth=1.2, zorder=5, label=Row["Reference"])
        P0 = S[S["Reference"] == S["Reference"].iloc[0]]
        for Cand in Style["candidates"]:
            P = P0[P0["Candidate"] == Cand]
            A1.plot(P["Phi"], P["Psi"], linestyle="none", marker=Style["markers"][Cand], color=C[Cand],
                    markersize=Style["marker_size"], markeredgecolor="white", markeredgewidth=0.5, zorder=4,
                    label=Style["candidate_labels"][Cand])
        A1.set_xlabel("φ (deg)")
        A1.set_ylabel("ψ (deg)")
        A1.legend(fontsize=Style["font_size"] - 2, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
        Grid(A1, Style, "both")
        # RMSD by atom set and dE, against the saddle when present
        RefName = "saddle" if "saddle" in set(S["Reference"]) else "ts_guess"
        Q = S[(S["Reference"] == RefName) & (S["Candidate"] == Style["candidates"][0])].sort_values(["Run", "Channel"])
        Sets = [("RMSD_sym", "all atoms"), ("RMSD_heavy", "heavy atoms"), ("RMSD_backbone", "φ/ψ atoms")]
        x = np.arange(len(Q))
        W = 0.8/len(Sets)
        Shades = ["#256abf", "#6da7ec", "#b7d3f6"]
        for k, (Col, Name) in enumerate(Sets):
            A2.bar(x + (k - 1)*W, Q[Col], width=W*0.9, color=Shades[k], label=Name)
        A2.set_xticks(x)
        A2.set_xticklabels(["ch%i" % (c,) for c in Q["Channel"]])
        A2.set_ylabel("RMSD vs %s (Å)" % (RefName,))
        A2.legend(fontsize=Style["font_size"] - 2, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
        Grid(A2, Style, "y")
        Refs = [r for r in ("saddle", "ts_guess") if r in set(S["Reference"])]
        for k, r in enumerate(Refs):
            Q2 = S[(S["Reference"] == r) & (S["Candidate"] == Style["candidates"][0])].sort_values(["Run", "Channel"])
            A3.bar(np.arange(len(Q2)) + (k - (len(Refs) - 1)/2.0)*0.38, Q2["dE_eV"], width=0.34,
                   color=[C["crossing"], C["ink_secondary"]][k], label="vs %s" % (r,))
        A3.axhline(0.0, color=C["ink_secondary"], linewidth=0.6)
        # vs ts_guess is ~1 eV, vs the saddle ~0.1 eV: symlog keeps both readable
        A3.set_yscale("symlog", linthresh=Style["dE_linthresh_eV"])
        A3.set_xticks(x)
        A3.set_xticklabels(["ch%i" % (c,) for c in Q["Channel"]])
        A3.set_ylabel("ΔE (eV)")
        A3.legend(fontsize=Style["font_size"] - 2, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
        Grid(A3, Style, "y")
        Fig.suptitle(Label(Style, Sys), fontsize=Style["font_size"] + 1)
        Fig.tight_layout()
        Save(Fig, Style, OutDir, "fig_peptide_%s" % (Sys,))

    return None

#%% 7. All classes together
def PlotScatter(Rows, Style, OutDir):

    """

    RMSD against |dE| on log axes, one colour per class, as Fig. 5a of the
    OA-ReactDiff paper, with the accuracy thresholds as guides.

    :return: None.

    """

    R = pd.concat([Selected(Rows, Style, c) for c in ("two_dimensional", "gas_phase", "peptide")])
    R = R[R["Candidate"] == Style["candidates"][0]]
    if R.empty:
        return None
    C = Style["colors"]
    Fig, Ax = plt.subplots(figsize=(Style["width_single_in"], Style["width_single_in"]*0.85))
    for Cl in ("two_dimensional", "gas_phase", "peptide"):
        P = R[R["Class"] == Cl]
        if P.empty:
            continue
        Ax.plot(np.maximum(P[Style["rmsd_column"]], 1e-4), np.maximum(np.abs(P["dE_eV"]), 1e-6), linestyle="none",
                marker="o", color=C[Cl], markersize=Style["marker_size"] + 0.5, markeredgecolor="white",
                markeredgewidth=0.6, label=Style["class_labels"][Cl])
    Ax.axhline(Style["threshold_dE_eV"], color=C["ink_secondary"], linewidth=0.6, linestyle=":")
    Ax.axvline(Style["threshold_rmsd_A"], color=C["ink_secondary"], linewidth=0.6, linestyle=":")
    Ax.set_xscale("log")
    Ax.set_yscale("log")
    Ax.set_xlabel("RMSD (Å)")
    Ax.set_ylabel("|ΔE| (eV)")
    Ax.set_title("%s vs reference" % (Style["candidate_labels"][Style["candidates"][0]],),
                 fontsize=Style["font_size"] + 1, loc="left")
    Ax.legend(fontsize=Style["font_size"] - 1, loc="lower right")
    Grid(Ax, Style, "both")
    Fig.tight_layout()
    Save(Fig, Style, OutDir, "fig_rmsd_vs_dE")

    return None

#%% 8. Tables
def WriteTables(Rows, Style, OutDir):

    """

    Summary over runs per class, system, channel, candidate and reference:
    medians and ranges of RMSD and dE, as CSV and Markdown.

    :return: None.

    """

    R = Rows[Rows["OnRoute"].astype(bool) | (Rows["Class"] == "two_dimensional")]
    if Style["only_primary"]:
        R = R[R["Primary"].astype(bool) | (R["Class"] == "two_dimensional")]
    Keys = ["Class", "System", "Pair", "Channel", "Candidate", "Reference"]
    Agg = {"Runs": ("Run", "nunique"), "RMSD_median": ("RMSD", "median"),
           "RMSD_sym_median": ("RMSD_sym", "median"), "RMSD_sym_max": ("RMSD_sym", "max"),
           "RMSD_heavy_median": ("RMSD_heavy", "median"), "dE_eV_median": ("dE_eV", "median"),
           "dE_eV_min": ("dE_eV", "min"), "dE_eV_max": ("dE_eV", "max"),
           "EUpMinusRef_eV_median": ("EUpMinusRef_eV", "median"), "Energies_median": ("Energies", "median")}
    T = R.groupby(Keys, sort=False).agg(**Agg).reset_index()
    T.to_csv(os.path.join(OutDir, "table_summary.csv"), index=False, float_format="%.6g")
    with open(os.path.join(OutDir, "table_summary.md"), "w") as f:
        f.write("| Class | System | Channel | Candidate | Reference | Runs | RMSD_sym (Å) | RMSD_heavy (Å) | ΔE (eV) | E_up − E_ref (eV) | Energies |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for _, t in T.iterrows():
            f.write("| %s | %s | %s ch%i | %s | %s | %i | %.3f | %s | %+.4f [%+.4f, %+.4f] | %+.4f | %.0f |\n"
                    % (Style["class_labels"].get(t["Class"], t["Class"]), PlainLabel(Style, t["System"]), t["Pair"],
                       t["Channel"], t["Candidate"], t["Reference"], t["Runs"], t["RMSD_sym_median"],
                       "" if pd.isna(t["RMSD_heavy_median"]) else "%.3f" % (t["RMSD_heavy_median"],),
                       t["dE_eV_median"], t["dE_eV_min"], t["dE_eV_max"], t["EUpMinusRef_eV_median"],
                       t["Energies_median"]))
    print("[plot_ts_compare] %s/table_summary.{csv,md}" % (OutDir,))

    return None

#%% 9. Main
def main():

    """

    Draw every figure and table from the files of ts_compare.py.

    :return: None.

    """

    Args = parse_args()
    InDir = os.path.abspath(Args.indir)
    OutDir = os.path.join(InDir, "figures")
    os.makedirs(OutDir, exist_ok=True)
    Style = LoadStyle(Args.style or os.path.join(OutDir, "style.json"), Args.reset_style)
    Rows, Chains = LoadTables(InDir)
    Rows = ChooseReference(Rows, Style)
    Items = Args.only or ["two_dimensional", "gas_phase", "schematic", "peptide", "scatter", "profiles", "tables"]
    if "two_dimensional" in Items:
        PlotTwoDimensional(Rows, Chains, InDir, Style, OutDir)
        PlotTwoDimensionalSampling(Rows, InDir, Style, OutDir)
    if "gas_phase" in Items:
        PlotGasPhase(Rows, Style, OutDir)
    if "profiles" in Items:
        PlotProfiles(Rows, Chains, Style, OutDir)
    if "schematic" in Items:
        PlotGasPhaseSchematic(Rows, InDir, Style, OutDir)
    if "peptide" in Items:
        # the peptide panels compare against both references, so take them all
        PlotPeptide(LoadTables(InDir)[0], Chains, Style, OutDir)
    if "scatter" in Items:
        PlotScatter(Rows, Style, OutDir)
    if "tables" in Items:
        WriteTables(LoadTables(InDir)[0], Style, OutDir)

    return None

if __name__ == "__main__":
    main()
