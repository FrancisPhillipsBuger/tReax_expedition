#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

CLI summary of finished staged searches: for every search.json given, the
states, the certified upper bound of every channel, the separatrix crossing,
the committor and pB there when S4 ran, tau_explore and the energy calls of
every stage, as one table. With --grade, the saved grade lines are added.

Created on: Mon Sep 28 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import argparse
import json
import os

#%% 1. Report of one search
def SearchLines(Folder):

    """

    :param1 Folder: folder of a finished search, or its search.json.

    :return: Lines: text lines describing it.

    """

    File = Folder if Folder.endswith(".json") else os.path.join(Folder, "search.json")
    with open(File) as f:
        R = json.load(f)
    Unit = "kJ/mol" if R["System"] not in ("2d", "muller", "two_paths") else ""
    Lines = ["%s  [%s]" % (R["System"], os.path.dirname(os.path.abspath(File)))]
    Lines.append("  states: " + ", ".join("%s %s" % (s["Name"], s["Fingerprint"]) for s in R["States"]))
    Lines.append("  tau_explore %s fs per walker, the pair's midpoint at twice that" % (R["Stages"]["S1"]["TauExplore"],))
    for P in R["Pairs"]:
        Certified = sum(1 for c in P["Chains"] if c["Certified"])
        Lines.append("  %s -> %s along %s: %i/%i chains certified, %i channel(s)"
                     % (P["From"], P["To"], P["Coordinate"], Certified, len(P["Chains"]), len(P["Channels"])))
        for Ch in P["Channels"]:
            Line = ("    %-10s E_up fwd %10.4f bwd %10.4f %s"
                    % (Ch["File"], Ch["Forward"], Ch["Backward"], Unit))
            X = Ch.get("Crossing")
            if X is not None:
                Line += " | crossing fwd %10.4f" % (X["Forward"],)
                if "pB" in X:
                    Line += " q %.3f pB %.3f+-%.3f" % (X["q"], X["pB"], X["pBErr"])
            Curv = Ch.get("Curvature")
            if Curv is not None:
                Line += " | nu %.0f cm-1 T_c %.0f K" % (Curv["nu"], Curv["Tc"])
            Lines.append(Line)
            T = Ch.get("TSE")
            if T is not None:
                Lines.append("    %-10s TSE at tau* %.2f fs: <V(M)> fwd %10.4f %s, xi spread %.3g, ESS/N %.3f, %i midpoints"
                             % ("", T["TauPhys"], T["Forward"], Unit, T["XiSpread"], T["ESS"], T["N"]))
    A = R["AtoB"]
    if A["Forward"] is not None:
        Lines.append("  A -> B via %s: certified E_up fwd %.4f bwd %.4f %s"
                     % ("-".join(A["Route"]), A["Forward"], A["Backward"], Unit))
    else:
        Lines.append("  A -> B: no certified chain")
    Lines.append("  energies: " + ", ".join("%s %i" % kv for kv in R["Cost"].items())
                 + "; wall %.0f s" % (R["WallSeconds"],))

    return Lines

#%% 2. Main
def main():

    """

    Print the summary of every search folder given.

    :return: None.

    """

    p = argparse.ArgumentParser(description="Summary of finished staged searches")
    p.add_argument("folders", nargs="+", help="Search folders or search.json files.")
    p.add_argument("--grade", action="store_true", help="Also print grade.txt when present.")
    Args = p.parse_args()
    for Folder in Args.folders:
        for Line in SearchLines(Folder):
            print(Line)
        Grade = os.path.join(Folder, "grade.txt")
        if Args.grade and os.path.isfile(Grade):
            with open(Grade) as f:
                print(f.read().rstrip())
        print("")

    return None

if __name__ == "__main__":
    main()
