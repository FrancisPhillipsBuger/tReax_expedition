#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

The transition state estimate on a certified chain: the point where the
zero-temperature label changes, i.e. where the chain crosses the separatrix
(q = 1/2 at zero temperature), located by bisection with quenches. Its energy
lies between the saddle energy and the upper bound of the chain. The
curvature along the chain at its top comes from three energies.

Created on: Mon Sep 28 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import numpy as np
import scipy.constants as spc

#%% 1. Crossing of the separatrix
def LocateCrossing(System, Chain, Label, Refine=4):

    """

    Bisection of the zero-temperature label along a chain.

    The node labels are bisected first, then the straight link between the two
    adjacent nodes whose labels differ, Refine more times. Near the separatrix
    a quench outcome is a coin flip, so the point found is where the two
    outcomes are equally likely along the chain, to the last bisection step.

    :param1 System: AnalyticSystem or AtomsSystem.
    :param2 Chain:  certified Chain; its end labels are the two states.
    :param3 Label:  callable X -> state name (a Labeller).
    :param4 Refine: bisections of the final link.

    :return: Crossing: dict with X, Energy, Node (index of the node before the
             crossing), t (position along the link) and NQuench.

    """

    First, Last = Chain.Ends
    Labels = {0: First, len(Chain.Nodes) - 1: Last}
    NQuench = 0
    Lo, Hi = 0, len(Chain.Nodes) - 1
    while Hi - Lo > 1:
        Mid = (Lo + Hi)//2
        Labels[Mid] = Label(Chain.Nodes[Mid].X)
        NQuench += 1
        if Labels[Mid] == First:
            Lo = Mid
        else:
            Hi = Mid

    P = Chain.Nodes[Lo].X
    Q = System.Aligned(Chain.Nodes[Hi].X, P)
    a, b = 0.0, 1.0
    for _ in range(0, Refine):
        t = 0.5*(a + b)
        NQuench += 1
        if Label(P + t*(Q - P)) == First:
            a = t
        else:
            b = t
    t = 0.5*(a + b)
    X = P + t*(Q - P)

    return {"X": X, "Energy": System.Energy(X), "Node": Lo, "t": t, "NQuench": NQuench}

#%% 2. Curvature along the chain
def ChainCurvature(System, Chain, Index=None, Step=0.05):

    """

    Curvature of the energy along the chain tangent at a node, from three
    energies, and the harmonic frequency and crossover temperature it implies.

    :param1 System: AnalyticSystem or AtomsSystem.
    :param2 Chain:  Chain.
    :param3 Index:  node index, or None for the highest node.
    :param4 Step:   displacement along the tangent, in sqrt(amu) angstroms.

    :return: Curvature: dict with k in eV/(amu A^2), nu in cm-1 (negative for
             a maximum) and Tc in K, or None at an end of the chain.

    """

    E = Chain.Energies
    k = int(np.argmax(E)) if Index is None else int(Index)
    if k <= 0 or k >= len(Chain.Nodes) - 1:
        return None

    X0 = Chain.Nodes[k].X
    T = System.Aligned(Chain.Nodes[k + 1].X, X0) - System.Aligned(Chain.Nodes[k - 1].X, X0)
    Root = np.sqrt(System.Masses)
    u = Root*T
    Norm = float(np.linalg.norm(u))
    if Norm <= 0.0:
        return None
    dX = Step*(u/Norm)/Root
    Ep = System.Energy(X0 + dX)
    Em = System.Energy(X0 - dX)
    if Ep is None or Em is None:
        return None
    Curv = (Ep + Em - 2.0*Chain.Nodes[k].Energy)/Step**2

    # omega in rad/s from eV/(amu A^2), then wavenumber and crossover temperature
    Omega = np.sqrt(abs(Curv)*spc.eV/(spc.atomic_mass*1.0e-20))
    Nu = np.sign(Curv)*Omega/(2.0*np.pi*spc.c*1.0e2)
    Tc = spc.hbar*Omega/(2.0*np.pi*spc.k)

    return {"Node": k, "k": float(Curv), "nu": float(Nu), "Tc": float(Tc)}
