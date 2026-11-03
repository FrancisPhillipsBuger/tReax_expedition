#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Zero-temperature labels along a path: the state each configuration drains
into under a gradient-free quench. A bisection finds, with a logarithmic number
of quenches, every place where the label changes along an MCTSS path.

Created on: Mon Sep 28 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
from dataclasses import dataclass

import numpy as np

from utils.mc.quench import Quench, QuenchSchedule

#%% 1. Labeller
# a label names the basin a configuration drains into at zero temperature, so
# the quench starts cold and with small steps: a hot start or a long first
# move can carry a point near the ridge over it
LabelSchedule = QuenchSchedule(kT=(0.005, 0.002, 0.001, 0.0005), Steps=150, Patience=4)

@dataclass
class Transition:

    """

    Two adjacent configurations of a path that drain into different states.

    """

    Lo: int
    Hi: int
    From: str
    To: str

class Labeller:

    """

    Callable quenching a configuration and naming the state it drains into.

    """

    def __init__(self, System, Book, StepTau, Schedule=None, Screen=False, Votes=1):

        """

        :param1 System:   AnalyticSystem or AtomsSystem.
        :param2 Book:     StateBook, extended with every new state met.
        :param3 StepTau:  initial time parameter of the proposal width, in fs.
        :param4 Schedule: QuenchSchedule, or None for LabelSchedule.
        :param5 Screen:   screen proposals on the system's cheaper calculator.
        :param6 Votes:    quenches per label, the majority deciding; near the
                          separatrix a single quench is a coin flip.

        """

        self.System = System
        self.Book = Book
        self.StepTau = float(StepTau)
        self.Schedule = LabelSchedule if Schedule is None else Schedule
        self.Screen = Screen
        self.Votes = int(Votes)
        self.Records = []

    def __call__(self, X):

        """

        :param1 X: coordinates.

        :return: Name: the state the configuration drains into.

        """

        Names = []
        for _ in range(0, self.Votes):
            R = Quench(self.System, X, self.StepTau, Schedule=self.Schedule, Screen=self.Screen)
            Names.append(self.Book.Register(R.X, R.Energy, Converged=R.Converged))
            self.Records.append({"X": np.asarray(X, dtype=float).copy(), "Label": Names[-1],
                                 "XMin": R.X, "EMin": R.Energy, "NCalls": R.NCalls,
                                 "Converged": R.Converged})

        # a tie goes to the label met first, whatever the hash order of names
        return max(Names, key=Names.count)

#%% 2. Bisection along a path
def DistinctFrames(Path, Energies=None, Tol=1.0e-12):

    """

    Drop consecutive repeats; a sampler dumps the walker every step, moved or not.

    :param1 Path:     list of coordinates.
    :param2 Energies: matching energies, or None.
    :param3 Tol:      largest squared difference counted as a repeat.

    :return: Path, Energies: the thinned lists.

    """

    Keep = [0]
    for i in range(1, len(Path)):
        D = np.asarray(Path[i]) - np.asarray(Path[Keep[-1]])
        if float(np.dot(D.ravel(), D.ravel())) > Tol:
            Keep.append(i)
    P = [np.asarray(Path[i], dtype=float) for i in Keep]

    return P, (None if Energies is None else [Energies[i] for i in Keep])

def LabelPath(Path, Label, First, Last):

    """

    Every change of the zero-temperature label along a path, by bisection.

    The labels of the two ends are known. The middle of every interval whose
    ends differ is quenched, and the halves whose ends still differ are split
    again, until each change sits between two adjacent configurations. An
    excursion that leaves and re-enters a state between two quenched points is
    not seen, which only costs a channel, never a wrong one.

    :param1 Path:  list of coordinates, the first in state First.
    :param2 Label: callable X -> state name.
    :param3 First: state of the first configuration.
    :param4 Last:  state of the last configuration.

    :return: Transitions, Labels: the list of Transition in path order and the
             dict index -> label of every quenched configuration.

    """

    Labels = {0: First, len(Path) - 1: Last}
    Found = []

    def Split(Lo, Hi):
        if Labels[Lo] == Labels[Hi]:
            return
        if Hi - Lo <= 1:
            Found.append(Transition(Lo, Hi, Labels[Lo], Labels[Hi]))
            return
        Mid = (Lo + Hi)//2
        Labels[Mid] = Label(Path[Mid])
        Split(Lo, Mid)
        Split(Mid, Hi)

    Split(0, len(Path) - 1)
    Found.sort(key=lambda t: t.Lo)

    return Found, Labels

def IntervalLabels(NFrames, Labels):

    """

    Labels of every frame of a path, from the quenched ones: a frame between
    two quenched frames of one state takes that state, the others none.

    :param1 NFrames: number of frames of the path.
    :param2 Labels:  dict index -> label from LabelPath.

    :return: Names: list of the label of every frame, None where undecided.

    """

    Names = [None]*NFrames
    Known = sorted(Labels)
    for a, b in zip(Known[:-1], Known[1:]):
        if Labels[a] == Labels[b]:
            for i in range(a, b + 1):
                Names[i] = Labels[a]
        else:
            Names[a] = Labels[a]
            Names[b] = Labels[b]

    return Names
