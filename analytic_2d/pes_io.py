#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Writers for plottable views of a 2D potential: a gnuplot-style grid of the
surface, the two one-dimensional cuts through the first saddle, and the table
of stationary points. Energies are in the native unit of the potential.

Created on: Mon Aug 24 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import os

import numpy as np

#%% 1. Two-dimensional grid
def SavePES(Potential, FileName, Npts=100, Limits=None):

    """

    Write the potential on a grid, one block per x value.

    :param1 Potential: a Potential2D.
    :param2 FileName:  destination file name.
    :param3 Npts:      number of grid points per direction.
    :param4 Limits:    ((xmin, xmax), (ymin, ymax)), or None for the grid of the potential.

    :return: None.

    """

    (x0, x1), (y0, y1) = Potential.GridLimits if Limits is None else Limits
    y = np.linspace(y0, y1, Npts)
    with open(FileName, "w") as f:
        f.write("# x[$1] y[$2] V,%s[$3]\n" % (Potential.UnitName,))
        for xi in np.linspace(x0, x1, Npts):
            V = Potential.Total(np.stack([np.full_like(y, xi), y], axis=-1))
            for yj, vj in zip(y, V):
                f.write(" %8.4f %8.4f %15.4f\n" % (xi, yj, vj))
            f.write("\n")

    return None

#%% 2. One-dimensional cuts
def SaveCuts(Potential, FolderToSave=".", Npts=1000, Limits=None):

    """

    Write ``pes_x.dat`` and ``pes_y.dat``, the cuts through the first saddle.

    On the proton-transfer model these are the cuts along the transfer and
    the bath coordinate, since its saddle is the origin.

    :param1 Potential:    a Potential2D.
    :param2 FolderToSave: folder for the two output files.
    :param3 Npts:         number of points per cut.
    :param4 Limits:       ((xmin, xmax), (ymin, ymax)), or None for the grid of the potential.

    :return: None.

    """

    (x0, x1), (y0, y1) = Potential.GridLimits if Limits is None else Limits
    S = Potential.Saddles()[0]
    for Name, Axis, Grid in (("pes_x.dat", 0, np.linspace(x0, x1, Npts)),
                             ("pes_y.dat", 1, np.linspace(y0, y1, Npts))):
        R = np.tile(S, (Npts, 1))
        R[:, Axis] = Grid
        np.savetxt(os.path.join(FolderToSave, Name), np.stack([Grid, Potential.Total(R)], axis=-1),
                   fmt=" %8.4f %15.4f", header="%s[$1] V,%s[$2] through (%+.4f, %+.4f)"
                   % ("xy"[Axis], Potential.UnitName, S[0], S[1]))

    return None

#%% 3. Stationary points
def SaveStationaryPoints(Potential, FileName, Mass=None):

    """

    Write the minima and saddles with their energies and frequencies.

    :param1 Potential: a Potential2D.
    :param2 FileName:  destination file name.
    :param3 Mass:      mass entering the frequencies, or None for the default.

    :return: None.

    """

    Mass = Potential.DefaultMass if Mass is None else Mass
    Labels, Points = Potential.StationaryPoints()
    with open(FileName, "w") as f:
        f.write("# label[$1] x[$2] y[$3] V,%s[$4] V,eV[$5] nu1,cm-1[$6] nu2,cm-1[$7]\n"
                % (Potential.UnitName,))
        for Label, R in zip(Labels, Points):
            Nu, _ = Potential.NormalModes(R, Mass)
            f.write(" %-3s %9.5f %9.5f %15.4f %12.6f %9.1f %9.1f\n"
                    % (Label, R[0], R[1], Potential.Total(R), Potential.Energy(R), Nu[0], Nu[1]))

    return None
