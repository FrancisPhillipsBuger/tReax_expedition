#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

XYZ I/O over ASE for the SWS sampler. Reads the one or two endpoint
structures from a single file, from a folder holding one file per endpoint,
or from the bare name of a reaction under reaction/input, checking that both
share the same stoichiometry and atomic order, appends sampled frames to a
trajectory, and records every structure whose energy was actually computed.

Created on: Mon Aug 24 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import json
import os

import ase.io as asio
import numpy as np

#%% 1. Input layout
# Folder holding the reactions shipped with the package, one sub-folder each.
INPUT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "reaction", "input")

# Endpoint file names accepted inside a reaction folder, reactant first. The
# first pair that is complete wins, so a folder may carry more than one naming.
ENDPOINT_NAMES = (("reactant", "product"),
                  ("ini", "fin"),
                  ("initial", "final"),
                  ("start", "end"))

def listReactions(Root=INPUT_ROOT):

    """

    List the reaction names that can be given instead of a path.

    :param1 Root: folder holding one sub-folder per reaction (str).

    :return: sorted list of reaction names (list of str).

    """

    if not os.path.isdir(Root):
        return []

    return sorted(Name for Name in os.listdir(Root)
                  if os.path.isdir(os.path.join(Root, Name)))

def resolveInput(Source, Root=INPUT_ROOT):

    """

    Turn what the user typed into the XYZ files to read.

    Accepted, in this order: an .xyz file holding one or two structures, a
    folder holding two endpoint files, or the bare name of a reaction under
    the package input folder.

    :param1 Source: file, folder, or reaction name (str).
    :param2 Root:   folder holding one sub-folder per reaction (str).

    :return: list of one or two file names, endpoints in reactant-first order.

    """

    if Source is None:
        raise ValueError("No input was given.")

    # a plain file is passed through untouched, so old call sites keep working
    if os.path.isfile(Source):
        return [Source]

    Folder = None
    if os.path.isdir(Source):
        Folder = Source
    else:
        # a bare name is looked up among the reactions shipped with the package
        Candidate = os.path.join(Root, Source)
        if os.path.isdir(Candidate):
            Folder = Candidate
        elif os.path.isfile(Candidate+".xyz"):
            return [Candidate+".xyz"]
        else:
            Known = listReactions(Root)
            raise FileNotFoundError(
                "No file, folder, or reaction named '%s'. Known reactions: %s"
                % (Source, ", ".join(Known) if Known else "none"))

    return resolveFolder(Folder)

def resolveFolder(Folder):

    """

    Pick the XYZ files out of a folder that holds one reaction.

    :param1 Folder: folder to look in (str).

    :return: list of one or two file names, endpoints in reactant-first order.

    """

    # a complete endpoint pair is the point of the folder layout
    for First, Second in ENDPOINT_NAMES:
        Files = [os.path.join(Folder, Name+".xyz") for Name in (First, Second)]
        if all(os.path.isfile(Name) for Name in Files):
            return Files

    # otherwise a folder carrying a single .xyz is still unambiguous
    Loose = sorted(Name for Name in os.listdir(Folder) if Name.endswith(".xyz"))
    if len(Loose) == 1:
        return [os.path.join(Folder, Loose[0])]

    # and so is one named after the folder itself
    Named = os.path.join(Folder, os.path.basename(os.path.normpath(Folder))+".xyz")
    if os.path.isfile(Named):
        return [Named]

    raise FileNotFoundError(
        "Folder %s holds no usable input. Expected %s, a single .xyz, or %s.xyz, found: %s"
        % (Folder,
           " and ".join("%s.xyz/%s.xyz" % Pair for Pair in ENDPOINT_NAMES[:1]),
           os.path.basename(os.path.normpath(Folder)),
           ", ".join(Loose) if Loose else "nothing"))

#%% 2. Reaction metadata and output folder
# Folder receiving one sub-folder of dumps per reaction.
OUTPUT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "reaction", "output")

# Name of the optional file carrying what the input alone cannot say.
META_NAME = "meta.json"

# Dumps opened in append mode, which a rerun into the same folder would extend
# rather than replace. The energy logs are opened for writing and truncate on
# their own, so they are deliberately not in this list.
APPEND_DUMPS = ("sampled.trj", "sampled1.trj", "sampled2.trj",
                "equilibration.trj",
                "evaluated0.xyz", "evaluated1.xyz", "evaluated2.xyz")

def reactionName(Source):

    """

    Name the reaction an input refers to, used to name its output folder.

    :param1 Source: file, folder, or reaction name (str).

    :return: the name (str).

    """

    Name = os.path.basename(os.path.normpath(Source))
    if Name.endswith(".xyz"):
        # a lone file is named after itself, a pair after the folder holding it
        Stem = Name[:-4]
        Folder = os.path.basename(os.path.normpath(os.path.dirname(os.path.abspath(Source))))
        return Folder if Stem in sum(ENDPOINT_NAMES, ()) else Stem

    return Name

def readMeta(Source, Root=INPUT_ROOT):

    """

    Read the optional metadata of a reaction.

    The charge and the multiplicity are properties of the system, not of the
    run, but an .xyz cannot carry them. A reaction folder may state them once
    in meta.json so that the reaction can be run without repeating them on
    every command line.

    :param1 Source: file, folder, or reaction name (str).
    :param2 Root:   folder holding one sub-folder per reaction (str).

    :return: dictionary of what was found, empty when there is no metadata.

    """

    if Source is None:
        return {}

    # the metadata sits next to the endpoints, whichever of the two was named
    if os.path.isdir(Source):
        Folder = Source
    elif os.path.isfile(Source):
        Folder = os.path.dirname(os.path.abspath(Source))
    else:
        Folder = os.path.join(Root, Source)

    Name = os.path.join(Folder, META_NAME)
    if not os.path.isfile(Name):
        return {}

    with open(Name, encoding="utf-8") as Handle:
        Meta = json.load(Handle)
    if not isinstance(Meta, dict):
        raise RuntimeError("%s does not hold a JSON object!" % (Name,))

    return Meta

def prepareOutputFolder(Folder, PrintStuff=False):

    """

    Create the output folder and clear the dumps a rerun would append to.

    :param1 Folder:     folder for the dumps of one run (str).
    :param2 PrintStuff: flag to print stuff (bool).

    :return: the folder (str).

    """

    os.makedirs(Folder, exist_ok=True)

    # trajectories are written in append mode, so a leftover file from an
    # earlier run into the same folder would silently be extended
    Cleared = 0
    for Name in APPEND_DUMPS:
        Path = os.path.join(Folder, Name)
        if os.path.isfile(Path):
            os.remove(Path)
            Cleared += 1
    if Cleared > 0 and PrintStuff:
        print("output: cleared %i trajectory dump(s) left in %s by an earlier run"
              % (Cleared, Folder))

    return Folder

#%% 3. Reader
def checkSameStoichiometry(Structures):

    """

    Check that two structures share atom count, elements, and atomic order.

    :param1 Structures: list of two ase.Atom instances.

    :return: None.

    """

    Z1 = np.array(Structures[0].get_atomic_numbers())
    Z2 = np.array(Structures[1].get_atomic_numbers())
    if len(Z1) != len(Z2):
        raise RuntimeError("Different number of atoms in two structures!")
    if np.sum(np.abs(Z1 - Z2)) > 1.0e-15:
        raise RuntimeError("The stoichiometries (or atomic order) does not match for two structures!")

    return None

def readXYZFile(FileName, PrintStuff=False):

    """

    Read the endpoints, from a file, a reaction folder, or a reaction name.

    :param1 FileName:   .xyz file, folder with the endpoint files, or the name
                        of a reaction under reaction/input (str).
    :param2 PrintStuff: flag to print stuff (bool).

    :return: Res: a list of length 1 or 2 with ase.Atom instances,
             representing the molecules.

    """

    Sources = resolveInput(FileName)

    # two files means the endpoints were kept apart, one file may still hold both
    if len(Sources) == 2:
        Res = [asio.read(Name, index=0) for Name in Sources]
        if PrintStuff:
            print("Two structures were read from "
                  +" and ".join(str(Name) for Name in Sources))
    else:
        Res = []
        Res.append(asio.read(Sources[0], index=0))
        try:
            Res.append(asio.read(Sources[0], index=1))
            if PrintStuff:
                print("Two structures were found in file "+str(Sources[0]))
        except:
            if PrintStuff:
                print("Only one structure was found in file "+str(Sources[0]))

    # additional check, that the stoichiometries are the same
    if len(Res) == 2:
        checkSameStoichiometry(Res)

    return Res

#%% 4. Writer
def writeXYZFile(AtomInstance, FileName, Append=True, PrintStuff=False):

    """

    Save the XYZ file using ASE.

    :param1 AtomInstance: ase.Atom instance, for which the geometry will be saved.
    :param2 FileName:     file name of the XYZ trajectory file (str).
    :param3 Append:       flag to append (if True) or rewrite (if False) the file.
    :param4 PrintStuff:   flag to print stuff (bool).

    :return: None.

    """

    asio.write(FileName, AtomInstance, format="xyz", append=Append)

#%% 5. Evaluation log
class EvaluationLog:

    """

    Recorder for every structure whose energy was actually computed.

    The sampled trajectories only hold the structure a walker carries after a
    step, so the trial structures that were evaluated and then rejected are
    lost. They cost exactly as much as the accepted ones and, being higher in
    energy, they are the more informative half of the data for fitting a
    surrogate or a committor. This writes them out at no extra cost.

    """

    def __init__(self, Index, FolderToSave=".", Enabled=True):

        """

        Open the dump files of one walker.

        :param1 Index:        index of the walker, used in the file names.
        :param2 FolderToSave: folder for the dump files.
        :param3 Enabled:      when False nothing is written to disk.

        """

        self.Index = Index
        self.Enabled = Enabled
        self.Step = 0
        self.NEval = 0
        self.TrjFile = None
        self.EnDump = None
        if not self.Enabled:
            return
        self.TrjFile = FolderToSave+"/evaluated%i.xyz" % (Index)
        self.EnDump = open(FolderToSave+"/evaluated_energy%i.log" % (Index), "w")
        self.EnDump.write("# step[$1] eval[$2] E,eV[$3] accepted[$4]\n")

    def Record(self, Structure, Energy, Accepted):

        """

        Record one evaluated structure and its energy.

        :param1 Structure: ase.Atom instance that was evaluated.
        :param2 Energy:    its potential energy, in eV.
        :param3 Accepted:  whether the move to it was taken.

        :return: None.

        """

        if not self.Enabled:
            return None
        writeXYZFile(Structure, self.TrjFile, Append=True, PrintStuff=False)
        self.EnDump.write(" %8i %8i %15.10f %3i\n" %
                          (self.Step, self.NEval, Energy, int(Accepted)))
        self.EnDump.flush()
        self.NEval += 1

        return None

    def close(self):

        """

        Close the dump files.

        :return: None.

        """

        if self.EnDump is not None:
            self.EnDump.close()
            self.EnDump = None

        return None
