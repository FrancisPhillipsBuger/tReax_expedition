#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Tests for how the sampler finds its input: a single .xyz, a folder holding one
file per endpoint, or the bare name of a reaction under reaction/input. The
endpoint order matters, since everything downstream reads the reactant as the
first structure. Also covers the metadata a reaction folder may carry and the
output folder the dumps are routed into.

Created on: Thu Aug 27 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import os

import pytest

from treax_expedition.io_xyz import (INPUT_ROOT, listReactions, prepareOutputFolder,
                              reactionName, readMeta, readXYZFile,
                              resolveFolder, resolveInput)

#%% 1. Helpers
REACTANT = """2
reactant
  H   0.00000   0.00000   0.00000
  H   0.00000   0.00000   0.74000
"""

PRODUCT = """2
product
  H   0.00000   0.00000   0.00000
  H   0.00000   0.00000   1.40000
"""

def writeEndpoints(Folder, First="reactant", Second="product"):

    """

    Write a two-file endpoint pair into a folder.

    :param1 Folder: folder to write into (str).
    :param2 First:  base name of the reactant file (str).
    :param3 Second: base name of the product file (str).

    :return: None.

    """

    open(os.path.join(Folder, First+".xyz"), "w").write(REACTANT)
    open(os.path.join(Folder, Second+".xyz"), "w").write(PRODUCT)

    return None

#%% 2. Resolution
def test_file_is_passed_through(tmp_path):

    """A plain file still resolves to itself, so old call sites keep working."""

    Name = str(tmp_path/"both.xyz")
    open(Name, "w").write(REACTANT+PRODUCT)
    assert resolveInput(Name) == [Name]

def test_folder_gives_reactant_first(tmp_path):

    """A folder resolves to the endpoint pair with the reactant leading."""

    writeEndpoints(str(tmp_path))
    Files = resolveInput(str(tmp_path))
    assert [os.path.basename(Name) for Name in Files] == ["reactant.xyz", "product.xyz"]

def test_ini_fin_naming_is_accepted(tmp_path):

    """The ini/fin naming of the reference data resolves the same way."""

    writeEndpoints(str(tmp_path), First="ini", Second="fin")
    Files = resolveFolder(str(tmp_path))
    assert [os.path.basename(Name) for Name in Files] == ["ini.xyz", "fin.xyz"]

def test_lone_file_in_folder_is_taken(tmp_path):

    """A folder carrying a single .xyz is unambiguous even without a pair."""

    Name = str(tmp_path/"whatever.xyz")
    open(Name, "w").write(REACTANT)
    assert resolveFolder(str(tmp_path)) == [Name]

def test_reaction_name_is_looked_up(tmp_path):

    """A bare name is resolved against the folder of shipped reactions."""

    os.mkdir(str(tmp_path/"demo"))
    writeEndpoints(str(tmp_path/"demo"))
    Files = resolveInput("demo", Root=str(tmp_path))
    assert os.path.basename(Files[0]) == "reactant.xyz"

def test_unknown_name_lists_what_exists(tmp_path):

    """The error names the reactions that could have been asked for instead."""

    os.mkdir(str(tmp_path/"demo"))
    with pytest.raises(FileNotFoundError, match="demo"):
        resolveInput("nosuch", Root=str(tmp_path))

def test_empty_folder_is_an_error(tmp_path):

    """A folder with nothing usable fails loudly rather than half-running."""

    with pytest.raises(FileNotFoundError):
        resolveFolder(str(tmp_path))

#%% 3. Reading
def test_split_endpoints_read_as_two_structures(tmp_path):

    """Two files read into the same pair a single two-frame file would give."""

    writeEndpoints(str(tmp_path))
    Split = readXYZFile(str(tmp_path))
    Name = str(tmp_path/"both.xyz")
    open(Name, "w").write(REACTANT+PRODUCT)
    Joined = readXYZFile(Name)
    assert len(Split) == len(Joined) == 2
    for A, B in zip(Split, Joined):
        assert A.get_chemical_formula() == B.get_chemical_formula()
        assert (A.get_positions() == B.get_positions()).all()

def test_mismatched_endpoints_are_rejected(tmp_path):

    """Endpoints that are not the same molecule are caught before sampling."""

    open(str(tmp_path/"reactant.xyz"), "w").write(REACTANT)
    open(str(tmp_path/"product.xyz"), "w").write("""1
product
  H   0.00000   0.00000   0.00000
""")
    with pytest.raises(RuntimeError):
        readXYZFile(str(tmp_path))

#%% 4. The reaction shipped with the package
@pytest.mark.reaction("sn2")
def test_sn2_is_shipped_and_readable():

    """The sn2 reaction resolves by name and reads as a matched pair."""

    assert "sn2" in listReactions()
    Structures = readXYZFile("sn2")
    assert len(Structures) == 2
    assert Structures[0].get_chemical_formula() == "CH3BrCl"
    assert os.path.isdir(os.path.join(INPUT_ROOT, "sn2"))

#%% 5. Metadata
def test_meta_is_read_from_a_folder(tmp_path):

    """A reaction folder can state the charge its .xyz files cannot carry."""

    writeEndpoints(str(tmp_path))
    open(str(tmp_path/"meta.json"), "w").write('{"charge": -1, "mult": 1}')
    assert readMeta(str(tmp_path))["charge"] == -1

def test_meta_is_found_next_to_a_named_file(tmp_path):

    """Naming one endpoint file still finds the metadata beside it."""

    writeEndpoints(str(tmp_path))
    open(str(tmp_path/"meta.json"), "w").write('{"charge": -1}')
    assert readMeta(str(tmp_path/"reactant.xyz"))["charge"] == -1

def test_missing_meta_is_empty(tmp_path):

    """No metadata is not an error, the caller falls back to its defaults."""

    writeEndpoints(str(tmp_path))
    assert readMeta(str(tmp_path)) == {}

@pytest.mark.reaction("sn2")
def test_sn2_meta_states_the_anion():

    """The shipped sn2 reaction is Cl- + CH3Br, so it has to run at charge -1."""

    assert readMeta("sn2")["charge"] == -1
    assert readMeta("sn2")["mult"] == 1

#%% 6. Output folder
def test_reaction_name_from_folder_and_file(tmp_path):

    """The output folder is named after the reaction, not after an endpoint."""

    writeEndpoints(str(tmp_path))
    assert reactionName(str(tmp_path)) == tmp_path.name
    assert reactionName(str(tmp_path/"reactant.xyz")) == tmp_path.name
    assert reactionName(str(tmp_path/"whatever.xyz")) == "whatever"
    assert reactionName("sn2") == "sn2"

def test_output_folder_is_created(tmp_path):

    """The folder is made on demand, so a fresh reaction needs no setup."""

    Folder = str(tmp_path/"fresh")
    prepareOutputFolder(Folder)
    assert os.path.isdir(Folder)

def test_append_dumps_are_cleared(tmp_path):

    """A rerun must not extend the trajectories an earlier run left behind."""

    Folder = str(tmp_path)
    open(os.path.join(Folder, "sampled1.trj"), "w").write(REACTANT)
    open(os.path.join(Folder, "evaluated1.xyz"), "w").write(REACTANT)
    Kept = os.path.join(Folder, "sampled_energy1.log")
    open(Kept, "w").write("# left alone, it is opened for writing anyway\n")
    prepareOutputFolder(Folder)
    assert not os.path.isfile(os.path.join(Folder, "sampled1.trj"))
    assert not os.path.isfile(os.path.join(Folder, "evaluated1.xyz"))
    assert os.path.isfile(Kept)
