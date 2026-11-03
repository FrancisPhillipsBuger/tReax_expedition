#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Tests for the GFN2-xTB back end that is now the default, alongside the ORCA
back end kept for the runs that have to match a published level of theory:
the parametrisation names xtb accepts, that the charge and the
multiplicity really reach the electronic structure, that one calculator shared
between the two walkers still answers per structure, and that the sampler runs
end to end on the anionic SN2 system the reaction inputs describe.

Created on: Thu Aug 27 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import numpy as np
import pytest
from ase.units import Hartree

from analyse import (BinnedProfile, BinnedStructures, BootstrapBarriers,
                            LoadRun, LoadRunStructures)
from treax_expedition.calculators import (setCalculatorToStructure,
                                   wrapGetPotentialEnergy)
from treax_expedition.io_xyz import readMeta, readXYZFile
from treax_expedition.orca import getORCACalculator
from treax_expedition.sampler import doTwoStructureSWSMC
from treax_expedition.xtb import DEFAULT_METHOD, getXTBCalculator, normaliseMethod

#%% 1. Fixtures
@pytest.fixture
def Endpoints():

    """The two endpoints of the shipped SN2 reaction, without a calculator."""

    return readXYZFile("sn2")

@pytest.fixture
def Anion():

    """A GFN2-xTB calculator set up for Cl- + CH3Br, as meta.json states it."""

    Meta = readMeta("sn2")
    return getXTBCalculator(charge=Meta["charge"], mult=Meta["mult"])

#%% 2. Parametrisation names
def test_default_is_gfn2():

    """GFN2-xTB is what an unspecified method means."""

    assert DEFAULT_METHOD == "GFN2-xTB"
    assert normaliseMethod(None) == "GFN2-xTB"
    assert normaliseMethod("") == "GFN2-xTB"

def test_aliases_are_accepted():

    """Case and the -xtb suffix are not something a user should have to get right."""

    for Name in ("gfn2", "GFN2", "gfn2-xtb", "GFN2-xTB", " gfn2-xtb "):
        assert normaliseMethod(Name) == "GFN2-xTB"
    for Name in ("gfn1", "gfn1-xtb", "ipea1-xtb", "gfn-ff"):
        with pytest.raises(ValueError, match="GFN2"):
            normaliseMethod(Name)

def test_retired_method_is_rejected():

    """A level of theory of the old back end must not silently become GFN2."""

    with pytest.raises(ValueError, match="hf-3c"):
        normaliseMethod("hf-3c")
    with pytest.raises(ValueError):
        normaliseMethod("B3LYP")

#%% 3. Energies
@pytest.mark.reaction("sn2")
def test_energy_is_finite_and_in_eV(Endpoints, Anion):

    """Both endpoints evaluate, and the result is an ASE energy in eV."""

    for Structure in Endpoints:
        Structure = setCalculatorToStructure(Structure, Anion)
        Energy = wrapGetPotentialEnergy(Structure)
        assert np.isfinite(Energy)
        # six atoms of C, H, Br and Cl at GFN2 sit in the few hundred eV range
        assert -500. < Energy < -100.

@pytest.mark.reaction("sn2")
def test_product_is_the_more_stable_endpoint(Endpoints, Anion):

    """Cl- + CH3Br -> CH3Cl + Br- is downhill, as the reference data also has it."""

    Energies = []
    for Structure in Endpoints:
        Structure = setCalculatorToStructure(Structure, Anion)
        Energies.append(wrapGetPotentialEnergy(Structure))
    assert Energies[1] < Energies[0]

def test_charge_and_multiplicity_reach_the_back_end():

    """What meta.json states has to arrive at the electronic structure itself.

    The physical contrast cannot be measured on this system: the endpoints are
    geometries of the anion, and the SCF of the cation or of the neutral
    radical does not converge on them. What can be checked, and is what the
    wiring can get wrong, is that the numbers are handed over unchanged.

    """

    Calculator = getXTBCalculator(Method="gfn2", charge=-1, mult=1,
                                  Accuracy=0.5, ElectronicTemperature=500.)
    assert Calculator.parameters["charge"] == -1
    assert Calculator.parameters["multiplicity"] == 1
    assert Calculator.parameters["method"] == "GFN2-xTB"
    assert Calculator.parameters["accuracy"] == 0.5
    assert Calculator.parameters["electronic_temperature"] == 500.

@pytest.mark.reaction("sn2")
def test_forces_are_available(Endpoints, Anion):

    """The surrogate may be trained on forces, so the back end has to give them."""

    Structure = setCalculatorToStructure(Endpoints[0], Anion)
    Forces = Structure.get_forces()
    assert Forces.shape == (len(Structure), 3)
    assert np.all(np.isfinite(Forces))

@pytest.mark.reaction("sn2")
def test_one_calculator_serves_both_walkers(Endpoints, Anion):

    """The two-structure run shares a calculator; each structure keeps its own energy."""

    First, Second = (setCalculatorToStructure(S, Anion) for S in Endpoints)
    Seen = [(wrapGetPotentialEnergy(First), wrapGetPotentialEnergy(Second))
            for _ in range(3)]
    # the two walkers stay apart, and neither drifts as the other is evaluated
    for Pair in Seen:
        assert abs(Pair[0] - Pair[1]) > 0.01
    for Index in (0, 1):
        Values = [Pair[Index] for Pair in Seen]
        assert max(Values) - min(Values) < 1.0e-6

#%% 4. Failure handling
@pytest.mark.reaction("sn2")
def test_a_failing_evaluation_is_reported(Endpoints):

    """A back end that cannot answer raises rather than returning nonsense."""

    class Broken:

        """Calculator that always fails, standing in for a dead SCF."""

        def get_potential_energy(self, *Args, **Kwargs):
            raise RuntimeError("no")

    Structure = Endpoints[0]
    Structure.calc = Broken()
    with pytest.raises(RuntimeError, match="Failed to evaluate"):
        wrapGetPotentialEnergy(Structure, NumIter=2)

@pytest.mark.reaction("sn2")
def test_the_cause_of_a_failure_is_kept(Endpoints):

    """A file driven back end fails for reasons the caller cannot guess."""

    class Broken:

        """Calculator that fails for a reason worth reporting."""

        def get_potential_energy(self, *Args, **Kwargs):
            raise RuntimeError("licence server unreachable")

    Structure = Endpoints[0]
    Structure.calc = Broken()
    with pytest.raises(RuntimeError, match="licence server unreachable"):
        wrapGetPotentialEnergy(Structure, NumIter=2)

#%% 5. End to end
@pytest.mark.reaction("sn2")
def test_sampler_runs_on_the_sn2_system(Endpoints, Anion, tmp_path):

    """The whole point of the swap: the real reaction now samples for real."""

    Structures = [setCalculatorToStructure(S, Anion) for S in Endpoints]
    NTrue = doTwoStructureSWSMC(Structures=Structures, tau=4., tau12=40.,
                                Nsteps=4, Seed=1234,
                                FolderToSave=str(tmp_path))
    assert NTrue >= 2
    assert (tmp_path/"sampled1.trj").is_file()
    assert (tmp_path/"sampled2.trj").is_file()

@pytest.mark.reaction("sn2")
def test_the_run_dumps_a_coordinate_the_barrier_analysis_reads(Endpoints, Anion, tmp_path):

    """ic1.trj and ic2.trj match the energy logs row by row, the initial endpoint last."""

    Structures = [setCalculatorToStructure(S, Anion) for S in Endpoints]
    doTwoStructureSWSMC(Structures=Structures, tau=1., tau12=3., Nsteps=40,
                        Seed=2, LogEvaluations=False, FolderToSave=str(tmp_path))
    for n in (1, 2):
        ic = np.loadtxt(tmp_path/("ic%i.trj" % n))
        en = np.loadtxt(tmp_path/("sampled_energy%i.log" % n))
        assert len(ic) == len(en)
        assert np.allclose(ic[:, 0], en[:, 0])
    # BarriersFromProfile reads the initial endpoint from the last bin
    assert np.loadtxt(tmp_path/"ic1.trj")[0, 1] == pytest.approx(1.0)
    assert np.loadtxt(tmp_path/"ic2.trj")[0, 1] == pytest.approx(0.0)

    IC, En, tau = LoadRun(str(tmp_path))
    assert tau == pytest.approx(1.0)
    assert len(IC) == len(En)
    assert np.all(np.isfinite(BootstrapBarriers(IC, En, N=5, NBoot=20)))

    # the TS structure of the ESI: the structures are read row for row
    Frames = LoadRunStructures(str(tmp_path))
    assert len(Frames) == len(En)
    Centres = BinnedProfile(IC, En, N=5, Grid="esi")[0]
    assert len(BinnedStructures(IC, Frames, N=5, Grid="esi")) == len(Centres)

#%% 6. The ORCA back end alongside it
def test_orca_carries_its_own_settings():

    """The accurate back end is still wired, with its own naming and its blocks."""

    Calculator = getORCACalculator(LevelOfTheory="hf-3c", charge=-1, mult=1,
                                   nProcs=8, PathToOrca="/nonexistent/orca")
    assert Calculator.parameters["charge"] == -1
    assert Calculator.parameters["mult"] == 1
    assert "%pal nprocs 8 end" in Calculator.parameters["orcablocks"]
    assert Calculator.parameters["orcasimpleinput"] == "hf-3c"

def test_orca_casscf_block_is_built():

    """An active space asks for a CASSCF block, an excited state for its roots."""

    Calculator = getORCACalculator(LevelOfTheory="", charge=0, mult=1, nProcs=1,
                                   PathToOrca="/nonexistent/orca",
                                   numElectrons=4, numOrbital=4, actState=1)
    Blocks = Calculator.parameters["orcablocks"]
    assert "%casscf" in Blocks
    assert "nel  4" in Blocks
    assert "iroot  2" in Blocks

def test_a_missing_orca_says_what_to_do():

    """Without a path ASE reports a configuration problem; say what fixes it."""

    with pytest.raises(RuntimeError, match="orcapath"):
        getORCACalculator(PathToOrca=None)

def test_the_two_back_ends_do_not_share_a_method_name():

    """hf-3c is a level of theory for ORCA and meaningless for the tight binding."""

    assert getORCACalculator(LevelOfTheory="hf-3c",
                             PathToOrca="/nonexistent/orca") is not None
    with pytest.raises(ValueError):
        normaliseMethod("hf-3c")

@pytest.mark.reaction("sn2")
def test_the_orca_file_interface_round_trips(Endpoints, tmp_path):

    """The input ASE writes and the output it reads back have to line up.

    ORCA itself is not needed to check that: a stand-in on the same file
    interface, answering a known energy, exercises the input writing, the
    process call, the output parsing and the Hartree to eV conversion, which
    is every part of the path this repository is responsible for.

    """

    Stub = tmp_path/"orca"
    Stub.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "Lines = open(sys.argv[1]).read().splitlines()\n"
        "Start = next(i for i, L in enumerate(Lines) if L.strip().startswith('*xyz'))\n"
        "Geom = []\n"
        "for L in Lines[Start+1:]:\n"
        "    if L.strip().startswith('*'):\n"
        "        break\n"
        "    Geom.append(L.split())\n"
        "print('* Single Point Calculation *')\n"
        "print('Number of atoms   ...  %i' % len(Geom))\n"
        "print('CARTESIAN COORDINATES (ANGSTROEM)')\n"
        "print('---')\n"
        "for P in Geom:\n"
        "    print(' %s %s %s %s' % (P[0], P[1], P[2], P[3]))\n"
        "print('')\n"
        "print('FINAL SINGLE POINT ENERGY  -3068.773692800535')\n"
        "print('                    ****ORCA TERMINATED NORMALLY****')\n")
    Stub.chmod(0o755)

    Structure = Endpoints[0]
    Structure.calc = getORCACalculator(LevelOfTheory="hf-3c", charge=-1,
                                       mult=1, nProcs=2,
                                       PathToOrca=str(Stub))
    Structure.calc.directory = tmp_path
    Energy = wrapGetPotentialEnergy(Structure)

    # the energy the stub reported, in Hartree, has to come back in eV
    assert Energy == pytest.approx(-3068.773692800535 * Hartree, rel=1e-12)

    # and the input it was handed has to carry the charge and the process count
    Written = (tmp_path/"orca.inp").read_text()
    assert "*xyz -1 1" in Written
    assert "%pal nprocs 2 end" in Written
    assert "! hf-3c" in Written
