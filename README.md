<p align="center">
    <h2 align="center">tReax_expedition: Gradient-Free Transition State Search<br>
    by Monte-Carlo Transition State Sampling</h2>
</p>

<p align="center">
    <a href="https://francisphillipsbuger.github.io/"><strong> Chien-Chai (Francis) Chang</strong></a>
</p>

<p align="center">
<img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" alt="Linux">
<img src="https://img.shields.io/badge/Python-3.12.13-3776AB?logo=python&logoColor=white" alt="Python 3.12.13">
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.8.0">
<img src="https://img.shields.io/badge/ASE-3.28.0-0A7BBB" alt="ASE 3.28.0">
</p>

## What is tReax_expedition?

**tReax_expedition** locates the **transition state** (TS) and an upper bound on the **barrier**
of a reaction from its two endpoints, using **energies only**: no gradients, no Hessians and no
saddle optimiser enter the search. It is built on SWS Monte-Carlo transition state sampling
(MCTSS), in which two walkers start from the reactant and the product and are biased towards
each other until they meet.



## Why tReax_expedition?

MCTSS samples the reactive region without any gradient, but the published protocol estimates
the barrier from hot data: a single imaginary-time step `tau` sets the temperature, the step
width and the stiffness of the pair coupling at once, which forces the sampling to be hot,
and binning a hot sample is a biased estimate of the saddle, not a noisy one.

tReax_expedition separates the two jobs:

- **Exploration** — hot MCTSS finds which states exist and roughly where the paths run.
- **Estimation** — cold, gradient-free continuation along a bond-change coordinate builds a
  certified chain, whose maximum is a verified upper bound on the barrier, and whose
  `q = 1/2` crossing is the TS estimate.

This keeps the method gradient-free end to end, so any energy back end, including ones that
offer no forces, can be used.

## Requirements
Required Python packages library include:
- `torch==2.8.0+cu128`
- `numpy==2.5.1`
- `scipy==1.17.1`
- `ase==3.28.0`
- `tblite==0.7.0`
- `matplotlib==3.11.0`
- `pandas==3.0.3`
- `sella==2.5.0`
- `pytest==9.1.1`

Alternatively, install the environment using the provided .txt and .YAML files at [environment](/environment/).

> [!NOTE]
> `sella` is used only by the grading scripts `analyse/reference_saddle.py` and
> `analyse/reference_minimum.py`; the search itself never calls it. ORCA is optional and is
> not installed by these files: pass its executable with `--orcapath` or set `TREAX_ORCA`.

## How to use tReax_expedition?

### Installation

* Clone this respostory
    ```
    git clone https://github.com/FrancisPhillipsBuger/tReax_expedition.git
    ```
* Install dependencies
    ```
    pip install -r environment/requirements.txt
    ```
    or
    ```
    conda env create -f environment/environment.yaml
    ```

### Repository layout

```
tReax_expedition/
├── run_treax_expedition.py       # CLI: staged search (or one MCTSS run) on a reaction
├── run_2d.py                     # CLI: 2D proton-transfer model
├── run_muller.py                 # CLI: scaled Mueller-Brown potential
├── run_two_paths.py              # CLI: two-channel potential
├── run_*.sh / run_*.bat          # wrappers reading their settings from environment variables
├── analytic_2d/
│   ├── potential.py              # proton transfer, Mueller-Brown and two-channel potentials
│   ├── sampler.py                # SWS Monte-Carlo trajectory on a 2D potential
│   ├── pipeline.py               # staged-search spec and grading against the exact answer
│   ├── reference.py              # exact committor from the backward Kolmogorov equation
│   ├── scan.py                   # (tau, Ninter) parameter scan
│   ├── analysis.py               # mean path and scan statistics
│   ├── pes_io.py                 # PES grid, 1D cuts and stationary-point writers
│   └── driver.py                 # command-line driver shared by the run_*.py scripts
├── treax_expedition/
│   ├── pipeline.py               # staged-search spec of a reaction (back end from meta.json)
│   ├── sampler.py                # single- and two-structure SWS Monte-Carlo on any ASE calculator
│   ├── io_xyz.py                 # endpoint reading, meta.json and output folders
│   ├── geometry.py               # mass-weighted RMSD, Kabsch alignment, SWS displacement
│   ├── calculators.py            # calculator handling and the free EMT potential
│   ├── xtb.py                    # GFN2-xTB back end (tblite)
│   ├── orca.py                   # ORCA back end, optionally CASSCF or TDDFT
│   └── amber.py                  # AMBER ff99SB energy of alanine dipeptide (energies only)
├── utils/
│   ├── mc/
│   │   ├── kernel.py             # the one delayed-acceptance Monte-Carlo move
│   │   ├── probability.py        # kinetic and potential terms of the acceptance probability
│   │   ├── system.py             # energy-only system interface, no gradient
│   │   ├── quench.py             # gradient-free quench (tau -> infinity)
│   │   ├── walk.py               # single-walker chain with an optional bias
│   │   ├── ladder.py             # tau ladder of the exploration stage
│   │   ├── ensemble.py           # ensemble of a coupled pair and its midpoint weights
│   │   └── pair.py               # equilibrium sampling of a coupled pair
│   ├── path/
│   │   ├── search.py             # the staged search S0-S6 shared by every system
│   │   ├── states.py             # states and their fingerprints
│   │   ├── label.py              # zero-temperature labels along MCTSS paths
│   │   ├── coordinate.py         # bond-change, label and committor progress coordinates
│   │   ├── continuation.py       # certified gradient-free continuation chains
│   │   ├── bisect.py             # separatrix crossing by bisection with quenches
│   │   └── tse.py                # transition state ensemble at the physical temperature
│   ├── committor/
│   │   ├── stage.py              # the committor stage S4
│   │   ├── model.py              # neural parametrisation of the committor
│   │   ├── descriptors.py        # differentiable descriptor layers
│   │   ├── loss.py               # variational (Ritz) and shot losses
│   │   ├── train.py              # optimisation of the committor network
│   │   ├── bias.py               # Kolmogorov bias potential
│   │   ├── zbias.py              # adaptive bias along the committor latent
│   │   ├── data.py               # biased blocks pooled by MBAR, basin data and shots
│   │   ├── shots.py              # shooting with unbiased Monte-Carlo chains
│   │   └── validate.py           # diagnostics of a learned committor on a 2D grid
│   └── surrogate/
│       └── physical.py           # a cheaper level of theory screening an expensive one
├── analyse/
│   ├── barrier.py                # ESI-style barrier estimators and kT -> 0 extrapolation
│   ├── analyse_barrier.py        # CLI: barrier analysis of finished MCTSS runs
│   ├── report.py                 # CLI: summary table of finished staged searches
│   ├── ts_metrics.py             # Kabsch / permutation RMSD and peptide dihedrals
│   ├── ts_compare.py             # CLI: grade located TSs against references
│   ├── plot_ts_compare.py        # CLI: figures and tables from the ts_compare output
│   ├── reference_saddle.py       # CLI: grading-only reference saddles (Sella + Hessian)
│   └── reference_minimum.py      # CLI: grading-only reference minima (Sella + Hessian)
├── constants/units.py            # fs / amu / Angstrom unit conversions
├── train/train_committor.py      # CLI: staged search followed by the committor stage S4
├── tests/                        # pytest suite
└── environment/                  # requirements.txt and environment.yaml
```

### Prepare a reaction

Reaction inputs are not shipped with this repository. A reaction is a folder under
`reaction/input/<name>/` (or any folder passed with `-f`):

```
reaction/input/<name>/
├── reactant.xyz                  # first endpoint
├── product.xyz                   # second endpoint, same atoms in the same order
├── meta.json                     # optional: what the .xyz files cannot say
└── ts_guess.xyz                  # optional: reference TS for grading, never read by the search
```

A minimal `meta.json`:

```json
{
  "charge": 0,
  "mult": 1,
  "calculator": "xtb",
  "T_phys": 298.15
}
```

| Key | Default | Meaning |
|-----|---------|---------|
| `charge` | 0 | total charge |
| `mult` | 1 | spin multiplicity |
| `calculator` | `xtb` | energy back end: `xtb`, `orca`, `amber` or `emt` |
| `level` | `hf-3c` | level of theory on the ORCA back end |
| `T_phys` | 298.15 | physical temperature of S4 and S6, in K |
| `fingerprint` | `{"kind": "bonds"}` | how states are told apart: `bonds` (bond graph) or `dihedrals` (conformers) |
| `reference` | — | reference barriers for grading only, e.g. `forward_kJmol`, `source` |

A single `.xyz` holding one or two structures can also be passed directly with `-f`.

### Run the search

Analytic 2D models (no input files needed):

```bash
python run_2d.py                  # 2D proton transfer
python run_muller.py              # Mueller-Brown
python run_two_paths.py           # two-channel potential
python run_2d.py --scan           # (tau, Ninter) scan instead of the search
```

Molecular reactions:

```bash
python run_treax_expedition.py -f <reaction>                           # S0-S3, S5
python run_treax_expedition.py -f <reaction> --stages search+committor # adds S4
python run_treax_expedition.py -f <reaction> --tse                     # adds S6
python run_treax_expedition.py -f <reaction> --stages explore -t 1.0 -s 10   # one MCTSS run
```

Committor on any system:

```bash
python train/train_committor.py --system {2d,muller,two_paths,<reaction>} [search options]
```

Outputs are written to `reaction/output/<name>/` by default (`--outdir` to change).

### Configuration

The main options of `run_treax_expedition.py` (see `--help` for all of them):

| Group | Option | Default | Meaning |
|-------|--------|---------|---------|
| Run | `--stages` | `search` | `search`, `search+committor` or `explore` |
| | `-f, --file` | `sn2` | reaction name, folder or `.xyz` |
| | `--nworkers` | 4 | worker processes of the search |
| | `--seed` | None | Monte-Carlo seed |
| Back end | `--calculator` | from `meta.json` | `xtb`, `orca`, `amber`, `emt` |
| | `-m, --method` | `gfn2-xtb` / `hf-3c` | parametrisation or level of theory |
| | `-o, --orcapath` | None | ORCA executable (or `TREAX_ORCA`) |
| | `--screen` | off | screen ORCA moves on GFN2-xTB (delayed acceptance, exact) |
| Search | `--seeds_per_rung` | 4 | MCTSS runs per rung of the tau ladder (S1) |
| | `--explore_steps` | 5000 | Monte-Carlo steps per MCTSS run (S1) |
| | `--max_chains` | 4 | continuation chains per pair of states (S3) |
| | `--crossing_votes` | 3 | quenches per label of the separatrix bisection (S5) |
| | `--committor_iterations` | 6 | biased iterations of the committor (S4) |
| | `--tse`, `--tse_steps` | off, 20000 | run S6 and its recorded sweeps |
| MCTSS | `-t, --tau` | 1.0 fs | each walker samples `exp(-tau V/hbar)` |
| | `-s, --steps` | 10 | steps of an `explore` run |
| | `--t12` | 10 / 3 | tau of the pair coupling in units of tau (explore / search) |
| | `--step_scale` | 1.0 | multiplier of the free-particle step width |
| | `--threshold` | 1.0 | matching threshold for the walkers to meet |

> [!IMPORTANT]
> The search never reads `ts_guess.xyz` and never uses a gradient. Gradients and Sella appear
> only in `analyse/reference_saddle.py` and `analyse/reference_minimum.py`, which build the
> stationary points that located TSs are graded against.

### Analysis

```bash
python analyse/report.py <search folders>            # summary of finished searches
python analyse/reference_saddle.py [reactions]       # reference saddles for grading
python analyse/ts_compare.py [search folders]        # RMSD / energy of located TSs vs references
python analyse/plot_ts_compare.py                    # figures from the ts_compare output
python analyse/analyse_barrier.py <run folders>      # ESI-style barrier analysis of MCTSS runs
```

### Tests

```bash
pytest
```

Tests that read a reaction are marked `@pytest.mark.reaction("<name>")` and are skipped when
`reaction/input/<name>` is absent.

## Acknowledgements

tReax_expedition builds on the MCTSS sampler of Tikhonov & Santra
(*J. Chem. Phys.* **164**, 014114); the committor stage follows the self-consistent, Kolmogorov-biased
committor of committor_2.0 and validates it with shots in the sense of AIMMD. GFN2-xTB energies
come from [tblite](https://github.com/tblite/tblite), and the grading-only reference saddles are
located with [Sella](https://github.com/zadorlab/sella).

> tReax_expedition is available under a [MIT License.](LICENSE)\
> ©Copyright 2026. All rights reserved.
