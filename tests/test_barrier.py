#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Tests for the barrier estimators, using synthetic profiles whose answer is
known analytically.

Created on: Tue Aug 25 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import numpy as np
import pytest

import scipy.spatial.transform as spst
from ase.build import molecule

from analyse.barrier import (BarriersFromProfile, BinnedProfile,
                                    BinnedStructures, BootstrapBarriers,
                                    ExtrapolateToZeroKT, SpreadErrors,
                                    TauFromTimeColumn)

#%% 1. Binning
def test_binned_profile_recovers_a_known_barrier():

    """A noiseless double well sampled densely must give back its own barrier."""

    ic = np.linspace(-1.0, 1.0, 4001)
    # reactant at +1 (last bin), product at -1 (first bin), barrier at 0
    en = np.where(ic > 0, 0.0, -0.5) + 0.3*np.exp(-(ic/0.15)**2)
    # an even number of bins puts an edge, not a centre, on the step at zero
    Centres, Mean, _, Minimum, Counts = BinnedProfile(ic, en, N=40)

    assert np.all(Counts > 0)
    assert len(Centres) == 40
    dEf, dEb = BarriersFromProfile(Mean)
    assert dEf == pytest.approx(0.3, abs=0.02)
    assert dEb == pytest.approx(0.8, abs=0.02)

def test_reactant_is_the_last_bin():

    """The SN2 convention puts the reactant at large reaction coordinate."""

    profile = np.array([-1.0, 0.0, 2.0, 0.5, -0.2])
    dEf, dEb = BarriersFromProfile(profile)
    assert dEf == pytest.approx(2.2)   # top minus the last bin
    assert dEb == pytest.approx(3.0)   # top minus the first bin
    # their difference is the reaction energy
    assert dEb - dEf == pytest.approx(profile[-1] - profile[0])

def test_empty_bins_are_dropped():

    """Bins with no samples must not appear in the profile."""

    ic = np.concatenate([np.linspace(0.0, 0.1, 200), np.linspace(0.9, 1.0, 200)])
    en = np.zeros_like(ic)
    Centres, Mean, _, _, Counts = BinnedProfile(ic, en, N=21)
    assert np.all(Counts > 0)
    assert len(Centres) < 21

def test_end_bins_are_averages_rather_than_single_points():

    """

    The end bins are the reactant and product references of the barriers, so
    each has to hold its share of the sample and every point has to be counted
    exactly once. A width of (max-min)/(N-1) left the largest point alone in
    the last bin, and the forward barrier was then measured from one sample.

    """

    rng = np.random.default_rng(0)
    for N in (7, 15, 40):
        ic = rng.uniform(-1.5, 1.1, 500)
        en = rng.normal(0.0, 0.3, 500)
        Centres, Mean, _, _, Counts = BinnedProfile(ic, en, N=N)

        assert len(Centres) == N
        assert Counts.sum() == len(ic)
        assert Counts[0] > 1 and Counts[-1] > 1
        # equal widths tiling [min, max]
        dx = (ic.max() - ic.min())/N
        assert np.allclose(np.diff(Centres), dx)
        assert Centres[0] - 0.5*dx == pytest.approx(ic.min())
        assert Centres[-1] + 0.5*dx == pytest.approx(ic.max())
        # the last bin holds exactly the points of its own interval
        Last = ic >= ic.max() - dx*(1.0 - 1e-12)
        assert Mean[-1] == pytest.approx(en[Last].mean())

def test_a_coordinate_that_never_moved_has_no_profile():

    """A zero-width range must not divide by zero."""

    Centres, _, _, _, Counts = BinnedProfile(np.zeros(10), np.ones(10), N=5)
    assert len(Centres) == 0 and len(Counts) == 0

def test_the_esi_grid_keeps_only_the_maximum_in_the_last_bin():

    """

    The ESI's bins are (max - min)/(N - 1) wide, so the last one starts at the
    maximum and holds the unmoved initial endpoint alone, not a tail average.

    """

    ic = np.concatenate([np.linspace(0.0, 0.99, 100), np.ones(3)])
    en = np.concatenate([np.zeros(100), np.full(3, 7.0)])
    _, Mean, _, _, Counts = BinnedProfile(ic, en, N=5, Grid="esi")
    assert Counts[-1] == 3
    assert Mean[-1] == pytest.approx(7.0)
    _, Mean, _, _, Counts = BinnedProfile(ic, en, N=5)
    assert Counts[-1] > 3

def test_an_unknown_grid_is_refused():

    """A typo must not fall back silently to one of the two grids."""

    with pytest.raises(ValueError, match="Grid"):
        BinnedProfile(np.linspace(0.0, 1.0, 10), np.zeros(10), Grid="ESI ")

#%% 2. ESI errors and structures
def test_spread_errors_add_the_bin_spreads_in_quadrature():

    """The ESI's error combines the spread of the top bin with that of each end bin."""

    Mean = np.array([-1.0, 0.0, 2.0, 0.5, -0.2])
    Std = np.array([0.3, 0.1, 0.4, 0.2, 0.5])
    dEfErr, dEbErr = SpreadErrors(Mean, Std)
    assert dEfErr == pytest.approx(np.sqrt(0.4**2 + 0.5**2))
    assert dEbErr == pytest.approx(np.sqrt(0.4**2 + 0.3**2))

def test_the_bin_average_of_moved_copies_is_the_structure():

    """

    Rotated and translated copies of one molecule average back to it, since
    every copy is overlaid on the running average before it is added.

    """

    Base = molecule("CH3Cl")
    Frames = []
    for k in range(0, 6):
        Copy = Base.copy()
        Copy.set_positions(spst.Rotation.random(random_state=k).apply(Base.get_positions()) + k)
        Frames.append(Copy)
    ic = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])

    Averages = BinnedStructures(ic, Frames, N=2)
    assert len(Averages) == 2
    R = Base.get_positions()
    Distances = lambda X: np.linalg.norm(X[:, None] - X[None], axis=-1)
    for Average in Averages:
        assert np.allclose(Distances(Average.get_positions()), Distances(R), atol=1e-8)

def test_structures_follow_the_bins_of_the_profile():

    """One average per occupied bin, in the order of the profile arrays."""

    Base = molecule("CH3Cl")
    ic = np.concatenate([np.linspace(0.0, 0.1, 5), np.linspace(0.9, 1.0, 5)])
    Frames = [Base.copy() for _ in ic]
    Centres, _, _, _, _ = BinnedProfile(ic, np.zeros_like(ic), N=7)
    assert len(BinnedStructures(ic, Frames, N=7)) == len(Centres)

#%% 3. Extrapolation
def test_extrapolation_recovers_a_known_intercept():

    """A quantity linear in 1/tau must extrapolate back to its intercept."""

    taus = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    a, b = 5.5, 120.0
    values = a + b/taus

    Intercept, Slope, R2, Err = ExtrapolateToZeroKT(taus, values)
    assert Intercept == pytest.approx(a, abs=1e-8)
    assert Slope == pytest.approx(b, abs=1e-8)
    assert R2 == pytest.approx(1.0, abs=1e-12)
    assert Err == pytest.approx(0.0, abs=1e-6)

def test_extrapolation_needs_three_points():

    """Two points cannot support a fit with an uncertainty."""

    Intercept, _, _, _ = ExtrapolateToZeroKT([2.0, 4.0], [1.0, 2.0])
    assert np.isnan(Intercept)

def test_extrapolation_ignores_non_finite_values():

    """NaN entries must be dropped rather than poisoning the fit."""

    taus = [2.0, 3.0, 4.0, 5.0, 6.0]
    values = [5.5 + 120.0/t for t in taus]
    values[2] = np.nan
    Intercept, _, _, _ = ExtrapolateToZeroKT(taus, values)
    assert Intercept == pytest.approx(5.5, abs=1e-8)

def test_weighted_extrapolation_reports_the_statistics_of_its_own_fit():

    """

    With per-run errors the line is a weighted fit, so its error bar and R2
    must be those of the weighted fit. numpy's polyfit, with weights 1/sigma
    and a covariance scaled by the reduced chi squared, is the reference.

    """

    taus = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 8.0])
    x = 1.0/taus
    noise = np.array([0.05, -0.08, 0.03, 0.9, -0.02, 0.04])
    values = 5.5 + 120.0*x + noise
    errors = np.array([0.05, 0.05, 0.05, 1.0, 0.05, 0.05])

    Intercept, Slope, R2, Err = ExtrapolateToZeroKT(taus, values, Errors=errors)
    coef, cov = np.polyfit(x, values, 1, w=1.0/errors, cov=True)
    assert Slope == pytest.approx(coef[0], rel=1e-10)
    assert Intercept == pytest.approx(coef[1], rel=1e-10)
    assert Err == pytest.approx(np.sqrt(cov[1, 1]), rel=1e-8)

    # the one run with a large error barely counts, so the fit stays good
    w2 = 1.0/errors**2
    res = values - np.polyval(coef, x)
    mean = np.sum(w2*values)/np.sum(w2)
    assert R2 == pytest.approx(1.0 - np.sum(w2*res**2)/np.sum(w2*(values - mean)**2), rel=1e-10)

#%% 4. Run loading
def test_tau_recovered_from_the_time_column(tmp_path):

    """The samplers write i*tau, so consecutive rows are spaced by tau."""

    fn = tmp_path/"sampled_energy1.log"
    tau = 3.0
    rows = ["%8.2f %15.6f %2i\n" % (i*tau, -1.0, 1) for i in range(20)]
    fn.write_text("".join(rows))
    assert TauFromTimeColumn(str(fn)) == pytest.approx(tau)

#%% 5. Bootstrap
def test_bootstrap_is_reproducible_and_brackets_the_estimate():

    """The same seed must give the same error, and the error must be finite."""

    rng = np.random.default_rng(1)
    ic = rng.uniform(-1.0, 1.0, 4000)
    en = 0.3*np.exp(-(ic/0.2)**2) + np.where(ic > 0, 0.0, -0.5) + rng.normal(0, 0.01, 4000)

    # an even number of bins puts an edge, not a centre, on the step at zero
    a = BootstrapBarriers(ic, en, N=20, NBoot=50, Seed=7)
    b = BootstrapBarriers(ic, en, N=20, NBoot=50, Seed=7)
    assert a == pytest.approx(b)
    assert np.isfinite(a[1]) and a[1] > 0.0
    assert a[0] == pytest.approx(0.3, abs=0.05)

#%% 6. Statistics over trials of the 2D scan
def test_scan_error_is_the_standard_error_of_the_mean():

    """The error over trials uses the sample variance, and needs two trials."""

    from analytic_2d.analysis import ConvPairMeans

    v = np.array([1.0, 2.5, 0.5, 3.0, 2.0])
    Mean, Err = ConvPairMeans(float(v.sum()), float(np.sum(v*v)), len(v))
    assert Mean == pytest.approx(v.mean())
    assert Err == pytest.approx(v.std(ddof=1)/np.sqrt(len(v)))

    Mean, Err = ConvPairMeans(2.0, 4.0, 1)
    assert Mean == pytest.approx(2.0) and np.isnan(Err)
