"""locbench tests: synthetic scenarios locate correctly, sweeps cache."""
import numpy as np
import pandas as pd

from aelib import GeigerLocator

from locbench.evaluate import Scenario, run_trials, summarize
from locbench.sweep import sweep
from locbench.synth import (corners_layout, coplanar_layout, grid_sources,
                            make_arrivals, random_layout, random_sources)


def test_grid_sources_regular_coverage():
    """Grid sources: cell-centre grid over the sensor bounding box, never
    on the boundary; interior=False adds the faces; constant axis (coplanar)
    collapses to one layer."""
    sensors = corners_layout(size=(0.1, 0.1, 0.2))
    g = grid_sources(sensors, spacing=0.025)          # 100/25=4, 200/25=8
    assert g.shape == (4 * 4 * 8, 3)
    P = np.stack(list(sensors.values()))
    assert (g > P.min(axis=0)).all() and (g < P.max(axis=0)).all()
    g2 = grid_sources(sensors, spacing=0.1, interior=False)   # 2x2x3
    assert g2.shape == (2 * 2 * 3, 3)
    corners = {tuple(np.round(p, 9)) for p in P}
    assert corners <= {tuple(np.round(p, 9)) for p in g2}
    s2 = coplanar_layout(6, extent=(0.2, 0.1), seed=1)
    g3 = grid_sources(s2, spacing=0.05)
    assert g3.shape[1] == 3 and np.allclose(g3[:, 2], 0.0)
    sc = Scenario(sensors=sensors, sources=g[::21], velocity=5300.0,
                  sigma=1e-6, seed=1)
    df = run_trials(GeigerLocator(sensors, velocity=5300.0), sc)
    assert df["err"].notna().all() and df["err"].median() < 0.01


def test_corners_layout_specimen_scenario():
    """Default specimen case: 100x100x200 mm, sensors on the 8 vertices."""
    sensors = corners_layout(size=(0.1, 0.1, 0.2))
    assert len(sensors) == 8
    P = np.stack(list(sensors.values()))
    assert np.allclose(P.min(axis=0), (-0.05, -0.05, -0.10))
    assert np.allclose(P.max(axis=0), (0.05, 0.05, 0.10))
    sc = Scenario(sensors=sensors, sources=random_sources(5, sensors, seed=1),
                  velocity=5300.0, sigma=1e-6, repeats=2, seed=2)
    df = run_trials(GeigerLocator(sensors, velocity=5300.0), sc)
    assert len(df) == 10
    assert df["err"].notna().all()
    assert df["err"].median() < 0.01          # ~mm at 1 us pick noise
    rect = corners_layout(size=(0.3, 0.2))    # 2-D size -> 4 corners
    assert len(rect) == 4 and all(len(p) == 2 for p in rect.values())


def test_clean_scenario_recovers_sources():
    sensors = random_layout(8, extent=(1.0, 1.0, 1.0), seed=1)
    sources = random_sources(5, sensors, seed=2)
    sc = Scenario(sensors=sensors, sources=sources, velocity=5300.0, seed=3)
    df = run_trials(GeigerLocator(sensors, velocity=5300.0), sc)
    assert len(df) == 5
    assert df["err"].notna().all()
    assert (df["err"] < 1e-6).all()          # exact arrivals -> ~0 error
    assert df["converged"].all()
    s = summarize(df)
    assert s["failures"] == 0 and s["trials"] == 5


def test_noisy_trials_error_scales():
    sensors = random_layout(8, extent=(1.0, 1.0, 1.0), seed=1)
    sources = random_sources(4, sensors, seed=2)
    sc = Scenario(sensors=sensors, sources=sources, velocity=5300.0,
                  sigma=1e-6, repeats=3, seed=4)
    df = run_trials(GeigerLocator(sensors, velocity=5300.0), sc)
    assert len(df) == 12
    assert df["err"].median() < 0.01        # < 1 cm at 1 us pick noise
    assert df["t0_err"].median() < 5e-6


def test_outliers_dropout_and_coplanar_still_run():
    sensors = random_layout(10, extent=(1.0, 1.0, 1.0), seed=5)
    sources = random_sources(4, sensors, seed=6)
    sc = Scenario(sensors=sensors, sources=sources, velocity=5300.0,
                  sigma=1e-6, outlier_p=0.2, drop_p=0.2, repeats=2, seed=7)
    df = run_trials(GeigerLocator(sensors, velocity=5300.0), sc)
    assert len(df) == 8 and df["n_picks"].max() <= 10
    assert df["err"].notna().any()          # at least some trials survive

    sensors2 = coplanar_layout(8, extent=(1.0, 1.0), plane="z", height=0.0,
                               seed=8)
    src2 = random_sources(3, sensors2, seed=9)   # on-plane (x, y, 0)
    sc2 = Scenario(sensors=sensors2, sources=src2, velocity=5300.0,
                   sigma=1e-6, seed=10)
    loc2 = GeigerLocator(sensors2, velocity=5300.0, mode="2d",
                         fixed_coords={"z": 0.0})
    df2 = run_trials(loc2, sc2)
    assert df2["err"].notna().all()
    assert df2["err"].median() < 0.01      # ~mm typical; edge sources can
    assert (df2["err"] < 0.05).all()       # exceed this via geometric dilution


def test_make_arrivals_degradation_model():
    sensors = random_layout(6, extent=(1.0, 1.0, 1.0), seed=11)
    src = np.array([0.1, -0.1, 0.05])
    arr, info = make_arrivals(sensors, src, velocity=5000.0, t0=1.0,
                              drop_p=0.5, outlier_p=1.0,
                              outlier_jitter=1e-3, seed=12)
    assert len(arr) <= 6 and len(arr) >= 0
    assert set(info["dropped"]) | set(arr) <= set(sensors)
    exact, _ = make_arrivals(sensors, src, velocity=5000.0, t0=1.0)
    for ch in arr:                          # kept picks are polluted but bounded
        assert abs(arr[ch] - exact[ch]) <= 1e-3 + 1e-9


def test_sweep_runs_and_caches(tmp_path):
    sensors = random_layout(8, seed=1)
    sources = random_sources(4, sensors, seed=2)
    sc = Scenario(sensors=sensors, sources=sources, velocity=5300.0,
                  sigma=1e-6, seed=3)
    grid = {"grid_size": [8, 24]}
    df = sweep(GeigerLocator, grid, sc, cache_dir=str(tmp_path),
               locator_kwargs={"velocity": 5300.0})
    assert set(df["grid_size"]) == {8, 24}
    assert len(df) == 8
    assert df.groupby("grid_size")["runtime_ms"].mean().loc[8] >= 0
    df2 = sweep(GeigerLocator, grid, sc, cache_dir=str(tmp_path),
                locator_kwargs={"velocity": 5300.0})
    pd.testing.assert_frame_equal(df, df2)   # cache hit, identical table
