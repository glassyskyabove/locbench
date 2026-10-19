"""locbench tests: synthetic scenarios locate correctly, sweeps cache."""
import numpy as np
import pandas as pd

from aelib import GeigerLocator

from locbench.evaluate import Scenario, run_trials, summarize
from locbench.sweep import sweep
from locbench.synth import (corners_layout, coplanar_layout, grid_sources,
                            make_arrivals, parse_layout, random_int_layout,
                            random_layout, random_sources)


def test_random_int_layout_mm_grid():
    """Random integer layouts: all coordinates integers inside the extent,
    spacing floor respected, deterministic per seed, 2-D extents work."""
    lay = random_int_layout(8, extent=(100, 100, 200), seed=7)
    assert len(lay) == 8
    for p in lay.values():
        assert len(p) == 3
        assert all(isinstance(v, int) for v in p)
        assert all(0 <= v <= e for v, e in zip(p, (100, 100, 200)))
    pts = np.asarray(list(lay.values()), dtype=float)
    from itertools import combinations
    assert all(np.linalg.norm(a - b) >= 5.0            # floor: 5% of 100mm
               for a, b in combinations(pts, 2))
    lay2 = random_int_layout(8, extent=(100, 100, 200), seed=7)
    assert lay == lay2                                  # deterministic
    lay3 = random_int_layout(6, extent=(80, 60), seed=1)
    assert all(len(p) == 2 and all(isinstance(v, int) and 0 <= v <= 80
                                   for v in p[:1]) for p in lay3.values())


def test_parse_layout_custom_sensors():
    """Custom layout text: one sensor per line, comma-separated, unit
    conversion, comments/full-width commas tolerated, clear errors."""
    txt = "# 台阵\n0,0\n100,0\n0,100\n100,100"
    s = parse_layout(txt, n_components=2, to_m=1e-3)
    assert len(s) == 4
    assert np.allclose(s[1], (0.0, 0.0)) and np.allclose(s[4], (0.1, 0.1))
    s2 = parse_layout("0，0，0\n50,50,50", n_components=3, to_m=1e-3)
    assert np.allclose(s2[2], (0.05, 0.05, 0.05))
    try:
        parse_layout("1,2,3\n4,5", n_components=2)
        raise AssertionError("wrong component count must raise")
    except ValueError:
        pass
    try:
        parse_layout("a,b\nc,d", n_components=2)
        raise AssertionError("non-numeric row must raise")
    except ValueError:
        pass
    # 2-D custom layout locates end-to-end (pure 2-D sensors, mode='2d')
    sc = Scenario(sensors=s, sources=random_sources(3, s, seed=1),
                  velocity=5300.0, sigma=1e-6, seed=2)
    df = run_trials(GeigerLocator(s, velocity=5300.0, mode="2d"), sc)
    assert df["err"].notna().all() and df["err"].median() < 0.01


def test_corner_and_2d_layout_variants():
    """2-D corners give 4 sensors of dimension 2; 2-D random layout is
    pure 2-D and locates with mode='2d' (no fixed axis needed)."""
    rect = corners_layout(size=(0.1, 0.1))
    assert len(rect) == 4 and all(len(p) == 2 for p in rect.values())
    lay = random_layout(6, extent=(0.1, 0.1), min_spacing=0.005, seed=3)
    assert all(len(p) == 2 for p in lay.values())
    sc = Scenario(sensors=lay, sources=random_sources(3, lay, seed=4),
                  velocity=5300.0, sigma=1e-6, seed=5)
    df = run_trials(GeigerLocator(lay, velocity=5300.0, mode="2d"), sc)
    assert df["err"].notna().all() and df["err"].median() < 0.02


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


def test_display_unit_scaling():
    """Distance displays scale m -> mm without touching the data layer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from locbench.plots import plot_err_hist, plot_layout

    sensors = corners_layout(size=(0.1, 0.1, 0.2))
    sc = Scenario(sensors=sensors, sources=random_sources(3, sensors, seed=1),
                  velocity=5300.0, sigma=1e-6, seed=2)
    df = run_trials(GeigerLocator(sensors, velocity=5300.0), sc)
    for unit, factor in (("m", 1.0), ("mm", 1e3)):
        fig = plot_err_hist(df, unit=unit, factor=factor)
        assert fig.axes[0].get_xlabel() == f"定位误差 ({unit})"
        lim = fig.axes[0].get_xlim()
        fig2 = plot_layout(sensors, sc.sources, None, unit=unit, factor=factor)
        assert fig2.axes[0].get_xlabel() == f"x ({unit})"
        pts = fig2.axes[0].collections[0].get_offsets() if unit == "2d" else None
        plt.close("all")
        if unit == "mm":
            assert lim[1] > 1.0          # mm scale: errors render > 1
    from locbench.plots import display_table
    show = display_table(df, "mm", 1e3)
    assert "定位误差 (mm)" in show.columns
    assert np.allclose(show["定位误差 (mm)"].dropna(),
                       df["err"].dropna() * 1e3)
