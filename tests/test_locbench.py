"""locbench tests: synthetic scenarios locate correctly, sweeps cache."""
import numpy as np
import pandas as pd

from aelib import GeigerLocator, RefractionLocator

from locbench.evaluate import Scenario, run_trials, summarize
from locbench.sweep import sweep
from locbench.synth import (composite_layout, corners_layout,
                            coplanar_layout, feature_layout, grid_sources,
                            head_layout, make_arrivals, parse_layout,
                            random_int_layout, random_layout, random_sources)


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
    txt = "# 传感器阵列\n0,0\n100,0\n0,100\n100,100"
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
    """Default specimen case: 100x100x200 mm, sensors on the 8 vertices,
    whole specimen in the all-positive octant."""
    sensors = corners_layout(size=(0.1, 0.1, 0.2))
    assert len(sensors) == 8
    P = np.stack(list(sensors.values()))
    assert (P >= 0).all()
    assert np.allclose(P.min(axis=0), (0.0, 0.0, 0.0))
    assert np.allclose(P.max(axis=0), (0.1, 0.1, 0.2))
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
    # tol=1e-12 for the deep-convergence check: the 1e-6 default stops
    # ~0.1 mm short of the optimum (us-scale early stopping by design)
    df = run_trials(GeigerLocator(sensors, velocity=5300.0, tol=1e-12), sc)
    assert len(df) == 5
    assert df["err"].notna().all()
    assert (df["err"] < 1e-6).all()          # exact arrivals -> ~0 error
    assert df["converged"].all()
    s = summarize(df)
    assert s["failures"] == 0 and s["trials"] == 5
    # default tol=1e-6 on the same exact-arrival data: sub-mm early stop
    df_def = run_trials(GeigerLocator(sensors, velocity=5300.0), sc)
    assert (df_def["err"] < 1e-3).all()


def test_noisy_trials_error_scales():
    sensors = random_layout(8, extent=(1.0, 1.0, 1.0), seed=1)
    sources = random_sources(4, sensors, seed=2)
    sc = Scenario(sensors=sensors, sources=sources, velocity=5300.0,
                  sigma=1e-6, repeats=3, seed=4)
    df = run_trials(GeigerLocator(sensors, velocity=5300.0), sc)
    assert len(df) == 12
    assert df["err"].median() < 0.01        # < 1 cm at 1 us pick noise
    assert df["t0_err"].median() < 5e-6


def test_dropout_and_coplanar_still_run():
    sensors = random_layout(10, extent=(1.0, 1.0, 1.0), seed=5)
    sources = random_sources(4, sensors, seed=6)
    sc = Scenario(sensors=sensors, sources=sources, velocity=5300.0,
                  sigma=1e-6, drop_p=0.2, repeats=2, seed=7)
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
                              drop_p=0.5, seed=12)
    assert set(arr) <= set(sensors)
    assert set(info["dropped"]) | set(arr) == set(sensors)
    exact, _ = make_arrivals(sensors, src, velocity=5000.0, t0=1.0)
    for ch in arr:                          # sigma=0 -> kept picks exact
        assert arr[ch] == exact[ch]


def test_arrival_quantization_by_sampling_rate():
    """fs quantizes arrival times to the sampling period (10 MHz -> 0.1 us
    ticks); fs=0 keeps them continuous."""
    sensors = random_layout(6, extent=(1.0, 1.0, 1.0), seed=21)
    src = np.array([0.1, -0.2, 0.3])
    arr, _ = make_arrivals(sensors, src, velocity=5000.0, t0=1.0,
                           fs=10e6, seed=22)
    T = 1.0 / 10e6
    for t in arr.values():                  # every pick sits on the clock
        assert abs(t / T - round(t / T)) < 1e-6
    exact, _ = make_arrivals(sensors, src, velocity=5000.0, t0=1.0)
    assert max(abs(arr[c] - exact[c]) for c in arr) <= T / 2 + 1e-12
    cont, _ = make_arrivals(sensors, src, velocity=5000.0, t0=1.0, fs=0.0)
    assert cont == exact                    # fs=0 -> continuous, exact
    sc = Scenario(sensors=sensors, sources=src[None, :], velocity=5000.0,
                  fs=10e6, seed=23)         # flows through run_trials
    df = run_trials(GeigerLocator(sensors, velocity=5000.0), sc)
    assert df["err"].notna().all() and df["err"].iloc[0] < 0.02


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


def test_feature_layout_edges_and_faces():
    """Geometric-feature layouts: an edge feature covers ALL edges
    parallel to that axis (4 per axis in 3-D, 2 in 2-D), each carrying
    interior division points (k/(n+1)·span, corners never duplicated);
    a face feature covers the TWO faces with that normal (one centre
    sensor each); vertices=True adds every corner; 2-D rejects z
    features; sources decouple from the sensor bbox via box=."""
    lay = feature_layout((0.1, 0.1, 0.2), edges={"横边": 4})
    assert len(lay) == 16                       # 4 条 x 向边 × 4 点
    pts = np.round(np.array(list(lay.values())), 9)
    assert set(pts[:, 0]) == {0.02, 0.04, 0.06, 0.08}   # 内部等分点
    assert set(pts[:, 1]) == {0.0, 0.1} and set(pts[:, 2]) == {0.0, 0.2}
    top = feature_layout((0.1, 0.1, 0.2), faces=["顶面"])
    assert np.allclose(list(top.values()),
                       [(0.05, 0.05, 0.0), (0.05, 0.05, 0.2)])  # 两个面心
    two = feature_layout((0.1, 0.1, 0.2), faces=["侧面", "正面"])
    assert np.allclose(list(two.values()),
                       [(0.0, 0.05, 0.1), (0.1, 0.05, 0.1),
                        (0.05, 0.0, 0.1), (0.05, 0.1, 0.1)])
    lv = feature_layout((0.1, 0.1, 0.2), vertices=True)      # 8 角点
    assert len(lv) == 8
    assert {tuple(np.round(p, 9)) for p in lv.values()} == \
        {tuple(np.round(p, 9)) for p in corners_layout(
            size=(0.1, 0.1, 0.2)).values()}
    mix = feature_layout((0.1, 0.1, 0.2), edges={"横边": 3}, vertices=True)
    assert len(mix) == 20                       # 4×3 边点 + 8 角点，无重复
    assert len({tuple(np.round(p, 9)) for p in mix.values()}) == 20
    rect = feature_layout((0.1, 0.1), edges={"横边": 2, "纵边": 3},
                          vertices=True)
    assert len(rect) == 14                      # 2×2 + 2×3 + 4 角点
    assert all(len(p) == 2 for p in rect.values())
    for kwargs in ({"edges": {"垂边": 2}}, {"faces": ["顶面"]}):
        try:
            feature_layout((0.1, 0.1), **kwargs)
            raise AssertionError("2-D must reject z features")
        except ValueError:
            pass
    # 端到端：横边框架传感器阵列（4 条 x 向边各 3 点），震源布满试样盒
    sensors = feature_layout((0.1, 0.1, 0.2), edges={"横边": 3})
    assert len(sensors) == 12
    src = grid_sources(spacing=(0.02, 0.02, 0.04),
                       box=((0, 0, 0), (0.1, 0.1, 0.2)))
    assert src.shape == (5 * 5 * 5, 3)          # 不塌缩到边传感器阵列的包围盒
    rs = random_sources(50, box=((0, 0, 0), (0.1, 0.1, 0.2)), seed=3)
    assert rs.shape == (50, 3) and (rs >= 0).all()
    assert (rs <= np.array([0.1, 0.1, 0.2])).all()
    sc = Scenario(sensors=sensors, sources=src[::21], velocity=5300.0,
                  sigma=1e-6, seed=1)
    df = run_trials(GeigerLocator(sensors, velocity=5300.0), sc)
    assert df["err"].notna().all()              # 边框架传感器阵列可定位
    # 兼容：不给 box 时仍用传感器包围盒
    assert grid_sources(sensors, spacing=0.05).shape[1] == 3
    assert random_sources(5, sensors, seed=1).shape == (5, 3)


def test_sweep_plot_tol_log_axis():
    """tol 扫描图 x 轴：全正取值 -> log；含 0（关闭判据档）-> symlog。"""
    import matplotlib
    matplotlib.use("Agg")
    from locbench.plots import plot_metric_vs_param
    rows = []
    for tol in (0.0, 1e-8, 1e-7, 1e-6, 1e-5):      # 默认 tol 扫描取值
        rows += [{"tol": tol, "err": 0.001 + 1e5 * tol, "runtime_ms": 1.0}
                 for _ in range(3)]
    df = pd.DataFrame(rows)
    ax = plot_metric_vs_param(df, "tol", metric="err").axes[0]
    assert ax.get_xscale() == "symlog"             # 含 0 -> symlog
    df2 = df[df["tol"] > 0]
    ax2 = plot_metric_vs_param(df2, "tol", metric="err").axes[0]
    assert ax2.get_xscale() == "log"               # 全正 -> log


def test_parse_values_equidistant_shorthand():
    """扫描取值解析：等间距简写（括号式/冒号式、全角字符）与普通项混用，
    含两端；非法表达式给出中文 ValueError。"""
    from locbench.sweep import parse_values
    # 括号式（含全角）：2000..4000 间隔 100 -> 21 个
    v = parse_values("（2000，4000，100）")
    assert len(v) == 21 and v[0] == 2000.0 and v[-1] == 4000.0
    assert v[1] == 2100.0 and v[10] == 3000.0
    # 冒号式 + 与普通项混用 + 小数/负数
    v2 = parse_values("4, 8, 10:20:2.5, 48")
    assert v2[:2] == [4.0, 8.0] and v2[-1] == 48.0
    assert v2[2:7] == [10.0, 12.5, 15.0, 17.5, 20.0]  # 整除时含终点
    # 含端点不整除：1 到 2 间隔 0.3 -> 1, 1.3, 1.6, 1.9
    assert parse_values("(1,2,0.3)") == [1.0, 1.3, 1.6, 1.9]
    # 负步长
    assert parse_values("5:1:-2") == [5.0, 3.0, 1.0]
    for bad in ("(1,2)", "1:2", "(1,10,0)", "(10,1,1)", "(1,2,a)",
                "1,2,)", "((1,2,3)"):
        try:
            parse_values(bad)
            raise AssertionError(f"必须报错：{bad}")
        except ValueError:
            pass


def test_sweep_cache_prunes_old_files(tmp_path):
    """clean_cache：按 mtime 删除 7 天前的 sweep_*.pkl，新文件与非
    sweep 文件不动；不存在的目录返回 0。"""
    import os
    import time as time_mod
    from locbench.sweep import clean_cache
    d = tmp_path / "results"
    d.mkdir()
    old = d / "sweep_old.pkl"
    old.write_bytes(b"x")
    fresh = d / "sweep_fresh.pkl"
    fresh.write_bytes(b"x")
    other = d / "notes.txt"
    other.write_bytes(b"x")
    ago = time_mod.time() - 8 * 86400
    os.utime(old, (ago, ago))                 # 8 天前 -> 过期
    assert clean_cache(str(d), max_age_days=7) == 1
    assert not old.exists()
    assert fresh.exists() and other.exists()
    assert clean_cache(str(d)) == 0           # 幂等
    assert clean_cache(str(tmp_path / "nope")) == 0


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
        if unit == "mm":                    # 等轴：200mm 轴 ≈ 2× 100mm 轴
            ba = fig2.axes[0].get_box_aspect()
            assert 1.7 < ba[2] / ba[0] < 2.1
        pts = fig2.axes[0].collections[0].get_offsets() if unit == "2d" else None
        plt.close("all")
        if unit == "mm":
            assert lim[1] > 1.0          # mm scale: errors render > 1
    # 发散的定位解不参与取限也不被画出：matplotlib 三维不按坐标限
    # 裁剪散点，超视野的点会被投影进图内（反复调参时"图变形"的根源）
    bad = sc.sources.copy()
    bad[1] = (0.9, 0.4, 0.1)                # 跑到 900mm 外的发散解
    fig3 = plot_layout(sensors, sc.sources, bad, unit="mm", factor=1e3)
    ax3 = fig3.axes[0]
    assert max(abs(np.array(ax3.get_xlim()))).max() < 200
    assert max(abs(np.array(ax3.get_zlim()))).max() < 300
    assert 1.7 < ax3.get_box_aspect()[2] / ax3.get_box_aspect()[0] < 2.1
    red = next(c for c in ax3.collections if c.get_label() == "定位结果")
    assert len(red.get_offsets()) == len(bad) - 1    # 发散解未画
    # 误差线段按下标配对：过滤后各线段仍从自己的真值震源出发
    kept = np.flatnonzero((np.abs(bad) < 0.5).all(axis=1))  # 排除野点(米制)
    assert len(ax3.lines) == len(kept)
    for j, i in enumerate(kept):
        start = np.array(ax3.lines[j].get_data_3d())[:, 0]
        assert np.allclose(start, np.asarray(sc.sources)[i] * 1e3)
    from locbench.synth import corners_layout as _cl
    rect = _cl(size=(0.1, 0.1))
    fig4 = plot_layout(rect, random_sources(3, rect, seed=1),
                       np.array([[5.0, -3.0]]), unit="m", factor=1.0)
    assert abs(np.array(fig4.axes[0].get_xlim())).max() < 1.0   # 二维同理
    from locbench.plots import display_table
    show = display_table(df, "mm", 1e3)
    assert "定位误差 (mm)" in show.columns
    assert np.allclose(show["定位误差 (mm)"].dropna(),
                       df["err"].dropna() * 1e3)
    # 时间单位换算：t0 误差 (s -> ms)
    show2 = display_table(df, "m", 1.0, "ms", 1e3)
    assert "发震时刻误差 (ms)" in show2.columns
    assert "残差 RMS (ms)" in show2.columns
    assert np.allclose(show2["发震时刻误差 (ms)"].dropna(),
                       df["t0_err"].dropna() * 1e3)


def test_head_layout_sensors_outside_on_one_axis():
    """head_layout: six faces x per_face sensors, each strictly outside the
    box on exactly ONE axis at the offset distance (RefractionLocator's
    admissible layout); per_face=1 takes face centres; deterministic."""
    lay = head_layout((0.1, 0.1, 0.2), per_face=2, offset=0.012, seed=3)
    assert len(lay) == 12
    lo = np.zeros(3)
    hi = np.array([0.1, 0.1, 0.2])
    per_face = {a: {0: 0, 1: 0} for a in range(3)}
    for p in lay.values():
        out = [a for a in range(3) if p[a] < lo[a] or p[a] > hi[a]]
        assert len(out) == 1, f"sensor {p} outside on {out}"
        a = out[0]
        side = 1 if p[a] > hi[a] else 0
        assert abs(abs(p[a] - (hi[a] if side else lo[a])) - 0.012) < 1e-12
        in_plane = [p[i] for i in range(3) if i != a]
        bounds = [(lo[i], hi[i]) for i in range(3) if i != a]
        assert all(l < v < h for v, (l, h) in zip(in_plane, bounds))
        per_face[a][side] += 1
    assert all(c == 2 for f in per_face.values() for c in f.values())
    lay2 = head_layout((0.1, 0.1, 0.2), per_face=2, offset=0.012, seed=3)
    assert lay2 == lay                                          # deterministic
    centres = head_layout((0.1, 0.1, 0.1), per_face=1, offset=0.01, seed=9)
    assert np.allclose(list(centres.values()),
                       [(-0.01, 0.05, 0.05), (0.11, 0.05, 0.05),
                        (0.05, -0.01, 0.05), (0.05, 0.11, 0.05),
                        (0.05, 0.05, -0.01), (0.05, 0.05, 0.11)], atol=1e-12)
    for kwargs in ({"size": (0.1, 0.1)}, {"per_face": 0}):
        try:
            head_layout(**{**{"size": (0.1, 0.1, 0.1), "per_face": 1,
                              "offset": 0.01}, **kwargs})
            raise AssertionError("must reject bad args")
        except ValueError:
            pass


def test_make_arrivals_refraction_degenerates_to_homogeneous():
    """v1 == v2: the refraction forward model equals the straight-line
    model exactly (Snell point on the straight path); a mismatched
    unknown model raises."""
    sensors = head_layout((0.1, 0.1, 0.1), per_face=2, offset=0.012, seed=1)
    src = np.array([0.043, 0.061, 0.027])
    box = ((0.0, 0.0, 0.0), (0.1, 0.1, 0.1))
    refr, _ = make_arrivals(sensors, src, velocity=5300.0, t0=0.5,
                            model="refraction", head_velocity=5300.0,
                            box=box, seed=2)
    hom, _ = make_arrivals(sensors, src, velocity=5300.0, t0=0.5, seed=2)
    assert max(abs(refr[c] - hom[c]) for c in refr) < 1e-15
    try:
        make_arrivals(sensors, src, velocity=5300.0, model="refraction",
                      head_velocity=5605.0)                    # box missing
        raise AssertionError("missing box must raise")
    except ValueError:
        pass
    try:
        make_arrivals(sensors, src, velocity=5300.0, model="banana")
        raise AssertionError("unknown model must raise")
    except ValueError:
        pass


def test_refraction_scenario_round_trip():
    """End to end: head layout + refraction forward model + RefractionLocator
    recovers sources and the specimen velocity on clean arrivals."""
    sensors = head_layout((0.1, 0.1, 0.1), per_face=2, offset=0.012, seed=5)
    sources = np.array([[0.043, 0.061, 0.027], [0.07, 0.02, 0.08],
                        [0.02, 0.08, 0.05]])
    sc = Scenario(sensors=sensors, sources=sources, velocity=5300.0,
                  model="refraction", head_velocity=5605.0,
                  box=((0.0, 0.0, 0.0), (0.1, 0.1, 0.1)), seed=1)
    loc = RefractionLocator(sensors, box=((0, 0, 0), (0.1, 0.1, 0.1)),
                            velocity=5605.0, head_velocity=5605.0)
    df = run_trials(loc, sc)
    assert len(df) == 3
    assert df["err"].notna().all()
    assert (df["err"] < 1e-4).all()          # clean -> numerical precision
    assert df["converged"].all()
    assert (abs(df["velocity"] - 5300.0) < 1.0).all()   # v1 jointly inverted


def test_refraction_method_beats_geiger_on_refraction_data():
    """The paper's Fig. 11 story on synthetic bench data: with sensors on
    heads and a strong v1/v2 contrast, straight-line Geiger carries a
    several-mm systematic bias the refraction method does not (verified
    range: contrast sweep on clean-to-mildly-noisy data -- at 1 us noise
    with a strong contrast the joint v1 inversion is itself weakly
    conditioned, which the paper never claims to survive)."""
    sensors = head_layout((0.1, 0.1, 0.1), per_face=2, offset=0.012, seed=7)
    rng = np.random.default_rng(0)
    sources = rng.uniform(0.02, 0.08, size=(6, 3))
    v1, v2 = 5300.0, 7000.0                  # c1/c2 = 0.76: strong contrast
    sc = Scenario(sensors=sensors, sources=sources, velocity=v1, sigma=0.2e-6,
                  model="refraction", head_velocity=v2,
                  box=((0.0, 0.0, 0.0), (0.1, 0.1, 0.1)), seed=2, repeats=2)
    box_kw = dict(box=((0, 0, 0), (0.1, 0.1, 0.1)))
    df_r = run_trials(RefractionLocator(sensors, velocity=v2,
                                        head_velocity=v2, **box_kw), sc)
    df_g = run_trials(GeigerLocator(sensors, velocity=v2), sc)
    assert df_r["err"].notna().all()
    med_r, med_g = df_r["err"].median(), df_g["err"].median()
    assert med_r < 0.006, f"refraction median {med_r*1e3:.1f} mm"
    assert med_g > 0.008, f"geiger median {med_g*1e3:.1f} mm (expected bias)"
    assert med_r < med_g / 2                 # clearly better, not a coin flip
    # v1 comes back near truth despite starting at v2
    assert abs(df_r["velocity"].median() - v1) / v1 < 0.05


def test_refraction_sweep_over_v1_init(tmp_path):
    """Sweep machinery works for the refraction locator too: sweeping the
    v1 initial guess (velocity) yields a table with the parameter column."""
    sensors = head_layout((0.1, 0.1, 0.1), per_face=2, offset=0.012, seed=7)
    sc = Scenario(sensors=sensors, sources=np.array([[0.05, 0.05, 0.05]]),
                  velocity=5300.0, model="refraction", head_velocity=5605.0,
                  box=((0.0, 0.0, 0.0), (0.1, 0.1, 0.1)), seed=1)
    df = sweep(RefractionLocator, {"velocity": [4500.0, 5605.0]}, sc,
               cache_dir=str(tmp_path),
               locator_kwargs=dict(head_velocity=5605.0,
                                   box=((0, 0, 0), (0.1, 0.1, 0.1))))
    assert set(df["velocity"]) == {4500.0, 5605.0} and len(df) == 2
    assert (df["err"] < 1e-4).all()          # both starts reach the solution


def test_dashboard_refraction_method_needs_two_media():
    """AppTest 回归（两介质推广后语义）：(a) 只选折射定位器、布置仍是
    角点（正演保持均质）——角点传感器现在是合法的介质 1 直线射线传感器，
    面板应当直接跑通而不是报错；\"压头（折射）/组合体（两介质）\"布置
    选项此时就出现（不必先切正演模型）。(b) 自定义输入\"同时在两个轴向
    上出界\"的传感器 -> 侧栏中文警告 + 结果区中文错误并停止，aelib 的
    英文 ValueError 不外露。"""
    import pytest
    pytest.importorskip("streamlit")
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    # AppTest 的相对路径按调用文件解析，须给绝对路径
    app = Path(__file__).resolve().parents[1] / "locbench" / "app.py"
    at = AppTest.from_file(str(app), default_timeout=300)
    at.run()
    assert not at.exception
    # 缩小默认场景（网格 250 源 x 3 重复 -> 随机 1 源；扫描取值压到 1 个），
    # 只测 UI 分支，不测数值。震源数滑条只在随机模式下渲染，分两步设
    next(r for r in at.sidebar.radio if "震源布置" in r.label).set_value("随机震源")
    at.run()
    next(s for s in at.sidebar.slider if s.label == "震源数").set_value(1)
    next(t for t in at.text_input
         if t.label.startswith("取值")).set_value("5300")
    at.run()
    assert not at.exception

    # 只切方法 -> 角点布置（介质 1 表面）合法：直接可运行
    next(r for r in at.sidebar.radio
         if "定位算法" in r.label).set_value("折射路径 (locRP)")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error
    place_opts = next(r for r in at.sidebar.radio
                      if "传感器布置" in r.label).options
    assert "压头（折射）" in place_opts and "组合体（两介质）" in place_opts

    # 按指引切到压头布置（正演仍为均质 = 折射法吃错模型的数据）：可运行
    next(r for r in at.sidebar.radio
         if "传感器布置" in r.label).set_value("压头（折射）")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error

    # 自定义输入两轴出界的传感器 -> 中文警告 + 中文闸门，英文错误不外露
    next(r for r in at.sidebar.radio
         if "传感器布置" in r.label).set_value("自定义")
    at.run()                                   # 文本框变为可编辑
    bad = "\n".join(["-20,29,17", "-20,-20,50", "120,40,60",
                     "60,-25,80", "50,70,-20", "30,60,110"])
    next(t for t in at.sidebar.text_area
         if "传感器坐标" in t.label).set_value(bad)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("介质" in w.value for w in at.sidebar.warning)
    errs = [e.value for e in at.error]
    assert any("介质" in e for e in errs), errs
    assert not any("axes" in e or "sensor (" in e for e in errs)


def test_composite_layout_geometry():
    """composite_layout: per_face sensors on each block's 5 OUTER faces
    (shared interface carries none) -> 10*per_face total; block-A sensors
    lie on A's boundary (medium 1), block-B sensors beyond the interface
    on exactly one axis (medium 2); deterministic; rejects bad args."""
    lay = composite_layout((0.1, 0.1, 0.2), interface_frac=0.4, per_face=1,
                           seed=4)
    assert len(lay) == 10
    lo, hi = np.zeros(3), np.array([0.1, 0.1, 0.2])
    x0 = 0.4 * 0.1
    n_in = n_out = 0
    for p in lay.values():
        on_face = (np.isclose(p, lo).any() or np.isclose(p, hi).any()
                   or np.isclose(p[0], x0))
        # 出界按介质 1（块 A）边界判断：恰在一个轴向上出界才属介质 2
        out = [a for a in range(3)
               if p[a] < 0.0 or p[a] > (x0 if a == 0 else hi[a])]
        assert on_face, f"sensor {p} not on any outer face"
        if p[0] <= x0:                       # block A: on/in A's bounds
            assert not out
            n_in += 1
        else:                                # block B: beyond x0 on ONE axis
            assert out == [0], f"block-B sensor {p} outside on {out}"
            n_out += 1
    assert n_in == 5 and n_out == 5
    assert not any(np.isclose(p[0], x0) for p in lay.values())  # 界面无传感器
    assert composite_layout((0.1, 0.1, 0.2), 0.4, 1,
                            seed=4) == lay                     # deterministic
    for kwargs in ({"size": (0.1, 0.1)}, {"interface_frac": 0.0},
                   {"interface_frac": 1.0}, {"per_face": 0}):
        try:
            composite_layout(**{**{"size": (0.1, 0.1, 0.1),
                                   "interface_frac": 0.5, "per_face": 1,
                                   "seed": 1}, **kwargs})
            raise AssertionError("must reject bad args")
        except ValueError:
            pass


def test_composite_scenario_round_trip_both_blocks():
    """Composite scenario end to end: sensors on both blocks (medium 1
    surface + medium 2), sources in BOTH blocks, clean arrivals ->
    RefractionLocator recovers position + v1 and auto-detects the
    source's medium (aelib two-media generalization)."""
    sensors = composite_layout((0.1, 0.1, 0.1), interface_frac=0.5,
                               per_face=2, seed=6)
    src_a = np.array([0.03, 0.04, 0.06])       # block A (medium 1)
    src_b = np.array([0.07, 0.06, 0.04])       # block B (medium 2)
    box_a = ((0.0, 0.0, 0.0), (0.05, 0.1, 0.1))
    v1, v2 = 5300.0, 6800.0
    sc = Scenario(sensors=sensors, sources=np.stack([src_a, src_b]),
                  velocity=v1, model="refraction", head_velocity=v2,
                  box=box_a, seed=1)
    loc = RefractionLocator(sensors, box=box_a, velocity=v2,
                            head_velocity=v2)
    df = run_trials(loc, sc)
    assert len(df) == 2 and df["err"].notna().all()
    assert (df["err"] < 1e-4).all()            # clean -> numerical precision
    assert df["converged"].all()
    assert (abs(df["velocity"] - v1) < 1.0).all()
    # per-source medium detection through a direct locate call
    from locbench.synth import make_arrivals
    for src, side in ((src_a, 1), (src_b, 2)):
        arr, _ = make_arrivals(sensors, src, velocity=v1, model="refraction",
                               head_velocity=v2, box=box_a, seed=5)
        res = loc.locate(arr, true_coords=src)
        assert res.extra["source_medium"] == side
        assert res.extra["true_error"] < 1e-4


def test_dashboard_composite_flow():
    """AppTest：组合体（两介质）正演 + 组合体布置 + 折射定位器的完整
    流程可运行（无异常/无错误），切回 Geiger 在同一场景上也不报错。"""
    import pytest
    pytest.importorskip("streamlit")
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    app = Path(__file__).resolve().parents[1] / "locbench" / "app.py"
    at = AppTest.from_file(str(app), default_timeout=300)
    at.run()
    assert not at.exception
    next(r for r in at.sidebar.radio if "震源布置" in r.label).set_value("随机震源")
    at.run()
    next(s for s in at.sidebar.slider if s.label == "震源数").set_value(3)
    next(t for t in at.text_input
         if t.label.startswith("取值")).set_value("5300")
    next(r for r in at.sidebar.radio
         if "到时正演" in r.label).set_value("组合体（两介质）")
    at.run()
    place = next(r for r in at.sidebar.radio if "传感器布置" in r.label)
    assert "组合体（两介质）" in place.options
    place.set_value("组合体（两介质）")
    at.run()
    next(r for r in at.sidebar.radio
         if "定位算法" in r.label).set_value("折射路径 (locRP)")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert any(m.value == "100%" for m in at.metric)      # 收敛率卡片
    next(r for r in at.sidebar.radio
         if "定位算法" in r.label).set_value("Geiger")
    at.run()
    assert not at.exception and not at.error
