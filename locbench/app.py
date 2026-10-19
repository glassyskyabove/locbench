"""locbench dashboard -- locator testing / parameter tuning.

ALL data shown is simulated: synthetic array + synthetic sources + modelled
pick noise, outliers and channel dropout.  Run with

    streamlit run locbench/app.py        # or the `locbench` console script
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

from aelib import GeigerLocator
from locbench.evaluate import Scenario, run_trials, summarize
from locbench.plots import (DIST_ROWS, METRIC_LABELS, PARAM_LABELS,
                            SUMMARY_LABELS, display_table, plot_err_hist,
                            plot_layout, plot_metric_vs_param, unit_labels)
from locbench.sweep import sweep
from locbench.synth import (corners_layout, grid_sources, parse_layout,
                            random_layout, random_sources)

st.set_page_config(page_title="locbench", page_icon="\U0001F3AF",
                   layout="wide")
st.title("locbench — 定位算法测试 / 调参")
st.caption("所有数据均为数值模拟：合成台阵 + 合成震源 + 模拟拾取噪声 / 粗差 / 丢道。"
           "不含任何实测数据。")

# ------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("显示")
    disp_unit = st.radio("距离单位", ["mm", "m"], horizontal=True)
    factor = 1e3 if disp_unit == "mm" else 1.0
    st.header("场景（合成）")
    dim_label = st.radio("维度", ["三维", "二维"])
    is_3d = dim_label == "三维"
    lx_mm = st.number_input("试样尺寸 x (mm)", 10.0, 2000.0, 100.0, 10.0)
    ly_mm = st.number_input("试样尺寸 y (mm)", 10.0, 2000.0, 100.0, 10.0)
    lz_mm = (st.number_input("试样尺寸 z (mm)", 10.0, 2000.0, 200.0, 10.0)
             if is_3d else 0.0)
    lx, ly, lz = lx_mm * 1e-3, ly_mm * 1e-3, lz_mm * 1e-3     # -> m
    place_label = st.radio("传感器布置", ["顶点", "随机", "自定义"])
    kind = f"{'3d' if is_3d else '2d'}-" + {"顶点": "corners",
                                            "随机": "random",
                                            "自定义": "custom"}[place_label]
    custom_key = None
    if place_label == "顶点":
        n_sensors = 8 if is_3d else 4
    elif place_label == "随机":
        n_sensors = st.slider("传感器数", 4, 32, 8)
    else:
        default_txt = ("0,0,0\n100,0,0\n0,100,0\n100,100,0\n50,50,100"
                       if is_3d else "0,0\n100,0\n0,100\n100,100")
        custom_txt = st.text_area(
            "传感器坐标（每行一个，逗号分隔）", default_txt,
            key=f"custom_layout_{is_3d}", height=140,
            help=f"每行一个传感器：{'x,y,z' if is_3d else 'x,y'}，逗号分隔；"
                 f"单位与显示单位一致（当前 {disp_unit}）；支持 # 注释行")
        try:
            custom = parse_layout(custom_txt,
                                  n_components=3 if is_3d else 2,
                                  to_m=1.0 / factor)
        except ValueError as exc:
            st.error(f"传感器坐标有误：{exc}")
            st.stop()
        n_sensors = len(custom)
        custom_key = tuple(sorted(custom.items()))
    source_mode = st.radio("震源布置", ["随机震源", "网格震源（xyz 间距）"])
    if source_mode.startswith("随机"):
        n_sources = st.slider("震源数", 1, 50, 10)
        spacing_mm = 0.0
    else:
        spacing_mm = st.number_input("网格间距 (mm)", 5.0, 500.0, 50.0, 5.0)
        n_sources = 0
        counts = [max(1, int(round(s / (spacing_mm * 1e-3))))
                  for s in (lx, ly, lz)]
        st.caption(f"格数 ≈ {'×'.join(map(str, counts))} = "
                   f"{counts[0] * counts[1] * counts[2]} 个震源"
                   f"（格心布源，不含边界）")
    repeats = st.slider("震源重复次数", 1, 20, 3)
    seed = st.number_input("随机种子", 0, 1_000_000, 0)
    st.subheader("到时误差模拟")
    sigma_us = st.slider("拾取噪声 σ (µs)", 0.0, 50.0, 1.0, 0.5)
    outlier_p = st.slider("粗差比例", 0.0, 0.3, 0.0, 0.01)
    outlier_us = st.slider("粗差幅度 (µs)", 1.0, 500.0, 50.0)
    drop_p = st.slider("丢道比例", 0.0, 0.5, 0.0, 0.01)
    v_true = st.number_input("真实波速 (m/s)", 100.0, 10000.0, 5300.0, 100.0)

    st.header("定位器（Geiger）")
    v_used = st.number_input("采用波速 (m/s)", 100.0, 10000.0, 5300.0, 100.0,
                             help="与真实波速不一致时检验速度误差敏感性")
    grid_size = st.slider("grid_size", 4, 64, 24)
    margin = st.slider("margin", 0.0, 0.5, 0.05, 0.01)
    damping = st.checkbox("LM 阻尼", True)
    est_v = st.checkbox("联合反演速度", False)

SCENARIO_KEY = ("scenario", kind, n_sensors, lx, ly, lz, custom_key,
                source_mode, n_sources, spacing_mm * 1e-3, repeats,
                seed, sigma_us * 1e-6, outlier_p, outlier_us * 1e-6, drop_p,
                v_true)
PARAMS = dict(velocity=v_used, grid_size=grid_size, margin=margin,
              damping=damping, estimate_velocity=est_v)


def locator_kwargs(scenario_key, params, exclude=None):
    """Locator constructor kwargs: user params plus mode='2d' for 2-D
    layouts (pure 2-D sensor coordinates need no fixed axis)."""
    kw = dict(params)
    if exclude:
        kw.pop(exclude, None)          # a swept param must not repeat
    if scenario_key[1].startswith("2d"):
        kw.update(mode="2d")
    return kw


@st.cache_data(show_spinner="生成合成场景…")
def make_scenario(scenario_key):
    (_, kind, n_sensors, lx, ly, lz, custom_key, source_mode, n_sources,
     spacing, repeats, seed, sigma, outlier_p, outlier_jitter, drop_p,
     v_true) = scenario_key
    dim3, place = kind.split("-", 1)
    if place == "corners":
        sensors = (corners_layout(size=(lx, ly, lz)) if dim3 == "3d"
                   else corners_layout(size=(lx, ly)))
    elif place == "random":
        sensors = (random_layout(n_sensors, extent=(lx, ly, lz),
                                 min_spacing=0.05 * min(lx, ly, lz),
                                 seed=seed) if dim3 == "3d" else
                   random_layout(n_sensors, extent=(lx, ly),
                                 min_spacing=0.05 * min(lx, ly),
                                 seed=seed))
    else:
        sensors = dict(custom_key)
    if source_mode.startswith("网格"):
        sources = grid_sources(sensors, spacing=spacing)
    else:
        sources = random_sources(n_sources, sensors, seed=seed + 1)
    return Scenario(sensors=sensors, sources=sources, velocity=v_true,
                    sigma=sigma, outlier_p=outlier_p,
                    outlier_jitter=outlier_jitter, drop_p=drop_p,
                    seed=seed + 2, repeats=repeats)


@st.cache_data(show_spinner="定位试验运行中…")
def run_trials_cached(scenario_key, params):
    sc = make_scenario(scenario_key)
    kw = locator_kwargs(scenario_key, params)
    df = run_trials(GeigerLocator(sc.sensors, **kw), sc)
    return df


# ------------------------------------------------------- tab 1: results
tab_result, tab_sweep = st.tabs(["场景与结果", "参数敏感性"])

with tab_result:
    df = run_trials_cached(SCENARIO_KEY, PARAMS)
    s = summarize(df)
    c = st.columns(6)
    c[0].metric(f"定位误差中位 ({disp_unit})", f"{s.err_median * factor:.2f}")
    c[1].metric(f"误差 p95 ({disp_unit})", f"{s.err_p95 * factor:.2f}")
    c[2].metric("收敛率", f"{s.converged_rate * 100:.0f}%")
    c[3].metric("残差 rms 中位 (µs)", f"{s.rms_median * 1e6:.2f}")
    c[4].metric("平均迭代", f"{s.iter_mean:.1f}")
    c[5].metric("平均耗时 (ms)", f"{s.runtime_ms_mean:.2f}")
    if s.failures:
        st.warning(f"{s.failures} 个试验定位失败（通道过少等），详见表格 failure 列。")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("台阵与定位（红=定位, 绿=真值, 线段=误差向量）")
        sc = make_scenario(SCENARIO_KEY)
        located = np.full_like(np.asarray(sc.sources, dtype=float), np.nan)
        for r in df[df["err"].notna()].itertuples():
            located[int(r.source)] = r.coords
        st.pyplot(plot_layout(sc.sensors, sc.sources, located,
                              unit=disp_unit, factor=factor))
    with right:
        st.subheader("误差分布")
        st.pyplot(plot_err_hist(df, unit=disp_unit, factor=factor))
        st.subheader("汇总")
        s_show = s.copy()
        for k in DIST_ROWS:
            if pd.notna(s_show.get(k)):
                s_show[k] = s_show[k] * factor
        st.dataframe(s_show.rename(
            unit_labels(SUMMARY_LABELS, DIST_ROWS, disp_unit)).to_frame("值"))
    with st.expander("逐试验表格"):
        st.dataframe(display_table(df, disp_unit, factor))

# ---------------------------------------------------------- tab 2: sweep
with tab_sweep:
    st.subheader("单参数扫描（其余参数取侧栏当前值）")
    target = st.radio("扫描对象", ["定位器参数", "场景参数（噪声/鲁棒性）"])
    metric = st.selectbox("评估指标", list(METRIC_LABELS),
                          format_func=METRIC_LABELS.get)
    if target.startswith("定位器"):
        param = st.selectbox("参数", ["grid_size", "margin", "velocity"],
                             format_func=PARAM_LABELS.get)
        raw = st.text_input("取值（逗号分隔）",
                            {"grid_size": "4, 8, 16, 24, 48",
                             "margin": "0.0, 0.05, 0.1, 0.2",
                             "velocity": "4000, 4650, 5300, 5950, 6600"}[param])
        vals = [float(v) for v in raw.replace(";", ",").split(",") if v.strip()]
        if param == "grid_size":
            vals = [int(v) for v in vals]

        @st.cache_data(show_spinner="扫描运行中…（磁盘缓存）")
        def do_sweep(scenario_key, params, param, vals):
            sc = make_scenario(scenario_key)
            return sweep(GeigerLocator, {param: vals}, sc,
                         locator_kwargs=locator_kwargs(scenario_key, params,
                                                       exclude=param))

        df2 = do_sweep(SCENARIO_KEY, PARAMS, param, vals)
    else:
        param = st.selectbox("场景参数",
                             ["sigma", "outlier_p", "drop_p", "outlier_jitter"],
                             format_func=PARAM_LABELS.get)
        raw = st.text_input("取值（逗号分隔）",
                            {"sigma": "0, 1e-6, 2e-6, 5e-6, 1e-5",
                             "outlier_p": "0, 0.05, 0.1, 0.2",
                             "drop_p": "0, 0.1, 0.2, 0.3",
                             "outlier_jitter": "1e-5, 5e-5, 1e-4, 5e-4"}[param])
        vals = [float(v) for v in raw.replace(";", ",").split(",") if v.strip()]

        @st.cache_data(show_spinner="扫描运行中…")
        def do_scenario_sweep(scenario_key, params, param, vals):
            sc = make_scenario(scenario_key)
            loc = GeigerLocator(sc.sensors,
                                **locator_kwargs(scenario_key, params))
            frames = []
            for v in vals:
                d = run_trials(loc, replace(sc, **{param: v}))
                d[param] = v
                frames.append(d)
            return pd.concat(frames, ignore_index=True)

        df2 = do_scenario_sweep(SCENARIO_KEY, PARAMS, param, vals)

    agg_map = {"err_p95": ("err", lambda x: x.quantile(0.95)),
               "stderr_calibration": ("err", "median")}
    src_col, agg = agg_map.get(metric, (metric, "median"))
    gcol = "err" if metric == "stderr_calibration" else src_col
    if metric == "stderr_calibration":
        df2 = df2.assign(_cal=df2["err"] / df2["stderr_pos"])
        gcol = "_cal"
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_metric_vs_param(df2, param, metric=gcol, agg=agg,
                                       dist_unit=disp_unit,
                                       dist_factor=factor))
    with c2:
        st.pyplot(plot_metric_vs_param(df2, param, metric="runtime_ms"))
    with st.expander("扫描结果表"):
        st.dataframe(display_table(df2, disp_unit, factor))


def main():
    """Console entry point: ``locbench`` launches this dashboard.

    Deliberately NOT wired to an ``if __name__ == "__main__"`` guard:
    streamlit executes app scripts with ``__name__`` set to
    ``"__main__"``, so such a guard would spawn a nested server on every
    dashboard run."""
    import subprocess
    app = os.path.abspath(__file__)
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", app,
                              *sys.argv[1:]]))
