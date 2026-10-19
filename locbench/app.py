"""locbench dashboard -- locator testing / parameter tuning.

ALL data shown is simulated: synthetic array + synthetic sources + modelled
pick noise and channel dropout.  Run with

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
                            SUMMARY_LABELS, TIME_ROWS, display_table,
                            plot_err_hist, plot_layout,
                            plot_metric_vs_param, unit_labels)
from locbench.sweep import sweep
from locbench.synth import (feature_layout, grid_sources, parse_layout,
                            random_int_layout, random_sources)

st.set_page_config(page_title="locbench", page_icon="\U0001F3AF",
                   layout="wide")

# 滚轮穿透：悬停数字输入框时滚轮强制滚动所在容器，而不是被输入框吞掉
# （或误改数值）。经同源 iframe 注入到父页面，失败则静默跳过。
st.iframe("""
<script>
try {
  const doc = window.parent.document;
  doc.addEventListener("wheel", (e) => {
    const t = e.target;
    if (t && t.tagName === "INPUT" && t.type === "number") {
      if (doc.activeElement === t) t.blur();   // 已聚焦：先失焦防误改值
      e.preventDefault();
      let el = t.parentElement, sc = null;     // 找最近的滚动容器
      while (el && el !== doc.body) {
        if (/(auto|scroll)/.test(getComputedStyle(el).overflowY)) {
          sc = el; break;
        }
        el = el.parentElement;
      }
      if (sc) sc.scrollTop += e.deltaY;
    }
  }, { passive: false, capture: true });
} catch (err) { /* 沙箱限制时保持原生行为 */ }
</script>
""", height=1)

st.title("locbench — 定位算法测试 / 调参")
st.caption("所有数据均为数值模拟：合成台阵 + 合成震源 + 模拟拾取噪声 / 丢道。"
           "不含任何实测数据。")

# ------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("显示")
    disp_unit = st.radio("距离单位", ["mm", "m"], horizontal=True)
    factor = 1e3 if disp_unit == "mm" else 1.0
    time_unit = st.radio("时间单位", ["ms", "s", "µs", "ns"], horizontal=True)
    tfactor = {"s": 1.0, "ms": 1e3, "µs": 1e6, "ns": 1e9}[time_unit]
    st.header("场景（合成）")
    dim_label = st.radio("维度", ["三维", "二维"])
    is_3d = dim_label == "三维"
    sz_unit = st.selectbox("尺寸单位", ["mm", "m"])          # 尺寸与间距共用
    u_sz = 1e-3 if sz_unit == "mm" else 1.0

    def _auto_default(key, default):
        # 默认值跟随（单位/尺寸）联动，用户手动改过后不再跟随
        if key not in st.session_state or st.session_state.get(key + "_auto"):
            st.session_state[key] = default
            st.session_state[key + "_auto"] = True

    def _unset_auto(key):
        st.session_state[key + "_auto"] = False

    s_min, s_max, s_step = ((10.0, 2000.0, 10.0) if sz_unit == "mm"
                            else (0.01, 2.0, 0.01))
    _auto_default("size_x", 0.1 / u_sz)                    # 基准 100 mm
    lx_in = st.number_input("试样尺寸 x", s_min, s_max, key="size_x",
                            step=s_step, on_change=lambda: _unset_auto("size_x"))
    _auto_default("size_y", 0.1 / u_sz)
    ly_in = st.number_input("试样尺寸 y", s_min, s_max, key="size_y",
                            step=s_step, on_change=lambda: _unset_auto("size_y"))
    _auto_default("size_z", 0.2 / u_sz)                    # 基准 200 mm
    lz_in = (st.number_input("试样尺寸 z", s_min, s_max, key="size_z",
                             step=s_step,
                             on_change=lambda: _unset_auto("size_z"))
             if is_3d else 0.0)
    lx, ly, lz = lx_in * u_sz, ly_in * u_sz, lz_in * u_sz     # -> m
    place_label = st.radio("传感器布置", ["几何特征", "随机", "自定义"])
    kind = f"{'3d' if is_3d else '2d'}-" + {"几何特征": "feature",
                                            "随机": "random",
                                            "自定义": "custom"}[place_label]
    n_wanted = st.slider("传感器数", 4, 32, 8) if place_label == "随机" else 0

    def _fmt_layout(m_pts):
        """米制坐标 -> 文本（按显示单位换算，整数值不带小数点）。"""
        lines = []
        for p in m_pts:
            cells = []
            for v in p:
                d = v * factor
                cells.append(str(int(round(d))) if abs(d - round(d)) < 1e-9
                             else f"{d:.6g}")
            lines.append(",".join(cells))
        return "\n".join(lines)

    if place_label == "几何特征":
        feats = (["横边", "纵边", "垂边", "侧面", "正面", "顶面"] if is_3d
                 else ["横边", "纵边"])
        default_on = {"横边", "纵边", "垂边"} if is_3d else {"横边", "纵边"}
        fcols = st.columns(len(feats))
        on = {f: fcols[i].checkbox(f, value=f in default_on)
              for i, f in enumerate(feats)}
        edge_n = {}
        for f in ("横边", "纵边", "垂边"):
            if on.get(f):
                edge_n[f] = st.number_input(
                    f"{f}等分点数", 1, 32, 4,
                    help="将该边 n+1 等分，传感器放在 n 个等分点上"
                         "（不含两端端点，故不会与其他边的角点重复）")
        faces = [f for f in on if on[f] and f not in edge_n]
        st.session_state["layout_text"] = _fmt_layout(
            feature_layout((lx, ly, lz) if is_3d else (lx, ly),
                           edges=edge_n, faces=faces).values())
        st.caption("横边/纵边/垂边 = 过原点、与 x/y/z 平行的边；"
                   "侧面 = x=0 面、正面 = y=0 面（以法向命名）；"
                   "顶面 = z=Lz 面。选中的面各布 1 个传感器于面心")
    elif place_label == "随机":
        ext_mm = ((int(round(lx * 1e3)), int(round(ly * 1e3)),
                   int(round(lz * 1e3))) if is_3d else
                  (int(round(lx * 1e3)), int(round(ly * 1e3))))
        # 随机种子控件在侧栏更下方：先读 session_state（首跑用默认 0）
        seed_gen = st.session_state.get("seed", 0)
        rand = random_int_layout(n_wanted, extent=ext_mm, seed=seed_gen)
        st.session_state["layout_text"] = _fmt_layout(
            tuple(v * 1e-3 for v in p) for p in rand.values())
    elif "layout_text" not in st.session_state:
        ex = ([(0, 0, 0), (100, 0, 0), (0, 100, 0), (100, 100, 0), (50, 50, 100)]
              if is_3d else [(0, 0), (100, 0), (0, 100), (100, 100)])
        st.session_state["layout_text"] = _fmt_layout(
            tuple(v * 1e-3 for v in p) for p in ex)

    custom_txt = st.text_area(
        "传感器坐标（每行一个，逗号分隔）", key="layout_text", height=140,
        disabled=place_label != "自定义",
        help=f"每行一个传感器：{'x,y,z' if is_3d else 'x,y'}；单位与显示单位"
             f"一致（当前 {disp_unit}）；支持 # 注释行。几何特征/随机模式显示"
             f"生成的坐标（不可编辑），切到\"自定义\"可在此基础上修改")
    try:
        custom = parse_layout(custom_txt, n_components=3 if is_3d else 2,
                              to_m=1.0 / factor)
    except ValueError as exc:
        st.error(f"传感器坐标有误：{exc}")
        st.stop()
    n_sensors = len(custom)
    min_needed = 4 if is_3d else 3            # x,y[,z],t0 -> 最低通道数
    if n_sensors < min_needed:
        st.warning(f"当前仅 {n_sensors} 个传感器，{'三维' if is_3d else '二维'}"
                   f"定位至少需要 {min_needed} 个通道，请多选几项几何特征或"
                   f"增加等分点数")
    custom_key = tuple(sorted(custom.items()))
    source_mode = st.radio("震源布置", ["随机震源", "网格震源"], index=1)
    if source_mode == "随机震源":
        n_sources = st.slider("震源数", 1, 50, 10)
        grid_d_in = (0.0, 0.0, 0.0)
    else:
        d_min, d_max, d_step = ((0.5, 2000.0, 1.0) if sz_unit == "mm"
                                else (0.0005, 2.0, 0.001))

        def _spacing(label, size_in, key):
            # 默认 = 尺寸/5（当前单位）；联动规则同尺寸输入
            _auto_default(key, round(size_in / 5.0, 9))
            return st.number_input(label, d_min, d_max, key=key, step=d_step,
                                   on_change=lambda k=key: _unset_auto(k))

        dx_in = _spacing("网格间距 x", lx_in, "grid_dx")
        dy_in = _spacing("网格间距 y", ly_in, "grid_dy")
        dz_in = (_spacing("网格间距 z", lz_in, "grid_dz") if is_3d else 0.0)
        n_sources = 0
        spans = (lx, ly, lz) if is_3d else (lx, ly)
        d_ins = (dx_in, dy_in, dz_in) if is_3d else (dx_in, dy_in)
        counts = [max(1, int(round(s / (d * u_sz))))
                  for s, d in zip(spans, d_ins)]
        st.caption(f"格数 ≈ {'×'.join(map(str, counts))} = "
                   f"{int(np.prod(counts))} 个震源（格心布源，不含边界）")
        grid_d_in = (dx_in, dy_in, dz_in)
    repeats = st.slider("震源重复次数", 1, 20, 3)
    seed = st.number_input("随机种子", 0, 1_000_000, key="seed")
    st.subheader("到时误差模拟")
    fs_unit = st.selectbox("采样率单位", ["MHz", "kHz"])
    f_fs = 1e6 if fs_unit == "MHz" else 1e3
    fs_min, fs_max, fs_step = ((0.0, 200.0, 1.0) if fs_unit == "MHz"
                               else (0.0, 200_000.0, 100.0))
    _auto_default("fs_in", 10e6 / f_fs)                    # 基准 10 MHz
    fs_in = st.number_input("采样率", fs_min, fs_max, key="fs_in",
                            step=fs_step,
                            on_change=lambda: _unset_auto("fs_in"),
                            help="到时按采样周期 1/fs 量化（模拟仪器时间分"
                                 "辨率，如 10 MHz → 0.1 µs）；0 = 连续不量化")
    fs_hz = fs_in * f_fs
    err_unit = st.selectbox("误差单位", ["µs", "ms"])
    f_err = 1e-6 if err_unit == "µs" else 1e-3
    _auto_default("sigma_in", 1e-6 / f_err)                # 基准 1 µs
    sigma_in = st.slider("到时误差 σ", 0.0, 50e-6 / f_err, key="sigma_in",
                         step=0.5e-6 / f_err,
                         on_change=lambda: _unset_auto("sigma_in"))
    sigma_s = sigma_in * f_err
    drop_p = st.slider("丢道概率", 0.0, 0.5, 0.1, 0.01,
                       help="每个事件对每个通道独立抽一个 [0,1) 均匀随机数，"
                            "小于该值则此通道在此事件中被丢弃")
    v_true = st.number_input("真实波速 (m/s)", 100.0, 10000.0, 5300.0, 100.0)

    st.header("定位器（Geiger）")
    v_used = st.number_input("采用波速 (m/s)", 100.0, 10000.0, 5300.0, 100.0)
    grid_size = st.slider("初值点个数（grid size）", 4, 64, 4,
                          help="初值网格每条轴的格点【数】，不是间距：在搜"
                               "索盒内等分取点且含边界，实际间距 = 跨度/"
                               "(个数−1)，候选点总数 = 个数^维度（见下方注"
                               "释）。定位迭代前对每个候选点算加权残差代"
                               "价，取最优者作起点。个数越多初值越稳但初"
                               "值阶段越慢")
    st.caption(f"初值点 = {'×'.join([str(grid_size)] * (3 if is_3d else 2))}"
               f" = {grid_size ** (3 if is_3d else 2)} 个候选点"
               f"（{'三维' if is_3d else '二维'}网格搜索）")
    margin = st.slider("外延比例（margin）", 0.0, 0.5, 0.0, 0.01,
                       help="初值搜索盒 = 传感器包围盒各方向向外扩大的"
                            "比例（0.05 = 每边外扩 5% 跨度），默认 0 即"
                            "只在包围盒内搜索。震源通常在台阵包围盒内；"
                            "预判震源会越出台阵时才需加大。只影响初值搜"
                            "索范围，不限制最终解的位置")
    max_iter = st.number_input("迭代上限（max_iter）", 5, 500, 60, 5,
                               help="外层迭代次数上限，两种模式共用；"
                                    "跑满且末步 RMS 改善仍大于 tol 则标记"
                                    "未收敛")
    tol = st.number_input("收敛判据 tol (s)", 0.0, 1e-3, 1e-6, 1e-7,
                          format="%.1e",
                          help="相邻两次迭代加权残差 RMS 的改善量小于 tol "
                               "即判定收敛。默认 1e-6 s（1 µs，与到时误差"
                               "同量级）——更小的 tol 会沿弱辨识方向过度"
                               "迭代、拟合噪声；0 = 关闭该判据，仅靠停滞"
                               "判据或 max_iter 终止")
    damping = st.checkbox("LM 阻尼", True)
    est_v = st.checkbox("联合反演速度", False)

SCENARIO_KEY = ("scenario", kind, n_sensors, lx, ly, lz, custom_key,
                source_mode, n_sources,
                tuple(d * u_sz for d in grid_d_in), repeats,
                seed, sigma_s, drop_p, fs_hz, v_true)
PARAMS = dict(velocity=v_used, grid_size=grid_size, margin=margin,
              max_iter=int(max_iter), tol=tol,
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
     spacing, repeats, seed, sigma, drop_p, fs, v_true) = scenario_key
    sensors = dict(custom_key)          # the text box is the single source
    dim = 3 if kind.startswith("3d") else 2
    # 震源域 = 试样盒 [0, size]，与传感器包围盒解耦（部分布置如单边
    # 台阵的包围盒会退化成线/点，不能拿来布源）
    box = (((0.0, 0.0, 0.0), (lx, ly, lz)) if dim == 3
           else ((0.0, 0.0), (lx, ly)))
    if source_mode.startswith("网格"):
        sources = grid_sources(spacing=tuple(spacing[:dim]), box=box)
    else:
        sources = random_sources(n_sources, box=box, seed=seed + 1)
    return Scenario(sensors=sensors, sources=sources, velocity=v_true,
                    sigma=sigma, drop_p=drop_p, fs=fs,
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
    try:
        df = run_trials_cached(SCENARIO_KEY, PARAMS)
    except Exception as exc:            # e.g. degenerate (collinear) layout
        st.error(f"定位无法进行：{exc}")
        st.stop()
    s = summarize(df)
    c = st.columns(6)
    c[0].metric(f"定位误差中位 ({disp_unit})", f"{s.err_median * factor:.2f}")
    c[1].metric(f"误差 p95 ({disp_unit})", f"{s.err_p95 * factor:.2f}")
    c[2].metric("收敛率", f"{s.converged_rate * 100:.0f}%")
    c[3].metric(f"残差 rms 中位 ({time_unit})",
                f"{s.rms_median * tfactor:.3g}")
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
        for rows, f in ((DIST_ROWS, factor), (TIME_ROWS, tfactor)):
            for k in rows:
                if pd.notna(s_show.get(k)):
                    s_show[k] = s_show[k] * f
        labels = unit_labels(SUMMARY_LABELS, DIST_ROWS, disp_unit, "(m)")
        labels = unit_labels(labels, TIME_ROWS, time_unit, "(s)")
        st.dataframe(s_show.rename(labels).to_frame("值"))
    with st.expander("逐试验表格"):
        st.dataframe(display_table(df, disp_unit, factor,
                                   time_unit, tfactor))

# ---------------------------------------------------------- tab 2: sweep
with tab_sweep:
    st.subheader("单参数扫描（其余参数取侧栏当前值）")
    target = st.radio("扫描对象", ["定位器参数", "场景参数（噪声/鲁棒性）"])
    metric = st.selectbox("评估指标", list(METRIC_LABELS),
                          format_func=METRIC_LABELS.get)
    if target.startswith("定位器"):
        param = st.selectbox("参数",
                             ["grid_size", "margin", "velocity", "max_iter",
                              "tol"],
                             format_func=PARAM_LABELS.get)
        raw = st.text_input("取值（逗号分隔）",
                            {"grid_size": "4, 8, 16, 24, 48",
                             "margin": "0.0, 0.05, 0.1, 0.2",
                             "velocity": "4000, 4650, 5300, 5950, 6600",
                             "max_iter": "5, 10, 20, 60, 150",
                             "tol": "0, 1e-8, 1e-7, 1e-6, 1e-5"}[param])
        vals = [float(v) for v in raw.replace(";", ",").split(",") if v.strip()]
        if param in ("grid_size", "max_iter"):
            vals = [int(v) for v in vals]

        @st.cache_data(show_spinner="扫描运行中…（磁盘缓存）")
        def do_sweep(scenario_key, params, param, vals):
            sc = make_scenario(scenario_key)
            return sweep(GeigerLocator, {param: vals}, sc,
                         locator_kwargs=locator_kwargs(scenario_key, params,
                                                       exclude=param))
    else:
        param = st.selectbox("场景参数", ["sigma", "fs", "drop_p"],
                             format_func=PARAM_LABELS.get)
        raw = st.text_input("取值（逗号分隔）",
                            {"sigma": "0, 1e-6, 2e-6, 5e-6, 1e-5",
                             "fs": "0, 2e6, 5e6, 1e7, 4e7",
                             "drop_p": "0, 0.1, 0.2, 0.3"}[param])
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

    agg_map = {"err_p95": ("err", lambda x: x.quantile(0.95)),
               "stderr_calibration": ("err", "median")}
    try:
        df2 = (do_sweep(SCENARIO_KEY, PARAMS, param, vals)
               if target.startswith("定位器")
               else do_scenario_sweep(SCENARIO_KEY, PARAMS, param, vals))
    except Exception as exc:
        st.error(f"扫描无法进行：{exc}")
        st.stop()
    src_col, agg = agg_map.get(metric, (metric, "median"))
    gcol = "err" if metric == "stderr_calibration" else src_col
    if metric == "stderr_calibration":
        df2 = df2.assign(_cal=df2["err"] / df2["stderr_pos"])
        gcol = "_cal"
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_metric_vs_param(df2, param, metric=gcol, agg=agg,
                                       dist_unit=disp_unit,
                                       dist_factor=factor,
                                       time_unit=time_unit,
                                       time_factor=tfactor))
    with c2:
        st.pyplot(plot_metric_vs_param(df2, param, metric="runtime_ms"))
    with st.expander("扫描结果表"):
        st.dataframe(display_table(df2, disp_unit, factor,
                                   time_unit, tfactor))


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
