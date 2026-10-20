"""locbench dashboard -- locator testing / parameter tuning.

ALL data shown is simulated: synthetic array + synthetic sources + modelled
pick noise and channel dropout.  Run with

    streamlit run locbench/app.py        # or the `locbench` console script
"""
from __future__ import annotations

import os
import sys

import streamlit as st


def main():
    """Console entry point: the ``locbench`` command launches this
    dashboard via a streamlit subprocess.

    The dashboard body below must ONLY execute under ``streamlit run``
    (widgets and session_state do not function in bare mode).  This
    module is IMPORTED -- not run -- by the console script, and a plain
    import must not trigger the body either: it used to, and crashed
    with a confusing ``NameError: name 'custom' is not defined`` because
    bare-mode widgets return empty values and ``st.stop()`` is a no-op
    outside a running script runner."""
    import subprocess
    app = os.path.abspath(__file__)
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", app,
                              *sys.argv[1:]]))


if __name__ == "__main__":
    # streamlit 以 __name__=='__main__' 执行本文件；裸 python（无运行时）
    # 在此友好退出；被控制台命令导入时整段跳过，只暴露上面的 main()
    if not st.runtime.exists():
        raise SystemExit(
            "\nlocbench 是 streamlit 应用，请用以下任一方式启动：\n"
            "    locbench\n"
            "    streamlit run locbench/app.py\n")

    # ------------------------------------------- dashboard body
    from dataclasses import replace

    import numpy as np
    import pandas as pd

    from aelib import GeigerLocator, RefractionLocator
    from locbench.evaluate import Scenario, run_trials, summarize
    from locbench.plots import (DIST_ROWS, METRIC_LABELS, PARAM_LABELS,
                                SUMMARY_LABELS, TIME_ROWS, display_table,
                                plot_err_hist, plot_layout,
                                plot_metric_vs_param, unit_labels)
    from locbench.plots import close_fig
    from locbench.sweep import parse_values, sweep
    from locbench.synth import (composite_layout, feature_layout, grid_sources,
                                head_layout, parse_layout, random_int_layout,
                                random_sources)

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
    st.caption("所有数据均为数值模拟：合成传感器阵列 + 合成震源 + 模拟拾取噪声 / 丢道。"
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
        st.subheader("波速模型")
        model_label = st.radio(
            "到时正演", ["均质（直线传播）", "折射（试样 + 压头）",
                         "组合体（两介质）"],
            help="合成到时用的传播模型。均质：t = d/v，单一波速直线传播；"
                 "折射（Rui et al. 2024）：试样盒内波速 v₁，压头已知波速 "
                 "v₂，波在界面按 Snell 折射——传感器须装在压头上"
                 "（布置选\"压头\"）；组合体：试样沿 x 向分成两块（界面比例"
                 "可调），块 A 波速 v₁、块 B 已知波速 v₂，传感器贴两块表面"
                 "（布置选\"组合体\"），震源可在任一块内。非均质模型下 "
                 "Geiger 有随 v₁/v₂ 偏离 1 增大的系统偏差，折射定位器则"
                 "联合反演 v₁ 并自动判别震源所在介质")
        is_refr = not model_label.startswith("均质")
        is_comp = model_label.startswith("组合体")
        # 介质 1（块 A）边界，米：组合体 = 界面左侧块，其余 = 整试样盒
        refr_box = ((0.0, 0.0, 0.0), (lx, ly, lz))
        if is_comp:
            st.caption(f"块 A = [0, 界面]×[0, y]×[0, z]，块 B = [界面, x]"
                       f"×[0, y]×[0, z]，界面为 x = 比例×x 尺寸的平面")
            frac = st.slider("界面位置（x 向比例）", 0.1, 0.9, 0.5, 0.05)
            refr_box = ((0.0, 0.0, 0.0), (frac * lx, ly, lz))
        # 定位算法单选框在侧栏更下方才创建，但用户改动后 session_state
        # 在本次重跑一开始就已更新——提前读它，布置选项就能跟随方法联动
        is_rp_now = str(st.session_state.get("loc_method", "Geiger")
                        ).startswith("折射")
        place_opts = ["几何特征", "随机", "自定义"]
        if is_3d and not is_comp and (is_refr or is_rp_now):
            place_opts.append("压头（折射）")
        if is_3d and (is_comp or is_rp_now):
            place_opts.append("组合体（两介质）")
        place_label = st.radio("传感器布置", place_opts,
                               help=("压头（折射）＝试样盒六个面外各布若干"
                                     "传感器（折射正演的经典布置）；组合体"
                                     "（两介质）＝两块的外表面各布若干传感器"
                                     "（介质 1/2 表面贴装）"
                                     if (is_refr or is_rp_now) else None))
        kind = f"{'3d' if is_3d else '2d'}-" + {"几何特征": "feature",
                                                "随机": "random",
                                                "自定义": "custom",
                                                "压头（折射）": "head",
                                                "组合体（两介质）": "comp"}[place_label]
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
            n_edges = 4 if is_3d else 2               # 每个边特征覆盖的边数
            edge_names = ["横边", "纵边", "垂边"][: 3 if is_3d else 2]
            face_names = ["侧面", "正面", "顶面"] if is_3d else []
            ecols = st.columns(len(edge_names))          # 边一行
            on = {f: ecols[i].checkbox(f, value=False)
                  for i, f in enumerate(edge_names)}
            if face_names:
                fcols = st.columns(len(face_names))      # 面一行
                for i, f in enumerate(face_names):
                    on[f] = fcols[i].checkbox(f, value=False)
            on["顶点"] = st.checkbox(                     # 顶点另起一行
                "顶点", value=True,
                help=f"全部{'8' if is_3d else '4'}个角点各布 1 个传感器"
                     f"（经典试样监测布置）")
            edge_n = {}
            for f in ("横边", "纵边", "垂边"):
                if on.get(f):
                    edge_n[f] = st.number_input(
                        f"{f}等分点数（每条边）", 1, 32, 4,
                        help=f"与{f[0]}轴平行的边共 {n_edges} 条，每条边"
                             " n+1 等分取 n 个内部等分点（不含端点角点）")
            faces = [f for f in on if on[f] and f not in edge_n
                     and f != "顶点"]
            st.session_state["layout_text"] = _fmt_layout(
                feature_layout((lx, ly, lz) if is_3d else (lx, ly),
                               edges=edge_n, faces=faces,
                               vertices=on.get("顶点", False)).values())
            st.caption(f"横边/纵边/垂边 = 与 x/y/z 平行的全部"
                       f"{'4' if is_3d else '2'}条边；侧面/正面/顶面 = "
                       f"以 x/y/z 为法向的两个面（各取面心）；顶点 = "
                       f"全部{'8' if is_3d else '4'}个角点")
        elif place_label == "随机":
            ext_mm = ((int(round(lx * 1e3)), int(round(ly * 1e3)),
                       int(round(lz * 1e3))) if is_3d else
                      (int(round(lx * 1e3)), int(round(ly * 1e3))))
            # 随机种子控件在侧栏更下方：先读 session_state（首跑用默认 0）
            seed_gen = st.session_state.get("seed", 0)
            rand = random_int_layout(n_wanted, extent=ext_mm, seed=seed_gen)
            st.session_state["layout_text"] = _fmt_layout(
                tuple(v * 1e-3 for v in p) for p in rand.values())
        elif place_label == "压头（折射）":
            n_per_face = st.number_input("每个压头（面）传感器数", 1, 8, 2,
                                         help="试样盒六个面外各布 n 个传感器"
                                              "（面内随机位置），共 6n 个")
            o_min, o_max, o_step = ((0.5, 100.0, 0.5) if sz_unit == "mm"
                                    else (0.0005, 0.1, 0.0005))
            _auto_default("head_off_in", round(0.2 * min(lx_in, ly_in,
                                                         lz_in), 9))
            head_off_in = st.number_input(
                "压头外距", o_min, o_max, key="head_off_in", step=o_step,
                on_change=lambda: _unset_auto("head_off_in"),
                help=f"传感器离试样面的垂直距离（当前单位 {sz_unit}）；"
                     "默认 0.2×最短边。偏距越大（论文 5–50 mm 不等），压头"
                     "段走时差异越大，v₁ 反演越可分辨；全部传感器小偏距"
                     "且速度对比度大时，v₁-位置权衡会病态（折射法误差与"
                     "Geiger 相当甚至更差）")
            # 随机种子控件在侧栏更下方：先读 session_state（首跑用默认 0）
            seed_gen = st.session_state.get("seed", 0)
            st.session_state["layout_text"] = _fmt_layout(
                head_layout((lx, ly, lz), per_face=int(n_per_face),
                            offset=head_off_in * u_sz, seed=seed_gen).values())
            st.caption(f"压头布置 = 六个面（±x/±y/±z）各 {int(n_per_face)} 个"
                       f"传感器，共 {6 * int(n_per_face)} 个，全部在试样盒外")
        elif place_label == "组合体（两介质）":
            n_per_face_c = st.number_input(
                "每块每个面传感器数", 1, 8, 1,
                help="两块各自 5 个外表面（不含界面）上各布 n 个传感器"
                     "（面内随机位置），共 10n 个")
            seed_gen = st.session_state.get("seed", 0)
            st.session_state["layout_text"] = _fmt_layout(
                composite_layout((lx, ly, lz), interface_frac=frac,
                                 per_face=int(n_per_face_c),
                                 seed=seed_gen).values())
            st.caption(f"组合体布置 = 块 A（x < {frac * lx_in:.4g}{sz_unit}）"
                       f"与块 B 各 5 个外表面 × {int(n_per_face_c)} 个，共 "
                       f"{10 * int(n_per_face_c)} 个，贴两块表面（无压头）")
        elif "layout_text" not in st.session_state:
            ex = ([(0, 0, 0), (100, 0, 0), (0, 100, 0), (100, 100, 0), (50, 50, 100)]
                  if is_3d else [(0, 0), (100, 0), (0, 100), (100, 100)])
            st.session_state["layout_text"] = _fmt_layout(
                tuple(v * 1e-3 for v in p) for p in ex)

        custom_txt = st.text_area(
            "传感器坐标（每行一个，逗号分隔）", key="layout_text", height=140,
            disabled=place_label != "自定义",
            help=f"每行一个传感器：{'x,y,z' if is_3d else 'x,y'}；单位与显示单位"
                 f"一致（当前 {disp_unit}）；支持 # 注释行。几何特征/随机/压头/"
                 f"组合体模式显示生成的坐标（不可编辑），切到\"自定义\"可在此基础上"
                 f"修改")
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

        # 两介质几何预检：折射/组合体正演与折射定位器都要求每个传感器
        # 恰在 0 或 1 个轴向上处于介质 1 边界（refr_box）之外——即装在
        # 其中一个介质上（与 aelib classify_heads 同判据）。提前在这里
        # 判掉并给中文引导，避免跑到一半才冒出 aelib 的英文错误
        def _head_problem():
            if not is_3d:
                return "两介质定位为三维布置，当前维度为二维"
            lo, hi = (np.asarray(b, dtype=float) for b in refr_box)
            for p in np.asarray(list(custom.values()), dtype=float):
                out = [a for a in range(3) if p[a] < lo[a] or p[a] > hi[a]]
                if len(out) > 1:
                    disp = ", ".join(f"{v * factor:.6g}" for v in p)
                    return (f"首个不满足的传感器（{disp_unit}）：({disp}) 同时"
                            f"在 {len(out)} 个轴向上处于介质 1 边界外——每个"
                            "传感器须装在其中一个介质上（介质 1 表面或恰在"
                            "一个轴向上出界）")
            return None

        if (is_refr or is_rp_now):
            heads_problem = _head_problem()
            if heads_problem:
                st.warning(
                    f"折射{'正演' if is_refr else '定位器'}需要传感器装在两个"
                    f"介质上（每个恰在 0 或 1 个轴向上处于介质 1 边界外）。"
                    f"{heads_problem}。请把\"传感器布置\"切到\"压头（折射）\""
                    f"或\"组合体（两介质）\"，或用\"自定义\"修改坐标")
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
        sigma_in = st.slider("到时误差 σ", 0.0, 6e-6 / f_err, key="sigma_in",
                             step=0.1e-6 / f_err,
                             on_change=lambda: _unset_auto("sigma_in"))
        sigma_s = sigma_in * f_err
        drop_p = st.slider("丢道概率", 0.0, 0.5, 0.1, 0.01,
                           help="每个事件对每个通道独立抽一个 [0,1) 均匀随机数，"
                                "小于该值则此通道在此事件中被丢弃")
        v_true = st.number_input(
            "真实波速 (m/s)", 100.0, 10000.0, 5300.0, 100.0,
            help="合成到时用的波速。折射模型下为试样波速 v₁（压头内为 "
                 "v₂，见下）")
        if is_refr:
            v2_true = st.number_input(
                "第二介质波速 v₂ (m/s)", 100.0, 10000.0, 5605.0, 100.0,
                help="折射模型 = 压头/耦合块波速、组合体 = 第二块波速"
                     "（已知量，如铁 5605 m/s）；v₁/v₂ 偏离 1 越远，均质"
                     "直线模型（Geiger）的系统偏差越大")

        st.header("定位器")
        method_label = st.radio(
            "定位算法", ["Geiger", "折射路径 (locRP)"], key="loc_method",
            help="Geiger = 均质直线模型线性化加权最小二乘（grid 初值 + LM）；"
                 "折射路径 = aelib RefractionLocator（Rui et al. 2024, IJMST "
                 "34:521-530 的两介质推广）：按 (源, 传感器) 所在介质分派"
                 "直线/折射射线，联合反演震源、试样波速 v₁ 与震源所在介质"
                 "（第二介质波速 v₂ 已知），LM + 迭代鲁棒加权——对粗差"
                 "几乎免疫。需要传感器装在两个介质上（布置选\"压头（折射）\""
                 "或\"组合体（两介质）\"）；到时正演再选\"折射/组合体\"即"
                 "双层介质的完整场景，正演保持\"均质\"则相当于给折射法喂"
                 "错模型的数据")
        is_rp = method_label.startswith("折射")
        if is_rp:
            v_used = st.number_input(
                "试样波速初值 v₁ (m/s)", 100.0, 10000.0, 5605.0, 100.0,
                help="v₁ 的迭代初值（论文从压头波速起步，默认同 v₂）；"
                     "反演结果见逐试验表格的 velocity 列")
            v2_used = st.number_input(
                "第二介质波速 v₂ (m/s)", 100.0, 10000.0, 5605.0, 100.0,
                help="已知的第二介质波速（压头 / 组合体第二块，正演模型 "
                     "v₂）；定位与合成场景的 v₂ 独立给定——不一致即模拟"
                     "\"v₂ 标定错\"的情形")
            grid_size = st.slider(
                "初值点个数（grid size）", 1, 64, 4,
                help="初值网格每条轴的格点【数】，在试样盒内等分取点且含"
                     "边界；1 = 只用试样中心（论文的单一初值）。折射法"
                     "的初值网格覆盖试样盒而非传感器包围盒（传感器全在盒外）")
            st.caption(f"初值点 = {'×'.join([str(grid_size)] * 3)}"
                       f" = {grid_size ** 3} 个候选点（试样盒内三维网格搜索）")
            margin = st.slider(
                "外延比例（margin）", 0.0, 0.5, 0.0, 0.01,
                help="初值搜索盒 = 试样盒各方向向外扩大的比例；预判震源在"
                     "试样面与压头内面之间时才需加大")
            max_iter = st.number_input("迭代上限（max_iter）", 5, 500, 60, 5,
                                       help="LM 迭代次数上限（论文为 20；"
                                            "默认 60 让鲁棒权重的重估充分"
                                            "收敛）")
            tol = st.number_input(
                "收敛判据 tol (m)", 0.0, 1e-3, 1e-5, step=1e-7,
                format="%.1e",
                help="【位置步长】判据（与 Geiger 的时间 RMS 判据不同）："
                     "一次被接受的迭代把震源移动 Σ|Δx|+|Δy|+|Δz| < tol "
                     "即收敛。默认 1e-5 m = 0.01 mm——刻意偏松：鲁棒权重"
                     "逐次重估时追噪声级以下的改进会让迭代沿位置-速度弱"
                     "约束方向漂移；0 = 关闭该判据")
        else:
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
                                    "只在包围盒内搜索。震源通常在传感器阵列包围盒内；"
                                    "预判震源会越出传感器阵列时才需加大。只影响初值搜"
                                    "索范围，不限制最终解的位置")
            max_iter = st.number_input("迭代上限（max_iter）", 5, 500, 60, 5,
                                       help="外层迭代次数上限，两种模式共用；"
                                            "跑满且末步 RMS 改善仍大于 tol 则标记"
                                            "未收敛")
            # 默认 = 1/fs（一个采样周期，与 aelib v0.9.39 的默认语义一致）；
            # 未手动改过 tol 时随采样率联动，改过即脱钩；fs=0 回退 1e-6
            _auto_default("tol_in", 1.0 / fs_hz if fs_hz > 0 else 1e-6)
            tol = st.number_input("收敛判据 tol (s)", 0.0, 1e-3, key="tol_in",
                                  step=1e-7, format="%.1e",
                                  on_change=lambda: _unset_auto("tol_in"),
                                  help="相邻两次迭代加权残差 RMS 的改善量小于 tol "
                                       "即判定收敛。默认 1/fs = 一个采样周期"
                                       "（如 10 MHz → 1e-7 s，采样分辨率以下的"
                                       "改善是噪声）；改采样率会联动此默认，手动"
                                       "改过 tol 后不再联动；0 = 关闭该判据，"
                                       "仅靠停滞判据或 max_iter 终止")
        damping = st.checkbox("LM 阻尼", True) if not is_rp else True
        est_v = st.checkbox("联合反演速度", False) if not is_rp else False

    box3 = refr_box          # 介质 1（块 A）边界（米）：折射界面/折射初值域
    SCENARIO_KEY = ("scenario", kind, n_sensors, lx, ly, lz, custom_key,
                    source_mode, n_sources,
                    tuple(d * u_sz for d in grid_d_in), repeats,
                    seed, sigma_s, drop_p, fs_hz, v_true,
                    "refraction" if is_refr else "homogeneous",
                    (v2_true if is_refr else 0.0),
                    tuple(map(tuple, box3)))
    if is_rp:
        PARAMS = dict(velocity=v_used, head_velocity=v2_used,
                      grid_size=int(grid_size), margin=margin,
                      max_iter=int(max_iter), tol=tol)
    else:
        PARAMS = dict(velocity=v_used, grid_size=grid_size, margin=margin,
                      max_iter=int(max_iter), tol=tol,
                      damping=damping, estimate_velocity=est_v)


    def locator_kwargs(scenario_key, params, exclude=None, refr=None):
        """Locator constructor kwargs: user params plus per-method extras --
        ``mode='2d'`` for 2-D Geiger layouts (pure 2-D sensor coordinates
        need no fixed axis); ``box`` (specimen bounds, metres) for the
        refraction locator."""
        if refr is None:
            refr = is_rp                     # default: the selected method
        kw = dict(params)
        if exclude:
            kw.pop(exclude, None)            # a swept param must not repeat
        if refr:
            kw.update(box=box3)              # 试样盒 = 折射界面（米）
        elif scenario_key[1].startswith("2d"):
            kw.update(mode="2d")
        return kw


    @st.cache_data(show_spinner="生成合成场景…")
    def make_scenario(scenario_key):
        (_, kind, n_sensors, lx, ly, lz, custom_key, source_mode, n_sources,
         spacing, repeats, seed, sigma, drop_p, fs, v_true, model,
         head_v, refr_b) = scenario_key
        sensors = dict(custom_key)          # the text box is the single source
        dim = 3 if kind.startswith("3d") else 2
        # 震源域 = 整个试样盒 [0, size]（组合体的震源可在任一块内），与
        # 传感器包围盒解耦（部分布置如单边传感器阵列的包围盒会退化成
        # 线/点，不能拿来布源）
        box = (((0.0, 0.0, 0.0), (lx, ly, lz)) if dim == 3
               else ((0.0, 0.0), (lx, ly)))
        if source_mode.startswith("网格"):
            sources = grid_sources(spacing=tuple(spacing[:dim]), box=box)
        else:
            sources = random_sources(n_sources, box=box, seed=seed + 1)
        return Scenario(sensors=sensors, sources=sources, velocity=v_true,
                        sigma=sigma, drop_p=drop_p, fs=fs,
                        seed=seed + 2, repeats=repeats,
                        model=model, head_velocity=head_v,
                        box=(refr_b if model == "refraction" else None))


    @st.cache_data(show_spinner="定位试验运行中…")
    def run_trials_cached(scenario_key, method, params):
        sc = make_scenario(scenario_key)
        kw = locator_kwargs(scenario_key, params,
                            refr=method.startswith("折射"))
        cls = RefractionLocator if method.startswith("折射") else GeigerLocator
        return run_trials(cls(sc.sensors, **kw), sc)


    # ------------------------------------------------------- tab 1: results
    # 折射正演 / 折射定位器的两介质几何闸门：不满足时直接给中文引导，
    # 不再让 aelib 的英文 ValueError 从构造器里冒出来
    if (is_refr or is_rp) and _head_problem():
        st.error(f"折射{'正演' if is_refr else '定位器'}无法运行：{_head_problem()}。"
                 "请在侧栏把\"传感器布置\"切到\"压头（折射）\"或\"组合体"
                 "（两介质）\"，或切到\"自定义\"把不满足的坐标改掉")
        st.stop()

    tab_result, tab_sweep = st.tabs(["场景与结果", "参数敏感性"])

    with tab_result:
        try:
            df = run_trials_cached(SCENARIO_KEY, method_label, PARAMS)
        except Exception as exc:            # e.g. degenerate (collinear) layout
            st.error(f"定位无法进行：{exc}")
            st.stop()
        s = summarize(df)
        c = st.columns(7)
        c[0].metric(f"定位误差中位 ({disp_unit})", f"{s.err_median * factor:.2f}")
        c[1].metric(f"定位误差均值 ({disp_unit})", f"{s.err_mean * factor:.2f}")
        c[2].metric(f"误差 p95 ({disp_unit})", f"{s.err_p95 * factor:.2f}")
        c[3].metric("收敛率", f"{s.converged_rate * 100:.0f}%")
        c[4].metric(f"残差 rms 中位 ({time_unit})",
                    f"{s.rms_median * tfactor:.3g}")
        c[5].metric("平均迭代", f"{s.iter_mean:.1f}")
        c[6].metric("平均耗时 (ms)", f"{s.runtime_ms_mean:.2f}")
        if s.failures:
            st.warning(f"{s.failures} 个试验定位失败（通道过少等），详见表格 failure 列。")

        left, right = st.columns([3, 2])
        with left:
            st.subheader("定位结果（红=定位，绿=真值，黑=传感器）")
            sc = make_scenario(SCENARIO_KEY)
            located = np.full_like(np.asarray(sc.sources, dtype=float), np.nan)
            for r in df[df["err"].notna()].itertuples():
                located[int(r.source)] = r.coords
            fig = plot_layout(sc.sensors, sc.sources, located,
                              unit=disp_unit, factor=factor)
            st.pyplot(fig)
            close_fig(fig)
        with right:
            st.subheader("误差分布")
            fig = plot_err_hist(df, unit=disp_unit, factor=factor)
            st.pyplot(fig)
            close_fig(fig)
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
            if is_rp:
                param_opts = ["velocity", "head_velocity", "grid_size",
                              "margin", "max_iter", "tol"]
                raw_defaults = {"velocity": "3000, 5300, 5605, 7000",
                                "head_velocity": "4000, 4800, 5605, 6500",
                                "grid_size": "1, 2, 4, 8, 16",
                                "margin": "0.0, 0.05, 0.1, 0.2",
                                "max_iter": "5, 10, 20, 60, 150",
                                "tol": "0, 1e-7, 1e-6, 1e-5, 1e-4"}
            else:
                param_opts = ["grid_size", "margin", "velocity", "max_iter",
                              "tol"]
                raw_defaults = {"grid_size": "4, 8, 16, 24, 48",
                                "margin": "0.0, 0.05, 0.1, 0.2",
                                "velocity": "1500, 2600:8000:300, 10000",
                                "max_iter": "5, 10, 20, 60, 150",
                                "tol": "0, 1e-8, 1e-7, 1e-6, 1e-5"}
            param = st.selectbox("参数", param_opts,
                                 format_func=PARAM_LABELS.get)
            raw = st.text_input(
                "取值（逗号分隔，支持等间距 a:b:c 或（a,b,c））",
                raw_defaults[param])
            vals = parse_values(raw)   # 等间距 (a,b,c) = 从 a 到 b 含端点间隔 c，可与普通项混用
            if param in ("grid_size", "max_iter"):
                vals = [int(v) for v in vals]

            @st.cache_data(show_spinner="扫描运行中…（磁盘缓存）")
            def do_sweep(scenario_key, method, params, param, vals):
                sc = make_scenario(scenario_key)
                cls = (RefractionLocator if method.startswith("折射")
                       else GeigerLocator)
                return sweep(cls, {param: vals}, sc,
                             locator_kwargs=locator_kwargs(
                                 scenario_key, params, exclude=param,
                                 refr=method.startswith("折射")))
        else:
            param = st.selectbox("场景参数", ["sigma", "fs", "drop_p"],
                                 format_func=PARAM_LABELS.get)
            raw = st.text_input(
                "取值（逗号分隔，支持等间距 a:b:c 或（a,b,c））",
                                {"sigma": "0, 1e-6, 2e-6, 5e-6, 1e-5",
                                 "fs": "0, 2e6, 5e6, 1e7, 4e7",
                                 "drop_p": "0, 0.1, 0.2, 0.3"}[param])
            vals = parse_values(raw)   # 等间距 (a,b,c) = 从 a 到 b 含端点间隔 c，可与普通项混用

            @st.cache_data(show_spinner="扫描运行中…")
            def do_scenario_sweep(scenario_key, method, params, param, vals):
                sc = make_scenario(scenario_key)
                cls = (RefractionLocator if method.startswith("折射")
                       else GeigerLocator)
                loc = cls(sc.sensors,
                          **locator_kwargs(scenario_key, params,
                                           refr=method.startswith("折射")))
                frames = []
                for v in vals:
                    d = run_trials(loc, replace(sc, **{param: v}))
                    d[param] = v
                    frames.append(d)
                return pd.concat(frames, ignore_index=True)

        agg_map = {"err_p95": ("err", lambda x: x.quantile(0.95)),
                   "stderr_calibration": ("err", "median")}
        try:
            df2 = (do_sweep(SCENARIO_KEY, method_label, PARAMS, param, vals)
                   if target.startswith("定位器")
                   else do_scenario_sweep(SCENARIO_KEY, method_label, PARAMS,
                                          param, vals))
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
            fig = plot_metric_vs_param(df2, param, metric=gcol, agg=agg,
                                       dist_unit=disp_unit,
                                       dist_factor=factor,
                                       time_unit=time_unit,
                                       time_factor=tfactor)
            st.pyplot(fig)
            close_fig(fig)
        with c2:
            fig = plot_metric_vs_param(df2, param, metric="runtime_ms")
            st.pyplot(fig)
            close_fig(fig)
        with st.expander("扫描结果表"):
            st.dataframe(display_table(df2, disp_unit, factor,
                                       time_unit, tfactor))
