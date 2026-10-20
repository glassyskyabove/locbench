"""Matplotlib figures with Chinese labels.

A CJK-capable sans font is auto-selected at import (Windows/macOS/Linux
common fonts) so Chinese axis labels and legends render instead of tofu;
without one installed this is a harmless no-op and labels fall back.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")                   # 无头渲染；须在 pyplot 之前
import matplotlib.pyplot as plt         # 仅 rcParams / 兼容关闭用
from matplotlib.figure import Figure

from .evaluate import TRIAL_COLS

__all__ = ["plot_layout", "plot_metric_vs_param", "plot_heatmap",
           "plot_err_hist", "display_table", "unit_labels",
           "DIST_COLS", "DIST_ROWS", "TIME_COLS", "TIME_ROWS",
           "METRIC_LABELS", "PARAM_LABELS", "SUMMARY_LABELS", "TRIAL_LABELS"]

#: 距离列/行（米 -> 显示单位缩放，单位注记同步替换）
DIST_COLS = ("err", "dx", "dy", "dz", "stderr_pos")
DIST_ROWS = ("err_median", "err_mean", "err_p95")
#: 时间列/行（秒 -> 显示单位缩放，'(s)' 注记同步替换）
TIME_COLS = ("t0_err", "rms", "sigma")
TIME_ROWS = ("t0_err_median", "rms_median")


def unit_labels(labels, keys, unit, marker="(m)"):
    """把 labels 中 keys 的 marker 注记（如 '(m)'/'(s)'）替换为显示单位
    （unit 与 marker 的基准单位一致时原样返回）。"""
    if unit == marker[1:-1]:
        return labels
    return {k: (v.replace(marker, f"({unit})") if k in keys else v)
            for k, v in labels.items()}


def display_table(df, unit="m", factor=1.0, time_unit="s", time_factor=1.0):
    """表格中文显示：列名映射、距离/时间列各自换算成显示单位（米 x
    factor、秒 x time_factor）、收敛标记转'是/否'。仅显示层，不改动原
    DataFrame。"""
    show = df.drop(columns=["coords"], errors="ignore").copy()
    for cols, f in ((DIST_COLS, factor), (TIME_COLS, time_factor)):
        if f != 1.0:
            for c in cols:
                if c in show:
                    show[c] = show[c] * f
    if "converged" in show:
        show = show.replace({"converged": {True: "是", False: "否"}})
    labels = unit_labels({**TRIAL_LABELS, **METRIC_LABELS, **PARAM_LABELS},
                         DIST_COLS, unit, "(m)")
    labels = unit_labels(labels, TIME_COLS, time_unit, "(s)")
    return show.rename(columns=labels)

#: 中文显示名（试用指标 / 扫描参数；未列出的键原样显示）
METRIC_LABELS = {
    "err": "定位误差 (m)",
    "err_p95": "定位误差 p95 (m)",
    "t0_err": "发震时刻误差 (s)",
    "rms": "残差 RMS (s)",
    "n_iter": "迭代次数",
    "runtime_ms": "单次定位耗时 (ms)",
    "stderr_calibration": "形式误差校准比",
}
PARAM_LABELS = {
    "grid_size": "初值点个数（grid size）",
    "margin": "外延比例（margin）",
    "velocity": "波速 (m/s)",
    "max_iter": "迭代上限（max_iter）",
    "tol": "收敛判据（tol s）",
    "sigma": "到时误差 σ (s)",
    "fs": "采样率 (Hz)",
    "drop_p": "丢道概率",
}
#: 汇总统计（summarize 输出）的中文行名
SUMMARY_LABELS = {
    "trials": "试验数",
    "failures": "失败数",
    "converged_rate": "收敛率",
    "err_median": "定位误差中位 (m)",
    "err_mean": "定位误差均值 (m)",
    "err_p95": "定位误差 p95 (m)",
    "t0_err_median": "t0 误差中位 (s)",
    "rms_median": "残差 RMS 中位 (s)",
    "iter_mean": "平均迭代次数",
    "runtime_ms_mean": "平均单次耗时 (ms)",
    "stderr_calibration": "形式误差校准比",
}
#: 逐试验表的中文列名（dataframe 本身的列名保持英文键，仅显示时映射）
TRIAL_LABELS = {
    "source": "震源序号",
    "repeat": "噪声重复",
    "n_picks": "到时通道数",
    "n_used": "参与定位通道数",
    "converged": "是否收敛",
    "err": "定位误差 (m)",
    "t0_err": "t0 误差 (s)",
    "rms": "残差 RMS (s)",
    "n_iter": "迭代次数",
    "stderr_pos": "形式误差模长 (m)",
    "velocity": "波速 (m/s)",
    "runtime_ms": "耗时 (ms)",
    "failure": "失败原因",
    "dx": "x 偏差 (m)",
    "dy": "y 偏差 (m)",
    "dz": "z 偏差 (m)",
}


def _cjk_font():
    from matplotlib import font_manager
    names = ["Microsoft YaHei", "SimHei", "PingFang SC", "Hiragino Sans GB",
             "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Micro Hei"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    for n in names:
        if n in have:
            sans = plt.rcParams["font.sans-serif"]
            plt.rcParams["font.sans-serif"] = [n] + [x for x in sans if x != n]
            plt.rcParams["axes.unicode_minus"] = False
            return


_cjk_font()


def plot_layout(sensors: dict, sources=None, located=None,
                unit: str = "m", factor: float = 1.0):
    """台阵布置 + 真值/定位震源；真值->定位的线段即误差向量。2-D/3-D
    跟随传感器维度。坐标按显示单位缩放（数据本身是米，``factor``
    米->显示单位，如 mm 取 1e3），轴标注带单位。"""
    P = np.stack([np.asarray(p, dtype=float) for p in sensors.values()]) * factor
    dim = P.shape[1]
    # 纯 OO Figure：不碰 pyplot 的全局 figure 注册表。streamlit 每个会话
    # 一个脚本线程，多会话并发时 pyplot 全局状态会互相踩踏（tight bbox
    # 偶发塌缩 -> PNG 尺寸漂移 -> 浏览器拉伸后"点的大小每次不一样"）。
    fig = Figure(figsize=(6, 5))
    if dim == 3:
        ax = fig.add_subplot(projection="3d")
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], marker="s", c="k",
                   label="传感器", s=28)
    else:
        ax = fig.add_subplot()
        ax.scatter(P[:, 0], P[:, 1], marker="s", c="k", label="传感器", s=28)
        ax.set_aspect("equal")
    # 视野由台阵与真值震源决定：个别发散到远处的定位解不参与取限
    # ——否则整幅图被拉爆（等轴盒子被压成薄片、固定大小的标记显得
    # 巨大、长刻度数字溢出画布）。matplotlib 三维不按坐标限裁剪散点，
    # 超视野的发散解会被投影画进图内，因此先算好视野再过滤再画。
    # 被滤掉的定位解不显示，其误差仍见逐试验表格。
    base = np.vstack([P] + ([np.atleast_2d(np.asarray(sources, dtype=float))
                             * factor] if sources is not None else []))
    lo, hi = base.min(axis=0), base.max(axis=0)
    m = 0.05 * max(float((hi - lo).max()), 1e-12)      # 5% 最大跨度边距
    if sources is not None:
        S = np.atleast_2d(np.asarray(sources, dtype=float)) * factor
        _scatter(ax, S, c="tab:green", label="真值震源")
    if located is not None:
        L = np.atleast_2d(np.asarray(located, dtype=float)) * factor
        keep = np.isfinite(L).all(axis=1)
        keep &= ((L >= lo - m) & (L <= hi + m)).all(axis=1)
        _scatter(ax, L[keep], c="tab:red", label="定位结果")
        if sources is not None:
            for i in np.where(keep)[0]:      # 按原下标配对：过滤后仍对应
                if i >= len(S):              # 同一个真值震源
                    break
                s, l = S[i], L[i]
                if dim == 3:
                    ax.plot([s[0], l[0]], [s[1], l[1]], [s[2], l[2]],
                            c="tab:red", lw=0.8, alpha=0.7)
                else:
                    ax.plot([s[0], l[0]], [s[1], l[1]],
                            c="tab:red", lw=0.8, alpha=0.7)
    for name in ("x", "y", "z")[:dim]:
        getattr(ax, f"set_{name}label")(f"{name} ({unit})")
    if dim == 3:
        for i, name in enumerate("xyz"):
            getattr(ax, f"set_{name}lim")(lo[i] - m, hi[i] + m)
        # 等轴：盒子三边比例 = 数据物理跨度比例（默认立方体会把
        # 100x100x200 的试样画成正方体）
        span = np.maximum(hi - lo + 2 * m, 1e-9 * (hi - lo + 2 * m).max())
        ax.set_box_aspect(tuple(span))
    else:
        ax.set_xlim(lo[0] - m, hi[0] + m)
        ax.set_ylim(lo[1] - m, hi[1] + m)
    return fig        # 不画图例：颜色含义写在界面小标题里


def _scatter(ax, pts, **kw):
    if pts.shape[1] == 3:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], **kw)
    else:
        ax.scatter(pts[:, 0], pts[:, 1], **kw)


def close_fig(fig) -> None:
    """Backward-compat hook: OO Figures never enter pyplot's global
    registry, so plt.close is a harmless no-op on them.  Kept so the app
    code reads the same after every plot call."""
    plt.close(fig)


def plot_metric_vs_param(df, param: str, metric: str = "err",
                         agg: str = "median", ax=None,
                         dist_unit: str = "m", dist_factor: float = 1.0,
                         time_unit: str = "s", time_factor: float = 1.0):
    """指标随单个扫描参数的变化：中位线 + 四分位带。其余扫描参数
    （不属于试用指标的列）各画一条线。距离类指标（err）与时间类指标
    （t0_err/rms/sigma）按各自显示单位缩放并标注。"""
    agg_cn = {"median": "中位", "mean": "均值"}.get(agg, agg)
    if ax is None:
        fig = Figure(figsize=(6.5, 4))
        ax = fig.add_subplot()
    others = [c for c in df.columns
              if c not in TRIAL_COLS and c != param and df[c].nunique() > 1]
    sub = df.dropna(subset=[metric])
    if metric == "err":
        sub = sub.assign(err=sub["err"] * dist_factor)
    elif metric in TIME_COLS:
        sub = sub.assign(**{metric: sub[metric] * time_factor})
    for key, grp in (sub.groupby(others) if others
                     else [(None, sub)]):
        g = grp.groupby(param)[metric]
        center = g.agg(agg).sort_index()
        lo = g.quantile(0.25).loc[center.index]
        hi = g.quantile(0.75).loc[center.index]
        x = np.asarray(center.index, dtype=float)
        label = (PARAM_LABELS.get(param, param) if key is None else str(
            dict(zip(others, key if isinstance(key, tuple) else (key,)))))
        ax.plot(x, center.values, marker="o", label=label)
        ax.fill_between(x, lo.values, hi.values, alpha=0.2)
    ax.set_xlabel(PARAM_LABELS.get(param, param))
    if param == "tol":
        # tol 跨数量级（如 0 ~ 1e-5），线性轴会挤在 0 附近。全为正时用
        # 对数轴；含 0（"关闭判据"档）时用 symlog——0 落在最左端的线性
        # 段，正值区仍呈对数间距
        xs = df[param].dropna()
        pos = xs[xs > 0]
        if len(pos):
            if (xs == 0).any():
                ax.set_xscale("symlog", linthresh=float(pos.min()) / 10)
            else:
                ax.set_xscale("log")
    if metric == "err":
        ylab = f"定位误差 ({dist_unit})"
    else:
        ylab = METRIC_LABELS.get(metric, metric)
        if metric in TIME_COLS:
            ylab = ylab.replace("(s)", f"({time_unit})")
    ax.set_ylabel(f"{ylab}（{agg_cn}）")
    if others:
        ax.legend(fontsize=8)
    return fig


def plot_heatmap(df, px: str, py: str, metric: str = "err",
                 agg: str = "median", ax=None):
    """两个扫描参数 x 一个指标的带数值标注热力图。"""
    piv = df.pivot_table(index=py, columns=px, values=metric, aggfunc=agg)
    if ax is None:
        fig = Figure(figsize=(6, 4.5))
        ax = fig.add_subplot()
    im = ax.imshow(piv.values, aspect="auto", origin="lower")
    ax.set_xticks(range(len(piv.columns)), [str(c) for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [str(i) for i in piv.index])
    ax.set_xlabel(PARAM_LABELS.get(px, px))
    ax.set_ylabel(PARAM_LABELS.get(py, py))
    vmax = np.nanmax(piv.values) if np.isfinite(np.nanmax(piv.values)) else 1
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.3g}" if np.isfinite(v) else "-",
                    ha="center", va="center", fontsize=7,
                    color=("w" if v > 0.6 * vmax else "k") if np.isfinite(v) else "k")
    fig.colorbar(im, ax=ax, label=f"{METRIC_LABELS.get(metric, metric)}")
    return fig


def plot_err_hist(df, ax=None, unit: str = "m", factor: float = 1.0):
    if ax is None:
        fig = Figure(figsize=(5.5, 4))
        ax = fig.add_subplot()
    errs = df["err"].dropna() * factor
    ax.hist(errs, bins=30)
    ax.set_xlabel(f"定位误差 ({unit})")
    ax.set_ylabel("试验数")
    return fig
