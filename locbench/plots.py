"""Matplotlib figures with Chinese labels.

A CJK-capable sans font is auto-selected at import (Windows/macOS/Linux
common fonts) so Chinese axis labels and legends render instead of tofu;
without one installed this is a harmless no-op and labels fall back.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .evaluate import TRIAL_COLS

__all__ = ["plot_layout", "plot_metric_vs_param", "plot_heatmap",
           "plot_err_hist", "METRIC_LABELS", "PARAM_LABELS"]

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
    "grid_size": "grid_size（网格点数/轴）",
    "margin": "margin（包围盒外扩比例）",
    "velocity": "波速 (m/s)",
    "sigma": "拾取噪声 σ (s)",
    "outlier_p": "粗差比例",
    "outlier_jitter": "粗差幅度 (s)",
    "drop_p": "丢道比例",
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


def plot_layout(sensors: dict, sources=None, located=None):
    """台阵布置 + 真值/定位震源；真值->定位的线段即误差向量。2-D/3-D
    跟随传感器维度。"""
    P = np.stack([np.asarray(p, dtype=float) for p in sensors.values()])
    dim = P.shape[1]
    if dim == 3:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], marker="s", c="k",
                   label="传感器", s=28)
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(P[:, 0], P[:, 1], marker="s", c="k", label="传感器", s=28)
        ax.set_aspect("equal")
    if sources is not None:
        S = np.atleast_2d(np.asarray(sources, dtype=float))
        _scatter(ax, S, c="tab:green", label="真值震源")
    if located is not None:
        L = np.atleast_2d(np.asarray(located, dtype=float))
        finite = np.isfinite(L).all(axis=1)
        _scatter(ax, L[finite], c="tab:red", label="定位结果")
        if sources is not None:
            for s, l in zip(S[finite], L[finite]):
                if dim == 3:
                    ax.plot([s[0], l[0]], [s[1], l[1]], [s[2], l[2]],
                            c="tab:red", lw=0.8, alpha=0.7)
                else:
                    ax.plot([s[0], l[0]], [s[1], l[1]],
                            c="tab:red", lw=0.8, alpha=0.7)
    ax.legend()
    return fig


def _scatter(ax, pts, **kw):
    if pts.shape[1] == 3:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], **kw)
    else:
        ax.scatter(pts[:, 0], pts[:, 1], **kw)


def plot_metric_vs_param(df, param: str, metric: str = "err",
                         agg: str = "median", ax=None):
    """指标随单个扫描参数的变化：中位线 + 四分位带。其余扫描参数
    （不属于试用指标的列）各画一条线。"""
    agg_cn = {"median": "中位", "mean": "均值"}.get(agg, agg)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4))
    else:
        fig = ax.figure
    others = [c for c in df.columns
              if c not in TRIAL_COLS and c != param and df[c].nunique() > 1]
    sub = df.dropna(subset=[metric])
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
    ax.set_ylabel(f"{METRIC_LABELS.get(metric, metric)}（{agg_cn}）")
    if others:
        ax.legend(fontsize=8)
    return fig


def plot_heatmap(df, px: str, py: str, metric: str = "err",
                 agg: str = "median", ax=None):
    """两个扫描参数 x 一个指标的带数值标注热力图。"""
    piv = df.pivot_table(index=py, columns=px, values=metric, aggfunc=agg)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4.5))
    else:
        fig = ax.figure
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


def plot_err_hist(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4))
    else:
        fig = ax.figure
    errs = df["err"].dropna()
    ax.hist(errs, bins=30)
    ax.set_xlabel("定位误差 (m)")
    ax.set_ylabel("试验数")
    return fig
