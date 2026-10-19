"""Matplotlib figures (English labels -- default matplotlib fonts lack CJK)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .evaluate import TRIAL_COLS

__all__ = ["plot_layout", "plot_metric_vs_param", "plot_heatmap",
           "plot_err_hist"]


def plot_layout(sensors: dict, sources=None, located=None):
    """Sensor layout with true and located sources; true->located segments
    visualise the localisation error vectors.  2-D or 3-D, following the
    sensor dimensionality."""
    P = np.stack([np.asarray(p, dtype=float) for p in sensors.values()])
    dim = P.shape[1]
    if dim == 3:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], marker="s", c="k",
                   label="sensors", s=28)
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(P[:, 0], P[:, 1], marker="s", c="k", label="sensors", s=28)
        ax.set_aspect("equal")
    if sources is not None:
        S = np.atleast_2d(np.asarray(sources, dtype=float))
        _scatter(ax, S, c="tab:green", label="true source")
    if located is not None:
        L = np.atleast_2d(np.asarray(located, dtype=float))
        finite = np.isfinite(L).all(axis=1)
        _scatter(ax, L[finite], c="tab:red", label="located")
        if sources is not None:
            for s, l in zip(S[finite] if sources is not None else [],
                            L[finite]):
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
    """Metric vs one swept parameter: central line + interquartile band.
    Other swept parameters (columns outside the trial metrics) split the
    data into one line each."""
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
        label = param if key is None else str(
            dict(zip(others, key if isinstance(key, tuple) else (key,))))
        ax.plot(x, center.values, marker="o", label=label)
        ax.fill_between(x, lo.values, hi.values, alpha=0.2)
    ax.set_xlabel(param)
    ax.set_ylabel(f"{metric} ({agg})")
    if others:
        ax.legend(fontsize=8)
    return fig


def plot_heatmap(df, px: str, py: str, metric: str = "err",
                 agg: str = "median", ax=None):
    """Two swept parameters vs one metric as an annotated heatmap."""
    piv = df.pivot_table(index=py, columns=px, values=metric, aggfunc=agg)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4.5))
    else:
        fig = ax.figure
    im = ax.imshow(piv.values, aspect="auto", origin="lower")
    ax.set_xticks(range(len(piv.columns)), [str(c) for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [str(i) for i in piv.index])
    ax.set_xlabel(px)
    ax.set_ylabel(py)
    vmax = np.nanmax(piv.values) if np.isfinite(np.nanmax(piv.values)) else 1
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.3g}" if np.isfinite(v) else "-",
                    ha="center", va="center", fontsize=7,
                    color=("w" if v > 0.6 * vmax else "k") if np.isfinite(v) else "k")
    fig.colorbar(im, ax=ax, label=f"{metric} ({agg})")
    return fig


def plot_err_hist(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4))
    else:
        fig = ax.figure
    errs = df["err"].dropna()
    ax.hist(errs, bins=30)
    ax.set_xlabel("location error (m)")
    ax.set_ylabel("trials")
    return fig
