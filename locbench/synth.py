"""Synthetic data generators: sensor layouts, sources, arrival times.

Everything here is simulated -- locbench never touches measured data.  All
quantities are SI (metres, seconds, m/s), so sensor mappings are plain
dicts (the unit aelib assumes for them); the unit contract is aelib's
business, not the bench's.
"""
from __future__ import annotations

import itertools
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = ["corners_layout", "random_layout", "coplanar_layout",
           "random_sources", "make_arrivals"]


def corners_layout(size=(0.1, 0.1, 0.2), center=(0.0, 0.0, 0.0)) -> Dict[int, tuple]:
    """Sensors at the box corners -- the classic specimen monitoring setup
    (default 100 x 100 x 200 mm with one sensor on each of the 8 vertices;
    ``size`` is the full side length per axis).  A 2-entry ``size`` gives
    the 4 corners of a rectangle.  Keys are 1..n."""
    size = np.asarray(size, dtype=float)
    dim = size.size
    center = np.asarray(center, dtype=float)[:dim]
    pts = [center + np.array(s) * size
           for s in itertools.product((-0.5, 0.5), repeat=dim)]
    return {i + 1: tuple(p) for i, p in enumerate(pts)}


def _rng(rng, seed):
    return np.random.default_rng(rng if rng is not None else seed)


def random_layout(n_sensors: int, extent=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0),
                  min_spacing: float = 0.05, seed=None, rng=None) -> Dict[int, tuple]:
    """Uniform random sensors in a box, rejection-sampled to keep pairwise
    spacing >= ``min_spacing`` (degenerate/near-collinear layouts make
    locating ill-posed; the spacing floor plus randomness avoids them).
    Keys are 1..n.  ``extent`` length sets the dimensionality (2 or 3)."""
    rng = _rng(rng, seed)
    extent = np.asarray(extent, dtype=float)
    center = np.asarray(center, dtype=float)[:extent.size]
    dim = extent.size
    pts: list = []
    tries = 0
    while len(pts) < n_sensors:
        p = center + (rng.random(dim) - 0.5) * extent
        if all(np.linalg.norm(p - q) >= min_spacing for q in pts):
            pts.append(p)
        tries += 1
        if tries > 20_000:                 # box too crowded: relax the floor
            min_spacing *= 0.5
            tries = 0
    return {i + 1: tuple(p) for i, p in enumerate(pts)}


def coplanar_layout(n_sensors: int, extent=(1.0, 1.0), plane: str = "z",
                    height: float = 0.0, min_spacing: float = 0.05,
                    seed=None, rng=None) -> Dict[int, tuple]:
    """Layout on one plane (axis ``plane`` constant at ``height``) -- the
    geometry that needs ``mode='2d', fixed_coords={plane: height}``."""
    rng = _rng(rng, seed)
    flat = random_layout(n_sensors, extent=extent, min_spacing=min_spacing,
                         rng=rng)
    axis = {"x": 0, "y": 1, "z": 2}[plane]
    free = [i for i in range(3) if i != axis]
    return {ch: tuple(height if i == axis else p[free.index(i)]
                      for i in range(3))
            for ch, p in flat.items()}


def random_sources(n: int, sensors: dict, margin: float = 0.0,
                   seed=None, rng=None) -> np.ndarray:
    """Random source points, shape (n, dim), uniform in the sensors'
    bounding box.  ``margin`` > 0 expands the box (sources outside the
    array -- extrapolation stress), < 0 shrinks it (well-conditioned)."""
    rng = _rng(rng, seed)
    P = np.stack([np.asarray(p, dtype=float) for p in sensors.values()])
    lo, hi = P.min(axis=0), P.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    lo, hi = lo - margin * span, hi + margin * span
    return rng.uniform(lo, hi, size=(n, P.shape[1]))


def make_arrivals(sensors: dict, source, velocity: float, t0: float = 0.0,
                  sigma: float = 0.0, outlier_p: float = 0.0,
                  outlier_jitter: float = 50e-6, drop_p: float = 0.0,
                  seed=None, rng=None) -> Tuple[dict, dict]:
    """Exact arrivals ``t_i = t0 + |s_i - x| / v`` plus simulated pick
    errors.

    - ``sigma``: Gaussian picking noise per pick (s)
    - ``outlier_p``: fraction of picks hit by a gross error, uniform in
      +/- ``outlier_jitter`` (cycle slips / false triggers)
    - ``drop_p``: fraction of channels with no pick at all

    Returns ``({channel: t}, info)``; ``info`` lists the dropped and
    outlier channels so robustness metrics can separate the cases."""
    rng = _rng(rng, seed)
    chs = list(sensors)
    S = np.stack([np.asarray(sensors[c], dtype=float) for c in chs])
    d = np.linalg.norm(S - np.asarray(source, dtype=float), axis=1)
    t = t0 + d / velocity
    if sigma:
        t = t + rng.normal(0.0, sigma, t.size)
    is_out = rng.random(t.size) < outlier_p
    t = t + is_out * rng.uniform(-1.0, 1.0, t.size) * outlier_jitter
    keep = rng.random(t.size) >= drop_p
    arrivals = {c: float(ti) for c, ti, k in zip(chs, t, keep) if k}
    info = {"dropped": [c for c, k in zip(chs, keep) if not k],
            "outliers": [c for c, k, o in zip(chs, keep, is_out) if k and o]}
    return arrivals, info
