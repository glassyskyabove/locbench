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

__all__ = ["corners_layout", "feature_layout", "head_layout",
           "composite_layout", "interface_plane", "random_layout",
           "random_int_layout", "coplanar_layout", "random_sources",
           "grid_sources", "parse_layout", "make_arrivals"]


def random_int_layout(n_sensors: int, extent=(100, 100, 200),
                      min_spacing=None, seed=None, rng=None) -> Dict[int, tuple]:
    """Uniform random INTEGER sensor coordinates on the lattice
    ``[0, extent]`` per axis (extent in integer units -- naturally a
    millimetre grid when the sizes are in mm).  ``min_spacing`` (same
    integer unit, default 5% of the smallest extent, at least 1) keeps
    points apart via rejection sampling, relaxing if the lattice is too
    crowded.  Keys are 1..n.  Deterministic given ``seed``."""
    rng = _rng(rng, seed)
    extent = np.asarray(extent, dtype=int)
    dim = extent.size
    if min_spacing is None:
        min_spacing = max(1, int(round(0.05 * extent.min())))
    pts: list = []
    tries = 0
    while len(pts) < n_sensors:
        p = rng.integers(0, extent + 1)
        if all(np.linalg.norm(p - np.asarray(q)) >= min_spacing for q in pts):
            pts.append(p)
        tries += 1
        if tries > 20_000:
            min_spacing = max(1, min_spacing // 2)
            tries = 0
    return {i + 1: tuple(int(v) for v in p) for i, p in enumerate(pts)}


def parse_layout(text: str, n_components: int = 3,
                 to_m: float = 1.0) -> Dict[int, tuple]:
    """Parse a free-form sensor layout: one sensor per line, coordinates
    comma-separated (full-width commas and ';' tolerated), blank lines and
    '#' comments skipped.  Values are multiplied by ``to_m`` (e.g. 1e-3
    when entered in mm).  Keys are 1..n in input order; raises ValueError
    (Chinese, UI-facing) with the offending line number on bad rows or
    fewer than 2 sensors."""
    sensors: Dict[int, tuple] = {}
    for i, raw in enumerate(text.splitlines(), 1):
        line = raw.strip().replace("，", ",").replace(";", ",")
        if not line or line.startswith("#"):
            continue
        try:
            vals = [float(p) * to_m for p in line.split(",") if p.strip()]
        except ValueError:
            raise ValueError(f"第 {i} 行无法解析为数字：{raw!r}") from None
        if len(vals) != n_components:
            raise ValueError(f"第 {i} 行需要 {n_components} 个坐标分量，"
                             f"得到 {len(vals)}：{raw!r}")
        sensors[len(sensors) + 1] = tuple(vals)
    if len(sensors) < 2:
        raise ValueError("至少需要 2 个传感器（每行一个）")
    return sensors


def grid_sources(sensors: dict = None, spacing=1, interior: bool = True,
                 box=None) -> np.ndarray:
    """Regular grid of source points over a box.

    The box defaults to the sensors' bounding box; pass ``box=(lo, hi)``
    (per-axis sequences, e.g. the specimen box) to decouple the source
    region from the sensor positions -- a partial array (one edge, say)
    must not shrink the source grid to its own bounding box.

    ``spacing`` is the target cell size (one value for all axes or a
    per-axis sequence); cells per axis = ``round(span / spacing)`` (>= 1).
    ``interior=True`` (default) takes cell CENTRES -- strictly inside the
    box, so a grid point never coincides with a corner sensor;
    ``interior=False`` includes the faces (``n+1`` points per axis).
    Constant axes give a single layer.  Returns ``(n_points, dim)``;
    deterministic (no seed).
    """
    lo, hi = _box_bounds(sensors, box)
    span = hi - lo
    dim = lo.size
    sp = np.broadcast_to(np.asarray(spacing, dtype=float), (dim,))
    n = np.maximum(1, np.round(span / sp).astype(int))
    axes = []
    for i in range(dim):
        if interior:
            axes.append(lo[i] + (np.arange(n[i]) + 0.5) * span[i] / n[i])
        else:
            axes.append(np.linspace(lo[i], hi[i], n[i] + 1))
    return np.stack(np.meshgrid(*axes, indexing="ij"),
                    axis=-1).reshape(-1, dim)


def _box_bounds(sensors, box):
    """(lo, hi) float arrays: explicit ``box`` or the sensors' bbox."""
    if box is not None:
        lo = np.asarray(box[0], dtype=float)
        hi = np.asarray(box[1], dtype=float)
        if lo.shape != hi.shape or lo.size < 2:
            raise ValueError("box 需要 (lo, hi) 两个等长序列（每轴一个值）")
        return lo, hi
    P = np.stack([np.asarray(p, dtype=float) for p in sensors.values()])
    return P.min(axis=0), P.max(axis=0)


def corners_layout(size=(0.1, 0.1, 0.2), origin=(0.0, 0.0, 0.0)) -> Dict[int, tuple]:
    """Sensors at the box corners -- the classic specimen monitoring setup
    (default 100 x 100 x 200 mm with one sensor on each of the 8 vertices;
    ``size`` is the full side length per axis).  The box spans
    ``[origin, origin + size]`` per axis, so the default puts the whole
    specimen in the all-positive octant.  A 2-entry ``size`` gives the 4
    corners of a rectangle.  Keys are 1..n."""
    size = np.asarray(size, dtype=float)
    dim = size.size
    origin = np.asarray(origin, dtype=float)[:dim]
    pts = [origin + np.array(s) * size
           for s in itertools.product((0.0, 1.0), repeat=dim)]
    return {i + 1: tuple(p) for i, p in enumerate(pts)}


#: 边特征 -> 平行的轴向（该向的全部边：三维 4 条、二维 2 条）
EDGE_FEATURES = {"横边": 0, "纵边": 1, "垂边": 2}
#: 面特征 -> 法向轴向（该向的两个面，各取面心）
FACE_FEATURES = {"侧面": 0, "正面": 1, "顶面": 2}


def feature_layout(size, edges: dict = None, faces=(),
                   vertices: bool = False) -> Dict[int, tuple]:
    """Sensors on named geometric features of the box ``[0, size]``.

    ``edges`` maps an edge feature name (``横边``/``纵边``/``垂边`` =
    edges parallel to x/y/z -- ALL of them: four per axis in 3-D, two
    in 2-D) to the number of equally spaced points PER EDGE: ``n``
    sensors at ``k/(n+1) · span``, k = 1..n -- the interior division
    points, box corners excluded, so edges never duplicate a corner.
    ``faces`` lists face feature names (``侧面``/``正面``/``顶面`` =
    the TWO faces with x/y/z normal); each face contributes ONE sensor
    at its centre.  ``vertices=True`` adds one sensor on every box
    corner (8 in 3-D, 4 in 2-D) -- the classic specimen setup,
    combinable with edges since edge points are interior.  Points come
    out edges first (x, y, z order, each edge's points grouped), then
    face centres (low side first), then corners; keys are 1..n.
    A 2-entry ``size`` allows only 横边/纵边 and 顶点 (no z features).
    """
    size = np.asarray(size, dtype=float)
    dim = size.size
    if dim not in (2, 3):
        raise ValueError("size 需要 2 或 3 个分量")
    pts: list = []
    for name, ax in EDGE_FEATURES.items():
        n = (edges or {}).get(name)
        if n is None:
            continue
        n = int(n)
        if n < 1:
            raise ValueError(f"{name}等分点数至少为 1")
        if dim == 2 and ax == 2:
            raise ValueError("二维布置没有垂边（z 方向的边）")
        others = [i for i in range(dim) if i != ax]
        for combo in itertools.product((0.0, 1.0), repeat=dim - 1):
            for k in range(1, n + 1):        # k/(n+1): 内部等分点
                p = np.zeros(dim)
                p[ax] = k * size[ax] / (n + 1)
                for i, c in zip(others, combo):
                    p[i] = c * size[i]
                pts.append(tuple(p))
    for name in faces or ():
        ax = FACE_FEATURES[name]              # KeyError = unknown name
        if dim == 2 and ax == 2:
            raise ValueError("二维布置没有顶面（z 方向的面）")
        for side in (0.0, 1.0):               # 法向两侧各一个面心
            c = size / 2.0
            c[ax] = side * size[ax]
            pts.append(tuple(c))
    if vertices:
        pts.extend(tuple(np.array(s) * size)
                   for s in itertools.product((0.0, 1.0), repeat=dim))
    return {i + 1: p for i, p in enumerate(pts)}


def head_layout(size, per_face: int = 2, offset: float = 0.01,
                min_spacing: float = None, seed=None, rng=None) -> Dict[int, tuple]:
    """Sensors on the pressure heads of the box ``[0, size]`` (3-D only) --
    the layout RefractionLocator needs: every sensor strictly outside the
    box on exactly ONE axis (``offset`` m out along that face's normal),
    in-plane coordinates uniform over the face with a ``min_spacing``
    floor (default 10 % of the smallest face side), rejection-sampled.
    ``per_face=1`` takes each face centre.  Six faces -> ``6 * per_face``
    sensors, keys 1..n, deterministic given ``seed``; SI units.

    Note: a UNIFORM small offset makes the head-path lengths similar and
    the joint v1 inversion weakly conditioned under noise at strong
    v1/v2 contrast -- the paper's layouts use varied offsets (5-50 mm
    on a 100 mm cube).  Use a generous ``offset`` (or edit the generated
    coordinates via 自定义) when testing strong contrasts."""
    rng = _rng(rng, seed)
    size = np.asarray(size, dtype=float)
    if size.size != 3:
        raise ValueError("压头布置需要三维尺寸（x, y, z）")
    if per_face < 1:
        raise ValueError("每个压头的传感器数至少为 1")
    face_lo, face_hi = 0.05 * size, 0.95 * size   # keep in-plane off edges
    if min_spacing is None:
        min_spacing = 0.1 * size.min()
    pts: list = []
    tries = 0
    for ax in range(3):
        for side in (0.0, 1.0):
            for _ in range(per_face):
                while True:
                    p = rng.uniform(face_lo, face_hi)
                    if per_face == 1:
                        p = 0.5 * size            # single sensor: face centre
                    p[ax] = side * size[ax] + (offset if side else -offset)
                    same_face = [q for q in pts
                                 if np.sign(q[ax] - size[ax] / 2) ==
                                 np.sign(p[ax] - size[ax] / 2)]
                    if all(np.linalg.norm(p - q) >= min_spacing
                           for q in same_face):
                        break
                    tries += 1
                    if tries > 20_000:            # crowded face: relax floor
                        min_spacing *= 0.5
                        tries = 0
                pts.append(p)
    return {i + 1: tuple(float(v) for v in p) for i, p in enumerate(pts)}


def composite_layout(size, per_face: int = 2, seed=None, rng=None) -> Dict[int, tuple]:
    """Sensors on the 6 outer faces of the specimen box (3-D only):
    ``per_face`` random in-plane positions per face -> 6*per_face sensors,
    keys 1..n, deterministic given ``seed``; SI units.  Deliberately
    DECOUPLED from the interface: whichever side of the interface plane a
    sensor lands on is its medium (classified at use time by aelib), so
    the plane can move/tilt freely without regenerating the layout --
    pair with ``Scenario(model="refraction", interface=(point, normal),
    head_velocity=<medium-2 velocity>)`` and aelib's plane-mode
    RefractionLocator."""
    rng = _rng(rng, seed)
    size = np.asarray(size, dtype=float)
    if size.size != 3:
        raise ValueError("组合体布置需要三维尺寸（x, y, z）")
    if per_face < 1:
        raise ValueError("每个面的传感器数至少为 1")
    pts: list = []
    for ax in range(3):
        for side in (0.0, 1.0):
            for _ in range(per_face):
                p = rng.uniform(0.05 * size, 0.95 * size)
                p[ax] = side * size[ax]
                pts.append(p)
    return {i + 1: tuple(float(v) for v in p) for i, p in enumerate(pts)}


def interface_plane(size, phi_deg: float = 0.0, psi_deg: float = 0.0,
                    offset: float = 0.0):
    """The composite interface as ``(point, normal)`` (SI), from slider
    controls: ``phi_deg`` = azimuth of the normal's horizontal projection
    (0=+x, 90=+y), ``psi_deg`` = tilt of the normal off the horizontal
    (+ toward +z; +-90 deg = horizontal interface, normal exactly +-z,
    azimuth irrelevant),
    ``offset`` = the plane's signed DISTANCE (m) from the specimen centre
    along the normal (positive = toward medium 2).  Normal points into
    medium 2."""
    size = np.asarray(size, dtype=float)
    if abs(abs(psi_deg) - 90.0) < 1e-9:      # 水平界面：法向精确 ±z
        n = np.array([0.0, 0.0, 1.0 if psi_deg > 0 else -1.0])
    else:
        phi, psi = np.radians(phi_deg), np.radians(psi_deg)
        n = np.array([np.cos(psi) * np.cos(phi),
                      np.cos(psi) * np.sin(phi),
                      np.sin(psi)])
    n_hat = n / np.linalg.norm(n)
    point = size / 2.0 + offset * n_hat
    return (tuple(float(v) for v in point), tuple(float(v) for v in n_hat))


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


def random_sources(n: int, sensors: dict = None, margin: float = 0.0,
                   box=None, seed=None, rng=None) -> np.ndarray:
    """Random source points, shape (n, dim), uniform in a box.

    The box defaults to the sensors' bounding box; ``box=(lo, hi)``
    overrides it (e.g. the specimen box for a partial sensor array).
    ``margin`` > 0 expands the box (sources outside the array --
    extrapolation stress), < 0 shrinks it (well-conditioned)."""
    rng = _rng(rng, seed)
    lo, hi = _box_bounds(sensors, box)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    lo, hi = lo - margin * span, hi + margin * span
    return rng.uniform(lo, hi, size=(n, lo.size))


def make_arrivals(sensors: dict, source, velocity: float, t0: float = 0.0,
                  sigma: float = 0.0, drop_p: float = 0.0, fs: float = 0.0,
                  seed=None, rng=None, model: str = "homogeneous",
                  head_velocity: float = 0.0, box=None,
                  interface=None) -> Tuple[dict, dict]:
    """Exact arrivals plus simulated pick errors.

    - ``model="homogeneous"`` (default): straight-line propagation
      ``t_i = t0 + |s_i - x| / v``
    - ``model="refraction"``: two-media forward model (aelib's
      RefractionLocator forward model, SI).  Either box-faces geometry
      (sensors must sit on heads outside ``box`` = ``((x0,y0,z0),
      (x1,y1,z1))``, required) or, when ``interface = (point, normal)``
      is given (normal into medium 2), a SINGLE arbitrarily-oriented
      plane splits the media and ``box`` is only a fallback.  In both
      cases ``velocity`` is the medium-1 velocity v1 and
      ``head_velocity`` the known medium-2 velocity v2;  ``v1 == v2``
      degenerates to the homogeneous model.
    - ``sigma``: Gaussian picking noise per pick (s)
    - ``fs``: sampling rate (Hz) -- when > 0 the arrival times are
      quantized to the sampling period 1/fs AFTER the Gaussian noise,
      modelling the instrument's time resolution (e.g. 10 MHz -> 0.1 us
      ticks); ``fs = 0`` keeps the times continuous
    - ``drop_p``: per-channel drop probability -- each channel draws an
      independent uniform [0,1) random number per call (event) and is
      dropped when it falls below ``drop_p``

    Returns ``({channel: t}, info)``; ``info`` lists the dropped
    channels."""
    rng = _rng(rng, seed)
    chs = list(sensors)
    S = np.stack([np.asarray(sensors[c], dtype=float) for c in chs])
    if model == "refraction":
        from aelib.locate.refraction import travel_times_refraction
        if interface is not None:
            # 单一平面分界：介质由位置决定，box/dirs 不参与
            lo = hi = dirs = None
        else:
            if box is None:
                raise ValueError("折射模型需要 box=((x0,y0,z0),(x1,y1,z1))"
                                 "（介质 1 边界，米）或 interface=(点, 法向)"
                                 "（单一分界面）")
            from aelib.locate.refraction import classify_heads
            lo, hi = (np.asarray(b, dtype=float) for b in box)
            dirs = classify_heads(S, lo, hi)
        t = t0 + travel_times_refraction(
            np.asarray(source, dtype=float), S, dirs, velocity,
            head_velocity, lo, hi, interface=interface)
    elif model == "homogeneous":
        d = np.linalg.norm(S - np.asarray(source, dtype=float), axis=1)
        t = t0 + d / velocity
    else:
        raise ValueError(f"未知波速模型 {model!r}（'homogeneous' 或 'refraction'）")
    if sigma:
        t = t + rng.normal(0.0, sigma, t.size)
    if fs:
        t = np.round(t * fs) / fs          # sample-clock quantization
    keep = rng.random(t.size) >= drop_p
    arrivals = {c: float(ti) for c, ti, k in zip(chs, t, keep) if k}
    info = {"dropped": [c for c, k in zip(chs, keep) if not k]}
    return arrivals, info
