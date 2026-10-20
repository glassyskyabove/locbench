"""Parameter-grid sweeps over a locator constructor, with disk caching."""
from __future__ import annotations

import hashlib
import itertools
import os
import time
from typing import Dict, Optional, Sequence

import pandas as pd

__all__ = ["sweep", "parse_values", "clean_cache"]


def clean_cache(cache_dir: str = "results", max_age_days: float = 7.0) -> int:
    """删除 ``cache_dir`` 下修改时间早于 ``max_age_days`` 天的
    ``sweep_*.pkl`` 缓存文件，返回删除数。

    缓存按（定位器、参数网格、场景、aelib 版本）哈希命名：任一设置
    变化后旧文件永远不会再命中，只会占硬盘，因此由 :func:`sweep`
    在每次运行入口自动清理（7 天保留期，可通过 ``max_age_days``
    调整）。只动 ``sweep_*.pkl``，目录里其他文件不碰。"""
    if not os.path.isdir(cache_dir):
        return 0
    cutoff = time.time() - max_age_days * 86400.0
    removed = 0
    for name in os.listdir(cache_dir):
        if not (name.startswith("sweep_") and name.endswith(".pkl")):
            continue
        path = os.path.join(cache_dir, name)
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
                removed += 1
        except OSError:
            pass
    return removed


def parse_values(text: str) -> list:
    """逗号分隔的扫描取值列表，支持等间距简写。

    等间距项可写成 ``2000:4000:100``（冒号式）或 ``(2000, 4000, 100)``
    / 全角 ``（2000，4000，100）``（括号式），都表示从 2000 到 4000
    **含两端**、间隔 100，即 2000, 2100, ..., 4000（共 21 个值）。
    等间距项可与普通项混用，如 ``4, 8, (10, 20, 2), 48``；负步长需
    起点 > 终点。全角逗号/分号、中英文括号均可。非法表达式抛
    ValueError（中文，界面可直接显示）。
    """
    s = (text.replace("，", ",").replace("；", ";").replace(";", ",")
             .replace("（", "(").replace("）", ")"))
    # 括号外的逗号才是取值分隔符
    tokens, depth, cur = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("括号不匹配：多余的右括号")
        if ch == "," and depth == 0:
            tokens.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    tokens.append("".join(cur))
    if depth != 0:
        raise ValueError("括号不匹配：缺少右括号")

    vals: list = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if tok.startswith("("):
            inner = tok[1:-1] if tok.endswith(")") else None
            if inner is None:
                raise ValueError(f"等间距写法缺右括号：{tok!r}")
            parts = inner.split(",")
            if len(parts) != 3:
                raise ValueError(f"等间距写法需要 3 个数（起点,终点,间隔）：{tok!r}")
            triple = [float(p) for p in parts]
        elif ":" in tok:
            parts = tok.split(":")
            if len(parts) != 3:
                raise ValueError(f"等间距写法需要 起点:终点:间隔：{tok!r}")
            triple = [float(p) for p in parts]
        else:
            vals.append(float(tok))
            continue
        start, stop, step = triple
        if step == 0:
            raise ValueError(f"等间距的间隔不能为 0：{tok!r}")
        n = int((stop - start) / step + 1e-9) + 1
        if n < 1:
            raise ValueError(f"等间距区间为空（检查起点/终点/间隔方向）：{tok!r}")
        vals.extend(start + step * k for k in range(n))
    if not vals:
        raise ValueError("取值列表为空")
    return vals


def sweep(locator_cls, param_grid: Dict[str, Sequence],
          scenario, cache_dir: str = "results",
          locator_kwargs: Optional[dict] = None) -> pd.DataFrame:
    """Run :func:`locbench.run_trials` for the cartesian product of
    ``param_grid`` and return one table with the parameters as columns.

    Parameters
    ----------
    locator_cls :
        A locator class (e.g. :class:`aelib.GeigerLocator`); one instance
        is built per combination as
        ``locator_cls(scenario.sensors, **combo, **locator_kwargs)`` --
        so ``param_grid`` keys are constructor keyword arguments
        (``grid_size``, ``margin``, ``velocity``, ``damping``, ...).
    param_grid :
        ``{param_name: [values]}``; at least one entry.
    scenario :
        :class:`locbench.Scenario`.
    cache_dir :
        Sweep results are pickled per (locator, grid, scenario, aelib
        version) hash under this directory; a repeat call with the same
        setup returns the cached table unchanged.
    locator_kwargs :
        Fixed constructor kwargs shared by all combinations (must not
        repeat a ``param_grid`` key -- the constructor would raise).

    Returns
    -------
    pandas.DataFrame -- trial rows of all combinations with one column
    per swept parameter.
    """
    import aelib

    from .evaluate import run_trials, scenario_payload

    if not param_grid:
        raise ValueError("param_grid is empty")
    clean_cache(cache_dir)            # 先清 7 天前的旧缓存再查/写
    locator_kwargs = dict(locator_kwargs or {})
    clash = set(param_grid) & set(locator_kwargs)
    if clash:
        raise ValueError(f"param_grid and locator_kwargs overlap: {sorted(clash)}")

    payload = dict(scenario_payload(scenario), locator=locator_cls.__name__,
                   grid={k: list(v) for k, v in param_grid.items()},
                   kwargs={k: repr(v) for k, v in locator_kwargs.items()},
                   aelib=aelib.__version__)
    key = hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()[:16]
    path = os.path.join(cache_dir, f"sweep_{key}.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)

    keys = sorted(param_grid)
    frames = []
    for combo in itertools.product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        loc = locator_cls(scenario.sensors, **params, **locator_kwargs)
        df = run_trials(loc, scenario)
        for k, v in params.items():
            df[k] = v
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    os.makedirs(cache_dir, exist_ok=True)
    out.to_pickle(path)
    return out
