"""Parameter-grid sweeps over a locator constructor, with disk caching."""
from __future__ import annotations

import hashlib
import itertools
import os
from typing import Dict, Optional, Sequence

import pandas as pd

__all__ = ["sweep"]


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
