"""Locating trials on synthetic scenarios -> one metric table row per trial."""
from __future__ import annotations

import time
import warnings
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd

__all__ = ["Scenario", "run_trials", "summarize", "TRIAL_COLS"]

#: columns produced by :func:`run_trials` that are NOT sweep parameters
#: (used by plots to tell swept params from trial metrics)
TRIAL_COLS = {"source", "repeat", "n_picks", "n_used", "converged", "err",
              "t0_err", "rms", "n_iter", "stderr_pos", "velocity",
              "runtime_ms", "failure", "coords"} | {f"d{a}" for a in "xyz"}


@dataclass
class Scenario:
    """A synthetic test scenario (all values SI).

    One trial = one source x one noise realisation: exact arrivals from
    ``sensors``/``source``/``velocity``, degraded by ``sigma`` /
    ``outlier_p`` / ``drop_p`` and located.  ``repeats`` noise seeds per
    source give error statistics.
    """
    sensors: dict
    sources: np.ndarray            # (n, dim) known truth
    velocity: float                # true velocity used to synthesize
    sigma: float = 0.0             # pick noise, s
    outlier_p: float = 0.0         # gross-error fraction
    outlier_jitter: float = 50e-6  # gross-error magnitude, s
    drop_p: float = 0.0            # dropped-channel fraction
    seed: int = 0
    repeats: int = 1


def run_trials(locator, scenario: Scenario) -> pd.DataFrame:
    """Locate every (source, repeat) of ``scenario``; one row per trial.

    The locator is a ready-built instance (e.g.
    ``GeigerLocator(sensors, velocity=...)``); it is reused across trials
    (locate is stateless).  Per-row metrics: position error, per-axis
    errors, origin-time error, weighted rms, iterations, convergence,
    formal-error norm, runtime.  A locator exception becomes a failed row
    (``err`` NaN, ``failure`` carries the message), never an abort.
    """
    from .synth import make_arrivals
    rng = np.random.default_rng(scenario.seed)
    rows = []
    for i, src in enumerate(np.atleast_2d(scenario.sources)):
        for rep in range(scenario.repeats):
            t0_true = float(rng.uniform(0.0, 1.0))
            arrivals, _ = make_arrivals(
                scenario.sensors, src, scenario.velocity, t0=t0_true,
                sigma=scenario.sigma, outlier_p=scenario.outlier_p,
                outlier_jitter=scenario.outlier_jitter,
                drop_p=scenario.drop_p, rng=rng)
            row = dict(source=i, repeat=rep, n_picks=len(arrivals))
            t_start = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = locator.locate(arrivals, true_coords=src)
                off = res.extra.get("true_offset") or [np.nan] * len(src)
                row.update(
                    n_used=res.n_used, converged=bool(res.converged),
                    err=float(res.extra.get("true_error", np.nan)),
                    t0_err=float(abs(res.origin_time - t0_true)),
                    rms=float(res.rms), n_iter=int(res.n_iter),
                    stderr_pos=(float(np.linalg.norm(res.stderr[:len(res.coords)]))
                                if res.stderr is not None else np.nan),
                    velocity=float(res.velocity), failure=None,
                    coords=np.asarray(res.coords, dtype=float))
                for k, ax in enumerate("xyz"[:len(off)]):
                    row[f"d{ax}"] = float(off[k])
            except Exception as exc:                    # keep the sweep alive
                row.update(n_used=len(arrivals), converged=False, err=np.nan,
                           t0_err=np.nan, rms=np.nan, n_iter=0,
                           stderr_pos=np.nan, velocity=np.nan,
                           failure=str(exc), coords=None)
            row["runtime_ms"] = (time.perf_counter() - t_start) * 1e3
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.Series:
    """Headline statistics of a trial table."""
    ok = df["err"].notna()
    return pd.Series({
        "trials": len(df),
        "failures": int((~ok).sum()),
        "converged_rate": float(df["converged"].mean()),
        "err_median": df["err"].median(),
        "err_mean": df["err"].mean(),
        "err_p95": df["err"].quantile(0.95),
        "t0_err_median": df["t0_err"].median(),
        "rms_median": df["rms"].median(),
        "iter_mean": df["n_iter"].mean(),
        "runtime_ms_mean": df["runtime_ms"].mean(),
        # median |true error| / formal error: ~1 means the covariance is
        # honest, <1 pessimistic, >1 optimistic (trust with care)
        "stderr_calibration": (float((df["err"] / df["stderr_pos"]).median())
                               if df["stderr_pos"].notna().any() else np.nan),
    })


def scenario_payload(scenario: Scenario) -> dict:
    """Hashable-friendly snapshot of a scenario (for sweep cache keys)."""
    payload = asdict(scenario)
    payload["sensors"] = {str(k): [float(v) for v in p]
                          for k, p in scenario.sensors.items()}
    payload["sources"] = np.asarray(scenario.sources,
                                    dtype=float).round(12).tolist()
    return payload
