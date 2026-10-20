"""locbench -- locator testing / parameter tuning on synthetic data only."""

from .evaluate import Scenario, run_trials, summarize
from .plots import plot_err_hist, plot_heatmap, plot_layout, plot_metric_vs_param
from .sweep import sweep
from .synth import (composite_layout, corners_layout, coplanar_layout,
                    grid_sources, head_layout, make_arrivals, parse_layout,
                    random_int_layout, random_layout, random_sources)

__version__ = "0.2.25"

__all__ = [
    "Scenario", "run_trials", "summarize", "sweep",
    "corners_layout", "random_layout", "random_int_layout",
    "coplanar_layout", "head_layout", "composite_layout",
    "random_sources", "grid_sources", "parse_layout", "make_arrivals",
    "plot_layout", "plot_metric_vs_param", "plot_heatmap", "plot_err_hist",
    "__version__",
]
