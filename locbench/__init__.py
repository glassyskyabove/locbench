"""locbench -- locator testing / parameter tuning on synthetic data only."""

from .evaluate import Scenario, run_trials, summarize
from .plots import plot_err_hist, plot_heatmap, plot_layout, plot_metric_vs_param
from .sweep import sweep
from .synth import (corners_layout, coplanar_layout, grid_sources,
                    make_arrivals, random_layout, random_sources)

__version__ = "0.1.0"

__all__ = [
    "Scenario", "run_trials", "summarize", "sweep",
    "corners_layout", "random_layout", "coplanar_layout",
    "random_sources", "grid_sources", "make_arrivals",
    "plot_layout", "plot_metric_vs_param", "plot_heatmap", "plot_err_hist",
    "__version__",
]
