# Visualization Module
"""
Visualization Module
=====================

Publication-quality plotting for experimental results.

Components:
- plotting: Core plotting functions
- figures: Pre-defined figure generators
"""

from .plotting import (
    PlotConfig,
    ResultsPlotter,
    plot_algorithm_comparison,
    plot_scalability,
    plot_speedup_bars,
    plot_corridor_visualization,
    plot_sensitivity_heatmap,
)

__all__ = [
    'PlotConfig',
    'ResultsPlotter',
    'plot_algorithm_comparison',
    'plot_scalability',
    'plot_speedup_bars',
    'plot_corridor_visualization',
    'plot_sensitivity_heatmap',
]
