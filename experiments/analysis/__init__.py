# Statistical Analysis Module
"""
Statistical Analysis Module
============================

Comprehensive statistical analysis for experimental results.

Components:
- statistics: Descriptive and inferential statistics
- hypothesis: Statistical hypothesis testing
- effect_size: Effect size calculations
- latex_export: LaTeX table generation
"""

from .statistics import StatisticalAnalyzer
from .hypothesis import HypothesisTester
from .effect_size import EffectSizeCalculator
from .latex_export import LaTeXExporter

__all__ = [
    'StatisticalAnalyzer',
    'HypothesisTester',
    'EffectSizeCalculator',
    'LaTeXExporter',
]
