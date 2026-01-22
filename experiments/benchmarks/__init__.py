# Benchmark Experiments
"""
Benchmark Experiments Module
=============================

Comprehensive benchmark experiments for evaluating AILS performance.

Experiments:
- performance: Execution time and node comparisons
- scalability: Performance across grid sizes
- sensitivity: Parameter sensitivity analysis
- patterns: Performance across obstacle patterns
- comparative: Full algorithm comparison
"""

from .performance import PerformanceBenchmark
from .scalability import ScalabilityBenchmark
from .sensitivity import ParameterSensitivityBenchmark
from .patterns import PatternBenchmark
from .comparative import ComparativeStudy

__all__ = [
    'PerformanceBenchmark',
    'ScalabilityBenchmark',
    'ParameterSensitivityBenchmark',
    'PatternBenchmark',
    'ComparativeStudy',
]
