"""
Large-Scale Grid Experiments
==============================

Evaluate AILS algorithm behavior on large grid maps (1000x1000, 5000x5000, 10000x10000)
to assess scalability beyond the standard experimental range.
"""

from .large_scale_benchmark import (
    LargeScaleBenchmark,
    LargeScaleConfig,
    LargeScaleResult,
)
