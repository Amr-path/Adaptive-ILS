"""
Paper Experiment Configurations
================================

Pre-defined configurations for Q1 paper experiments.
These configurations are designed to generate comprehensive
results suitable for publication.
"""

from benchmarks.performance import BenchmarkConfig
from benchmarks.scalability import ScalabilityConfig
from benchmarks.sensitivity import SensitivityConfig
from benchmarks.patterns import PatternConfig
from benchmarks.comparative import ComparativeConfig


# =============================================================================
# FULL PAPER CONFIGURATIONS
# =============================================================================

FULL_PERFORMANCE_CONFIG = BenchmarkConfig(
    grid_sizes=[
        (50, 50), (100, 100), (150, 150), (200, 200),
        (250, 250), (300, 300), (400, 400), (500, 500)
    ],
    densities=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    patterns=["random", "maze", "clustered", "open", "mixed", "room", "corridor"],
    trials_per_config=100,
    algorithms=["AILS", "A*", "Dijkstra", "BFS", "JPS", "Bidirectional-A*"],
    seed=42,
    output_dir="results/performance_full",
)

FULL_SCALABILITY_CONFIG = ScalabilityConfig(
    grid_sizes=[
        (50, 50), (75, 75), (100, 100), (125, 125), (150, 150),
        (175, 175), (200, 200), (250, 250), (300, 300), (350, 350),
        (400, 400), (500, 500), (600, 600), (750, 750), (1000, 1000)
    ],
    density=0.20,
    pattern="random",
    trials_per_size=100,
    algorithms=["AILS", "A*", "Dijkstra", "BFS", "JPS", "Bidirectional-A*"],
    track_memory=True,
    seed=42,
    output_dir="results/scalability_full",
)

FULL_SENSITIVITY_CONFIG = SensitivityConfig(
    r_min_values=[1, 2, 3, 4, 5, 6, 8, 10],
    r_max_values=[8, 10, 12, 15, 18, 20, 25, 30, 40],
    alpha_values=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0],
    beta_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0],
    window_size_values=[2, 3, 5, 7, 10, 12, 15, 20],
    strategies=["base", "density_adaptive", "gradient_based"],
    grid_sizes=[(100, 100), (200, 200), (300, 300)],
    densities=[0.15, 0.25, 0.35],
    patterns=["random", "clustered", "mixed"],
    trials_per_config=50,
    seed=42,
    output_dir="results/sensitivity_full",
)

FULL_PATTERN_CONFIG = PatternConfig(
    patterns=["random", "maze", "clustered", "open", "mixed", "room", "corridor"],
    grid_sizes=[(100, 100), (200, 200), (300, 300), (400, 400)],
    densities=[0.10, 0.20, 0.30, 0.40],
    algorithms=["AILS", "A*", "Dijkstra", "BFS", "JPS", "Bidirectional-A*"],
    trials_per_config=100,
    seed=42,
    output_dir="results/patterns_full",
)

FULL_COMPARATIVE_CONFIG = ComparativeConfig(
    synthetic_sizes=[
        (50, 50), (100, 100), (150, 150), (200, 200),
        (250, 250), (300, 300), (400, 400), (500, 500)
    ],
    synthetic_densities=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    synthetic_patterns=[
        "random", "maze", "clustered", "open", "mixed", "room", "corridor"
    ],
    algorithms=["AILS", "A*", "Dijkstra", "BFS", "JPS", "Bidirectional-A*"],
    ails_strategies=["base", "density_adaptive", "gradient_based"],
    trials_per_synthetic=100,
    use_benchmark_maps=True,
    seed=42,
    output_dir="results/comparative_full",
)


# =============================================================================
# QUICK TEST CONFIGURATIONS
# =============================================================================

QUICK_PERFORMANCE_CONFIG = BenchmarkConfig(
    grid_sizes=[(100, 100), (200, 200)],
    densities=[0.2, 0.3],
    patterns=["random", "clustered", "maze"],
    trials_per_config=20,
    algorithms=["AILS", "A*", "Dijkstra"],
    seed=42,
    output_dir="results/performance_quick",
)

QUICK_SCALABILITY_CONFIG = ScalabilityConfig(
    grid_sizes=[(50, 50), (100, 100), (200, 200), (300, 300)],
    trials_per_size=20,
    algorithms=["AILS", "A*", "Dijkstra"],
    track_memory=False,
    seed=42,
    output_dir="results/scalability_quick",
)

QUICK_SENSITIVITY_CONFIG = SensitivityConfig(
    r_min_values=[2, 5],
    r_max_values=[10, 20],
    alpha_values=[0.5, 1.0, 1.5],
    beta_values=[0.0, 0.3],
    window_size_values=[3, 7],
    strategies=["density_adaptive", "gradient_based"],
    grid_sizes=[(100, 100)],
    densities=[0.2],
    patterns=["random"],
    trials_per_config=15,
    seed=42,
    output_dir="results/sensitivity_quick",
)


# =============================================================================
# SPECIALIZED CONFIGURATIONS
# =============================================================================

# Focused on large grids for robotics applications
ROBOTICS_CONFIG = BenchmarkConfig(
    grid_sizes=[(500, 500), (750, 750), (1000, 1000), (1500, 1500)],
    densities=[0.15, 0.25, 0.35],
    patterns=["room", "corridor", "mixed"],
    trials_per_config=50,
    algorithms=["AILS", "A*", "JPS"],
    seed=42,
    output_dir="results/robotics",
)

# Focused on maze-like environments
MAZE_FOCUSED_CONFIG = BenchmarkConfig(
    grid_sizes=[(100, 100), (200, 200), (300, 300), (400, 400)],
    densities=[0.25, 0.30, 0.35, 0.40],
    patterns=["maze", "corridor", "room"],
    trials_per_config=100,
    algorithms=["AILS", "A*", "Dijkstra", "BFS", "JPS"],
    seed=42,
    output_dir="results/maze_focused",
)

# Strategy comparison only
STRATEGY_COMPARISON_CONFIG = ComparativeConfig(
    synthetic_sizes=[(200, 200), (300, 300)],
    synthetic_densities=[0.15, 0.25, 0.35],
    synthetic_patterns=["random", "clustered", "maze", "mixed"],
    algorithms=["AILS"],
    ails_strategies=["base", "density_adaptive", "gradient_based"],
    trials_per_synthetic=100,
    use_benchmark_maps=False,
    seed=42,
    output_dir="results/strategy_comparison",
)


# =============================================================================
# CONFIGURATION GETTER
# =============================================================================

CONFIGS = {
    'full_performance': FULL_PERFORMANCE_CONFIG,
    'full_scalability': FULL_SCALABILITY_CONFIG,
    'full_sensitivity': FULL_SENSITIVITY_CONFIG,
    'full_patterns': FULL_PATTERN_CONFIG,
    'full_comparative': FULL_COMPARATIVE_CONFIG,
    'quick_performance': QUICK_PERFORMANCE_CONFIG,
    'quick_scalability': QUICK_SCALABILITY_CONFIG,
    'quick_sensitivity': QUICK_SENSITIVITY_CONFIG,
    'robotics': ROBOTICS_CONFIG,
    'maze_focused': MAZE_FOCUSED_CONFIG,
    'strategy_comparison': STRATEGY_COMPARISON_CONFIG,
}


def get_config(name: str):
    """
    Get a configuration by name.

    Args:
        name: Configuration name

    Returns:
        Configuration object

    Raises:
        ValueError: If configuration not found
    """
    if name not in CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name]


def list_configs():
    """List available configurations."""
    return list(CONFIGS.keys())
