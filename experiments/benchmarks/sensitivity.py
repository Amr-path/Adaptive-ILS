"""
Parameter Sensitivity Analysis
===============================

Analyze the impact of AILS parameters on performance.

Parameters Analyzed:
- r_min: Minimum corridor radius
- r_max: Maximum corridor radius
- alpha: Density exponent
- beta: Gradient sensitivity (for gradient-based strategy)
- window_size: Local density window size
- strategy: Corridor construction strategy
"""

import time
import json
import csv
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from datetime import datetime
from itertools import product
import sys
sys.path.insert(0, '..')
from core.data_structures import Grid, Point, PathResult
from core.ails_algorithm import AILS, AILSConfig
from maps.generators import MapGenerator


@dataclass
class SensitivityResult:
    """Result of a parameter sensitivity test."""
    # Parameter values
    r_min: int
    r_max: int
    alpha: float
    beta: float
    window_size: int
    strategy: str

    # Test configuration
    grid_size: str
    density: float
    pattern: str
    trial: int

    # Performance metrics
    path_found: bool
    path_cost: float
    visited_nodes: int
    execution_time_ms: float
    corridor_size: int
    corridor_efficiency: float
    expansions: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SensitivityConfig:
    """Configuration for sensitivity analysis."""
    # Parameter ranges to test
    r_min_values: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 8])
    r_max_values: List[int] = field(default_factory=lambda: [8, 12, 15, 20, 25, 30])
    alpha_values: List[float] = field(default_factory=lambda: [0.5, 0.75, 1.0, 1.5, 2.0])
    beta_values: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.5, 0.7])
    window_size_values: List[int] = field(default_factory=lambda: [3, 5, 7, 10, 15])
    strategies: List[str] = field(default_factory=lambda: [
        "base", "density_adaptive", "gradient_based"
    ])

    # Test grid configurations
    grid_sizes: List[Tuple[int, int]] = field(default_factory=lambda: [
        (100, 100), (200, 200), (300, 300)
    ])
    densities: List[float] = field(default_factory=lambda: [0.15, 0.25, 0.35])
    patterns: List[str] = field(default_factory=lambda: ["random", "clustered", "mixed"])

    # Number of trials per parameter combination
    trials_per_config: int = 30

    # Random seed
    seed: int = 42

    # Full factorial or sampled
    full_factorial: bool = False  # If False, test parameters individually

    # Output directory
    output_dir: str = "results/sensitivity"


class ParameterSensitivityBenchmark:
    """
    Parameter sensitivity analysis for AILS.

    Systematically evaluates the impact of each AILS parameter
    on performance metrics.
    """

    def __init__(self, config: Optional[SensitivityConfig] = None):
        """Initialize benchmark."""
        self.config = config or SensitivityConfig()
        self.results: List[SensitivityResult] = []
        self.map_generator = MapGenerator(seed=self.config.seed)

        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def run(self, verbose: bool = True) -> List[SensitivityResult]:
        """
        Run sensitivity analysis.

        Args:
            verbose: Print progress

        Returns:
            List of results
        """
        self.results = []

        if self.config.full_factorial:
            self._run_full_factorial(verbose)
        else:
            self._run_one_at_a_time(verbose)

        return self.results

    def _run_full_factorial(self, verbose: bool):
        """Run full factorial experiment (all parameter combinations)."""
        # This can be very expensive - use sparingly
        param_combinations = list(product(
            self.config.r_min_values,
            self.config.r_max_values,
            self.config.alpha_values,
            self.config.strategies,
        ))

        # Filter invalid combinations (r_min >= r_max)
        param_combinations = [
            (r_min, r_max, alpha, strategy)
            for r_min, r_max, alpha, strategy in param_combinations
            if r_min < r_max
        ]

        total = len(param_combinations) * len(self.config.grid_sizes) * \
                len(self.config.densities) * len(self.config.patterns)

        if verbose:
            print(f"Full factorial: {len(param_combinations)} parameter combinations")
            print(f"Total configurations: {total}")

        config_num = 0
        for width, height in self.config.grid_sizes:
            for density in self.config.densities:
                for pattern in self.config.patterns:
                    # Generate map and test cases
                    grid = self.map_generator.generate(width, height, pattern, density)
                    min_dist = int(0.3 * np.sqrt(width**2 + height**2))
                    test_cases = self.map_generator.generate_valid_endpoints(
                        grid, min_dist, self.config.trials_per_config
                    )

                    for r_min, r_max, alpha, strategy in param_combinations:
                        config_num += 1
                        if verbose and config_num % 100 == 0:
                            print(f"  Progress: {config_num}/{total}")

                        results = self._run_parameter_config(
                            grid, test_cases,
                            r_min=r_min, r_max=r_max, alpha=alpha,
                            beta=0.3, window_size=5, strategy=strategy,
                            width=width, height=height,
                            density=density, pattern=pattern
                        )
                        self.results.extend(results)

    def _run_one_at_a_time(self, verbose: bool):
        """
        Run one-at-a-time sensitivity analysis.

        Vary one parameter at a time while keeping others at defaults.
        """
        # Default parameter values
        defaults = {
            'r_min': 2,
            'r_max': 15,
            'alpha': 1.0,
            'beta': 0.3,
            'window_size': 5,
            'strategy': 'density_adaptive',
        }

        # Parameters to vary
        param_variations = [
            ('r_min', self.config.r_min_values),
            ('r_max', self.config.r_max_values),
            ('alpha', self.config.alpha_values),
            ('beta', self.config.beta_values),
            ('window_size', self.config.window_size_values),
            ('strategy', self.config.strategies),
        ]

        total_variations = sum(len(values) for _, values in param_variations)
        total_configs = total_variations * len(self.config.grid_sizes) * \
                       len(self.config.densities) * len(self.config.patterns)

        if verbose:
            print(f"One-at-a-time analysis: {total_variations} parameter variations")
            print(f"Total configurations: {total_configs}")

        config_num = 0
        for width, height in self.config.grid_sizes:
            for density in self.config.densities:
                for pattern in self.config.patterns:
                    if verbose:
                        print(f"\nGrid: {width}x{height}, Density: {density:.0%}, Pattern: {pattern}")

                    # Generate map and test cases
                    grid = self.map_generator.generate(width, height, pattern, density)
                    min_dist = int(0.3 * np.sqrt(width**2 + height**2))
                    test_cases = self.map_generator.generate_valid_endpoints(
                        grid, min_dist, self.config.trials_per_config
                    )

                    for param_name, param_values in param_variations:
                        if verbose:
                            print(f"  Varying {param_name}...", end=" ", flush=True)

                        for value in param_values:
                            config_num += 1

                            # Create config with one parameter varied
                            params = defaults.copy()
                            params[param_name] = value

                            # Skip invalid combinations
                            if params['r_min'] >= params['r_max']:
                                continue

                            results = self._run_parameter_config(
                                grid, test_cases,
                                width=width, height=height,
                                density=density, pattern=pattern,
                                **params
                            )
                            self.results.extend(results)

                        if verbose:
                            print(f"Done ({len(param_values)} values)")

    def _run_parameter_config(self, grid: Grid,
                              test_cases: List[Tuple[Point, Point]],
                              r_min: int, r_max: int, alpha: float,
                              beta: float, window_size: int, strategy: str,
                              width: int, height: int,
                              density: float, pattern: str) -> List[SensitivityResult]:
        """Run AILS with specific parameter configuration."""
        results = []

        # Create AILS with parameters
        config = AILSConfig(
            r_min=r_min,
            r_max=r_max,
            alpha=alpha,
            beta=beta,
            window_size=window_size,
            strategy=strategy,
        )
        ails = AILS(grid, config)

        for trial, (start, goal) in enumerate(test_cases):
            result = ails.find_path(start, goal)

            sensitivity_result = SensitivityResult(
                r_min=r_min,
                r_max=r_max,
                alpha=alpha,
                beta=beta,
                window_size=window_size,
                strategy=strategy,
                grid_size=f"{width}x{height}",
                density=density,
                pattern=pattern,
                trial=trial,
                path_found=result.found,
                path_cost=result.cost if result.found else float('inf'),
                visited_nodes=result.visited_nodes,
                execution_time_ms=result.execution_time_ms,
                corridor_size=result.corridor_size,
                corridor_efficiency=result.corridor_efficiency,
                expansions=result.expansions,
            )
            results.append(sensitivity_result)

        return results

    def save_results(self, filename: Optional[str] = None) -> str:
        """Save results to files."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sensitivity_analysis_{timestamp}"

        csv_path = Path(self.config.output_dir) / f"{filename}.csv"
        json_path = Path(self.config.output_dir) / f"{filename}.json"

        with open(csv_path, 'w', newline='') as f:
            if self.results:
                writer = csv.DictWriter(f, fieldnames=self.results[0].to_dict().keys())
                writer.writeheader()
                for result in self.results:
                    writer.writerow(result.to_dict())

        with open(json_path, 'w') as f:
            json.dump({
                'config': asdict(self.config),
                'results': [r.to_dict() for r in self.results],
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2, default=str)

        return str(csv_path)

    def analyze_parameter_effects(self) -> Dict[str, Any]:
        """
        Analyze the effect of each parameter on performance.

        Returns:
            Analysis results with effect sizes and recommendations
        """
        analysis = {}

        # Parameters to analyze
        params = ['r_min', 'r_max', 'alpha', 'beta', 'window_size', 'strategy']

        for param in params:
            param_values = set()
            value_to_results = {}

            for result in self.results:
                value = getattr(result, param)
                param_values.add(value)
                if value not in value_to_results:
                    value_to_results[value] = []
                value_to_results[value].append(result)

            # Compute statistics for each value
            value_stats = {}
            for value, results in value_to_results.items():
                times = [r.execution_time_ms for r in results]
                nodes = [r.visited_nodes for r in results]
                corridors = [r.corridor_efficiency for r in results]

                value_stats[value] = {
                    'count': len(results),
                    'time_mean': np.mean(times),
                    'time_std': np.std(times),
                    'nodes_mean': np.mean(nodes),
                    'nodes_std': np.std(nodes),
                    'corridor_efficiency_mean': np.mean(corridors),
                }

            # Determine optimal value
            best_value = min(value_stats.keys(),
                            key=lambda v: value_stats[v]['time_mean'])

            # Compute effect size (range of effect)
            time_range = (
                max(s['time_mean'] for s in value_stats.values()) -
                min(s['time_mean'] for s in value_stats.values())
            )
            avg_time = np.mean([s['time_mean'] for s in value_stats.values()])
            effect_pct = (time_range / avg_time * 100) if avg_time > 0 else 0

            analysis[param] = {
                'values_tested': sorted(list(param_values), key=str),
                'value_statistics': value_stats,
                'best_value': best_value,
                'effect_range_pct': effect_pct,
                'sensitivity': 'high' if effect_pct > 30 else 'medium' if effect_pct > 10 else 'low',
            }

        return analysis

    def get_recommendations(self) -> Dict[str, Any]:
        """
        Get parameter recommendations based on analysis.

        Returns:
            Recommended parameter values with justification
        """
        analysis = self.analyze_parameter_effects()

        recommendations = {}
        for param, data in analysis.items():
            recommendations[param] = {
                'recommended_value': data['best_value'],
                'sensitivity': data['sensitivity'],
                'justification': f"Provides best execution time. "
                                f"Parameter has {data['sensitivity']} sensitivity "
                                f"({data['effect_range_pct']:.1f}% effect range)."
            }

        return recommendations


def run_sensitivity_analysis(seed: int = 42, quick: bool = False) -> Dict[str, Any]:
    """
    Run parameter sensitivity analysis.

    Args:
        seed: Random seed
        quick: Use reduced parameters

    Returns:
        Analysis results
    """
    if quick:
        config = SensitivityConfig(
            r_min_values=[2, 5],
            r_max_values=[10, 20],
            alpha_values=[0.5, 1.0, 1.5],
            beta_values=[0.0, 0.3],
            window_size_values=[3, 7],
            grid_sizes=[(100, 100)],
            densities=[0.2],
            patterns=["random"],
            trials_per_config=15,
            seed=seed,
        )
    else:
        config = SensitivityConfig(seed=seed)

    benchmark = ParameterSensitivityBenchmark(config)
    benchmark.run(verbose=True)
    benchmark.save_results()

    return benchmark.analyze_parameter_effects()


if __name__ == "__main__":
    analysis = run_sensitivity_analysis(quick=True)
    print("\n=== Parameter Sensitivity Analysis ===")
    for param, data in analysis.items():
        print(f"\n{param}:")
        print(f"  Best value: {data['best_value']}")
        print(f"  Sensitivity: {data['sensitivity']}")
        print(f"  Effect range: {data['effect_range_pct']:.1f}%")
