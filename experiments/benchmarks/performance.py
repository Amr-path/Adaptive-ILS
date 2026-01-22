"""
Performance Benchmark Experiments
==================================

Comprehensive performance evaluation comparing AILS with baseline algorithms.

Metrics:
- Execution time (ms)
- Visited nodes (count)
- Path cost (weighted distance)
- Optimality rate (%)
- Corridor efficiency (%)
"""

import time
import json
import csv
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from datetime import datetime
import sys
sys.path.insert(0, '..')
from core.data_structures import Grid, Point, PathResult
from core.ails_algorithm import AILS, AILSConfig
from algorithms.astar import AStar
from algorithms.dijkstra import Dijkstra
from algorithms.bfs import BFS
from algorithms.jps import JumpPointSearch
from algorithms.bidirectional import BidirectionalAStar
from maps.generators import MapGenerator


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    algorithm: str
    map_name: str
    map_size: str
    pattern: str
    density: float
    trial: int
    start: Tuple[int, int]
    goal: Tuple[int, int]
    path_found: bool
    path_cost: float
    path_length: int
    visited_nodes: int
    execution_time_ms: float
    corridor_size: int = 0
    corridor_efficiency: float = 0.0
    expansions: int = 0
    strategy_used: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark experiments."""
    # Grid sizes to test: (width, height)
    grid_sizes: List[Tuple[int, int]] = field(default_factory=lambda: [
        (50, 50), (100, 100), (200, 200), (300, 300), (400, 400), (500, 500)
    ])

    # Obstacle densities to test
    densities: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4])

    # Patterns to test
    patterns: List[str] = field(default_factory=lambda: [
        "random", "maze", "clustered", "open", "mixed", "room"
    ])

    # Number of trials per configuration
    trials_per_config: int = 100

    # Random seed for reproducibility
    seed: int = 42

    # Algorithms to benchmark
    algorithms: List[str] = field(default_factory=lambda: [
        "AILS", "A*", "Dijkstra", "BFS", "JPS", "Bidirectional-A*"
    ])

    # AILS configurations to test
    ails_configs: List[Dict] = field(default_factory=lambda: [
        {"strategy": "density_adaptive", "r_min": 2, "r_max": 15},
    ])

    # Minimum distance between start and goal (fraction of grid diagonal)
    min_distance_fraction: float = 0.3

    # Output directory
    output_dir: str = "results/performance"


class PerformanceBenchmark:
    """
    Performance benchmark for pathfinding algorithms.

    Runs comprehensive experiments comparing AILS with baseline algorithms
    across various grid configurations.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """
        Initialize benchmark.

        Args:
            config: Benchmark configuration
        """
        self.config = config or BenchmarkConfig()
        self.results: List[BenchmarkResult] = []
        self.map_generator = MapGenerator(seed=self.config.seed)

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def run(self, verbose: bool = True) -> List[BenchmarkResult]:
        """
        Run the complete performance benchmark.

        Args:
            verbose: Print progress messages

        Returns:
            List of all benchmark results
        """
        self.results = []
        total_configs = (
            len(self.config.grid_sizes) *
            len(self.config.densities) *
            len(self.config.patterns)
        )

        config_num = 0
        start_time = time.time()

        for width, height in self.config.grid_sizes:
            for density in self.config.densities:
                for pattern in self.config.patterns:
                    config_num += 1

                    if verbose:
                        print(f"\n[{config_num}/{total_configs}] "
                              f"Grid: {width}x{height}, "
                              f"Density: {density:.0%}, "
                              f"Pattern: {pattern}")

                    # Generate map
                    grid = self.map_generator.generate(
                        width, height, pattern, density
                    )
                    map_name = f"{pattern}_{width}x{height}_{int(density*100)}pct"

                    # Generate test cases
                    min_dist = int(self.config.min_distance_fraction *
                                   np.sqrt(width**2 + height**2))
                    test_cases = self.map_generator.generate_valid_endpoints(
                        grid, min_dist, self.config.trials_per_config
                    )

                    if verbose:
                        print(f"  Generated {len(test_cases)} test cases")

                    # Run algorithms
                    for algo_name in self.config.algorithms:
                        if verbose:
                            print(f"  Running {algo_name}...", end=" ", flush=True)

                        algo_results = self._run_algorithm(
                            grid, algo_name, test_cases,
                            map_name, f"{width}x{height}", pattern, density
                        )
                        self.results.extend(algo_results)

                        if verbose:
                            avg_time = np.mean([r.execution_time_ms for r in algo_results])
                            avg_nodes = np.mean([r.visited_nodes for r in algo_results])
                            print(f"Avg time: {avg_time:.2f}ms, Avg nodes: {avg_nodes:.0f}")

        elapsed = time.time() - start_time
        if verbose:
            print(f"\nBenchmark completed in {elapsed:.1f}s")
            print(f"Total results: {len(self.results)}")

        return self.results

    def _run_algorithm(self, grid: Grid, algo_name: str,
                       test_cases: List[Tuple[Point, Point]],
                       map_name: str, map_size: str,
                       pattern: str, density: float) -> List[BenchmarkResult]:
        """Run a single algorithm on all test cases."""
        results = []

        # Create algorithm instance
        if algo_name == "AILS":
            algo = AILS(grid, AILSConfig(**self.config.ails_configs[0]))
        elif algo_name == "A*":
            algo = AStar(grid)
        elif algo_name == "Dijkstra":
            algo = Dijkstra(grid)
        elif algo_name == "BFS":
            algo = BFS(grid)
        elif algo_name == "JPS":
            algo = JumpPointSearch(grid)
        elif algo_name == "Bidirectional-A*":
            algo = BidirectionalAStar(grid)
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")

        for trial, (start, goal) in enumerate(test_cases):
            result = algo.find_path(start, goal)

            benchmark_result = BenchmarkResult(
                algorithm=algo_name,
                map_name=map_name,
                map_size=map_size,
                pattern=pattern,
                density=density,
                trial=trial,
                start=(start.x, start.y),
                goal=(goal.x, goal.y),
                path_found=result.found,
                path_cost=result.cost if result.found else float('inf'),
                path_length=result.path_length(),
                visited_nodes=result.visited_nodes,
                execution_time_ms=result.execution_time_ms,
                corridor_size=result.corridor_size,
                corridor_efficiency=result.corridor_efficiency,
                expansions=result.expansions,
                strategy_used=result.strategy_used,
            )
            results.append(benchmark_result)

        return results

    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Save results to CSV and JSON files.

        Args:
            filename: Base filename (without extension)

        Returns:
            Path to saved CSV file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_benchmark_{timestamp}"

        csv_path = Path(self.config.output_dir) / f"{filename}.csv"
        json_path = Path(self.config.output_dir) / f"{filename}.json"

        # Save CSV
        with open(csv_path, 'w', newline='') as f:
            if self.results:
                writer = csv.DictWriter(f, fieldnames=self.results[0].to_dict().keys())
                writer.writeheader()
                for result in self.results:
                    writer.writerow(result.to_dict())

        # Save JSON
        with open(json_path, 'w') as f:
            json.dump({
                'config': asdict(self.config),
                'results': [r.to_dict() for r in self.results],
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2, default=str)

        return str(csv_path)

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Compute summary statistics from results.

        Returns:
            Dictionary with summary statistics
        """
        if not self.results:
            return {}

        # Group by algorithm
        by_algorithm = {}
        for result in self.results:
            if result.algorithm not in by_algorithm:
                by_algorithm[result.algorithm] = []
            by_algorithm[result.algorithm].append(result)

        summary = {}
        for algo, results in by_algorithm.items():
            times = [r.execution_time_ms for r in results]
            nodes = [r.visited_nodes for r in results]
            costs = [r.path_cost for r in results if r.path_found]
            success_rate = sum(1 for r in results if r.path_found) / len(results)

            summary[algo] = {
                'count': len(results),
                'success_rate': success_rate,
                'time_mean': np.mean(times),
                'time_std': np.std(times),
                'time_median': np.median(times),
                'time_min': np.min(times),
                'time_max': np.max(times),
                'nodes_mean': np.mean(nodes),
                'nodes_std': np.std(nodes),
                'nodes_median': np.median(nodes),
                'cost_mean': np.mean(costs) if costs else 0,
                'cost_std': np.std(costs) if costs else 0,
            }

            # AILS-specific metrics
            if algo == "AILS":
                corridors = [r.corridor_efficiency for r in results]
                summary[algo]['corridor_efficiency_mean'] = np.mean(corridors)
                summary[algo]['corridor_efficiency_std'] = np.std(corridors)

        return summary

    def compute_speedup(self, baseline: str = "A*") -> Dict[str, Dict[str, float]]:
        """
        Compute speedup relative to baseline algorithm.

        Args:
            baseline: Baseline algorithm name

        Returns:
            Dictionary with speedup metrics for each algorithm
        """
        summary = self.get_summary_statistics()
        if baseline not in summary:
            return {}

        baseline_time = summary[baseline]['time_mean']
        baseline_nodes = summary[baseline]['nodes_mean']

        speedups = {}
        for algo, stats in summary.items():
            if algo == baseline:
                continue

            speedups[algo] = {
                'time_speedup': baseline_time / stats['time_mean'] if stats['time_mean'] > 0 else 0,
                'time_reduction_pct': (1 - stats['time_mean'] / baseline_time) * 100 if baseline_time > 0 else 0,
                'node_reduction_pct': (1 - stats['nodes_mean'] / baseline_nodes) * 100 if baseline_nodes > 0 else 0,
            }

        return speedups


def run_quick_benchmark(seed: int = 42) -> Dict[str, Any]:
    """
    Run a quick benchmark with reduced parameters.

    Useful for testing and quick evaluations.

    Args:
        seed: Random seed

    Returns:
        Summary statistics
    """
    config = BenchmarkConfig(
        grid_sizes=[(100, 100), (200, 200)],
        densities=[0.2, 0.3],
        patterns=["random", "clustered"],
        trials_per_config=20,
        seed=seed,
    )

    benchmark = PerformanceBenchmark(config)
    benchmark.run(verbose=True)
    benchmark.save_results("quick_benchmark")

    return benchmark.get_summary_statistics()


if __name__ == "__main__":
    # Run quick benchmark when executed directly
    summary = run_quick_benchmark()
    print("\n=== Summary Statistics ===")
    for algo, stats in summary.items():
        print(f"\n{algo}:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
