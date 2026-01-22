"""
Obstacle Pattern Analysis
==========================

Analyze AILS performance across different obstacle patterns.

Pattern Types:
- Random: Uniformly distributed obstacles
- Maze: Maze-like corridors
- Clustered: Grouped obstacle clusters
- Open: Sparse with large open areas
- Mixed: Combination of patterns
- Room: Room-like structures
- Corridor: Long narrow passages
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
from maps.generators import MapGenerator


@dataclass
class PatternResult:
    """Result from pattern analysis."""
    algorithm: str
    pattern: str
    density: float
    grid_size: str
    trial: int

    # Performance metrics
    path_found: bool
    path_cost: float
    visited_nodes: int
    execution_time_ms: float
    corridor_size: int = 0
    corridor_efficiency: float = 0.0
    strategy_used: str = ""

    # Pattern-specific metrics
    local_density_variance: float = 0.0
    path_complexity: float = 0.0  # Number of turns in path

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PatternConfig:
    """Configuration for pattern analysis."""
    # Patterns to analyze
    patterns: List[str] = field(default_factory=lambda: [
        "random", "maze", "clustered", "open", "mixed", "room", "corridor"
    ])

    # Grid sizes
    grid_sizes: List[Tuple[int, int]] = field(default_factory=lambda: [
        (100, 100), (200, 200), (300, 300)
    ])

    # Densities per pattern
    densities: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4])

    # Algorithms
    algorithms: List[str] = field(default_factory=lambda: [
        "AILS", "A*", "Dijkstra", "BFS", "JPS"
    ])

    # Trials per configuration
    trials_per_config: int = 50

    # Random seed
    seed: int = 42

    # Output directory
    output_dir: str = "results/patterns"


class PatternBenchmark:
    """
    Benchmark performance across different obstacle patterns.

    Analyzes how AILS adapts to different environmental structures.
    """

    def __init__(self, config: Optional[PatternConfig] = None):
        """Initialize benchmark."""
        self.config = config or PatternConfig()
        self.results: List[PatternResult] = []
        self.map_generator = MapGenerator(seed=self.config.seed)

        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def run(self, verbose: bool = True) -> List[PatternResult]:
        """Run pattern analysis."""
        self.results = []
        start_time = time.time()

        total = (len(self.config.patterns) * len(self.config.grid_sizes) *
                len(self.config.densities))
        config_num = 0

        for pattern in self.config.patterns:
            if verbose:
                print(f"\n=== Pattern: {pattern.upper()} ===")

            for width, height in self.config.grid_sizes:
                for density in self.config.densities:
                    config_num += 1
                    if verbose:
                        print(f"\n[{config_num}/{total}] "
                              f"{pattern} {width}x{height} density={density:.0%}")

                    # Generate map
                    grid = self.map_generator.generate(
                        width, height, pattern, density
                    )

                    # Compute local density variance
                    density_variance = self._compute_density_variance(grid)

                    # Generate test cases
                    min_dist = int(0.3 * np.sqrt(width**2 + height**2))
                    test_cases = self.map_generator.generate_valid_endpoints(
                        grid, min_dist, self.config.trials_per_config
                    )

                    if verbose:
                        print(f"  Density variance: {density_variance:.4f}")
                        print(f"  Test cases: {len(test_cases)}")

                    for algo_name in self.config.algorithms:
                        if verbose:
                            print(f"  Running {algo_name}...", end=" ", flush=True)

                        algo_results = self._run_algorithm(
                            grid, algo_name, test_cases,
                            pattern, density, f"{width}x{height}",
                            density_variance
                        )
                        self.results.extend(algo_results)

                        if verbose:
                            avg_time = np.mean([r.execution_time_ms for r in algo_results])
                            print(f"Avg: {avg_time:.2f}ms")

        elapsed = time.time() - start_time
        if verbose:
            print(f"\nPattern analysis completed in {elapsed:.1f}s")

        return self.results

    def _compute_density_variance(self, grid: Grid, window_size: int = 10) -> float:
        """Compute variance of local densities across the grid."""
        densities = []
        step = max(1, min(grid.width, grid.height) // 20)

        grid.compute_integral_image()

        for y in range(0, grid.height, step):
            for x in range(0, grid.width, step):
                local_density = grid.get_density_in_window(x, y, window_size)
                densities.append(local_density)

        return float(np.var(densities)) if densities else 0.0

    def _compute_path_complexity(self, path: List[Point]) -> float:
        """Compute path complexity (number of direction changes)."""
        if len(path) < 3:
            return 0.0

        turns = 0
        for i in range(1, len(path) - 1):
            dx1 = path[i].x - path[i-1].x
            dy1 = path[i].y - path[i-1].y
            dx2 = path[i+1].x - path[i].x
            dy2 = path[i+1].y - path[i].y

            if (dx1, dy1) != (dx2, dy2):
                turns += 1

        return turns / len(path)  # Normalized by path length

    def _run_algorithm(self, grid: Grid, algo_name: str,
                       test_cases: List[Tuple[Point, Point]],
                       pattern: str, density: float, grid_size: str,
                       density_variance: float) -> List[PatternResult]:
        """Run algorithm on test cases."""
        results = []

        # Create algorithm
        if algo_name == "AILS":
            algo = AILS(grid)
        elif algo_name == "A*":
            algo = AStar(grid)
        elif algo_name == "Dijkstra":
            algo = Dijkstra(grid)
        elif algo_name == "BFS":
            algo = BFS(grid)
        elif algo_name == "JPS":
            algo = JumpPointSearch(grid)
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")

        for trial, (start, goal) in enumerate(test_cases):
            result = algo.find_path(start, goal)

            path_complexity = 0.0
            if result.found and result.path:
                path_complexity = self._compute_path_complexity(result.path)

            pattern_result = PatternResult(
                algorithm=algo_name,
                pattern=pattern,
                density=density,
                grid_size=grid_size,
                trial=trial,
                path_found=result.found,
                path_cost=result.cost if result.found else float('inf'),
                visited_nodes=result.visited_nodes,
                execution_time_ms=result.execution_time_ms,
                corridor_size=result.corridor_size,
                corridor_efficiency=result.corridor_efficiency,
                strategy_used=result.strategy_used,
                local_density_variance=density_variance,
                path_complexity=path_complexity,
            )
            results.append(pattern_result)

        return results

    def save_results(self, filename: Optional[str] = None) -> str:
        """Save results."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pattern_analysis_{timestamp}"

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
            }, f, indent=2, default=str)

        return str(csv_path)

    def analyze_by_pattern(self) -> Dict[str, Any]:
        """Analyze results grouped by pattern."""
        analysis = {}

        # Group by pattern and algorithm
        for result in self.results:
            if result.pattern not in analysis:
                analysis[result.pattern] = {}
            if result.algorithm not in analysis[result.pattern]:
                analysis[result.pattern][result.algorithm] = []
            analysis[result.pattern][result.algorithm].append(result)

        # Compute statistics
        summary = {}
        for pattern, by_algo in analysis.items():
            summary[pattern] = {}
            for algo, results in by_algo.items():
                times = [r.execution_time_ms for r in results]
                nodes = [r.visited_nodes for r in results]

                summary[pattern][algo] = {
                    'count': len(results),
                    'time_mean': np.mean(times),
                    'time_std': np.std(times),
                    'nodes_mean': np.mean(nodes),
                    'nodes_std': np.std(nodes),
                }

                if algo == "AILS":
                    corridors = [r.corridor_efficiency for r in results]
                    summary[pattern][algo]['corridor_efficiency'] = np.mean(corridors)

        # Compute AILS advantage per pattern
        for pattern in summary:
            if "AILS" in summary[pattern] and "A*" in summary[pattern]:
                ails_time = summary[pattern]["AILS"]["time_mean"]
                astar_time = summary[pattern]["A*"]["time_mean"]
                speedup = astar_time / ails_time if ails_time > 0 else 0
                summary[pattern]["ails_speedup"] = speedup

        return summary

    def get_pattern_rankings(self) -> Dict[str, List[str]]:
        """
        Rank patterns by AILS performance advantage.

        Returns:
            Rankings showing which patterns AILS handles best/worst
        """
        analysis = self.analyze_by_pattern()

        # Get AILS speedup for each pattern
        speedups = []
        for pattern, data in analysis.items():
            if "ails_speedup" in data:
                speedups.append((pattern, data["ails_speedup"]))

        speedups.sort(key=lambda x: x[1], reverse=True)

        return {
            'best_patterns': [p for p, s in speedups[:3]],
            'worst_patterns': [p for p, s in speedups[-3:]],
            'rankings': speedups,
        }


def run_pattern_analysis(seed: int = 42, quick: bool = False) -> Dict[str, Any]:
    """Run pattern analysis."""
    if quick:
        config = PatternConfig(
            patterns=["random", "clustered", "maze"],
            grid_sizes=[(100, 100)],
            densities=[0.2, 0.3],
            trials_per_config=20,
            seed=seed,
        )
    else:
        config = PatternConfig(seed=seed)

    benchmark = PatternBenchmark(config)
    benchmark.run(verbose=True)
    benchmark.save_results()

    return benchmark.analyze_by_pattern()


if __name__ == "__main__":
    analysis = run_pattern_analysis(quick=True)
    print("\n=== Pattern Analysis Summary ===")
    for pattern, data in analysis.items():
        if isinstance(data, dict) and "ails_speedup" in data:
            print(f"\n{pattern}: AILS speedup = {data['ails_speedup']:.2f}x")
