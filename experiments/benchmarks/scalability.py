"""
Scalability Benchmark Experiments
==================================

Analyze how AILS performance scales with grid size and complexity.

Key Questions:
1. How does execution time scale with grid size?
2. How does corridor efficiency change with scale?
3. What is the memory footprint at different scales?
4. How does the improvement over baselines change with scale?
"""

import time
import json
import csv
import tracemalloc
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
class ScalabilityResult:
    """Result of a scalability benchmark run."""
    algorithm: str
    grid_width: int
    grid_height: int
    grid_cells: int
    density: float
    pattern: str
    trial: int
    path_found: bool
    path_cost: float
    visited_nodes: int
    execution_time_ms: float
    memory_peak_mb: float
    corridor_size: int = 0
    corridor_efficiency: float = 0.0
    time_per_cell_us: float = 0.0  # microseconds per grid cell
    nodes_per_cell: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScalabilityConfig:
    """Configuration for scalability experiments."""
    # Grid sizes to test - extended range for scalability analysis
    grid_sizes: List[Tuple[int, int]] = field(default_factory=lambda: [
        (50, 50), (100, 100), (150, 150), (200, 200), (250, 250),
        (300, 300), (400, 400), (500, 500), (600, 600), (750, 750),
        (1000, 1000)
    ])

    # Fixed density for scalability analysis
    density: float = 0.2

    # Pattern to use
    pattern: str = "random"

    # Number of trials per size
    trials_per_size: int = 50

    # Random seed
    seed: int = 42

    # Algorithms to benchmark
    algorithms: List[str] = field(default_factory=lambda: [
        "AILS", "A*", "Dijkstra", "BFS", "JPS"
    ])

    # Minimum path distance fraction
    min_distance_fraction: float = 0.4

    # Track memory usage
    track_memory: bool = True

    # Output directory
    output_dir: str = "results/scalability"


class ScalabilityBenchmark:
    """
    Scalability analysis benchmark.

    Measures how algorithm performance changes with grid size.
    """

    def __init__(self, config: Optional[ScalabilityConfig] = None):
        """
        Initialize benchmark.

        Args:
            config: Benchmark configuration
        """
        self.config = config or ScalabilityConfig()
        self.results: List[ScalabilityResult] = []
        self.map_generator = MapGenerator(seed=self.config.seed)

        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def run(self, verbose: bool = True) -> List[ScalabilityResult]:
        """
        Run scalability benchmark.

        Args:
            verbose: Print progress

        Returns:
            List of results
        """
        self.results = []
        start_time = time.time()

        for size_idx, (width, height) in enumerate(self.config.grid_sizes):
            if verbose:
                print(f"\n[{size_idx + 1}/{len(self.config.grid_sizes)}] "
                      f"Grid size: {width}x{height} ({width * height:,} cells)")

            # Generate map
            grid = self.map_generator.generate(
                width, height, self.config.pattern, self.config.density
            )

            # Generate test cases
            min_dist = int(self.config.min_distance_fraction *
                           np.sqrt(width**2 + height**2))
            test_cases = self.map_generator.generate_valid_endpoints(
                grid, min_dist, self.config.trials_per_size
            )

            if verbose:
                print(f"  Generated {len(test_cases)} test cases")

            for algo_name in self.config.algorithms:
                if verbose:
                    print(f"  Running {algo_name}...", end=" ", flush=True)

                algo_results = self._run_algorithm(
                    grid, algo_name, test_cases, width, height
                )
                self.results.extend(algo_results)

                if verbose:
                    avg_time = np.mean([r.execution_time_ms for r in algo_results])
                    avg_time_per_cell = np.mean([r.time_per_cell_us for r in algo_results])
                    print(f"Avg: {avg_time:.2f}ms ({avg_time_per_cell:.4f}us/cell)")

        elapsed = time.time() - start_time
        if verbose:
            print(f"\nScalability benchmark completed in {elapsed:.1f}s")

        return self.results

    def _run_algorithm(self, grid: Grid, algo_name: str,
                       test_cases: List[Tuple[Point, Point]],
                       width: int, height: int) -> List[ScalabilityResult]:
        """Run algorithm on test cases with optional memory tracking."""
        results = []
        grid_cells = width * height

        # Create algorithm instance
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
            # Track memory if enabled
            if self.config.track_memory:
                tracemalloc.start()

            result = algo.find_path(start, goal)

            memory_peak = 0.0
            if self.config.track_memory:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                memory_peak = peak / (1024 * 1024)  # Convert to MB

            # Compute normalized metrics
            time_per_cell = (result.execution_time_ms * 1000) / grid_cells  # us
            nodes_per_cell = result.visited_nodes / grid_cells

            scalability_result = ScalabilityResult(
                algorithm=algo_name,
                grid_width=width,
                grid_height=height,
                grid_cells=grid_cells,
                density=self.config.density,
                pattern=self.config.pattern,
                trial=trial,
                path_found=result.found,
                path_cost=result.cost if result.found else float('inf'),
                visited_nodes=result.visited_nodes,
                execution_time_ms=result.execution_time_ms,
                memory_peak_mb=memory_peak,
                corridor_size=result.corridor_size,
                corridor_efficiency=result.corridor_efficiency,
                time_per_cell_us=time_per_cell,
                nodes_per_cell=nodes_per_cell,
            )
            results.append(scalability_result)

        return results

    def save_results(self, filename: Optional[str] = None) -> str:
        """Save results to files."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scalability_benchmark_{timestamp}"

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

    def analyze_scaling(self) -> Dict[str, Any]:
        """
        Analyze scaling behavior of each algorithm.

        Fits power law: time = a * n^b where n is grid size.

        Returns:
            Scaling analysis results
        """
        analysis = {}

        # Group by algorithm
        by_algorithm = {}
        for result in self.results:
            if result.algorithm not in by_algorithm:
                by_algorithm[result.algorithm] = {}
            size = result.grid_cells
            if size not in by_algorithm[result.algorithm]:
                by_algorithm[result.algorithm][size] = []
            by_algorithm[result.algorithm][size].append(result)

        for algo, by_size in by_algorithm.items():
            sizes = sorted(by_size.keys())
            mean_times = []
            mean_nodes = []
            mean_efficiency = []

            for size in sizes:
                results = by_size[size]
                mean_times.append(np.mean([r.execution_time_ms for r in results]))
                mean_nodes.append(np.mean([r.visited_nodes for r in results]))
                if algo == "AILS":
                    mean_efficiency.append(np.mean([r.corridor_efficiency for r in results]))

            # Fit power law for time: log(time) = log(a) + b*log(n)
            if len(sizes) > 2:
                log_sizes = np.log(sizes)
                log_times = np.log(mean_times)
                coeffs = np.polyfit(log_sizes, log_times, 1)
                time_exponent = coeffs[0]
                time_constant = np.exp(coeffs[1])

                log_nodes = np.log(mean_nodes)
                node_coeffs = np.polyfit(log_sizes, log_nodes, 1)
                node_exponent = node_coeffs[0]
            else:
                time_exponent = 0
                time_constant = 0
                node_exponent = 0

            analysis[algo] = {
                'sizes': sizes,
                'mean_times': mean_times,
                'mean_nodes': mean_nodes,
                'time_scaling_exponent': time_exponent,
                'time_scaling_constant': time_constant,
                'node_scaling_exponent': node_exponent,
                'complexity_estimate': self._estimate_complexity(time_exponent),
            }

            if algo == "AILS":
                analysis[algo]['mean_corridor_efficiency'] = mean_efficiency

        return analysis

    def _estimate_complexity(self, exponent: float) -> str:
        """Estimate complexity class from scaling exponent."""
        if exponent < 0.6:
            return "O(n^0.5) - sublinear"
        elif exponent < 1.1:
            return "O(n) - linear"
        elif exponent < 1.6:
            return "O(n log n)"
        elif exponent < 2.1:
            return "O(n^2) - quadratic"
        else:
            return f"O(n^{exponent:.1f}) - superquadratic"

    def get_scaling_summary(self) -> Dict[str, Any]:
        """Get a summary of scaling behavior."""
        analysis = self.analyze_scaling()

        summary = {
            'algorithms': {},
            'ails_advantage': {},
        }

        for algo, data in analysis.items():
            summary['algorithms'][algo] = {
                'complexity': data['complexity_estimate'],
                'time_exponent': data['time_scaling_exponent'],
                'smallest_grid_time': data['mean_times'][0] if data['mean_times'] else 0,
                'largest_grid_time': data['mean_times'][-1] if data['mean_times'] else 0,
            }

        # Compute AILS advantage at each scale
        if 'AILS' in analysis and 'A*' in analysis:
            ails_data = analysis['AILS']
            astar_data = analysis['A*']

            for i, size in enumerate(ails_data['sizes']):
                if i < len(astar_data['mean_times']):
                    speedup = astar_data['mean_times'][i] / ails_data['mean_times'][i]
                    summary['ails_advantage'][size] = {
                        'speedup': speedup,
                        'time_reduction_pct': (1 - 1/speedup) * 100,
                    }

        return summary


def run_scalability_analysis(seed: int = 42, quick: bool = False) -> Dict[str, Any]:
    """
    Run scalability analysis.

    Args:
        seed: Random seed
        quick: Use reduced parameters for quick testing

    Returns:
        Scaling analysis results
    """
    if quick:
        config = ScalabilityConfig(
            grid_sizes=[(50, 50), (100, 100), (200, 200), (300, 300)],
            trials_per_size=20,
            seed=seed,
            track_memory=False,
        )
    else:
        config = ScalabilityConfig(seed=seed)

    benchmark = ScalabilityBenchmark(config)
    benchmark.run(verbose=True)
    benchmark.save_results()

    return benchmark.analyze_scaling()


if __name__ == "__main__":
    analysis = run_scalability_analysis(quick=True)
    print("\n=== Scaling Analysis ===")
    for algo, data in analysis.items():
        print(f"\n{algo}:")
        print(f"  Complexity: {data['complexity_estimate']}")
        print(f"  Time exponent: {data['time_scaling_exponent']:.3f}")
