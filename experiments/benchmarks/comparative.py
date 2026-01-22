"""
Comparative Study
==================

Comprehensive comparison of AILS with all baseline algorithms
across all experimental dimensions.

This module combines all benchmarks for complete Q1 paper analysis.
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
from maps.parser import MovingAIParser, parse_map_file


@dataclass
class ComparativeResult:
    """Comprehensive result for comparative study."""
    # Experiment identification
    experiment_id: str
    algorithm: str
    map_type: str  # synthetic or benchmark
    map_name: str
    grid_size: str
    pattern: str
    density: float
    trial: int

    # Query details
    start: Tuple[int, int]
    goal: Tuple[int, int]
    manhattan_distance: int

    # Performance metrics
    path_found: bool
    path_cost: float
    path_length: int
    visited_nodes: int
    execution_time_ms: float

    # AILS-specific
    corridor_size: int = 0
    corridor_efficiency: float = 0.0
    strategy_used: str = ""
    expansions: int = 0

    # Optimality
    is_optimal: bool = True
    optimality_ratio: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComparativeConfig:
    """Configuration for comprehensive comparative study."""
    # Synthetic map configurations
    synthetic_sizes: List[Tuple[int, int]] = field(default_factory=lambda: [
        (50, 50), (100, 100), (200, 200), (300, 300), (400, 400), (500, 500)
    ])
    synthetic_densities: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4])
    synthetic_patterns: List[str] = field(default_factory=lambda: [
        "random", "maze", "clustered", "open", "mixed", "room"
    ])

    # Benchmark maps (Moving AI)
    benchmark_map_dir: str = ""  # Will be set if maps available
    use_benchmark_maps: bool = True

    # Algorithms
    algorithms: List[str] = field(default_factory=lambda: [
        "AILS", "A*", "Dijkstra", "BFS", "JPS", "Bidirectional-A*"
    ])

    # AILS strategies to compare
    ails_strategies: List[str] = field(default_factory=lambda: [
        "base", "density_adaptive", "gradient_based"
    ])

    # Trials
    trials_per_synthetic: int = 100
    trials_per_benchmark: int = 200

    # Reproducibility
    seed: int = 42

    # Output
    output_dir: str = "results/comparative"


class ComparativeStudy:
    """
    Comprehensive comparative study for Q1 paper.

    Combines:
    - Performance benchmarks
    - Scalability analysis
    - Pattern analysis
    - Benchmark map evaluation
    - Strategy comparison
    """

    def __init__(self, config: Optional[ComparativeConfig] = None):
        """Initialize study."""
        self.config = config or ComparativeConfig()
        self.results: List[ComparativeResult] = []
        self.map_generator = MapGenerator(seed=self.config.seed)
        self.experiment_counter = 0

        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def run_full_study(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run the complete comparative study.

        Returns:
            Complete analysis results
        """
        start_time = time.time()
        self.results = []

        if verbose:
            print("=" * 60)
            print("AILS COMPREHENSIVE COMPARATIVE STUDY")
            print("=" * 60)

        # Phase 1: Synthetic maps
        if verbose:
            print("\n--- Phase 1: Synthetic Map Experiments ---")
        self._run_synthetic_experiments(verbose)

        # Phase 2: Benchmark maps (if available)
        if self.config.benchmark_map_dir and self.config.use_benchmark_maps:
            if verbose:
                print("\n--- Phase 2: Benchmark Map Experiments ---")
            self._run_benchmark_experiments(verbose)

        # Phase 3: AILS strategy comparison
        if verbose:
            print("\n--- Phase 3: AILS Strategy Comparison ---")
        self._run_strategy_comparison(verbose)

        elapsed = time.time() - start_time
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Study completed in {elapsed/60:.1f} minutes")
            print(f"Total experiments: {len(self.results)}")

        # Save results
        self.save_results()

        # Generate analysis
        return self.generate_full_analysis()

    def _run_synthetic_experiments(self, verbose: bool):
        """Run experiments on synthetic maps."""
        total = (len(self.config.synthetic_sizes) *
                len(self.config.synthetic_densities) *
                len(self.config.synthetic_patterns))

        config_num = 0
        for width, height in self.config.synthetic_sizes:
            for density in self.config.synthetic_densities:
                for pattern in self.config.synthetic_patterns:
                    config_num += 1

                    if verbose:
                        print(f"\n[{config_num}/{total}] {pattern} "
                              f"{width}x{height} density={density:.0%}")

                    grid = self.map_generator.generate(
                        width, height, pattern, density
                    )

                    min_dist = int(0.3 * np.sqrt(width**2 + height**2))
                    test_cases = self.map_generator.generate_valid_endpoints(
                        grid, min_dist, self.config.trials_per_synthetic
                    )

                    # Get optimal costs using A*
                    optimal_costs = {}
                    astar = AStar(grid)
                    for start, goal in test_cases:
                        result = astar.find_path(start, goal)
                        if result.found:
                            optimal_costs[(start, goal)] = result.cost

                    for algo_name in self.config.algorithms:
                        if verbose:
                            print(f"  {algo_name}...", end=" ", flush=True)

                        results = self._run_algorithm_synthetic(
                            grid, algo_name, test_cases, optimal_costs,
                            f"synthetic_{pattern}_{width}x{height}_{int(density*100)}",
                            f"{width}x{height}", pattern, density
                        )
                        self.results.extend(results)

                        avg_time = np.mean([r.execution_time_ms for r in results])
                        if verbose:
                            print(f"{avg_time:.2f}ms")

    def _run_benchmark_experiments(self, verbose: bool):
        """Run experiments on Moving AI benchmark maps."""
        import os

        map_files = []
        for f in os.listdir(self.config.benchmark_map_dir):
            if f.endswith('.map'):
                map_files.append(os.path.join(self.config.benchmark_map_dir, f))

        if verbose:
            print(f"Found {len(map_files)} benchmark maps")

        for map_path in map_files:
            map_name = os.path.basename(map_path)
            if verbose:
                print(f"\nMap: {map_name}")

            try:
                grid = parse_map_file(map_path)
            except Exception as e:
                if verbose:
                    print(f"  Error loading map: {e}")
                continue

            # Generate test cases
            min_dist = int(0.3 * np.sqrt(grid.width**2 + grid.height**2))
            test_cases = self.map_generator.generate_valid_endpoints(
                grid, min_dist, self.config.trials_per_benchmark
            )

            if len(test_cases) < 10:
                if verbose:
                    print(f"  Skipping - insufficient test cases ({len(test_cases)})")
                continue

            # Get optimal costs
            optimal_costs = {}
            astar = AStar(grid)
            for start, goal in test_cases:
                result = astar.find_path(start, goal)
                if result.found:
                    optimal_costs[(start, goal)] = result.cost

            for algo_name in self.config.algorithms:
                if verbose:
                    print(f"  {algo_name}...", end=" ", flush=True)

                results = self._run_algorithm_synthetic(
                    grid, algo_name, test_cases, optimal_costs,
                    f"benchmark_{map_name}",
                    f"{grid.width}x{grid.height}",
                    "benchmark", grid.get_obstacle_density()
                )
                self.results.extend(results)

                avg_time = np.mean([r.execution_time_ms for r in results])
                if verbose:
                    print(f"{avg_time:.2f}ms")

    def _run_strategy_comparison(self, verbose: bool):
        """Compare AILS corridor strategies."""
        # Use a representative set of configurations
        test_configs = [
            (200, 200, 0.2, "random"),
            (200, 200, 0.3, "clustered"),
            (300, 300, 0.25, "mixed"),
        ]

        for width, height, density, pattern in test_configs:
            if verbose:
                print(f"\nStrategy comparison: {pattern} {width}x{height}")

            grid = self.map_generator.generate(width, height, pattern, density)
            min_dist = int(0.3 * np.sqrt(width**2 + height**2))
            test_cases = self.map_generator.generate_valid_endpoints(
                grid, min_dist, 50
            )

            for strategy in self.config.ails_strategies:
                if verbose:
                    print(f"  AILS-{strategy}...", end=" ", flush=True)

                config = AILSConfig(strategy=strategy)
                ails = AILS(grid, config)

                strategy_results = []
                for trial, (start, goal) in enumerate(test_cases):
                    result = ails.find_path(start, goal)

                    self.experiment_counter += 1
                    comp_result = ComparativeResult(
                        experiment_id=f"strategy_{self.experiment_counter}",
                        algorithm=f"AILS-{strategy}",
                        map_type="synthetic",
                        map_name=f"{pattern}_{width}x{height}_{int(density*100)}",
                        grid_size=f"{width}x{height}",
                        pattern=pattern,
                        density=density,
                        trial=trial,
                        start=(start.x, start.y),
                        goal=(goal.x, goal.y),
                        manhattan_distance=start.manhattan_distance(goal),
                        path_found=result.found,
                        path_cost=result.cost if result.found else float('inf'),
                        path_length=result.path_length(),
                        visited_nodes=result.visited_nodes,
                        execution_time_ms=result.execution_time_ms,
                        corridor_size=result.corridor_size,
                        corridor_efficiency=result.corridor_efficiency,
                        strategy_used=result.strategy_used,
                        expansions=result.expansions,
                    )
                    strategy_results.append(comp_result)
                    self.results.append(comp_result)

                avg_time = np.mean([r.execution_time_ms for r in strategy_results])
                avg_eff = np.mean([r.corridor_efficiency for r in strategy_results])
                if verbose:
                    print(f"{avg_time:.2f}ms, efficiency={avg_eff:.2%}")

    def _run_algorithm_synthetic(self, grid: Grid, algo_name: str,
                                 test_cases: List[Tuple[Point, Point]],
                                 optimal_costs: Dict,
                                 map_name: str, grid_size: str,
                                 pattern: str, density: float) -> List[ComparativeResult]:
        """Run algorithm and collect comprehensive results."""
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
        elif algo_name == "Bidirectional-A*":
            algo = BidirectionalAStar(grid)
        else:
            return results

        for trial, (start, goal) in enumerate(test_cases):
            result = algo.find_path(start, goal)

            # Check optimality
            optimal = optimal_costs.get((start, goal), result.cost)
            is_optimal = abs(result.cost - optimal) < 1e-6 if result.found else False
            opt_ratio = result.cost / optimal if optimal > 0 and result.found else float('inf')

            self.experiment_counter += 1
            comp_result = ComparativeResult(
                experiment_id=f"exp_{self.experiment_counter}",
                algorithm=algo_name,
                map_type="synthetic" if "synthetic" in map_name else "benchmark",
                map_name=map_name,
                grid_size=grid_size,
                pattern=pattern,
                density=density,
                trial=trial,
                start=(start.x, start.y),
                goal=(goal.x, goal.y),
                manhattan_distance=start.manhattan_distance(goal),
                path_found=result.found,
                path_cost=result.cost if result.found else float('inf'),
                path_length=result.path_length(),
                visited_nodes=result.visited_nodes,
                execution_time_ms=result.execution_time_ms,
                corridor_size=result.corridor_size,
                corridor_efficiency=result.corridor_efficiency,
                strategy_used=result.strategy_used,
                expansions=result.expansions,
                is_optimal=is_optimal,
                optimality_ratio=opt_ratio,
            )
            results.append(comp_result)

        return results

    def save_results(self, filename: Optional[str] = None) -> Tuple[str, str]:
        """Save all results."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comparative_study_{timestamp}"

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

        return str(csv_path), str(json_path)

    def generate_full_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive analysis for paper."""
        analysis = {
            'summary': self._generate_summary(),
            'by_algorithm': self._analyze_by_algorithm(),
            'by_pattern': self._analyze_by_pattern(),
            'by_size': self._analyze_by_size(),
            'by_density': self._analyze_by_density(),
            'ails_strategies': self._analyze_ails_strategies(),
            'speedup_analysis': self._analyze_speedups(),
            'statistical_tests': self._statistical_tests(),
        }

        # Save analysis
        analysis_path = Path(self.config.output_dir) / "analysis_summary.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)

        return analysis

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate overall summary."""
        return {
            'total_experiments': len(self.results),
            'algorithms_tested': list(set(r.algorithm for r in self.results)),
            'patterns_tested': list(set(r.pattern for r in self.results)),
            'grid_sizes': list(set(r.grid_size for r in self.results)),
        }

    def _analyze_by_algorithm(self) -> Dict[str, Any]:
        """Analyze results by algorithm."""
        by_algo = {}
        for result in self.results:
            if result.algorithm not in by_algo:
                by_algo[result.algorithm] = []
            by_algo[result.algorithm].append(result)

        analysis = {}
        for algo, results in by_algo.items():
            times = [r.execution_time_ms for r in results]
            nodes = [r.visited_nodes for r in results]
            success = [1 if r.path_found else 0 for r in results]

            analysis[algo] = {
                'count': len(results),
                'success_rate': np.mean(success),
                'time_mean': np.mean(times),
                'time_std': np.std(times),
                'time_median': np.median(times),
                'time_p95': np.percentile(times, 95),
                'nodes_mean': np.mean(nodes),
                'nodes_std': np.std(nodes),
            }

            if algo.startswith("AILS"):
                corridors = [r.corridor_efficiency for r in results]
                analysis[algo]['corridor_efficiency_mean'] = np.mean(corridors)

        return analysis

    def _analyze_by_pattern(self) -> Dict[str, Any]:
        """Analyze by obstacle pattern."""
        by_pattern = {}
        for result in self.results:
            key = (result.pattern, result.algorithm)
            if key not in by_pattern:
                by_pattern[key] = []
            by_pattern[key].append(result)

        analysis = {}
        for (pattern, algo), results in by_pattern.items():
            if pattern not in analysis:
                analysis[pattern] = {}
            analysis[pattern][algo] = {
                'time_mean': np.mean([r.execution_time_ms for r in results]),
                'nodes_mean': np.mean([r.visited_nodes for r in results]),
            }

        return analysis

    def _analyze_by_size(self) -> Dict[str, Any]:
        """Analyze by grid size."""
        by_size = {}
        for result in self.results:
            key = (result.grid_size, result.algorithm)
            if key not in by_size:
                by_size[key] = []
            by_size[key].append(result)

        analysis = {}
        for (size, algo), results in by_size.items():
            if size not in analysis:
                analysis[size] = {}
            analysis[size][algo] = {
                'time_mean': np.mean([r.execution_time_ms for r in results]),
                'nodes_mean': np.mean([r.visited_nodes for r in results]),
            }

        return analysis

    def _analyze_by_density(self) -> Dict[str, Any]:
        """Analyze by obstacle density."""
        by_density = {}
        for result in self.results:
            density_bucket = round(result.density, 1)
            key = (density_bucket, result.algorithm)
            if key not in by_density:
                by_density[key] = []
            by_density[key].append(result)

        analysis = {}
        for (density, algo), results in by_density.items():
            if density not in analysis:
                analysis[density] = {}
            analysis[density][algo] = {
                'time_mean': np.mean([r.execution_time_ms for r in results]),
                'nodes_mean': np.mean([r.visited_nodes for r in results]),
            }

        return analysis

    def _analyze_ails_strategies(self) -> Dict[str, Any]:
        """Compare AILS strategies."""
        strategies = {}
        for result in self.results:
            if result.algorithm.startswith("AILS"):
                strategy = result.algorithm.replace("AILS-", "") or "default"
                if strategy not in strategies:
                    strategies[strategy] = []
                strategies[strategy].append(result)

        analysis = {}
        for strategy, results in strategies.items():
            analysis[strategy] = {
                'count': len(results),
                'time_mean': np.mean([r.execution_time_ms for r in results]),
                'corridor_efficiency_mean': np.mean([r.corridor_efficiency for r in results]),
                'expansions_mean': np.mean([r.expansions for r in results]),
            }

        return analysis

    def _analyze_speedups(self) -> Dict[str, Any]:
        """Compute speedup metrics relative to A*."""
        algo_stats = self._analyze_by_algorithm()

        if "A*" not in algo_stats:
            return {}

        baseline_time = algo_stats["A*"]["time_mean"]
        baseline_nodes = algo_stats["A*"]["nodes_mean"]

        speedups = {}
        for algo, stats in algo_stats.items():
            if algo == "A*":
                continue
            speedups[algo] = {
                'time_speedup': baseline_time / stats['time_mean'] if stats['time_mean'] > 0 else 0,
                'time_reduction_pct': (1 - stats['time_mean'] / baseline_time) * 100,
                'node_reduction_pct': (1 - stats['nodes_mean'] / baseline_nodes) * 100,
            }

        return speedups

    def _statistical_tests(self) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        from scipy import stats

        # Compare AILS vs A*
        ails_times = [r.execution_time_ms for r in self.results if r.algorithm == "AILS"]
        astar_times = [r.execution_time_ms for r in self.results if r.algorithm == "A*"]

        if len(ails_times) > 1 and len(astar_times) > 1:
            # Paired t-test (if same configurations)
            min_len = min(len(ails_times), len(astar_times))
            t_stat, p_value = stats.ttest_rel(ails_times[:min_len], astar_times[:min_len])

            # Cohen's d effect size
            pooled_std = np.sqrt((np.var(ails_times) + np.var(astar_times)) / 2)
            cohens_d = (np.mean(astar_times) - np.mean(ails_times)) / pooled_std if pooled_std > 0 else 0

            return {
                'ails_vs_astar': {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'significant': p_value < 0.05,
                    'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small',
                }
            }

        return {}


def run_comparative_study(seed: int = 42, quick: bool = False) -> Dict[str, Any]:
    """Run comparative study."""
    if quick:
        config = ComparativeConfig(
            synthetic_sizes=[(100, 100), (200, 200)],
            synthetic_densities=[0.2, 0.3],
            synthetic_patterns=["random", "clustered"],
            algorithms=["AILS", "A*", "Dijkstra"],
            trials_per_synthetic=30,
            seed=seed,
        )
    else:
        config = ComparativeConfig(seed=seed)

    study = ComparativeStudy(config)
    return study.run_full_study(verbose=True)


if __name__ == "__main__":
    analysis = run_comparative_study(quick=True)
    print("\n=== Study Complete ===")
    print(f"Total experiments: {analysis['summary']['total_experiments']}")
