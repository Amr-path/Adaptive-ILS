"""
Large-Scale Grid Benchmark
============================

Benchmark AILS and comparison algorithms on large grid maps
(1000x1000, 5000x5000, 10000x10000) to evaluate scalability behavior
at scales beyond the standard experimental range.

Design considerations for large grids:
- Memory tracking is critical (10000x10000 = 100M cells)
- Per-algorithm timeout to avoid hanging on slow algorithms at large scales
- Reduced trial counts for the largest grids to keep runtime practical
- AILS r_max is scaled proportionally to grid size
- Results are exported in formats suitable for paper updates
"""

import time
import json
import csv
import signal
import tracemalloc
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.data_structures import Grid, Point, PathResult
from core.ails_algorithm import AILS, AILSConfig
from algorithms.astar import AStar
from algorithms.dijkstra import Dijkstra
from algorithms.bfs import BFS
from algorithms.jps import JumpPointSearch
from algorithms.bidirectional import BidirectionalAStar
from maps.generators import MapGenerator


class TimeoutError(Exception):
    """Raised when an algorithm exceeds its time budget."""
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Algorithm exceeded time budget")


@dataclass
class LargeScaleResult:
    """Result from a single large-scale benchmark trial."""
    # Identification
    algorithm: str
    grid_width: int
    grid_height: int
    grid_cells: int
    density: float
    pattern: str
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
    memory_peak_mb: float

    # Normalized metrics
    time_per_million_cells_ms: float
    nodes_per_cell: float
    search_space_reduction_pct: float = 0.0

    # AILS-specific
    corridor_size: int = 0
    corridor_efficiency: float = 0.0
    strategy_used: str = ""
    expansions: int = 0

    # Status
    timed_out: bool = False
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LargeScaleConfig:
    """Configuration for large-scale grid experiments."""
    # Grid sizes to test
    grid_sizes: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1000, 1000),
        (5000, 5000),
        (10000, 10000),
    ])

    # Obstacle densities
    densities: List[float] = field(default_factory=lambda: [0.2, 0.3])

    # Map patterns to test
    patterns: List[str] = field(default_factory=lambda: [
        "random", "clustered", "mixed"
    ])

    # Algorithms to benchmark
    algorithms: List[str] = field(default_factory=lambda: [
        "AILS", "A*", "JPS"
    ])

    # Trials per grid size (automatically scaled down for larger grids)
    trials_by_size: Dict[int, int] = field(default_factory=lambda: {
        1000: 20,
        5000: 10,
        10000: 5,
    })

    # Per-trial timeout in seconds (per algorithm per trial).
    # With real preemptive signal.alarm timeouts these are now strictly
    # enforced, so use tighter values that still give A* a fair chance.
    timeout_per_trial: Dict[int, int] = field(default_factory=lambda: {
        1000: 30,
        5000: 120,
        10000: 180,
    })

    # AILS configuration scaling by grid size
    ails_r_max_by_size: Dict[int, int] = field(default_factory=lambda: {
        1000: 20,
        5000: 30,
        10000: 40,
    })

    # Minimum path distance as fraction of diagonal
    min_distance_fraction: float = 0.3

    # Random seed
    seed: int = 42

    # Track memory usage
    track_memory: bool = True

    # Output directory
    output_dir: str = "results/large_scale_grids"

    def get_trials(self, grid_width: int) -> int:
        """Get trial count for a given grid width."""
        return self.trials_by_size.get(grid_width, 10)

    def get_timeout(self, grid_width: int) -> int:
        """Get per-trial timeout for a given grid width."""
        return self.timeout_per_trial.get(grid_width, 300)

    def get_ails_r_max(self, grid_width: int) -> int:
        """Get AILS r_max for a given grid width."""
        return self.ails_r_max_by_size.get(grid_width, 15)


class LargeScaleBenchmark:
    """
    Large-scale grid benchmark for evaluating AILS at 1000-10000 grid sizes.

    This benchmark is specifically designed to answer:
    1. How does AILS execution time scale at 1K, 5K, 10K grid sizes?
    2. How does corridor efficiency change at large scales?
    3. What is the memory footprint at large scales?
    4. How does the AILS advantage over A*/JPS change with scale?
    5. At what scale does AILS become impractical (if any)?
    """

    def __init__(self, config: Optional[LargeScaleConfig] = None):
        self.config = config or LargeScaleConfig()
        self.results: List[LargeScaleResult] = []
        self.map_generator = MapGenerator(seed=self.config.seed)
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def run(self, verbose: bool = True) -> List[LargeScaleResult]:
        """Run the full large-scale benchmark."""
        self.results = []
        start_time = time.time()

        total_configs = (
            len(self.config.grid_sizes) *
            len(self.config.densities) *
            len(self.config.patterns)
        )
        config_num = 0

        if verbose:
            print("=" * 70)
            print("LARGE-SCALE GRID BENCHMARK")
            print("=" * 70)
            print(f"Grid sizes: {[f'{w}x{h}' for w, h in self.config.grid_sizes]}")
            print(f"Densities: {self.config.densities}")
            print(f"Patterns: {self.config.patterns}")
            print(f"Algorithms: {self.config.algorithms}")
            print(f"Seed: {self.config.seed}")
            print()

        for width, height in self.config.grid_sizes:
            for density in self.config.densities:
                for pattern in self.config.patterns:
                    config_num += 1
                    n_trials = self.config.get_trials(width)
                    timeout = self.config.get_timeout(width)

                    if verbose:
                        print(f"\n[{config_num}/{total_configs}] "
                              f"{width}x{height} ({width*height:,} cells) | "
                              f"density={density:.0%} | pattern={pattern} | "
                              f"trials={n_trials} | timeout={timeout}s")

                    # Generate grid
                    if verbose:
                        print(f"  Generating {pattern} grid...", end=" ", flush=True)
                    gen_start = time.time()
                    grid = self.map_generator.generate(
                        width, height, pattern, density
                    )
                    gen_elapsed = time.time() - gen_start
                    if verbose:
                        print(f"done ({gen_elapsed:.1f}s)")

                    # Generate test cases
                    min_dist = int(
                        self.config.min_distance_fraction *
                        np.sqrt(width**2 + height**2)
                    )
                    if verbose:
                        print(f"  Generating {n_trials} test cases "
                              f"(min_dist={min_dist})...", end=" ", flush=True)
                    test_cases = self.map_generator.generate_valid_endpoints(
                        grid, min_dist, n_trials
                    )
                    if verbose:
                        print(f"got {len(test_cases)}")

                    if len(test_cases) == 0:
                        if verbose:
                            print("  WARNING: No valid test cases generated, skipping.")
                        continue

                    # Run each algorithm
                    # NOTE: We no longer pre-run A* to collect "optimal costs"
                    # because that block had no timeout and would hang for minutes
                    # on 5 000×5 000+ grids. The reduction metric is computed
                    # directly from visited-node counts instead.
                    optimal_costs: dict = {}
                    for algo_name in self.config.algorithms:
                        if verbose:
                            print(f"  {algo_name}...", end=" ", flush=True)

                        algo_results = self._run_algorithm(
                            grid, algo_name, test_cases, optimal_costs,
                            width, height, density, pattern, timeout
                        )
                        self.results.extend(algo_results)

                        # Print summary
                        if verbose:
                            successful = [r for r in algo_results if not r.timed_out and r.path_found]
                            timed_out = sum(1 for r in algo_results if r.timed_out)
                            if successful:
                                avg_time = np.mean([r.execution_time_ms for r in successful])
                                avg_nodes = np.mean([r.visited_nodes for r in successful])
                                avg_mem = np.mean([r.memory_peak_mb for r in successful])
                                print(f"avg={avg_time:.1f}ms | "
                                      f"nodes={avg_nodes:,.0f} | "
                                      f"mem={avg_mem:.1f}MB", end="")
                                if algo_name == "AILS":
                                    avg_eff = np.mean([r.corridor_efficiency for r in successful])
                                    print(f" | corr_eff={avg_eff:.1%}", end="")
                                if timed_out > 0:
                                    print(f" | timeouts={timed_out}", end="")
                                print()
                            else:
                                print(f"ALL TIMED OUT ({timed_out})")

        elapsed = time.time() - start_time
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"Benchmark complete: {len(self.results)} results "
                  f"in {elapsed/60:.1f} minutes")
            print(f"{'=' * 70}")

        return self.results

    def _run_algorithm(
        self, grid: Grid, algo_name: str,
        test_cases: List[Tuple[Point, Point]],
        optimal_costs: Dict,
        width: int, height: int,
        density: float, pattern: str,
        timeout: int
    ) -> List[LargeScaleResult]:
        """Run an algorithm on all test cases with preemptive timeout.

        The timeout is enforced via signal.SIGALRM (Unix/macOS) so the
        algorithm is actually interrupted rather than checked post-hoc.
        Memory tracking is disabled for grids >= 5 000 × 5 000 because
        tracemalloc imposes a 2–5× slowdown on large heaps.
        """
        results = []
        grid_cells = width * height

        # Disable tracemalloc for very large grids to avoid 2-5x overhead
        use_memory = self.config.track_memory and (grid_cells < 5_000_000)

        # Register preemptive timeout handler (Unix/macOS only)
        _sigalrm_available = hasattr(signal, 'SIGALRM')
        if _sigalrm_available:
            signal.signal(signal.SIGALRM, _timeout_handler)

        # Create algorithm instance
        algo = self._create_algorithm(grid, algo_name, width)
        if algo is None:
            return results

        for trial, (start, goal) in enumerate(test_cases):
            manhattan_dist = abs(start.x - goal.x) + abs(start.y - goal.y)
            timed_out = False
            error_msg = ""
            memory_peak = 0.0

            try:
                if use_memory:
                    tracemalloc.start()

                # Arm the preemptive timer
                if _sigalrm_available:
                    signal.alarm(timeout)

                trial_start = time.perf_counter()
                result = algo.find_path(start, goal)
                trial_elapsed = time.perf_counter() - trial_start

                # Disarm the timer immediately after the call returns
                if _sigalrm_available:
                    signal.alarm(0)

                if use_memory:
                    _, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    memory_peak = peak / (1024 * 1024)

                # Compute normalised metrics
                time_per_m = (result.execution_time_ms / grid_cells) * 1_000_000
                nodes_per_cell = result.visited_nodes / grid_cells
                # Search-space reduction relative to the full grid
                reduction = (1 - result.visited_nodes / grid_cells) * 100 if result.found else 0.0

                lr = LargeScaleResult(
                    algorithm=algo_name,
                    grid_width=width,
                    grid_height=height,
                    grid_cells=grid_cells,
                    density=density,
                    pattern=pattern,
                    trial=trial,
                    start=(start.x, start.y),
                    goal=(goal.x, goal.y),
                    manhattan_distance=manhattan_dist,
                    path_found=result.found,
                    path_cost=result.cost if result.found else float('inf'),
                    path_length=result.path_length(),
                    visited_nodes=result.visited_nodes,
                    execution_time_ms=result.execution_time_ms,
                    memory_peak_mb=memory_peak,
                    time_per_million_cells_ms=time_per_m,
                    nodes_per_cell=nodes_per_cell,
                    search_space_reduction_pct=reduction,
                    corridor_size=result.corridor_size,
                    corridor_efficiency=result.corridor_efficiency,
                    strategy_used=result.strategy_used,
                    expansions=result.expansions,
                    timed_out=timed_out,
                    error=error_msg,
                )
                results.append(lr)

            except (TimeoutError, Exception) as e:
                # Cancel the alarm so it doesn't fire on the next trial
                if _sigalrm_available:
                    signal.alarm(0)

                if use_memory:
                    try:
                        tracemalloc.stop()
                    except RuntimeError:
                        pass

                is_timeout = isinstance(e, TimeoutError)
                results.append(LargeScaleResult(
                    algorithm=algo_name,
                    grid_width=width,
                    grid_height=height,
                    grid_cells=grid_cells,
                    density=density,
                    pattern=pattern,
                    trial=trial,
                    start=(start.x, start.y),
                    goal=(goal.x, goal.y),
                    manhattan_distance=manhattan_dist,
                    path_found=False,
                    path_cost=float('inf'),
                    path_length=0,
                    visited_nodes=0,
                    execution_time_ms=0,
                    memory_peak_mb=0,
                    time_per_million_cells_ms=0,
                    nodes_per_cell=0,
                    timed_out=True,
                    error="timeout" if is_timeout else str(e),
                ))

        return results

    def _create_algorithm(self, grid: Grid, algo_name: str, grid_width: int):
        """Create algorithm instance with appropriate configuration."""
        if algo_name == "AILS":
            r_max = self.config.get_ails_r_max(grid_width)
            config = AILSConfig(
                strategy="density_adaptive",
                r_min=2,
                r_max=r_max,
                alpha=1.0,
                beta=0.3,
                window_size=5,
            )
            return AILS(grid, config)
        elif algo_name == "A*":
            return AStar(grid)
        elif algo_name == "Dijkstra":
            return Dijkstra(grid)
        elif algo_name == "BFS":
            return BFS(grid)
        elif algo_name == "JPS":
            return JumpPointSearch(grid)
        elif algo_name == "Bidirectional-A*":
            return BidirectionalAStar(grid)
        else:
            return None

    # =========================================================================
    # Results saving
    # =========================================================================

    def save_results(self, filename: Optional[str] = None) -> Tuple[str, str]:
        """Save results to CSV and JSON."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"large_scale_benchmark_{timestamp}"

        csv_path = Path(self.config.output_dir) / f"{filename}.csv"
        json_path = Path(self.config.output_dir) / f"{filename}.json"

        # CSV
        with open(csv_path, 'w', newline='') as f:
            if self.results:
                writer = csv.DictWriter(
                    f, fieldnames=self.results[0].to_dict().keys()
                )
                writer.writeheader()
                for result in self.results:
                    writer.writerow(result.to_dict())

        # JSON
        with open(json_path, 'w') as f:
            json.dump({
                'config': asdict(self.config),
                'results': [r.to_dict() for r in self.results],
                'timestamp': datetime.now().isoformat(),
                'total_results': len(self.results),
            }, f, indent=2, default=str)

        return str(csv_path), str(json_path)

    # =========================================================================
    # Analysis methods
    # =========================================================================

    def analyze(self) -> Dict[str, Any]:
        """Generate comprehensive analysis of large-scale results."""
        return {
            'summary': self._summary(),
            'by_grid_size': self._analyze_by_grid_size(),
            'by_algorithm': self._analyze_by_algorithm(),
            'by_pattern': self._analyze_by_pattern(),
            'scaling_analysis': self._analyze_scaling(),
            'ails_corridor_analysis': self._analyze_corridors(),
            'speedup_vs_astar': self._analyze_speedup("A*"),
            'memory_analysis': self._analyze_memory(),
        }

    def _summary(self) -> Dict[str, Any]:
        successful = [r for r in self.results if not r.timed_out and r.path_found]
        return {
            'total_trials': len(self.results),
            'successful_trials': len(successful),
            'timed_out_trials': sum(1 for r in self.results if r.timed_out),
            'grid_sizes': sorted(set(f"{r.grid_width}x{r.grid_height}" for r in self.results)),
            'algorithms': sorted(set(r.algorithm for r in self.results)),
            'patterns': sorted(set(r.pattern for r in self.results)),
            'densities': sorted(set(r.density for r in self.results)),
        }

    def _analyze_by_grid_size(self) -> Dict[str, Dict[str, Any]]:
        """Analyze performance metrics grouped by grid size."""
        analysis = {}
        for result in self.results:
            size_key = f"{result.grid_width}x{result.grid_height}"
            algo = result.algorithm
            key = (size_key, algo)
            if key not in analysis:
                analysis[key] = []
            analysis[key].append(result)

        output = {}
        for (size_key, algo), results in analysis.items():
            if size_key not in output:
                output[size_key] = {}
            successful = [r for r in results if not r.timed_out and r.path_found]
            if successful:
                output[size_key][algo] = {
                    'trials': len(results),
                    'successful': len(successful),
                    'timeout_rate': (len(results) - len(successful)) / len(results),
                    'time_mean_ms': float(np.mean([r.execution_time_ms for r in successful])),
                    'time_std_ms': float(np.std([r.execution_time_ms for r in successful])),
                    'time_median_ms': float(np.median([r.execution_time_ms for r in successful])),
                    'time_p95_ms': float(np.percentile([r.execution_time_ms for r in successful], 95)),
                    'nodes_mean': float(np.mean([r.visited_nodes for r in successful])),
                    'nodes_std': float(np.std([r.visited_nodes for r in successful])),
                    'memory_mean_mb': float(np.mean([r.memory_peak_mb for r in successful])),
                    'memory_max_mb': float(np.max([r.memory_peak_mb for r in successful])),
                    'time_per_million_cells_ms': float(np.mean([r.time_per_million_cells_ms for r in successful])),
                    'nodes_per_cell': float(np.mean([r.nodes_per_cell for r in successful])),
                }
                if algo == "AILS":
                    output[size_key][algo]['corridor_efficiency_mean'] = float(
                        np.mean([r.corridor_efficiency for r in successful])
                    )
                    output[size_key][algo]['corridor_size_mean'] = float(
                        np.mean([r.corridor_size for r in successful])
                    )
            else:
                output[size_key][algo] = {
                    'trials': len(results),
                    'successful': 0,
                    'timeout_rate': 1.0,
                }

        return output

    def _analyze_by_algorithm(self) -> Dict[str, Dict[str, Any]]:
        """Overall per-algorithm statistics."""
        by_algo = {}
        for result in self.results:
            if result.algorithm not in by_algo:
                by_algo[result.algorithm] = []
            by_algo[result.algorithm].append(result)

        analysis = {}
        for algo, results in by_algo.items():
            successful = [r for r in results if not r.timed_out and r.path_found]
            if successful:
                analysis[algo] = {
                    'total_trials': len(results),
                    'successful_trials': len(successful),
                    'success_rate': len(successful) / len(results),
                    'time_mean_ms': float(np.mean([r.execution_time_ms for r in successful])),
                    'time_std_ms': float(np.std([r.execution_time_ms for r in successful])),
                    'nodes_mean': float(np.mean([r.visited_nodes for r in successful])),
                    'memory_mean_mb': float(np.mean([r.memory_peak_mb for r in successful])),
                }
        return analysis

    def _analyze_by_pattern(self) -> Dict[str, Dict[str, Any]]:
        """Performance broken down by obstacle pattern."""
        grouped = {}
        for r in self.results:
            key = (r.pattern, r.algorithm)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(r)

        analysis = {}
        for (pattern, algo), results in grouped.items():
            if pattern not in analysis:
                analysis[pattern] = {}
            successful = [r for r in results if not r.timed_out and r.path_found]
            if successful:
                analysis[pattern][algo] = {
                    'time_mean_ms': float(np.mean([r.execution_time_ms for r in successful])),
                    'nodes_mean': float(np.mean([r.visited_nodes for r in successful])),
                    'memory_mean_mb': float(np.mean([r.memory_peak_mb for r in successful])),
                }
        return analysis

    def _analyze_scaling(self) -> Dict[str, Any]:
        """Fit power-law scaling: time = a * n^b."""
        by_algo = {}
        for r in self.results:
            if r.timed_out or not r.path_found:
                continue
            if r.algorithm not in by_algo:
                by_algo[r.algorithm] = {}
            if r.grid_cells not in by_algo[r.algorithm]:
                by_algo[r.algorithm][r.grid_cells] = []
            by_algo[r.algorithm][r.grid_cells].append(r)

        analysis = {}
        for algo, by_size in by_algo.items():
            sizes = sorted(by_size.keys())
            mean_times = [np.mean([r.execution_time_ms for r in by_size[s]]) for s in sizes]
            mean_nodes = [np.mean([r.visited_nodes for r in by_size[s]]) for s in sizes]

            if len(sizes) >= 2:
                log_sizes = np.log(sizes)
                log_times = np.log(np.array(mean_times) + 1e-10)
                coeffs_time = np.polyfit(log_sizes, log_times, 1)

                log_nodes = np.log(np.array(mean_nodes) + 1e-10)
                coeffs_nodes = np.polyfit(log_sizes, log_nodes, 1)
            else:
                coeffs_time = [0, 0]
                coeffs_nodes = [0, 0]

            analysis[algo] = {
                'grid_cells': [int(s) for s in sizes],
                'mean_times_ms': [float(t) for t in mean_times],
                'mean_nodes': [float(n) for n in mean_nodes],
                'time_scaling_exponent': float(coeffs_time[0]),
                'node_scaling_exponent': float(coeffs_nodes[0]),
                'complexity_estimate': self._estimate_complexity(coeffs_time[0]),
            }

        return analysis

    def _estimate_complexity(self, exponent: float) -> str:
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

    def _analyze_corridors(self) -> Dict[str, Any]:
        """Analyze AILS corridor behavior at large scales."""
        ails_results = [
            r for r in self.results
            if r.algorithm == "AILS" and not r.timed_out and r.path_found
        ]
        if not ails_results:
            return {}

        by_size = {}
        for r in ails_results:
            size_key = f"{r.grid_width}x{r.grid_height}"
            if size_key not in by_size:
                by_size[size_key] = []
            by_size[size_key].append(r)

        analysis = {}
        for size_key, results in by_size.items():
            analysis[size_key] = {
                'corridor_efficiency_mean': float(np.mean([r.corridor_efficiency for r in results])),
                'corridor_efficiency_std': float(np.std([r.corridor_efficiency for r in results])),
                'corridor_size_mean': float(np.mean([r.corridor_size for r in results])),
                'corridor_size_std': float(np.std([r.corridor_size for r in results])),
                'corridor_to_grid_ratio': float(np.mean([
                    r.corridor_size / r.grid_cells for r in results
                ])),
                'nodes_per_corridor_cell': float(np.mean([
                    r.visited_nodes / max(r.corridor_size, 1) for r in results
                ])),
            }

        return analysis

    def _analyze_speedup(self, baseline: str = "A*") -> Dict[str, Any]:
        """Compute speedup metrics relative to a baseline algorithm."""
        by_size_algo = {}
        for r in self.results:
            if r.timed_out or not r.path_found:
                continue
            size_key = f"{r.grid_width}x{r.grid_height}"
            key = (size_key, r.algorithm)
            if key not in by_size_algo:
                by_size_algo[key] = []
            by_size_algo[key].append(r)

        speedups = {}
        for (size_key, algo), results in by_size_algo.items():
            if algo == baseline:
                continue
            baseline_key = (size_key, baseline)
            if baseline_key not in by_size_algo:
                continue

            baseline_results = by_size_algo[baseline_key]
            baseline_time = np.mean([r.execution_time_ms for r in baseline_results])
            baseline_nodes = np.mean([r.visited_nodes for r in baseline_results])
            algo_time = np.mean([r.execution_time_ms for r in results])
            algo_nodes = np.mean([r.visited_nodes for r in results])

            if size_key not in speedups:
                speedups[size_key] = {}
            speedups[size_key][algo] = {
                'time_speedup': float(baseline_time / algo_time) if algo_time > 0 else 0,
                'time_reduction_pct': float((1 - algo_time / baseline_time) * 100) if baseline_time > 0 else 0,
                'node_reduction_pct': float((1 - algo_nodes / baseline_nodes) * 100) if baseline_nodes > 0 else 0,
            }

        return speedups

    def _analyze_memory(self) -> Dict[str, Any]:
        """Analyze memory usage at each scale."""
        by_size_algo = {}
        for r in self.results:
            if r.timed_out or not r.path_found:
                continue
            size_key = f"{r.grid_width}x{r.grid_height}"
            key = (size_key, r.algorithm)
            if key not in by_size_algo:
                by_size_algo[key] = []
            by_size_algo[key].append(r)

        analysis = {}
        for (size_key, algo), results in by_size_algo.items():
            if size_key not in analysis:
                analysis[size_key] = {}
            analysis[size_key][algo] = {
                'memory_mean_mb': float(np.mean([r.memory_peak_mb for r in results])),
                'memory_max_mb': float(np.max([r.memory_peak_mb for r in results])),
                'memory_std_mb': float(np.std([r.memory_peak_mb for r in results])),
            }

        return analysis

    # =========================================================================
    # Paper-friendly output generation
    # =========================================================================

    def generate_paper_summary(self) -> str:
        """
        Generate a markdown summary of results suitable for sharing
        with an AI assistant to update the paper.
        """
        analysis = self.analyze()
        lines = []

        lines.append("# Large-Scale Grid Experiment Results")
        lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"**Seed:** {self.config.seed}")
        lines.append("")

        # Summary
        s = analysis['summary']
        lines.append("## Experiment Summary")
        lines.append(f"- Total trials: {s['total_trials']}")
        lines.append(f"- Successful trials: {s['successful_trials']}")
        lines.append(f"- Timed-out trials: {s['timed_out_trials']}")
        lines.append(f"- Grid sizes: {', '.join(s['grid_sizes'])}")
        lines.append(f"- Algorithms: {', '.join(s['algorithms'])}")
        lines.append(f"- Patterns: {', '.join(s['patterns'])}")
        lines.append(f"- Densities: {s['densities']}")
        lines.append("")

        # Performance by grid size
        lines.append("## Performance by Grid Size")
        lines.append("")
        by_size = analysis['by_grid_size']
        for size_key in sorted(by_size.keys(), key=lambda x: int(x.split('x')[0])):
            lines.append(f"### {size_key}")
            lines.append("")
            lines.append("| Algorithm | Time (ms) | Time SD | Nodes | Memory (MB) | Timeout Rate |")
            lines.append("|-----------|-----------|---------|-------|-------------|--------------|")
            for algo, data in by_size[size_key].items():
                if data.get('successful', 0) == 0:
                    lines.append(f"| {algo} | - | - | - | - | {data.get('timeout_rate', 1.0):.0%} |")
                else:
                    time_ms = data['time_mean_ms']
                    time_sd = data['time_std_ms']
                    nodes = data['nodes_mean']
                    mem = data['memory_mean_mb']
                    to_rate = data['timeout_rate']
                    extra = ""
                    if 'corridor_efficiency_mean' in data:
                        extra = f" (corr_eff={data['corridor_efficiency_mean']:.1%})"
                    lines.append(
                        f"| {algo}{extra} | {time_ms:,.1f} | {time_sd:,.1f} | "
                        f"{nodes:,.0f} | {mem:.1f} | {to_rate:.0%} |"
                    )
            lines.append("")

        # Scaling analysis
        lines.append("## Scaling Analysis (Power Law Fit: time ~ n^b)")
        lines.append("")
        lines.append("| Algorithm | Time Exponent (b) | Complexity Estimate |")
        lines.append("|-----------|-------------------|---------------------|")
        for algo, data in analysis['scaling_analysis'].items():
            lines.append(
                f"| {algo} | {data['time_scaling_exponent']:.3f} | "
                f"{data['complexity_estimate']} |"
            )
        lines.append("")

        # Speedup vs A*
        if analysis['speedup_vs_astar']:
            lines.append("## AILS Speedup vs A* by Grid Size")
            lines.append("")
            lines.append("| Grid Size | Algorithm | Speedup | Time Reduction | Node Reduction |")
            lines.append("|-----------|-----------|---------|----------------|----------------|")
            for size_key in sorted(analysis['speedup_vs_astar'].keys(),
                                   key=lambda x: int(x.split('x')[0])):
                for algo, data in analysis['speedup_vs_astar'][size_key].items():
                    lines.append(
                        f"| {size_key} | {algo} | "
                        f"{data['time_speedup']:.2f}x | "
                        f"{data['time_reduction_pct']:.1f}% | "
                        f"{data['node_reduction_pct']:.1f}% |"
                    )
            lines.append("")

        # Corridor analysis
        if analysis['ails_corridor_analysis']:
            lines.append("## AILS Corridor Behavior at Scale")
            lines.append("")
            lines.append("| Grid Size | Corridor Efficiency | Corridor Size | Corridor/Grid Ratio | Nodes/Corridor Cell |")
            lines.append("|-----------|--------------------:|---------------:|--------------------:|--------------------:|")
            for size_key in sorted(analysis['ails_corridor_analysis'].keys(),
                                   key=lambda x: int(x.split('x')[0])):
                data = analysis['ails_corridor_analysis'][size_key]
                lines.append(
                    f"| {size_key} | {data['corridor_efficiency_mean']:.1%} | "
                    f"{data['corridor_size_mean']:,.0f} | "
                    f"{data['corridor_to_grid_ratio']:.4%} | "
                    f"{data['nodes_per_corridor_cell']:.2f} |"
                )
            lines.append("")

        # Memory analysis
        if analysis['memory_analysis']:
            lines.append("## Memory Usage")
            lines.append("")
            lines.append("| Grid Size | Algorithm | Mean (MB) | Max (MB) |")
            lines.append("|-----------|-----------|-----------|----------|")
            for size_key in sorted(analysis['memory_analysis'].keys(),
                                   key=lambda x: int(x.split('x')[0])):
                for algo, data in analysis['memory_analysis'][size_key].items():
                    lines.append(
                        f"| {size_key} | {algo} | "
                        f"{data['memory_mean_mb']:.1f} | {data['memory_max_mb']:.1f} |"
                    )
            lines.append("")

        # Pattern analysis
        if analysis['by_pattern']:
            lines.append("## Performance by Obstacle Pattern")
            lines.append("")
            for pattern, algos in sorted(analysis['by_pattern'].items()):
                lines.append(f"### {pattern.capitalize()}")
                lines.append("")
                lines.append("| Algorithm | Time (ms) | Nodes | Memory (MB) |")
                lines.append("|-----------|-----------|-------|-------------|")
                for algo, data in algos.items():
                    lines.append(
                        f"| {algo} | {data['time_mean_ms']:,.1f} | "
                        f"{data['nodes_mean']:,.0f} | {data['memory_mean_mb']:.1f} |"
                    )
                lines.append("")

        # Raw data for paper tables
        lines.append("## Raw Data for Paper Tables")
        lines.append("")
        lines.append("The complete raw results are saved in:")
        lines.append(f"- CSV: `{self.config.output_dir}/<timestamp>.csv`")
        lines.append(f"- JSON: `{self.config.output_dir}/<timestamp>.json`")
        lines.append("")
        lines.append("Load the CSV in pandas for further analysis:")
        lines.append("```python")
        lines.append("import pandas as pd")
        lines.append("df = pd.read_csv('results/large_scale_grids/<filename>.csv')")
        lines.append("```")

        return "\n".join(lines)

    def save_paper_summary(self, filename: Optional[str] = None) -> str:
        """Save paper-friendly markdown summary."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"paper_summary_{timestamp}.md"

        path = Path(self.config.output_dir) / filename
        content = self.generate_paper_summary()
        with open(path, 'w') as f:
            f.write(content)

        return str(path)

    def save_analysis_json(self, filename: Optional[str] = None) -> str:
        """Save structured analysis JSON for programmatic use."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_{timestamp}.json"

        path = Path(self.config.output_dir) / filename
        analysis = self.analyze()
        with open(path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)

        return str(path)


# =============================================================================
# Convenience functions
# =============================================================================

def run_large_scale_benchmark(
    seed: int = 42,
    quick: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run large-scale benchmark with sensible defaults.

    Args:
        seed: Random seed for reproducibility
        quick: If True, run reduced configuration (1000 only, fewer trials)
        verbose: Print progress

    Returns:
        Analysis results dictionary
    """
    if quick:
        config = LargeScaleConfig(
            grid_sizes=[(1000, 1000)],
            densities=[0.2],
            patterns=["random"],
            algorithms=["AILS", "A*", "JPS"],
            trials_by_size={1000: 10},
            seed=seed,
        )
    else:
        config = LargeScaleConfig(seed=seed)

    benchmark = LargeScaleBenchmark(config)
    benchmark.run(verbose=verbose)

    # Save everything
    csv_path, json_path = benchmark.save_results()
    summary_path = benchmark.save_paper_summary()
    analysis_path = benchmark.save_analysis_json()

    if verbose:
        print(f"\nResults saved:")
        print(f"  CSV:      {csv_path}")
        print(f"  JSON:     {json_path}")
        print(f"  Summary:  {summary_path}")
        print(f"  Analysis: {analysis_path}")

    return benchmark.analyze()


if __name__ == "__main__":
    analysis = run_large_scale_benchmark(quick=True)
    print("\n=== Quick Large-Scale Analysis ===")
    for algo, data in analysis.get('scaling_analysis', {}).items():
        print(f"{algo}: {data['complexity_estimate']}")
