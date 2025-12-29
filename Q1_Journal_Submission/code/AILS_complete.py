"""
============================================================================
AILS: Adaptive Incremental Line Search
Complete Implementation for Q1 Journal Submission

Authors: Amr Elshahed
Institution: Universiti Sains Malaysia

This file contains the complete, documented implementation of the AILS
algorithm for corridor-based pathfinding optimization.
============================================================================
"""

import numpy as np
import heapq
import time
from collections import deque, defaultdict
from typing import List, Tuple, Optional, Dict, Set, Any
from dataclasses import dataclass
from enum import Enum
import warnings

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Movement directions for 8-connectivity grid
DIRECTIONS = [
    (0, 1), (0, -1), (1, 0), (-1, 0),    # Cardinal directions
    (1, 1), (1, -1), (-1, 1), (-1, -1)   # Diagonal directions
]

# Movement costs
COST_CARDINAL = 1.0
COST_DIAGONAL = 1.41421356  # sqrt(2)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class AILSConfig:
    """Configuration parameters for AILS algorithm."""
    r_min: int = 5           # Minimum corridor radius
    r_max: int = 15          # Maximum corridor radius
    alpha: float = 0.8       # Density scaling exponent
    window_size: int = 7     # Local density estimation window
    max_iterations: int = 5  # Maximum corridor expansion iterations


@dataclass
class PathResult:
    """Result of a pathfinding operation."""
    path: List[Tuple[int, int]]
    cost: float
    time_ms: float
    nodes_visited: int
    path_found: bool
    corridor_size: int = 0
    iterations: int = 1


# ============================================================================
# HEURISTIC FUNCTIONS
# ============================================================================

class Heuristics:
    """Collection of admissible heuristics for A* search."""

    @staticmethod
    def euclidean(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance (L2 norm) - admissible for 8-connectivity."""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    @staticmethod
    def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance (L1 norm) - admissible for 4-connectivity."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def chebyshev(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Chebyshev distance (L-infinity norm) - tighter for 8-connectivity."""
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    @staticmethod
    def octile(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Octile distance - optimal for 8-connectivity with diagonal costs."""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return max(dx, dy) + (COST_DIAGONAL - 1) * min(dx, dy)


# ============================================================================
# A* SEARCH ENGINE
# ============================================================================

class AStarEngine:
    """
    A* pathfinding algorithm with optional corridor constraint.

    This implementation supports both standard A* (full grid search)
    and AILS-constrained search (corridor-limited).
    """

    def __init__(self, grid: np.ndarray, heuristic=Heuristics.octile):
        """
        Initialize A* engine.

        Args:
            grid: 2D numpy array where 0=traversable, 1=obstacle
            heuristic: Heuristic function for cost estimation
        """
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.h = heuristic

    def _get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        """
        Get valid neighbors with movement costs.

        Args:
            node: Current cell coordinates (row, col)

        Returns:
            List of (neighbor, cost) tuples
        """
        neighbors = []
        r, c = node

        for i, (dr, dc) in enumerate(DIRECTIONS):
            nr, nc = r + dr, c + dc

            # Check bounds
            if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                continue

            # Check if traversable
            if self.grid[nr, nc] == 1:
                continue

            # Diagonal movement cost
            if i >= 4:  # Diagonal directions
                # Check corner-cutting (both adjacent cells must be free)
                if self.grid[r + dr, c] == 1 or self.grid[r, c + dc] == 1:
                    continue
                cost = COST_DIAGONAL
            else:
                cost = COST_CARDINAL

            neighbors.append(((nr, nc), cost))

        return neighbors

    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int],
                  corridor: Optional[Set[Tuple[int, int]]] = None) -> PathResult:
        """
        Find shortest path from start to goal.

        Args:
            start: Starting cell (row, col)
            goal: Goal cell (row, col)
            corridor: Optional set of allowed cells (AILS constraint)

        Returns:
            PathResult with path, cost, time, and metrics
        """
        start_time = time.perf_counter()

        # Validate start and goal
        if self.grid[start] == 1 or self.grid[goal] == 1:
            return PathResult(
                path=[], cost=0, time_ms=0,
                nodes_visited=0, path_found=False
            )

        # Priority queue: (f_score, g_score, counter, node)
        counter = 0  # Tie-breaker for heap stability
        open_set = [(0, 0, counter, start)]

        # Cost tracking
        g_scores: Dict[Tuple[int, int], float] = defaultdict(lambda: float('inf'))
        g_scores[start] = 0

        # Path reconstruction
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        # Closed set for visited nodes
        closed_set: Set[Tuple[int, int]] = set()
        nodes_visited = 0

        while open_set:
            _, current_g, _, current = heapq.heappop(open_set)

            # Skip if already processed with better cost
            if current in closed_set:
                continue

            closed_set.add(current)
            nodes_visited += 1

            # Goal reached
            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()

                end_time = time.perf_counter()
                return PathResult(
                    path=path,
                    cost=current_g,
                    time_ms=(end_time - start_time) * 1000,
                    nodes_visited=nodes_visited,
                    path_found=True,
                    corridor_size=len(corridor) if corridor else 0
                )

            # Expand neighbors
            for neighbor, step_cost in self._get_neighbors(current):
                # AILS corridor constraint
                if corridor is not None and neighbor not in corridor:
                    continue

                # Skip already visited
                if neighbor in closed_set:
                    continue

                tentative_g = g_scores[current] + step_cost

                if tentative_g < g_scores[neighbor]:
                    came_from[neighbor] = current
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self.h(neighbor, goal)
                    counter += 1
                    heapq.heappush(open_set, (f_score, tentative_g, counter, neighbor))

        # No path found
        end_time = time.perf_counter()
        return PathResult(
            path=[], cost=0,
            time_ms=(end_time - start_time) * 1000,
            nodes_visited=nodes_visited,
            path_found=False,
            corridor_size=len(corridor) if corridor else 0
        )


# ============================================================================
# AILS CORRIDOR BUILDER
# ============================================================================

class AILSCorridorBuilder:
    """
    Adaptive Incremental Line Search corridor generation.

    The corridor builder creates a search space constraint based on:
    1. Bresenham line from start to goal (initial corridor)
    2. Local obstacle density estimation
    3. Adaptive radius computation
    4. Incremental expansion when needed
    """

    def __init__(self, grid: np.ndarray, config: AILSConfig = None):
        """
        Initialize corridor builder.

        Args:
            grid: 2D numpy array (0=traversable, 1=obstacle)
            config: AILS configuration parameters
        """
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.config = config or AILSConfig()

    def bresenham_line(self, start: Tuple[int, int],
                       goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Generate line points using Bresenham's algorithm.

        Args:
            start: Starting point (row, col)
            goal: Goal point (row, col)

        Returns:
            List of points along the line
        """
        points = []
        r0, c0 = start
        r1, c1 = goal

        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1

        err = dr - dc
        r, c = r0, c0

        while True:
            points.append((r, c))

            if r == r1 and c == c1:
                break

            e2 = 2 * err

            if e2 > -dc:
                err -= dc
                r += sr

            if e2 < dr:
                err += dr
                c += sc

        return points

    def compute_local_density(self, point: Tuple[int, int]) -> float:
        """
        Compute obstacle density in local window around point.

        Args:
            point: Center point (row, col)

        Returns:
            Normalized density in [0, 1]
        """
        r, c = point
        w = self.config.window_size

        obstacles = 0
        total = 0

        for dr in range(-w, w + 1):
            for dc in range(-w, w + 1):
                nr, nc = r + dr, c + dc

                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    total += 1
                    if self.grid[nr, nc] == 1:
                        obstacles += 1

        return obstacles / total if total > 0 else 0

    def adaptive_radius(self, density: float) -> int:
        """
        Compute adaptive corridor radius based on local density.

        Formula: r(p) = r_min + floor((r_max - r_min) * sigma(p)^alpha)

        Args:
            density: Local obstacle density sigma(p)

        Returns:
            Radius value for corridor expansion
        """
        r_range = self.config.r_max - self.config.r_min
        scaled = density ** self.config.alpha
        return self.config.r_min + int(r_range * scaled)

    def _add_disk(self, center: Tuple[int, int], radius: int,
                  corridor: Set[Tuple[int, int]]) -> None:
        """
        Add circular disk around center point to corridor.

        Args:
            center: Center of disk (row, col)
            radius: Disk radius
            corridor: Corridor set to update (modified in-place)
        """
        cr, cc = center
        r2 = radius * radius

        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr * dr + dc * dc <= r2:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        if self.grid[nr, nc] == 0:
                            corridor.add((nr, nc))

    def build_base_corridor(self, start: Tuple[int, int],
                           goal: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """
        Build base corridor using Bresenham line with fixed radius.

        Args:
            start: Starting point
            goal: Goal point

        Returns:
            Set of corridor cells
        """
        corridor: Set[Tuple[int, int]] = set()
        line_points = self.bresenham_line(start, goal)

        for point in line_points:
            self._add_disk(point, self.config.r_min, corridor)

        # Always include start and goal
        corridor.add(start)
        corridor.add(goal)

        return corridor

    def build_adaptive_corridor(self, start: Tuple[int, int],
                                goal: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """
        Build adaptive corridor with density-based radius.

        This is the main AILS corridor generation method that adapts
        the corridor width based on local obstacle density.

        Args:
            start: Starting point
            goal: Goal point

        Returns:
            Set of corridor cells
        """
        corridor: Set[Tuple[int, int]] = set()
        line_points = self.bresenham_line(start, goal)

        for point in line_points:
            if 0 <= point[0] < self.rows and 0 <= point[1] < self.cols:
                density = self.compute_local_density(point)
                radius = self.adaptive_radius(density)
                self._add_disk(point, radius, corridor)

        # Always include start and goal
        corridor.add(start)
        corridor.add(goal)

        return corridor

    def expand_corridor(self, corridor: Set[Tuple[int, int]],
                       expansion_factor: float = 1.5) -> Set[Tuple[int, int]]:
        """
        Expand existing corridor by a factor.

        Used for incremental expansion when initial corridor
        doesn't contain a valid path.

        Args:
            corridor: Current corridor set
            expansion_factor: How much to expand

        Returns:
            Expanded corridor set
        """
        expanded = set(corridor)
        expansion_radius = int(self.config.r_min * expansion_factor)

        for point in list(corridor):
            self._add_disk(point, expansion_radius, expanded)

        return expanded


# ============================================================================
# AILS PATHFINDER (MAIN INTERFACE)
# ============================================================================

class AILSPathfinder:
    """
    Main AILS pathfinding interface.

    This class provides the complete AILS algorithm with:
    - Initial adaptive corridor generation
    - Constrained A* search
    - Incremental corridor expansion on failure
    - Fallback to unconstrained search
    """

    def __init__(self, grid: np.ndarray, config: AILSConfig = None):
        """
        Initialize AILS pathfinder.

        Args:
            grid: 2D numpy array (0=traversable, 1=obstacle)
            config: AILS configuration parameters
        """
        self.grid = grid
        self.config = config or AILSConfig()
        self.astar = AStarEngine(grid, Heuristics.octile)
        self.corridor_builder = AILSCorridorBuilder(grid, self.config)

    def find_path_standard(self, start: Tuple[int, int],
                          goal: Tuple[int, int]) -> PathResult:
        """
        Find path using standard A* (baseline comparison).

        Args:
            start: Starting cell
            goal: Goal cell

        Returns:
            PathResult from unconstrained A*
        """
        return self.astar.find_path(start, goal)

    def find_path_ails(self, start: Tuple[int, int],
                       goal: Tuple[int, int],
                       strategy: str = 'adaptive') -> PathResult:
        """
        Find path using AILS corridor-constrained search.

        Args:
            start: Starting cell
            goal: Goal cell
            strategy: 'base' for fixed radius, 'adaptive' for density-based

        Returns:
            PathResult with corridor constraint
        """
        total_start = time.perf_counter()

        # Build initial corridor
        if strategy == 'base':
            corridor = self.corridor_builder.build_base_corridor(start, goal)
        else:
            corridor = self.corridor_builder.build_adaptive_corridor(start, goal)

        iteration = 1

        while iteration <= self.config.max_iterations:
            # Try to find path within corridor
            result = self.astar.find_path(start, goal, corridor)

            if result.path_found:
                # Path found within corridor
                total_time = (time.perf_counter() - total_start) * 1000
                return PathResult(
                    path=result.path,
                    cost=result.cost,
                    time_ms=total_time,
                    nodes_visited=result.nodes_visited,
                    path_found=True,
                    corridor_size=len(corridor),
                    iterations=iteration
                )

            # Expand corridor and retry
            expansion_factor = 1.0 + 0.5 * iteration
            corridor = self.corridor_builder.expand_corridor(corridor, expansion_factor)
            iteration += 1

        # Fallback to unconstrained A*
        result = self.astar.find_path(start, goal)
        total_time = (time.perf_counter() - total_start) * 1000

        return PathResult(
            path=result.path,
            cost=result.cost,
            time_ms=total_time,
            nodes_visited=result.nodes_visited,
            path_found=result.path_found,
            corridor_size=0,  # Fallback used full grid
            iterations=iteration
        )

    def compare_methods(self, start: Tuple[int, int],
                       goal: Tuple[int, int]) -> Dict[str, PathResult]:
        """
        Compare standard A*, base AILS, and adaptive AILS.

        Args:
            start: Starting cell
            goal: Goal cell

        Returns:
            Dictionary with results for each method
        """
        return {
            'standard_astar': self.find_path_standard(start, goal),
            'ails_base': self.find_path_ails(start, goal, strategy='base'),
            'ails_adaptive': self.find_path_ails(start, goal, strategy='adaptive')
        }


# ============================================================================
# MAP LOADER FOR MOVING AI BENCHMARKS
# ============================================================================

class MovingAIMapLoader:
    """
    Loader for Moving AI Lab benchmark maps.

    Supports .map format with octile grid type.
    """

    @staticmethod
    def load_map(filepath: str) -> np.ndarray:
        """
        Load a Moving AI Lab .map file.

        Args:
            filepath: Path to .map file

        Returns:
            2D numpy array (0=traversable, 1=obstacle)
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Parse header
        map_type = lines[0].strip().split()[1]
        height = int(lines[1].strip().split()[1])
        width = int(lines[2].strip().split()[1])

        assert lines[3].strip() == 'map', "Expected 'map' keyword"

        # Parse grid
        grid = np.zeros((height, width), dtype=int)

        for i, line in enumerate(lines[4:4+height]):
            for j, char in enumerate(line.strip()):
                if j < width:
                    # Traversable: '.', 'G', 'S' (ground, swamp)
                    # Obstacle: 'T', '@', 'O', 'W' (tree, out of bounds, water)
                    if char in '.GS':
                        grid[i, j] = 0
                    else:
                        grid[i, j] = 1

        return grid

    @staticmethod
    def load_scenarios(filepath: str) -> List[Dict]:
        """
        Load scenario file for benchmarking.

        Args:
            filepath: Path to .scen file

        Returns:
            List of scenario dictionaries
        """
        scenarios = []

        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Skip header
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) >= 9:
                scenarios.append({
                    'bucket': int(parts[0]),
                    'map': parts[1],
                    'width': int(parts[2]),
                    'height': int(parts[3]),
                    'start_col': int(parts[4]),
                    'start_row': int(parts[5]),
                    'goal_col': int(parts[6]),
                    'goal_row': int(parts[7]),
                    'optimal_length': float(parts[8])
                })

        return scenarios


# ============================================================================
# GRID GENERATOR FOR SYNTHETIC EXPERIMENTS
# ============================================================================

class GridGenerator:
    """
    Generator for synthetic test grids.

    Supports multiple obstacle patterns for controlled experiments.
    """

    @staticmethod
    def generate_random(size: int, density: float = 0.2,
                       seed: int = None) -> np.ndarray:
        """
        Generate grid with random obstacle distribution.

        Args:
            size: Grid dimension (size x size)
            density: Obstacle density [0, 1]
            seed: Random seed for reproducibility

        Returns:
            2D numpy array grid
        """
        if seed is not None:
            np.random.seed(seed)

        grid = (np.random.rand(size, size) < density).astype(int)

        # Ensure borders are blocked
        grid[0, :] = grid[-1, :] = 1
        grid[:, 0] = grid[:, -1] = 1

        return grid

    @staticmethod
    def generate_clustered(size: int, density: float = 0.3,
                          num_clusters: int = 10,
                          seed: int = None) -> np.ndarray:
        """
        Generate grid with clustered obstacles.

        Args:
            size: Grid dimension
            density: Target obstacle density
            num_clusters: Number of obstacle clusters
            seed: Random seed

        Returns:
            2D numpy array grid
        """
        if seed is not None:
            np.random.seed(seed)

        grid = np.zeros((size, size), dtype=int)
        target_obstacles = int(size * size * density)

        # Generate cluster centers
        centers = [(np.random.randint(size), np.random.randint(size))
                   for _ in range(num_clusters)]

        placed = 0
        while placed < target_obstacles:
            # Pick random cluster center
            cr, cc = centers[np.random.randint(len(centers))]

            # Add obstacle near center (Gaussian spread)
            dr = int(np.random.randn() * size * 0.1)
            dc = int(np.random.randn() * size * 0.1)
            r, c = cr + dr, cc + dc

            if 0 < r < size - 1 and 0 < c < size - 1:
                if grid[r, c] == 0:
                    grid[r, c] = 1
                    placed += 1

        # Ensure borders
        grid[0, :] = grid[-1, :] = 1
        grid[:, 0] = grid[:, -1] = 1

        return grid

    @staticmethod
    def generate_maze(size: int, seed: int = None) -> np.ndarray:
        """
        Generate maze using recursive backtracking.

        Args:
            size: Grid dimension (should be odd for proper maze)
            seed: Random seed

        Returns:
            2D numpy array maze grid
        """
        if seed is not None:
            np.random.seed(seed)

        # Ensure odd size for maze generation
        if size % 2 == 0:
            size += 1

        # Start with all walls
        grid = np.ones((size, size), dtype=int)

        def carve(r, c):
            grid[r, c] = 0
            directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
            np.random.shuffle(directions)

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 < nr < size - 1 and 0 < nc < size - 1:
                    if grid[nr, nc] == 1:
                        grid[r + dr // 2, c + dc // 2] = 0
                        carve(nr, nc)

        # Start from center
        carve(1, 1)

        return grid


# ============================================================================
# EXPERIMENT UTILITIES
# ============================================================================

def run_benchmark(grid: np.ndarray, num_pairs: int = 100,
                  seed: int = 42) -> Dict[str, List[PathResult]]:
    """
    Run benchmark comparing A* and AILS on random pairs.

    Args:
        grid: Test grid
        num_pairs: Number of start-goal pairs
        seed: Random seed

    Returns:
        Dictionary with results for each method
    """
    np.random.seed(seed)

    # Find all traversable cells
    traversable = np.argwhere(grid == 0)
    if len(traversable) < 2:
        raise ValueError("Grid has insufficient traversable cells")

    # Generate random pairs
    pairs = []
    for _ in range(num_pairs):
        idx = np.random.choice(len(traversable), 2, replace=False)
        start = tuple(traversable[idx[0]])
        goal = tuple(traversable[idx[1]])
        pairs.append((start, goal))

    # Initialize pathfinder
    pathfinder = AILSPathfinder(grid)

    results = {
        'standard_astar': [],
        'ails_base': [],
        'ails_adaptive': []
    }

    for start, goal in pairs:
        comparison = pathfinder.compare_methods(start, goal)
        for method, result in comparison.items():
            results[method].append(result)

    return results


def compute_statistics(results: Dict[str, List[PathResult]]) -> Dict[str, Dict]:
    """
    Compute aggregate statistics from benchmark results.

    Args:
        results: Dictionary of method -> list of PathResults

    Returns:
        Statistics for each method
    """
    stats = {}

    for method, result_list in results.items():
        times = [r.time_ms for r in result_list if r.path_found]
        nodes = [r.nodes_visited for r in result_list if r.path_found]
        costs = [r.cost for r in result_list if r.path_found]
        success_rate = sum(1 for r in result_list if r.path_found) / len(result_list)

        if times:
            stats[method] = {
                'time_mean': np.mean(times),
                'time_std': np.std(times),
                'nodes_mean': np.mean(nodes),
                'nodes_std': np.std(nodes),
                'cost_mean': np.mean(costs),
                'success_rate': success_rate * 100,
                'num_samples': len(times)
            }
        else:
            stats[method] = {
                'time_mean': 0, 'time_std': 0,
                'nodes_mean': 0, 'nodes_std': 0,
                'cost_mean': 0, 'success_rate': 0,
                'num_samples': 0
            }

    return stats


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AILS: Adaptive Incremental Line Search - Demonstration")
    print("=" * 70)

    # Generate test grid
    print("\n1. Generating 200x200 test grid with 25% obstacle density...")
    grid = GridGenerator.generate_random(200, density=0.25, seed=42)
    print(f"   Grid size: {grid.shape}")
    print(f"   Traversable cells: {np.sum(grid == 0)}")
    print(f"   Obstacle cells: {np.sum(grid == 1)}")

    # Single path comparison
    print("\n2. Single path comparison...")
    pathfinder = AILSPathfinder(grid)

    # Find valid start and goal
    traversable = np.argwhere(grid == 0)
    np.random.seed(123)
    idx = np.random.choice(len(traversable), 2, replace=False)
    start = tuple(traversable[idx[0]])
    goal = tuple(traversable[idx[1]])

    print(f"   Start: {start}")
    print(f"   Goal: {goal}")

    results = pathfinder.compare_methods(start, goal)

    print("\n   Results:")
    print("   " + "-" * 60)
    print(f"   {'Method':<20} {'Time (ms)':<12} {'Nodes':<10} {'Path Found'}")
    print("   " + "-" * 60)

    for method, result in results.items():
        print(f"   {method:<20} {result.time_ms:<12.3f} {result.nodes_visited:<10} {result.path_found}")

    # Benchmark
    print("\n3. Running benchmark (100 random pairs)...")
    benchmark_results = run_benchmark(grid, num_pairs=100, seed=42)
    stats = compute_statistics(benchmark_results)

    print("\n   Benchmark Statistics:")
    print("   " + "-" * 70)
    print(f"   {'Method':<20} {'Time Mean':<12} {'Time Std':<10} {'Nodes Mean':<12} {'Success %'}")
    print("   " + "-" * 70)

    for method, s in stats.items():
        print(f"   {method:<20} {s['time_mean']:<12.3f} {s['time_std']:<10.3f} "
              f"{s['nodes_mean']:<12.1f} {s['success_rate']:.1f}%")

    # Node reduction calculation
    if stats['standard_astar']['nodes_mean'] > 0:
        base_reduction = (1 - stats['ails_base']['nodes_mean'] /
                         stats['standard_astar']['nodes_mean']) * 100
        adaptive_reduction = (1 - stats['ails_adaptive']['nodes_mean'] /
                             stats['standard_astar']['nodes_mean']) * 100

        print("\n   Node Reduction vs Standard A*:")
        print(f"   - AILS Base: {base_reduction:.1f}%")
        print(f"   - AILS Adaptive: {adaptive_reduction:.1f}%")

    print("\n" + "=" * 70)
    print("Demonstration complete.")
    print("=" * 70)
