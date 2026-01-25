"""
============================================================================
AILS: Adaptive Incremental Line Search
Complete Implementation for Analysis and Experiments

Authors: Amr Elshahed
Institution: Universiti Sains Malaysia

This module contains the complete AILS algorithm implementation with:
- Core data structures
- A* and other pathfinding algorithms
- Corridor building strategies
- Grid generators for synthetic experiments
- Benchmark utilities
- Statistical analysis functions
============================================================================
"""

import numpy as np
import heapq
import time
from collections import deque, defaultdict
from typing import List, Tuple, Optional, Dict, Set, Any, Callable
from dataclasses import dataclass, field
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
    algorithm: str = ""


@dataclass
class BenchmarkResult:
    """Result of a complete benchmark run."""
    grid_size: int
    obstacle_density: float
    pattern: str
    num_pairs: int
    results: Dict[str, List[PathResult]]
    statistics: Dict[str, Dict]


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
        """Get valid neighbors with movement costs."""
        neighbors = []
        r, c = node

        for i, (dr, dc) in enumerate(DIRECTIONS):
            nr, nc = r + dr, c + dc

            if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                continue

            if self.grid[nr, nc] == 1:
                continue

            if i >= 4:  # Diagonal directions
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

        if self.grid[start] == 1 or self.grid[goal] == 1:
            return PathResult(
                path=[], cost=0, time_ms=0,
                nodes_visited=0, path_found=False, algorithm="A*"
            )

        counter = 0
        open_set = [(0, 0, counter, start)]

        g_scores: Dict[Tuple[int, int], float] = defaultdict(lambda: float('inf'))
        g_scores[start] = 0

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        closed_set: Set[Tuple[int, int]] = set()
        nodes_visited = 0

        while open_set:
            _, current_g, _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue

            closed_set.add(current)
            nodes_visited += 1

            if current == goal:
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
                    corridor_size=len(corridor) if corridor else 0,
                    algorithm="A*"
                )

            for neighbor, step_cost in self._get_neighbors(current):
                if corridor is not None and neighbor not in corridor:
                    continue

                if neighbor in closed_set:
                    continue

                tentative_g = g_scores[current] + step_cost

                if tentative_g < g_scores[neighbor]:
                    came_from[neighbor] = current
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self.h(neighbor, goal)
                    counter += 1
                    heapq.heappush(open_set, (f_score, tentative_g, counter, neighbor))

        end_time = time.perf_counter()
        return PathResult(
            path=[], cost=0,
            time_ms=(end_time - start_time) * 1000,
            nodes_visited=nodes_visited,
            path_found=False,
            corridor_size=len(corridor) if corridor else 0,
            algorithm="A*"
        )


# ============================================================================
# DIJKSTRA'S ALGORITHM
# ============================================================================

class DijkstraEngine:
    """Dijkstra's shortest path algorithm."""

    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.rows, self.cols = grid.shape

    def _get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        neighbors = []
        r, c = node

        for i, (dr, dc) in enumerate(DIRECTIONS):
            nr, nc = r + dr, c + dc

            if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                continue

            if self.grid[nr, nc] == 1:
                continue

            if i >= 4:
                if self.grid[r + dr, c] == 1 or self.grid[r, c + dc] == 1:
                    continue
                cost = COST_DIAGONAL
            else:
                cost = COST_CARDINAL

            neighbors.append(((nr, nc), cost))

        return neighbors

    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int],
                  corridor: Optional[Set[Tuple[int, int]]] = None) -> PathResult:
        start_time = time.perf_counter()

        if self.grid[start] == 1 or self.grid[goal] == 1:
            return PathResult(
                path=[], cost=0, time_ms=0,
                nodes_visited=0, path_found=False, algorithm="Dijkstra"
            )

        counter = 0
        open_set = [(0, counter, start)]

        g_scores: Dict[Tuple[int, int], float] = defaultdict(lambda: float('inf'))
        g_scores[start] = 0

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        closed_set: Set[Tuple[int, int]] = set()
        nodes_visited = 0

        while open_set:
            current_g, _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue

            closed_set.add(current)
            nodes_visited += 1

            if current == goal:
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
                    corridor_size=len(corridor) if corridor else 0,
                    algorithm="Dijkstra"
                )

            for neighbor, step_cost in self._get_neighbors(current):
                if corridor is not None and neighbor not in corridor:
                    continue

                if neighbor in closed_set:
                    continue

                tentative_g = g_scores[current] + step_cost

                if tentative_g < g_scores[neighbor]:
                    came_from[neighbor] = current
                    g_scores[neighbor] = tentative_g
                    counter += 1
                    heapq.heappush(open_set, (tentative_g, counter, neighbor))

        end_time = time.perf_counter()
        return PathResult(
            path=[], cost=0,
            time_ms=(end_time - start_time) * 1000,
            nodes_visited=nodes_visited,
            path_found=False,
            corridor_size=len(corridor) if corridor else 0,
            algorithm="Dijkstra"
        )


# ============================================================================
# BFS ALGORITHM
# ============================================================================

class BFSEngine:
    """Breadth-First Search algorithm (unweighted shortest path)."""

    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.rows, self.cols = grid.shape

    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int],
                  corridor: Optional[Set[Tuple[int, int]]] = None) -> PathResult:
        start_time = time.perf_counter()

        if self.grid[start] == 1 or self.grid[goal] == 1:
            return PathResult(
                path=[], cost=0, time_ms=0,
                nodes_visited=0, path_found=False, algorithm="BFS"
            )

        queue = deque([start])
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        visited: Set[Tuple[int, int]] = {start}
        nodes_visited = 0

        while queue:
            current = queue.popleft()
            nodes_visited += 1

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()

                # Calculate actual path cost
                cost = 0.0
                for i in range(len(path) - 1):
                    dr = abs(path[i+1][0] - path[i][0])
                    dc = abs(path[i+1][1] - path[i][1])
                    if dr + dc == 2:
                        cost += COST_DIAGONAL
                    else:
                        cost += COST_CARDINAL

                end_time = time.perf_counter()
                return PathResult(
                    path=path,
                    cost=cost,
                    time_ms=(end_time - start_time) * 1000,
                    nodes_visited=nodes_visited,
                    path_found=True,
                    corridor_size=len(corridor) if corridor else 0,
                    algorithm="BFS"
                )

            r, c = current
            for i, (dr, dc) in enumerate(DIRECTIONS):
                nr, nc = r + dr, c + dc

                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue

                if self.grid[nr, nc] == 1:
                    continue

                neighbor = (nr, nc)

                if neighbor in visited:
                    continue

                if corridor is not None and neighbor not in corridor:
                    continue

                if i >= 4:
                    if self.grid[r + dr, c] == 1 or self.grid[r, c + dc] == 1:
                        continue

                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

        end_time = time.perf_counter()
        return PathResult(
            path=[], cost=0,
            time_ms=(end_time - start_time) * 1000,
            nodes_visited=nodes_visited,
            path_found=False,
            corridor_size=len(corridor) if corridor else 0,
            algorithm="BFS"
        )


# ============================================================================
# BIDIRECTIONAL A* ALGORITHM
# ============================================================================

class BidirectionalAStarEngine:
    """Bidirectional A* search algorithm."""

    def __init__(self, grid: np.ndarray, heuristic=Heuristics.octile):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.h = heuristic

    def _get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        neighbors = []
        r, c = node

        for i, (dr, dc) in enumerate(DIRECTIONS):
            nr, nc = r + dr, c + dc

            if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                continue

            if self.grid[nr, nc] == 1:
                continue

            if i >= 4:
                if self.grid[r + dr, c] == 1 or self.grid[r, c + dc] == 1:
                    continue
                cost = COST_DIAGONAL
            else:
                cost = COST_CARDINAL

            neighbors.append(((nr, nc), cost))

        return neighbors

    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int],
                  corridor: Optional[Set[Tuple[int, int]]] = None) -> PathResult:
        start_time = time.perf_counter()

        if self.grid[start] == 1 or self.grid[goal] == 1:
            return PathResult(
                path=[], cost=0, time_ms=0,
                nodes_visited=0, path_found=False, algorithm="Bidirectional A*"
            )

        # Forward search
        counter = 0
        open_forward = [(0, 0, counter, start)]
        g_forward: Dict[Tuple[int, int], float] = defaultdict(lambda: float('inf'))
        g_forward[start] = 0
        came_from_forward: Dict[Tuple[int, int], Tuple[int, int]] = {}
        closed_forward: Set[Tuple[int, int]] = set()

        # Backward search
        counter += 1
        open_backward = [(0, 0, counter, goal)]
        g_backward: Dict[Tuple[int, int], float] = defaultdict(lambda: float('inf'))
        g_backward[goal] = 0
        came_from_backward: Dict[Tuple[int, int], Tuple[int, int]] = {}
        closed_backward: Set[Tuple[int, int]] = set()

        best_path_cost = float('inf')
        meeting_point = None
        nodes_visited = 0

        while open_forward and open_backward:
            # Expand forward
            if open_forward:
                _, current_g, _, current = heapq.heappop(open_forward)

                if current not in closed_forward:
                    closed_forward.add(current)
                    nodes_visited += 1

                    # Check if meets backward search
                    if current in closed_backward:
                        total_cost = g_forward[current] + g_backward[current]
                        if total_cost < best_path_cost:
                            best_path_cost = total_cost
                            meeting_point = current

                    for neighbor, step_cost in self._get_neighbors(current):
                        if corridor is not None and neighbor not in corridor:
                            continue

                        if neighbor in closed_forward:
                            continue

                        tentative_g = g_forward[current] + step_cost

                        if tentative_g < g_forward[neighbor]:
                            came_from_forward[neighbor] = current
                            g_forward[neighbor] = tentative_g
                            f_score = tentative_g + self.h(neighbor, goal)
                            counter += 1
                            heapq.heappush(open_forward, (f_score, tentative_g, counter, neighbor))

            # Expand backward
            if open_backward:
                _, current_g, _, current = heapq.heappop(open_backward)

                if current not in closed_backward:
                    closed_backward.add(current)
                    nodes_visited += 1

                    # Check if meets forward search
                    if current in closed_forward:
                        total_cost = g_forward[current] + g_backward[current]
                        if total_cost < best_path_cost:
                            best_path_cost = total_cost
                            meeting_point = current

                    for neighbor, step_cost in self._get_neighbors(current):
                        if corridor is not None and neighbor not in corridor:
                            continue

                        if neighbor in closed_backward:
                            continue

                        tentative_g = g_backward[current] + step_cost

                        if tentative_g < g_backward[neighbor]:
                            came_from_backward[neighbor] = current
                            g_backward[neighbor] = tentative_g
                            f_score = tentative_g + self.h(neighbor, start)
                            counter += 1
                            heapq.heappush(open_backward, (f_score, tentative_g, counter, neighbor))

            # Early termination check
            if meeting_point is not None:
                min_f_forward = open_forward[0][0] if open_forward else float('inf')
                min_f_backward = open_backward[0][0] if open_backward else float('inf')

                if best_path_cost <= min_f_forward and best_path_cost <= min_f_backward:
                    break

        if meeting_point is not None:
            # Reconstruct path
            path_forward = [meeting_point]
            node = meeting_point
            while node in came_from_forward:
                node = came_from_forward[node]
                path_forward.append(node)
            path_forward.reverse()

            path_backward = []
            node = meeting_point
            while node in came_from_backward:
                node = came_from_backward[node]
                path_backward.append(node)

            path = path_forward + path_backward

            end_time = time.perf_counter()
            return PathResult(
                path=path,
                cost=best_path_cost,
                time_ms=(end_time - start_time) * 1000,
                nodes_visited=nodes_visited,
                path_found=True,
                corridor_size=len(corridor) if corridor else 0,
                algorithm="Bidirectional A*"
            )

        end_time = time.perf_counter()
        return PathResult(
            path=[], cost=0,
            time_ms=(end_time - start_time) * 1000,
            nodes_visited=nodes_visited,
            path_found=False,
            corridor_size=len(corridor) if corridor else 0,
            algorithm="Bidirectional A*"
        )


# ============================================================================
# AILS CORRIDOR BUILDER
# ============================================================================

class AILSCorridorBuilder:
    """Adaptive Incremental Line Search corridor generation."""

    def __init__(self, grid: np.ndarray, config: AILSConfig = None):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.config = config or AILSConfig()

    def bresenham_line(self, start: Tuple[int, int],
                       goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Generate line points using Bresenham's algorithm."""
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
        """Compute obstacle density in local window around point."""
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
        """Compute adaptive corridor radius based on local density."""
        r_range = self.config.r_max - self.config.r_min
        scaled = density ** self.config.alpha
        return self.config.r_min + int(r_range * scaled)

    def _add_disk(self, center: Tuple[int, int], radius: int,
                  corridor: Set[Tuple[int, int]]) -> None:
        """Add circular disk around center point to corridor."""
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
        """Build base corridor using Bresenham line with fixed radius."""
        corridor: Set[Tuple[int, int]] = set()
        line_points = self.bresenham_line(start, goal)

        for point in line_points:
            self._add_disk(point, self.config.r_min, corridor)

        corridor.add(start)
        corridor.add(goal)

        return corridor

    def build_adaptive_corridor(self, start: Tuple[int, int],
                                goal: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """Build adaptive corridor with density-based radius."""
        corridor: Set[Tuple[int, int]] = set()
        line_points = self.bresenham_line(start, goal)

        for point in line_points:
            if 0 <= point[0] < self.rows and 0 <= point[1] < self.cols:
                density = self.compute_local_density(point)
                radius = self.adaptive_radius(density)
                self._add_disk(point, radius, corridor)

        corridor.add(start)
        corridor.add(goal)

        return corridor

    def expand_corridor(self, corridor: Set[Tuple[int, int]],
                       expansion_factor: float = 1.5) -> Set[Tuple[int, int]]:
        """Expand existing corridor by a factor."""
        expanded = set(corridor)
        expansion_radius = int(self.config.r_min * expansion_factor)

        for point in list(corridor):
            self._add_disk(point, expansion_radius, expanded)

        return expanded


# ============================================================================
# AILS PATHFINDER (MAIN INTERFACE)
# ============================================================================

class AILSPathfinder:
    """Main AILS pathfinding interface."""

    def __init__(self, grid: np.ndarray, config: AILSConfig = None):
        self.grid = grid
        self.config = config or AILSConfig()
        self.astar = AStarEngine(grid, Heuristics.octile)
        self.dijkstra = DijkstraEngine(grid)
        self.bfs = BFSEngine(grid)
        self.bidirectional = BidirectionalAStarEngine(grid, Heuristics.octile)
        self.corridor_builder = AILSCorridorBuilder(grid, self.config)

    def find_path_standard(self, start: Tuple[int, int],
                          goal: Tuple[int, int], algorithm: str = 'astar') -> PathResult:
        """Find path using standard algorithms (baseline comparison)."""
        if algorithm == 'astar':
            return self.astar.find_path(start, goal)
        elif algorithm == 'dijkstra':
            return self.dijkstra.find_path(start, goal)
        elif algorithm == 'bfs':
            return self.bfs.find_path(start, goal)
        elif algorithm == 'bidirectional':
            return self.bidirectional.find_path(start, goal)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def find_path_ails(self, start: Tuple[int, int],
                       goal: Tuple[int, int],
                       strategy: str = 'adaptive') -> PathResult:
        """Find path using AILS corridor-constrained search."""
        total_start = time.perf_counter()

        if strategy == 'base':
            corridor = self.corridor_builder.build_base_corridor(start, goal)
        else:
            corridor = self.corridor_builder.build_adaptive_corridor(start, goal)

        iteration = 1

        while iteration <= self.config.max_iterations:
            result = self.astar.find_path(start, goal, corridor)

            if result.path_found:
                total_time = (time.perf_counter() - total_start) * 1000
                return PathResult(
                    path=result.path,
                    cost=result.cost,
                    time_ms=total_time,
                    nodes_visited=result.nodes_visited,
                    path_found=True,
                    corridor_size=len(corridor),
                    iterations=iteration,
                    algorithm=f"AILS-{strategy}"
                )

            expansion_factor = 1.0 + 0.5 * iteration
            corridor = self.corridor_builder.expand_corridor(corridor, expansion_factor)
            iteration += 1

        result = self.astar.find_path(start, goal)
        total_time = (time.perf_counter() - total_start) * 1000

        return PathResult(
            path=result.path,
            cost=result.cost,
            time_ms=total_time,
            nodes_visited=result.nodes_visited,
            path_found=result.path_found,
            corridor_size=0,
            iterations=iteration,
            algorithm=f"AILS-{strategy} (fallback)"
        )

    def compare_methods(self, start: Tuple[int, int],
                       goal: Tuple[int, int]) -> Dict[str, PathResult]:
        """Compare all methods."""
        return {
            'A*': self.find_path_standard(start, goal, 'astar'),
            'Dijkstra': self.find_path_standard(start, goal, 'dijkstra'),
            'BFS': self.find_path_standard(start, goal, 'bfs'),
            'Bidirectional A*': self.find_path_standard(start, goal, 'bidirectional'),
            'AILS-Base': self.find_path_ails(start, goal, strategy='base'),
            'AILS-Adaptive': self.find_path_ails(start, goal, strategy='adaptive')
        }


# ============================================================================
# GRID GENERATORS
# ============================================================================

class GridGenerator:
    """Generator for synthetic test grids."""

    @staticmethod
    def generate_random(size: int, density: float = 0.2,
                       seed: int = None) -> np.ndarray:
        """Generate grid with random obstacle distribution."""
        if seed is not None:
            np.random.seed(seed)

        grid = (np.random.rand(size, size) < density).astype(int)

        grid[0, :] = grid[-1, :] = 1
        grid[:, 0] = grid[:, -1] = 1

        return grid

    @staticmethod
    def generate_clustered(size: int, density: float = 0.3,
                          num_clusters: int = 10,
                          seed: int = None) -> np.ndarray:
        """Generate grid with clustered obstacles."""
        if seed is not None:
            np.random.seed(seed)

        grid = np.zeros((size, size), dtype=int)
        target_obstacles = int(size * size * density)

        centers = [(np.random.randint(size), np.random.randint(size))
                   for _ in range(num_clusters)]

        placed = 0
        while placed < target_obstacles:
            cr, cc = centers[np.random.randint(len(centers))]
            dr = int(np.random.randn() * size * 0.1)
            dc = int(np.random.randn() * size * 0.1)
            r, c = cr + dr, cc + dc

            if 0 < r < size - 1 and 0 < c < size - 1:
                if grid[r, c] == 0:
                    grid[r, c] = 1
                    placed += 1

        grid[0, :] = grid[-1, :] = 1
        grid[:, 0] = grid[:, -1] = 1

        return grid

    @staticmethod
    def generate_maze(size: int, seed: int = None) -> np.ndarray:
        """Generate maze using recursive backtracking."""
        if seed is not None:
            np.random.seed(seed)

        if size % 2 == 0:
            size += 1

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

        carve(1, 1)

        return grid

    @staticmethod
    def generate_room(size: int, num_rooms: int = 5,
                      room_size_range: Tuple[int, int] = (10, 30),
                      seed: int = None) -> np.ndarray:
        """Generate room-based map with corridors."""
        if seed is not None:
            np.random.seed(seed)

        grid = np.ones((size, size), dtype=int)
        rooms = []

        for _ in range(num_rooms * 3):
            if len(rooms) >= num_rooms:
                break

            room_w = np.random.randint(room_size_range[0], room_size_range[1])
            room_h = np.random.randint(room_size_range[0], room_size_range[1])
            x = np.random.randint(1, size - room_w - 1)
            y = np.random.randint(1, size - room_h - 1)

            overlap = False
            for rx, ry, rw, rh in rooms:
                if (x < rx + rw + 2 and x + room_w + 2 > rx and
                    y < ry + rh + 2 and y + room_h + 2 > ry):
                    overlap = True
                    break

            if not overlap:
                rooms.append((x, y, room_w, room_h))
                grid[y:y+room_h, x:x+room_w] = 0

        # Connect rooms with corridors
        for i in range(len(rooms) - 1):
            x1, y1, w1, h1 = rooms[i]
            x2, y2, w2, h2 = rooms[i + 1]

            cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
            cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2

            # Horizontal then vertical corridor
            for x in range(min(cx1, cx2), max(cx1, cx2) + 1):
                grid[cy1, x] = 0
            for y in range(min(cy1, cy2), max(cy1, cy2) + 1):
                grid[y, cx2] = 0

        return grid

    @staticmethod
    def generate_open(size: int, density: float = 0.1,
                     seed: int = None) -> np.ndarray:
        """Generate open environment with sparse obstacles."""
        return GridGenerator.generate_random(size, density, seed)


# ============================================================================
# BENCHMARK UTILITIES
# ============================================================================

def run_benchmark(grid: np.ndarray, num_pairs: int = 100,
                  seed: int = 42, algorithms: List[str] = None) -> Dict[str, List[PathResult]]:
    """Run benchmark comparing algorithms on random pairs."""
    np.random.seed(seed)

    if algorithms is None:
        algorithms = ['A*', 'Dijkstra', 'BFS', 'Bidirectional A*', 'AILS-Base', 'AILS-Adaptive']

    traversable = np.argwhere(grid == 0)
    if len(traversable) < 2:
        raise ValueError("Grid has insufficient traversable cells")

    pairs = []
    for _ in range(num_pairs):
        idx = np.random.choice(len(traversable), 2, replace=False)
        start = tuple(traversable[idx[0]])
        goal = tuple(traversable[idx[1]])
        pairs.append((start, goal))

    pathfinder = AILSPathfinder(grid)
    results = {alg: [] for alg in algorithms}

    for start, goal in pairs:
        if 'A*' in algorithms:
            results['A*'].append(pathfinder.find_path_standard(start, goal, 'astar'))
        if 'Dijkstra' in algorithms:
            results['Dijkstra'].append(pathfinder.find_path_standard(start, goal, 'dijkstra'))
        if 'BFS' in algorithms:
            results['BFS'].append(pathfinder.find_path_standard(start, goal, 'bfs'))
        if 'Bidirectional A*' in algorithms:
            results['Bidirectional A*'].append(pathfinder.find_path_standard(start, goal, 'bidirectional'))
        if 'AILS-Base' in algorithms:
            results['AILS-Base'].append(pathfinder.find_path_ails(start, goal, strategy='base'))
        if 'AILS-Adaptive' in algorithms:
            results['AILS-Adaptive'].append(pathfinder.find_path_ails(start, goal, strategy='adaptive'))

    return results


def compute_statistics(results: Dict[str, List[PathResult]]) -> Dict[str, Dict]:
    """Compute aggregate statistics from benchmark results."""
    stats = {}

    for method, result_list in results.items():
        times = [r.time_ms for r in result_list if r.path_found]
        nodes = [r.nodes_visited for r in result_list if r.path_found]
        costs = [r.cost for r in result_list if r.path_found]
        success_rate = sum(1 for r in result_list if r.path_found) / len(result_list) if result_list else 0

        if times:
            stats[method] = {
                'time_mean': np.mean(times),
                'time_std': np.std(times),
                'time_median': np.median(times),
                'time_min': np.min(times),
                'time_max': np.max(times),
                'nodes_mean': np.mean(nodes),
                'nodes_std': np.std(nodes),
                'nodes_median': np.median(nodes),
                'cost_mean': np.mean(costs),
                'cost_std': np.std(costs),
                'success_rate': success_rate * 100,
                'num_samples': len(times)
            }
        else:
            stats[method] = {
                'time_mean': 0, 'time_std': 0, 'time_median': 0,
                'time_min': 0, 'time_max': 0,
                'nodes_mean': 0, 'nodes_std': 0, 'nodes_median': 0,
                'cost_mean': 0, 'cost_std': 0,
                'success_rate': 0, 'num_samples': 0
            }

    return stats


def compute_improvement(stats: Dict[str, Dict], baseline: str = 'A*') -> Dict[str, Dict]:
    """Compute improvement percentages relative to baseline."""
    improvements = {}
    baseline_stats = stats.get(baseline, {})

    if not baseline_stats or baseline_stats['nodes_mean'] == 0:
        return improvements

    for method, method_stats in stats.items():
        if method == baseline:
            continue

        improvements[method] = {
            'time_improvement': ((baseline_stats['time_mean'] - method_stats['time_mean']) /
                                baseline_stats['time_mean'] * 100) if baseline_stats['time_mean'] > 0 else 0,
            'nodes_improvement': ((baseline_stats['nodes_mean'] - method_stats['nodes_mean']) /
                                 baseline_stats['nodes_mean'] * 100) if baseline_stats['nodes_mean'] > 0 else 0,
        }

    return improvements


# ============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# ============================================================================

def paired_t_test(sample1: List[float], sample2: List[float]) -> Tuple[float, float]:
    """Perform paired t-test."""
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(sample1, sample2)
    return t_stat, p_value


def cohens_d(sample1: List[float], sample2: List[float]) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(sample1), len(sample2)
    var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0

    return (np.mean(sample1) - np.mean(sample2)) / pooled_std


def wilcoxon_test(sample1: List[float], sample2: List[float]) -> Tuple[float, float]:
    """Perform Wilcoxon signed-rank test (non-parametric alternative to t-test)."""
    from scipy import stats
    stat, p_value = stats.wilcoxon(sample1, sample2)
    return stat, p_value


# ============================================================================
# MOVING AI MAP LOADER
# ============================================================================

class MovingAIMapLoader:
    """Loader for Moving AI Lab benchmark maps."""

    @staticmethod
    def load_map(filepath: str) -> np.ndarray:
        """Load a Moving AI Lab .map file."""
        with open(filepath, 'r') as f:
            lines = f.readlines()

        height = int(lines[1].strip().split()[1])
        width = int(lines[2].strip().split()[1])

        grid = np.zeros((height, width), dtype=int)

        for i, line in enumerate(lines[4:4+height]):
            for j, char in enumerate(line.strip()):
                if j < width:
                    if char in '.GS':
                        grid[i, j] = 0
                    else:
                        grid[i, j] = 1

        return grid

    @staticmethod
    def load_scenarios(filepath: str) -> List[Dict]:
        """Load scenario file for benchmarking."""
        scenarios = []

        with open(filepath, 'r') as f:
            lines = f.readlines()

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
# MAIN DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AILS: Adaptive Incremental Line Search - Core Module")
    print("=" * 70)

    # Generate test grid
    print("\nGenerating 100x100 test grid with 25% obstacle density...")
    grid = GridGenerator.generate_random(100, density=0.25, seed=42)
    print(f"Grid size: {grid.shape}")
    print(f"Traversable cells: {np.sum(grid == 0)}")
    print(f"Obstacle cells: {np.sum(grid == 1)}")

    # Single path comparison
    print("\nRunning single path comparison...")
    pathfinder = AILSPathfinder(grid)

    traversable = np.argwhere(grid == 0)
    np.random.seed(123)
    idx = np.random.choice(len(traversable), 2, replace=False)
    start = tuple(traversable[idx[0]])
    goal = tuple(traversable[idx[1]])

    print(f"Start: {start}")
    print(f"Goal: {goal}")

    results = pathfinder.compare_methods(start, goal)

    print("\nResults:")
    print("-" * 70)
    print(f"{'Method':<20} {'Time (ms)':<12} {'Nodes':<10} {'Cost':<10} {'Found'}")
    print("-" * 70)

    for method, result in results.items():
        print(f"{method:<20} {result.time_ms:<12.3f} {result.nodes_visited:<10} "
              f"{result.cost:<10.2f} {result.path_found}")

    print("\n" + "=" * 70)
    print("Module loaded successfully. Import this module to use AILS in your experiments.")
    print("=" * 70)
