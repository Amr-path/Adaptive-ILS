"""
AILS Algorithm Implementation
==============================

Complete implementation of the Adaptive Incremental Line Search (AILS)
framework for grid-based pathfinding optimization.

AILS is an algorithm-agnostic wrapper that enhances any graph search
algorithm by constraining the search to an adaptive corridor.
"""

import time
from dataclasses import dataclass, field
from typing import Set, Dict, List, Callable, Optional, Tuple, Any
from .data_structures import Grid, Point, PathResult, PriorityQueue
from .heuristics import octile, manhattan, euclidean, chebyshev, get_heuristic
from .corridor_builder import CorridorBuilder, CorridorConfig, CorridorStrategy


@dataclass
class AILSConfig:
    """Configuration for AILS algorithm."""
    # Corridor parameters
    r_min: int = 2
    r_max: int = 15
    alpha: float = 1.0
    beta: float = 0.3
    window_size: int = 5

    # Strategy selection
    strategy: str = "density_adaptive"  # "base", "density_adaptive", "gradient_based"

    # Fallback parameters
    max_expansions: int = 10
    expansion_step: int = 2

    # Search parameters
    heuristic: str = "octile"  # "manhattan", "euclidean", "chebyshev", "octile"


class AILS:
    """
    Adaptive Incremental Line Search (AILS) Framework.

    A wrapper framework that enhances any graph search algorithm by
    constraining the search to an adaptive corridor based on local
    obstacle density.

    Features:
    - Algorithm-agnostic design
    - Multi-strategy corridor construction
    - Automatic fallback with corridor expansion
    - Completeness guarantee
    """

    def __init__(self, grid: Grid, config: Optional[AILSConfig] = None):
        """
        Initialize AILS.

        Args:
            grid: The grid to search on
            config: AILS configuration (uses defaults if None)
        """
        self.grid = grid
        self.config = config or AILSConfig()

        # Get heuristic function
        self._heuristic = get_heuristic(self.config.heuristic)

        # Create corridor builder
        strategy_map = {
            "base": CorridorStrategy.BASE,
            "density_adaptive": CorridorStrategy.DENSITY_ADAPTIVE,
            "gradient_based": CorridorStrategy.GRADIENT_BASED,
        }

        corridor_config = CorridorConfig(
            r_min=self.config.r_min,
            r_max=self.config.r_max,
            alpha=self.config.alpha,
            beta=self.config.beta,
            window_size=self.config.window_size,
            strategy=strategy_map.get(self.config.strategy, CorridorStrategy.DENSITY_ADAPTIVE)
        )

        self.corridor_builder = CorridorBuilder(grid, corridor_config)

    def find_path(self, start: Point, goal: Point) -> PathResult:
        """
        Find a path from start to goal using AILS.

        This is the main entry point for pathfinding.

        Args:
            start: Starting point
            goal: Goal point

        Returns:
            PathResult with path and statistics
        """
        start_time = time.perf_counter()

        # Validate start and goal
        if not self.grid.is_free(start.x, start.y):
            return PathResult(found=False, algorithm="AILS")
        if not self.grid.is_free(goal.x, goal.y):
            return PathResult(found=False, algorithm="AILS")

        # Build initial corridor
        corridor, radii, strategy_used = self.corridor_builder.build_corridor(start, goal)

        # Ensure start and goal are in corridor
        corridor.add(start)
        corridor.add(goal)

        # Try to find path within corridor
        total_visited = 0
        expansions = 0

        for expansion in range(self.config.max_expansions + 1):
            path, cost, visited = self._astar_in_corridor(start, goal, corridor)
            total_visited += visited

            if path:
                end_time = time.perf_counter()
                return PathResult(
                    path=path,
                    found=True,
                    cost=cost,
                    visited_nodes=total_visited,
                    execution_time_ms=(end_time - start_time) * 1000,
                    corridor_size=len(corridor),
                    corridor_efficiency=len(corridor) / (self.grid.width * self.grid.height),
                    expansions=expansions,
                    strategy_used=strategy_used.value,
                    algorithm="AILS"
                )

            # Expand corridor for next attempt
            expansions += 1
            corridor = self.corridor_builder.expand_corridor(
                corridor, self.config.expansion_step
            )

        # Failed to find path even after max expansions
        end_time = time.perf_counter()
        return PathResult(
            found=False,
            visited_nodes=total_visited,
            execution_time_ms=(end_time - start_time) * 1000,
            corridor_size=len(corridor),
            corridor_efficiency=len(corridor) / (self.grid.width * self.grid.height),
            expansions=expansions,
            strategy_used=strategy_used.value,
            algorithm="AILS"
        )

    def _astar_in_corridor(self, start: Point, goal: Point,
                           corridor: Set[Point]) -> Tuple[List[Point], float, int]:
        """
        Run A* search constrained to a corridor.

        Args:
            start: Starting point
            goal: Goal point
            corridor: Set of points to constrain search to

        Returns:
            Tuple of (path, cost, visited_count)
        """
        # Priority queue: (f_score, point)
        open_set = PriorityQueue()
        open_set.push(start, 0.0)

        # Tracking
        came_from: Dict[Point, Point] = {}
        g_score: Dict[Point, float] = {start: 0.0}
        visited = 0

        while open_set:
            current, _ = open_set.pop()
            visited += 1

            if current == goal:
                # Reconstruct path
                path = self._reconstruct_path(came_from, current)
                return path, g_score[current], visited

            # Get neighbors within corridor
            for neighbor, edge_cost in self.grid.get_neighbors_in_corridor(current, corridor):
                tentative_g = g_score[current] + edge_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, goal)
                    open_set.push(neighbor, f_score)

        return [], float('inf'), visited

    def _reconstruct_path(self, came_from: Dict[Point, Point], current: Point) -> List[Point]:
        """Reconstruct path from came_from map."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def find_path_with_base_algorithm(self, start: Point, goal: Point,
                                      algorithm: str = "astar") -> PathResult:
        """
        Find path using specified base algorithm within AILS corridor.

        Args:
            start: Starting point
            goal: Goal point
            algorithm: Algorithm to use ("astar", "dijkstra", "bfs", "greedy")

        Returns:
            PathResult
        """
        start_time = time.perf_counter()

        if not self.grid.is_free(start.x, start.y) or not self.grid.is_free(goal.x, goal.y):
            return PathResult(found=False, algorithm=f"AILS-{algorithm}")

        # Build corridor
        corridor, radii, strategy_used = self.corridor_builder.build_corridor(start, goal)
        corridor.add(start)
        corridor.add(goal)

        total_visited = 0
        expansions = 0

        for expansion in range(self.config.max_expansions + 1):
            if algorithm == "astar":
                path, cost, visited = self._astar_in_corridor(start, goal, corridor)
            elif algorithm == "dijkstra":
                path, cost, visited = self._dijkstra_in_corridor(start, goal, corridor)
            elif algorithm == "bfs":
                path, cost, visited = self._bfs_in_corridor(start, goal, corridor)
            elif algorithm == "greedy":
                path, cost, visited = self._greedy_in_corridor(start, goal, corridor)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            total_visited += visited

            if path:
                end_time = time.perf_counter()
                return PathResult(
                    path=path,
                    found=True,
                    cost=cost,
                    visited_nodes=total_visited,
                    execution_time_ms=(end_time - start_time) * 1000,
                    corridor_size=len(corridor),
                    corridor_efficiency=len(corridor) / (self.grid.width * self.grid.height),
                    expansions=expansions,
                    strategy_used=strategy_used.value,
                    algorithm=f"AILS-{algorithm}"
                )

            expansions += 1
            corridor = self.corridor_builder.expand_corridor(corridor, self.config.expansion_step)

        end_time = time.perf_counter()
        return PathResult(
            found=False,
            visited_nodes=total_visited,
            execution_time_ms=(end_time - start_time) * 1000,
            corridor_size=len(corridor),
            expansions=expansions,
            strategy_used=strategy_used.value,
            algorithm=f"AILS-{algorithm}"
        )

    def _dijkstra_in_corridor(self, start: Point, goal: Point,
                              corridor: Set[Point]) -> Tuple[List[Point], float, int]:
        """Dijkstra's algorithm within corridor (A* with h=0)."""
        open_set = PriorityQueue()
        open_set.push(start, 0.0)

        came_from: Dict[Point, Point] = {}
        g_score: Dict[Point, float] = {start: 0.0}
        visited = 0

        while open_set:
            current, _ = open_set.pop()
            visited += 1

            if current == goal:
                path = self._reconstruct_path(came_from, current)
                return path, g_score[current], visited

            for neighbor, edge_cost in self.grid.get_neighbors_in_corridor(current, corridor):
                tentative_g = g_score[current] + edge_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    open_set.push(neighbor, tentative_g)  # No heuristic

        return [], float('inf'), visited

    def _bfs_in_corridor(self, start: Point, goal: Point,
                         corridor: Set[Point]) -> Tuple[List[Point], float, int]:
        """BFS within corridor (unweighted shortest path)."""
        from collections import deque

        queue = deque([start])
        came_from: Dict[Point, Point] = {}
        visited_set = {start}
        visited = 0

        while queue:
            current = queue.popleft()
            visited += 1

            if current == goal:
                path = self._reconstruct_path(came_from, current)
                # Compute actual path cost
                cost = sum(
                    1.41421356237 if abs(path[i].x - path[i+1].x) + abs(path[i].y - path[i+1].y) == 2 else 1.0
                    for i in range(len(path) - 1)
                )
                return path, cost, visited

            for neighbor, _ in self.grid.get_neighbors_in_corridor(current, corridor):
                if neighbor not in visited_set:
                    visited_set.add(neighbor)
                    came_from[neighbor] = current
                    queue.append(neighbor)

        return [], float('inf'), visited

    def _greedy_in_corridor(self, start: Point, goal: Point,
                            corridor: Set[Point]) -> Tuple[List[Point], float, int]:
        """Greedy best-first search within corridor."""
        open_set = PriorityQueue()
        open_set.push(start, self._heuristic(start, goal))

        came_from: Dict[Point, Point] = {}
        visited_set = {start}
        visited = 0

        while open_set:
            current, _ = open_set.pop()
            visited += 1

            if current == goal:
                path = self._reconstruct_path(came_from, current)
                # Compute actual path cost
                cost = 0.0
                for i in range(len(path) - 1):
                    if abs(path[i].x - path[i+1].x) + abs(path[i].y - path[i+1].y) == 2:
                        cost += 1.41421356237
                    else:
                        cost += 1.0
                return path, cost, visited

            for neighbor, _ in self.grid.get_neighbors_in_corridor(current, corridor):
                if neighbor not in visited_set:
                    visited_set.add(neighbor)
                    came_from[neighbor] = current
                    h = self._heuristic(neighbor, goal)
                    open_set.push(neighbor, h)

        return [], float('inf'), visited


def create_ails(grid: Grid, **kwargs) -> AILS:
    """
    Factory function to create an AILS instance.

    Args:
        grid: The grid to search on
        **kwargs: Configuration parameters (see AILSConfig)

    Returns:
        Configured AILS instance
    """
    config = AILSConfig(**kwargs)
    return AILS(grid, config)
