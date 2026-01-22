"""
Breadth-First Search Implementation
=====================================

Classic BFS for unweighted shortest path finding.
"""

import time
from typing import Dict, List
from collections import deque
import sys
sys.path.insert(0, '..')
from core.data_structures import Grid, Point, PathResult


class BFS:
    """
    Breadth-First Search Algorithm.

    Finds the shortest path in unweighted graphs (minimum number of edges).
    For weighted graphs, the path cost is computed but may not be optimal.
    """

    def __init__(self, grid: Grid):
        """
        Initialize BFS.

        Args:
            grid: The grid to search on
        """
        self.grid = grid

    def find_path(self, start: Point, goal: Point) -> PathResult:
        """
        Find the shortest path from start to goal.

        Args:
            start: Starting point
            goal: Goal point

        Returns:
            PathResult with path and statistics
        """
        start_time = time.perf_counter()

        # Validate start and goal
        if not self.grid.is_free(start.x, start.y):
            return PathResult(found=False, algorithm="BFS")
        if not self.grid.is_free(goal.x, goal.y):
            return PathResult(found=False, algorithm="BFS")

        # BFS queue
        queue = deque([start])
        came_from: Dict[Point, Point] = {}
        visited_set = {start}
        visited = 0

        while queue:
            current = queue.popleft()
            visited += 1

            if current == goal:
                path = self._reconstruct_path(came_from, current)
                cost = self._compute_path_cost(path)
                end_time = time.perf_counter()

                return PathResult(
                    path=path,
                    found=True,
                    cost=cost,
                    visited_nodes=visited,
                    execution_time_ms=(end_time - start_time) * 1000,
                    algorithm="BFS"
                )

            for neighbor, _ in self.grid.get_neighbors(current):
                if neighbor not in visited_set:
                    visited_set.add(neighbor)
                    came_from[neighbor] = current
                    queue.append(neighbor)

        end_time = time.perf_counter()
        return PathResult(
            found=False,
            visited_nodes=visited,
            execution_time_ms=(end_time - start_time) * 1000,
            algorithm="BFS"
        )

    def _reconstruct_path(self, came_from: Dict[Point, Point], current: Point) -> List[Point]:
        """Reconstruct path from came_from map."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _compute_path_cost(self, path: List[Point]) -> float:
        """Compute the actual cost of a path."""
        if len(path) < 2:
            return 0.0

        cost = 0.0
        SQRT2 = 1.41421356237

        for i in range(len(path) - 1):
            dx = abs(path[i].x - path[i + 1].x)
            dy = abs(path[i].y - path[i + 1].y)
            if dx + dy == 2:  # Diagonal move
                cost += SQRT2
            else:
                cost += 1.0

        return cost


def bfs(grid: Grid, start: Point, goal: Point) -> PathResult:
    """
    Convenience function for BFS pathfinding.

    Args:
        grid: The grid
        start: Starting point
        goal: Goal point

    Returns:
        PathResult
    """
    return BFS(grid).find_path(start, goal)
