"""
Dijkstra's Algorithm Implementation
====================================

Classic Dijkstra's shortest path algorithm for weighted graphs.
"""

import time
from typing import Dict, List
import sys
sys.path.insert(0, '..')
from core.data_structures import Grid, Point, PathResult, PriorityQueue


class Dijkstra:
    """
    Dijkstra's Shortest Path Algorithm.

    An uninformed search algorithm that finds the shortest path in
    weighted graphs. Equivalent to A* with h(n) = 0.
    """

    def __init__(self, grid: Grid):
        """
        Initialize Dijkstra.

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
            return PathResult(found=False, algorithm="Dijkstra")
        if not self.grid.is_free(goal.x, goal.y):
            return PathResult(found=False, algorithm="Dijkstra")

        # Priority queue
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
                path = self._reconstruct_path(came_from, current)
                end_time = time.perf_counter()

                return PathResult(
                    path=path,
                    found=True,
                    cost=g_score[current],
                    visited_nodes=visited,
                    execution_time_ms=(end_time - start_time) * 1000,
                    algorithm="Dijkstra"
                )

            for neighbor, edge_cost in self.grid.get_neighbors(current):
                tentative_g = g_score[current] + edge_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    open_set.push(neighbor, tentative_g)

        end_time = time.perf_counter()
        return PathResult(
            found=False,
            visited_nodes=visited,
            execution_time_ms=(end_time - start_time) * 1000,
            algorithm="Dijkstra"
        )

    def _reconstruct_path(self, came_from: Dict[Point, Point], current: Point) -> List[Point]:
        """Reconstruct path from came_from map."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


def dijkstra(grid: Grid, start: Point, goal: Point) -> PathResult:
    """
    Convenience function for Dijkstra pathfinding.

    Args:
        grid: The grid
        start: Starting point
        goal: Goal point

    Returns:
        PathResult
    """
    return Dijkstra(grid).find_path(start, goal)
