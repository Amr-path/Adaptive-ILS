"""
A* Algorithm Implementation
============================

Standard A* pathfinding algorithm with configurable heuristics.
"""

import time
from typing import Dict, List, Tuple, Optional, Callable
import sys
sys.path.insert(0, '..')
from core.data_structures import Grid, Point, PathResult, PriorityQueue
from core.heuristics import octile, get_heuristic


class AStar:
    """
    A* (A-Star) Pathfinding Algorithm.

    A* is an informed search algorithm that uses heuristics to guide
    the search towards the goal, guaranteeing optimal paths when using
    admissible heuristics.
    """

    def __init__(self, grid: Grid, heuristic: str = "octile"):
        """
        Initialize A*.

        Args:
            grid: The grid to search on
            heuristic: Heuristic function name
        """
        self.grid = grid
        self._heuristic = get_heuristic(heuristic)
        self.heuristic_name = heuristic

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
            return PathResult(found=False, algorithm="A*")
        if not self.grid.is_free(goal.x, goal.y):
            return PathResult(found=False, algorithm="A*")

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
                # Reconstruct path
                path = self._reconstruct_path(came_from, current)
                end_time = time.perf_counter()

                return PathResult(
                    path=path,
                    found=True,
                    cost=g_score[current],
                    visited_nodes=visited,
                    execution_time_ms=(end_time - start_time) * 1000,
                    algorithm="A*"
                )

            for neighbor, edge_cost in self.grid.get_neighbors(current):
                tentative_g = g_score[current] + edge_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, goal)
                    open_set.push(neighbor, f_score)

        end_time = time.perf_counter()
        return PathResult(
            found=False,
            visited_nodes=visited,
            execution_time_ms=(end_time - start_time) * 1000,
            algorithm="A*"
        )

    def _reconstruct_path(self, came_from: Dict[Point, Point], current: Point) -> List[Point]:
        """Reconstruct path from came_from map."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


def astar(grid: Grid, start: Point, goal: Point, heuristic: str = "octile") -> PathResult:
    """
    Convenience function for A* pathfinding.

    Args:
        grid: The grid
        start: Starting point
        goal: Goal point
        heuristic: Heuristic name

    Returns:
        PathResult
    """
    return AStar(grid, heuristic).find_path(start, goal)
