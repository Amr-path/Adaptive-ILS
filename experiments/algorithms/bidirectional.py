"""
Bidirectional A* Implementation
================================

A* search from both start and goal simultaneously.
"""

import time
from typing import Dict, List, Set, Tuple, Optional
import sys
sys.path.insert(0, '..')
from core.data_structures import Grid, Point, PathResult, PriorityQueue
from core.heuristics import octile, get_heuristic


class BidirectionalAStar:
    """
    Bidirectional A* Search Algorithm.

    Searches from both start and goal simultaneously, meeting in the middle.
    Can find paths faster than unidirectional A* when the search space
    expands rapidly.
    """

    def __init__(self, grid: Grid, heuristic: str = "octile"):
        """
        Initialize Bidirectional A*.

        Args:
            grid: The grid to search on
            heuristic: Heuristic function name
        """
        self.grid = grid
        self._heuristic = get_heuristic(heuristic)

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
            return PathResult(found=False, algorithm="Bidirectional-A*")
        if not self.grid.is_free(goal.x, goal.y):
            return PathResult(found=False, algorithm="Bidirectional-A*")

        # Same point
        if start == goal:
            end_time = time.perf_counter()
            return PathResult(
                path=[start],
                found=True,
                cost=0.0,
                visited_nodes=1,
                execution_time_ms=(end_time - start_time) * 1000,
                algorithm="Bidirectional-A*"
            )

        # Forward search (from start)
        open_f = PriorityQueue()
        open_f.push(start, 0.0)
        came_from_f: Dict[Point, Point] = {}
        g_score_f: Dict[Point, float] = {start: 0.0}
        closed_f: Set[Point] = set()

        # Backward search (from goal)
        open_b = PriorityQueue()
        open_b.push(goal, 0.0)
        came_from_b: Dict[Point, Point] = {}
        g_score_b: Dict[Point, float] = {goal: 0.0}
        closed_b: Set[Point] = set()

        visited = 0
        best_path_cost = float('inf')
        meeting_point = None

        while open_f and open_b:
            # Check termination condition
            if open_f:
                min_f = min(g_score_f.get(p, float('inf'))
                           for p in g_score_f if p not in closed_f)
            else:
                min_f = float('inf')

            if open_b:
                min_b = min(g_score_b.get(p, float('inf'))
                           for p in g_score_b if p not in closed_b)
            else:
                min_b = float('inf')

            if min_f + min_b >= best_path_cost:
                break

            # Alternate between forward and backward search
            # Forward step
            if open_f:
                current_f, _ = open_f.pop()
                visited += 1
                closed_f.add(current_f)

                # Check if this node was reached from backward search
                if current_f in g_score_b:
                    path_cost = g_score_f[current_f] + g_score_b[current_f]
                    if path_cost < best_path_cost:
                        best_path_cost = path_cost
                        meeting_point = current_f

                # Expand forward
                for neighbor, edge_cost in self.grid.get_neighbors(current_f):
                    if neighbor in closed_f:
                        continue

                    tentative_g = g_score_f[current_f] + edge_cost

                    if neighbor not in g_score_f or tentative_g < g_score_f[neighbor]:
                        came_from_f[neighbor] = current_f
                        g_score_f[neighbor] = tentative_g
                        f_score = tentative_g + self._heuristic(neighbor, goal)
                        open_f.push(neighbor, f_score)

                        # Check connection
                        if neighbor in g_score_b:
                            path_cost = tentative_g + g_score_b[neighbor]
                            if path_cost < best_path_cost:
                                best_path_cost = path_cost
                                meeting_point = neighbor

            # Backward step
            if open_b:
                current_b, _ = open_b.pop()
                visited += 1
                closed_b.add(current_b)

                # Check if this node was reached from forward search
                if current_b in g_score_f:
                    path_cost = g_score_f[current_b] + g_score_b[current_b]
                    if path_cost < best_path_cost:
                        best_path_cost = path_cost
                        meeting_point = current_b

                # Expand backward
                for neighbor, edge_cost in self.grid.get_neighbors(current_b):
                    if neighbor in closed_b:
                        continue

                    tentative_g = g_score_b[current_b] + edge_cost

                    if neighbor not in g_score_b or tentative_g < g_score_b[neighbor]:
                        came_from_b[neighbor] = current_b
                        g_score_b[neighbor] = tentative_g
                        f_score = tentative_g + self._heuristic(neighbor, start)
                        open_b.push(neighbor, f_score)

                        # Check connection
                        if neighbor in g_score_f:
                            path_cost = g_score_f[neighbor] + tentative_g
                            if path_cost < best_path_cost:
                                best_path_cost = path_cost
                                meeting_point = neighbor

        end_time = time.perf_counter()

        if meeting_point is not None:
            # Reconstruct path
            path = self._reconstruct_bidirectional_path(
                came_from_f, came_from_b, meeting_point, start, goal
            )
            return PathResult(
                path=path,
                found=True,
                cost=best_path_cost,
                visited_nodes=visited,
                execution_time_ms=(end_time - start_time) * 1000,
                algorithm="Bidirectional-A*"
            )
        else:
            return PathResult(
                found=False,
                visited_nodes=visited,
                execution_time_ms=(end_time - start_time) * 1000,
                algorithm="Bidirectional-A*"
            )

    def _reconstruct_bidirectional_path(self,
                                        came_from_f: Dict[Point, Point],
                                        came_from_b: Dict[Point, Point],
                                        meeting: Point,
                                        start: Point,
                                        goal: Point) -> List[Point]:
        """Reconstruct path from bidirectional search."""
        # Forward path (start to meeting)
        path_f = []
        current = meeting
        while current in came_from_f:
            path_f.append(current)
            current = came_from_f[current]
        path_f.append(start)
        path_f.reverse()

        # Backward path (meeting to goal)
        path_b = []
        current = meeting
        while current in came_from_b:
            current = came_from_b[current]
            path_b.append(current)

        return path_f + path_b


def bidirectional_astar(grid: Grid, start: Point, goal: Point,
                        heuristic: str = "octile") -> PathResult:
    """
    Convenience function for Bidirectional A* pathfinding.

    Args:
        grid: The grid
        start: Starting point
        goal: Goal point
        heuristic: Heuristic name

    Returns:
        PathResult
    """
    return BidirectionalAStar(grid, heuristic).find_path(start, goal)
