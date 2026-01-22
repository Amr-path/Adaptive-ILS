"""
Jump Point Search (JPS) Implementation
=======================================

An optimization of A* for uniform-cost grids that exploits
symmetry to reduce the number of nodes expanded.

Reference:
    Harabor, D., & Grastien, A. (2011).
    Online graph pruning for pathfinding on grid maps.
    AAAI Conference on Artificial Intelligence.
"""

import time
from typing import Dict, List, Tuple, Optional, Set
import sys
sys.path.insert(0, '..')
from core.data_structures import Grid, Point, PathResult, PriorityQueue
from core.heuristics import octile


class JumpPointSearch:
    """
    Jump Point Search (JPS) Algorithm.

    JPS is an optimization of A* that reduces the number of nodes
    expanded by exploiting path symmetry in uniform-cost grids.

    Note: JPS only works correctly on uniform-cost 8-connected grids.
    """

    def __init__(self, grid: Grid):
        """
        Initialize JPS.

        Args:
            grid: The grid to search on (must be 8-connected)
        """
        self.grid = grid
        if not grid.eight_connected:
            raise ValueError("JPS requires an 8-connected grid")

    def find_path(self, start: Point, goal: Point) -> PathResult:
        """
        Find the shortest path from start to goal using JPS.

        Args:
            start: Starting point
            goal: Goal point

        Returns:
            PathResult with path and statistics
        """
        start_time = time.perf_counter()

        # Validate start and goal
        if not self.grid.is_free(start.x, start.y):
            return PathResult(found=False, algorithm="JPS")
        if not self.grid.is_free(goal.x, goal.y):
            return PathResult(found=False, algorithm="JPS")

        # Priority queue
        open_set = PriorityQueue()
        open_set.push(start, 0.0)

        # Tracking
        came_from: Dict[Point, Tuple[Point, int, int]] = {}  # point -> (parent, dx, dy)
        g_score: Dict[Point, float] = {start: 0.0}
        visited = 0

        while open_set:
            current, _ = open_set.pop()
            visited += 1

            if current == goal:
                # Reconstruct path with intermediate points
                path = self._reconstruct_full_path(came_from, start, current)
                end_time = time.perf_counter()

                return PathResult(
                    path=path,
                    found=True,
                    cost=g_score[current],
                    visited_nodes=visited,
                    execution_time_ms=(end_time - start_time) * 1000,
                    algorithm="JPS"
                )

            # Get successors through jumping
            successors = self._get_successors(current, came_from, goal)

            for successor, dx, dy in successors:
                # Compute distance to successor
                dist = self._distance(current, successor)
                tentative_g = g_score[current] + dist

                if successor not in g_score or tentative_g < g_score[successor]:
                    came_from[successor] = (current, dx, dy)
                    g_score[successor] = tentative_g
                    f_score = tentative_g + octile(successor, goal)
                    open_set.push(successor, f_score)

        end_time = time.perf_counter()
        return PathResult(
            found=False,
            visited_nodes=visited,
            execution_time_ms=(end_time - start_time) * 1000,
            algorithm="JPS"
        )

    def _get_successors(self, current: Point,
                        came_from: Dict[Point, Tuple[Point, int, int]],
                        goal: Point) -> List[Tuple[Point, int, int]]:
        """
        Get jump point successors of current node.

        Args:
            current: Current node
            came_from: Parent tracking dict
            goal: Goal point

        Returns:
            List of (successor, dx, dy) tuples
        """
        successors = []

        # Get directions to explore based on parent
        if current in came_from:
            parent, dx, dy = came_from[current]
            directions = self._get_pruned_directions(current, dx, dy)
        else:
            # Start node - explore all directions
            directions = [
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1)
            ]

        for dx, dy in directions:
            jump_point = self._jump(current, dx, dy, goal)
            if jump_point:
                successors.append((jump_point, dx, dy))

        return successors

    def _get_pruned_directions(self, current: Point, dx: int, dy: int) -> List[Tuple[int, int]]:
        """
        Get pruned set of directions based on movement direction.

        This is the key to JPS's efficiency - we only need to explore
        directions that could lead to new optimal paths.
        """
        directions = []

        if dx != 0 and dy != 0:
            # Diagonal move - prune to natural neighbors and forced neighbors
            # Natural: continue diagonal, and the two cardinal components
            directions.append((dx, dy))  # Continue diagonal
            directions.append((dx, 0))   # Horizontal component
            directions.append((0, dy))   # Vertical component

            # Check for forced neighbors
            x, y = current.x, current.y
            if not self.grid.is_free(x - dx, y):
                directions.append((-dx, dy))
            if not self.grid.is_free(x, y - dy):
                directions.append((dx, -dy))

        else:
            # Cardinal move
            if dx != 0:
                # Horizontal
                directions.append((dx, 0))  # Continue horizontal
                # Check forced neighbors
                if not self.grid.is_free(current.x, current.y + 1):
                    directions.append((dx, 1))
                if not self.grid.is_free(current.x, current.y - 1):
                    directions.append((dx, -1))
            else:
                # Vertical
                directions.append((0, dy))  # Continue vertical
                # Check forced neighbors
                if not self.grid.is_free(current.x + 1, current.y):
                    directions.append((1, dy))
                if not self.grid.is_free(current.x - 1, current.y):
                    directions.append((-1, dy))

        return directions

    def _jump(self, current: Point, dx: int, dy: int, goal: Point) -> Optional[Point]:
        """
        Jump in direction (dx, dy) to find a jump point.

        A jump point is found when:
        1. The goal is reached
        2. A forced neighbor exists
        3. (For diagonal) A cardinal scan finds a jump point
        """
        next_x = current.x + dx
        next_y = current.y + dy

        # Check if next position is valid
        if not self.grid.is_free(next_x, next_y):
            return None

        next_point = Point(next_x, next_y)

        # Check if goal reached
        if next_point == goal:
            return next_point

        # Diagonal move
        if dx != 0 and dy != 0:
            # Check for forced neighbors
            if ((not self.grid.is_free(next_x - dx, next_y) and
                 self.grid.is_free(next_x - dx, next_y + dy)) or
                (not self.grid.is_free(next_x, next_y - dy) and
                 self.grid.is_free(next_x + dx, next_y - dy))):
                return next_point

            # Check cardinal directions
            if self._jump(next_point, dx, 0, goal) or self._jump(next_point, 0, dy, goal):
                return next_point

        else:
            # Cardinal move
            if dx != 0:
                # Horizontal
                if ((not self.grid.is_free(next_x, next_y + 1) and
                     self.grid.is_free(next_x + dx, next_y + 1)) or
                    (not self.grid.is_free(next_x, next_y - 1) and
                     self.grid.is_free(next_x + dx, next_y - 1))):
                    return next_point
            else:
                # Vertical
                if ((not self.grid.is_free(next_x + 1, next_y) and
                     self.grid.is_free(next_x + 1, next_y + dy)) or
                    (not self.grid.is_free(next_x - 1, next_y) and
                     self.grid.is_free(next_x - 1, next_y + dy))):
                    return next_point

        # Continue jumping in same direction
        return self._jump(next_point, dx, dy, goal)

    def _distance(self, p1: Point, p2: Point) -> float:
        """Compute the actual grid distance between two points."""
        dx = abs(p1.x - p2.x)
        dy = abs(p1.y - p2.y)
        # Diagonal moves cost sqrt(2), cardinal moves cost 1
        diagonal = min(dx, dy)
        cardinal = abs(dx - dy)
        return diagonal * 1.41421356237 + cardinal

    def _reconstruct_full_path(self, came_from: Dict[Point, Tuple[Point, int, int]],
                                start: Point, goal: Point) -> List[Point]:
        """
        Reconstruct the full path including intermediate points.

        JPS only stores jump points, so we need to fill in the
        intermediate points between them.
        """
        # First get the jump point path
        jump_path = [goal]
        current = goal

        while current in came_from:
            parent, _, _ = came_from[current]
            jump_path.append(parent)
            current = parent

        jump_path.reverse()

        # Now fill in intermediate points
        full_path = [jump_path[0]]

        for i in range(len(jump_path) - 1):
            p1, p2 = jump_path[i], jump_path[i + 1]
            intermediate = self._interpolate(p1, p2)
            full_path.extend(intermediate[1:])  # Skip first as it's already added

        return full_path

    def _interpolate(self, p1: Point, p2: Point) -> List[Point]:
        """Generate intermediate points between two jump points."""
        path = [p1]

        dx = 0 if p2.x == p1.x else (1 if p2.x > p1.x else -1)
        dy = 0 if p2.y == p1.y else (1 if p2.y > p1.y else -1)

        x, y = p1.x, p1.y

        while x != p2.x or y != p2.y:
            if x != p2.x:
                x += dx
            if y != p2.y:
                y += dy
            path.append(Point(x, y))

        return path


def jps(grid: Grid, start: Point, goal: Point) -> PathResult:
    """
    Convenience function for JPS pathfinding.

    Args:
        grid: The grid (must be 8-connected)
        start: Starting point
        goal: Goal point

    Returns:
        PathResult
    """
    return JumpPointSearch(grid).find_path(start, goal)
