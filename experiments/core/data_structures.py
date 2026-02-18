"""
Data Structures for AILS Algorithm
===================================

Core data structures used throughout the AILS framework.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Set, Dict, Optional, Any
import heapq
import numpy as np


@dataclass(frozen=True)
class Point:
    """Immutable 2D point representation."""
    x: int
    y: int

    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Point') -> 'Point':
        return Point(self.x - other.x, self.y - other.y)

    def __iter__(self):
        yield self.x
        yield self.y

    def to_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)

    def manhattan_distance(self, other: 'Point') -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

    def euclidean_distance(self, other: 'Point') -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


class PriorityQueue:
    """
    Efficient priority queue using heapq with pure lazy deletion.

    Each push assigns a unique counter (generation). On pop, stale
    entries whose counter doesn't match the latest in entry_finder
    are silently discarded. This avoids the O(n) list.index() call
    that the old mark-as-removed approach required.
    """

    def __init__(self):
        self._heap: List[Tuple[float, int, Any]] = []
        # item -> latest counter assigned to it
        self._entry_finder: Dict[Any, int] = {}
        self._counter = 0

    def push(self, item: Any, priority: float) -> None:
        """Add item with priority, or update priority if already exists."""
        self._counter += 1
        self._entry_finder[item] = self._counter
        heapq.heappush(self._heap, (priority, self._counter, item))

    def pop(self) -> Tuple[Any, float]:
        """Remove and return the item with lowest priority."""
        while self._heap:
            priority, count, item = heapq.heappop(self._heap)
            # Accept only the most recently pushed entry for this item
            if self._entry_finder.get(item) == count:
                del self._entry_finder[item]
                return item, priority
            # Stale entry — skip it (lazy deletion)
        raise KeyError('Priority queue is empty')

    def __len__(self) -> int:
        return len(self._entry_finder)

    def __bool__(self) -> bool:
        return bool(self._entry_finder)

    def __contains__(self, item: Any) -> bool:
        return item in self._entry_finder


class Grid:
    """
    2D Grid representation for pathfinding.

    Supports:
    - Obstacle placement and querying
    - Integral image for efficient density computation
    - 4-connected and 8-connected neighborhoods
    """

    # Direction vectors for 4-connected grid
    DIRECTIONS_4 = [
        Point(0, 1),   # Up
        Point(0, -1),  # Down
        Point(1, 0),   # Right
        Point(-1, 0),  # Left
    ]

    # Direction vectors for 8-connected grid
    DIRECTIONS_8 = [
        Point(0, 1),   # Up
        Point(0, -1),  # Down
        Point(1, 0),   # Right
        Point(-1, 0),  # Left
        Point(1, 1),   # Up-Right
        Point(1, -1),  # Down-Right
        Point(-1, 1),  # Up-Left
        Point(-1, -1), # Down-Left
    ]

    def __init__(self, width: int, height: int, eight_connected: bool = True):
        """
        Initialize grid.

        Args:
            width: Grid width
            height: Grid height
            eight_connected: Use 8-connected grid (default True)
        """
        self.width = width
        self.height = height
        self.eight_connected = eight_connected

        # Obstacle grid: True = obstacle, False = free
        self._obstacles: np.ndarray = np.zeros((height, width), dtype=bool)

        # Integral image for density computation (computed on demand)
        self._integral_image: Optional[np.ndarray] = None
        self._integral_valid = False

        # Directions based on connectivity
        self.directions = self.DIRECTIONS_8 if eight_connected else self.DIRECTIONS_4

        # Edge costs (1.0 for cardinal, sqrt(2) for diagonal)
        self._diagonal_cost = 1.41421356237  # sqrt(2)

    def set_obstacle(self, x: int, y: int, is_obstacle: bool = True) -> None:
        """Set or clear obstacle at position."""
        if self.in_bounds(x, y):
            self._obstacles[y, x] = is_obstacle
            self._integral_valid = False

    def set_obstacles_from_array(self, obstacles: np.ndarray) -> None:
        """Set obstacles from a 2D boolean array."""
        if obstacles.shape == (self.height, self.width):
            self._obstacles = obstacles.astype(bool)
            self._integral_valid = False
        else:
            raise ValueError(f"Array shape {obstacles.shape} doesn't match grid ({self.height}, {self.width})")

    def is_obstacle(self, x: int, y: int) -> bool:
        """Check if position is an obstacle."""
        if not self.in_bounds(x, y):
            return True  # Out of bounds treated as obstacle
        return self._obstacles[y, x]

    def is_free(self, x: int, y: int) -> bool:
        """Check if position is free (not an obstacle)."""
        return self.in_bounds(x, y) and not self._obstacles[y, x]

    def in_bounds(self, x: int, y: int) -> bool:
        """Check if position is within grid bounds."""
        return 0 <= x < self.width and 0 <= y < self.height

    def get_neighbors(self, point: Point) -> List[Tuple[Point, float]]:
        """
        Get valid neighbors of a point with edge costs.

        Returns:
            List of (neighbor, cost) tuples
        """
        neighbors = []
        for i, d in enumerate(self.directions):
            nx, ny = point.x + d.x, point.y + d.y
            if self.is_free(nx, ny):
                # Diagonal moves cost sqrt(2)
                cost = self._diagonal_cost if i >= 4 else 1.0
                neighbors.append((Point(nx, ny), cost))
        return neighbors

    def get_neighbors_in_corridor(self, point: Point, corridor: np.ndarray) -> List[Tuple[Point, float]]:
        """
        Get valid neighbors within a corridor constraint.

        Args:
            point: Current point
            corridor: Boolean numpy mask (height × width); True = in corridor

        Returns:
            List of (neighbor, cost) tuples
        """
        neighbors = []
        for i, d in enumerate(self.directions):
            nx, ny = point.x + d.x, point.y + d.y
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if corridor[ny, nx] and not self._obstacles[ny, nx]:
                    cost = self._diagonal_cost if i >= 4 else 1.0
                    neighbors.append((Point(nx, ny), cost))
        return neighbors

    def compute_integral_image(self) -> None:
        """Compute integral image for efficient density queries.

        Uses numpy double-cumsum instead of a Python double-loop,
        reducing runtime from O(n) Python iterations to two fast
        numpy vectorised passes — ~1000x faster on large grids.
        """
        if self._integral_valid:
            return

        # Pad with a zero row/col so that boundary queries are seamless.
        padded = np.zeros((self.height + 1, self.width + 1), dtype=np.int32)
        padded[1:, 1:] = self._obstacles.astype(np.int32)
        self._integral_image = np.cumsum(np.cumsum(padded, axis=0), axis=1)

        self._integral_valid = True

    def get_density_in_window(self, x: int, y: int, window_size: int) -> float:
        """
        Get obstacle density in a window centered at (x, y) using integral image.

        Args:
            x, y: Center of window
            window_size: Half-size of window (full size = 2*window_size + 1)

        Returns:
            Density value between 0.0 and 1.0
        """
        if not self._integral_valid:
            self.compute_integral_image()

        # Compute window bounds with clamping
        x1 = max(0, x - window_size)
        y1 = max(0, y - window_size)
        x2 = min(self.width - 1, x + window_size)
        y2 = min(self.height - 1, y + window_size)

        # Query integral image
        obstacle_count = (
            self._integral_image[y2 + 1, x2 + 1] -
            self._integral_image[y1, x2 + 1] -
            self._integral_image[y2 + 1, x1] +
            self._integral_image[y1, x1]
        )

        window_area = (x2 - x1 + 1) * (y2 - y1 + 1)
        return obstacle_count / window_area if window_area > 0 else 0.0

    def get_obstacle_count(self) -> int:
        """Get total number of obstacles."""
        return int(np.sum(self._obstacles))

    def get_free_count(self) -> int:
        """Get total number of free cells."""
        return self.width * self.height - self.get_obstacle_count()

    def get_obstacle_density(self) -> float:
        """Get overall obstacle density."""
        return self.get_obstacle_count() / (self.width * self.height)

    def get_obstacles_array(self) -> np.ndarray:
        """Get a copy of the obstacles array."""
        return self._obstacles.copy()

    def get_free_cells(self) -> List[Point]:
        """Get list of all free cells (uses numpy for speed)."""
        ys, xs = np.where(~self._obstacles)
        return [Point(int(x), int(y)) for x, y in zip(xs, ys)]

    def get_free_cells_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (xs, ys) numpy arrays of free-cell coordinates.

        Much cheaper than get_free_cells() for large grids because it
        avoids creating millions of Python Point objects.
        """
        ys, xs = np.where(~self._obstacles)
        return xs, ys

    def copy(self) -> 'Grid':
        """Create a deep copy of the grid."""
        new_grid = Grid(self.width, self.height, self.eight_connected)
        new_grid._obstacles = self._obstacles.copy()
        new_grid._integral_valid = False
        return new_grid

    def __repr__(self) -> str:
        return f"Grid({self.width}x{self.height}, obstacles={self.get_obstacle_count()})"


@dataclass
class PathResult:
    """
    Result of a pathfinding operation.

    Attributes:
        path: List of points from start to goal (empty if no path)
        found: Whether a path was found
        cost: Total path cost
        visited_nodes: Number of nodes expanded during search
        execution_time_ms: Time taken in milliseconds
        corridor_size: Size of the search corridor (AILS only)
        corridor_efficiency: Ratio of corridor to grid size (AILS only)
        expansions: Number of corridor expansions needed (AILS only)
        strategy_used: Which AILS strategy was used
        algorithm: Name of the algorithm used
    """
    path: List[Point] = field(default_factory=list)
    found: bool = False
    cost: float = float('inf')
    visited_nodes: int = 0
    execution_time_ms: float = 0.0
    corridor_size: int = 0
    corridor_efficiency: float = 0.0
    expansions: int = 0
    strategy_used: str = ""
    algorithm: str = ""

    def path_length(self) -> int:
        """Get the number of points in the path."""
        return len(self.path)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'found': self.found,
            'cost': self.cost,
            'path_length': self.path_length(),
            'visited_nodes': self.visited_nodes,
            'execution_time_ms': self.execution_time_ms,
            'corridor_size': self.corridor_size,
            'corridor_efficiency': self.corridor_efficiency,
            'expansions': self.expansions,
            'strategy_used': self.strategy_used,
            'algorithm': self.algorithm,
        }
