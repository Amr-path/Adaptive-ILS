"""
Heuristic Functions for Pathfinding
====================================

Various admissible heuristics for A* and related algorithms.
All heuristics are consistent (monotonic) for grid-based pathfinding.
"""

import math
from typing import Tuple, Union
from .data_structures import Point


def manhattan(p1: Union[Point, Tuple[int, int]], p2: Union[Point, Tuple[int, int]]) -> float:
    """
    Manhattan (L1) distance heuristic.

    Admissible and consistent for 4-connected grids.
    For 8-connected grids, this is an overestimate.

    Args:
        p1: First point
        p2: Second point

    Returns:
        Manhattan distance
    """
    if isinstance(p1, Point):
        x1, y1 = p1.x, p1.y
    else:
        x1, y1 = p1

    if isinstance(p2, Point):
        x2, y2 = p2.x, p2.y
    else:
        x2, y2 = p2

    return abs(x1 - x2) + abs(y1 - y2)


def euclidean(p1: Union[Point, Tuple[int, int]], p2: Union[Point, Tuple[int, int]]) -> float:
    """
    Euclidean (L2) distance heuristic.

    Admissible for both 4-connected and 8-connected grids.
    Slightly slower to compute than Manhattan or Chebyshev.

    Args:
        p1: First point
        p2: Second point

    Returns:
        Euclidean distance
    """
    if isinstance(p1, Point):
        x1, y1 = p1.x, p1.y
    else:
        x1, y1 = p1

    if isinstance(p2, Point):
        x2, y2 = p2.x, p2.y
    else:
        x2, y2 = p2

    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx * dx + dy * dy)


def chebyshev(p1: Union[Point, Tuple[int, int]], p2: Union[Point, Tuple[int, int]]) -> float:
    """
    Chebyshev (L-infinity) distance heuristic.

    Optimal for 8-connected grids with uniform costs.
    Underestimates for grids with diagonal cost > 1.

    Args:
        p1: First point
        p2: Second point

    Returns:
        Chebyshev distance
    """
    if isinstance(p1, Point):
        x1, y1 = p1.x, p1.y
    else:
        x1, y1 = p1

    if isinstance(p2, Point):
        x2, y2 = p2.x, p2.y
    else:
        x2, y2 = p2

    return max(abs(x1 - x2), abs(y1 - y2))


def octile(p1: Union[Point, Tuple[int, int]], p2: Union[Point, Tuple[int, int]]) -> float:
    """
    Octile distance heuristic.

    The optimal heuristic for 8-connected grids with diagonal cost sqrt(2).
    This is the most accurate heuristic for standard 8-way movement.

    Formula: max(dx, dy) + (sqrt(2) - 1) * min(dx, dy)

    Args:
        p1: First point
        p2: Second point

    Returns:
        Octile distance
    """
    if isinstance(p1, Point):
        x1, y1 = p1.x, p1.y
    else:
        x1, y1 = p1

    if isinstance(p2, Point):
        x2, y2 = p2.x, p2.y
    else:
        x2, y2 = p2

    dx = abs(x1 - x2)
    dy = abs(y1 - y2)

    # Diagonal cost factor: sqrt(2) - 1 ≈ 0.414
    SQRT2_MINUS_1 = 0.41421356237

    return max(dx, dy) + SQRT2_MINUS_1 * min(dx, dy)


def diagonal(p1: Union[Point, Tuple[int, int]], p2: Union[Point, Tuple[int, int]],
             d1: float = 1.0, d2: float = 1.41421356237) -> float:
    """
    Generalized diagonal distance heuristic.

    Allows custom costs for cardinal (d1) and diagonal (d2) moves.

    Args:
        p1: First point
        p2: Second point
        d1: Cost of cardinal move (default 1.0)
        d2: Cost of diagonal move (default sqrt(2))

    Returns:
        Diagonal distance with custom costs
    """
    if isinstance(p1, Point):
        x1, y1 = p1.x, p1.y
    else:
        x1, y1 = p1

    if isinstance(p2, Point):
        x2, y2 = p2.x, p2.y
    else:
        x2, y2 = p2

    dx = abs(x1 - x2)
    dy = abs(y1 - y2)

    return d1 * (dx + dy) + (d2 - 2 * d1) * min(dx, dy)


def zero_heuristic(p1: Union[Point, Tuple[int, int]], p2: Union[Point, Tuple[int, int]]) -> float:
    """
    Zero heuristic (for Dijkstra's algorithm).

    Always returns 0, making A* equivalent to Dijkstra.

    Args:
        p1: First point (unused)
        p2: Second point (unused)

    Returns:
        Always 0.0
    """
    return 0.0


# Heuristic registry for easy access
HEURISTICS = {
    'manhattan': manhattan,
    'euclidean': euclidean,
    'chebyshev': chebyshev,
    'octile': octile,
    'diagonal': diagonal,
    'zero': zero_heuristic,
}


def get_heuristic(name: str):
    """
    Get a heuristic function by name.

    Args:
        name: Name of the heuristic

    Returns:
        Heuristic function

    Raises:
        ValueError: If heuristic name is unknown
    """
    if name not in HEURISTICS:
        raise ValueError(f"Unknown heuristic: {name}. Available: {list(HEURISTICS.keys())}")
    return HEURISTICS[name]
