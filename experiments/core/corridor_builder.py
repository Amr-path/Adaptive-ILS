"""
Corridor Builder for AILS Algorithm
=====================================

Implements the adaptive corridor construction mechanism that is the
core innovation of the AILS framework.

Key Features:
- Bresenham line generation for reference path
- Per-point local density estimation using integral images
- Adaptive corridor radius based on density
- Gradient-based predictive expansion
- Multi-strategy selection
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Set, Tuple, Optional
import numpy as np
from .data_structures import Grid, Point


class CorridorStrategy(Enum):
    """Corridor construction strategies."""
    BASE = "base"              # Fixed-width corridor (r_min only)
    DENSITY_ADAPTIVE = "density_adaptive"  # Standard density-based
    GRADIENT_BASED = "gradient_based"      # Predictive with gradients


@dataclass
class CorridorConfig:
    """Configuration for corridor construction."""
    r_min: int = 2           # Minimum corridor radius
    r_max: int = 15          # Maximum corridor radius
    alpha: float = 1.0       # Density exponent for radius computation
    beta: float = 0.3        # Gradient sensitivity factor
    window_size: int = 5     # Window size for density estimation
    strategy: CorridorStrategy = CorridorStrategy.DENSITY_ADAPTIVE


class CorridorBuilder:
    """
    Builds adaptive corridors for constrained pathfinding.

    The corridor is constructed in four phases:
    1. Generate reference path using Bresenham line
    2. Estimate local density at each point
    3. Compute adaptive radius based on density
    4. Build corridor by expanding around reference path
    """

    def __init__(self, grid: Grid, config: Optional[CorridorConfig] = None):
        """
        Initialize corridor builder.

        Args:
            grid: The grid to build corridors on
            config: Corridor configuration (uses defaults if None)
        """
        self.grid = grid
        self.config = config or CorridorConfig()

        # Ensure integral image is computed
        self.grid.compute_integral_image()

    def bresenham_line(self, start: Point, goal: Point) -> List[Point]:
        """
        Generate a reference path using Bresenham's line algorithm.

        This produces a connected sequence of grid cells from start to goal
        using only integer arithmetic.

        Args:
            start: Starting point
            goal: Goal point

        Returns:
            List of points along the line
        """
        points = []

        x0, y0 = start.x, start.y
        x1, y1 = goal.x, goal.y

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            points.append(Point(x0, y0))

            if x0 == x1 and y0 == y1:
                break

            e2 = 2 * err

            if e2 > -dy:
                err -= dy
                x0 += sx

            if e2 < dx:
                err += dx
                y0 += sy

        return points

    def compute_local_density(self, point: Point) -> float:
        """
        Compute local obstacle density around a point.

        Uses integral image for O(1) query complexity.

        Args:
            point: Center point for density estimation

        Returns:
            Density value between 0.0 and 1.0
        """
        return self.grid.get_density_in_window(
            point.x, point.y, self.config.window_size
        )

    def compute_density_gradient(self, point: Point, prev_density: float) -> float:
        """
        Compute gradient of density change.

        Uses forward difference approximation.

        Args:
            point: Current point
            prev_density: Density at previous point

        Returns:
            Gradient magnitude (always non-negative)
        """
        current_density = self.compute_local_density(point)
        return abs(current_density - prev_density)

    def compute_adaptive_radius(self, density: float, gradient: float = 0.0) -> int:
        """
        Compute adaptive corridor radius based on local density.

        Formula (standard): r(p) = r_min + floor((r_max - r_min) * density^alpha)
        Formula (gradient): r(p) = r_min + floor((r_max - r_min) * (density + beta * gradient)^alpha)

        Args:
            density: Local obstacle density (0.0 to 1.0)
            gradient: Density gradient magnitude (optional)

        Returns:
            Corridor radius for this point
        """
        r_min = self.config.r_min
        r_max = self.config.r_max
        alpha = self.config.alpha
        beta = self.config.beta

        if self.config.strategy == CorridorStrategy.BASE:
            return r_min

        if self.config.strategy == CorridorStrategy.GRADIENT_BASED:
            effective_density = min(1.0, density + beta * gradient)
        else:
            effective_density = density

        # Apply power function and compute radius
        radius = r_min + int((r_max - r_min) * (effective_density ** alpha))
        return min(max(radius, r_min), r_max)

    def build_corridor(self, start: Point, goal: Point) -> Tuple[Set[Point], List[int], CorridorStrategy]:
        """
        Build an adaptive corridor from start to goal.

        Returns:
            Tuple of:
            - Set of points in the corridor
            - List of radii used at each reference point
            - Strategy that was used
        """
        # Generate reference path
        reference_path = self.bresenham_line(start, goal)

        # Check if reference path is obstacle-free (use BASE strategy)
        path_clear = all(self.grid.is_free(p.x, p.y) for p in reference_path)

        if path_clear and self.config.strategy != CorridorStrategy.BASE:
            # Temporarily use base strategy for clear paths
            actual_strategy = CorridorStrategy.BASE
        else:
            actual_strategy = self.config.strategy

        # Compute densities and radii for each point
        corridor = set()
        radii = []
        prev_density = 0.0

        for i, point in enumerate(reference_path):
            density = self.compute_local_density(point)

            if actual_strategy == CorridorStrategy.GRADIENT_BASED and i > 0:
                gradient = abs(density - prev_density)
            else:
                gradient = 0.0

            if actual_strategy == CorridorStrategy.BASE:
                radius = self.config.r_min
            else:
                radius = self.compute_adaptive_radius(density, gradient)

            radii.append(radius)

            # Expand corridor around this point
            self._expand_corridor_at_point(point, radius, corridor)

            prev_density = density

        return corridor, radii, actual_strategy

    def _expand_corridor_at_point(self, center: Point, radius: int, corridor: Set[Point]) -> None:
        """
        Expand corridor around a center point with given radius.

        Uses a circular expansion pattern.

        Args:
            center: Center point
            radius: Expansion radius
            corridor: Set to add points to (modified in place)
        """
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                # Circular expansion (use squared distance to avoid sqrt)
                if dx * dx + dy * dy <= radius * radius:
                    nx, ny = center.x + dx, center.y + dy
                    if self.grid.in_bounds(nx, ny):
                        corridor.add(Point(nx, ny))

    def expand_corridor(self, corridor: Set[Point], expansion_amount: int = 1) -> Set[Point]:
        """
        Expand an existing corridor by a fixed amount.

        Used for fallback when initial corridor doesn't contain a valid path.

        Args:
            corridor: Existing corridor
            expansion_amount: How much to expand

        Returns:
            Expanded corridor (new set)
        """
        expanded = set(corridor)

        for point in corridor:
            for dy in range(-expansion_amount, expansion_amount + 1):
                for dx in range(-expansion_amount, expansion_amount + 1):
                    nx, ny = point.x + dx, point.y + dy
                    if self.grid.in_bounds(nx, ny):
                        expanded.add(Point(nx, ny))

        return expanded

    def get_corridor_stats(self, corridor: Set[Point]) -> dict:
        """
        Get statistics about a corridor.

        Args:
            corridor: The corridor to analyze

        Returns:
            Dictionary with corridor statistics
        """
        total_cells = self.grid.width * self.grid.height
        free_in_corridor = sum(1 for p in corridor if self.grid.is_free(p.x, p.y))

        return {
            'total_cells': len(corridor),
            'free_cells': free_in_corridor,
            'efficiency': len(corridor) / total_cells,
            'free_ratio': free_in_corridor / len(corridor) if corridor else 0.0,
        }


def create_corridor_builder(grid: Grid,
                            r_min: int = 2,
                            r_max: int = 15,
                            alpha: float = 1.0,
                            beta: float = 0.3,
                            window_size: int = 5,
                            strategy: str = "density_adaptive") -> CorridorBuilder:
    """
    Factory function to create a corridor builder.

    Args:
        grid: The grid
        r_min: Minimum radius
        r_max: Maximum radius
        alpha: Density exponent
        beta: Gradient sensitivity
        window_size: Density window size
        strategy: Strategy name ("base", "density_adaptive", "gradient_based")

    Returns:
        Configured CorridorBuilder
    """
    strategy_map = {
        "base": CorridorStrategy.BASE,
        "density_adaptive": CorridorStrategy.DENSITY_ADAPTIVE,
        "gradient_based": CorridorStrategy.GRADIENT_BASED,
    }

    config = CorridorConfig(
        r_min=r_min,
        r_max=r_max,
        alpha=alpha,
        beta=beta,
        window_size=window_size,
        strategy=strategy_map.get(strategy, CorridorStrategy.DENSITY_ADAPTIVE)
    )

    return CorridorBuilder(grid, config)
