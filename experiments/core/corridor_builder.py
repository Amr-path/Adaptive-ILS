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
from typing import List, Tuple, Optional
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

    # Cache of precomputed disk offsets per radius: radius -> (dy_array, dx_array)
    _disk_offset_cache: dict = {}

    def _get_disk_offsets(self, radius: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (dy, dx) offset arrays for a filled circle of given radius.

        Results are cached so each radius is computed at most once per
        CorridorBuilder instance (and shared across all instances via the
        class-level dict).
        """
        if radius not in CorridorBuilder._disk_offset_cache:
            r = radius
            ys, xs = np.mgrid[-r:r + 1, -r:r + 1]
            mask = xs * xs + ys * ys <= r * r
            CorridorBuilder._disk_offset_cache[radius] = (
                ys[mask].astype(np.int32),
                xs[mask].astype(np.int32),
            )
        return CorridorBuilder._disk_offset_cache[radius]

    def build_corridor(self, start: Point, goal: Point) -> Tuple[np.ndarray, List[int], CorridorStrategy]:
        """
        Build an adaptive corridor from start to goal.

        Returns:
            Tuple of:
            - Boolean numpy mask (height × width); True = inside corridor
            - List of radii used at each reference point
            - Strategy that was used
        """
        h, w = self.grid.height, self.grid.width

        # Generate reference path
        reference_path = self.bresenham_line(start, goal)

        # Check if reference path is obstacle-free (use BASE strategy)
        path_clear = all(self.grid.is_free(p.x, p.y) for p in reference_path)

        if path_clear and self.config.strategy != CorridorStrategy.BASE:
            actual_strategy = CorridorStrategy.BASE
        else:
            actual_strategy = self.config.strategy

        # Numpy boolean mask for the corridor
        corridor = np.zeros((h, w), dtype=bool)
        radii = []
        prev_density = 0.0

        for i, point in enumerate(reference_path):
            density = self.compute_local_density(point)

            if actual_strategy == CorridorStrategy.GRADIENT_BASED and i > 0:
                gradient = abs(density - prev_density)
            else:
                gradient = 0.0

            radius = self.config.r_min if actual_strategy == CorridorStrategy.BASE \
                else self.compute_adaptive_radius(density, gradient)

            radii.append(radius)
            self._stamp_disk(corridor, point.x, point.y, radius)
            prev_density = density

        # Always include start and goal
        corridor[start.y, start.x] = True
        corridor[goal.y, goal.x] = True

        return corridor, radii, actual_strategy

    def _stamp_disk(self, corridor: np.ndarray, cx: int, cy: int, radius: int) -> None:
        """Paint a filled circle of radius *radius* centred at (cx, cy) into *corridor*.

        Uses precomputed offset arrays and numpy fancy-indexing — much faster
        than the old Python double-loop + set.add() approach.
        """
        h, w = corridor.shape
        dy_offsets, dx_offsets = self._get_disk_offsets(radius)

        ys = cy + dy_offsets
        xs = cx + dx_offsets

        # Clip to grid bounds
        valid = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
        corridor[ys[valid], xs[valid]] = True

    def expand_corridor(self, corridor: np.ndarray, expansion_amount: int = 1) -> np.ndarray:
        """
        Expand an existing corridor mask by *expansion_amount* cells.

        Uses binary dilation via scipy when available, falling back to a
        numpy-only implementation so there is no hard scipy dependency.

        Args:
            corridor: Boolean numpy mask (height × width)
            expansion_amount: Dilation radius in cells

        Returns:
            New expanded boolean numpy mask
        """
        try:
            from scipy.ndimage import binary_dilation
            r = expansion_amount
            dy, dx = self._get_disk_offsets(r)
            struct = np.zeros((2 * r + 1, 2 * r + 1), dtype=bool)
            struct[dy + r, dx + r] = True
            return binary_dilation(corridor, structure=struct)
        except ImportError:
            # Pure-numpy fallback: expand by stamping each True cell
            expanded = corridor.copy()
            ys, xs = np.where(corridor)
            for cy, cx in zip(ys.tolist(), xs.tolist()):
                self._stamp_disk(expanded, int(cx), int(cy), expansion_amount)
            return expanded

    def get_corridor_stats(self, corridor: np.ndarray) -> dict:
        """
        Get statistics about a corridor.

        Args:
            corridor: Boolean numpy mask (height × width)

        Returns:
            Dictionary with corridor statistics
        """
        total_cells = self.grid.width * self.grid.height
        corridor_cells = int(np.count_nonzero(corridor))
        free_in_corridor = int(np.count_nonzero(corridor & ~self.grid._obstacles))

        return {
            'total_cells': corridor_cells,
            'free_cells': free_in_corridor,
            'efficiency': corridor_cells / total_cells,
            'free_ratio': free_in_corridor / corridor_cells if corridor_cells else 0.0,
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
