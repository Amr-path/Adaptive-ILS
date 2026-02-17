"""
Synthetic Map Generators
=========================

Generators for creating various obstacle patterns for benchmarking.

Pattern Types:
- Random: Uniformly distributed obstacles
- Maze: Maze-like structures with corridors
- Clustered: Grouped obstacle clusters
- Open: Sparse obstacles with large open areas
- Mixed: Combination of patterns
- Room: Room-like structures with walls
- Corridor: Long narrow passages
"""

import numpy as np
from typing import Tuple, Optional, List
import random
import sys
sys.path.insert(0, '..')
from core.data_structures import Grid, Point


class MapGenerator:
    """
    Generator for synthetic benchmark maps.

    Supports various obstacle patterns and densities for
    comprehensive algorithm evaluation.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def generate(self, width: int, height: int,
                 pattern: str = "random",
                 density: float = 0.2,
                 **kwargs) -> Grid:
        """
        Generate a map with specified pattern.

        Args:
            width: Map width
            height: Map height
            pattern: Pattern type ("random", "maze", "clustered", "open", "mixed", "room", "corridor")
            density: Obstacle density (0.0 to 1.0)
            **kwargs: Pattern-specific parameters

        Returns:
            Generated Grid
        """
        pattern_funcs = {
            "random": self._generate_random,
            "maze": self._generate_maze,
            "clustered": self._generate_clustered,
            "open": self._generate_open,
            "mixed": self._generate_mixed,
            "room": self._generate_room,
            "corridor": self._generate_corridor,
        }

        if pattern not in pattern_funcs:
            raise ValueError(f"Unknown pattern: {pattern}. Available: {list(pattern_funcs.keys())}")

        grid = Grid(width, height, eight_connected=True)
        obstacles = pattern_funcs[pattern](width, height, density, **kwargs)
        grid.set_obstacles_from_array(obstacles)

        return grid

    def _generate_random(self, width: int, height: int, density: float, **kwargs) -> np.ndarray:
        """Generate uniformly random obstacles."""
        return np.random.random((height, width)) < density

    def _generate_maze(self, width: int, height: int, density: float,
                       wall_thickness: int = 1, passage_width: int = 2, **kwargs) -> np.ndarray:
        """
        Generate maze-like structure using recursive backtracking.

        Args:
            wall_thickness: Thickness of maze walls
            passage_width: Width of passages
        """
        obstacles = np.ones((height, width), dtype=bool)

        # Adjusted cell size
        cell_size = wall_thickness + passage_width

        # Calculate maze dimensions
        maze_height = (height - wall_thickness) // cell_size
        maze_width = (width - wall_thickness) // cell_size

        if maze_height < 2 or maze_width < 2:
            # Fallback to random if too small
            return self._generate_random(width, height, density)

        # Create maze using recursive backtracking
        maze = np.zeros((maze_height, maze_width), dtype=bool)
        visited = np.zeros((maze_height, maze_width), dtype=bool)

        stack = [(0, 0)]
        visited[0, 0] = True

        while stack:
            cy, cx = stack[-1]

            # Get unvisited neighbors
            neighbors = []
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < maze_height and 0 <= nx < maze_width and not visited[ny, nx]:
                    neighbors.append((ny, nx, dy, dx))

            if neighbors:
                # Choose random neighbor
                ny, nx, dy, dx = random.choice(neighbors)
                visited[ny, nx] = True
                stack.append((ny, nx))

                # Carve passage in obstacles array
                for py in range(passage_width):
                    for px in range(passage_width):
                        oy = wall_thickness + cy * cell_size + py
                        ox = wall_thickness + cx * cell_size + px
                        if 0 <= oy < height and 0 <= ox < width:
                            obstacles[oy, ox] = False

                # Carve wall between cells
                wy = wall_thickness + cy * cell_size + (dy * passage_width if dy > 0 else 0)
                wx = wall_thickness + cx * cell_size + (dx * passage_width if dx > 0 else 0)
                for py in range(passage_width):
                    for px in range(passage_width):
                        oy = wy + py
                        ox = wx + px
                        if 0 <= oy < height and 0 <= ox < width:
                            obstacles[oy, ox] = False
            else:
                stack.pop()

        # Carve final cell passages
        for cy in range(maze_height):
            for cx in range(maze_width):
                for py in range(passage_width):
                    for px in range(passage_width):
                        oy = wall_thickness + cy * cell_size + py
                        ox = wall_thickness + cx * cell_size + px
                        if 0 <= oy < height and 0 <= ox < width:
                            obstacles[oy, ox] = False

        return obstacles

    def _generate_clustered(self, width: int, height: int, density: float,
                            num_clusters: Optional[int] = None,
                            cluster_size: Optional[int] = None, **kwargs) -> np.ndarray:
        """
        Generate clustered obstacles using Gaussian blobs.

        Args:
            num_clusters: Number of obstacle clusters
            cluster_size: Average cluster radius
        """
        obstacles = np.zeros((height, width), dtype=bool)

        if num_clusters is None:
            num_clusters = max(3, int(density * 20))

        if cluster_size is None:
            cluster_size = min(width, height) // 8

        total_area = width * height
        target_obstacles = int(total_area * density)
        current_obstacles = 0

        for _ in range(num_clusters):
            if current_obstacles >= target_obstacles:
                break

            # Random cluster center
            cx = np.random.randint(0, width)
            cy = np.random.randint(0, height)

            # Cluster radius with some variance
            radius = max(1, int(cluster_size * (0.5 + np.random.random())))

            # Fill cluster
            for y in range(max(0, cy - radius), min(height, cy + radius + 1)):
                for x in range(max(0, cx - radius), min(width, cx + radius + 1)):
                    dist_sq = (x - cx) ** 2 + (y - cy) ** 2
                    if dist_sq <= radius ** 2:
                        # Probability decreases with distance from center
                        prob = 1.0 - (dist_sq / (radius ** 2))
                        if np.random.random() < prob and not obstacles[y, x]:
                            obstacles[y, x] = True
                            current_obstacles += 1

        return obstacles

    def _generate_open(self, width: int, height: int, density: float,
                       min_gap: int = 10, **kwargs) -> np.ndarray:
        """
        Generate sparse obstacles with guaranteed open areas.

        Args:
            min_gap: Minimum gap between obstacle regions
        """
        obstacles = np.zeros((height, width), dtype=bool)

        # Divide into regions
        region_size = min_gap * 3
        regions_y = max(1, height // region_size)
        regions_x = max(1, width // region_size)

        for ry in range(regions_y):
            for rx in range(regions_x):
                # Random chance to have obstacles in this region
                if np.random.random() < density * 2:
                    # Place a small obstacle group
                    cy = ry * region_size + region_size // 2
                    cx = rx * region_size + region_size // 2

                    # Small cluster
                    radius = np.random.randint(2, min_gap // 2)
                    for y in range(max(0, cy - radius), min(height, cy + radius + 1)):
                        for x in range(max(0, cx - radius), min(width, cx + radius + 1)):
                            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                                if np.random.random() < 0.8:
                                    obstacles[y, x] = True

        return obstacles

    def _generate_mixed(self, width: int, height: int, density: float, **kwargs) -> np.ndarray:
        """Generate a mix of different patterns."""
        obstacles = np.zeros((height, width), dtype=bool)

        # Divide into quadrants with different patterns
        mid_y, mid_x = height // 2, width // 2

        # Top-left: Random
        random_obs = self._generate_random(mid_x, mid_y, density)
        obstacles[:mid_y, :mid_x] = random_obs

        # Top-right: Clustered
        clustered_obs = self._generate_clustered(width - mid_x, mid_y, density)
        obstacles[:mid_y, mid_x:] = clustered_obs

        # Bottom-left: Open
        open_obs = self._generate_open(mid_x, height - mid_y, density)
        obstacles[mid_y:, :mid_x] = open_obs

        # Bottom-right: More random obstacles
        random_obs2 = self._generate_random(width - mid_x, height - mid_y, density * 1.2)
        obstacles[mid_y:, mid_x:] = random_obs2

        return obstacles

    def _generate_room(self, width: int, height: int, density: float,
                       room_size: int = 20, door_width: int = 3, **kwargs) -> np.ndarray:
        """
        Generate room-like structures with walls and doorways.

        Args:
            room_size: Approximate size of rooms
            door_width: Width of doorways
        """
        obstacles = np.zeros((height, width), dtype=bool)

        # Create walls
        wall_positions_x = list(range(room_size, width - room_size // 2, room_size))
        wall_positions_y = list(range(room_size, height - room_size // 2, room_size))

        # Vertical walls
        for wx in wall_positions_x:
            for y in range(height):
                obstacles[y, wx] = True

            # Add doors
            num_doors = max(1, len(wall_positions_y) + 1)
            for i in range(num_doors):
                door_y = np.random.randint(door_width, height - door_width)
                for dy in range(-door_width // 2, door_width // 2 + 1):
                    if 0 <= door_y + dy < height:
                        obstacles[door_y + dy, wx] = False

        # Horizontal walls
        for wy in wall_positions_y:
            for x in range(width):
                obstacles[wy, x] = True

            # Add doors
            num_doors = max(1, len(wall_positions_x) + 1)
            for i in range(num_doors):
                door_x = np.random.randint(door_width, width - door_width)
                for dx in range(-door_width // 2, door_width // 2 + 1):
                    if 0 <= door_x + dx < width:
                        obstacles[wy, door_x + dx] = False

        # Add some random obstacles inside rooms based on density
        random_obs = self._generate_random(width, height, density * 0.3)
        obstacles = obstacles | random_obs

        return obstacles

    def _generate_corridor(self, width: int, height: int, density: float,
                           num_corridors: int = 5, corridor_width: int = 4, **kwargs) -> np.ndarray:
        """
        Generate maps with long narrow passages.

        Args:
            num_corridors: Number of corridors
            corridor_width: Width of corridors
        """
        # Start with filled
        obstacles = np.ones((height, width), dtype=bool)

        for _ in range(num_corridors):
            # Random corridor
            if np.random.random() < 0.5:
                # Horizontal corridor
                y = np.random.randint(corridor_width, height - corridor_width)
                for dy in range(-corridor_width // 2, corridor_width // 2 + 1):
                    if 0 <= y + dy < height:
                        obstacles[y + dy, :] = False
            else:
                # Vertical corridor
                x = np.random.randint(corridor_width, width - corridor_width)
                for dx in range(-corridor_width // 2, corridor_width // 2 + 1):
                    if 0 <= x + dx < width:
                        obstacles[:, x + dx] = False

        # Add connecting passages
        for _ in range(num_corridors * 2):
            x = np.random.randint(corridor_width, width - corridor_width)
            y = np.random.randint(corridor_width, height - corridor_width)
            length = np.random.randint(10, max(width, height) // 2)

            if np.random.random() < 0.5:
                # Horizontal
                for i in range(length):
                    if 0 <= x + i < width:
                        for dy in range(-corridor_width // 4, corridor_width // 4 + 1):
                            if 0 <= y + dy < height:
                                obstacles[y + dy, x + i] = False
            else:
                # Vertical
                for i in range(length):
                    if 0 <= y + i < height:
                        for dx in range(-corridor_width // 4, corridor_width // 4 + 1):
                            if 0 <= x + dx < width:
                                obstacles[y + i, x + dx] = False

        return obstacles

    def generate_valid_endpoints(self, grid: Grid,
                                 min_distance: Optional[int] = None,
                                 num_pairs: int = 1) -> List[Tuple[Point, Point]]:
        """
        Generate valid start-goal pairs.

        Uses numpy index-based sampling instead of a Python list of Point
        objects so it stays fast even for 5 000×5 000+ grids (where the
        old approach would create ~20 M Python objects).

        Args:
            grid: The grid to generate endpoints for
            min_distance: Minimum Manhattan distance between endpoints
            num_pairs: Number of pairs to generate

        Returns:
            List of (start, goal) point pairs
        """
        if min_distance is None:
            min_distance = (grid.width + grid.height) // 4

        # --- Fast numpy path ---
        free_xs, free_ys = grid.get_free_cells_arrays()
        n_free = len(free_xs)
        if n_free < 2:
            raise ValueError("Grid has fewer than 2 free cells")

        rng = np.random.default_rng(self.seed)

        pairs: List[Tuple[Point, Point]] = []
        max_attempts = num_pairs * 200

        for _ in range(max_attempts):
            if len(pairs) >= num_pairs:
                break
            i, j = rng.integers(0, n_free, size=2)
            if i == j:
                continue
            sx, sy = int(free_xs[i]), int(free_ys[i])
            gx, gy = int(free_xs[j]), int(free_ys[j])
            if abs(sx - gx) + abs(sy - gy) >= min_distance:
                pairs.append((Point(sx, sy), Point(gx, gy)))

        # Relaxed-constraint fallback if we couldn't satisfy min_distance
        if len(pairs) < num_pairs:
            for _ in range((num_pairs - len(pairs)) * 50):
                if len(pairs) >= num_pairs:
                    break
                i, j = rng.integers(0, n_free, size=2)
                if i != j:
                    pairs.append((
                        Point(int(free_xs[i]), int(free_ys[i])),
                        Point(int(free_xs[j]), int(free_ys[j])),
                    ))

        return pairs[:num_pairs]


# Convenience functions
def generate_random_map(width: int, height: int, density: float = 0.2,
                        seed: Optional[int] = None) -> Grid:
    """Generate a random obstacle map."""
    return MapGenerator(seed).generate(width, height, "random", density)


def generate_maze_map(width: int, height: int, density: float = 0.3,
                      seed: Optional[int] = None) -> Grid:
    """Generate a maze-like map."""
    return MapGenerator(seed).generate(width, height, "maze", density)


def generate_clustered_map(width: int, height: int, density: float = 0.2,
                           seed: Optional[int] = None) -> Grid:
    """Generate a map with clustered obstacles."""
    return MapGenerator(seed).generate(width, height, "clustered", density)


def generate_open_map(width: int, height: int, density: float = 0.1,
                      seed: Optional[int] = None) -> Grid:
    """Generate an open map with sparse obstacles."""
    return MapGenerator(seed).generate(width, height, "open", density)


def generate_mixed_map(width: int, height: int, density: float = 0.2,
                       seed: Optional[int] = None) -> Grid:
    """Generate a map with mixed obstacle patterns."""
    return MapGenerator(seed).generate(width, height, "mixed", density)


def generate_room_map(width: int, height: int, density: float = 0.15,
                      seed: Optional[int] = None) -> Grid:
    """Generate a map with room-like structures."""
    return MapGenerator(seed).generate(width, height, "room", density)


def generate_corridor_map(width: int, height: int, density: float = 0.4,
                          seed: Optional[int] = None) -> Grid:
    """Generate a map with corridor patterns."""
    return MapGenerator(seed).generate(width, height, "corridor", density)
