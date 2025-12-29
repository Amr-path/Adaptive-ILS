"""
Adaptive Incremental Line Search (AILS): A Comprehensive Research Framework
============================================================================
Research-Grade Implementation for Q1 Publication
Authors: [Your Names]
Institution: [Your Institution]
Date: 2024

This notebook provides a comprehensive experimental framework for evaluating
the Adaptive Incremental Line Search (AILS) algorithm against classical
pathfinding algorithms across diverse scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
import heapq
from collections import deque, defaultdict
import time
import random
from typing import List, Tuple, Optional, Dict, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import pickle
import json
import warnings
import os
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import itertools
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import networkx as nx

warnings.filterwarnings('ignore')

# Set publication-quality plot parameters
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# Create output directories
os.makedirs('results', exist_ok=True)
os.makedirs('results/data', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/analysis', exist_ok=True)

# ============================================================================
# EXPERIMENTAL CONFIGURATION
# ============================================================================

@dataclass
class ExperimentConfig:
    """Comprehensive configuration for experiments"""
    
    # Grid configurations
    grid_sizes: List[int] = field(default_factory=lambda: [50, 100, 200, 300, 500])
    obstacle_densities: List[float] = field(default_factory=lambda: 
        [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
    
    # Trial configurations
    trials_per_config: int = 50
    random_seed: int = 42
    
    # Corridor parameters
    min_corridor_width: int = 1
    max_corridor_width_ratio: float = 0.1  # Ratio of grid size
    window_size: int = 7
    lookahead_distance: int = 5
    adaptive_threshold: float = 0.3  # Density threshold for expansion
    
    # ILS parameters
    ils_initial_width: int = 3
    ils_step_size: int = 2
    ils_max_iterations: int = 10
    
    # Performance parameters
    use_multiprocessing: bool = True
    num_workers: int = mp.cpu_count() - 1
    
    # Analysis parameters
    statistical_confidence: float = 0.95
    outlier_threshold: float = 3.0  # Standard deviations
    
    # Export parameters
    export_csv: bool = True
    export_pickle: bool = True
    generate_latex_tables: bool = True
    
    def get_max_corridor_width(self, grid_size: int) -> int:
        """Calculate maximum corridor width based on grid size"""
        return max(5, int(grid_size * self.max_corridor_width_ratio))

# ============================================================================
# ENHANCED GRID MAP GENERATION
# ============================================================================

class AdvancedGridGenerator:
    """Advanced grid map generation with multiple patterns"""
    
    class ObstaclePattern(Enum):
        RANDOM = "random"
        CLUSTERED = "clustered"
        MAZE = "maze"
        ROOMS = "rooms"
        MIXED = "mixed"
    
    @staticmethod
    def generate(size: int, density: float, pattern: 'ObstaclePattern' = None, 
                 seed: Optional[int] = None) -> np.ndarray:
        """Generate grid with specified pattern"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        if pattern is None:
            pattern = AdvancedGridGenerator.ObstaclePattern.RANDOM
        
        if pattern == AdvancedGridGenerator.ObstaclePattern.RANDOM:
            return AdvancedGridGenerator._generate_random(size, density)
        elif pattern == AdvancedGridGenerator.ObstaclePattern.CLUSTERED:
            return AdvancedGridGenerator._generate_clustered(size, density)
        elif pattern == AdvancedGridGenerator.ObstaclePattern.MAZE:
            return AdvancedGridGenerator._generate_maze(size, density)
        elif pattern == AdvancedGridGenerator.ObstaclePattern.ROOMS:
            return AdvancedGridGenerator._generate_rooms(size, density)
        else:  # MIXED
            return AdvancedGridGenerator._generate_mixed(size, density)
    
    @staticmethod
    def _generate_random(size: int, density: float) -> np.ndarray:
        """Generate random obstacles"""
        grid = np.random.random((size, size)) < density
        grid = grid.astype(int)
        grid[0, 0] = 0  # Clear start
        grid[size-1, size-1] = 0  # Clear goal
        
        # Ensure path exists
        if not AdvancedGridGenerator._path_exists(grid):
            grid = AdvancedGridGenerator._ensure_path(grid)
        
        return grid
    
    @staticmethod
    def _generate_clustered(size: int, density: float) -> np.ndarray:
        """Generate clustered obstacles"""
        grid = np.zeros((size, size), dtype=int)
        num_clusters = int(size * density)
        
        for _ in range(num_clusters):
            cx, cy = random.randint(0, size-1), random.randint(0, size-1)
            radius = random.randint(2, max(3, size//20))
            
            for i in range(max(0, cx-radius), min(size, cx+radius+1)):
                for j in range(max(0, cy-radius), min(size, cy+radius+1)):
                    if (i-cx)**2 + (j-cy)**2 <= radius**2:
                        if not ((i == 0 and j == 0) or (i == size-1 and j == size-1)):
                            grid[i, j] = 1
        
        if not AdvancedGridGenerator._path_exists(grid):
            grid = AdvancedGridGenerator._ensure_path(grid)
        
        return grid
    
    @staticmethod
    def _generate_maze(size: int, density: float) -> np.ndarray:
        """Generate maze-like pattern"""
        grid = np.zeros((size, size), dtype=int)
        
        # Create walls at regular intervals
        wall_spacing = max(3, int(10 * (1 - density)))
        
        for i in range(0, size, wall_spacing):
            if i > 0:
                grid[i, :] = 1
                grid[:, i] = 1
                
                # Create openings
                for j in range(wall_spacing//2, size, wall_spacing):
                    if j < size:
                        grid[i, j] = 0
                        grid[j, i] = 0
        
        grid[0, 0] = 0
        grid[size-1, size-1] = 0
        
        if not AdvancedGridGenerator._path_exists(grid):
            grid = AdvancedGridGenerator._ensure_path(grid)
        
        return grid
    
    @staticmethod
    def _generate_rooms(size: int, density: float) -> np.ndarray:
        """Generate room-like structure"""
        grid = np.zeros((size, size), dtype=int)
        
        num_rooms = int(4 + density * 10)
        room_size = size // int(np.sqrt(num_rooms))
        
        for i in range(0, size, room_size):
            for j in range(0, size, room_size):
                # Create room walls
                if i > 0:
                    grid[i, j:min(j+room_size, size)] = 1
                if j > 0:
                    grid[i:min(i+room_size, size), j] = 1
                
                # Create doorways
                if i > 0 and i < size - room_size:
                    door_pos = j + room_size//2
                    if door_pos < size:
                        grid[i, door_pos] = 0
                if j > 0 and j < size - room_size:
                    door_pos = i + room_size//2
                    if door_pos < size:
                        grid[door_pos, j] = 0
        
        grid[0, 0] = 0
        grid[size-1, size-1] = 0
        
        if not AdvancedGridGenerator._path_exists(grid):
            grid = AdvancedGridGenerator._ensure_path(grid)
        
        return grid
    
    @staticmethod
    def _generate_mixed(size: int, density: float) -> np.ndarray:
        """Generate mixed pattern combining multiple styles"""
        # Divide grid into quadrants with different patterns
        grid = np.zeros((size, size), dtype=int)
        half = size // 2
        
        # Random quadrant
        grid[:half, :half] = AdvancedGridGenerator._generate_random(half, density)[:half, :half]
        
        # Clustered quadrant
        grid[half:, :half] = AdvancedGridGenerator._generate_clustered(half, density)[:half, :half]
        
        # Maze quadrant
        grid[:half, half:] = AdvancedGridGenerator._generate_maze(half, density)[:half, :half]
        
        # Rooms quadrant
        grid[half:, half:] = AdvancedGridGenerator._generate_rooms(half, density)[:half, :half]
        
        grid[0, 0] = 0
        grid[size-1, size-1] = 0
        
        if not AdvancedGridGenerator._path_exists(grid):
            grid = AdvancedGridGenerator._ensure_path(grid)
        
        return grid
    
    @staticmethod
    def _path_exists(grid: np.ndarray) -> bool:
        """Check if path exists from top-left to bottom-right"""
        size = len(grid)
        visited = set()
        queue = deque([(0, 0)])
        visited.add((0, 0))
        
        while queue:
            x, y = queue.popleft()
            if x == size-1 and y == size-1:
                return True
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < size and 0 <= ny < size and 
                    grid[nx, ny] == 0 and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        return False
    
    @staticmethod
    def _ensure_path(grid: np.ndarray) -> np.ndarray:
        """Ensure a path exists by clearing a corridor"""
        size = len(grid)
        new_grid = grid.copy()
        
        # Clear diagonal with some width
        for i in range(size):
            for offset in [-1, 0, 1]:
                x, y = i, min(max(0, i + offset), size-1)
                if x < size and y < size:
                    new_grid[x, y] = 0
        
        return new_grid

# ============================================================================
# ENHANCED BRESENHAM'S LINE WITH INTERPOLATION
# ============================================================================

class BresenhamLine:
    """Enhanced Bresenham's line algorithm with smooth interpolation"""
    
    @staticmethod
    def compute(start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Compute line points using Bresenham's algorithm"""
        x0, y0 = start
        x1, y1 = goal
        points = []
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            points.append((x0, y0))
            
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
    
    @staticmethod
    def compute_with_thickness(start: Tuple[int, int], goal: Tuple[int, int], 
                              thickness: int = 1) -> Set[Tuple[int, int]]:
        """Compute thick line for visualization"""
        line_points = BresenhamLine.compute(start, goal)
        thick_points = set()
        
        for x, y in line_points:
            for dx in range(-thickness, thickness + 1):
                for dy in range(-thickness, thickness + 1):
                    if dx*dx + dy*dy <= thickness*thickness:
                        thick_points.add((x + dx, y + dy))
        
        return thick_points

# ============================================================================
# ADVANCED ADAPTIVE CORRIDOR SYSTEM
# ============================================================================

class AdvancedAdaptiveCorridor:
    """Advanced adaptive corridor with multiple strategies"""
    
    def __init__(self, grid: np.ndarray, config: ExperimentConfig):
        self.grid = grid
        self.config = config
        self.size = len(grid)
        self.density_cache = {}
        self._precompute_density_map()
    
    def _precompute_density_map(self):
        """Precompute density for entire grid"""
        window = self.config.window_size
        half_window = window // 2
        
        self.density_map = np.zeros_like(self.grid, dtype=float)
        
        for i in range(self.size):
            for j in range(self.size):
                x_min = max(0, i - half_window)
                x_max = min(self.size, i + half_window + 1)
                y_min = max(0, j - half_window)
                y_max = min(self.size, j + half_window + 1)
                
                window_area = self.grid[x_min:x_max, y_min:y_max]
                self.density_map[i, j] = np.mean(window_area)
    
    def compute_adaptive_corridor(self, start: Tuple[int, int], goal: Tuple[int, int],
                                 strategy: str = 'standard') -> Set[Tuple[int, int]]:
        """Compute adaptive corridor with different strategies"""
        if strategy == 'standard':
            return self._standard_adaptive_corridor(start, goal)
        elif strategy == 'predictive':
            return self._predictive_adaptive_corridor(start, goal)
        elif strategy == 'gradient':
            return self._gradient_based_corridor(start, goal)
        else:
            return self._hybrid_corridor(start, goal)
    
    def _standard_adaptive_corridor(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """Standard adaptive corridor based on local density"""
        line_points = BresenhamLine.compute(start, goal)
        corridor = set()
        max_width = self.config.get_max_corridor_width(self.size)
        
        for point in line_points:
            x, y = point
            density = self.density_map[x, y]
            
            # Adaptive width calculation
            width = self.config.min_corridor_width + int(
                (max_width - self.config.min_corridor_width) * density
            )
            
            # Add points within radius
            for dx in range(-width, width + 1):
                for dy in range(-width, width + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        corridor.add((nx, ny))
        
        return corridor
    
    def _predictive_adaptive_corridor(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """Predictive corridor that looks ahead for obstacles"""
        line_points = BresenhamLine.compute(start, goal)
        corridor = set()
        max_width = self.config.get_max_corridor_width(self.size)
        
        for i, point in enumerate(line_points):
            x, y = point
            
            # Look ahead for maximum density
            lookahead_density = self.density_map[x, y]
            for j in range(1, min(self.config.lookahead_distance + 1, len(line_points) - i)):
                next_point = line_points[i + j]
                lookahead_density = max(lookahead_density, self.density_map[next_point[0], next_point[1]])
            
            # Use maximum of current and lookahead
            width = self.config.min_corridor_width + int(
                (max_width - self.config.min_corridor_width) * lookahead_density
            )
            
            for dx in range(-width, width + 1):
                for dy in range(-width, width + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        corridor.add((nx, ny))
        
        return corridor
    
    def _gradient_based_corridor(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """Corridor based on density gradient"""
        line_points = BresenhamLine.compute(start, goal)
        corridor = set()
        max_width = self.config.get_max_corridor_width(self.size)
        
        # Compute density gradient
        grad_x = np.gradient(self.density_map, axis=0)
        grad_y = np.gradient(self.density_map, axis=1)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        for point in line_points:
            x, y = point
            
            # Use gradient to determine width
            grad_mag = gradient_magnitude[x, y]
            normalized_grad = min(1.0, grad_mag / (np.max(gradient_magnitude) + 1e-6))
            
            width = self.config.min_corridor_width + int(
                (max_width - self.config.min_corridor_width) * max(self.density_map[x, y], normalized_grad)
            )
            
            for dx in range(-width, width + 1):
                for dy in range(-width, width + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        corridor.add((nx, ny))
        
        return corridor
    
    def _hybrid_corridor(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """Hybrid approach combining multiple strategies"""
        # Combine all three strategies
        standard = self._standard_adaptive_corridor(start, goal)
        predictive = self._predictive_adaptive_corridor(start, goal)
        gradient = self._gradient_based_corridor(start, goal)
        
        # Use intersection for conservative approach or union for aggressive
        # Here we use a weighted combination
        all_points = standard | predictive | gradient
        common_points = standard & predictive & gradient
        
        # Include all common points plus some percentage of unique points
        corridor = common_points.copy()
        unique_points = all_points - common_points
        
        for point in unique_points:
            if random.random() < 0.5:  # Include 50% of unique points
                corridor.add(point)
        
        return corridor

# ============================================================================
# COMPREHENSIVE PATHFINDING ALGORITHMS
# ============================================================================

class ComprehensivePathfinding:
    """Complete suite of pathfinding algorithms with enhancements"""
    
    @staticmethod
    def euclidean_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance heuristic"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    @staticmethod
    def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    @staticmethod
    def chebyshev_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Chebyshev distance heuristic"""
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
    
    @staticmethod
    def get_neighbors(pos: Tuple[int, int], grid: np.ndarray, 
                     corridor: Optional[Set[Tuple[int, int]]] = None,
                     connectivity: int = 8) -> List[Tuple[Tuple[int, int], float]]:
        """Get valid neighbors with costs"""
        neighbors = []
        
        if connectivity == 4:
            moves = [(0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0)]
        else:  # 8-connectivity
            moves = [
                (0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0),
                (1, 1, 1.414), (-1, -1, 1.414), (1, -1, 1.414), (-1, 1, 1.414)
            ]
        
        for dx, dy, cost in moves:
            new_pos = (pos[0] + dx, pos[1] + dy)
            if (0 <= new_pos[0] < len(grid) and 
                0 <= new_pos[1] < len(grid[0]) and
                grid[new_pos[0]][new_pos[1]] == 0):
                if corridor is None or new_pos in corridor:
                    neighbors.append((new_pos, cost))
        
        return neighbors
    
    @staticmethod
    def reconstruct_path(came_from: Dict, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from dictionary"""
        path = []
        current = goal
        
        while current is not None:
            path.append(current)
            current = came_from.get(current)
            if current == start:
                path.append(start)
                break
        
        path.reverse()
        return path if path[0] == start else []
    
    @staticmethod
    def a_star(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
              corridor: Optional[Set[Tuple[int, int]]] = None,
              heuristic: str = 'euclidean') -> Dict[str, Any]:
        """A* pathfinding with comprehensive metrics"""
        start_time = time.time()
        
        # Select heuristic
        if heuristic == 'manhattan':
            h_func = ComprehensivePathfinding.manhattan_distance
        elif heuristic == 'chebyshev':
            h_func = ComprehensivePathfinding.chebyshev_distance
        else:
            h_func = ComprehensivePathfinding.euclidean_distance
        
        # Initialize
        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        visited = set()
        expanded = 0
        
        while frontier:
            current_priority, current = heapq.heappop(frontier)
            
            if current in visited:
                continue
            
            visited.add(current)
            expanded += 1
            
            if current == goal:
                path = ComprehensivePathfinding.reconstruct_path(came_from, start, goal)
                return {
                    'path': path,
                    'length': len(path),
                    'cost': cost_so_far[goal],
                    'expanded': expanded,
                    'visited': len(visited),
                    'time': time.time() - start_time,
                    'success': True
                }
            
            for next_pos, move_cost in ComprehensivePathfinding.get_neighbors(current, grid, corridor):
                new_cost = cost_so_far[current] + move_cost
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + h_func(next_pos, goal)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        return {
            'path': [],
            'length': 0,
            'cost': float('inf'),
            'expanded': expanded,
            'visited': len(visited),
            'time': time.time() - start_time,
            'success': False
        }
    
    @staticmethod
    def dijkstra(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                corridor: Optional[Set[Tuple[int, int]]] = None) -> Dict[str, Any]:
        """Dijkstra's algorithm"""
        start_time = time.time()
        
        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        visited = set()
        expanded = 0
        
        while frontier:
            current_cost, current = heapq.heappop(frontier)
            
            if current in visited:
                continue
            
            visited.add(current)
            expanded += 1
            
            if current == goal:
                path = ComprehensivePathfinding.reconstruct_path(came_from, start, goal)
                return {
                    'path': path,
                    'length': len(path),
                    'cost': cost_so_far[goal],
                    'expanded': expanded,
                    'visited': len(visited),
                    'time': time.time() - start_time,
                    'success': True
                }
            
            for next_pos, move_cost in ComprehensivePathfinding.get_neighbors(current, grid, corridor):
                new_cost = cost_so_far[current] + move_cost
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    heapq.heappush(frontier, (new_cost, next_pos))
                    came_from[next_pos] = current
        
        return {
            'path': [],
            'length': 0,
            'cost': float('inf'),
            'expanded': expanded,
            'visited': len(visited),
            'time': time.time() - start_time,
            'success': False
        }
    
    @staticmethod
    def bfs(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
           corridor: Optional[Set[Tuple[int, int]]] = None) -> Dict[str, Any]:
        """Breadth-First Search"""
        start_time = time.time()
        
        queue = deque([start])
        came_from = {start: None}
        visited = set([start])
        expanded = 0
        
        while queue:
            current = queue.popleft()
            expanded += 1
            
            if current == goal:
                path = ComprehensivePathfinding.reconstruct_path(came_from, start, goal)
                return {
                    'path': path,
                    'length': len(path),
                    'cost': len(path) - 1,
                    'expanded': expanded,
                    'visited': len(visited),
                    'time': time.time() - start_time,
                    'success': True
                }
            
            for next_pos, _ in ComprehensivePathfinding.get_neighbors(current, grid, corridor):
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append(next_pos)
                    came_from[next_pos] = current
        
        return {
            'path': [],
            'length': 0,
            'cost': float('inf'),
            'expanded': expanded,
            'visited': len(visited),
            'time': time.time() - start_time,
            'success': False
        }
    
    @staticmethod
    def dfs(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
           corridor: Optional[Set[Tuple[int, int]]] = None) -> Dict[str, Any]:
        """Depth-First Search"""
        start_time = time.time()
        
        stack = [start]
        came_from = {start: None}
        visited = set()
        expanded = 0
        
        while stack:
            current = stack.pop()
            
            if current in visited:
                continue
            
            visited.add(current)
            expanded += 1
            
            if current == goal:
                path = ComprehensivePathfinding.reconstruct_path(came_from, start, goal)
                return {
                    'path': path,
                    'length': len(path),
                    'cost': len(path) - 1,
                    'expanded': expanded,
                    'visited': len(visited),
                    'time': time.time() - start_time,
                    'success': True
                }
            
            for next_pos, _ in ComprehensivePathfinding.get_neighbors(current, grid, corridor):
                if next_pos not in visited:
                    stack.append(next_pos)
                    if next_pos not in came_from:
                        came_from[next_pos] = current
        
        return {
            'path': [],
            'length': 0,
            'cost': float('inf'),
            'expanded': expanded,
            'visited': len(visited),
            'time': time.time() - start_time,
            'success': False
        }
    
    @staticmethod
    def greedy_best_first(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                         corridor: Optional[Set[Tuple[int, int]]] = None) -> Dict[str, Any]:
        """Greedy Best-First Search"""
        start_time = time.time()
        
        h_func = ComprehensivePathfinding.euclidean_distance
        frontier = [(h_func(start, goal), start)]
        came_from = {start: None}
        visited = set()
        expanded = 0
        
        while frontier:
            _, current = heapq.heappop(frontier)
            
            if current in visited:
                continue
            
            visited.add(current)
            expanded += 1
            
            if current == goal:
                path = ComprehensivePathfinding.reconstruct_path(came_from, start, goal)
                return {
                    'path': path,
                    'length': len(path),
                    'cost': len(path) - 1,
                    'expanded': expanded,
                    'visited': len(visited),
                    'time': time.time() - start_time,
                    'success': True
                }
            
            for next_pos, _ in ComprehensivePathfinding.get_neighbors(current, grid, corridor):
                if next_pos not in visited:
                    priority = h_func(next_pos, goal)
                    heapq.heappush(frontier, (priority, next_pos))
                    if next_pos not in came_from:
                        came_from[next_pos] = current
        
        return {
            'path': [],
            'length': 0,
            'cost': float('inf'),
            'expanded': expanded,
            'visited': len(visited),
            'time': time.time() - start_time,
            'success': False
        }
    
    @staticmethod
    def bidirectional_search(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                           corridor: Optional[Set[Tuple[int, int]]] = None) -> Dict[str, Any]:
        """Bidirectional Search"""
        start_time = time.time()
        
        # Forward search
        forward_queue = deque([start])
        forward_visited = {start: None}
        
        # Backward search
        backward_queue = deque([goal])
        backward_visited = {goal: None}
        
        expanded = 0
        
        while forward_queue and backward_queue:
            # Forward step
            if forward_queue:
                current_f = forward_queue.popleft()
                expanded += 1
                
                if current_f in backward_visited:
                    # Path found - reconstruct
                    path1 = []
                    curr = current_f
                    while curr is not None:
                        path1.append(curr)
                        curr = forward_visited[curr]
                    path1.reverse()
                    
                    path2 = []
                    curr = backward_visited[current_f]
                    while curr is not None:
                        path2.append(curr)
                        curr = backward_visited[curr]
                    
                    path = path1 + path2
                    return {
                        'path': path,
                        'length': len(path),
                        'cost': len(path) - 1,
                        'expanded': expanded,
                        'visited': len(forward_visited) + len(backward_visited),
                        'time': time.time() - start_time,
                        'success': True
                    }
                
                for next_pos, _ in ComprehensivePathfinding.get_neighbors(current_f, grid, corridor):
                    if next_pos not in forward_visited:
                        forward_visited[next_pos] = current_f
                        forward_queue.append(next_pos)
            
            # Backward step
            if backward_queue:
                current_b = backward_queue.popleft()
                expanded += 1
                
                if current_b in forward_visited:
                    # Path found - reconstruct
                    path1 = []
                    curr = current_b
                    while curr is not None:
                        path1.append(curr)
                        curr = forward_visited[curr]
                    path1.reverse()
                    
                    path2 = []
                    curr = backward_visited[current_b]
                    while curr is not None:
                        path2.append(curr)
                        curr = backward_visited[curr]
                    
                    path = path1 + path2
                    return {
                        'path': path,
                        'length': len(path),
                        'cost': len(path) - 1,
                        'expanded': expanded,
                        'visited': len(forward_visited) + len(backward_visited),
                        'time': time.time() - start_time,
                        'success': True
                    }
                
                for next_pos, _ in ComprehensivePathfinding.get_neighbors(current_b, grid, corridor):
                    if next_pos not in backward_visited:
                        backward_visited[next_pos] = current_b
                        backward_queue.append(next_pos)
        
        return {
            'path': [],
            'length': 0,
            'cost': float('inf'),
            'expanded': expanded,
            'visited': len(forward_visited) + len(backward_visited),
            'time': time.time() - start_time,
            'success': False
        }

# ============================================================================
# ORIGINAL ILS IMPLEMENTATION
# ============================================================================

class OriginalILS:
    """Original Incremental Line Search with fixed corridor"""
    
    def __init__(self, grid: np.ndarray, config: ExperimentConfig):
        self.grid = grid
        self.config = config
        self.size = len(grid)
    
    def build_corridor(self, start: Tuple[int, int], goal: Tuple[int, int], width: int) -> Set[Tuple[int, int]]:
        """Build fixed-width corridor"""
        line_points = BresenhamLine.compute(start, goal)
        corridor = set()
        
        for x, y in line_points:
            for dx in range(-width, width + 1):
                for dy in range(-width, width + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        corridor.add((nx, ny))
        
        return corridor
    
    def search(self, algorithm_func, start: Tuple[int, int], goal: Tuple[int, int]) -> Dict[str, Any]:
        """Execute ILS search with incremental expansion"""
        total_time = 0
        total_expanded = 0
        total_visited = 0
        
        width = self.config.ils_initial_width
        
        for iteration in range(self.config.ils_max_iterations):
            corridor = self.build_corridor(start, goal, width)
            
            result = algorithm_func(self.grid, start, goal, corridor)
            total_time += result['time']
            total_expanded += result['expanded']
            total_visited += result['visited']
            
            if result['success']:
                result['time'] = total_time
                result['expanded'] = total_expanded
                result['visited'] = total_visited
                result['iterations'] = iteration + 1
                return result
            
            width += self.config.ils_step_size
        
        # Final attempt without corridor
        result = algorithm_func(self.grid, start, goal, None)
        result['time'] += total_time
        result['expanded'] += total_expanded
        result['visited'] += total_visited
        result['iterations'] = self.config.ils_max_iterations + 1
        return result

# ============================================================================
# ADAPTIVE ILS IMPLEMENTATION
# ============================================================================

class AdaptiveILS:
    """Adaptive Incremental Line Search"""
    
    def __init__(self, grid: np.ndarray, config: ExperimentConfig):
        self.grid = grid
        self.config = config
        self.corridor_builder = AdvancedAdaptiveCorridor(grid, config)
    
    def search(self, algorithm_func, start: Tuple[int, int], goal: Tuple[int, int],
              strategy: str = 'predictive') -> Dict[str, Any]:
        """Execute adaptive ILS search"""
        # Build adaptive corridor
        corridor = self.corridor_builder.compute_adaptive_corridor(start, goal, strategy)
        
        # Try with adaptive corridor
        result = algorithm_func(self.grid, start, goal, corridor)
        
        if result['success']:
            result['iterations'] = 1
            result['corridor_size'] = len(corridor)
            return result
        
        # Fallback: gradually expand
        total_time = result['time']
        total_expanded = result['expanded']
        total_visited = result['visited']
        
        for iteration in range(1, 4):  # Max 3 expansions
            # Increase corridor size
            old_max = self.config.get_max_corridor_width(len(self.grid))
            self.config.max_corridor_width_ratio *= 1.5
            
            corridor = self.corridor_builder.compute_adaptive_corridor(start, goal, strategy)
            
            # Reset config
            self.config.max_corridor_width_ratio = old_max / len(self.grid)
            
            result = algorithm_func(self.grid, start, goal, corridor)
            total_time += result['time']
            total_expanded += result['expanded']
            total_visited += result['visited']
            
            if result['success']:
                result['time'] = total_time
                result['expanded'] = total_expanded
                result['visited'] = total_visited
                result['iterations'] = iteration + 1
                result['corridor_size'] = len(corridor)
                return result
        
        # Final fallback
        result = algorithm_func(self.grid, start, goal, None)
        result['time'] += total_time
        result['expanded'] += total_expanded
        result['visited'] += total_visited
        result['iterations'] = 4
        result['corridor_size'] = self.grid.size
        return result

# ============================================================================
# EXPERIMENT RUNNER WITH MULTIPROCESSING
# ============================================================================

class ExperimentRunner:
    """Comprehensive experiment runner with parallel processing"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def run_single_trial(self, params: Tuple) -> List[Dict]:
        """Run a single trial (for multiprocessing)"""
        grid_size, density, pattern, trial_id, seed = params
        
        # Generate grid
        grid = AdvancedGridGenerator.generate(grid_size, density, pattern, seed)
        
        # Define start and goal
        start = (0, 0)
        goal = (grid_size - 1, grid_size - 1)
        
        # Algorithms to test
        algorithms = {
            'A*': ComprehensivePathfinding.a_star,
            'Dijkstra': ComprehensivePathfinding.dijkstra,
            'BFS': ComprehensivePathfinding.bfs,
            'DFS': ComprehensivePathfinding.dfs,
            'Greedy': ComprehensivePathfinding.greedy_best_first,
            'Bidirectional': ComprehensivePathfinding.bidirectional_search
        }
        
        trial_results = []
        
        for alg_name, alg_func in algorithms.items():
            # Standard algorithm
            result_std = alg_func(grid, start, goal, None)
            
            # Original ILS
            ils = OriginalILS(grid, self.config)
            result_ils = ils.search(alg_func, start, goal)
            
            # Adaptive ILS with different strategies
            ails = AdaptiveILS(grid, self.config)
            result_ails_std = ails.search(alg_func, start, goal, 'standard')
            result_ails_pred = ails.search(alg_func, start, goal, 'predictive')
            result_ails_grad = ails.search(alg_func, start, goal, 'gradient')
            
            # Compile results
            trial_results.append({
                'grid_size': grid_size,
                'obstacle_density': density,
                'pattern': pattern.value,
                'trial_id': trial_id,
                'algorithm': alg_name,
                
                # Standard metrics
                'std_time': result_std['time'],
                'std_expanded': result_std['expanded'],
                'std_visited': result_std['visited'],
                'std_path_length': result_std['length'],
                'std_path_cost': result_std['cost'],
                'std_success': result_std['success'],
                
                # ILS metrics
                'ils_time': result_ils['time'],
                'ils_expanded': result_ils['expanded'],
                'ils_visited': result_ils['visited'],
                'ils_path_length': result_ils['length'],
                'ils_path_cost': result_ils['cost'],
                'ils_success': result_ils['success'],
                'ils_iterations': result_ils.get('iterations', 0),
                
                # AILS Standard metrics
                'ails_std_time': result_ails_std['time'],
                'ails_std_expanded': result_ails_std['expanded'],
                'ails_std_visited': result_ails_std['visited'],
                'ails_std_path_length': result_ails_std['length'],
                'ails_std_path_cost': result_ails_std['cost'],
                'ails_std_success': result_ails_std['success'],
                'ails_std_iterations': result_ails_std.get('iterations', 0),
                'ails_std_corridor_size': result_ails_std.get('corridor_size', 0),
                
                # AILS Predictive metrics
                'ails_pred_time': result_ails_pred['time'],
                'ails_pred_expanded': result_ails_pred['expanded'],
                'ails_pred_visited': result_ails_pred['visited'],
                'ails_pred_path_length': result_ails_pred['length'],
                'ails_pred_path_cost': result_ails_pred['cost'],
                'ails_pred_success': result_ails_pred['success'],
                
                # AILS Gradient metrics
                'ails_grad_time': result_ails_grad['time'],
                'ails_grad_expanded': result_ails_grad['expanded'],
                'ails_grad_visited': result_ails_grad['visited'],
                'ails_grad_path_length': result_ails_grad['length'],
                'ails_grad_path_cost': result_ails_grad['cost'],
                'ails_grad_success': result_ails_grad['success'],
                
                # Improvements
                'ils_time_improvement': ((result_std['time'] - result_ils['time']) / 
                                        result_std['time'] * 100) if result_std['time'] > 0 else 0,
                'ails_pred_time_improvement': ((result_std['time'] - result_ails_pred['time']) / 
                                              result_std['time'] * 100) if result_std['time'] > 0 else 0,
                'ails_pred_vs_ils_improvement': ((result_ils['time'] - result_ails_pred['time']) / 
                                                result_ils['time'] * 100) if result_ils['time'] > 0 else 0,
            })
        
        return trial_results
    
    def run_experiments(self) -> pd.DataFrame:
        """Run all experiments with progress tracking"""
        print("=" * 80)
        print("COMPREHENSIVE PATHFINDING EXPERIMENT")
        print("=" * 80)
        print(f"Grid Sizes: {self.config.grid_sizes}")
        print(f"Obstacle Densities: {[f'{d*100:.0f}%' for d in self.config.obstacle_densities]}")
        print(f"Trials per configuration: {self.config.trials_per_config}")
        print(f"Total experiments: {len(self.config.grid_sizes) * len(self.config.obstacle_densities) * self.config.trials_per_config * 5}")
        print("-" * 80)
        
        # Prepare experiment parameters
        experiment_params = []
        
        patterns = [
            AdvancedGridGenerator.ObstaclePattern.RANDOM,
            AdvancedGridGenerator.ObstaclePattern.CLUSTERED,
            AdvancedGridGenerator.ObstaclePattern.MAZE,
            AdvancedGridGenerator.ObstaclePattern.ROOMS,
            AdvancedGridGenerator.ObstaclePattern.MIXED
        ]
        
        for grid_size in self.config.grid_sizes:
            for density in self.config.obstacle_densities:
                for pattern in patterns:
                    for trial in range(self.config.trials_per_config):
                        seed = self.config.random_seed + trial + grid_size + int(density * 1000)
                        experiment_params.append((grid_size, density, pattern, trial, seed))
        
        # Run experiments
        if self.config.use_multiprocessing:
            print(f"Running experiments in parallel with {self.config.num_workers} workers...")
            
            with mp.Pool(self.config.num_workers) as pool:
                results_nested = list(tqdm(
                    pool.imap(self.run_single_trial, experiment_params),
                    total=len(experiment_params),
                    desc="Running experiments"
                ))
            
            # Flatten results
            for result_list in results_nested:
                self.results.extend(result_list)
        else:
            print("Running experiments sequentially...")
            
            for params in tqdm(experiment_params, desc="Running experiments"):
                trial_results = self.run_single_trial(params)
                self.results.extend(trial_results)
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        # Save results
        if self.config.export_csv:
            csv_path = f'results/data/experiment_results_{self.timestamp}.csv'
            df.to_csv(csv_path, index=False)
            print(f"\nResults saved to: {csv_path}")
        
        if self.config.export_pickle:
            pickle_path = f'results/data/experiment_results_{self.timestamp}.pkl'
            df.to_pickle(pickle_path)
            print(f"Pickle saved to: {pickle_path}")
        
        return df

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

class StatisticalAnalysis:
    """Comprehensive statistical analysis of results"""
    
    @staticmethod
    def compute_confidence_intervals(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
        """Compute mean and confidence interval"""
        mean = np.mean(data)
        sem = stats.sem(data)
        interval = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
        return mean, mean - interval, mean + interval
    
    @staticmethod
    def perform_anova(df: pd.DataFrame, groups: str, values: str) -> Dict:
        """Perform one-way ANOVA"""
        groups_data = [group[values].values for name, group in df.groupby(groups)]
        f_stat, p_value = stats.f_oneway(*groups_data)
        
        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def perform_paired_t_test(data1: np.ndarray, data2: np.ndarray) -> Dict:
        """Perform paired t-test"""
        t_stat, p_value = stats.ttest_rel(data1, data2)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'mean_difference': np.mean(data1 - data2),
            'std_difference': np.std(data1 - data2)
        }
    
    @staticmethod
    def compute_effect_size(data1: np.ndarray, data2: np.ndarray) -> float:
        """Compute Cohen's d effect size"""
        mean_diff = np.mean(data1) - np.mean(data2)
        pooled_std = np.sqrt((np.var(data1) + np.var(data2)) / 2)
        return mean_diff / pooled_std if pooled_std > 0 else 0
    
    @staticmethod
    def analyze_results(df: pd.DataFrame) -> Dict:
        """Perform comprehensive statistical analysis"""
        results = {}
        
        # Overall statistics
        algorithms = df['algorithm'].unique()
        
        for alg in algorithms:
            alg_data = df[df['algorithm'] == alg]
            
            results[alg] = {
                'time_improvement': {
                    'ils_vs_std': StatisticalAnalysis.compute_confidence_intervals(
                        alg_data['ils_time_improvement'].values
                    ),
                    'ails_vs_std': StatisticalAnalysis.compute_confidence_intervals(
                        alg_data['ails_pred_time_improvement'].values
                    ),
                    'ails_vs_ils': StatisticalAnalysis.compute_confidence_intervals(
                        alg_data['ails_pred_vs_ils_improvement'].values
                    )
                },
                't_tests': {
                    'ils_vs_std': StatisticalAnalysis.perform_paired_t_test(
                        alg_data['std_time'].values,
                        alg_data['ils_time'].values
                    ),
                    'ails_vs_std': StatisticalAnalysis.perform_paired_t_test(
                        alg_data['std_time'].values,
                        alg_data['ails_pred_time'].values
                    ),
                    'ails_vs_ils': StatisticalAnalysis.perform_paired_t_test(
                        alg_data['ils_time'].values,
                        alg_data['ails_pred_time'].values
                    )
                },
                'effect_sizes': {
                    'ils_vs_std': StatisticalAnalysis.compute_effect_size(
                        alg_data['std_time'].values,
                        alg_data['ils_time'].values
                    ),
                    'ails_vs_std': StatisticalAnalysis.compute_effect_size(
                        alg_data['std_time'].values,
                        alg_data['ails_pred_time'].values
                    )
                }
            }
        
        # ANOVA for algorithm comparison
        results['anova'] = {
            'algorithms': StatisticalAnalysis.perform_anova(df, 'algorithm', 'ails_pred_time_improvement'),
            'grid_sizes': StatisticalAnalysis.perform_anova(df, 'grid_size', 'ails_pred_time_improvement'),
            'densities': StatisticalAnalysis.perform_anova(df, 'obstacle_density', 'ails_pred_time_improvement')
        }
        
        return results

# ============================================================================
# COMPREHENSIVE VISUALIZATION
# ============================================================================

class ComprehensiveVisualization:
    """Publication-quality visualizations"""
    
    @staticmethod
    def plot_performance_comparison(df: pd.DataFrame):
        """Create comprehensive performance comparison plot"""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Time comparison by algorithm
        ax1 = fig.add_subplot(gs[0, :])
        algorithm_means = df.groupby(['algorithm'])[
            ['std_time', 'ils_time', 'ails_pred_time']
        ].mean()
        
        x = np.arange(len(algorithm_means.index))
        width = 0.25
        
        ax1.bar(x - width, algorithm_means['std_time'], width, label='Standard', color='#e74c3c')
        ax1.bar(x, algorithm_means['ils_time'], width, label='ILS', color='#f39c12')
        ax1.bar(x + width, algorithm_means['ails_pred_time'], width, label='AILS', color='#27ae60')
        
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Execution Time (s)')
        ax1.set_title('Average Execution Time Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(algorithm_means.index, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Visited nodes comparison
        ax2 = fig.add_subplot(gs[1, 0])
        visited_means = df.groupby(['algorithm'])[
            ['std_visited', 'ils_visited', 'ails_pred_visited']
        ].mean()
        
        ax2.bar(x - width, visited_means['std_visited'], width, label='Standard', color='#e74c3c')
        ax2.bar(x, visited_means['ils_visited'], width, label='ILS', color='#f39c12')
        ax2.bar(x + width, visited_means['ails_pred_visited'], width, label='AILS', color='#27ae60')
        
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('Visited Nodes')
        ax2.set_title('Average Visited Nodes')
        ax2.set_xticks(x)
        ax2.set_xticklabels(visited_means.index, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Improvement percentages
        ax3 = fig.add_subplot(gs[1, 1])
        improvements = df.groupby(['algorithm'])[
            ['ils_time_improvement', 'ails_pred_time_improvement']
        ].mean()
        
        x = np.arange(len(improvements.index))
        ax3.bar(x - width/2, improvements['ils_time_improvement'], width, 
                label='ILS vs Standard', color='#3498db')
        ax3.bar(x + width/2, improvements['ails_pred_time_improvement'], width,
                label='AILS vs Standard', color='#9b59b6')
        
        ax3.set_xlabel('Algorithm')
        ax3.set_ylabel('Improvement (%)')
        ax3.set_title('Time Improvement Percentages')
        ax3.set_xticks(x)
        ax3.set_xticklabels(improvements.index, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # AILS vs ILS improvement
        ax4 = fig.add_subplot(gs[1, 2])
        ails_vs_ils = df.groupby(['algorithm'])['ails_pred_vs_ils_improvement'].mean()
        
        bars = ax4.bar(range(len(ails_vs_ils)), ails_vs_ils.values, color='#16a085')
        ax4.set_xlabel('Algorithm')
        ax4.set_ylabel('Improvement (%)')
        ax4.set_title('AILS vs ILS Improvement')
        ax4.set_xticks(range(len(ails_vs_ils)))
        ax4.set_xticklabels(ails_vs_ils.index, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, ails_vs_ils.values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # Performance by grid size
        ax5 = fig.add_subplot(gs[2, 0])
        for alg in df['algorithm'].unique():
            alg_data = df[df['algorithm'] == alg].groupby('grid_size')['ails_pred_time_improvement'].mean()
            ax5.plot(alg_data.index, alg_data.values, marker='o', label=alg)
        
        ax5.set_xlabel('Grid Size')
        ax5.set_ylabel('AILS Improvement (%)')
        ax5.set_title('Performance Scaling with Grid Size')
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax5.grid(True, alpha=0.3)
        
        # Performance by obstacle density
        ax6 = fig.add_subplot(gs[2, 1])
        for alg in df['algorithm'].unique():
            alg_data = df[df['algorithm'] == alg].groupby('obstacle_density')['ails_pred_time_improvement'].mean()
            ax6.plot(alg_data.index * 100, alg_data.values, marker='o', label=alg)
        
        ax6.set_xlabel('Obstacle Density (%)')
        ax6.set_ylabel('AILS Improvement (%)')
        ax6.set_title('Performance vs Obstacle Density')
        ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax6.grid(True, alpha=0.3)
        
        # Path quality comparison
        ax7 = fig.add_subplot(gs[2, 2])
        path_quality = df.groupby(['algorithm'])[
            ['std_path_length', 'ils_path_length', 'ails_pred_path_length']
        ].mean()
        
        # Normalize to standard path length
        path_quality_norm = path_quality.div(path_quality['std_path_length'], axis=0)
        
        x = np.arange(len(path_quality_norm.index))
        ax7.bar(x - width, path_quality_norm['std_path_length'], width, label='Standard', color='#e74c3c')
        ax7.bar(x, path_quality_norm['ils_path_length'], width, label='ILS', color='#f39c12')
        ax7.bar(x + width, path_quality_norm['ails_pred_path_length'], width, label='AILS', color='#27ae60')
        
        ax7.set_xlabel('Algorithm')
        ax7.set_ylabel('Normalized Path Length')
        ax7.set_title('Path Quality Comparison')
        ax7.set_xticks(x)
        ax7.set_xticklabels(path_quality_norm.index, rotation=45, ha='right')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        plt.suptitle('Comprehensive Performance Analysis: AILS vs ILS vs Standard', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save figure
        fig.savefig(f'results/figures/performance_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_heatmaps(df: pd.DataFrame):
        """Create heatmaps for different metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Prepare pivot tables
        metrics = ['ails_pred_time_improvement', 'ails_pred_vs_ils_improvement', 
                  'ails_pred_visited', 'ils_iterations', 'ails_pred_path_length']
        titles = ['AILS vs Standard Time Improvement (%)', 'AILS vs ILS Time Improvement (%)',
                 'AILS Visited Nodes', 'ILS Iterations Required', 'AILS Path Length']
        
        for idx, (metric, title) in enumerate(zip(metrics[:5], titles)):
            ax = axes[idx // 3, idx % 3]
            
            pivot = df.pivot_table(
                values=metric,
                index='obstacle_density',
                columns='grid_size',
                aggfunc='mean'
            )
            
            sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, cbar_kws={'label': metric})
            ax.set_title(title)
            ax.set_xlabel('Grid Size')
            ax.set_ylabel('Obstacle Density')
            
            # Format y-axis as percentages
            ax.set_yticklabels([f'{float(label.get_text())*100:.0f}%' for label in ax.get_yticklabels()])
        
        # Remove empty subplot
        fig.delaxes(axes[1, 2])
        
        plt.suptitle('Performance Heatmaps: Grid Size vs Obstacle Density', fontsize=14)
        plt.tight_layout()
        
        # Save figure
        fig.savefig(f'results/figures/heatmaps_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_statistical_analysis(df: pd.DataFrame, stats_results: Dict):
        """Plot statistical analysis results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Confidence intervals
        ax1 = axes[0, 0]
        algorithms = list(stats_results.keys())
        if 'anova' in algorithms:
            algorithms.remove('anova')
        
        positions = np.arange(len(algorithms))
        
        for i, alg in enumerate(algorithms):
            if alg in stats_results:
                mean, lower, upper = stats_results[alg]['time_improvement']['ails_vs_std']
                ax1.errorbar(i, mean, yerr=[[mean - lower], [upper - mean]], 
                           fmt='o', capsize=5, capthick=2, label=alg)
        
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Time Improvement (%)')
        ax1.set_title('95% Confidence Intervals for AILS Time Improvement')
        ax1.set_xticks(positions)
        ax1.set_xticklabels(algorithms, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Effect sizes
        ax2 = axes[0, 1]
        effect_sizes_ils = []
        effect_sizes_ails = []
        
        for alg in algorithms:
            if alg in stats_results:
                effect_sizes_ils.append(stats_results[alg]['effect_sizes']['ils_vs_std'])
                effect_sizes_ails.append(stats_results[alg]['effect_sizes']['ails_vs_std'])
        
        x = np.arange(len(algorithms))
        width = 0.35
        
        ax2.bar(x - width/2, effect_sizes_ils, width, label='ILS vs Standard', color='#3498db')
        ax2.bar(x + width/2, effect_sizes_ails, width, label='AILS vs Standard', color='#9b59b6')
        
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel("Cohen's d")
        ax2.set_title('Effect Sizes')
        ax2.set_xticks(x)
        ax2.set_xticklabels(algorithms, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add reference lines for effect size interpretation
        ax2.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small')
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium')
        ax2.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Large')
        
        # P-values from t-tests
        ax3 = axes[0, 2]
        p_values_data = []
        
        for alg in algorithms:
            if alg in stats_results:
                p_values_data.append([
                    stats_results[alg]['t_tests']['ils_vs_std']['p_value'],
                    stats_results[alg]['t_tests']['ails_vs_std']['p_value'],
                    stats_results[alg]['t_tests']['ails_vs_ils']['p_value']
                ])
        
        p_values_array = np.array(p_values_data).T
        
        im = ax3.imshow(p_values_array, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=0.1)
        ax3.set_xticks(range(len(algorithms)))
        ax3.set_xticklabels(algorithms, rotation=45, ha='right')
        ax3.set_yticks(range(3))
        ax3.set_yticklabels(['ILS vs Std', 'AILS vs Std', 'AILS vs ILS'])
        ax3.set_title('P-values from Paired T-tests')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('p-value')
        
        # Add significance markers
        for i in range(len(algorithms)):
            for j in range(3):
                if p_values_array[j, i] < 0.05:
                    ax3.text(i, j, '*', ha='center', va='center', color='white', fontsize=14)
        
        # Distribution plots
        ax4 = axes[1, 0]
        sample_alg = 'A*'
        if sample_alg in df['algorithm'].values:
            alg_data = df[df['algorithm'] == sample_alg]
            ax4.hist(alg_data['std_time'], bins=30, alpha=0.5, label='Standard', color='red')
            ax4.hist(alg_data['ils_time'], bins=30, alpha=0.5, label='ILS', color='orange')
            ax4.hist(alg_data['ails_pred_time'], bins=30, alpha=0.5, label='AILS', color='green')
            ax4.set_xlabel('Execution Time (s)')
            ax4.set_ylabel('Frequency')
            ax4.set_title(f'Time Distribution for {sample_alg}')
            ax4.legend()
        
        # Correlation matrix
        ax5 = axes[1, 1]
        correlation_cols = ['grid_size', 'obstacle_density', 'ails_pred_time_improvement', 
                          'ails_pred_visited', 'ails_std_corridor_size']
        
        if all(col in df.columns for col in correlation_cols):
            corr_matrix = df[correlation_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, ax=ax5, square=True)
            ax5.set_title('Correlation Matrix')
        
        # Pattern comparison
        ax6 = axes[1, 2]
        if 'pattern' in df.columns:
            pattern_performance = df.groupby('pattern')['ails_pred_time_improvement'].mean().sort_values()
            bars = ax6.barh(range(len(pattern_performance)), pattern_performance.values)
            ax6.set_yticks(range(len(pattern_performance)))
            ax6.set_yticklabels(pattern_performance.index)
            ax6.set_xlabel('AILS Time Improvement (%)')
            ax6.set_title('Performance by Obstacle Pattern')
            
            # Color bars by value
            for bar, val in zip(bars, pattern_performance.values):
                bar.set_color('#27ae60' if val > 80 else '#f39c12' if val > 60 else '#e74c3c')
        
        plt.suptitle('Statistical Analysis Results', fontsize=14)
        plt.tight_layout()
        
        # Save figure
        fig.savefig(f'results/figures/statistical_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

# ============================================================================
# LATEX TABLE GENERATOR
# ============================================================================

class LatexTableGenerator:
    """Generate LaTeX tables for publication"""
    
    @staticmethod
    def generate_summary_table(df: pd.DataFrame) -> str:
        """Generate summary statistics table"""
        summary = df.groupby(['algorithm']).agg({
            'std_time': ['mean', 'std'],
            'ils_time': ['mean', 'std'],
            'ails_pred_time': ['mean', 'std'],
            'ails_pred_time_improvement': ['mean', 'std'],
            'ails_pred_vs_ils_improvement': ['mean', 'std']
        }).round(4)
        
        latex = "\\begin{table}[h]\n"
        latex += "\\centering\n"
        latex += "\\caption{Summary Statistics for All Algorithms}\n"
        latex += "\\label{tab:summary}\n"
        latex += "\\begin{tabular}{l|cc|cc|cc|cc|cc}\n"
        latex += "\\toprule\n"
        latex += "\\multirow{2}{*}{Algorithm} & \\multicolumn{2}{c|}{Standard} & \\multicolumn{2}{c|}{ILS} & "
        latex += "\\multicolumn{2}{c|}{AILS} & \\multicolumn{2}{c|}{AILS vs Std} & \\multicolumn{2}{c}{AILS vs ILS} \\\\\n"
        latex += "& Time (s) & SD & Time (s) & SD & Time (s) & SD & Imp (\\%) & SD & Imp (\\%) & SD \\\\\n"
        latex += "\\midrule\n"
        
        for alg in summary.index:
            row_data = summary.loc[alg]
            latex += f"{alg} & "
            latex += f"{row_data[('std_time', 'mean')]:.4f} & {row_data[('std_time', 'std')]:.4f} & "
            latex += f"{row_data[('ils_time', 'mean')]:.4f} & {row_data[('ils_time', 'std')]:.4f} & "
            latex += f"{row_data[('ails_pred_time', 'mean')]:.4f} & {row_data[('ails_pred_time', 'std')]:.4f} & "
            latex += f"{row_data[('ails_pred_time_improvement', 'mean')]:.2f} & {row_data[('ails_pred_time_improvement', 'std')]:.2f} & "
            latex += f"{row_data[('ails_pred_vs_ils_improvement', 'mean')]:.2f} & {row_data[('ails_pred_vs_ils_improvement', 'std')]:.2f} \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    @staticmethod
    def generate_scaling_table(df: pd.DataFrame) -> str:
        """Generate scaling performance table"""
        scaling = df.groupby(['grid_size']).agg({
            'ails_pred_time_improvement': 'mean',
            'ails_pred_visited': 'mean',
            'ails_std_corridor_size': 'mean'
        }).round(2)
        
        latex = "\\begin{table}[h]\n"
        latex += "\\centering\n"
        latex += "\\caption{Performance Scaling with Grid Size}\n"
        latex += "\\label{tab:scaling}\n"
        latex += "\\begin{tabular}{c|ccc}\n"
        latex += "\\toprule\n"
        latex += "Grid Size & Time Improvement (\\%) & Avg Visited Nodes & Avg Corridor Size \\\\\n"
        latex += "\\midrule\n"
        
        for size in scaling.index:
            row = scaling.loc[size]
            latex += f"{size}×{size} & {row['ails_pred_time_improvement']:.1f} & "
            latex += f"{row['ails_pred_visited']:.0f} & {row['ails_std_corridor_size']:.0f} \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    @staticmethod
    def save_tables(df: pd.DataFrame):
        """Save all LaTeX tables to file"""
        with open(f'results/analysis/latex_tables_{datetime.now().strftime("%Y%m%d_%H%M%S")}.tex', 'w') as f:
            f.write("% Auto-generated LaTeX tables\n")
            f.write("% " + "="*50 + "\n\n")
            
            f.write(LatexTableGenerator.generate_summary_table(df))
            f.write("\n\n")
            
            f.write(LatexTableGenerator.generate_scaling_table(df))
            
        print("LaTeX tables saved to results/analysis/")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_comprehensive_experiment():
    """Execute the complete experimental framework"""
    
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║     ADAPTIVE INCREMENTAL LINE SEARCH (AILS) FRAMEWORK         ║
    ║            Comprehensive Research Evaluation                   ║
    ╔════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize configuration
    config = ExperimentConfig(
        grid_sizes=[50, 100, 200],  # Adjust based on computational resources
        obstacle_densities=[0.10, 0.20, 0.30],
        trials_per_config=10,  # Adjust based on time constraints
        use_multiprocessing=True
    )
    
    # Run experiments
    runner = ExperimentRunner(config)
    df = runner.run_experiments()
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED")
    print("="*80)
    print(f"Total experiments run: {len(df)}")
    print(f"Unique configurations: {len(df.groupby(['grid_size', 'obstacle_density', 'pattern']))}")
    
    # Statistical analysis
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)
    
    stats_results = StatisticalAnalysis.analyze_results(df)
    
    # Print key findings
    for alg in ['A*', 'Dijkstra', 'BFS']:
        if alg in stats_results:
            mean, lower, upper = stats_results[alg]['time_improvement']['ails_vs_std']
            print(f"\n{alg}:")
            print(f"  AILS vs Standard: {mean:.2f}% improvement (95% CI: [{lower:.2f}, {upper:.2f}])")
            print(f"  P-value: {stats_results[alg]['t_tests']['ails_vs_std']['p_value']:.6f}")
            print(f"  Effect size: {stats_results[alg]['effect_sizes']['ails_vs_std']:.3f}")
    
    # Visualization
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    ComprehensiveVisualization.plot_performance_comparison(df)
    ComprehensiveVisualization.plot_heatmaps(df)
    ComprehensiveVisualization.plot_statistical_analysis(df, stats_results)
    
    # Generate LaTeX tables
    if config.generate_latex_tables:
        print("\n" + "="*80)
        print("GENERATING LATEX TABLES")
        print("="*80)
        LatexTableGenerator.save_tables(df)
    
    # Summary report
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    overall_improvement = df['ails_pred_time_improvement'].mean()
    best_case = df['ails_pred_time_improvement'].max()
    worst_case = df['ails_pred_time_improvement'].min()
    
    print(f"Overall AILS Performance:")
    print(f"  Average improvement: {overall_improvement:.2f}%")
    print(f"  Best case improvement: {best_case:.2f}%")
    print(f"  Worst case improvement: {worst_case:.2f}%")
    
    print(f"\nBest performing algorithm: {df.groupby('algorithm')['ails_pred_time_improvement'].mean().idxmax()}")
    print(f"Best performing pattern: {df.groupby('pattern')['ails_pred_time_improvement'].mean().idxmax()}")
    
    print("\nFiles saved:")
    print(f"  - CSV: results/data/experiment_results_*.csv")
    print(f"  - Figures: results/figures/*.png")
    print(f"  - LaTeX: results/analysis/latex_tables_*.tex")
    
    return df, stats_results

# ============================================================================
# QUICK DEMO
# ============================================================================

def quick_demo():
    """Run a quick demonstration with visualization"""
    print("Running Quick Demo...")
    
    # Generate sample grid
    grid = AdvancedGridGenerator.generate(
        100, 0.2, AdvancedGridGenerator.ObstaclePattern.CLUSTERED, seed=42
    )
    
    start = (0, 0)
    goal = (99, 99)
    
    # Initialize algorithms
    config = ExperimentConfig()
    ails = AdaptiveILS(grid, config)
    
    # Run A* with different configurations
    result_std = ComprehensivePathfinding.a_star(grid, start, goal, None)
    result_ails = ails.search(ComprehensivePathfinding.a_star, start, goal, 'predictive')
    
    print(f"\nResults:")
    print(f"Standard A*: Time={result_std['time']:.4f}s, Visited={result_std['visited']}")
    print(f"AILS A*: Time={result_ails['time']:.4f}s, Visited={result_ails['visited']}")
    print(f"Improvement: {((result_std['time'] - result_ails['time'])/result_std['time']*100):.1f}%")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot grid with paths
    axes[0].imshow(grid, cmap='binary')
    if result_std['path']:
        path_array = np.array(result_std['path'])
        axes[0].plot(path_array[:, 1], path_array[:, 0], 'r-', linewidth=2, label='Standard A*')
    axes[0].set_title('Standard A* Path')
    axes[0].axis('off')
    
    axes[1].imshow(grid, cmap='binary')
    if result_ails['path']:
        path_array = np.array(result_ails['path'])
        axes[1].plot(path_array[:, 1], path_array[:, 0], 'g-', linewidth=2, label='AILS')
    
    # Show corridor
    corridor = ails.corridor_builder.compute_adaptive_corridor(start, goal, 'predictive')
    corridor_mask = np.zeros_like(grid)
    for point in corridor:
        if 0 <= point[0] < 100 and 0 <= point[1] < 100:
            corridor_mask[point] = 1
    axes[1].imshow(corridor_mask, cmap='Greens', alpha=0.3)
    axes[1].set_title('AILS Path with Adaptive Corridor')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# NOTEBOOK INTERFACE
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("Select an option:")
    print("1. Quick Demo")
    print("2. Comprehensive Experiment (WARNING: This will take considerable time)")
    print("3. Exit")
    
    # For Jupyter notebook, call functions directly:
    # quick_demo()  # For a quick demonstration
    # df, stats = run_comprehensive_experiment()  # For full analysis
    
    # Default: Run quick demo
    quick_demo()
