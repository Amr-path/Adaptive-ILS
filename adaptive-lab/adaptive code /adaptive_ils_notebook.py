# Adaptive Incremental Line Search (AILS) - Complete Implementation
# A smart pathfinding optimization that uses narrow corridors by default
# and widens only near obstacles for maximum efficiency

import numpy as np
import matplotlib.pyplot as plt
import heapq
from collections import deque
import time
import random
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
import pandas as pd
import seaborn as sns
from IPython.display import display, HTML
import warnings
warnings.filterwarnings('ignore')

# Configuration
@dataclass
class Config:
    """Configuration parameters for the experiments"""
    grid_size: int = 200
    obstacle_densities: List[float] = None
    num_trials: int = 100
    min_corridor_width: int = 1
    max_corridor_width: int = 20
    window_size: int = 7
    lookahead_distance: int = 5
    
    def __post_init__(self):
        if self.obstacle_densities is None:
            self.obstacle_densities = [0.1, 0.2, 0.3]

config = Config()

# ============================================================================
# GRID MAP GENERATION
# ============================================================================

class GridMapGenerator:
    """Generate random grid maps with specified obstacle density"""
    
    @staticmethod
    def generate(size: int, obstacle_density: float, seed: Optional[int] = None) -> np.ndarray:
        """Generate a random grid map with guaranteed path from top-left to bottom-right"""
        if seed is not None:
            np.random.seed(seed)
            
        grid = np.zeros((size, size), dtype=int)
        num_obstacles = int(size * size * obstacle_density)
        
        # Place random obstacles
        positions = [(i, j) for i in range(size) for j in range(size)]
        positions.remove((0, 0))  # Keep start free
        positions.remove((size-1, size-1))  # Keep goal free
        
        obstacle_positions = random.sample(positions, min(num_obstacles, len(positions)))
        for i, j in obstacle_positions:
            grid[i][j] = 1
            
        # Ensure path exists using BFS
        if not GridMapGenerator._path_exists(grid, (0, 0), (size-1, size-1)):
            # Create a guaranteed path
            grid = GridMapGenerator._ensure_path(grid, size)
            
        return grid
    
    @staticmethod
    def _path_exists(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        """Check if a path exists using BFS"""
        visited = set()
        queue = deque([start])
        visited.add(start)
        
        while queue:
            current = queue.popleft()
            if current == goal:
                return True
                
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                ni, nj = current[0] + di, current[1] + dj
                if (0 <= ni < len(grid) and 0 <= nj < len(grid[0]) and 
                    grid[ni][nj] == 0 and (ni, nj) not in visited):
                    visited.add((ni, nj))
                    queue.append((ni, nj))
                    
        return False
    
    @staticmethod
    def _ensure_path(grid: np.ndarray, size: int) -> np.ndarray:
        """Ensure a path exists by clearing a diagonal path"""
        new_grid = grid.copy()
        # Clear diagonal path
        for i in range(size):
            new_grid[i, min(i, size-1)] = 0
            if i < size - 1:
                new_grid[i, min(i+1, size-1)] = 0
        return new_grid

# ============================================================================
# BRESENHAM'S LINE ALGORITHM
# ============================================================================

def bresenham_line(start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Generate points along a line using Bresenham's algorithm"""
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

# ============================================================================
# ADAPTIVE CORRIDOR CALCULATION
# ============================================================================

class AdaptiveCorridor:
    """Calculate adaptive corridor based on local obstacle density"""
    
    def __init__(self, grid: np.ndarray, config: Config):
        self.grid = grid
        self.config = config
        self.size = len(grid)
        
    def compute_local_density(self, point: Tuple[int, int], window_size: int) -> float:
        """Compute obstacle density in a window around a point"""
        x, y = point
        half_window = window_size // 2
        
        x_min = max(0, x - half_window)
        x_max = min(self.size, x + half_window + 1)
        y_min = max(0, y - half_window)
        y_max = min(self.size, y + half_window + 1)
        
        window = self.grid[x_min:x_max, y_min:y_max]
        if window.size == 0:
            return 0.0
        
        return np.sum(window) / window.size
    
    def compute_adaptive_width(self, line_points: List[Tuple[int, int]]) -> List[int]:
        """Compute adaptive corridor width for each point on the line"""
        widths = []
        
        for i, point in enumerate(line_points):
            # Compute local density
            density = self.compute_local_density(point, self.config.window_size)
            
            # Look ahead for obstacles
            lookahead_density = 0.0
            for j in range(1, min(self.config.lookahead_distance + 1, len(line_points) - i)):
                lookahead_density = max(lookahead_density, 
                                       self.compute_local_density(line_points[i + j], 
                                                                 self.config.window_size))
            
            # Use maximum of current and lookahead density
            effective_density = max(density, lookahead_density)
            
            # Calculate width based on density (narrow in sparse, wide in dense)
            width = self.config.min_corridor_width + int(
                (self.config.max_corridor_width - self.config.min_corridor_width) * effective_density
            )
            
            widths.append(width)
            
        return widths
    
    def build_corridor(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """Build adaptive corridor from start to goal"""
        line_points = bresenham_line(start, goal)
        widths = self.compute_adaptive_width(line_points)
        
        corridor = set()
        for point, width in zip(line_points, widths):
            # Add all points within width distance from the line point
            x, y = point
            for dx in range(-width, width + 1):
                for dy in range(-width, width + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        corridor.add((nx, ny))
                        
        return corridor

# ============================================================================
# PATHFINDING ALGORITHMS
# ============================================================================

class PathfindingAlgorithms:
    """Implementation of various pathfinding algorithms"""
    
    @staticmethod
    def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance heuristic"""
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5
    
    @staticmethod
    def get_neighbors(pos: Tuple[int, int], grid: np.ndarray, 
                      corridor: Optional[Set[Tuple[int, int]]] = None) -> List[Tuple[int, int]]:
        """Get valid neighbors, optionally restricted to corridor"""
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
            new_pos = (pos[0] + dx, pos[1] + dy)
            if (0 <= new_pos[0] < len(grid) and 0 <= new_pos[1] < len(grid[0]) and
                grid[new_pos[0]][new_pos[1]] == 0):
                if corridor is None or new_pos in corridor:
                    neighbors.append(new_pos)
        return neighbors
    
    @staticmethod
    def a_star(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], 
               corridor: Optional[Set[Tuple[int, int]]] = None) -> Tuple[Optional[List], int, float]:
        """A* pathfinding algorithm"""
        start_time = time.time()
        visited_count = 0
        
        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current_cost, current = heapq.heappop(frontier)
            visited_count += 1
            
            if current == goal:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path, visited_count, time.time() - start_time
            
            for next_pos in PathfindingAlgorithms.get_neighbors(current, grid, corridor):
                new_cost = cost_so_far[current] + 1
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + PathfindingAlgorithms.heuristic(next_pos, goal)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        return None, visited_count, time.time() - start_time
    
    @staticmethod
    def dijkstra(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 corridor: Optional[Set[Tuple[int, int]]] = None) -> Tuple[Optional[List], int, float]:
        """Dijkstra's algorithm"""
        start_time = time.time()
        visited_count = 0
        
        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current_cost, current = heapq.heappop(frontier)
            visited_count += 1
            
            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path, visited_count, time.time() - start_time
            
            for next_pos in PathfindingAlgorithms.get_neighbors(current, grid, corridor):
                new_cost = cost_so_far[current] + 1
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    heapq.heappush(frontier, (new_cost, next_pos))
                    came_from[next_pos] = current
        
        return None, visited_count, time.time() - start_time
    
    @staticmethod
    def bfs(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
            corridor: Optional[Set[Tuple[int, int]]] = None) -> Tuple[Optional[List], int, float]:
        """Breadth-First Search"""
        start_time = time.time()
        visited_count = 0
        
        queue = deque([start])
        came_from = {start: None}
        
        while queue:
            current = queue.popleft()
            visited_count += 1
            
            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path, visited_count, time.time() - start_time
            
            for next_pos in PathfindingAlgorithms.get_neighbors(current, grid, corridor):
                if next_pos not in came_from:
                    queue.append(next_pos)
                    came_from[next_pos] = current
        
        return None, visited_count, time.time() - start_time
    
    @staticmethod
    def dfs(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
            corridor: Optional[Set[Tuple[int, int]]] = None) -> Tuple[Optional[List], int, float]:
        """Depth-First Search"""
        start_time = time.time()
        visited_count = 0
        
        stack = [start]
        came_from = {start: None}
        
        while stack:
            current = stack.pop()
            visited_count += 1
            
            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path, visited_count, time.time() - start_time
            
            for next_pos in PathfindingAlgorithms.get_neighbors(current, grid, corridor):
                if next_pos not in came_from:
                    stack.append(next_pos)
                    came_from[next_pos] = current
        
        return None, visited_count, time.time() - start_time
    
    @staticmethod
    def best_first(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                   corridor: Optional[Set[Tuple[int, int]]] = None) -> Tuple[Optional[List], int, float]:
        """Best-First Search (Greedy)"""
        start_time = time.time()
        visited_count = 0
        
        frontier = [(PathfindingAlgorithms.heuristic(start, goal), start)]
        came_from = {start: None}
        
        while frontier:
            _, current = heapq.heappop(frontier)
            visited_count += 1
            
            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path, visited_count, time.time() - start_time
            
            for next_pos in PathfindingAlgorithms.get_neighbors(current, grid, corridor):
                if next_pos not in came_from:
                    priority = PathfindingAlgorithms.heuristic(next_pos, goal)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        return None, visited_count, time.time() - start_time

# ============================================================================
# INCREMENTAL LINE SEARCH (Original Fixed-Width Version)
# ============================================================================

class IncrementalLineSearch:
    """Original ILS with fixed corridor width"""
    
    def __init__(self, grid: np.ndarray, initial_width: int = 5, max_width: int = 20, step: int = 2):
        self.grid = grid
        self.initial_width = initial_width
        self.max_width = max_width
        self.step = step
        self.size = len(grid)
    
    def build_fixed_corridor(self, start: Tuple[int, int], goal: Tuple[int, int], width: int) -> Set[Tuple[int, int]]:
        """Build fixed-width corridor"""
        line_points = bresenham_line(start, goal)
        corridor = set()
        
        for point in line_points:
            x, y = point
            for dx in range(-width, width + 1):
                for dy in range(-width, width + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        corridor.add((nx, ny))
        
        return corridor
    
    def search(self, algorithm_func, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[Optional[List], int, float]:
        """Search with incremental corridor expansion"""
        total_visited = 0
        start_time = time.time()
        
        width = self.initial_width
        while width <= self.max_width:
            corridor = self.build_fixed_corridor(start, goal, width)
            path, visited, _ = algorithm_func(self.grid, start, goal, corridor)
            total_visited += visited
            
            if path is not None:
                return path, total_visited, time.time() - start_time
            
            width += self.step
        
        # Fallback to full search
        path, visited, _ = algorithm_func(self.grid, start, goal, None)
        total_visited += visited
        return path, total_visited, time.time() - start_time

# ============================================================================
# ADAPTIVE INCREMENTAL LINE SEARCH (New Enhanced Version)
# ============================================================================

class AdaptiveIncrementalLineSearch:
    """Enhanced ILS with adaptive corridor based on obstacle density"""
    
    def __init__(self, grid: np.ndarray, config: Config):
        self.grid = grid
        self.config = config
        self.adaptive_corridor = AdaptiveCorridor(grid, config)
        
    def search(self, algorithm_func, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[Optional[List], int, float]:
        """Search with adaptive corridor"""
        start_time = time.time()
        
        # Build initial adaptive corridor
        corridor = self.adaptive_corridor.build_corridor(start, goal)
        path, visited, _ = algorithm_func(self.grid, start, goal, corridor)
        
        if path is not None:
            return path, visited, time.time() - start_time
        
        # If no path found, gradually expand corridor
        expansion_factor = 1.5
        max_expansions = 3
        
        for i in range(max_expansions):
            # Increase corridor widths
            old_max = self.config.max_corridor_width
            self.config.max_corridor_width = int(self.config.max_corridor_width * expansion_factor)
            corridor = self.adaptive_corridor.build_corridor(start, goal)
            self.config.max_corridor_width = old_max  # Reset for next iteration
            
            path, new_visited, _ = algorithm_func(self.grid, start, goal, corridor)
            visited += new_visited
            
            if path is not None:
                return path, visited, time.time() - start_time
        
        # Final fallback to full search
        path, new_visited, _ = algorithm_func(self.grid, start, goal, None)
        visited += new_visited
        return path, visited, time.time() - start_time

# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

class ExperimentRunner:
    """Run experiments and collect results"""
    
    def __init__(self, config: Config):
        self.config = config
        self.results = []
        
    def run_single_experiment(self, grid: np.ndarray, algorithm_name: str, algorithm_func) -> Dict:
        """Run a single experiment with all three variants"""
        start = (0, 0)
        goal = (self.config.grid_size - 1, self.config.grid_size - 1)
        
        # Standard algorithm
        path_std, visited_std, time_std = algorithm_func(grid, start, goal, None)
        
        # Original ILS
        ils = IncrementalLineSearch(grid)
        path_ils, visited_ils, time_ils = ils.search(algorithm_func, start, goal)
        
        # Adaptive ILS
        ails = AdaptiveIncrementalLineSearch(grid, self.config)
        path_ails, visited_ails, time_ails = ails.search(algorithm_func, start, goal)
        
        return {
            'algorithm': algorithm_name,
            'path_length_standard': len(path_std) if path_std else 0,
            'path_length_ils': len(path_ils) if path_ils else 0,
            'path_length_ails': len(path_ails) if path_ails else 0,
            'visited_standard': visited_std,
            'visited_ils': visited_ils,
            'visited_ails': visited_ails,
            'time_standard': time_std,
            'time_ils': time_ils,
            'time_ails': time_ails,
            'improvement_ils_vs_standard': ((time_std - time_ils) / time_std * 100) if time_std > 0 else 0,
            'improvement_ails_vs_standard': ((time_std - time_ails) / time_std * 100) if time_std > 0 else 0,
            'improvement_ails_vs_ils': ((time_ils - time_ails) / time_ils * 100) if time_ils > 0 else 0
        }
    
    def run_experiments(self) -> pd.DataFrame:
        """Run all experiments"""
        algorithms = {
            'A*': PathfindingAlgorithms.a_star,
            'Dijkstra': PathfindingAlgorithms.dijkstra,
            'BFS': PathfindingAlgorithms.bfs,
            'DFS': PathfindingAlgorithms.dfs,
            'Best-First': PathfindingAlgorithms.best_first
        }
        
        print("Running experiments...")
        print(f"Grid size: {self.config.grid_size}x{self.config.grid_size}")
        print(f"Trials per configuration: {self.config.num_trials}")
        print("-" * 60)
        
        for density in self.config.obstacle_densities:
            print(f"\nObstacle Density: {density*100:.0f}%")
            
            for trial in range(self.config.num_trials):
                # Generate random grid
                grid = GridMapGenerator.generate(self.config.grid_size, density, seed=trial)
                
                for alg_name, alg_func in algorithms.items():
                    result = self.run_single_experiment(grid, alg_name, alg_func)
                    result['obstacle_density'] = density
                    result['trial'] = trial
                    self.results.append(result)
                
                if (trial + 1) % 20 == 0:
                    print(f"  Completed {trial + 1}/{self.config.num_trials} trials")
        
        return pd.DataFrame(self.results)

# ============================================================================
# VISUALIZATION
# ============================================================================

class Visualizer:
    """Visualization utilities"""
    
    @staticmethod
    def plot_grid_with_paths(grid: np.ndarray, paths: Dict[str, List], title: str = "Pathfinding Comparison"):
        """Plot grid with multiple paths"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        colors = {'Standard': 'red', 'ILS': 'yellow', 'AILS': 'lime'}
        
        for idx, (name, path) in enumerate(paths.items()):
            ax = axes[idx]
            ax.imshow(grid, cmap='binary')
            
            if path:
                path_array = np.array(path)
                ax.plot(path_array[:, 1], path_array[:, 0], color=colors[name], linewidth=2, label=name)
                ax.scatter([0], [0], color='green', s=100, marker='o', label='Start')
                ax.scatter([grid.shape[0]-1], [grid.shape[1]-1], color='red', s=100, marker='*', label='Goal')
            
            ax.set_title(f"{name} (Path Length: {len(path) if path else 'N/A'})")
            ax.legend()
            ax.axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_corridor_adaptation(grid: np.ndarray, config: Config):
        """Visualize adaptive corridor"""
        start = (0, 0)
        goal = (config.grid_size - 1, config.grid_size - 1)
        
        adaptive_corridor = AdaptiveCorridor(grid, config)
        line_points = bresenham_line(start, goal)
        widths = adaptive_corridor.compute_adaptive_width(line_points)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot corridor on grid
        corridor = adaptive_corridor.build_corridor(start, goal)
        corridor_mask = np.zeros_like(grid)
        for point in corridor:
            corridor_mask[point] = 1
        
        combined = grid.copy().astype(float)
        combined[corridor_mask == 1] = 0.5
        
        ax1.imshow(combined, cmap='viridis')
        ax1.plot([p[1] for p in line_points], [p[0] for p in line_points], 'r-', linewidth=2)
        ax1.set_title("Adaptive Corridor Visualization")
        ax1.axis('off')
        
        # Plot width variation
        ax2.plot(widths, linewidth=2)
        ax2.set_xlabel("Position along Bresenham line")
        ax2.set_ylabel("Corridor width")
        ax2.set_title("Adaptive Width Along Path")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_results(df: pd.DataFrame):
        """Plot comprehensive results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Aggregate results
        agg_df = df.groupby(['algorithm', 'obstacle_density']).agg({
            'time_standard': 'mean',
            'time_ils': 'mean',
            'time_ails': 'mean',
            'visited_standard': 'mean',
            'visited_ils': 'mean',
            'visited_ails': 'mean'
        }).reset_index()
        
        # Plot execution time comparison
        ax = axes[0, 0]
        width = 0.25
        x = np.arange(len(agg_df['algorithm'].unique()))
        
        for i, density in enumerate(config.obstacle_densities):
            density_data = agg_df[agg_df['obstacle_density'] == density]
            ax.bar(x - width + i*width, density_data['time_standard'], width, label=f'{density*100:.0f}% Standard')
            ax.bar(x + i*width, density_data['time_ils'], width, label=f'{density*100:.0f}% ILS', alpha=0.7)
            ax.bar(x + width + i*width, density_data['time_ails'], width, label=f'{density*100:.0f}% AILS', alpha=0.5)
        
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Execution Time (s)')
        ax.set_title('Execution Time Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(agg_df['algorithm'].unique())
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot visited nodes comparison
        ax = axes[0, 1]
        for i, density in enumerate(config.obstacle_densities):
            density_data = agg_df[agg_df['obstacle_density'] == density]
            ax.bar(x - width + i*width, density_data['visited_standard'], width, label=f'{density*100:.0f}% Standard')
            ax.bar(x + i*width, density_data['visited_ils'], width, label=f'{density*100:.0f}% ILS', alpha=0.7)
            ax.bar(x + width + i*width, density_data['visited_ails'], width, label=f'{density*100:.0f}% AILS', alpha=0.5)
        
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Visited Nodes')
        ax.set_title('Visited Nodes Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(agg_df['algorithm'].unique())
        
        # Plot improvement percentages
        ax = axes[0, 2]
        improvement_data = df.groupby(['algorithm'])['improvement_ails_vs_standard'].mean()
        bars = ax.bar(range(len(improvement_data)), improvement_data.values, color='green', alpha=0.7)
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Improvement (%)')
        ax.set_title('AILS vs Standard - Average Improvement')
        ax.set_xticks(range(len(improvement_data)))
        ax.set_xticklabels(improvement_data.index, rotation=45)
        
        for bar, val in zip(bars, improvement_data.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}%', ha='center', va='bottom')
        
        # Plot AILS vs ILS improvement
        ax = axes[1, 0]
        improvement_data = df.groupby(['algorithm'])['improvement_ails_vs_ils'].mean()
        bars = ax.bar(range(len(improvement_data)), improvement_data.values, color='blue', alpha=0.7)
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Improvement (%)')
        ax.set_title('AILS vs ILS - Average Improvement')
        ax.set_xticks(range(len(improvement_data)))
        ax.set_xticklabels(improvement_data.index, rotation=45)
        
        for bar, val in zip(bars, improvement_data.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}%', ha='center', va='bottom')
        
        # Plot path length comparison
        ax = axes[1, 1]
        path_data = df.groupby(['algorithm'])[['path_length_standard', 'path_length_ils', 'path_length_ails']].mean()
        path_data.plot(kind='bar', ax=ax)
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Average Path Length')
        ax.set_title('Path Length Comparison')
        ax.legend(['Standard', 'ILS', 'AILS'])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        
        # Plot density impact
        ax = axes[1, 2]
        for alg in df['algorithm'].unique():
            alg_data = df[df['algorithm'] == alg].groupby('obstacle_density')['improvement_ails_vs_standard'].mean()
            ax.plot([d*100 for d in alg_data.index], alg_data.values, marker='o', label=alg)
        
        ax.set_xlabel('Obstacle Density (%)')
        ax.set_ylabel('AILS Improvement (%)')
        ax.set_title('Impact of Obstacle Density on AILS Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# ============================================================================
# MAIN EXPERIMENT EXECUTION
# ============================================================================

def run_complete_experiment():
    """Run the complete experimental evaluation"""
    
    print("=" * 60)
    print("ADAPTIVE INCREMENTAL LINE SEARCH (AILS)")
    print("Complete Experimental Evaluation")
    print("=" * 60)
    
    # Run experiments
    runner = ExperimentRunner(config)
    results_df = runner.run_experiments()
    
    # Display summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    summary = results_df.groupby(['algorithm']).agg({
        'improvement_ails_vs_standard': ['mean', 'std'],
        'improvement_ails_vs_ils': ['mean', 'std'],
        'time_ails': 'mean',
        'visited_ails': 'mean'
    }).round(2)
    
    display(summary)
    
    # Detailed comparison table
    print("\n" + "=" * 60)
    print("DETAILED COMPARISON BY OBSTACLE DENSITY")
    print("=" * 60)
    
    detailed = results_df.groupby(['algorithm', 'obstacle_density']).agg({
        'time_standard': 'mean',
        'time_ils': 'mean',
        'time_ails': 'mean',
        'improvement_ails_vs_standard': 'mean',
        'improvement_ails_vs_ils': 'mean'
    }).round(4)
    
    display(detailed)
    
    # Visualize results
    print("\n" + "=" * 60)
    print("VISUALIZATION")
    print("=" * 60)
    
    Visualizer.plot_results(results_df)
    
    # Demo with single grid
    print("\n" + "=" * 60)
    print("DEMO: ADAPTIVE CORRIDOR VISUALIZATION")
    print("=" * 60)
    
    demo_grid = GridMapGenerator.generate(config.grid_size, 0.2, seed=42)
    Visualizer.plot_corridor_adaptation(demo_grid, config)
    
    # Compare paths on same grid
    print("\n" + "=" * 60)
    print("DEMO: PATH COMPARISON")
    print("=" * 60)
    
    start = (0, 0)
    goal = (config.grid_size - 1, config.grid_size - 1)
    
    path_std, _, _ = PathfindingAlgorithms.a_star(demo_grid, start, goal, None)
    
    ils = IncrementalLineSearch(demo_grid)
    path_ils, _, _ = ils.search(PathfindingAlgorithms.a_star, start, goal)
    
    ails = AdaptiveIncrementalLineSearch(demo_grid, config)
    path_ails, _, _ = ails.search(PathfindingAlgorithms.a_star, start, goal)
    
    paths = {
        'Standard': path_std,
        'ILS': path_ils,
        'AILS': path_ails
    }
    
    Visualizer.plot_grid_with_paths(demo_grid, paths)
    
    return results_df

# ============================================================================
# QUICK DEMO FUNCTION
# ============================================================================

def quick_demo():
    """Run a quick demonstration"""
    
    print("Quick Demo: Adaptive ILS vs Original ILS vs Standard A*")
    print("-" * 60)
    
    # Generate test grid
    grid = GridMapGenerator.generate(50, 0.2, seed=123)
    start = (0, 0)
    goal = (49, 49)
    
    # Run standard A*
    path_std, visited_std, time_std = PathfindingAlgorithms.a_star(grid, start, goal, None)
    print(f"Standard A*: Time={time_std:.4f}s, Visited={visited_std}, Path Length={len(path_std)}")
    
    # Run original ILS
    ils = IncrementalLineSearch(grid, initial_width=3, max_width=15, step=2)
    path_ils, visited_ils, time_ils = ils.search(PathfindingAlgorithms.a_star, start, goal)
    print(f"Original ILS: Time={time_ils:.4f}s, Visited={visited_ils}, Path Length={len(path_ils)}")
    
    # Run adaptive ILS
    test_config = Config(grid_size=50, num_trials=1, min_corridor_width=1, max_corridor_width=10)
    ails = AdaptiveIncrementalLineSearch(grid, test_config)
    path_ails, visited_ails, time_ails = ails.search(PathfindingAlgorithms.a_star, start, goal)
    print(f"Adaptive ILS: Time={time_ails:.4f}s, Visited={visited_ails}, Path Length={len(path_ails)}")
    
    print("-" * 60)
    print(f"ILS Improvement over Standard: {(time_std - time_ils)/time_std*100:.1f}%")
    print(f"AILS Improvement over Standard: {(time_std - time_ails)/time_std*100:.1f}%")
    print(f"AILS Improvement over ILS: {(time_ils - time_ails)/time_ils*100:.1f}%")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (name, path, visited) in enumerate([
        ("Standard A*", path_std, visited_std),
        ("Original ILS", path_ils, visited_ils),
        ("Adaptive ILS", path_ails, visited_ails)
    ]):
        ax = axes[idx]
        ax.imshow(grid, cmap='binary')
        if path:
            path_array = np.array(path)
            ax.plot(path_array[:, 1], path_array[:, 0], 'r-', linewidth=2)
        ax.set_title(f"{name}\nVisited: {visited}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# RUN THE EXPERIMENTS
# ============================================================================

if __name__ == "__main__":
    # You can run either the quick demo or the full experiment
    
    print("Choose an option:")
    print("1. Quick Demo (fast)")
    print("2. Full Experiment (comprehensive but slower)")
    
    # For Jupyter notebook, just call the function directly:
    # quick_demo()  # For a quick demonstration
    # OR
    # results = run_complete_experiment()  # For full experimental evaluation
    
    # Default: run quick demo
    quick_demo()
