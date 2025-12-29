"""
Authors: Amr Elshahed 
Institution: Universiti Sains Malaysia

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
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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
os.makedirs('results/plots', exist_ok=True)
os.makedirs('results/data', exist_ok=True)
os.makedirs('results/analysis', exist_ok=True)


# ============================================================================
# GLOBAL EXPERIMENT CONSTANTS 
# ============================================================================

# The grid sizes to be tested (in number of cells per side)
grid_sizes: List[int] = [50, 100, 200, 300, 400, 500]

# The grid patterns to be tested
patterns: List[str] = ['open', 'maze', 'uniform', 'clustered']

# The pathfinding algorithms to be compared
algorithms: List[str] = [
    'astar_std',             # Standard A* baseline
    'astar_ails_base',       # AILS with basic line-of-sight corridor
    'astar_ails_predictive', # AILS with advanced, predictive corridor
]

# ============================================================================
# DATACLASSES AND ENUMS
# ============================================================================

@dataclass
class PathfindingConfig:
    """Configuration for a single pathfinding test run."""
    grid_size: int
    pattern: str
    density: float
    algorithm: str
    num_random_pairs: int = 100
    
@dataclass
class GridConfig:
    """Configuration for the grid generation."""
    size: int
    density: float
    pattern: str

# Define the possible movement directions (8-connectivity)
DIRECTIONS = [
    (0, 1), (0, -1), (1, 0), (-1, 0),  # Cardinal
    (1, 1), (1, -1), (-1, 1), (-1, -1) # Diagonal
]

# Define costs for movement
COST_CARDINAL = 1.0
COST_DIAGONAL = 1.41421356 # sqrt(2)

# ============================================================================
# CORE COMPONENTS (GRID, HEURISTICS, A*)
# ============================================================================

class AdvancedGridGenerator:
    """
    Generates complex, customizable 2D grids for pathfinding experiments.
    """
    @staticmethod
    def create_grid(config: GridConfig) -> np.ndarray:
        """Generates a grid based on the provided configuration."""
        if config.pattern == 'open':
            grid = AdvancedGridGenerator._generate_open(config.size, config.density)
        elif config.pattern == 'maze':
            grid = AdvancedGridGenerator._generate_maze(config.size, config.density)
        elif config.pattern == 'uniform':
            grid = AdvancedGridGenerator._generate_uniform(config.size, config.density)
        elif config.pattern == 'clustered':
            grid = AdvancedGridGenerator._generate_clustered(config.size, config.density)
        else:
            raise ValueError(f"Unknown pattern: {config.pattern}")
        
        # Ensure the edges are always blocked for boundary stability
        grid[0, :] = grid[-1, :] = 1
        grid[:, 0] = grid[:, -1] = 1
        
        return grid

    @staticmethod
    def _generate_open(size: int, density: float) -> np.ndarray:
        """Generates a grid with a low, uniform obstacle distribution."""
        grid = np.zeros((size, size), dtype=int)
        num_obstacles = int(size * size * density)
        
        # Simple random placement
        coords = np.random.choice(size * size, num_obstacles, replace=False)
        rows, cols = np.unravel_index(coords, (size, size))
        grid[rows, cols] = 1
        return grid

    @staticmethod
    def _generate_uniform(size: int, density: float) -> np.ndarray:
        """Generates a grid with obstacles uniformly distributed."""
        return (np.random.rand(size, size) < density).astype(int)

    @staticmethod
    def _generate_maze(size: int, density: float) -> np.ndarray:
        """
        Generates a maze-like structure using Randomized Prim's Algorithm.
        Density controls the amount of open space (inverted logic).
        """
        # Start with a grid of walls
        grid = np.ones((size, size), dtype=int)
        
        # Ensure size is odd for maze generation, handle small sizes
        if size < 5: 
            return AdvancedGridGenerator._generate_uniform(size, density)
        
        # Choose a random starting cell (must be odd coordinates)
        start_row, start_col = (random.randrange(1, size-1, 2), 
                                random.randrange(1, size-1, 2))
        
        grid[start_row, start_col] = 0 # Mark as passage
        walls = []
        
        # Add the starting cell's walls to the list
        for dr, dc in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
            r, c = start_row + dr, start_col + dc
            if 0 < r < size - 1 and 0 < c < size - 1:
                walls.append((r, c, dr//2, dc//2)) # (wall_r, wall_c, dir_r, dir_c)

        while walls:
            # Pick a random wall
            wall_index = random.randrange(len(walls))
            wr, wc, dr, dc = walls.pop(wall_index)
            
            # The cell on the other side of the wall
            nr, nc = wr + dr, wc + dc
            
            if grid[nr, nc] == 1: # If the cell is a wall
                grid[wr, wc] = 0  # Knock down the wall (passage)
                grid[nr, nc] = 0  # Make the new cell a passage
                
                # Add the new cell's walls
                for dr_new, dc_new in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                    r, c = nr + dr_new, nc + dc_new
                    if 0 < r < size - 1 and 0 < c < size - 1 and grid[r, c] == 1:
                        # Only add walls that lead to unvisited (wall) cells
                        if grid[nr + dr_new//2, nc + dc_new//2] == 1:
                            walls.append((r, c, dr_new//2, dc_new//2))
        
        # Optionally add extra obstacles to meet density requirement (if density > 0)
        # This is a simplification; for proper mazes, density should be ignored.
        # Here we use it as a 'difficulty' multiplier for the maze walls
        if density < 1.0:
            grid[grid == 1] = (np.random.rand(grid.sum()) < (1 - density)).astype(int)
            
        return grid

    @staticmethod
    def _generate_clustered(size: int, density: float) -> np.ndarray:
        """Generates a grid with obstacles tending to form clusters."""
        grid = np.zeros((size, size), dtype=int)
        num_obstacles = int(size * size * density)
        
        if num_obstacles == 0:
            return grid
        
        # Start a few "seeds"
        num_seeds = max(1, int(size * density / 5))
        seeds = [(random.randint(1, size-2), random.randint(1, size-2)) for _ in range(num_seeds)]
        
        placed_count = 0
        grid_copy = np.zeros_like(grid)
        
        # Simple iterative clustering (similar to fire spreading)
        while placed_count < num_obstacles:
            if not seeds:
                # If seeds run out, pick a random unblocked cell to start a new cluster
                unblocked = np.argwhere(grid == 0)
                if len(unblocked) > 0:
                    seeds.append(tuple(unblocked[random.randint(0, len(unblocked)-1)]))
                else:
                    break # Grid is full
            
            seed_r, seed_c = seeds.pop(random.randrange(len(seeds)))
            
            # Try to place an obstacle at or near the seed
            for _ in range(5): # Try 5 times to place near the seed
                dr, dc = random.choice(DIRECTIONS)
                r, c = seed_r + dr, seed_c + dc
                
                if 0 < r < size - 1 and 0 < c < size - 1 and grid[r, c] == 0:
                    # Place obstacle
                    grid[r, c] = 1
                    placed_count += 1
                    
                    # Add its neighbors to the potential next seeds
                    for n_dr, n_dc in DIRECTIONS:
                        nr, nc = r + n_dr, c + n_dc
                        if 0 < nr < size - 1 and 0 < nc < size - 1 and grid[nr, nc] == 0:
                            if (nr, nc) not in seeds:
                                seeds.append((nr, nc))
                            
                    if placed_count >= num_obstacles:
                        break
            
        return grid


class Heuristics:
    """Collection of pathfinding heuristics."""
    @staticmethod
    def euclidean(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance heuristic (L2 norm)."""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    @staticmethod
    def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic (L1 norm)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    @staticmethod
    def chebyshev(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Chebyshev distance heuristic (max norm)."""
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


class AStar:
    """Standard A* pathfinding algorithm implementation."""
    
    def __init__(self, grid: np.ndarray, heuristic_func):
        """Initializes A* with a grid and heuristic function."""
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.h = heuristic_func

    def _get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Returns valid neighbors of a given node."""
        neighbors = []
        r, c = node
        
        for i, (dr, dc) in enumerate(DIRECTIONS):
            nr, nc = r + dr, c + dc
            
            # Check bounds and if the cell is not an obstacle (0 is open, 1 is obstacle)
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr, nc] == 0:
                neighbors.append((nr, nc))
                
        return neighbors

    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int], 
                  corridor: Optional[Set[Tuple[int, int]]] = None) -> Dict[str, Any]:
        """
        Finds the shortest path from start to goal.
        
        Args:
            start: Starting coordinates (row, col).
            goal: Goal coordinates (row, col).
            corridor: An optional set of allowed nodes (AILS corridor).
                      If None, the entire grid is the search space (Standard A*).
                      
        Returns:
            A dictionary containing path, cost, time, and metrics.
        """
        start_time = time.time()
        
        if self.grid[start] == 1 or self.grid[goal] == 1:
            return {'path': [], 'cost': 0, 'time': 0, 'nodes_visited': 0, 'path_found': False}
        
        # Priority queue: (f_score, g_score, node)
        open_set = [(0, 0, start)]
        
        # g_score: actual cost from start to node
        g_scores: Dict[Tuple[int, int], float] = defaultdict(lambda: float('inf'))
        g_scores[start] = 0
        
        # came_from: tracks the path
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        nodes_visited = 0

        while open_set:
            current_f, current_g, current_node = heapq.heappop(open_set)
            
            if current_node == goal:
                # Reconstruct path
                path = []
                temp = current_node
                while temp in came_from:
                    path.append(temp)
                    temp = came_from[temp]
                path.append(start)
                path.reverse()
                
                end_time = time.time()
                return {
                    'path': path, 
                    'cost': current_g, 
                    'time': end_time - start_time, 
                    'nodes_visited': nodes_visited, 
                    'path_found': True
                }

            nodes_visited += 1

            for neighbor in self._get_neighbors(current_node):
                
                # AILS check: If a corridor is active, only search within it
                if corridor is not None and neighbor not in corridor:
                    continue
                    
                # Calculate step cost
                dr, dc = neighbor[0] - current_node[0], neighbor[1] - current_node[1]
                step_cost = COST_DIAGONAL if abs(dr) == 1 and abs(dc) == 1 else COST_CARDINAL
                
                tentative_g_score = current_g + step_cost
                
                if tentative_g_score < g_scores[neighbor]:
                    # This path is better. Record it.
                    came_from[neighbor] = current_node
                    g_scores[neighbor] = tentative_g_score
                    
                    h_score = self.h(neighbor, goal)
                    f_score = tentative_g_score + h_score
                    
                    heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))

        # Path not found
        end_time = time.time()
        return {
            'path': [], 
            'cost': 0, 
            'time': end_time - start_time, 
            'nodes_visited': nodes_visited, 
            'path_found': False
        }


# ============================================================================
# AILS COMPONENTS (CORRIDOR BUILDING)
# ============================================================================

class CorridorBuilder:
    """
    Handles the generation of the Adaptive Incremental Line Search (AILS) corridor.
    """
    
    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.rows, self.cols = grid.shape
        
    def is_los(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
        """
        Checks for Line-of-Sight (LoS) between two points using Bresenham's algorithm.
        Returns True if LoS exists (no obstacles in between), False otherwise.
        """
        r1, c1 = p1
        r2, c2 = p2
        
        # If the points are the same, LoS exists
        if p1 == p2: return True
        
        dr = abs(r2 - r1)
        dc = abs(c2 - c1)
        sr = 1 if r1 < r2 else -1
        sc = 1 if c1 < c2 else -1
        
        r, c = r1, c1
        
        if dr > dc:
            d = dr / 2
            while r != r2:
                r += sr
                d -= dc
                if d < 0:
                    c += sc
                    d += dr
                
                if self.grid[r, c] == 1:
                    return False
        else:
            d = dc / 2
            while c != c2:
                c += sc
                d -= dr
                if d < 0:
                    r += sr
                    d += dc
                    
                if self.grid[r, c] == 1:
                    return False

        return True

    def _get_los_neighbors(self, point: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Returns neighbors that are visible from the point."""
        los_neighbors = []
        r, c = point
        
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr, nc] == 0:
                # Optimized LoS check: only check if the move is non-cardinal, 
                # as cardinal moves are always LoS to immediate neighbors.
                # However, for the expanded corridor, we rely on full LoS check to boundary of LOS.
                # Here, we only use immediate neighbors for BFS expansion.
                
                # A full check is safer for the intended use in this complex environment
                if self.is_los(point, (nr, nc)):
                    los_neighbors.append((nr, nc))
                    
        return los_neighbors
    
    def _expand_corridor_bfs(self, seed: Tuple[int, int], max_radius: int) -> Set[Tuple[int, int]]:
        """Performs a bounded BFS to expand the corridor from a seed point."""
        corridor = {seed}
        queue = deque([(seed, 0)]) # (node, distance)
        visited = {seed}
        
        while queue:
            current_node, dist = queue.popleft()
            
            if dist >= max_radius:
                continue
            
            # IMPORTANT: We use standard 8-connectivity neighbors here, NOT LoS neighbors,
            # to expand the corridor outwards like a "buffer."
            r, c = current_node
            for dr, dc in DIRECTIONS:
                neighbor = (r + dr, c + dc)
                nr, nc = neighbor
            
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr, nc] == 0:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        corridor.add(neighbor)
                        queue.append((neighbor, dist + 1))
        
        return corridor

    def compute_adaptive_corridor(self, start: Tuple[int, int], goal: Tuple[int, int], 
                                  strategy: str) -> Set[Tuple[int, int]]:
        """
        Generates the AILS corridor based on the chosen strategy.
        
        Args:
            start: Starting coordinates.
            goal: Goal coordinates.
            strategy: 'base' (simple LOS) or 'predictive' (expanded/adaptive).
            
        Returns:
            A set of nodes that form the search corridor.
        """
        
        # 1. Compute the core Line-of-Sight corridor using Bresenham's line points
        core_los_nodes: Set[Tuple[int, int]] = set()
        r1, c1 = start
        r2, c2 = goal
        
        dr_abs = abs(r2 - r1)
        dc_abs = abs(c2 - c1)
        sr = 1 if r1 < r2 else -1
        sc = 1 if c1 < c2 else -1
        
        r, c = r1, c1
        
        if dr_abs >= dc_abs:
            d = dr_abs / 2
            while True:
                if 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r, c] == 0:
                    core_los_nodes.add((r, c))
                
                if r == r2 and c == c2: break
                
                r += sr
                d -= dc_abs
                if d < 0:
                    c += sc
                    d += dr_abs
        else:
            d = dc_abs / 2
            while True:
                if 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r, c] == 0:
                    core_los_nodes.add((r, c))
                
                if r == r2 and c == c2: break
                
                c += sc
                d -= dr_abs
                if d < 0:
                    r += sr
                    d += dc_abs

        
        if not core_los_nodes:
            # If the direct line is completely blocked (rare, but possible), return empty
            return set()
            
        if strategy == 'astar_ails_base' or strategy == 'base':
            # Base AILS: The corridor is just the core LoS line.
            return core_los_nodes
            
        elif strategy == 'astar_ails_predictive' or strategy == 'predictive':
            # Predictive AILS: LoS plus a bounded expansion.
            # The radius is adaptive based on grid size (e.g., 5% of grid size, min 5)
            max_radius = max(5, int(self.rows * 0.05)) 
            
            full_corridor: Set[Tuple[int, int]] = set(core_los_nodes)
            
            # Use points on the core line as seeds for the expansion
            for seed in core_los_nodes:
                # Expand a small 'bubble' around each core LoS point
                expanded_nodes = self._expand_corridor_bfs(seed, max_radius)
                full_corridor.update(expanded_nodes)
                
            return full_corridor
            
        else:
            raise ValueError(f"Unknown AILS strategy: {strategy}")


# ============================================================================
# EXPERIMENT FRAMEWORK
# ============================================================================

class ExperimentRunner:
    """Manages the configuration, execution, and data collection of experiments."""
    
    def __init__(self, config: PathfindingConfig):
        self.config = config
        self.grid = AdvancedGridGenerator.create_grid(
            GridConfig(config.grid_size, 0.3, config.pattern)
        )
        self.astar_engine = AStar(self.grid, Heuristics.chebyshev)
        self.corridor_builder = CorridorBuilder(self.grid)
        self.results: List[Dict[str, Any]] = []

    def _generate_random_pairs(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Generates a list of valid (start, goal) pairs."""
        valid_cells = np.argwhere(self.grid == 0)
        
        if len(valid_cells) < 2:
            warnings.warn(f"Grid {self.config.grid_size}x{self.config.grid_size} is too dense, cannot find pairs.")
            return []
            
        pairs = []
        for _ in range(self.config.num_random_pairs):
            start_idx, goal_idx = np.random.choice(len(valid_cells), 2, replace=False)
            start = tuple(valid_cells[start_idx])
            goal = tuple(valid_cells[goal_idx])
            pairs.append((start, goal))
            
        return pairs

    def _run_single_test(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Dict[str, Any]:
        """Runs the pathfinding algorithm and records results."""
        
        result: Dict[str, Any] = {
            'grid_size': self.config.grid_size,
            'pattern': self.config.pattern,
            'algorithm': self.config.algorithm,
            'start': start,
            'goal': goal,
            'time': 0,
            'cost': 0,
            'nodes_visited': 0,
            'path_found': False,
            'is_optimal': False, # Will be set by comparison
            'corridor_size': 0 # For AILS
        }
        
        corridor = None
        
        if 'ails' in self.config.algorithm:
            corridor = self.corridor_builder.compute_adaptive_corridor(start, goal, self.config.algorithm)
            result['corridor_size'] = len(corridor)
            if not corridor:
                # If no corridor is found, path is impossible in this strategy
                return result
                
        # Run the A* search
        search_result = self.astar_engine.find_path(start, goal, corridor)
        
        result.update({
            'time': search_result['time'],
            'cost': search_result['cost'],
            'nodes_visited': search_result['nodes_visited'],
            'path_found': search_result['path_found'],
        })
        
        return result

    def run_experiment(self) -> List[Dict[str, Any]]:
        """Runs the configured number of tests."""
        
        # Generate pairs once per grid configuration
        pairs = self._generate_random_pairs()
        
        if not pairs:
            print(f"Skipping {self.config.pattern} {self.config.grid_size} due to no valid start/goal pairs.")
            return []
            
        print(f"-> Running {self.config.algorithm} on {self.config.pattern} {self.config.grid_size}x{self.config.grid_size} ({len(pairs)} pairs)")
        
        for start, goal in tqdm(pairs, desc=f"   {self.config.algorithm}"):
            res = self._run_single_test(start, goal)
            self.results.append(res)
            
        return self.results

    @staticmethod
    def _process_experiment_chunk(config_tuple: Tuple[int, str, str]) -> List[Dict[str, Any]]:
        """Wrapper for multiprocessing to run a single configuration."""
        grid_size, pattern, algorithm = config_tuple
        
        # Use a fixed density for comprehensive test
        density = 0.3 
        
        runner = ExperimentRunner(PathfindingConfig(
            grid_size=grid_size,
            pattern=pattern,
            density=density,
            algorithm=algorithm,
            num_random_pairs=100
        ))
        return runner.run_experiment()

    @staticmethod
    def run_comprehensive_experiment() -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Runs the full suite of experiments across all grid sizes, patterns, and algorithms.
        Uses multiprocessing for parallel execution of configurations.
        """
        
        print("\n" + "="*80)
        print("COMPREHENSIVE PATHFINDING EXPERIMENT")
        print("="*80)
        print(f"Grid Sizes: {grid_sizes}")
        print(f"Patterns: {patterns}")
        print(f"Algorithms: {algorithms}")
        print(f"Random Pairs per Config: 100")
        
        # Number of unique configurations
        num_configs = len(grid_sizes) * len(patterns) * len(algorithms)
        total_runs = num_configs * 100 
        
        print(f"Total pathfinding runs (est): {total_runs:,}")
        print(f"Total unique configurations: {num_configs}")
        print("-" * 80)
        
        all_configs = list(itertools.product(grid_sizes, patterns, algorithms))
        
        # Use a maximum of 4 processes to avoid excessive overhead
        num_processes = min(4, mp.cpu_count()) 
        
        all_results = []
        
        # Non-parallel execution for stability in notebooks/environments where multiprocessing is complex
        print(f"Using sequential execution for stability (Num configs: {len(all_configs)})")
        for config in all_configs:
            results_chunk = ExperimentRunner._process_experiment_chunk(config)
            all_results.extend(results_chunk)
            
        df = pd.DataFrame(all_results)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Ensure 'path_found' and 'cost' are not correlated to time if path was not found
        df['time'] = np.where(df['path_found'], df['time'], 0)
        
        print("\nExperiment Complete. Performing Post-Processing...")
        
        # --- Post-Processing 1: Optimality Check ---
        df_optimal = df[df['algorithm'] == 'astar_std'].copy()
        df_optimal.rename(columns={'cost': 'optimal_cost'}, inplace=True)
        df_optimal = df_optimal[['grid_size', 'pattern', 'start', 'goal', 'optimal_cost']]
        
        # Merge on all relevant columns to get the standard A* cost
        df = pd.merge(df, df_optimal, on=['grid_size', 'pattern', 'start', 'goal'], how='left')
        
        # An algorithm's cost is optimal if it matches the standard A* cost
        df['is_optimal'] = (df['cost'] == df['optimal_cost'])
        df.drop(columns=['optimal_cost'], inplace=True)
        
        # --- Post-Processing 2: Data Cleaning and Summarization ---
        df = df[df['path_found']].copy() # Only keep successful runs
        
        # --- Statistical Analysis ---
        stats_results = StatsAnalyzer.analyze_data(df)
        
        print("Analysis Complete. Saving results...")
        df.to_pickle(f'results/data/comprehensive_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
        
        print(f"Results saved to results/data/")
        print("="*80)
        
        return df, stats_results

    @staticmethod
    def run_parameter_sensitivity_experiment() -> pd.DataFrame:
        """
        Runs a focused experiment to analyze the sensitivity of AILS to
        key parameters (e.g., density, corridor radius proxy).
        """
        
        print("\n" + "="*80)
        print("PARAMETER SENSITIVITY ANALYSIS (PSA)")
        print("="*80)
        
        # Parameters for PSA
        PSA_SIZES = [100, 300]
        PSA_PATTERNS = ['uniform', 'maze']
        PSA_DENSITIES = [0.1, 0.3, 0.5, 0.7]
        PSA_ALGORITHMS = ['astar_std', 'astar_ails_predictive']
        
        # Total runs
        num_configs = len(PSA_SIZES) * len(PSA_PATTERNS) * len(PSA_DENSITIES) * len(PSA_ALGORITHMS)
        total_runs = num_configs * 50 # Reduced pairs for PSA
        
        print(f"Sizes: {PSA_SIZES}, Patterns: {PSA_PATTERNS}, Densities: {PSA_DENSITIES}")
        print(f"Total pathfinding runs (est): {total_runs:,}")
        
        all_configs = list(itertools.product(PSA_SIZES, PSA_PATTERNS, PSA_DENSITIES, PSA_ALGORITHMS))
        
        all_results = []
        for size, pattern, density, algorithm in all_configs:
            runner = ExperimentRunner(PathfindingConfig(
                grid_size=size,
                pattern=pattern,
                density=density,
                algorithm=algorithm,
                num_random_pairs=50 # Use fewer pairs for faster PSA
            ))
            results_chunk = runner.run_experiment()
            all_results.extend(results_chunk)
            
        df = pd.DataFrame(all_results)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df[df['path_found']].copy()
        
        # Check optimality against the std A* for this density/size/pattern
        df_optimal = df[df['algorithm'] == 'astar_std'].copy()
        df_optimal.rename(columns={'cost': 'optimal_cost'}, inplace=True)
        df_optimal = df_optimal[['grid_size', 'pattern', 'start', 'goal', 'density', 'optimal_cost']]

        df = pd.merge(df, df_optimal, on=['grid_size', 'pattern', 'start', 'goal', 'density'], how='left')
        df['is_optimal'] = (df['cost'] == df['optimal_cost'])
        df.drop(columns=['optimal_cost'], inplace=True)

        df.to_pickle(f'results/data/psa_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
        print("PSA Results saved to results/data/")
        print("="*80)
        
        return df


# ============================================================================
# ANALYSIS AND PLOTTING
# ============================================================================

class StatsAnalyzer:
    """Performs statistical analysis on experiment results."""

    @staticmethod
    def analyze_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Performs statistical tests."""
        stats_results: Dict[str, Any] = {}
        
        # 1. Performance Metric Ratios (AILS/Std A*)
        print("\n--- Calculating Performance Ratios (AILS vs. Standard A*) ---")
        
        df_agg = df.groupby(['grid_size', 'pattern', 'algorithm']).agg(
            mean_time=('time', 'mean'),
            mean_nodes=('nodes_visited', 'mean'),
            std_time=('time', 'std'),
            std_nodes=('nodes_visited', 'std')
        ).reset_index()
        
        # Merge AILS and STD results for ratio calculation
        df_std = df_agg[df_agg['algorithm'] == 'astar_std'].copy()
        df_std.rename(columns={'mean_time': 'std_time_mean', 
                               'mean_nodes': 'std_nodes_mean'}, inplace=True)
        df_std = df_std.drop(columns=['algorithm', 'std_time', 'std_nodes'])

        df_ails = df_agg[df_agg['algorithm'] != 'astar_std'].copy()

        df_ratios = pd.merge(df_ails, df_std, on=['grid_size', 'pattern'])

        df_ratios['time_ratio'] = df_ratios['mean_time'] / df_ratios['std_time_mean']
        df_ratios['nodes_ratio'] = df_ratios['mean_nodes'] / df_ratios['std_nodes_mean']
        
        stats_results['ratios'] = df_ratios
        print("Ratios calculated.")

        # 2. ANOVA and Tukey HSD (Post-hoc for multiple comparisons)
        print("\n--- Performing ANOVA and Tukey HSD for Time and Nodes Visited ---")
        tukey_results = {}
        
        for metric in ['time', 'nodes_visited']:
            print(f"  > Analyzing {metric}...")
            # Perform ANOVA: null hypothesis is that all algorithm means are equal
            f_val, p_val = stats.f_oneway(
                df[df['algorithm'] == 'astar_std'][metric],
                df[df['algorithm'] == 'astar_ails_base'][metric],
                df[df['algorithm'] == 'astar_ails_predictive'][metric]
            )
            print(f"    - Overall ANOVA (F={f_val:.2f}, P={p_val:.4f}).")
            stats_results[f'anova_{metric}'] = {'F': f_val, 'P': p_val}

            # Perform Tukey HSD only if ANOVA suggests a difference (P < 0.05)
            if p_val < 0.05:
                tukey = pairwise_tukeyhsd(endog=df[metric], groups=df['algorithm'], alpha=0.05)
                tukey_results[metric] = pd.DataFrame(data=tukey._results_table.data[1:], 
                                                     columns=tukey._results_table.data[0])
                print("    - Tukey HSD results recorded.")
            else:
                tukey_results[metric] = None
                print("    - Tukey HSD skipped (ANOVA was non-significant).")
                
        stats_results['tukey'] = tukey_results
        
        return stats_results


class PlotGenerator:
    """Generates publication-quality plots from the experiment data."""

    @staticmethod
    def generate_all_plots(df: pd.DataFrame, stats_results: Dict):
        """Generates all standard plots."""
        print("\n" + "="*80)
        print("PLOTTING RESULTS")
        print("="*80)

        PlotGenerator._plot_scaling_performance(df, stats_results['ratios'])
        PlotGenerator._plot_optimality_ratio(df)
        PlotGenerator._plot_corridor_size_vs_nodes(df)
        PlotGenerator._plot_algorithm_comparison_boxplot(df)
        
        print("Plots saved to results/plots/")
        
    @staticmethod
    def _plot_scaling_performance(df: pd.DataFrame, df_ratios: pd.DataFrame):
        """Plots the time and node count ratios as a function of grid size."""
        fig, axes = plt.subplots(nrows=2, ncols=len(patterns), figsize=(3 * len(patterns), 7), 
                                 sharex=True, sharey='row')
        fig.suptitle('Performance Scaling (AILS / Standard A*)', y=1.02, fontsize=14)

        for col, pattern in enumerate(patterns):
            # --- Time Ratio Plot ---
            ax_time = axes[0, col]
            df_time = df_ratios[df_ratios['pattern'] == pattern]
            
            sns.lineplot(data=df_time, x='grid_size', y='time_ratio', 
                         hue='algorithm', style='algorithm', markers=True, ax=ax_time, legend=False)
            
            ax_time.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
            ax_time.set_title(pattern.capitalize())
            ax_time.set_xlabel('')
            ax_time.set_ylabel('Time Ratio')
            
            # --- Nodes Visited Ratio Plot ---
            ax_nodes = axes[1, col]
            df_nodes = df_ratios[df_ratios['pattern'] == pattern]
            
            sns.lineplot(data=df_nodes, x='grid_size', y='nodes_ratio', 
                         hue='algorithm', style='algorithm', markers=True, ax=ax_nodes)
            
            ax_nodes.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
            ax_nodes.set_xlabel('Grid Size (N)')
            ax_nodes.set_ylabel('Nodes Visited Ratio')
            
            if col < len(patterns) - 1:
                ax_nodes.get_legend().remove()

        # Place the legend in the last subplot
        handles, labels = axes[1, len(patterns) - 1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.95, 0.01), ncol=2, frameon=False)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        fig.savefig('results/plots/scaling_performance.png', bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def _plot_optimality_ratio(df: pd.DataFrame):
        """Plots the percentage of runs that found an optimal path."""
        fig, axes = plt.subplots(nrows=1, ncols=len(patterns), figsize=(3 * len(patterns), 4), sharey=True)
        fig.suptitle('Optimality Percentage', y=1.05)

        for col, pattern in enumerate(patterns):
            ax = axes[col]
            df_pattern = df[df['pattern'] == pattern].copy()
            
            # Calculate mean optimality for each algorithm and size
            optimality_df = df_pattern.groupby(['grid_size', 'algorithm'])['is_optimal'].mean().reset_index()
            optimality_df['is_optimal'] = optimality_df['is_optimal'] * 100 # Convert to percentage

            sns.lineplot(data=optimality_df, x='grid_size', y='is_optimal', 
                         hue='algorithm', style='algorithm', markers=True, ax=ax, legend=False)
            
            ax.set_title(pattern.capitalize())
            ax.set_xlabel('Grid Size (N)')
            ax.set_ylim(90, 101)
            
            if col == 0:
                ax.set_ylabel('Optimal Path Found (%)')
            else:
                ax.set_ylabel('')

        # Add legend
        handles, labels = axes[len(patterns) - 1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.95, 0.01), ncol=3, frameon=False)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        fig.savefig('results/plots/optimality_percentage.png', bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def _plot_corridor_size_vs_nodes(df: pd.DataFrame):
        """Plots correlation between corridor size and nodes visited for AILS."""
        fig, ax = plt.subplots(figsize=(6, 5))
        df_ails = df[df['algorithm'] != 'astar_std'].copy()

        # Use a small subset for plotting for better visualization
        df_plot = df_ails.sample(min(len(df_ails), 5000))

        sns.scatterplot(data=df_plot, x='corridor_size', y='nodes_visited', 
                        hue='algorithm', style='algorithm', alpha=0.6, ax=ax, 
                        s=10) # Smaller points for scatter

        ax.set_title('Corridor Size vs. Nodes Visited in AILS')
        ax.set_xlabel('Corridor Size (Number of Nodes)')
        ax.set_ylabel('Nodes Visited (Search Effort)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(title='Algorithm')
        plt.tight_layout()
        fig.savefig('results/plots/corridor_vs_nodes.png', bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def _plot_algorithm_comparison_boxplot(df: pd.DataFrame):
        """Box plot comparison of time and nodes across all configurations."""
        df_melt = df.melt(id_vars=['grid_size', 'pattern', 'algorithm'], 
                          value_vars=['time', 'nodes_visited'], 
                          var_name='metric', value_name='value')
        
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
        fig.suptitle('Overall Algorithm Performance Comparison', y=1.02)
        
        # Plot Time
        sns.boxplot(data=df_melt[df_melt['metric'] == 'time'], x='algorithm', y='value', 
                    hue='pattern', ax=axes[0], showfliers=False)
        axes[0].set_title('Time (s)')
        axes[0].set_xlabel('')
        axes[0].set_ylabel('Time (s)')
        axes[0].set_yscale('log')
        
        # Plot Nodes Visited
        sns.boxplot(data=df_melt[df_melt['metric'] == 'nodes_visited'], x='algorithm', y='value', 
                    hue='pattern', ax=axes[1], showfliers=False)
        axes[1].set_title('Nodes Visited')
        axes[1].set_xlabel('Algorithm')
        axes[1].set_ylabel('Nodes Visited')
        axes[1].set_yscale('log')
        
        # Adjust legend and layout
        axes[0].legend(title='Pattern', loc='upper right')
        axes[1].get_legend().remove()
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig('results/plots/algorithm_comparison_boxplot.png', bbox_inches='tight')
        plt.close(fig)


class LatexTableGenerator:
    """Generates LaTeX code for tables summarizing the results."""

    @staticmethod
    def generate_summary_table(df: pd.DataFrame) -> str:
        """Generates a summary table for mean time, nodes, and optimality."""
        summary = df.groupby(['algorithm']).agg(
            Time_Mean=('time', 'mean'),
            Time_Std=('time', 'std'),
            Nodes_Mean=('nodes_visited', 'mean'),
            Nodes_Std=('nodes_visited', 'std'),
            Optimality=('is_optimal', 'mean'),
            Path_Found=('path_found', 'mean')
        )
        
        # Convert Optimality and Path_Found to percentage display format
        summary['Optimality'] = summary['Optimality'] * 100
        summary['Path_Found'] = summary['Path_Found'] * 100
        
        latex = (
            "\\begin{table}[ht]\n"
            "\\centering\n"
            "\\caption{Overall Performance Summary of Pathfinding Algorithms}\n"
            "\\label{tab:overall_summary}\n"
            "\\begin{tabular}{lcccccc}\n"
            "\\toprule\n"
            "Algorithm & Time Mean (s) & Time Std (s) & Nodes Mean & Nodes Std & Optimality (\\%) & Success (\\%) \\\\ \n"
            "\\midrule\n"
        )
        
        for algo, row in summary.iterrows():
            latex += (
                f"{algo.replace('_', ' ').title()} & "
                f"${row['Time_Mean']:.4f}$ & ${row['Time_Std']:.4f}$ & "
                f"${row['Nodes_Mean']:.0f}$ & ${row['Nodes_Std']:.0f}$ & "
                f"${row['Optimality']:.2f}$ & ${row['Path_Found']:.2f}$ \\\\ \\hline\n"
            )
            
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        return latex

    @staticmethod
    def generate_scaling_table(df: pd.DataFrame) -> str:
        """Generates a table showing time and nodes visited scaling by grid size."""
        
        # Calculate ratios first
        df_agg = df.groupby(['grid_size', 'pattern', 'algorithm']).agg(
            mean_time=('time', 'mean'),
            mean_nodes=('nodes_visited', 'mean')
        ).reset_index()
        
        df_std = df_agg[df_agg['algorithm'] == 'astar_std'].copy()
        df_std.rename(columns={'mean_time': 'std_time_mean', 
                               'mean_nodes': 'std_nodes_mean'}, inplace=True)
        df_std = df_std.drop(columns=['algorithm'])

        df_ails = df_agg[df_agg['algorithm'] != 'astar_std'].copy()
        df_ratios = pd.merge(df_ails, df_std, on=['grid_size', 'pattern'])

        df_ratios['Time Ratio'] = df_ratios['mean_time'] / df_ratios['std_time_mean']
        df_ratios['Nodes Ratio'] = df_ratios['mean_nodes'] / df_ratios['std_nodes_mean']
        
        df_ratios['Algorithm'] = df_ratios['algorithm'].apply(lambda x: x.split('_')[-1].title())
        df_ratios['Algorithm'] = df_ratios['Algorithm'].replace({'Predictive': 'AILS-P', 'Base': 'AILS-B'})

        latex = (
            "\\begin{table}[ht]\n"
            "\\centering\n"
            "\\caption{Performance Scaling Ratios (Algorithm / Standard A*)}\n"
            "\\label{tab:scaling_ratios}\n"
            "\\begin{tabular}{llccc}\n"
            "\\toprule\n"
            "Grid Size (N) & Pattern & Algorithm & Time Ratio & Nodes Ratio \\\\ \n"
            "\\midrule\n"
        )
        
        # Sort for clean presentation
        df_ratios.sort_values(by=['grid_size', 'pattern', 'Algorithm'], inplace=True)
        
        current_size = None
        for _, row in df_ratios.iterrows():
            if current_size != row['grid_size']:
                if current_size is not None:
                    latex += "\\midrule\n"
                current_size = row['grid_size']

            latex += (
                f"${row['grid_size']}$ & {row['pattern'].capitalize()} & "
                f"{row['Algorithm']} & "
                f"${row['Time Ratio']:.3f}$ & ${row['Nodes Ratio']:.3f}$ \\\\ \n"
            )
            
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        return latex

    @staticmethod
    def generate_tukey_table(tukey_results: pd.DataFrame, metric: str) -> str:
        """Generates a LaTeX table for Tukey HSD post-hoc test results."""
        latex = (
            f"\\begin{table}[ht]\n"
            f"\\centering\n"
            f"\\caption{{Tukey HSD Post-hoc Analysis for {metric.replace('_', ' ').title()}}}\n"
            f"\\label{{tab:tukey_{metric}}}\n"
            f"\\begin{tabular}{lcccccc}\n"
            f"\\toprule\n"
            "Group 1 & Group 2 & Mean Diff. & Lower CI & Upper CI & p-adj & Reject H$_0$ \\\\ \n"
            "\\midrule\n"
        )
        
        for _, row in tukey_results.iterrows():
            latex += (f"{row['group1'].replace('_', ' ').title()} & {row['group2'].replace('_', ' ').title()} & ${row['meandiff']:.2f}$ & "
                      f"${row['lower']:.2f}$ & ${row['upper']:.2f}$ & "
                      f"${row['p-adj']:.4f}$ & {row['reject']} \\\\ \n")
            
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        return latex

    @staticmethod
    def save_tables(df: pd.DataFrame, stats_results: Dict):
        """Save all LaTeX tables to file"""
        try:
            with open(f'results/analysis/latex_tables_{datetime.now().strftime("%Y%m%d_%H%M%S")}.tex', 'w') as f:
                f.write("% Auto-generated LaTeX tables\n")
                f.write("% " + "="*50 + "\n\n")
                
                f.write(LatexTableGenerator.generate_summary_table(df))
                f.write("\n\n")
                
                f.write(LatexTableGenerator.generate_scaling_table(df))
                f.write("\n\n")

                if 'tukey' in stats_results:
                    for group_name, tukey_df in stats_results['tukey'].items():
                        if tukey_df is not None:
                            f.write(LatexTableGenerator.generate_tukey_table(tukey_df, group_name))
                            f.write("\n\n")

            print("LaTeX tables generated successfully in results/analysis/")
        except Exception as e:
            print(f"Error generating LaTeX tables: {e}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compare_single_run(grid_size: int = 100, pattern: str = 'uniform', 
                       density: float = 0.3, start_seed: int = 42):
    """
    Runs a single comparison test between A* and AILS on a generated grid.
    Helpful for debugging and quick demonstration.
    """
    print("\n" + "="*50)
    print(f"Quick Demo: {grid_size}x{grid_size} | {pattern.capitalize()}")
    print("="*50)
    
    np.random.seed(start_seed)
    random.seed(start_seed)
    
    # 1. Setup
    grid = AdvancedGridGenerator.create_grid(GridConfig(grid_size, density, pattern))
    astar = AStar(grid, Heuristics.chebyshev)
    corridor_builder = CorridorBuilder(grid)

    # Find valid start/goal (must be open cells)
    valid_cells = np.argwhere(grid == 0)
    if len(valid_cells) < 2:
        print("Error: Grid too dense, cannot find valid start/goal.")
        return

    # Pick two distant points
    idx1, idx2 = np.random.choice(len(valid_cells), 2, replace=False)
    start = tuple(valid_cells[idx1])
    goal = tuple(valid_cells[idx2])

    print(f"Start: {start}, Goal: {goal}")
    
    # 2. Run Standard A*
    print("\n--- Running Standard A* ---")
    result_std = astar.find_path(start, goal)
    
    # 3. Run Predictive AILS
    print("--- Running Predictive AILS ---")
    # AILS is A* restricted to the corridor
    corridor = corridor_builder.compute_adaptive_corridor(start, goal, 'astar_ails_predictive')
    result_ails = astar.find_path(start, goal, corridor=corridor)

    # 4. Report
    print("\n--- Results Summary ---")
    
    print(f"Standard A*: Path found: {result_std['path_found']}, Cost: {result_std['cost']:.2f}, Time: {result_std['time'] * 1000:.2f} ms, Nodes: {result_std['nodes_visited']}")
    print(f"AILS (Predictive): Path found: {result_ails['path_found']}, Cost: {result_ails['cost']:.2f}, Time: {result_ails['time'] * 1000:.2f} ms, Nodes: {result_ails['nodes_visited']}")
    print(f"Corridor Size: {len(corridor) if corridor else 0} nodes")

    # 5. Visualizer
    # NOTE: PlotGenerator.plot_paths is not included in this file but would be used here.
    # if result_std['path_found'] or result_ails['path_found']:
    #     PlotGenerator.plot_paths(grid, start, goal, result_std, result_ails, corridor_builder)
    # else:
    #     print("No path found by either algorithm.")


def quick_demo():
    """Runs a series of quick demos for visual comparison."""
    # This function is used to visually test the algorithms on different grid types.
    # It does not include plotting functions in this file, but relies on them
    # being present if the user wants to uncomment and use this block.
    print("Running a series of quick demos (requires PlotGenerator.plot_paths to show visuals, but logic will run).")
    compare_single_run(grid_size=100, pattern='uniform', density=0.3, start_seed=10)
    compare_single_run(grid_size=150, pattern='maze', density=0.5, start_seed=20)
    compare_single_run(grid_size=200, pattern='open', density=0.1, start_seed=30)
    compare_single_run(grid_size=200, pattern='clustered', density=0.4, start_seed=40)


if __name__ == "__main__":
    
    # --- EXPERIMENT EXECUTION OPTIONS ---
    
    print("\nSelect an experiment option by uncommenting the relevant lines:")
    print("1. Run quick_demo() (Visual comparison, default run)")
    print("2. Comprehensive Experiment (WARNING: Long running time)")
    print("3. Parameter Sensitivity Analysis (Medium running time)")
    print("-" * 50)

    # 1. Quick Demo (Visual comparison) - Leave this uncommented to run by default:
    quick_demo()
    
    # 2. Comprehensive Experiment (Uncomment the lines below to run the main study)
#    try:
#        print("\nStarting Comprehensive Experiment... (This will take a long time)\n")
#        df, stats = ExperimentRunner.run_comprehensive_experiment()
#        PlotGenerator.generate_all_plots(df, stats)
#        LatexTableGenerator.save_tables(df, stats)
#        print("\nComprehensive Experiment Finished. All data, plots, and tables saved.")
#    except Exception as e:
#        print(f"An error occurred during Comprehensive Experiment: {e}")
    
    # 3. Parameter Sensitivity Analysis (Uncomment the block below to run the PSA)
#    try:
#        print("\nStarting Parameter Sensitivity Analysis...\n")
#        df_psa = ExperimentRunner.run_parameter_sensitivity_experiment()
#        print("\nPSA Finished. Data saved.")
#    except Exception as e:
#        print(f"An error occurred during Parameter Sensitivity Analysis: {e}")

    print("\nScript execution finished. Remember to uncomment the desired experiment function.")