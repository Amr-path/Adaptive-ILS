# AILS Analysis - Jupyter Notebooks for MacBook

This folder contains comprehensive Jupyter notebooks for analyzing and experimenting with the **Adaptive Incremental Line Search (AILS)** algorithm.

**Author:** Amr Elshahed
**Institution:** Universiti Sains Malaysia

---

## Quick Start

### 1. Install Dependencies

```bash
cd analysis
pip install -r requirements.txt
```

### 2. Start Jupyter

```bash
jupyter notebook
```

### 3. Run Notebooks in Order

1. `01_quick_start.ipynb` - Introduction and basic usage
2. `02_experiments.ipynb` - Comprehensive experiments
3. `03_statistical_analysis.ipynb` - Statistical testing
4. `04_visualization.ipynb` - Publication-quality figures

---

## Notebook Descriptions

### 01_quick_start.ipynb
A beginner-friendly introduction to AILS:
- Basic pathfinding examples
- Grid visualization
- Simple benchmarks
- AILS configuration options

### 02_experiments.ipynb
Comprehensive experiments for research:
- **Scalability Analysis**: Grid sizes 50x50 to 500x500
- **Density Analysis**: Obstacle densities 10-40%
- **Pattern Analysis**: Random, Clustered, Maze, Room, Open
- **Parameter Sensitivity**: r_min, r_max, alpha, window_size
- **Algorithm Comparison**: A*, Dijkstra, BFS, Bidirectional A*

### 03_statistical_analysis.ipynb
Rigorous statistical testing:
- Descriptive statistics
- Paired t-tests and Wilcoxon tests
- Effect size (Cohen's d) calculations
- 95% confidence intervals
- Bonferroni correction for multiple comparisons
- LaTeX table generation

### 04_visualization.ipynb
Publication-quality figures:
- Corridor visualization
- Performance heatmaps
- Scalability line plots
- Algorithm comparison bar charts
- Obstacle pattern visualizations
- Step-by-step corridor construction

---

## Core Module: ails_core.py

The `ails_core.py` module contains all algorithm implementations:

### Algorithms
- `AILSPathfinder` - Main AILS interface
- `AStarEngine` - A* search algorithm
- `DijkstraEngine` - Dijkstra's algorithm
- `BFSEngine` - Breadth-First Search
- `BidirectionalAStarEngine` - Bidirectional A*

### Grid Generators
- `GridGenerator.generate_random()` - Random obstacles
- `GridGenerator.generate_clustered()` - Clustered obstacles
- `GridGenerator.generate_maze()` - Maze patterns
- `GridGenerator.generate_room()` - Room-based maps
- `GridGenerator.generate_open()` - Sparse obstacles

### Utilities
- `run_benchmark()` - Run benchmark experiments
- `compute_statistics()` - Calculate statistics
- `compute_improvement()` - Calculate improvement percentages
- `paired_t_test()` - Statistical hypothesis testing
- `cohens_d()` - Effect size calculation

---

## Example Usage

```python
from ails_core import AILSPathfinder, GridGenerator

# Create a grid
grid = GridGenerator.generate_random(size=100, density=0.25, seed=42)

# Initialize pathfinder
pathfinder = AILSPathfinder(grid)

# Find path using AILS
result = pathfinder.find_path_ails(
    start=(10, 10),
    goal=(90, 90),
    strategy='adaptive'
)

print(f"Path found: {result.path_found}")
print(f"Nodes visited: {result.nodes_visited}")
print(f"Time: {result.time_ms:.3f} ms")
```

---

## Configuration Options

AILS can be configured using `AILSConfig`:

```python
from ails_core import AILSConfig, AILSPathfinder

config = AILSConfig(
    r_min=5,           # Minimum corridor radius
    r_max=15,          # Maximum corridor radius
    alpha=0.8,         # Density scaling exponent
    window_size=7,     # Local density window
    max_iterations=5   # Max corridor expansions
)

pathfinder = AILSPathfinder(grid, config=config)
```

---

## Output Directories

- `results/` - CSV files and statistical reports
- `figures/` - PNG figures for publication

---

## System Requirements

- Python 3.8+
- macOS, Linux, or Windows
- 4GB RAM recommended for large grids
- Jupyter Notebook or JupyterLab

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{elshahed2024ails,
  title={Adaptive Incremental Line Search: A Corridor-Based Optimization for Grid Pathfinding},
  author={Elshahed, Amr},
  journal={[Journal Name]},
  year={2024}
}
```

---

## Support

For questions or issues, please contact the author or open an issue in the repository.
