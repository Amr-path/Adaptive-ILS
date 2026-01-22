# AILS Experiments

Comprehensive experiment suite for evaluating the Adaptive Incremental Line Search (AILS) algorithm.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run quick test
python run_experiments.py --quick --all

# Run full experiments
python run_experiments.py --all
```

## Directory Structure

```
experiments/
├── core/                    # Core AILS implementation
│   ├── ails_algorithm.py    # Main AILS algorithm
│   ├── corridor_builder.py  # Adaptive corridor construction
│   ├── data_structures.py   # Grid, Point, PathResult
│   └── heuristics.py        # Heuristic functions
│
├── algorithms/              # Comparison algorithms
│   ├── astar.py             # A* algorithm
│   ├── dijkstra.py          # Dijkstra's algorithm
│   ├── bfs.py               # Breadth-first search
│   ├── jps.py               # Jump Point Search
│   └── bidirectional.py     # Bidirectional A*
│
├── maps/                    # Map generation and parsing
│   ├── generators.py        # Synthetic map generators
│   └── parser.py            # Moving AI Lab parser
│
├── benchmarks/              # Benchmark experiments
│   ├── performance.py       # Performance benchmarks
│   ├── scalability.py       # Scalability analysis
│   ├── sensitivity.py       # Parameter sensitivity
│   ├── patterns.py          # Obstacle pattern analysis
│   └── comparative.py       # Full comparative study
│
├── analysis/                # Statistical analysis
│   ├── statistics.py        # Descriptive statistics
│   ├── hypothesis.py        # Hypothesis testing
│   ├── effect_size.py       # Effect size calculations
│   └── latex_export.py      # LaTeX table generation
│
├── visualization/           # Plotting and visualization
│   └── plotting.py          # Publication-quality plots
│
├── run_experiments.py       # Main experiment runner
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Available Experiments

### 1. Performance Benchmark (`--performance`)
Comprehensive performance comparison of all algorithms across:
- Grid sizes: 50x50 to 500x500
- Obstacle densities: 10% to 40%
- Patterns: random, maze, clustered, open, mixed, room

### 2. Scalability Analysis (`--scalability`)
Analyzes algorithm scaling with grid size:
- Grid sizes up to 1000x1000
- Time complexity estimation
- Memory usage tracking
- Scaling exponent computation

### 3. Parameter Sensitivity (`--sensitivity`)
One-at-a-time sensitivity analysis for AILS parameters:
- r_min: minimum corridor radius
- r_max: maximum corridor radius
- alpha: density exponent
- beta: gradient sensitivity
- window_size: density window size
- strategy: corridor construction strategy

### 4. Pattern Analysis (`--patterns`)
Performance comparison across obstacle patterns:
- Random uniform
- Maze structures
- Clustered obstacles
- Open areas
- Mixed environments
- Room-like structures

### 5. Comparative Study (`--comparative`)
Full comparative study combining all analyses:
- Synthetic map experiments
- Benchmark map experiments (if available)
- Strategy comparison
- Statistical significance testing

## Usage Examples

```bash
# Quick test (reduced parameters)
python run_experiments.py --quick --all

# Full performance benchmark
python run_experiments.py --performance

# Scalability with specific seed
python run_experiments.py --scalability --seed 123

# Multiple experiments
python run_experiments.py --performance --scalability --patterns

# Full study for paper
python run_experiments.py --all
```

## Output Files

Results are saved in the `results/` directory:

```
results/
├── performance/           # Performance benchmark results
│   ├── *.csv              # Raw results
│   └── *.json             # Full results with config
├── scalability/           # Scalability analysis
├── sensitivity/           # Parameter sensitivity
├── patterns/              # Pattern analysis
├── comparative/           # Comparative study
├── figures/               # Generated plots (PDF/PNG)
├── latex/                 # LaTeX tables
└── analysis/              # Combined analysis files
```

## Adding New Experiments

1. Create a new benchmark class in `benchmarks/`
2. Implement `run()`, `save_results()`, and analysis methods
3. Add to `run_experiments.py`

Example:
```python
from benchmarks.performance import PerformanceBenchmark, BenchmarkConfig

config = BenchmarkConfig(
    grid_sizes=[(100, 100), (200, 200)],
    densities=[0.2],
    trials_per_config=50,
)

benchmark = PerformanceBenchmark(config)
benchmark.run(verbose=True)
benchmark.save_results()
summary = benchmark.get_summary_statistics()
```

## Statistical Analysis

```python
from analysis.statistics import StatisticalAnalyzer
from analysis.hypothesis import HypothesisTester

# Load results
analyzer = StatisticalAnalyzer()
analyzer.load_csv('results/performance/results.csv')

# Compare algorithms
comparison = analyzer.compare_algorithms()

# Hypothesis testing
tester = HypothesisTester()
result = tester.paired_ttest(ails_times, astar_times)
print(f"p-value: {result.p_value}, Effect size: {result.effect_size}")
```

## Visualization

```python
from visualization.plotting import ResultsPlotter

plotter = ResultsPlotter()

# Algorithm comparison bar chart
plotter.algorithm_comparison_bar(summary, filename='comparison')

# Scalability line plot
plotter.scalability_line(scaling_data, filename='scalability')

# Corridor visualization
plotter.corridor_visualization(grid, corridor, path, start, goal)
```

## For Paper Submission

1. Run full experiments:
   ```bash
   python run_experiments.py --all --seed 42
   ```

2. LaTeX tables are in `results/latex/`

3. Figures are in `results/figures/`

4. Raw data for reproducibility in `results/*/`

## Citation

If you use this code, please cite:
```bibtex
@article{ails2024,
  title={Adaptive Incremental Line Search: A Dynamic Corridor-Based
         Optimization Framework for Grid-Based Pathfinding},
  author={Elshahed, Amr and Ali, Majid Khan and Mohamed, Ahmad Sufril Azlan
          and Abdullah, Farah Aini and Aun, Lee Jian},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2024}
}
```
