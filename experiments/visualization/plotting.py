"""
Visualization and Plotting
===========================

Publication-quality plots for experimental results.
Optimized for Q1 journal standards.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns


@dataclass
class PlotConfig:
    """Configuration for publication-quality plots."""
    # Figure dimensions (in inches)
    fig_width: float = 7.0
    fig_height: float = 5.0
    dpi: int = 300

    # Font sizes
    title_size: int = 14
    label_size: int = 12
    tick_size: int = 10
    legend_size: int = 10

    # Colors (colorblind-friendly palette)
    colors: List[str] = field(default_factory=lambda: [
        '#0077BB',  # Blue
        '#EE7733',  # Orange
        '#009988',  # Teal
        '#CC3311',  # Red
        '#33BBEE',  # Cyan
        '#EE3377',  # Magenta
        '#BBBBBB',  # Gray
    ])

    # Style
    style: str = 'seaborn-v0_8-whitegrid'
    font_family: str = 'serif'

    # Output
    output_dir: str = 'results/figures'
    save_format: str = 'pdf'  # pdf, png, svg

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


class ResultsPlotter:
    """
    Generate publication-quality plots for experimental results.
    """

    def __init__(self, config: Optional[PlotConfig] = None):
        """
        Initialize plotter.

        Args:
            config: Plot configuration
        """
        self.config = config or PlotConfig()
        self._setup_style()

    def _setup_style(self):
        """Configure matplotlib style."""
        try:
            plt.style.use(self.config.style)
        except:
            plt.style.use('seaborn-v0_8-whitegrid')

        plt.rcParams.update({
            'font.family': self.config.font_family,
            'font.size': self.config.tick_size,
            'axes.titlesize': self.config.title_size,
            'axes.labelsize': self.config.label_size,
            'xtick.labelsize': self.config.tick_size,
            'ytick.labelsize': self.config.tick_size,
            'legend.fontsize': self.config.legend_size,
            'figure.dpi': self.config.dpi,
        })

    def algorithm_comparison_bar(self, data: Dict[str, Dict],
                                  metric: str = 'time_mean',
                                  ylabel: str = 'Execution Time (ms)',
                                  title: str = 'Algorithm Performance Comparison',
                                  filename: str = 'algorithm_comparison') -> str:
        """
        Create bar chart comparing algorithms.

        Args:
            data: Dictionary mapping algorithm names to metrics
            metric: Metric to plot
            ylabel: Y-axis label
            title: Plot title
            filename: Output filename (without extension)

        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(self.config.fig_width, self.config.fig_height))

        algorithms = list(data.keys())
        values = [data[algo].get(metric, 0) for algo in algorithms]
        errors = [data[algo].get(f'{metric.replace("_mean", "_std")}', 0) for algo in algorithms]

        x = np.arange(len(algorithms))
        bars = ax.bar(x, values, yerr=errors, capsize=4,
                     color=self.config.colors[:len(algorithms)],
                     edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Algorithm')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=45, ha='right')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        filepath = Path(self.config.output_dir) / f"{filename}.{self.config.save_format}"
        plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()

        return str(filepath)

    def scalability_line(self, data: Dict[str, List],
                         xlabel: str = 'Grid Size',
                         ylabel: str = 'Execution Time (ms)',
                         title: str = 'Scalability Analysis',
                         filename: str = 'scalability',
                         log_scale: bool = True) -> str:
        """
        Create line plot for scalability analysis.

        Args:
            data: Dictionary mapping algorithm names to (sizes, values) tuples
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            filename: Output filename
            log_scale: Use logarithmic scale

        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(self.config.fig_width, self.config.fig_height))

        for i, (algo, (sizes, values)) in enumerate(data.items()):
            color = self.config.colors[i % len(self.config.colors)]
            ax.plot(sizes, values, 'o-', color=color, label=algo,
                   linewidth=2, markersize=6)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        if log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')

        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        filepath = Path(self.config.output_dir) / f"{filename}.{self.config.save_format}"
        plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()

        return str(filepath)

    def speedup_bar(self, speedups: Dict[str, float],
                    baseline: str = 'A*',
                    title: str = 'AILS Speedup Over Baseline',
                    filename: str = 'speedup') -> str:
        """
        Create bar chart of speedup factors.

        Args:
            speedups: Dictionary mapping algorithm names to speedup values
            baseline: Baseline algorithm name
            title: Plot title
            filename: Output filename

        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(self.config.fig_width, self.config.fig_height))

        algorithms = list(speedups.keys())
        values = list(speedups.values())

        x = np.arange(len(algorithms))
        colors = [self.config.colors[0] if v > 1 else self.config.colors[3] for v in values]

        bars = ax.bar(x, values, color=colors, edgecolor='black', linewidth=0.5)

        # Add reference line at 1
        ax.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.7)

        ax.set_xlabel('Baseline Algorithm')
        ax.set_ylabel('Speedup Factor')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=45, ha='right')

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}x',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        filepath = Path(self.config.output_dir) / f"{filename}.{self.config.save_format}"
        plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()

        return str(filepath)

    def heatmap(self, data: np.ndarray,
                row_labels: List[str],
                col_labels: List[str],
                title: str = 'Parameter Sensitivity',
                xlabel: str = 'Parameter 1',
                ylabel: str = 'Parameter 2',
                cmap: str = 'viridis',
                filename: str = 'heatmap') -> str:
        """
        Create heatmap for parameter sensitivity or interaction effects.

        Args:
            data: 2D numpy array of values
            row_labels: Row labels
            col_labels: Column labels
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            cmap: Colormap name
            filename: Output filename

        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(self.config.fig_width, self.config.fig_height))

        im = ax.imshow(data, cmap=cmap, aspect='auto')
        cbar = ax.figure.colorbar(im, ax=ax)

        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # Add text annotations
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                text = ax.text(j, i, f'{data[i, j]:.2f}',
                              ha="center", va="center", color="white",
                              fontsize=8)

        plt.tight_layout()

        filepath = Path(self.config.output_dir) / f"{filename}.{self.config.save_format}"
        plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()

        return str(filepath)

    def box_plot(self, data: Dict[str, List[float]],
                 ylabel: str = 'Execution Time (ms)',
                 title: str = 'Algorithm Performance Distribution',
                 filename: str = 'boxplot') -> str:
        """
        Create box plot comparing algorithm distributions.

        Args:
            data: Dictionary mapping algorithm names to value lists
            ylabel: Y-axis label
            title: Plot title
            filename: Output filename

        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(self.config.fig_width, self.config.fig_height))

        algorithms = list(data.keys())
        values = [data[algo] for algo in algorithms]

        bp = ax.boxplot(values, labels=algorithms, patch_artist=True)

        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(self.config.colors[i % len(self.config.colors)])

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticklabels(algorithms, rotation=45, ha='right')

        plt.tight_layout()

        filepath = Path(self.config.output_dir) / f"{filename}.{self.config.save_format}"
        plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()

        return str(filepath)

    def pattern_comparison(self, data: Dict[str, Dict[str, float]],
                           patterns: List[str],
                           algorithms: List[str],
                           ylabel: str = 'Execution Time (ms)',
                           title: str = 'Performance by Obstacle Pattern',
                           filename: str = 'pattern_comparison') -> str:
        """
        Create grouped bar chart for pattern comparison.

        Args:
            data: Nested dict: {pattern: {algorithm: value}}
            patterns: List of patterns
            algorithms: List of algorithms
            ylabel: Y-axis label
            title: Plot title
            filename: Output filename

        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(self.config.fig_width * 1.2, self.config.fig_height))

        x = np.arange(len(patterns))
        width = 0.8 / len(algorithms)

        for i, algo in enumerate(algorithms):
            values = [data.get(pattern, {}).get(algo, {}).get('time_mean', 0)
                     for pattern in patterns]
            offset = (i - len(algorithms) / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=algo,
                         color=self.config.colors[i % len(self.config.colors)])

        ax.set_xlabel('Obstacle Pattern')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([p.capitalize() for p in patterns])
        ax.legend(loc='upper right')

        plt.tight_layout()

        filepath = Path(self.config.output_dir) / f"{filename}.{self.config.save_format}"
        plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()

        return str(filepath)

    def corridor_visualization(self, grid_array: np.ndarray,
                               corridor_points: set,
                               path: List[Tuple[int, int]],
                               start: Tuple[int, int],
                               goal: Tuple[int, int],
                               title: str = 'AILS Corridor Visualization',
                               filename: str = 'corridor') -> str:
        """
        Visualize AILS corridor on a grid.

        Args:
            grid_array: 2D boolean array of obstacles
            corridor_points: Set of (x, y) corridor points
            path: List of path points
            start: Start point
            goal: Goal point
            title: Plot title
            filename: Output filename

        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(self.config.fig_width, self.config.fig_width))

        height, width = grid_array.shape

        # Create display array
        display = np.ones((height, width, 3))

        # Obstacles (dark gray)
        display[grid_array] = [0.3, 0.3, 0.3]

        # Corridor (light blue)
        for x, y in corridor_points:
            if 0 <= y < height and 0 <= x < width and not grid_array[y, x]:
                display[y, x] = [0.7, 0.85, 1.0]

        # Path (green)
        for x, y in path:
            if 0 <= y < height and 0 <= x < width:
                display[y, x] = [0.2, 0.8, 0.2]

        ax.imshow(display)

        # Mark start and goal
        ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
        ax.plot(goal[0], goal[1], 'r*', markersize=12, label='Goal')

        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.axis('off')

        plt.tight_layout()

        filepath = Path(self.config.output_dir) / f"{filename}.{self.config.save_format}"
        plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()

        return str(filepath)


# Convenience functions
def plot_algorithm_comparison(data: Dict, **kwargs) -> str:
    """Quick algorithm comparison plot."""
    return ResultsPlotter().algorithm_comparison_bar(data, **kwargs)


def plot_scalability(data: Dict, **kwargs) -> str:
    """Quick scalability plot."""
    return ResultsPlotter().scalability_line(data, **kwargs)


def plot_speedup_bars(speedups: Dict, **kwargs) -> str:
    """Quick speedup bar plot."""
    return ResultsPlotter().speedup_bar(speedups, **kwargs)


def plot_corridor_visualization(grid, corridor, path, start, goal, **kwargs) -> str:
    """Quick corridor visualization."""
    return ResultsPlotter().corridor_visualization(grid, corridor, path, start, goal, **kwargs)


def plot_sensitivity_heatmap(data: np.ndarray, rows: List, cols: List, **kwargs) -> str:
    """Quick sensitivity heatmap."""
    return ResultsPlotter().heatmap(data, rows, cols, **kwargs)
