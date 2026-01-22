"""
LaTeX Export Module
====================

Generate publication-ready LaTeX tables from experimental results.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd


class LaTeXExporter:
    """
    Export experimental results to LaTeX tables.

    Generates publication-quality tables formatted for
    IEEE, ACM, or custom journal styles.
    """

    def __init__(self, output_dir: str = "results/latex"):
        """
        Initialize exporter.

        Args:
            output_dir: Directory for LaTeX output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def algorithm_comparison_table(self, data: Dict[str, Dict],
                                    metrics: List[str] = None,
                                    caption: str = "Algorithm Performance Comparison",
                                    label: str = "tab:algorithm_comparison") -> str:
        """
        Generate comparison table for algorithms.

        Args:
            data: Dictionary mapping algorithm names to metric dictionaries
            metrics: List of metrics to include
            caption: Table caption
            label: Table label for references

        Returns:
            LaTeX table string
        """
        if metrics is None:
            metrics = ['time_mean', 'time_std', 'nodes_mean', 'nodes_std']

        algorithms = list(data.keys())
        n_metrics = len(metrics)

        # Build table
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\begin{tabular}{l" + "r" * n_metrics + "}",
            "\\toprule",
        ]

        # Header
        header_names = {
            'time_mean': 'Time (ms)',
            'time_std': 'Time SD',
            'nodes_mean': 'Nodes',
            'nodes_std': 'Nodes SD',
            'success_rate': 'Success',
            'corridor_efficiency_mean': 'Corr. Eff.',
        }
        header = "Algorithm & " + " & ".join(
            header_names.get(m, m) for m in metrics
        ) + " \\\\"
        lines.append(header)
        lines.append("\\midrule")

        # Data rows
        for algo in algorithms:
            algo_data = data[algo]
            values = []
            for metric in metrics:
                val = algo_data.get(metric, 0)
                if 'pct' in metric or 'rate' in metric or 'efficiency' in metric:
                    values.append(f"{val:.1f}\\%")
                elif isinstance(val, float):
                    if val > 1000:
                        values.append(f"{val:.0f}")
                    elif val > 10:
                        values.append(f"{val:.1f}")
                    else:
                        values.append(f"{val:.3f}")
                else:
                    values.append(str(val))

            # Bold best values
            row = f"{algo} & " + " & ".join(values) + " \\\\"
            lines.append(row)

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

        return "\n".join(lines)

    def speedup_table(self, speedups: Dict[str, Dict],
                      caption: str = "AILS Speedup Over Baseline Algorithms",
                      label: str = "tab:speedup") -> str:
        """
        Generate speedup comparison table.

        Args:
            speedups: Dictionary with speedup data
            caption: Table caption
            label: Table label

        Returns:
            LaTeX table string
        """
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\begin{tabular}{lrrr}",
            "\\toprule",
            "Algorithm & Speedup & Time Red. (\\%) & Node Red. (\\%) \\\\",
            "\\midrule",
        ]

        for algo, data in speedups.items():
            speedup = data.get('time_speedup', 0)
            time_red = data.get('time_reduction_pct', 0)
            node_red = data.get('node_reduction_pct', 0)

            row = f"{algo} & {speedup:.2f}$\\times$ & {time_red:.1f}\\% & {node_red:.1f}\\% \\\\"
            lines.append(row)

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

        return "\n".join(lines)

    def scalability_table(self, data: Dict[str, Dict],
                          grid_sizes: List[str],
                          caption: str = "Scalability Analysis",
                          label: str = "tab:scalability") -> str:
        """
        Generate scalability comparison table.

        Args:
            data: Scalability data by algorithm and grid size
            grid_sizes: List of grid sizes to include
            caption: Table caption
            label: Table label

        Returns:
            LaTeX table string
        """
        algorithms = list(data.keys())

        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\begin{tabular}{l" + "r" * len(grid_sizes) + "}",
            "\\toprule",
            "Algorithm & " + " & ".join(grid_sizes) + " \\\\",
            "\\midrule",
        ]

        for algo in algorithms:
            values = []
            for size in grid_sizes:
                val = data.get(algo, {}).get(size, {}).get('time_mean', 0)
                if val > 1000:
                    values.append(f"{val:.0f}")
                else:
                    values.append(f"{val:.2f}")
            row = f"{algo} & " + " & ".join(values) + " \\\\"
            lines.append(row)

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

        return "\n".join(lines)

    def statistical_tests_table(self, tests: Dict[str, Any],
                                caption: str = "Statistical Significance Tests",
                                label: str = "tab:statistics") -> str:
        """
        Generate table of statistical test results.

        Args:
            tests: Statistical test results
            caption: Table caption
            label: Table label

        Returns:
            LaTeX table string
        """
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\begin{tabular}{llrrl}",
            "\\toprule",
            "Comparison & Test & Statistic & $p$-value & Effect Size \\\\",
            "\\midrule",
        ]

        for comparison, test_data in tests.items():
            test_name = test_data.get('test', 'Unknown')
            statistic = test_data.get('statistic', 0)
            p_value = test_data.get('p_value', 1)
            effect = test_data.get('effect_size', 0)
            effect_interp = test_data.get('effect_interpretation', '')

            # Format p-value
            if p_value < 0.001:
                p_str = "$< 0.001$"
            else:
                p_str = f"${p_value:.4f}$"

            # Add significance marker
            sig = "$^{***}$" if p_value < 0.001 else "$^{**}$" if p_value < 0.01 else "$^{*}$" if p_value < 0.05 else ""

            row = f"{comparison} & {test_name} & {statistic:.3f} & {p_str}{sig} & {effect:.3f} ({effect_interp}) \\\\"
            lines.append(row)

        lines.extend([
            "\\bottomrule",
            "\\multicolumn{5}{l}{\\footnotesize $^{*}p < 0.05$, $^{**}p < 0.01$, $^{***}p < 0.001$} \\\\",
            "\\end{tabular}",
            "\\end{table}",
        ])

        return "\n".join(lines)

    def pattern_comparison_table(self, data: Dict[str, Dict],
                                  algorithms: List[str],
                                  caption: str = "Performance by Obstacle Pattern",
                                  label: str = "tab:patterns") -> str:
        """
        Generate table comparing performance across patterns.

        Args:
            data: Pattern analysis data
            algorithms: Algorithms to include
            caption: Table caption
            label: Table label

        Returns:
            LaTeX table string
        """
        patterns = list(data.keys())

        lines = [
            "\\begin{table*}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\begin{tabular}{l" + "rr" * len(algorithms) + "}",
            "\\toprule",
        ]

        # Header
        header1 = "Pattern"
        for algo in algorithms:
            header1 += f" & \\multicolumn{{2}}{{c}}{{{algo}}}"
        header1 += " \\\\"
        lines.append(header1)

        header2 = " & " + " & ".join(["Time & Nodes"] * len(algorithms)) + " \\\\"
        lines.append(header2)
        lines.append("\\midrule")

        for pattern in patterns:
            row_parts = [pattern.capitalize()]
            for algo in algorithms:
                algo_data = data[pattern].get(algo, {})
                time_val = algo_data.get('time_mean', 0)
                nodes_val = algo_data.get('nodes_mean', 0)
                row_parts.append(f"{time_val:.2f}")
                row_parts.append(f"{nodes_val:.0f}")
            row = " & ".join(row_parts) + " \\\\"
            lines.append(row)

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
        ])

        return "\n".join(lines)

    def sensitivity_table(self, data: Dict[str, Dict],
                          caption: str = "Parameter Sensitivity Analysis",
                          label: str = "tab:sensitivity") -> str:
        """
        Generate parameter sensitivity table.

        Args:
            data: Sensitivity analysis data
            caption: Table caption
            label: Table label

        Returns:
            LaTeX table string
        """
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\begin{tabular}{llrl}",
            "\\toprule",
            "Parameter & Best Value & Effect (\\%) & Sensitivity \\\\",
            "\\midrule",
        ]

        for param, param_data in data.items():
            best = param_data.get('best_value', 'N/A')
            effect = param_data.get('effect_range_pct', 0)
            sensitivity = param_data.get('sensitivity', 'unknown')

            row = f"{param} & {best} & {effect:.1f}\\% & {sensitivity} \\\\"
            lines.append(row)

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

        return "\n".join(lines)

    def save_table(self, table_content: str, filename: str) -> str:
        """
        Save LaTeX table to file.

        Args:
            table_content: LaTeX table string
            filename: Output filename (without extension)

        Returns:
            Path to saved file
        """
        filepath = self.output_dir / f"{filename}.tex"
        with open(filepath, 'w') as f:
            f.write(table_content)
        return str(filepath)

    def generate_all_tables(self, analysis_results: Dict[str, Any],
                            prefix: str = "ails") -> List[str]:
        """
        Generate all standard tables from analysis results.

        Args:
            analysis_results: Complete analysis results dictionary
            prefix: Filename prefix

        Returns:
            List of saved file paths
        """
        saved_files = []

        # Algorithm comparison
        if 'by_algorithm' in analysis_results:
            table = self.algorithm_comparison_table(
                analysis_results['by_algorithm'],
                metrics=['time_mean', 'time_std', 'nodes_mean', 'success_rate'],
                caption="Algorithm Performance Comparison",
                label="tab:algorithm_comparison"
            )
            saved_files.append(self.save_table(table, f"{prefix}_algorithm_comparison"))

        # Speedup table
        if 'speedup_analysis' in analysis_results:
            table = self.speedup_table(
                analysis_results['speedup_analysis'],
                caption="AILS Performance Improvement Over Baseline Algorithms",
                label="tab:speedup"
            )
            saved_files.append(self.save_table(table, f"{prefix}_speedup"))

        # Pattern comparison
        if 'by_pattern' in analysis_results:
            algorithms = list(analysis_results.get('by_algorithm', {}).keys())[:4]
            table = self.pattern_comparison_table(
                analysis_results['by_pattern'],
                algorithms=algorithms,
                caption="Algorithm Performance Across Obstacle Patterns",
                label="tab:patterns"
            )
            saved_files.append(self.save_table(table, f"{prefix}_patterns"))

        # Statistical tests
        if 'statistical_tests' in analysis_results:
            table = self.statistical_tests_table(
                analysis_results['statistical_tests'],
                caption="Statistical Significance of AILS Improvements",
                label="tab:statistics"
            )
            saved_files.append(self.save_table(table, f"{prefix}_statistics"))

        return saved_files


def export_to_latex(analysis_results: Dict[str, Any],
                    output_dir: str = "results/latex") -> List[str]:
    """
    Convenience function to export all tables.

    Args:
        analysis_results: Analysis results dictionary
        output_dir: Output directory

    Returns:
        List of saved file paths
    """
    exporter = LaTeXExporter(output_dir)
    return exporter.generate_all_tables(analysis_results)
