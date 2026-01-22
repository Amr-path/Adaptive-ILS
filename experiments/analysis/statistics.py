"""
Statistical Analysis
=====================

Comprehensive statistical analysis for experimental results.
Includes descriptive statistics, confidence intervals, and
distribution analysis.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import pandas as pd


@dataclass
class DescriptiveStats:
    """Descriptive statistics for a dataset."""
    n: int
    mean: float
    std: float
    sem: float  # Standard error of mean
    median: float
    q1: float
    q3: float
    iqr: float
    min_val: float
    max_val: float
    skewness: float
    kurtosis: float
    cv: float  # Coefficient of variation

    def to_dict(self) -> Dict[str, float]:
        return {
            'n': self.n,
            'mean': self.mean,
            'std': self.std,
            'sem': self.sem,
            'median': self.median,
            'q1': self.q1,
            'q3': self.q3,
            'iqr': self.iqr,
            'min': self.min_val,
            'max': self.max_val,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'cv': self.cv,
        }


class StatisticalAnalyzer:
    """
    Comprehensive statistical analyzer for experimental results.
    """

    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Initialize analyzer.

        Args:
            data: Optional DataFrame with experimental results
        """
        self.data = data

    def load_csv(self, filepath: str) -> pd.DataFrame:
        """Load results from CSV file."""
        self.data = pd.read_csv(filepath)
        return self.data

    def descriptive_stats(self, values: List[float]) -> DescriptiveStats:
        """
        Compute comprehensive descriptive statistics.

        Args:
            values: List of numeric values

        Returns:
            DescriptiveStats object
        """
        values = np.array(values)
        n = len(values)

        if n == 0:
            return DescriptiveStats(
                n=0, mean=0, std=0, sem=0, median=0,
                q1=0, q3=0, iqr=0, min_val=0, max_val=0,
                skewness=0, kurtosis=0, cv=0
            )

        mean = np.mean(values)
        std = np.std(values, ddof=1)
        sem = std / np.sqrt(n)
        median = np.median(values)
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)

        return DescriptiveStats(
            n=n,
            mean=mean,
            std=std,
            sem=sem,
            median=median,
            q1=q1,
            q3=q3,
            iqr=q3 - q1,
            min_val=np.min(values),
            max_val=np.max(values),
            skewness=float(stats.skew(values)) if n > 2 else 0,
            kurtosis=float(stats.kurtosis(values)) if n > 3 else 0,
            cv=(std / mean * 100) if mean != 0 else 0,
        )

    def confidence_interval(self, values: List[float],
                           confidence: float = 0.95) -> Tuple[float, float]:
        """
        Compute confidence interval for the mean.

        Args:
            values: List of values
            confidence: Confidence level (default 0.95 for 95% CI)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        values = np.array(values)
        n = len(values)
        if n < 2:
            return (np.mean(values), np.mean(values))

        mean = np.mean(values)
        sem = stats.sem(values)
        h = sem * stats.t.ppf((1 + confidence) / 2, n - 1)

        return (mean - h, mean + h)

    def bootstrap_ci(self, values: List[float],
                     n_bootstrap: int = 10000,
                     confidence: float = 0.95) -> Tuple[float, float]:
        """
        Compute bootstrap confidence interval.

        More robust for non-normal distributions.

        Args:
            values: List of values
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        values = np.array(values)
        n = len(values)

        # Bootstrap resampling
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

        return (lower, upper)

    def normality_test(self, values: List[float]) -> Dict[str, Any]:
        """
        Test for normality using multiple tests.

        Args:
            values: List of values

        Returns:
            Dictionary with test results
        """
        values = np.array(values)
        n = len(values)

        results = {}

        # Shapiro-Wilk test (best for n < 50)
        if 3 <= n <= 5000:
            stat, p = stats.shapiro(values)
            results['shapiro_wilk'] = {
                'statistic': stat,
                'p_value': p,
                'normal': p > 0.05,
            }

        # D'Agostino-Pearson test (requires n >= 20)
        if n >= 20:
            stat, p = stats.normaltest(values)
            results['dagostino_pearson'] = {
                'statistic': stat,
                'p_value': p,
                'normal': p > 0.05,
            }

        # Anderson-Darling test
        result = stats.anderson(values, dist='norm')
        # Use 5% significance level (index 2)
        results['anderson_darling'] = {
            'statistic': result.statistic,
            'critical_value_5pct': result.critical_values[2],
            'normal': result.statistic < result.critical_values[2],
        }

        return results

    def analyze_by_group(self, group_column: str,
                         value_column: str) -> Dict[str, DescriptiveStats]:
        """
        Compute descriptive statistics grouped by a column.

        Args:
            group_column: Column to group by
            value_column: Column with values to analyze

        Returns:
            Dictionary mapping group name to statistics
        """
        if self.data is None:
            raise ValueError("No data loaded. Use load_csv() first.")

        results = {}
        for group_name, group_data in self.data.groupby(group_column):
            values = group_data[value_column].dropna().tolist()
            results[str(group_name)] = self.descriptive_stats(values)

        return results

    def compare_algorithms(self, algorithm_column: str = 'algorithm',
                          time_column: str = 'execution_time_ms',
                          nodes_column: str = 'visited_nodes') -> Dict[str, Any]:
        """
        Compare algorithms with comprehensive statistics.

        Args:
            algorithm_column: Column with algorithm names
            time_column: Column with execution times
            nodes_column: Column with visited nodes

        Returns:
            Comparison results
        """
        if self.data is None:
            raise ValueError("No data loaded")

        results = {}

        for algo in self.data[algorithm_column].unique():
            algo_data = self.data[self.data[algorithm_column] == algo]

            times = algo_data[time_column].dropna().tolist()
            nodes = algo_data[nodes_column].dropna().tolist()

            time_stats = self.descriptive_stats(times)
            node_stats = self.descriptive_stats(nodes)
            time_ci = self.confidence_interval(times)

            results[algo] = {
                'time': time_stats.to_dict(),
                'nodes': node_stats.to_dict(),
                'time_95ci': {'lower': time_ci[0], 'upper': time_ci[1]},
            }

        return results

    def compute_improvement(self, baseline: str, target: str,
                           metric_column: str) -> Dict[str, float]:
        """
        Compute improvement of target over baseline.

        Args:
            baseline: Baseline algorithm name
            target: Target algorithm name
            metric_column: Column to compare

        Returns:
            Improvement metrics
        """
        if self.data is None:
            raise ValueError("No data loaded")

        baseline_data = self.data[self.data['algorithm'] == baseline][metric_column]
        target_data = self.data[self.data['algorithm'] == target][metric_column]

        baseline_mean = baseline_data.mean()
        target_mean = target_data.mean()

        improvement = baseline_mean - target_mean
        improvement_pct = (improvement / baseline_mean * 100) if baseline_mean != 0 else 0
        speedup = baseline_mean / target_mean if target_mean != 0 else float('inf')

        return {
            'baseline_mean': baseline_mean,
            'target_mean': target_mean,
            'absolute_improvement': improvement,
            'improvement_pct': improvement_pct,
            'speedup': speedup,
        }

    def outlier_analysis(self, values: List[float],
                        method: str = 'iqr') -> Dict[str, Any]:
        """
        Detect and analyze outliers.

        Args:
            values: List of values
            method: Detection method ('iqr' or 'zscore')

        Returns:
            Outlier analysis results
        """
        values = np.array(values)

        if method == 'iqr':
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
        else:  # zscore
            mean = np.mean(values)
            std = np.std(values)
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std

        outliers = values[(values < lower_bound) | (values > upper_bound)]

        return {
            'method': method,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'n_outliers': len(outliers),
            'outlier_pct': len(outliers) / len(values) * 100,
            'outlier_values': outliers.tolist(),
        }


def analyze_results(csv_path: str) -> Dict[str, Any]:
    """
    Convenience function to analyze experimental results.

    Args:
        csv_path: Path to results CSV

    Returns:
        Comprehensive analysis
    """
    analyzer = StatisticalAnalyzer()
    analyzer.load_csv(csv_path)

    return {
        'algorithm_comparison': analyzer.compare_algorithms(),
        'ails_improvement': analyzer.compute_improvement('A*', 'AILS', 'execution_time_ms'),
    }
