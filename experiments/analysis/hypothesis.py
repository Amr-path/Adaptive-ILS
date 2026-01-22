"""
Hypothesis Testing
===================

Statistical hypothesis tests for comparing algorithm performance.
Includes parametric and non-parametric tests.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import warnings


@dataclass
class TestResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float = 0.05
    effect_size: Optional[float] = None
    effect_interpretation: str = ""
    additional_info: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'test': self.test_name,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'alpha': self.alpha,
            'significant': self.significant,
        }
        if self.effect_size is not None:
            result['effect_size'] = self.effect_size
            result['effect_interpretation'] = self.effect_interpretation
        if self.additional_info:
            result.update(self.additional_info)
        return result


class HypothesisTester:
    """
    Comprehensive hypothesis testing for algorithm comparison.

    Supports:
    - Paired and independent t-tests
    - Mann-Whitney U test
    - Wilcoxon signed-rank test
    - ANOVA with post-hoc tests
    - Multiple comparison corrections
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize tester.

        Args:
            alpha: Significance level (default 0.05)
        """
        self.alpha = alpha

    def paired_ttest(self, group1: List[float], group2: List[float],
                     alternative: str = 'two-sided') -> TestResult:
        """
        Paired samples t-test.

        Use when the same subjects are measured under two conditions.
        Requires normally distributed differences.

        Args:
            group1: First group measurements
            group2: Second group measurements
            alternative: 'two-sided', 'less', or 'greater'

        Returns:
            TestResult
        """
        group1, group2 = np.array(group1), np.array(group2)

        if len(group1) != len(group2):
            raise ValueError("Paired t-test requires equal sample sizes")

        t_stat, p_value = stats.ttest_rel(group1, group2, alternative=alternative)

        # Effect size: Cohen's d for paired samples
        diff = group1 - group2
        d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0

        return TestResult(
            test_name="Paired t-test",
            statistic=t_stat,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=d,
            effect_interpretation=self._interpret_cohens_d(d),
            additional_info={
                'mean_difference': np.mean(diff),
                'std_difference': np.std(diff, ddof=1),
                'n': len(group1),
            }
        )

    def independent_ttest(self, group1: List[float], group2: List[float],
                          equal_var: bool = False,
                          alternative: str = 'two-sided') -> TestResult:
        """
        Independent samples t-test.

        Use when comparing two independent groups.

        Args:
            group1: First group measurements
            group2: Second group measurements
            equal_var: Assume equal variances (default False for Welch's test)
            alternative: 'two-sided', 'less', or 'greater'

        Returns:
            TestResult
        """
        group1, group2 = np.array(group1), np.array(group2)

        t_stat, p_value = stats.ttest_ind(
            group1, group2, equal_var=equal_var, alternative=alternative
        )

        # Cohen's d for independent samples
        pooled_std = np.sqrt(
            ((len(group1) - 1) * np.var(group1, ddof=1) +
             (len(group2) - 1) * np.var(group2, ddof=1)) /
            (len(group1) + len(group2) - 2)
        )
        d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

        return TestResult(
            test_name="Welch's t-test" if not equal_var else "Student's t-test",
            statistic=t_stat,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=d,
            effect_interpretation=self._interpret_cohens_d(d),
            additional_info={
                'n1': len(group1),
                'n2': len(group2),
                'mean1': np.mean(group1),
                'mean2': np.mean(group2),
            }
        )

    def mann_whitney(self, group1: List[float], group2: List[float],
                     alternative: str = 'two-sided') -> TestResult:
        """
        Mann-Whitney U test (non-parametric).

        Use when normality assumption is violated.

        Args:
            group1: First group
            group2: Second group
            alternative: 'two-sided', 'less', or 'greater'

        Returns:
            TestResult
        """
        group1, group2 = np.array(group1), np.array(group2)

        u_stat, p_value = stats.mannwhitneyu(
            group1, group2, alternative=alternative
        )

        # Effect size: rank-biserial correlation
        n1, n2 = len(group1), len(group2)
        r = 1 - (2 * u_stat) / (n1 * n2)

        return TestResult(
            test_name="Mann-Whitney U test",
            statistic=u_stat,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=r,
            effect_interpretation=self._interpret_r(r),
            additional_info={
                'n1': n1,
                'n2': n2,
            }
        )

    def wilcoxon(self, group1: List[float], group2: List[float],
                 alternative: str = 'two-sided') -> TestResult:
        """
        Wilcoxon signed-rank test (non-parametric paired test).

        Use for paired samples when normality is violated.

        Args:
            group1: First group
            group2: Second group
            alternative: 'two-sided', 'less', or 'greater'

        Returns:
            TestResult
        """
        group1, group2 = np.array(group1), np.array(group2)

        if len(group1) != len(group2):
            raise ValueError("Wilcoxon test requires equal sample sizes")

        # Handle zero differences
        diff = group1 - group2
        non_zero = diff != 0

        if np.sum(non_zero) < 10:
            warnings.warn("Very few non-zero differences")

        try:
            w_stat, p_value = stats.wilcoxon(
                group1, group2, alternative=alternative
            )
        except ValueError:
            # All differences are zero
            return TestResult(
                test_name="Wilcoxon signed-rank test",
                statistic=0,
                p_value=1.0,
                significant=False,
                alpha=self.alpha,
            )

        # Effect size: r = Z / sqrt(N)
        n = len(group1)
        z = stats.norm.ppf(1 - p_value / 2)  # Approximate Z
        r = z / np.sqrt(n)

        return TestResult(
            test_name="Wilcoxon signed-rank test",
            statistic=w_stat,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=r,
            effect_interpretation=self._interpret_r(r),
            additional_info={
                'n': n,
                'n_non_zero': np.sum(non_zero),
            }
        )

    def one_way_anova(self, *groups: List[float]) -> TestResult:
        """
        One-way ANOVA.

        Compare means across multiple groups.

        Args:
            *groups: Variable number of groups

        Returns:
            TestResult
        """
        groups = [np.array(g) for g in groups]

        f_stat, p_value = stats.f_oneway(*groups)

        # Effect size: eta-squared
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_total = np.sum((all_data - grand_mean)**2)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0

        return TestResult(
            test_name="One-way ANOVA",
            statistic=f_stat,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=eta_sq,
            effect_interpretation=self._interpret_eta_squared(eta_sq),
            additional_info={
                'n_groups': len(groups),
                'group_sizes': [len(g) for g in groups],
                'group_means': [np.mean(g) for g in groups],
            }
        )

    def kruskal_wallis(self, *groups: List[float]) -> TestResult:
        """
        Kruskal-Wallis H test (non-parametric ANOVA).

        Non-parametric alternative to one-way ANOVA.

        Args:
            *groups: Variable number of groups

        Returns:
            TestResult
        """
        groups = [np.array(g) for g in groups]

        h_stat, p_value = stats.kruskal(*groups)

        # Effect size: epsilon-squared
        n = sum(len(g) for g in groups)
        k = len(groups)
        epsilon_sq = (h_stat - k + 1) / (n - k)

        return TestResult(
            test_name="Kruskal-Wallis H test",
            statistic=h_stat,
            p_value=p_value,
            significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=epsilon_sq,
            effect_interpretation=self._interpret_epsilon_squared(epsilon_sq),
            additional_info={
                'n_groups': len(groups),
            }
        )

    def tukey_hsd(self, data: Dict[str, List[float]]) -> Dict[str, TestResult]:
        """
        Tukey's HSD post-hoc test.

        Pairwise comparisons after significant ANOVA.

        Args:
            data: Dictionary mapping group names to values

        Returns:
            Dictionary of pairwise comparison results
        """
        from scipy.stats import tukey_hsd as scipy_tukey

        groups = list(data.keys())
        values = [np.array(data[g]) for g in groups]

        result = scipy_tukey(*values)

        comparisons = {}
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                pair = f"{groups[i]}_vs_{groups[j]}"
                p_val = result.pvalue[i, j]
                ci = result.confidence_interval()

                comparisons[pair] = TestResult(
                    test_name="Tukey HSD",
                    statistic=result.statistic[i, j],
                    p_value=p_val,
                    significant=p_val < self.alpha,
                    alpha=self.alpha,
                    additional_info={
                        'ci_low': ci.low[i, j],
                        'ci_high': ci.high[i, j],
                    }
                )

        return comparisons

    def bonferroni_correction(self, p_values: List[float]) -> Tuple[List[float], float]:
        """
        Bonferroni correction for multiple comparisons.

        Args:
            p_values: List of p-values

        Returns:
            Tuple of (adjusted p-values, adjusted alpha)
        """
        n = len(p_values)
        adjusted_alpha = self.alpha / n
        adjusted_pvalues = [min(p * n, 1.0) for p in p_values]

        return adjusted_pvalues, adjusted_alpha

    def holm_bonferroni(self, p_values: List[float]) -> List[Tuple[float, bool]]:
        """
        Holm-Bonferroni correction (step-down method).

        Less conservative than Bonferroni.

        Args:
            p_values: List of p-values

        Returns:
            List of (adjusted p-value, significant) tuples
        """
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_pvalues = np.array(p_values)[sorted_indices]

        adjusted = []
        cumulative_max = 0

        for i, (idx, p) in enumerate(zip(sorted_indices, sorted_pvalues)):
            adjusted_p = p * (n - i)
            adjusted_p = max(adjusted_p, cumulative_max)
            adjusted_p = min(adjusted_p, 1.0)
            cumulative_max = adjusted_p
            adjusted.append((adjusted_p, adjusted_p < self.alpha))

        # Reorder to original order
        result = [None] * n
        for i, idx in enumerate(sorted_indices):
            result[idx] = adjusted[i]

        return result

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"

    def _interpret_r(self, r: float) -> str:
        """Interpret correlation coefficient effect size."""
        r = abs(r)
        if r < 0.1:
            return "negligible"
        elif r < 0.3:
            return "small"
        elif r < 0.5:
            return "medium"
        else:
            return "large"

    def _interpret_eta_squared(self, eta_sq: float) -> str:
        """Interpret eta-squared effect size."""
        if eta_sq < 0.01:
            return "negligible"
        elif eta_sq < 0.06:
            return "small"
        elif eta_sq < 0.14:
            return "medium"
        else:
            return "large"

    def _interpret_epsilon_squared(self, eps_sq: float) -> str:
        """Interpret epsilon-squared effect size."""
        # Similar interpretation to eta-squared
        return self._interpret_eta_squared(eps_sq)


def compare_algorithms(results_df, baseline: str = "A*",
                       target: str = "AILS",
                       metric: str = "execution_time_ms") -> Dict[str, Any]:
    """
    Comprehensive comparison of two algorithms.

    Args:
        results_df: DataFrame with results
        baseline: Baseline algorithm
        target: Algorithm to compare
        metric: Metric column to compare

    Returns:
        Comparison results with multiple tests
    """
    tester = HypothesisTester()

    baseline_data = results_df[results_df['algorithm'] == baseline][metric].tolist()
    target_data = results_df[results_df['algorithm'] == target][metric].tolist()

    # Check if data is paired
    is_paired = len(baseline_data) == len(target_data)

    results = {
        'baseline': baseline,
        'target': target,
        'metric': metric,
        'n_baseline': len(baseline_data),
        'n_target': len(target_data),
        'tests': {},
    }

    if is_paired:
        results['tests']['paired_ttest'] = tester.paired_ttest(
            baseline_data, target_data
        ).to_dict()
        results['tests']['wilcoxon'] = tester.wilcoxon(
            baseline_data, target_data
        ).to_dict()
    else:
        results['tests']['independent_ttest'] = tester.independent_ttest(
            baseline_data, target_data
        ).to_dict()
        results['tests']['mann_whitney'] = tester.mann_whitney(
            baseline_data, target_data
        ).to_dict()

    return results
