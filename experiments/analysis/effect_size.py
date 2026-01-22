"""
Effect Size Calculations
=========================

Comprehensive effect size measures for algorithm comparison.
Effect sizes quantify the magnitude of differences, independent
of sample size.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EffectSizeResult:
    """Result of an effect size calculation."""
    name: str
    value: float
    interpretation: str
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'name': self.name,
            'value': self.value,
            'interpretation': self.interpretation,
        }
        if self.ci_lower is not None:
            result['ci_lower'] = self.ci_lower
            result['ci_upper'] = self.ci_upper
        return result


class EffectSizeCalculator:
    """
    Calculate various effect size measures.

    Supports:
    - Cohen's d (standardized mean difference)
    - Glass's delta
    - Hedges' g (bias-corrected d)
    - Cliff's delta (non-parametric)
    - Common language effect size
    - Relative improvement metrics
    """

    def cohens_d(self, group1: List[float], group2: List[float],
                 paired: bool = False) -> EffectSizeResult:
        """
        Calculate Cohen's d.

        Measures the standardized difference between two means.

        Args:
            group1: First group
            group2: Second group
            paired: Whether samples are paired

        Returns:
            EffectSizeResult
        """
        group1, group2 = np.array(group1), np.array(group2)

        if paired:
            # For paired samples, use the SD of differences
            diff = group1 - group2
            d = np.mean(diff) / np.std(diff, ddof=1)
        else:
            # Pooled standard deviation
            n1, n2 = len(group1), len(group2)
            var1 = np.var(group1, ddof=1)
            var2 = np.var(group2, ddof=1)

            pooled_std = np.sqrt(
                ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
            )
            d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

        return EffectSizeResult(
            name="Cohen's d",
            value=d,
            interpretation=self._interpret_cohens_d(d)
        )

    def hedges_g(self, group1: List[float], group2: List[float]) -> EffectSizeResult:
        """
        Calculate Hedges' g (bias-corrected Cohen's d).

        Better for small samples (n < 20).

        Args:
            group1: First group
            group2: Second group

        Returns:
            EffectSizeResult
        """
        group1, group2 = np.array(group1), np.array(group2)

        # Calculate Cohen's d first
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(
            ((n1 - 1) * np.var(group1, ddof=1) +
             (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2)
        )
        d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

        # Correction factor for small samples
        df = n1 + n2 - 2
        j = 1 - (3 / (4 * df - 1))  # Hedges' correction factor
        g = d * j

        return EffectSizeResult(
            name="Hedges' g",
            value=g,
            interpretation=self._interpret_cohens_d(g)
        )

    def glass_delta(self, treatment: List[float],
                    control: List[float]) -> EffectSizeResult:
        """
        Calculate Glass's delta.

        Uses control group SD only, useful when variances differ.

        Args:
            treatment: Treatment group
            control: Control group

        Returns:
            EffectSizeResult
        """
        treatment, control = np.array(treatment), np.array(control)

        control_std = np.std(control, ddof=1)
        delta = (np.mean(treatment) - np.mean(control)) / control_std if control_std > 0 else 0

        return EffectSizeResult(
            name="Glass's delta",
            value=delta,
            interpretation=self._interpret_cohens_d(delta)
        )

    def cliffs_delta(self, group1: List[float], group2: List[float]) -> EffectSizeResult:
        """
        Calculate Cliff's delta (non-parametric effect size).

        Robust to outliers and non-normality.
        Range: -1 to +1

        Args:
            group1: First group
            group2: Second group

        Returns:
            EffectSizeResult
        """
        group1, group2 = np.array(group1), np.array(group2)

        n1, n2 = len(group1), len(group2)

        # Count dominance
        greater = sum(1 for x in group1 for y in group2 if x > y)
        less = sum(1 for x in group1 for y in group2 if x < y)

        delta = (greater - less) / (n1 * n2)

        return EffectSizeResult(
            name="Cliff's delta",
            value=delta,
            interpretation=self._interpret_cliffs_delta(delta)
        )

    def common_language_effect_size(self, group1: List[float],
                                     group2: List[float]) -> EffectSizeResult:
        """
        Calculate Common Language Effect Size (CLES).

        Probability that a random sample from group1 is greater
        than a random sample from group2.

        Args:
            group1: First group (expected to be larger)
            group2: Second group

        Returns:
            EffectSizeResult with probability
        """
        group1, group2 = np.array(group1), np.array(group2)

        n1, n2 = len(group1), len(group2)

        # Count wins
        wins = sum(1 for x in group1 for y in group2 if x > y)
        ties = sum(1 for x in group1 for y in group2 if x == y)

        cles = (wins + 0.5 * ties) / (n1 * n2)

        return EffectSizeResult(
            name="Common Language Effect Size",
            value=cles,
            interpretation=f"{cles:.1%} probability of superiority"
        )

    def relative_improvement(self, baseline: List[float],
                             improved: List[float]) -> Dict[str, float]:
        """
        Calculate relative improvement metrics.

        Args:
            baseline: Baseline measurements
            improved: Improved measurements

        Returns:
            Dictionary with improvement metrics
        """
        baseline, improved = np.array(baseline), np.array(improved)

        baseline_mean = np.mean(baseline)
        improved_mean = np.mean(improved)

        absolute_diff = baseline_mean - improved_mean
        relative_diff = absolute_diff / baseline_mean if baseline_mean != 0 else 0
        speedup = baseline_mean / improved_mean if improved_mean != 0 else float('inf')

        return {
            'baseline_mean': baseline_mean,
            'improved_mean': improved_mean,
            'absolute_improvement': absolute_diff,
            'relative_improvement_pct': relative_diff * 100,
            'speedup_factor': speedup,
        }

    def eta_squared(self, groups: List[List[float]]) -> EffectSizeResult:
        """
        Calculate eta-squared for ANOVA.

        Proportion of variance explained by group membership.

        Args:
            groups: List of groups

        Returns:
            EffectSizeResult
        """
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)

        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_total = np.sum((all_data - grand_mean)**2)

        eta_sq = ss_between / ss_total if ss_total > 0 else 0

        return EffectSizeResult(
            name="Eta-squared",
            value=eta_sq,
            interpretation=self._interpret_eta_squared(eta_sq)
        )

    def omega_squared(self, groups: List[List[float]]) -> EffectSizeResult:
        """
        Calculate omega-squared (less biased than eta-squared).

        Args:
            groups: List of groups

        Returns:
            EffectSizeResult
        """
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        n_total = len(all_data)
        k = len(groups)

        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_within = sum(np.sum((np.array(g) - np.mean(g))**2) for g in groups)
        ms_within = ss_within / (n_total - k)

        omega_sq = (ss_between - (k - 1) * ms_within) / (ss_within + ss_between + ms_within)
        omega_sq = max(0, omega_sq)  # Can't be negative

        return EffectSizeResult(
            name="Omega-squared",
            value=omega_sq,
            interpretation=self._interpret_eta_squared(omega_sq)
        )

    def bootstrap_effect_size_ci(self, group1: List[float], group2: List[float],
                                  n_bootstrap: int = 10000,
                                  confidence: float = 0.95) -> Tuple[float, float, float]:
        """
        Bootstrap confidence interval for Cohen's d.

        Args:
            group1: First group
            group2: Second group
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level

        Returns:
            Tuple of (d, ci_lower, ci_upper)
        """
        group1, group2 = np.array(group1), np.array(group2)
        n1, n2 = len(group1), len(group2)

        bootstrap_ds = []

        for _ in range(n_bootstrap):
            sample1 = np.random.choice(group1, size=n1, replace=True)
            sample2 = np.random.choice(group2, size=n2, replace=True)

            pooled_std = np.sqrt(
                ((n1 - 1) * np.var(sample1, ddof=1) +
                 (n2 - 1) * np.var(sample2, ddof=1)) / (n1 + n2 - 2)
            )
            if pooled_std > 0:
                d = (np.mean(sample1) - np.mean(sample2)) / pooled_std
                bootstrap_ds.append(d)

        # Original effect size
        result = self.cohens_d(group1.tolist(), group2.tolist())

        # Confidence interval
        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_ds, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_ds, 100 * (1 - alpha / 2))

        return result.value, ci_lower, ci_upper

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d."""
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"

    def _interpret_cliffs_delta(self, delta: float) -> str:
        """Interpret Cliff's delta."""
        delta = abs(delta)
        if delta < 0.147:
            return "negligible"
        elif delta < 0.33:
            return "small"
        elif delta < 0.474:
            return "medium"
        else:
            return "large"

    def _interpret_eta_squared(self, eta_sq: float) -> str:
        """Interpret eta-squared."""
        if eta_sq < 0.01:
            return "negligible"
        elif eta_sq < 0.06:
            return "small"
        elif eta_sq < 0.14:
            return "medium"
        else:
            return "large"


def compute_all_effect_sizes(baseline: List[float],
                              treatment: List[float]) -> Dict[str, EffectSizeResult]:
    """
    Compute all relevant effect sizes for a comparison.

    Args:
        baseline: Baseline group
        treatment: Treatment group

    Returns:
        Dictionary of effect size results
    """
    calculator = EffectSizeCalculator()

    return {
        'cohens_d': calculator.cohens_d(baseline, treatment),
        'hedges_g': calculator.hedges_g(baseline, treatment),
        'glass_delta': calculator.glass_delta(treatment, baseline),
        'cliffs_delta': calculator.cliffs_delta(baseline, treatment),
        'cles': calculator.common_language_effect_size(baseline, treatment),
    }
