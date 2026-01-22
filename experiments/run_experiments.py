#!/usr/bin/env python3
"""
AILS Comprehensive Experiment Runner
=====================================

Main script for running all experiments for the Q1 journal paper.

Usage:
    python run_experiments.py --all                    # Run all experiments
    python run_experiments.py --quick                  # Quick test run
    python run_experiments.py --performance            # Performance benchmark
    python run_experiments.py --scalability            # Scalability analysis
    python run_experiments.py --sensitivity            # Parameter sensitivity
    python run_experiments.py --patterns               # Pattern analysis
    python run_experiments.py --comparative            # Full comparative study

Author: AILS Research Team
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from benchmarks.performance import PerformanceBenchmark, BenchmarkConfig
from benchmarks.scalability import ScalabilityBenchmark, ScalabilityConfig
from benchmarks.sensitivity import ParameterSensitivityBenchmark, SensitivityConfig
from benchmarks.patterns import PatternBenchmark, PatternConfig
from benchmarks.comparative import ComparativeStudy, ComparativeConfig
from analysis.statistics import StatisticalAnalyzer
from analysis.hypothesis import HypothesisTester, compare_algorithms
from analysis.effect_size import EffectSizeCalculator, compute_all_effect_sizes
from analysis.latex_export import LaTeXExporter, export_to_latex
from visualization.plotting import ResultsPlotter, PlotConfig


def create_output_dirs():
    """Create all output directories."""
    dirs = [
        'results/performance',
        'results/scalability',
        'results/sensitivity',
        'results/patterns',
        'results/comparative',
        'results/figures',
        'results/latex',
        'results/analysis',
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def run_performance_benchmark(quick: bool = False, seed: int = 42) -> dict:
    """Run performance benchmark experiments."""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)

    if quick:
        config = BenchmarkConfig(
            grid_sizes=[(100, 100), (200, 200)],
            densities=[0.2, 0.3],
            patterns=["random", "clustered", "maze"],
            trials_per_config=30,
            algorithms=["AILS", "A*", "Dijkstra", "BFS"],
            seed=seed,
        )
    else:
        config = BenchmarkConfig(
            grid_sizes=[(50, 50), (100, 100), (200, 200), (300, 300), (400, 400), (500, 500)],
            densities=[0.1, 0.2, 0.3, 0.4],
            patterns=["random", "maze", "clustered", "open", "mixed", "room"],
            trials_per_config=100,
            seed=seed,
        )

    benchmark = PerformanceBenchmark(config)
    benchmark.run(verbose=True)
    csv_path = benchmark.save_results()

    summary = benchmark.get_summary_statistics()
    speedups = benchmark.compute_speedup("A*")

    print("\n--- Summary ---")
    for algo, stats in summary.items():
        print(f"{algo}: {stats['time_mean']:.2f}ms ({stats['nodes_mean']:.0f} nodes)")

    print("\n--- AILS Speedups ---")
    if "AILS" in speedups:
        print(f"vs A*: {speedups.get('AILS', {}).get('time_speedup', 0):.2f}x")

    return {
        'csv_path': csv_path,
        'summary': summary,
        'speedups': speedups,
    }


def run_scalability_benchmark(quick: bool = False, seed: int = 42) -> dict:
    """Run scalability analysis experiments."""
    print("\n" + "=" * 60)
    print("SCALABILITY ANALYSIS")
    print("=" * 60)

    if quick:
        config = ScalabilityConfig(
            grid_sizes=[(50, 50), (100, 100), (200, 200), (300, 300)],
            trials_per_size=30,
            algorithms=["AILS", "A*", "Dijkstra"],
            track_memory=False,
            seed=seed,
        )
    else:
        config = ScalabilityConfig(
            grid_sizes=[
                (50, 50), (100, 100), (150, 150), (200, 200),
                (250, 250), (300, 300), (400, 400), (500, 500),
                (600, 600), (750, 750), (1000, 1000)
            ],
            trials_per_size=50,
            seed=seed,
        )

    benchmark = ScalabilityBenchmark(config)
    benchmark.run(verbose=True)
    csv_path = benchmark.save_results()

    analysis = benchmark.analyze_scaling()
    summary = benchmark.get_scaling_summary()

    print("\n--- Scaling Analysis ---")
    for algo, data in analysis.items():
        print(f"{algo}: {data['complexity_estimate']}")

    return {
        'csv_path': csv_path,
        'analysis': analysis,
        'summary': summary,
    }


def run_sensitivity_analysis(quick: bool = False, seed: int = 42) -> dict:
    """Run parameter sensitivity experiments."""
    print("\n" + "=" * 60)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 60)

    if quick:
        config = SensitivityConfig(
            r_min_values=[2, 4],
            r_max_values=[10, 20],
            alpha_values=[0.5, 1.0, 1.5],
            beta_values=[0.0, 0.3],
            window_size_values=[3, 7],
            grid_sizes=[(100, 100)],
            densities=[0.2],
            patterns=["random"],
            trials_per_config=20,
            seed=seed,
        )
    else:
        config = SensitivityConfig(seed=seed)

    benchmark = ParameterSensitivityBenchmark(config)
    benchmark.run(verbose=True)
    csv_path = benchmark.save_results()

    analysis = benchmark.analyze_parameter_effects()
    recommendations = benchmark.get_recommendations()

    print("\n--- Parameter Effects ---")
    for param, data in analysis.items():
        print(f"{param}: Best={data['best_value']}, "
              f"Sensitivity={data['sensitivity']} ({data['effect_range_pct']:.1f}%)")

    return {
        'csv_path': csv_path,
        'analysis': analysis,
        'recommendations': recommendations,
    }


def run_pattern_analysis(quick: bool = False, seed: int = 42) -> dict:
    """Run pattern analysis experiments."""
    print("\n" + "=" * 60)
    print("OBSTACLE PATTERN ANALYSIS")
    print("=" * 60)

    if quick:
        config = PatternConfig(
            patterns=["random", "maze", "clustered", "mixed"],
            grid_sizes=[(100, 100), (200, 200)],
            densities=[0.2, 0.3],
            algorithms=["AILS", "A*", "Dijkstra"],
            trials_per_config=30,
            seed=seed,
        )
    else:
        config = PatternConfig(seed=seed)

    benchmark = PatternBenchmark(config)
    benchmark.run(verbose=True)
    csv_path = benchmark.save_results()

    analysis = benchmark.analyze_by_pattern()
    rankings = benchmark.get_pattern_rankings()

    print("\n--- Pattern Rankings (AILS speedup) ---")
    for pattern, speedup in rankings['rankings']:
        print(f"{pattern}: {speedup:.2f}x")

    return {
        'csv_path': csv_path,
        'analysis': analysis,
        'rankings': rankings,
    }


def run_comparative_study(quick: bool = False, seed: int = 42) -> dict:
    """Run full comparative study."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE COMPARATIVE STUDY")
    print("=" * 60)

    if quick:
        config = ComparativeConfig(
            synthetic_sizes=[(100, 100), (200, 200)],
            synthetic_densities=[0.2, 0.3],
            synthetic_patterns=["random", "clustered", "maze"],
            algorithms=["AILS", "A*", "Dijkstra", "BFS"],
            trials_per_synthetic=30,
            use_benchmark_maps=False,
            seed=seed,
        )
    else:
        config = ComparativeConfig(seed=seed)

    study = ComparativeStudy(config)
    analysis = study.run_full_study(verbose=True)

    return analysis


def generate_all_outputs(results: dict):
    """Generate all output files from results."""
    print("\n" + "=" * 60)
    print("GENERATING OUTPUTS")
    print("=" * 60)

    output_dir = Path('results/analysis')

    # Save combined results
    combined_path = output_dir / 'all_results.json'
    with open(combined_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved combined results to {combined_path}")

    # Generate LaTeX tables
    if 'comparative' in results:
        print("\nGenerating LaTeX tables...")
        latex_files = export_to_latex(results['comparative'], 'results/latex')
        print(f"Generated {len(latex_files)} LaTeX files")

    # Generate figures
    print("\nGenerating figures...")
    plotter = ResultsPlotter(PlotConfig(output_dir='results/figures'))

    if 'performance' in results and 'summary' in results['performance']:
        plotter.algorithm_comparison_bar(
            results['performance']['summary'],
            metric='time_mean',
            ylabel='Execution Time (ms)',
            title='Algorithm Performance Comparison',
            filename='performance_comparison'
        )
        print("  - performance_comparison.pdf")

    if 'performance' in results and 'speedups' in results['performance']:
        speedup_vals = {k: v.get('time_speedup', 1) for k, v in results['performance']['speedups'].items()}
        if speedup_vals:
            plotter.speedup_bar(
                speedup_vals,
                title='AILS Speedup Over Baseline Algorithms',
                filename='speedup_comparison'
            )
            print("  - speedup_comparison.pdf")

    print("\nAll outputs generated successfully!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='AILS Comprehensive Experiment Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py --all                    Run all experiments
  python run_experiments.py --quick                  Quick test run
  python run_experiments.py --performance            Performance only
  python run_experiments.py --scalability --quick    Quick scalability
        """
    )

    parser.add_argument('--all', action='store_true',
                       help='Run all experiments')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode with reduced parameters')
    parser.add_argument('--performance', action='store_true',
                       help='Run performance benchmark')
    parser.add_argument('--scalability', action='store_true',
                       help='Run scalability analysis')
    parser.add_argument('--sensitivity', action='store_true',
                       help='Run parameter sensitivity analysis')
    parser.add_argument('--patterns', action='store_true',
                       help='Run pattern analysis')
    parser.add_argument('--comparative', action='store_true',
                       help='Run full comparative study')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--no-outputs', action='store_true',
                       help='Skip generating output files')

    args = parser.parse_args()

    # If no specific experiment selected, show help
    if not any([args.all, args.performance, args.scalability,
                args.sensitivity, args.patterns, args.comparative]):
        parser.print_help()
        print("\nError: Please select at least one experiment to run.")
        sys.exit(1)

    # Create output directories
    create_output_dirs()

    # Track results
    results = {}
    start_time = time.time()

    print("=" * 60)
    print("AILS EXPERIMENT SUITE")
    print("=" * 60)
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print(f"Seed: {args.seed}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run selected experiments
    if args.all or args.performance:
        results['performance'] = run_performance_benchmark(args.quick, args.seed)

    if args.all or args.scalability:
        results['scalability'] = run_scalability_benchmark(args.quick, args.seed)

    if args.all or args.sensitivity:
        results['sensitivity'] = run_sensitivity_analysis(args.quick, args.seed)

    if args.all or args.patterns:
        results['patterns'] = run_pattern_analysis(args.quick, args.seed)

    if args.all or args.comparative:
        results['comparative'] = run_comparative_study(args.quick, args.seed)

    # Generate outputs
    if not args.no_outputs:
        generate_all_outputs(results)

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("EXPERIMENT SUITE COMPLETE")
    print("=" * 60)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Results saved in: results/")


if __name__ == "__main__":
    main()
