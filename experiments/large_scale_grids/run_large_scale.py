#!/usr/bin/env python3
"""
Large-Scale Grid Experiment Runner
====================================

CLI entry point for running AILS large-scale grid experiments.

Usage:
    python run_large_scale.py --full                  # Full benchmark (1K, 5K, 10K)
    python run_large_scale.py --quick                 # Quick test (1K only)
    python run_large_scale.py --medium                # Medium run (1K, 5K)
    python run_large_scale.py --size 1000 5000        # Custom sizes
    python run_large_scale.py --full --seed 123       # Custom seed
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from large_scale_grids.large_scale_benchmark import (
    LargeScaleBenchmark,
    LargeScaleConfig,
)


def main():
    parser = argparse.ArgumentParser(
        description='AILS Large-Scale Grid Experiment Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_large_scale.py --full                  Full benchmark (1K, 5K, 10K)
  python run_large_scale.py --quick                 Quick test (1K only, 10 trials)
  python run_large_scale.py --medium                Medium run (1K, 5K)
  python run_large_scale.py --size 1000 5000 10000  Custom sizes
  python run_large_scale.py --full --all-algos      Include Dijkstra, BFS, Bidir.
  python run_large_scale.py --full --seed 123       Custom seed
        """
    )

    # Run mode
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--full', action='store_true',
                      help='Full benchmark: 1000, 5000, 10000 grids')
    mode.add_argument('--quick', action='store_true',
                      help='Quick test: 1000 grid only, 10 trials, 1 density')
    mode.add_argument('--medium', action='store_true',
                      help='Medium run: 1000 and 5000 grids')
    mode.add_argument('--size', type=int, nargs='+', metavar='N',
                      help='Custom grid sizes (e.g., --size 1000 5000)')

    # Options
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--all-algos', action='store_true',
                        help='Include all algorithms (adds Dijkstra, BFS, Bidirectional-A*)')
    parser.add_argument('--patterns', nargs='+',
                        default=None,
                        help='Patterns to test (default: random clustered mixed)')
    parser.add_argument('--densities', type=float, nargs='+',
                        default=None,
                        help='Densities to test (default: 0.2 0.3)')
    parser.add_argument('--no-memory', action='store_true',
                        help='Disable memory tracking (faster)')
    parser.add_argument('--output-dir', type=str,
                        default='results/large_scale_grids',
                        help='Output directory')

    args = parser.parse_args()

    # Build configuration
    if args.quick:
        grid_sizes = [(1000, 1000)]
        trials_by_size = {1000: 10}
        timeout_by_size = {1000: 30}
        densities = [0.2]
        patterns = ["random"]
    elif args.medium:
        grid_sizes = [(1000, 1000), (5000, 5000)]
        trials_by_size = {1000: 20, 5000: 10}
        timeout_by_size = {1000: 30, 5000: 120}
        densities = args.densities or [0.2, 0.3]
        patterns = args.patterns or ["random", "clustered", "mixed"]
    elif args.size:
        grid_sizes = [(s, s) for s in args.size]
        trials_by_size = {s: max(5, 20 - s // 1000) for s in args.size}
        timeout_by_size = {s: min(180, 30 + s // 50) for s in args.size}
        densities = args.densities or [0.2, 0.3]
        patterns = args.patterns or ["random", "clustered", "mixed"]
    else:  # --full
        grid_sizes = [(1000, 1000), (5000, 5000), (10000, 10000)]
        trials_by_size = {1000: 20, 5000: 10, 10000: 5}
        timeout_by_size = {1000: 30, 5000: 120, 10000: 180}
        densities = args.densities or [0.2, 0.3]
        patterns = args.patterns or ["random", "clustered", "mixed"]

    algorithms = ["AILS", "A*", "JPS"]
    if args.all_algos:
        algorithms = ["AILS", "A*", "Dijkstra", "BFS", "JPS", "Bidirectional-A*"]

    config = LargeScaleConfig(
        grid_sizes=grid_sizes,
        densities=densities,
        patterns=patterns,
        algorithms=algorithms,
        trials_by_size=trials_by_size,
        timeout_per_trial=timeout_by_size,
        seed=args.seed,
        track_memory=not args.no_memory,
        output_dir=args.output_dir,
    )

    # Print configuration
    print("=" * 70)
    print("AILS LARGE-SCALE GRID EXPERIMENT")
    print("=" * 70)
    print(f"Grid sizes:  {[f'{w}x{h}' for w, h in config.grid_sizes]}")
    print(f"Densities:   {config.densities}")
    print(f"Patterns:    {config.patterns}")
    print(f"Algorithms:  {config.algorithms}")
    print(f"Trials:      {config.trials_by_size}")
    print(f"Timeouts:    {config.timeout_per_trial}")
    print(f"Seed:        {config.seed}")
    print(f"Memory:      {'ON' if config.track_memory else 'OFF'}")
    print(f"Output:      {config.output_dir}")
    print(f"Started:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Run
    start = time.time()
    benchmark = LargeScaleBenchmark(config)
    benchmark.run(verbose=True)

    # Save results
    csv_path, json_path = benchmark.save_results()
    summary_path = benchmark.save_paper_summary()
    analysis_path = benchmark.save_analysis_json()

    elapsed = time.time() - start

    # Print final summary
    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Duration: {elapsed/60:.1f} minutes")
    print()
    print("Output files:")
    print(f"  Raw CSV:           {csv_path}")
    print(f"  Raw JSON:          {json_path}")
    print(f"  Paper Summary:     {summary_path}")
    print(f"  Analysis JSON:     {analysis_path}")
    print()
    print("To share results for paper updates, provide the content of:")
    print(f"  {summary_path}")
    print()

    # Print the paper summary to stdout
    print("-" * 70)
    print("PAPER SUMMARY (copy-paste this for paper updates):")
    print("-" * 70)
    print(benchmark.generate_paper_summary())


if __name__ == "__main__":
    main()
