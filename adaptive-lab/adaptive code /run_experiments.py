"""
AILS Experiment Runner
======================

This is the main executable script for running the AILS experiments.
It imports all core logic from 'ails_core.py' and runs the
experiments within an 'if __name__ == "__main__"' block.

This structure is required to make multiprocessing compatible with
Jupyter notebooks, Windows, and macOS.

To run:
1.  Save this file as 'run_experiments.py'.
2.  Save the other file as 'ails_core.py' in the same directory.
3.  Run this file from your terminal:
    python run_experiments.py
    
    ...or copy the contents of this file into a single
    Jupyter Notebook cell and run it.
"""

import sys
import numpy as np
from ails_core import *

# ============================================================================
# MAIN EXPERIMENT FUNCTION
# ============================================================================




def run_comprehensive_experiment():
    """Execute the complete experimental framework"""
    
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║     ADAPTIVE INCREMENTAL LINE SEARCH (AILS) FRAMEWORK         ║
    ║         Q1 Journal-Grade Comprehensive Evaluation              ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize Q1-level configuration
    config = ExperimentConfig(
        grid_sizes=[50, 100, 200, 300, 400], # Extensive grid sizes
        obstacle_densities=list(np.linspace(0.05, 0.40, 8)), # 8 density levels
        trials_per_config=20,  # More trials for statistical power
        start_goal_strategy='random', # CRITICAL: use random pairs
        num_random_pairs=10,          # 10 random pairs per grid
        use_multiprocessing=True,
        num_workers=max(1, mp.cpu_count() - 2) # Leave 2 CPUs free
    )
    
    # Run experiments
    runner = ExperimentRunner(config)
    df = runner.run_experiments()
    
    if df.empty:
        print("\n" + "="*80)
        print("EXPERIMENT FAILED: No results generated.")
        print("="*80)
        return None, None
        
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED")
    print("="*80)
    print(f"Total experiment data points (rows): {len(df)}")
    print(f"Unique configurations: {len(df.groupby(['grid_size', 'obstacle_density', 'pattern', 'trial_id']))}")
    
    # Statistical analysis
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)
    
    stats_results = StatisticalAnalysis.analyze_results(df)
    
    # Print key findings
    for alg in ['A*', 'Dijkstra', 'BFS']:
        if alg in stats_results:
            print(f"\n--- Analysis for {alg} ---")
            mean_imp, lower_imp, upper_imp = stats_results[alg]['time_improvement']['ails_vs_std']
            print(f"  AILS vs Standard Time: {mean_imp:.2f}% improvement (95% CI: [{lower_imp:.2f}, {upper_imp:.2f}])")
            print(f"    - P-value (t-test): {stats_results[alg]['t_tests']['ails_vs_std']['p_value']:.2e}")
            print(f"    - Effect size (Cohen's d): {stats_results[alg]['effect_sizes']['ails_vs_std']:.3f}")
            
            mean_qual, lower_qual, upper_qual = stats_results[alg]['path_quality_ratio']
            print(f"  AILS Path Quality Ratio: {mean_qual:.3f} (95% CI: [{lower_qual:.3f}, {upper_qual:.3f}])")
            print(f"    (1.0 = same as standard, >1.0 = worse path)")
    
    if stats_results['tukey'].get('algorithm') is not None:
        print("\n--- Tukey's HSD (Algorithm) ---")
        print(stats_results['tukey']['algorithm'])

    # Visualization
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Run all standard and new Q1-level plots
    ComprehensiveVisualization.plot_performance_comparison(df)
    ComprehensiveVisualization.plot_heatmaps(df)
    ComprehensiveVisualization.plot_performance_distributions(df)
    ComprehensiveVisualization.plot_scalability_analysis(df)
    ComprehensiveVisualization.plot_tradeoff_analysis(df)
    ComprehensiveVisualization.plot_faceted_heatmaps_by_pattern(df)
    
    # Generate LaTeX tables
    if config.generate_latex_tables:
        print("\n" + "="*80)
        print("GENERATING LATEX TABLES")
        print("="*80)
        LatexTableGenerator.save_tables(df, stats_results)
    
    # Summary report
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    overall_improvement = df['ails_pred_time_improvement'].mean()
    best_case = df['ails_pred_time_improvement'].max()
    worst_case = df['ails_pred_time_improvement'].min()
    
    print(f"Overall AILS Performance:")
    print(f"  Average improvement: {overall_improvement:.2f}%")
    print(f"  Best case improvement: {best_case:.2f}%")
    print(f"  Worst case improvement: {worst_case:.2f}%")
    
    print(f"\nBest performing algorithm: {df.groupby('algorithm')['ails_pred_time_improvement'].mean().idxmax()}")
    print(f"Best performing pattern: {df.groupby('pattern')['ails_pred_time_improvement'].mean().idxmax()}")
    
    print("\nFiles saved:")
    print(f"  - CSV: results/data/experiment_results_*.csv")
    print(f"  - Figures: results/figures/*.png")
    print(f"  - LaTeX: results/analysis/latex_tables_*.tex")
    
    return df, stats_results

# ============================================================================
# PARAMETER SENSITIVITY ANALYSIS (PSA)
# ============================================================================

def run_parameter_sensitivity_experiment():
    """Execute the PSA framework"""
    
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║         PARAMETER SENSITIVITY ANALYSIS (PSA)                  ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    # PSA Configuration
    psa_config = ExperimentConfig(
        use_multiprocessing=True,
        num_workers=max(1, mp.cpu_count() - 2)
    )
    psa_runner = ExperimentRunner(psa_config)
    
    # Define parameter ranges to test
    window_sizes = [3, 5, 7, 9, 11]
    lookaheads = [0, 3, 5, 7, 10]
    
    # Run the PSA
    df_psa = psa_runner.run_sensitivity_analysis(window_sizes, lookaheads)
    
    if df_psa.empty:
        print("\n" + "="*80)
        print("PSA FAILED: No results generated.")
        print("="*80)
        return None
    
    # Plot PSA results
    print("\n" + "="*80)
    print("GENERATING PSA VISUALIZATIONS")
    print("="*80)
    ComprehensiveVisualization.plot_parameter_sensitivity(df_psa)
    
    print("\nFiles saved:")
    print(f"  - CSV: results/data/psa_results_*.csv")
    print(f"  - Figures: results/figures/psa_results_*.png")
    
    return df_psa

# ============================================================================
# QUICK DEMO
# ============================================================================

def quick_demo():
    """Run a quick demonstration with visualization"""
    print("Running Quick Demo...")
    
    # Generate sample grid
    grid = AdvancedGridGenerator.generate(
        100, 0.2, AdvancedGridGenerator.ObstaclePattern.CLUSTERED, seed=42
    )
    
    start = (0, 0)
    goal = (99, 99)
    
    # Initialize algorithms
    config = ExperimentConfig()
    ails = AdaptiveILS(grid, config)
    
    # Run A* with different configurations
    result_std = ComprehensivePathfinding.a_star(grid, start, goal, None)
    result_ails = ails.search(ComprehensivePathfinding.a_star, start, goal, 'predictive')
    
    print(f"\nResults:")
    print(f"Standard A*: Time={result_std['time']:.4f}s, Visited={result_std['visited']}, Cost={result_std['cost']:.2f}")
    print(f"AILS A*:     Time={result_ails['time']:.4f}s, Visited={result_ails['visited']}, Cost={result_ails['cost']:.2f}")
    if result_std['time'] > 0:
        print(f"Time Improvement: {((result_std['time'] - result_ails['time'])/result_std['time']*100):.1f}%")
    if result_std['cost'] > 0:
        print(f"Path Cost Ratio:  {(result_ails['cost']/result_std['cost']):.3f}")

    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    
    # Plot grid with paths
    axes[0].imshow(grid, cmap='binary')
    if result_std['path']:
        path_array = np.array(result_std['path'])
        axes[0].plot(path_array[:, 1], path_array[:, 0], 'r-', linewidth=2, label='Standard A*')
    axes[0].set_title(f"Standard A* Path\nCost: {result_std['cost']:.2f}")
    axes[0].axis('off')
    
    axes[1].imshow(grid, cmap='binary')
    
    # Show corridor
    corridor = ails.corridor_builder.compute_adaptive_corridor(start, goal, 'predictive')
    corridor_mask = np.ones_like(grid, dtype=float) * np.nan # Use NaN for empty
    for point in corridor:
        if 0 <= point[0] < 100 and 0 <= point[1] < 100:
            corridor_mask[point[0], point[1]] = 1
    axes[1].imshow(corridor_mask, cmap='Greens', alpha=0.3)
    
    if result_ails['path']:
        path_array = np.array(result_ails['path'])
        axes[1].plot(path_array[:, 1], path_array[:, 0], 'g-', linewidth=2, label='AILS')
        
    axes[1].set_title(f"AILS Path with Adaptive Corridor\nCost: {result_ails['cost']:.2f}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# NOTEBOOK INTERFACE
# ============================================================================

if __name__ == "__main__":
    
    # This __name__ == "__main__" block is ESSENTIAL for
    # multiprocessing to work correctly, especially in
    # Jupyter notebooks or when running as a script.
    
    print("Select an option in the 'if __name__ == \"__main__\":' block:")
    print("1. Run quick_demo()")
    print("2. Uncomment and run run_comprehensive_experiment()")
    print("3. Uncomment and run run_parameter_sensitivity_experiment()")
    print("-" * 60)

    # ------------------------------------------------------------------------
    # Your requested lines are below:
    # ------------------------------------------------------------------------
    
    # 1. To run a short experiment (quick demo):
    #quick_demo()
    
    # 2. To run the whole experiment (uncomment the line below):
     print("\nStarting Comprehensive Experiment... (This will take a long time)\n")
     df, stats = run_comprehensive_experiment()
     print("\nComprehensive Experiment Finished.")
    
    # 3. To run the parameter sensitivity analysis (uncomment the line below):
    # print("\nStarting Parameter Sensitivity Analysis...\n")
    # df_psa = run_parameter_sensitivity_experiment()
    # print("\nPSA Finished.")
    
    print("\nScript finished.")