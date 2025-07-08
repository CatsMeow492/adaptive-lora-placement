#!/usr/bin/env python3
"""
Results analysis script for comparing adaptive LoRA strategies.
Analyzes performance, efficiency, and trade-offs across all experiments.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze results from adaptive LoRA experiments"
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/analysis",
        help="Output directory for analysis results"
    )
    parser.add_argument(
        "--figures_dir",
        type=str,
        default="paper/figures",
        help="Directory to save figures for paper"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "pdf", "svg"],
        default="png",
        help="Figure format"
    )
    
    return parser.parse_args()


def load_experiment_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all experiment results from the results directory."""
    logger.info(f"Loading experiment results from: {results_dir}")
    
    results = []
    
    # Find all experiment directories
    for exp_dir in results_dir.iterdir():
        if exp_dir.is_dir():
            results_file = exp_dir / "results.json"
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        result = json.load(f)
                    
                    # Add experiment directory name
                    result['experiment_dir'] = exp_dir.name
                    
                    # Load allocation info if available
                    allocation_file = exp_dir / "allocation.json"
                    if allocation_file.exists():
                        with open(allocation_file, 'r') as f:
                            allocation = json.load(f)
                        result['allocation'] = allocation
                    
                    results.append(result)
                    logger.info(f"Loaded: {exp_dir.name}")
                    
                except Exception as e:
                    logger.warning(f"Failed to load {results_file}: {e}")
    
    logger.info(f"Loaded {len(results)} experiment results")
    return results


def create_results_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a pandas DataFrame from experiment results."""
    logger.info("Creating results DataFrame...")
    
    # Extract key metrics for comparison
    data = []
    for result in results:
        row = {
            'experiment_name': result['experiment_name'],
            'experiment_dir': result['experiment_dir'],
            'strategy': result.get('strategy', 'baseline'),
            'lora_rank': result.get('lora_rank', result.get('base_rank', 0)),
            'final_train_loss': result['final_train_loss'],
            'final_eval_loss': result['final_eval_loss'],
            'final_perplexity': result['final_perplexity'],
            'trainable_parameters': result['trainable_parameters'],
            'total_parameters': result['total_parameters'],
            'parameter_efficiency': result['parameter_efficiency'],
            'training_time_seconds': result['training_time_seconds'],
            'dataset_size': result['dataset_size'],
            'num_epochs': result['num_epochs'],
            'learning_rate': result['learning_rate'],
            'seed': result['seed'],
        }
        
        # Add adaptive-specific info
        if 'allocation' in result:
            allocation = result['allocation']
            if 'ranks' in allocation:
                ranks = allocation['ranks']
                row['avg_rank'] = np.mean(ranks)
                row['min_rank'] = min(ranks)
                row['max_rank'] = max(ranks)
                row['rank_std'] = np.std(ranks)
        
        data.append(row)
    
    df = pd.DataFrame(data)
    logger.info(f"Created DataFrame with {len(df)} experiments and {len(df.columns)} columns")
    
    return df


def analyze_performance_efficiency(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze performance vs efficiency trade-offs."""
    logger.info("Analyzing performance-efficiency trade-offs...")
    
    analysis = {
        'summary_stats': df.groupby('strategy').agg({
            'final_eval_loss': ['mean', 'std'],
            'final_perplexity': ['mean', 'std'],
            'trainable_parameters': ['mean', 'std'],
            'parameter_efficiency': ['mean', 'std'],
            'training_time_seconds': ['mean', 'std']
        }).round(4),
        
        'best_performance': {
            'lowest_loss': df.loc[df['final_eval_loss'].idxmin()],
            'lowest_perplexity': df.loc[df['final_perplexity'].idxmin()],
            'most_efficient': df.loc[df['parameter_efficiency'].idxmin()],
            'fastest_training': df.loc[df['training_time_seconds'].idxmin()]
        },
        
        'strategy_rankings': {}
    }
    
    # Rank strategies by different criteria
    criteria = ['final_eval_loss', 'final_perplexity', 'parameter_efficiency', 'training_time_seconds']
    for criterion in criteria:
        ascending = True if criterion in ['final_eval_loss', 'final_perplexity', 'parameter_efficiency', 'training_time_seconds'] else False
        if criterion == 'training_time_seconds':
            ascending = True  # Lower is better
        else:
            ascending = True if 'loss' in criterion or 'perplexity' in criterion or 'efficiency' in criterion else False
        
        ranking = df.groupby('strategy')[criterion].mean().sort_values(ascending=ascending)
        analysis['strategy_rankings'][criterion] = ranking.to_dict()
    
    return analysis


def calculate_efficiency_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate additional efficiency metrics."""
    logger.info("Calculating efficiency metrics...")
    
    # Efficiency ratio: performance per parameter
    df['perplexity_per_param'] = df['final_perplexity'] / (df['trainable_parameters'] / 1e6)  # Per million params
    
    # Normalized metrics (relative to baseline rank 8)
    baseline_8 = df[df['strategy'] == 'baseline'].copy()
    if len(baseline_8) > 0:
        baseline_loss = baseline_8['final_eval_loss'].iloc[0]
        baseline_params = baseline_8['trainable_parameters'].iloc[0]
        
        df['loss_improvement'] = (baseline_loss - df['final_eval_loss']) / baseline_loss
        df['param_ratio'] = df['trainable_parameters'] / baseline_params
    
    return df


def create_comparison_plots(df: pd.DataFrame, output_dir: Path, figures_dir: Path, format: str):
    """Create comparison plots for the analysis."""
    logger.info("Creating comparison plots...")
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up the plotting style
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10
    })
    
    # 1. Performance vs Parameters scatter plot
    plt.figure(figsize=(10, 6))
    strategies = df['strategy'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(strategies)))
    
    for i, strategy in enumerate(strategies):
        strategy_data = df[df['strategy'] == strategy]
        plt.scatter(
            strategy_data['trainable_parameters'] / 1e6,  # Convert to millions
            strategy_data['final_perplexity'],
            label=strategy.replace('_', ' ').title(),
            alpha=0.7,
            s=100,
            color=colors[i]
        )
    
    plt.xlabel('Trainable Parameters (Millions)')
    plt.ylabel('Final Perplexity')
    plt.title('Performance vs Parameter Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_dir / f'performance_vs_parameters.{format}', dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / f'performance_vs_parameters.{format}', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Strategy comparison bar plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Evaluation Loss
    strategy_means = df.groupby('strategy')['final_eval_loss'].mean()
    strategy_stds = df.groupby('strategy')['final_eval_loss'].std()
    axes[0, 0].bar(range(len(strategy_means)), strategy_means.values, 
                   yerr=strategy_stds.values, capsize=5, alpha=0.7)
    axes[0, 0].set_title('Final Evaluation Loss')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_xticks(range(len(strategy_means)))
    axes[0, 0].set_xticklabels([s.replace('_', ' ').title() for s in strategy_means.index], rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Perplexity
    strategy_means = df.groupby('strategy')['final_perplexity'].mean()
    strategy_stds = df.groupby('strategy')['final_perplexity'].std()
    axes[0, 1].bar(range(len(strategy_means)), strategy_means.values, 
                   yerr=strategy_stds.values, capsize=5, alpha=0.7)
    axes[0, 1].set_title('Final Perplexity')
    axes[0, 1].set_ylabel('Perplexity')
    axes[0, 1].set_xticks(range(len(strategy_means)))
    axes[0, 1].set_xticklabels([s.replace('_', ' ').title() for s in strategy_means.index], rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Parameters
    strategy_means = df.groupby('strategy')['trainable_parameters'].mean() / 1e6
    strategy_stds = df.groupby('strategy')['trainable_parameters'].std() / 1e6
    axes[1, 0].bar(range(len(strategy_means)), strategy_means.values, 
                   yerr=strategy_stds.values, capsize=5, alpha=0.7)
    axes[1, 0].set_title('Trainable Parameters')
    axes[1, 0].set_ylabel('Parameters (Millions)')
    axes[1, 0].set_xticks(range(len(strategy_means)))
    axes[1, 0].set_xticklabels([s.replace('_', ' ').title() for s in strategy_means.index], rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Training Time
    strategy_means = df.groupby('strategy')['training_time_seconds'].mean() / 60  # Convert to minutes
    strategy_stds = df.groupby('strategy')['training_time_seconds'].std() / 60
    axes[1, 1].bar(range(len(strategy_means)), strategy_means.values, 
                   yerr=strategy_stds.values, capsize=5, alpha=0.7)
    axes[1, 1].set_title('Training Time')
    axes[1, 1].set_ylabel('Time (Minutes)')
    axes[1, 1].set_xticks(range(len(strategy_means)))
    axes[1, 1].set_xticklabels([s.replace('_', ' ').title() for s in strategy_means.index], rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'strategy_comparison.{format}', dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / f'strategy_comparison.{format}', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Efficiency frontier plot
    if 'perplexity_per_param' in df.columns:
        plt.figure(figsize=(10, 6))
        for i, strategy in enumerate(strategies):
            strategy_data = df[df['strategy'] == strategy]
            plt.scatter(
                strategy_data['parameter_efficiency'] * 100,  # Convert to percentage
                strategy_data['final_perplexity'],
                label=strategy.replace('_', ' ').title(),
                alpha=0.7,
                s=100,
                color=colors[i]
            )
        
        plt.xlabel('Parameter Efficiency (%)')
        plt.ylabel('Final Perplexity')
        plt.title('Efficiency Frontier: Perplexity vs Parameter Efficiency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(output_dir / f'efficiency_frontier.{format}', dpi=300, bbox_inches='tight')
        plt.savefig(figures_dir / f'efficiency_frontier.{format}', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Plots saved to {output_dir} and {figures_dir}")


def generate_summary_report(df: pd.DataFrame, analysis: Dict[str, Any], output_dir: Path):
    """Generate a comprehensive summary report."""
    logger.info("Generating summary report...")
    
    report_file = output_dir / "analysis_report.md"
    
    with open(report_file, 'w') as f:
        f.write("# Adaptive LoRA: Experimental Results Analysis\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overview
        f.write("## Overview\n\n")
        f.write(f"- **Total Experiments:** {len(df)}\n")
        f.write(f"- **Strategies Tested:** {', '.join(df['strategy'].unique())}\n")
        f.write(f"- **Dataset Size:** {df['dataset_size'].iloc[0]} examples\n")
        f.write(f"- **Training Epochs:** {df['num_epochs'].iloc[0]}\n\n")
        
        # Summary Statistics
        f.write("## Summary Statistics\n\n")
        f.write("### Performance Metrics by Strategy\n\n")
        summary_stats = analysis['summary_stats']
        f.write(summary_stats.to_string())
        f.write("\n\n")
        
        # Best Performers
        f.write("## Best Performers\n\n")
        best = analysis['best_performance']
        
        f.write("### Lowest Evaluation Loss\n")
        f.write(f"- **Strategy:** {best['lowest_loss']['strategy']}\n")
        f.write(f"- **Loss:** {best['lowest_loss']['final_eval_loss']:.4f}\n")
        f.write(f"- **Perplexity:** {best['lowest_loss']['final_perplexity']:.2f}\n")
        f.write(f"- **Parameters:** {best['lowest_loss']['trainable_parameters']:,}\n\n")
        
        f.write("### Most Parameter Efficient\n")
        f.write(f"- **Strategy:** {best['most_efficient']['strategy']}\n")
        f.write(f"- **Efficiency:** {best['most_efficient']['parameter_efficiency']:.1%}\n")
        f.write(f"- **Loss:** {best['most_efficient']['final_eval_loss']:.4f}\n")
        f.write(f"- **Parameters:** {best['most_efficient']['trainable_parameters']:,}\n\n")
        
        # Strategy Rankings
        f.write("## Strategy Rankings\n\n")
        rankings = analysis['strategy_rankings']
        
        for criterion, ranking in rankings.items():
            f.write(f"### {criterion.replace('_', ' ').title()}\n")
            for i, (strategy, value) in enumerate(ranking.items(), 1):
                f.write(f"{i}. **{strategy.replace('_', ' ').title()}**: {value:.4f}\n")
            f.write("\n")
        
        # Detailed Results Table
        f.write("## Detailed Results\n\n")
        # Select key columns for the report
        report_cols = [
            'experiment_name', 'strategy', 'final_eval_loss', 'final_perplexity',
            'trainable_parameters', 'parameter_efficiency', 'training_time_seconds'
        ]
        detailed_df = df[report_cols].copy()
        detailed_df['parameter_efficiency'] = detailed_df['parameter_efficiency'].apply(lambda x: f"{x:.1%}")
        detailed_df['training_time_seconds'] = detailed_df['training_time_seconds'].apply(lambda x: f"{x:.1f}s")
        
        f.write(detailed_df.to_markdown(index=False))
        f.write("\n\n")
        
        # Conclusions
        f.write("## Key Findings\n\n")
        
        # Find best strategy for each metric
        best_loss_strategy = min(rankings['final_eval_loss'].items(), key=lambda x: x[1])
        best_efficiency_strategy = min(rankings['parameter_efficiency'].items(), key=lambda x: x[1])
        
        f.write(f"1. **Best Performance:** {best_loss_strategy[0].replace('_', ' ').title()} achieved the lowest evaluation loss ({best_loss_strategy[1]:.4f})\n")
        f.write(f"2. **Best Efficiency:** {best_efficiency_strategy[0].replace('_', ' ').title()} achieved the highest parameter efficiency ({best_efficiency_strategy[1]:.1%})\n")
        
        # Calculate improvement vs baseline
        baseline_strategies = [s for s in df['strategy'].unique() if 'baseline' in s]
        if baseline_strategies:
            baseline_df = df[df['strategy'].isin(baseline_strategies)]
            adaptive_df = df[~df['strategy'].isin(baseline_strategies)]
            
            if len(adaptive_df) > 0:
                avg_baseline_loss = baseline_df['final_eval_loss'].mean()
                avg_adaptive_loss = adaptive_df['final_eval_loss'].mean()
                improvement = ((avg_baseline_loss - avg_adaptive_loss) / avg_baseline_loss) * 100
                
                f.write(f"3. **Adaptive vs Baseline:** Adaptive strategies show {improvement:.1f}% {'improvement' if improvement > 0 else 'degradation'} in average loss\n")
        
        f.write("\n")
    
    logger.info(f"Summary report saved to: {report_file}")


def main():
    """Main analysis function."""
    args = parse_args()
    
    # Set up paths
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return
    
    # Load results
    results = load_experiment_results(results_dir)
    
    if not results:
        logger.error("No experiment results found!")
        return
    
    # Create DataFrame
    df = create_results_dataframe(results)
    
    # Calculate additional metrics
    df = calculate_efficiency_metrics(df)
    
    # Analyze results
    analysis = analyze_performance_efficiency(df)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save DataFrame
    df.to_csv(output_dir / "experiment_results.csv", index=False)
    logger.info(f"Results DataFrame saved to: {output_dir / 'experiment_results.csv'}")
    
    # Save analysis
    with open(output_dir / "analysis.json", 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        analysis_json = json.loads(json.dumps(analysis, default=str))
        json.dump(analysis_json, f, indent=2)
    
    # Create plots
    create_comparison_plots(df, output_dir, figures_dir, args.format)
    
    # Generate report
    generate_summary_report(df, analysis, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"📊 Analyzed {len(df)} experiments")
    print(f"📁 Output directory: {output_dir}")
    print(f"🖼️  Figures directory: {figures_dir}")
    print(f"📄 Report: {output_dir / 'analysis_report.md'}")
    print("="*60)
    
    logger.info("Analysis completed successfully!")


if __name__ == "__main__":
    main() 