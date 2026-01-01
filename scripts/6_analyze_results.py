#!/usr/bin/env python3
"""
Script 6: Analyze unlearning results and generate paper-ready tables
"""

import json
import os
import argparse
import numpy as np
from typing import Dict, List
from pathlib import Path


def load_evaluation_results(results_dir: str, suite_name: str, seeds: List[int]) -> Dict:
    """Load evaluation results for multiple seeds"""
    all_results = {}

    for seed in seeds:
        result_file = os.path.join(results_dir, f"{suite_name}_seed_{seed}_evaluation.json")
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                all_results[seed] = json.load(f)
        else:
            print(f"Warning: Results not found for seed {seed}: {result_file}")

    return all_results


def aggregate_metrics(all_results: Dict) -> Dict:
    """Aggregate metrics across seeds (mean ± std)"""
    # Collect metrics per model
    model_metrics = {}

    for seed, results in all_results.items():
        for model_name, model_data in results.items():
            if model_name not in model_metrics:
                model_metrics[model_name] = {
                    'forget_acc': [],
                    'retain_acc': []
                }

            # Extract clean accuracy metrics
            if 'clean' in model_data:
                clean_data = model_data['clean']
                model_metrics[model_name]['forget_acc'].append(clean_data['forget_acc'])
                model_metrics[model_name]['retain_acc'].append(clean_data['retain_acc'])

    # Compute mean ± std
    aggregated = {}
    for model_name, metrics in model_metrics.items():
        aggregated[model_name] = {}
        for metric_name, values in metrics.items():
            if len(values) > 0:
                mean = np.mean(values) * 100  # Convert to percentage
                std = np.std(values) * 100
                aggregated[model_name][metric_name] = {
                    'mean': mean,
                    'std': std,
                    'values': values
                }

    return aggregated


def format_result(mean: float, std: float, bold_best: bool = False) -> str:
    """Format result as mean ± std with optional bold for best"""
    formatted = f"{mean:.2f} ± {std:.2f}"
    if bold_best:
        formatted = f"**{formatted}**"
    return formatted


def generate_comparison_table(aggregated: Dict, output_path: str = None):
    """Generate LaTeX/Markdown comparison table"""

    # Sort models: base first, then unlearned methods alphabetically
    model_order = []
    unlearn_models = []

    for model_name in aggregated.keys():
        if 'base' in model_name:
            model_order.insert(0, model_name)
        else:
            unlearn_models.append(model_name)

    model_order.extend(sorted(unlearn_models))

    # Map internal names to paper names
    name_mapping = {
        'base_vit_cifar10': 'Base Model (No Unlearning)',
        'unlearn_lora_vit_cifar10_forget0': 'CE Ascent',
        'unlearn_kl_vit_cifar10_forget0': 'Uniform KL',
        'unlearn_salun_vit_cifar10_forget0': 'SalUn (Fan et al. 2024)',
        'unlearn_scrub_distill_vit_cifar10_forget0': 'SCRUB (Kurmanji et al. 2023)'
    }

    # Find best performing method (lowest forget_acc, highest retain_acc)
    best_forget = min([(model, data['forget_acc']['mean'])
                       for model, data in aggregated.items()
                       if 'unlearn' in model],
                      key=lambda x: x[1])[0]

    best_retain = max([(model, data['retain_acc']['mean'])
                       for model, data in aggregated.items()
                       if 'unlearn' in model],
                      key=lambda x: x[1])[0]

    # Generate Markdown table
    markdown = "# Machine Unlearning Results - CIFAR-10 Class 0\n\n"
    markdown += "## Comparison of Unlearning Methods\n\n"
    markdown += "| Method | Forget Acc (%) [lower] | Retain Acc (%) [higher] | Delta Utility |\n"
    markdown += "|--------|------------------------|-------------------------|---------------|\n"

    base_retain = None
    for model in model_order:
        if model not in aggregated:
            continue

        data = aggregated[model]

        # Get display name
        display_name = name_mapping.get(model, model)

        # Format forget accuracy (lower is better)
        forget_mean = data['forget_acc']['mean']
        forget_std = data['forget_acc']['std']
        forget_str = format_result(forget_mean, forget_std, model == best_forget)

        # Format retain accuracy (higher is better)
        retain_mean = data['retain_acc']['mean']
        retain_std = data['retain_acc']['std']
        retain_str = format_result(retain_mean, retain_std, model == best_retain)

        # Compute utility delta (positive = gained utility, negative = lost utility)
        if 'base' in model:
            base_retain = retain_mean
            utility_str = "—"
        else:
            if base_retain is not None:
                utility_delta = retain_mean - base_retain
                utility_str = f"{utility_delta:+.2f}%"  # Force sign display
            else:
                utility_str = "N/A"

        markdown += f"| {display_name} | {forget_str} | {retain_str} | {utility_str} |\n"

    markdown += "\n**[lower]** Lower is better (successful forgetting)\n"
    markdown += "**[higher]** Higher is better (utility preservation)\n"
    markdown += "**Bold** indicates best performance among unlearning methods\n\n"

    # Generate LaTeX table
    latex = "\\begin{table}[t]\n"
    latex += "\\centering\n"
    latex += "\\caption{Comparison of machine unlearning methods on CIFAR-10. "
    latex += "Forget Acc measures accuracy on forgotten class (lower is better). "
    latex += "Retain Acc measures accuracy on remaining classes (higher is better).}\n"
    latex += "\\label{tab:unlearning_results}\n"
    latex += "\\begin{tabular}{lccc}\n"
    latex += "\\toprule\n"
    latex += "Method & Forget Acc (\\%) $\\downarrow$ & Retain Acc (\\%) $\\uparrow$ & $\\Delta$ Utility \\\\\n"
    latex += "\\midrule\n"

    for model in model_order:
        if model not in aggregated:
            continue

        data = aggregated[model]
        display_name = name_mapping.get(model, model)

        forget_mean = data['forget_acc']['mean']
        forget_std = data['forget_acc']['std']
        retain_mean = data['retain_acc']['mean']
        retain_std = data['retain_acc']['std']

        # LaTeX formatting
        forget_str = f"{forget_mean:.2f} $\\pm$ {forget_std:.2f}"
        retain_str = f"{retain_mean:.2f} $\\pm$ {retain_std:.2f}"

        # Bold best results
        if model == best_forget:
            forget_str = f"\\textbf{{{forget_str}}}"
        if model == best_retain:
            retain_str = f"\\textbf{{{retain_str}}}"

        # Utility delta
        if 'base' in model:
            utility_str = "—"
        else:
            if base_retain is not None:
                utility_delta = retain_mean - base_retain
                utility_str = f"{utility_delta:+.2f}\\%"  # Force sign display
            else:
                utility_str = "N/A"

        latex += f"{display_name} & {forget_str} & {retain_str} & {utility_str} \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    # Print to console
    print(markdown)
    print("\n" + "="*80 + "\n")
    print("LaTeX Table:")
    print(latex)

    # Save to file if path provided
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path.replace('.txt', '_markdown.txt'), 'w', encoding='utf-8') as f:
            f.write(markdown)

        with open(output_path.replace('.txt', '_latex.txt'), 'w', encoding='utf-8') as f:
            f.write(latex)

        print(f"\nResults saved to:")
        print(f"  - {output_path.replace('.txt', '_markdown.txt')}")
        print(f"  - {output_path.replace('.txt', '_latex.txt')}")


def print_detailed_analysis(aggregated: Dict):
    """Print detailed analysis for each method"""
    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80 + "\n")

    for model_name, data in sorted(aggregated.items()):
        print(f"\n{model_name}:")
        print("-" * 40)

        for metric_name, metric_data in data.items():
            mean = metric_data['mean']
            std = metric_data['std']
            values = metric_data['values']

            print(f"  {metric_name}:")
            print(f"    Mean: {mean:.2f}%")
            print(f"    Std:  {std:.2f}%")
            print(f"    Values: {[f'{v*100:.2f}' for v in values]}")


def main():
    parser = argparse.ArgumentParser(description="Analyze unlearning evaluation results")
    parser.add_argument("--results_dir", type=str, default="results/logs",
                       help="Directory containing evaluation results")
    parser.add_argument("--suite", type=str, default="eval_quick_baselines_vit_cifar10_forget0",
                       help="Evaluation suite name")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                       help="Seeds to aggregate")
    parser.add_argument("--output", type=str, default="results/analysis/comparison_table.txt",
                       help="Output file for tables")

    args = parser.parse_args()

    print("="*80)
    print("ForgetGate-V: Unlearning Results Analysis")
    print("="*80)
    print(f"Suite: {args.suite}")
    print(f"Seeds: {args.seeds}")
    print(f"Results directory: {args.results_dir}")
    print()

    # Load results
    all_results = load_evaluation_results(args.results_dir, args.suite, args.seeds)

    if not all_results:
        print("Error: No results found!")
        return

    print(f"Loaded results for {len(all_results)} seed(s)")

    # Aggregate metrics
    aggregated = aggregate_metrics(all_results)

    # Generate comparison table
    generate_comparison_table(aggregated, args.output)

    # Print detailed analysis
    print_detailed_analysis(aggregated)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
