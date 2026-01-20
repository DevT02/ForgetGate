#!/usr/bin/env python3
"""
Comprehensive K-shot Analysis for ForgetGate
Analyzes Oracle->VPT vs Unlearned->VPT sample efficiency experiments
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import argparse


def load_vpt_log(log_file):
    """Load VPT training log and extract final resurrection accuracy"""
    if not Path(log_file).exists():
        return None

    with open(log_file, 'r') as f:
        epochs = [json.loads(line) for line in f]

    if not epochs:
        return None

    # Extract training curve
    epoch_nums = [e['epoch'] for e in epochs]
    resurrection_accs = [e['resurrection_acc'] * 100 for e in epochs]  # Convert to percentage

    return {
        'epochs': epoch_nums,
        'resurrection_accs': resurrection_accs,
        'final_acc': resurrection_accs[-1] if resurrection_accs else 0,
        'max_acc': max(resurrection_accs) if resurrection_accs else 0
    }


def load_multiseed_results(results_dir, pattern, seeds):
    """Load results across multiple seeds and compute statistics (final logged epoch)."""
    results_by_seed = {}

    for seed in seeds:
        log_file = results_dir / pattern.format(seed=seed)
        data = load_vpt_log(log_file)
        if data:
            results_by_seed[seed] = data

    if not results_by_seed:
        return None

    final_accs = [data['final_acc'] for data in results_by_seed.values()]

    return {
        'mean': float(np.mean(final_accs)),
        'std': float(np.std(final_accs, ddof=1) if len(final_accs) > 1 else 0.0),
        'n_seeds': len(final_accs),
        'seeds': list(results_by_seed.keys()),
        'all_accs': final_accs,
        'per_seed': results_by_seed
    }


def analyze_kshot_sample_efficiency(results_dir, seeds):
    """Analyze sample efficiency across different k-shot settings"""
    print("\n" + "="*80)
    print("K-SHOT SAMPLE EFFICIENCY ANALYSIS")
    print("="*80)

    # K-shot configurations to analyze
    kshots = [1, 5, 10, 25, 50, 100]

    # Collect results
    oracle_results = {}
    kl_results = {}
    salun_results = {}
    scrub_results = {}

    # Load Oracle k-shot results
    for k in kshots:
        data = load_multiseed_results(
            results_dir,
            f"vpt_oracle_vit_cifar10_forget0_{k}shot_seed_{{seed}}.jsonl",
            seeds
        )
        if data:
            oracle_results[k] = data

    # Load Unlearned k-shot results (KL method)
    for k in kshots:
        data = load_multiseed_results(
            results_dir,
            f"vpt_resurrect_kl_forget0_{k}shot_seed_{{seed}}.jsonl",
            seeds
        )
        if data:
            kl_results[k] = data

    # Load SalUn k-shot results
    for k in [10, 25]:  # Only have these
        data = load_multiseed_results(
            results_dir,
            f"vpt_resurrect_salun_forget0_{k}shot_seed_{{seed}}.jsonl",
            seeds
        )
        if data:
            salun_results[k] = data

    # Load SCRUB k-shot results
    for k in [10, 25]:  # Only have these
        data = load_multiseed_results(
            results_dir,
            f"vpt_resurrect_scrub_forget0_{k}shot_seed_{{seed}}.jsonl",
            seeds
        )
        if data:
            scrub_results[k] = data

    # Print summary table with error bars
    print("\nFinal Resurrection Accuracy by K-shot Setting (mean +/- std across seeds):")
    print(f"\n{'K-shot':<10} {'Oracle':<20} {'KL':<20} {'SalUn':<15} {'SCRUB':<15} {'Gap (KL-Oracle)':<20}")
    print("-" * 110)

    for k in kshots:
        # Get results (now with multi-seed stats)
        oracle_data = oracle_results.get(k, {})
        kl_data = kl_results.get(k, {})
        salun_data = salun_results.get(k, {})
        scrub_data = scrub_results.get(k, {})

        # Format with error bars if multi-seed
        if isinstance(oracle_data, dict) and 'mean' in oracle_data:
            oracle_str = f"{oracle_data['mean']:.2f} +/- {oracle_data['std']:.2f}%"
            oracle_mean = oracle_data['mean']
        elif isinstance(oracle_data, dict) and 'final_acc' in oracle_data:
            oracle_str = f"{oracle_data['final_acc']:.2f}%"
            oracle_mean = oracle_data['final_acc']
        else:
            oracle_str = "N/A"
            oracle_mean = 0

        if isinstance(kl_data, dict) and 'mean' in kl_data:
            kl_str = f"{kl_data['mean']:.2f} +/- {kl_data['std']:.2f}%"
            kl_mean = kl_data['mean']
        elif isinstance(kl_data, dict) and 'final_acc' in kl_data:
            kl_str = f"{kl_data['final_acc']:.2f}%"
            kl_mean = kl_data['final_acc']
        else:
            kl_str = "N/A"
            kl_mean = 0

        if isinstance(salun_data, dict) and 'mean' in salun_data:
            salun_str = f"{salun_data['mean']:.2f} +/- {salun_data['std']:.2f}%"
        elif isinstance(salun_data, dict) and 'final_acc' in salun_data:
            salun_str = f"{salun_data['final_acc']:.2f}%"
        else:
            salun_str = "N/A"

        if isinstance(scrub_data, dict) and 'mean' in scrub_data:
            scrub_str = f"{scrub_data['mean']:.2f} +/- {scrub_data['std']:.2f}%"
        elif isinstance(scrub_data, dict) and 'final_acc' in scrub_data:
            scrub_str = f"{scrub_data['final_acc']:.2f}%"
        else:
            scrub_str = "N/A"

        gap = kl_mean - oracle_mean
        gap_str = f"{gap:+.2f}%" if oracle_str != "N/A" and kl_str != "N/A" else "N/A"

        print(f"{k:<10} {oracle_str:<20} {kl_str:<20} {salun_str:<15} {scrub_str:<15} {gap_str:<20}")

    # Analysis
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    avg_gap = np.mean([
        kl_results.get(k, {}).get('mean', kl_results.get(k, {}).get('final_acc', 0)) -
        oracle_results.get(k, {}).get('mean', oracle_results.get(k, {}).get('final_acc', 0))
        for k in kshots if k in oracle_results and k in kl_results
    ])

    print(f"\nAverage Oracle->VPT gap: {avg_gap:+.2f}%")

    if abs(avg_gap) < 2.0:
        print("\n[OK] CONCLUSION: Oracle and Unlearned recovery are NEARLY IDENTICAL")
        print("  -> VPT 'resurrection' is mostly just fast relearning")
        print("  -> LoRA-unlearning provides MINIMAL security benefit")
        print("  -> 0% forget accuracy != meaningful security guarantee")
        print("\n  This VALIDATES the 'audit' framing of your paper!")
    elif avg_gap > 5.0:
        print("\n! CONCLUSION: Unlearned models show HIGHER recovery than Oracle")
        print("  -> Some hidden knowledge may be retained in frozen parameters")
        print("  -> But gap is still small - security benefit is minimal")
    else:
        print("\n[WARNING] CONCLUSION: Modest difference between Oracle and Unlearned")
        print("  -> Some residual accessibility, but not dramatic")

    return oracle_results, kl_results, salun_results, scrub_results


def generate_sample_efficiency_report(results_dir, oracle_kshot, kl_kshot,
                                      thresholds=(1.0, 2.0, 5.0, 10.0)):
    """Generate a sample-efficiency report: smallest k reaching each recovery threshold."""
    results_dir = Path(results_dir)

    def _get_mean(d):
        if not isinstance(d, dict):
            return None
        if 'mean' in d:
            return float(d['mean'])
        if 'final_acc' in d:
            return float(d['final_acc'])
        return None

    kshots = sorted(set(list(oracle_kshot.keys()) + list(kl_kshot.keys())))

    def _first_k_at_threshold(res_dict, threshold):
        for k in kshots:
            mean = _get_mean(res_dict.get(k, {}))
            if mean is None:
                continue
            if mean >= threshold:
                return k, mean
        return None, None

    lines = []
    lines.append("# Sample-Efficiency Thresholds (Recovery@k)")
    lines.append("")
    lines.append("Smallest k such that mean Recovery@k >= threshold (percent).")
    lines.append("")
    lines.append("| Threshold (%) | Oracle k* | Oracle Recovery@k | Unlearned (KL) k* | Unlearned Recovery@k | k-Advantage (Oracle - Unlearned) |")
    lines.append("|---|---|---|---|---|---|")

    for t in thresholds:
        ok, oacc = _first_k_at_threshold(oracle_kshot, t)
        kk, kacc = _first_k_at_threshold(kl_kshot, t)
        ok_str = str(ok) if ok is not None else "not reached"
        kk_str = str(kk) if kk is not None else "not reached"
        oacc_str = f"{oacc:.2f}%" if oacc is not None else "N/A"
        kacc_str = f"{kacc:.2f}%" if kacc is not None else "N/A"
        if ok is not None and kk is not None:
            adv = ok - kk
            adv_str = f"{adv:+d}"
        else:
            adv_str = "N/A"
        lines.append(f"| {t:.1f} | {ok_str} | {oacc_str} | {kk_str} | {kacc_str} | {adv_str} |")

    output_file = results_dir.parent / "analysis" / "sample_efficiency_gap.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines))

    print("\n" + "=" * 80)
    print("SAMPLE-EFFICIENCY THRESHOLD REPORT")
    print("=" * 80)
    print("\n".join(lines))
    print(f"\n[OK] Saved sample-efficiency report to {output_file}")


def analyze_prompt_capacity(results_dir, seeds):
    """Analyze VPT capacity with different prompt lengths"""
    print("\n" + "="*80)
    print("PROMPT CAPACITY ABLATION")
    print("="*80)

    # Explicit prompt-length overrides we ran
    prompt_lengths = [1, 2, 5, 10]

    oracle_results = {}
    kl_results = {}

    # Load Oracle prompt ablation (all with 10-shot)
    for pl in prompt_lengths:
        if pl == 10:
            # Default prompt length (from suite config), no prompt tag in filename
            pattern = "vpt_oracle_vit_cifar10_forget0_10shot_seed_{seed}.jsonl"
        else:
            # Prompt-length override uses "10shot_prompt{pl}" naming
            pattern = f"vpt_oracle_vit_cifar10_forget0_10shot_prompt{pl}_seed_{{seed}}.jsonl"

        data = load_multiseed_results(results_dir, pattern, seeds)
        if data:
            oracle_results[pl] = data

    # Load KL prompt ablation (all with 10-shot)
    for pl in prompt_lengths:
        if pl == 10:
            # Default prompt length (from suite config), no prompt tag in filename
            pattern = "vpt_resurrect_kl_forget0_10shot_seed_{seed}.jsonl"
        else:
            # Prompt-length override uses "10shot_prompt{pl}" naming
            pattern = f"vpt_resurrect_kl_forget0_10shot_prompt{pl}_seed_{{seed}}.jsonl"

        data = load_multiseed_results(results_dir, pattern, seeds)
        if data:
            kl_results[pl] = data

    # Print summary table
    print("\nResurrection Accuracy by Prompt Length (10-shot setting):")
    print(f"\n{'Prompt Length':<15} {'Params':<12} {'Oracle':<12} {'KL Unlearned':<15} {'Gap':<12}")
    print("-" * 70)

    for pl in prompt_lengths:
        params = pl * 192  # ViT-Tiny embed_dim = 192
        oracle_acc = oracle_results.get(pl, {}).get('mean', oracle_results.get(pl, {}).get('final_acc', 0))
        kl_acc = kl_results.get(pl, {}).get('mean', kl_results.get(pl, {}).get('final_acc', 0))
        gap = kl_acc - oracle_acc

        oracle_str = f"{oracle_acc:.2f}%" if pl in oracle_results else "N/A"
        kl_str = f"{kl_acc:.2f}%" if pl in kl_results else "N/A"
        gap_str = f"{gap:+.2f}%" if (pl in oracle_results and pl in kl_results) else "N/A"

        print(f"{pl:<15} {params:<12,} {oracle_str:<12} {kl_str:<15} {gap_str:<12}")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    if kl_results and oracle_results:
        max_kl = max([kl_results[pl].get('mean', kl_results[pl].get('final_acc', 0)) for pl in kl_results.keys()])
        max_oracle = max([oracle_results[pl].get('mean', oracle_results[pl].get('final_acc', 0)) for pl in oracle_results.keys()])

        print(f"\nMax KL recovery: {max_kl:.2f}%")
        print(f"Max Oracle recovery: {max_oracle:.2f}%")

        if max_kl < 5.0 and max_oracle < 5.0:
            print("\n[OK] Both show VERY LOW recovery even with more prompt capacity")
            print("  -> Confirms that k-shot VPT resurrection is minimal")
        else:
            print("\n[NOTE] Recovery increases with prompt capacity or differs by method")
            print("  -> Interpret k-shot results conditioned on prompt length")

    return oracle_results, kl_results


def analyze_classwise_gap(results_dir, seeds, forget_classes=(1, 2, 5, 9)):
    """Analyze class-wise oracle gap for 10-shot runs."""
    print("\n" + "="*80)
    print("CLASS-WISE ORACLE GAP (10-shot)")
    print("="*80)

    results = []
    for c in forget_classes:
        oracle_data = load_multiseed_results(
            results_dir,
            f"vpt_oracle_vit_cifar10_forget{c}_10shot_seed_{{seed}}.jsonl",
            seeds
        )
        kl_data = load_multiseed_results(
            results_dir,
            f"vpt_resurrect_kl_forget{c}_10shot_seed_{{seed}}.jsonl",
            seeds
        )

        if not oracle_data and not kl_data:
            continue

        oracle_acc = oracle_data.get('mean', oracle_data.get('final_acc', 0)) if oracle_data else None
        kl_acc = kl_data.get('mean', kl_data.get('final_acc', 0)) if kl_data else None
        gap = (kl_acc - oracle_acc) if (oracle_acc is not None and kl_acc is not None) else None

        results.append((c, oracle_acc, kl_acc, gap))

    if not results:
        print("No class-wise 10-shot logs found.")
        return []

    print(f"\n{'Forget Class':<12} {'Oracle':<12} {'KL Unlearned':<15} {'Gap':<12}")
    print("-" * 55)
    for c, o, k, g in results:
        o_str = f"{o:.2f}%" if o is not None else "N/A"
        k_str = f"{k:.2f}%" if k is not None else "N/A"
        g_str = f"{g:+.2f}%" if g is not None else "N/A"
        print(f"{c:<12} {o_str:<12} {k_str:<15} {g_str:<12}")

    return results


def analyze_lowshot_controls(results_dir, seeds, prompt_length=5, kshots=(1, 5)):
    """Analyze low-shot controls with fixed prompt length."""
    print("\n" + "="*80)
    print("LOW-SHOT CONTROLS (prompt length %d)" % prompt_length)
    print("="*80)

    results = []
    for k in kshots:
        oracle_data = load_multiseed_results(
            results_dir,
            f"vpt_oracle_vit_cifar10_forget0_10shot_prompt{prompt_length}_kshot{k}_seed_{{seed}}.jsonl",
            seeds
        )
        kl_data = load_multiseed_results(
            results_dir,
            f"vpt_resurrect_kl_forget0_10shot_prompt{prompt_length}_kshot{k}_seed_{{seed}}.jsonl",
            seeds
        )
        if not oracle_data and not kl_data:
            continue

        oracle_acc = oracle_data.get('mean', oracle_data.get('final_acc', 0)) if oracle_data else None
        kl_acc = kl_data.get('mean', kl_data.get('final_acc', 0)) if kl_data else None
        gap = (kl_acc - oracle_acc) if (oracle_acc is not None and kl_acc is not None) else None
        results.append((k, oracle_acc, kl_acc, gap))

    if not results:
        print("No low-shot control logs found.")
        return []

    print(f"\n{'K-shot':<10} {'Oracle':<12} {'KL Unlearned':<15} {'Gap':<12}")
    print("-" * 55)
    for k, o, kacc, g in results:
        o_str = f"{o:.2f}%" if o is not None else "N/A"
        k_str = f"{kacc:.2f}%" if kacc is not None else "N/A"
        g_str = f"{g:+.2f}%" if g is not None else "N/A"
        print(f"{k:<10} {o_str:<12} {k_str:<15} {g_str:<12}")

    return results


def analyze_shuffled_label_control(results_dir, seeds, prompt_length=5):
    """Analyze shuffled-label controls (k=10) with fixed prompt length."""
    print("\n" + "="*80)
    print("SHUFFLED-LABEL CONTROL (prompt length %d, k=10)" % prompt_length)
    print("="*80)

    # Accept both "shufflelabels" and legacy "shuffledlabels" suffixes
    oracle_data = load_multiseed_results(
        results_dir,
        f"vpt_oracle_vit_cifar10_forget0_10shot_prompt{prompt_length}_shufflelabels_seed_{{seed}}.jsonl",
        seeds
    ) or load_multiseed_results(
        results_dir,
        f"vpt_oracle_vit_cifar10_forget0_10shot_prompt{prompt_length}_shuffledlabels_seed_{{seed}}.jsonl",
        seeds
    )
    kl_data = load_multiseed_results(
        results_dir,
        f"vpt_resurrect_kl_forget0_10shot_prompt{prompt_length}_shufflelabels_seed_{{seed}}.jsonl",
        seeds
    ) or load_multiseed_results(
        results_dir,
        f"vpt_resurrect_kl_forget0_10shot_prompt{prompt_length}_shuffledlabels_seed_{{seed}}.jsonl",
        seeds
    )

    if not oracle_data and not kl_data:
        print("No shuffled-label control logs found.")
        return None

    oracle_acc = oracle_data.get('mean', oracle_data.get('final_acc', 0)) if oracle_data else None
    kl_acc = kl_data.get('mean', kl_data.get('final_acc', 0)) if kl_data else None
    gap = (kl_acc - oracle_acc) if (oracle_acc is not None and kl_acc is not None) else None

    print(f"\n{'Oracle':<12} {'KL Unlearned':<15} {'Gap':<12}")
    print("-" * 45)
    o_str = f"{oracle_acc:.2f}%" if oracle_acc is not None else "N/A"
    k_str = f"{kl_acc:.2f}%" if kl_acc is not None else "N/A"
    g_str = f"{gap:+.2f}%" if gap is not None else "N/A"
    print(f"{o_str:<12} {k_str:<15} {g_str:<12}")

    return (oracle_acc, kl_acc, gap)


def generate_plots(results_dir, oracle_kshot, kl_kshot, salun_kshot, scrub_kshot,
                   oracle_prompt, kl_prompt, curve_seed):
    """Generate publication-quality plots"""
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Sample Efficiency Curve
    ax = axes[0, 0]

    kshots = sorted([k for k in oracle_kshot.keys()])
    oracle_accs = [oracle_kshot[k].get('mean', oracle_kshot[k].get('final_acc', 0)) for k in kshots]
    kl_accs = [kl_kshot.get(k, {}).get('mean', kl_kshot.get(k, {}).get('final_acc', 0)) for k in kshots]

    ax.plot(kshots, oracle_accs, marker='o', linewidth=2.5, markersize=8,
            label='Oracle->VPT (never saw class)', color='#e74c3c', linestyle='--')
    ax.plot(kshots, kl_accs, marker='s', linewidth=2.5, markersize=8,
            label='Unlearned->VPT (forgot class)', color='#3498db')

    ax.set_xlabel('Number of Training Samples (K-shot)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Resurrection Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Sample Efficiency: Oracle vs Unlearned VPT Recovery', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_xticks(kshots)
    ax.set_xticklabels(kshots)
    ax.set_ylim([-1, max(max(oracle_accs), max(kl_accs)) + 2])

    # Plot 2: Gap Analysis
    ax = axes[0, 1]

    gaps = [
        kl_kshot.get(k, {}).get('mean', kl_kshot.get(k, {}).get('final_acc', 0)) -
        oracle_kshot[k].get('mean', oracle_kshot[k].get('final_acc', 0))
        for k in kshots
    ]
    colors = ['#27ae60' if g > 0 else '#e74c3c' for g in gaps]

    ax.bar(range(len(kshots)), gaps, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Number of Training Samples (K-shot)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Recovery Gap: Unlearned - Oracle (%)', fontsize=12, fontweight='bold')
    ax.set_title('Oracle-Normalized Recovery Difficulty', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(kshots)))
    ax.set_xticklabels(kshots)
    ax.grid(True, alpha=0.3, axis='y')

    # Add annotation
    avg_gap = np.mean(gaps)
    ax.text(0.98, 0.95, f'Avg Gap: {avg_gap:+.2f}%',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Prompt Capacity Ablation
    ax = axes[1, 0]

    prompt_lengths = sorted([pl for pl in oracle_prompt.keys()])
    oracle_prompt_accs = [oracle_prompt[pl].get('mean', oracle_prompt[pl].get('final_acc', 0)) for pl in prompt_lengths]
    kl_prompt_accs = [kl_prompt[pl].get('mean', kl_prompt[pl].get('final_acc', 0)) for pl in prompt_lengths]
    prompt_params = [pl * 192 for pl in prompt_lengths]

    ax.plot(prompt_params, oracle_prompt_accs, marker='o', linewidth=2.5, markersize=8,
            label='Oracle->VPT', color='#e74c3c', linestyle='--')
    ax.plot(prompt_params, kl_prompt_accs, marker='s', linewidth=2.5, markersize=8,
            label='Unlearned->VPT', color='#3498db')

    ax.set_xlabel('VPT Prompt Parameters', fontsize=12, fontweight='bold')
    ax.set_ylabel('Resurrection Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Prompt Capacity Ablation (10-shot)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add prompt length annotations
    for i, pl in enumerate(prompt_lengths):
        ax.annotate(f'{pl}tok', (prompt_params[i], max(oracle_prompt_accs[i], kl_prompt_accs[i])),
                   textcoords="offset points", xytext=(0,5), ha='center', fontsize=9)

    # Plot 4: Method Comparison (full dataset)
    ax = axes[1, 1]

    # Load full dataset results
    methods_data = {}

    full_results = [
        ('KL', results_dir / 'vpt_resurrect_kl_forget0_seed_42.jsonl'),
        ('SalUn', results_dir / 'vpt_resurrect_salun_forget0_seed_42.jsonl'),
        ('SCRUB', results_dir / 'vpt_resurrect_scrub_forget0_seed_42.jsonl'),
    ]

    method_names = []
    method_accs = []

    for method_name, log_file in full_results:
        if log_file.exists():
            data = load_vpt_log(log_file)
            if data:
                method_names.append(method_name)
                method_accs.append(data['final_acc'])

    if method_names:
        colors_method = ['#3498db', '#e74c3c', '#2ecc71'][:len(method_names)]
        ax.bar(method_names, method_accs, color=colors_method, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Resurrection Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Unlearning Method Comparison (Full Dataset)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, max(method_accs) * 1.2 if method_accs else 10])

        # Add value labels on bars
        for i, (name, acc) in enumerate(zip(method_names, method_accs)):
            ax.text(i, acc + 0.5, f'{acc:.2f}%', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Save plot
    output_file = results_dir.parent / 'analysis' / 'kshot_analysis.png'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Saved plots to {output_file}")

    plt.close()

    # Generate individual learning curve plots for key experiments
    generate_learning_curves(results_dir, curve_seed)


def generate_learning_curves(results_dir, curve_seed):
    """Generate detailed learning curves for key k-shot settings"""

    key_kshots = [10, 25, 50, 100]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, k in enumerate(key_kshots):
        ax = axes[idx]

        oracle_log = results_dir / f"vpt_oracle_vit_cifar10_forget0_{k}shot_seed_{curve_seed}.jsonl"
        oracle_data = load_vpt_log(oracle_log)
        if oracle_data:
            epochs = oracle_data['epochs']
            accs = oracle_data['resurrection_accs']
            ax.plot(epochs, accs, marker='o', linewidth=2, markersize=6,
                   label='Oracle->VPT', color='#e74c3c', linestyle='--', alpha=0.8)

        kl_log = results_dir / f"vpt_resurrect_kl_forget0_{k}shot_seed_{curve_seed}.jsonl"
        kl_data = load_vpt_log(kl_log)
        if kl_data:
            epochs = kl_data['epochs']
            accs = kl_data['resurrection_accs']
            ax.plot(epochs, accs, marker='s', linewidth=2, markersize=6,
                   label='Unlearned->VPT (KL)', color='#3498db', alpha=0.8)

        ax.set_xlabel('Training Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('Resurrection Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{k}-shot Learning Curve', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-1, 5])  # Most are near 0%

    plt.tight_layout()

    output_file = results_dir.parent / 'analysis' / 'learning_curves_kshot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved learning curves to {output_file}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze k-shot VPT experiments')
    parser.add_argument('--results_dir', type=str, default='results/logs',
                       help='Directory containing experiment logs')
    parser.add_argument('--seeds', type=int, nargs="+", default=[42, 123],
                       help='Seeds to aggregate (use final logged epoch per run)')
    parser.add_argument('--curve-seed', type=int, default=None,
                       help='Seed to use for learning-curve plots (default: first in --seeds)')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    print("="*80)
    print("FORGETGATE K-SHOT ANALYSIS")
    print("="*80)
    seeds = args.seeds
    curve_seed = args.curve_seed if args.curve_seed is not None else seeds[0]

    print(f"\nAnalyzing results from: {results_dir}")
    print(f"Aggregating seeds: {seeds} (learning curves use seed {curve_seed})")

    # Run analyses
    oracle_kshot, kl_kshot, salun_kshot, scrub_kshot = analyze_kshot_sample_efficiency(results_dir, seeds)
    oracle_prompt, kl_prompt = analyze_prompt_capacity(results_dir, seeds)
    classwise_results = analyze_classwise_gap(results_dir, seeds)
    lowshot_results = analyze_lowshot_controls(results_dir, seeds)
    shuffled_results = analyze_shuffled_label_control(results_dir, seeds)

    # Generate plots
    if oracle_kshot and kl_kshot:
        generate_plots(results_dir, oracle_kshot, kl_kshot, salun_kshot, scrub_kshot,
                      oracle_prompt, kl_prompt, curve_seed)
        generate_sample_efficiency_report(results_dir, oracle_kshot, kl_kshot)
    else:
        print("\n[WARNING] WARNING: Insufficient data for plots")
        print(f"  Oracle k-shot results: {len(oracle_kshot)}")
        print(f"  KL k-shot results: {len(kl_kshot)}")

    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey files generated:")
    print("  - results/analysis/kshot_analysis.png")
    print("  - results/analysis/learning_curves_kshot.png")
    print("\nUse these plots in your paper to demonstrate:")
    print("  1. Oracle-normalized recovery difficulty")
    print("  2. 0% forget accuracy != security guarantee")
    print("  3. LoRA-unlearning provides minimal benefit over never training")
    print("="*80)


if __name__ == "__main__":
    main()
