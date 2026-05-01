#!/usr/bin/env python3
"""
Run high-value additions:
  - Prompt-length ablation for forget0 (seed 123)
  - Random-label control for forget0 (seed 123)
  - Prompt-length ablation for forget1 (seed 42)
All steps are resume-safe.
"""

import argparse
import os
import subprocess
import sys


def run(cmd):
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_prompt_ablation(config, suite, seed, prompt_lengths):
    for pl in prompt_lengths:
        log_path = f"results/logs/{suite}_prompt{pl}_seed_{seed}.jsonl"
        if os.path.exists(log_path):
            print(f"[SKIP] {suite} prompt{pl} seed {seed}")
        else:
            run([
                "python", "scripts/3_train_vpt_resurrector.py",
                "--config", config,
                "--suite", suite,
                "--seed", str(seed),
                "--prompt-length", str(pl),
            ])


def main():
    parser = argparse.ArgumentParser(description="Run high-value additions (resume-safe)")
    parser.add_argument("--config", default="configs/experiment_suites.yaml")
    parser.add_argument("--seed-main", type=int, default=123)
    parser.add_argument("--seed-classwise", type=int, default=42)
    parser.add_argument("--prompt-lengths", nargs="+", type=int, default=[1, 2, 5])
    args = parser.parse_args()

    config = args.config
    seed_main = args.seed_main
    seed_classwise = args.seed_classwise
    prompt_lengths = args.prompt_lengths
    print("Run: High-value additions (resume-safe)")
    # Prompt-length ablation for forget0 (seed 123)
    print(f"[Prompt Ablation] forget0 seed {seed_main}")
    run_prompt_ablation(config, "vpt_oracle_vit_cifar10_forget0_10shot", seed_main, prompt_lengths)
    run_prompt_ablation(config, "vpt_resurrect_kl_forget0_10shot", seed_main, prompt_lengths)

    # Random-label control (forget0, seed 123, prompt length 5, k=10)
    print(f"[Random-label] forget0 seed {seed_main}")
    oracle_log = f"results/logs/vpt_oracle_vit_cifar10_forget0_10shot_prompt5_randomlabels_seed_{seed_main}.jsonl"
    if os.path.exists(oracle_log):
        print("[SKIP] oracle random labels")
    else:
        run([
            "python", "scripts/3_train_vpt_resurrector.py",
            "--config", config,
            "--suite", "vpt_oracle_vit_cifar10_forget0_10shot",
            "--seed", str(seed_main),
            "--prompt-length", "5",
            "--label-mode", "random",
        ])

    kl_log = f"results/logs/vpt_resurrect_kl_forget0_10shot_prompt5_randomlabels_seed_{seed_main}.jsonl"
    if os.path.exists(kl_log):
        print("[SKIP] kl random labels")
    else:
        run([
            "python", "scripts/3_train_vpt_resurrector.py",
            "--config", config,
            "--suite", "vpt_resurrect_kl_forget0_10shot",
            "--seed", str(seed_main),
            "--prompt-length", "5",
            "--label-mode", "random",
        ])

    # Prompt-length ablation for another class (forget1, seed 42)
    print(f"[Prompt Ablation] forget1 seed {seed_classwise}")
    run_prompt_ablation(config, "vpt_oracle_vit_cifar10_forget1_10shot", seed_classwise, prompt_lengths)
    run_prompt_ablation(config, "vpt_resurrect_kl_forget1_10shot", seed_classwise, prompt_lengths)
    print("Done.")
if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: Pipeline failed with exit code {exc.returncode}.")
        sys.exit(exc.returncode)
