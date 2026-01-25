#!/usr/bin/env python3
"""
Run missing k-shot suites to remove N/A entries.
Resume-safe: skips logs that already exist.
"""

import argparse
import os
import subprocess
import sys


def run(cmd):
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run missing k-shot suites (resume-safe)")
    parser.add_argument("--config", default="configs/experiment_suites.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = args.config
    seed = args.seed

    print("=" * 70)
    print(f"ForgetGate: Fill missing k-shot suites (seed {seed})")
    print("=" * 70)

    suites = [
        # Oracle + KL (k=1,5)
        "vpt_oracle_vit_cifar10_forget0_1shot",
        "vpt_oracle_vit_cifar10_forget0_5shot",
        "vpt_resurrect_kl_forget0_1shot",
        "vpt_resurrect_kl_forget0_5shot",

        # SalUn/SCRUB (k=50,100)
        "vpt_resurrect_salun_forget0_50shot",
        "vpt_resurrect_salun_forget0_100shot",
        "vpt_resurrect_scrub_forget0_50shot",
        "vpt_resurrect_scrub_forget0_100shot",
    ]

    for suite in suites:
        log_path = f"results/logs/{suite}_seed_{seed}.jsonl"
        if os.path.exists(log_path):
            print(f"[SKIP] {suite}")
        else:
            run([
                "python", "scripts/3_train_vpt_resurrector.py",
                "--config", config,
                "--suite", suite,
                "--seed", str(seed),
            ])

    print("=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print("=" * 70)
        print(f"ERROR: Pipeline failed with exit code {exc.returncode}.")
        print("=" * 70)
        sys.exit(exc.returncode)
