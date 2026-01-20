#!/usr/bin/env python3
"""
Run low-shot + shuffled-label controls (resume-safe).
"""

import argparse
import os
import subprocess
import sys


def run(cmd):
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Controls (low-shot + shuffled labels)")
    parser.add_argument("--config", default="configs/experiment_suites.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-length", type=int, default=5)
    args = parser.parse_args()

    config = args.config
    seed = args.seed
    pl = args.prompt_length

    print("=" * 60)
    print(f"ForgetGate: Controls (low-shot + shuffled labels) seed {seed}")
    print("=" * 60)

    # Low-shot controls
    print("[Low-shot] Oracle + KL (prompt length %d, k=1/5)" % pl)
    for k in (1, 5):
        log_path = f"results/logs/vpt_oracle_vit_cifar10_forget0_10shot_prompt{pl}_kshot{k}_seed_{seed}.jsonl"
        if os.path.exists(log_path):
            print(f"[SKIP] oracle k={k}")
        else:
            run([
                "python", "scripts/3_train_vpt_resurrector.py",
                "--config", config,
                "--suite", "vpt_oracle_vit_cifar10_forget0_10shot",
                "--seed", str(seed),
                "--prompt-length", str(pl),
                "--k-shot", str(k),
            ])

    for k in (1, 5):
        log_path = f"results/logs/vpt_resurrect_kl_forget0_10shot_prompt{pl}_kshot{k}_seed_{seed}.jsonl"
        if os.path.exists(log_path):
            print(f"[SKIP] kl k={k}")
        else:
            run([
                "python", "scripts/3_train_vpt_resurrector.py",
                "--config", config,
                "--suite", "vpt_resurrect_kl_forget0_10shot",
                "--seed", str(seed),
                "--prompt-length", str(pl),
                "--k-shot", str(k),
            ])

    # Shuffled-label control
    print("[Shuffled labels] Oracle + KL (prompt length %d, k=10)" % pl)
    oracle_log = f"results/logs/vpt_oracle_vit_cifar10_forget0_10shot_prompt{pl}_shufflelabels_seed_{seed}.jsonl"
    if os.path.exists(oracle_log):
        print("[SKIP] oracle shuffled labels")
    else:
        run([
            "python", "scripts/3_train_vpt_resurrector.py",
            "--config", config,
            "--suite", "vpt_oracle_vit_cifar10_forget0_10shot",
            "--seed", str(seed),
            "--prompt-length", str(pl),
            "--label-mode", "shuffle",
        ])

    kl_log = f"results/logs/vpt_resurrect_kl_forget0_10shot_prompt{pl}_shufflelabels_seed_{seed}.jsonl"
    if os.path.exists(kl_log):
        print("[SKIP] kl shuffled labels")
    else:
        run([
            "python", "scripts/3_train_vpt_resurrector.py",
            "--config", config,
            "--suite", "vpt_resurrect_kl_forget0_10shot",
            "--seed", str(seed),
            "--prompt-length", str(pl),
            "--label-mode", "shuffle",
        ])

    print("=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print("=" * 60)
        print(f"ERROR: Pipeline failed with exit code {exc.returncode}.")
        print("=" * 60)
        sys.exit(exc.returncode)
