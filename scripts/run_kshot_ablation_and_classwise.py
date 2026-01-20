#!/usr/bin/env python3
"""
Run prompt-length ablation + class-wise gap (resume-safe).
"""

import argparse
import os
import subprocess
import sys


def run(cmd):
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="K-shot ablation + class-wise gap (resume-safe)")
    parser.add_argument("--config", default="configs/experiment_suites.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = args.config
    seed = args.seed

    print("=" * 60)
    print(f"ForgetGate: K-shot prompt ablation + class-wise gap (seed {seed})")
    print("=" * 60)

    # Prompt ablation (oracle + KL) for 10-shot
    print("[Prompt Ablation] Oracle 10-shot (prompt length 1/2/5)")
    for pl in (1, 2, 5):
        log_path = f"results/logs/vpt_oracle_vit_cifar10_forget0_10shot_prompt{pl}_seed_{seed}.jsonl"
        if os.path.exists(log_path):
            print(f"[SKIP] Oracle prompt{pl}")
        else:
            run([
                "python", "scripts/3_train_vpt_resurrector.py",
                "--config", config,
                "--suite", "vpt_oracle_vit_cifar10_forget0_10shot",
                "--seed", str(seed),
                "--prompt-length", str(pl),
            ])

    print("[Prompt Ablation] Unlearned KL 10-shot (prompt length 1/2/5)")
    for pl in (1, 2, 5):
        log_path = f"results/logs/vpt_resurrect_kl_forget0_10shot_prompt{pl}_seed_{seed}.jsonl"
        if os.path.exists(log_path):
            print(f"[SKIP] KL prompt{pl}")
        else:
            run([
                "python", "scripts/3_train_vpt_resurrector.py",
                "--config", config,
                "--suite", "vpt_resurrect_kl_forget0_10shot",
                "--seed", str(seed),
                "--prompt-length", str(pl),
            ])

    # Class-wise oracles
    print("[Class-wise] Train oracles (forget classes 1/2/5/9)")
    for c in (1, 2, 5, 9):
        ckpt = f"checkpoints/oracle/oracle_vit_cifar10_forget{c}_seed_{seed}_final.pt"
        if os.path.exists(ckpt):
            print(f"[SKIP] oracle forget{c}")
        else:
            run([
                "python", "scripts/1b_train_retrained_oracle.py",
                "--config", config,
                "--suite", f"oracle_vit_cifar10_forget{c}",
                "--seed", str(seed),
            ])

    # Class-wise unlearned KL
    print("[Class-wise] Train unlearned KL (forget classes 1/2/5/9)")
    for c in (1, 2, 5, 9):
        log_path = f"results/logs/unlearn_kl_vit_cifar10_forget{c}_seed_{seed}_history.json"
        if os.path.exists(log_path):
            print(f"[SKIP] unlearn KL forget{c}")
        else:
            run([
                "python", "scripts/2_train_unlearning_lora.py",
                "--config", config,
                "--suite", f"unlearn_kl_vit_cifar10_forget{c}",
                "--seed", str(seed),
            ])

    # Class-wise VPT oracle
    print("[Class-wise] VPT oracle (10-shot)")
    for c in (1, 2, 5, 9):
        log_path = f"results/logs/vpt_oracle_vit_cifar10_forget{c}_10shot_seed_{seed}.jsonl"
        if os.path.exists(log_path):
            print(f"[SKIP] vpt oracle forget{c}")
        else:
            run([
                "python", "scripts/3_train_vpt_resurrector.py",
                "--config", config,
                "--suite", f"vpt_oracle_vit_cifar10_forget{c}_10shot",
                "--seed", str(seed),
            ])

    # Class-wise VPT unlearned KL
    print("[Class-wise] VPT unlearned KL (10-shot)")
    for c in (1, 2, 5, 9):
        log_path = f"results/logs/vpt_resurrect_kl_forget{c}_10shot_seed_{seed}.jsonl"
        if os.path.exists(log_path):
            print(f"[SKIP] vpt kl forget{c}")
        else:
            run([
                "python", "scripts/3_train_vpt_resurrector.py",
                "--config", config,
                "--suite", f"vpt_resurrect_kl_forget{c}_10shot",
                "--seed", str(seed),
            ])

    print("[Analysis] k-shot summary")
    run(["python", "scripts/analyze_kshot_experiments.py", "--seeds", str(seed)])

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
