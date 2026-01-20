#!/usr/bin/env python3
"""
Run the full paper pipeline (no skipping).
"""

import argparse
import subprocess
import sys


def run(cmd):
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="ForgetGate paper pipeline (no skipping)")
    parser.add_argument("--config", default="configs/experiment_suites.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123])
    parser.add_argument("--kshots", nargs="+", type=int, default=[10, 25, 50, 100])
    parser.add_argument("--run-fulldata-vpt", action="store_true", default=True)
    parser.add_argument("--run-salun", action="store_true", default=True)
    parser.add_argument("--run-scrub", action="store_true", default=True)
    args = parser.parse_args()

    config = args.config
    seeds = args.seeds
    kshots = args.kshots

    base_suite = "base_vit_cifar10"
    oracle_suite = "oracle_vit_cifar10_forget0"

    unlearn_ce = "unlearn_lora_vit_cifar10_forget0"
    unlearn_kl = "unlearn_kl_vit_cifar10_forget0"
    unlearn_salun = "unlearn_salun_vit_cifar10_forget0"
    unlearn_scrub = "unlearn_scrub_distill_vit_cifar10_forget0"

    vpt_ce = "vpt_resurrect_vit_cifar10_forget0"
    vpt_kl = "vpt_resurrect_kl_forget0"
    vpt_salun = "vpt_resurrect_salun_forget0"
    vpt_scrub = "vpt_resurrect_scrub_forget0"

    eval_baselines = "eval_paper_baselines_vit_cifar10_forget0"

    print("=" * 60)
    print("ForgetGate paper pipeline")
    print(f"Seeds: {seeds}")
    print(f"K-shots: {kshots}")
    print("=" * 60)

    # 1) Base + Oracle + Unlearning
    for seed in seeds:
        run(["python", "scripts/1_train_base.py", "--config", config, "--suite", base_suite, "--seed", str(seed)])
        run(["python", "scripts/1b_train_retrained_oracle.py", "--config", config, "--suite", oracle_suite, "--seed", str(seed)])
        run(["python", "scripts/2_train_unlearning_lora.py", "--config", config, "--suite", unlearn_ce, "--seed", str(seed)])
        run(["python", "scripts/2_train_unlearning_lora.py", "--config", config, "--suite", unlearn_kl, "--seed", str(seed)])
        if args.run_salun:
            run(["python", "scripts/2_train_unlearning_lora.py", "--config", config, "--suite", unlearn_salun, "--seed", str(seed)])
        if args.run_scrub:
            run(["python", "scripts/2_train_unlearning_lora.py", "--config", config, "--suite", unlearn_scrub, "--seed", str(seed)])

    # 2) VPT k-shot (oracle + KL)
    for seed in seeds:
        for k in kshots:
            run(["python", "scripts/3_train_vpt_resurrector.py", "--config", config,
                 "--suite", f"vpt_oracle_vit_cifar10_forget0_{k}shot", "--seed", str(seed)])
            run(["python", "scripts/3_train_vpt_resurrector.py", "--config", config,
                 "--suite", f"vpt_resurrect_kl_forget0_{k}shot", "--seed", str(seed)])

    # 3) VPT full-data recovery
    if args.run_fulldata_vpt:
        for seed in seeds:
            run(["python", "scripts/3_train_vpt_resurrector.py", "--config", config, "--suite", vpt_ce, "--seed", str(seed)])
            run(["python", "scripts/3_train_vpt_resurrector.py", "--config", config, "--suite", vpt_kl, "--seed", str(seed)])
            if args.run_salun:
                run(["python", "scripts/3_train_vpt_resurrector.py", "--config", config, "--suite", vpt_salun, "--seed", str(seed)])
            if args.run_scrub:
                run(["python", "scripts/3_train_vpt_resurrector.py", "--config", config, "--suite", vpt_scrub, "--seed", str(seed)])

    # 4) Evaluation
    for seed in seeds:
        run(["python", "scripts/4_adv_evaluate.py", "--config", config, "--suite", eval_baselines, "--seed", str(seed)])

    # 5) Analysis
    run(["python", "scripts/analyze_kshot_experiments.py", "--seeds"] + [str(s) for s in seeds])
    run(["python", "scripts/6_analyze_results.py", "--suite", eval_baselines, "--seeds"] + [str(s) for s in seeds])

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
