#!/usr/bin/env python3
"""
Run VPT tradeoff sweep over lambda_retain values.
Writes a summary JSON with final resurrection_acc per lambda.
"""

import argparse
import json
import os
import subprocess
from datetime import datetime


def run_train(config, suite, seed, lam, extra_args):
    cmd = [
        "python",
        "scripts/3_train_vpt_resurrector.py",
        "--config",
        config,
        "--suite",
        suite,
        "--seed",
        str(seed),
        "--lambda-retain",
        str(lam),
    ] + extra_args
    print("[RUN]", " ".join(cmd))
    subprocess.check_call(cmd)


def load_last_metric(log_path):
    last = None
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            last = json.loads(line)
    return last or {}


def main():
    parser = argparse.ArgumentParser(description="VPT tradeoff sweep")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--suite", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambdas", type=str, default="0.0,0.1,0.5,1.0,2.0,5.0")
    parser.add_argument("--k-shot", type=int, default=None)
    parser.add_argument("--prompt-length", type=int, default=None)
    args = parser.parse_args()

    lam_vals = [float(x.strip()) for x in args.lambdas.split(",") if x.strip()]
    extra_args = []
    if args.k_shot is not None:
        extra_args += ["--k-shot", str(args.k_shot)]
    if args.prompt_length is not None:
        extra_args += ["--prompt-length", str(args.prompt_length)]

    summary = {
        "suite": args.suite,
        "seed": args.seed,
        "lambdas": lam_vals,
        "generated_utc": datetime.utcnow().isoformat() + "Z",
        "results": {},
    }

    for lam in lam_vals:
        lam_tag = str(lam).replace(".", "p")
        run_train(args.config, args.suite, args.seed, lam, extra_args)
        log_path = f"results/logs/{args.suite}_lam{lam_tag}_seed_{args.seed}.jsonl"
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"Expected log not found: {log_path}")
        last = load_last_metric(log_path)
        summary["results"][str(lam)] = last

    os.makedirs("results/analysis", exist_ok=True)
    out_path = f"results/analysis/vpt_tradeoff_sweep_seed_{args.seed}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote sweep summary to {out_path}")


if __name__ == "__main__":
    main()
