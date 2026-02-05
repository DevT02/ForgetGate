#!/usr/bin/env python3
"""
Run forget-neighborhood eval across multiple noise_std values.
Copies per-run logs with a noise suffix and writes a summary.
"""

import argparse
import json
import os
import subprocess
from datetime import datetime


def run_eval(config, suite, seed, noise_std):
    cmd = [
        "python",
        "scripts/4_adv_evaluate.py",
        "--config",
        config,
        "--suite",
        suite,
        "--seed",
        str(seed),
        "--forget-neighborhood-noise-std",
        str(noise_std),
    ]
    print("[RUN]", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser(description="Forget-neighborhood sweep")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--suite", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-stds", type=str, default="0.0,0.01,0.02,0.05")
    args = parser.parse_args()

    noise_vals = [float(x.strip()) for x in args.noise_stds.split(",") if x.strip()]
    base_log = f"results/logs/{args.suite}_seed_{args.seed}_evaluation.json"

    summary = {
        "suite": args.suite,
        "seed": args.seed,
        "noise_stds": noise_vals,
        "generated_utc": datetime.utcnow().isoformat() + "Z",
        "results": {},
    }

    for noise in noise_vals:
        run_eval(args.config, args.suite, args.seed, noise)
        if not os.path.exists(base_log):
            raise FileNotFoundError(f"Expected log not found: {base_log}")
        noise_tag = str(noise).replace(".", "p")
        out_log = base_log.replace(
            f"_seed_{args.seed}_evaluation.json",
            f"_seed_{args.seed}_noise{noise_tag}_evaluation.json"
        )
        os.replace(base_log, out_log)
        with open(out_log, "r", encoding="utf-8") as f:
            data = json.load(f)
        summary["results"][str(noise)] = data

    os.makedirs("results/analysis", exist_ok=True)
    out_summary = f"results/analysis/forget_neighborhood_sweep_seed_{args.seed}.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote sweep summary to {out_summary}")


if __name__ == "__main__":
    main()
