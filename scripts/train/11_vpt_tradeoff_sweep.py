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
    parser.add_argument(
        "--feature-guide",
        type=str,
        default=None,
        choices=["linear_probe", "centroid_margin", "prototype_margin", "oracle_contrastive_probe"],
        help="Optional feature-guided attack mode to use for all sweep runs.",
    )
    parser.add_argument(
        "--feature-guide-weight",
        type=float,
        default=None,
        help="Optional weight for the feature-guidance loss term.",
    )
    parser.add_argument(
        "--feature-contrast-weight",
        type=float,
        default=None,
        help="Optional oracle-contrastive penalty weight for oracle_contrastive_probe mode.",
    )
    parser.add_argument(
        "--feature-contrast-margin",
        type=float,
        default=None,
        help="Optional oracle-contrastive margin for oracle_contrastive_probe mode.",
    )
    args = parser.parse_args()

    lam_vals = [float(x.strip()) for x in args.lambdas.split(",") if x.strip()]
    extra_args = []
    if args.k_shot is not None:
        extra_args += ["--k-shot", str(args.k_shot)]
    if args.prompt_length is not None:
        extra_args += ["--prompt-length", str(args.prompt_length)]
    if args.feature_guide is not None:
        extra_args += ["--feature-guide", args.feature_guide]
    if args.feature_guide_weight is not None:
        extra_args += ["--feature-guide-weight", str(args.feature_guide_weight)]
    if args.feature_contrast_weight is not None:
        extra_args += ["--feature-contrast-weight", str(args.feature_contrast_weight)]
    if args.feature_contrast_margin is not None:
        extra_args += ["--feature-contrast-margin", str(args.feature_contrast_margin)]

    summary = {
        "suite": args.suite,
        "seed": args.seed,
        "lambdas": lam_vals,
        "feature_guide": args.feature_guide,
        "feature_guide_weight": args.feature_guide_weight,
        "generated_utc": datetime.utcnow().isoformat() + "Z",
        "results": {},
    }

    for lam in lam_vals:
        lam_tag = str(lam).replace(".", "p")
        run_train(args.config, args.suite, args.seed, lam, extra_args)
        suite_name_for_io = args.suite
        if args.prompt_length is not None:
            suite_name_for_io = f"{suite_name_for_io}_prompt{args.prompt_length}"
        if args.k_shot is not None:
            suite_name_for_io = f"{suite_name_for_io}_kshot{args.k_shot}"
        suite_name_for_io = f"{suite_name_for_io}_lam{lam_tag}"
        if args.feature_guide is not None:
            fg_tag = args.feature_guide.replace("_", "")
            suite_name_for_io = f"{suite_name_for_io}_fg{fg_tag}"
        log_path = f"results/logs/{suite_name_for_io}_seed_{args.seed}.jsonl"
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"Expected log not found: {log_path}")
        last = load_last_metric(log_path)
        summary["results"][str(lam)] = last

    os.makedirs("results/analysis", exist_ok=True)
    out_name = f"{args.suite}_tradeoff_sweep_seed_{args.seed}"
    if args.feature_guide is not None:
        out_name += f"_fg{args.feature_guide.replace('_', '')}"
    out_path = f"results/analysis/{out_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote sweep summary to {out_path}")


if __name__ == "__main__":
    main()
