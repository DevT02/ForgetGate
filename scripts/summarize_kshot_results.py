#!/usr/bin/env python
import argparse
import json
import os
from collections import defaultdict


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def last_resurrection_acc(path):
    rows = read_jsonl(path)
    if not rows:
        return None
    return rows[-1].get("resurrection_acc")


def fmt_pct(x):
    if x is None:
        return "N/A"
    return f"{x * 100:.2f}%"


def mean_std(values):
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, var ** 0.5


def load_runs(log_dir):
    runs = {}
    for name in os.listdir(log_dir):
        if not name.endswith(".jsonl"):
            continue
        if "_seed_" not in name:
            continue
        suite, seed_str = name.rsplit("_seed_", 1)
        seed = int(seed_str.replace(".jsonl", ""))
        runs[(suite, seed)] = last_resurrection_acc(os.path.join(log_dir, name))
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--log-dir", default="results/logs")
    parser.add_argument("--out", default="results/analysis/kshot_summary.md")
    args = parser.parse_args()

    runs = load_runs(args.log_dir)

    # Expected suite patterns
    kshots_default = [10, 25, 50, 100]
    kshots_controls = [1, 5]
    seeds = args.seeds

    def collect(suite):
        vals = []
        missing = []
        for seed in seeds:
            key = (suite, seed)
            if key not in runs:
                missing.append(seed)
            else:
                vals.append(runs[key])
        return vals, missing

    def get_val(suite, seed):
        return runs.get((suite, seed))

    lines = []
    lines.append("# K-shot Summary\n")
    lines.append(f"Seeds: {', '.join(map(str, seeds))}\n")

    # Default prompt length (10 tokens)
    lines.append("## Default prompt length (10 tokens)\n")
    lines.append("| K-shot | Oracle | KL | Missing (Oracle/KL) |")
    lines.append("|---|---|---|---|")
    for k in kshots_default:
        oracle_vals, oracle_missing = collect(f"vpt_oracle_vit_cifar10_forget0_{k}shot")
        kl_vals, kl_missing = collect(f"vpt_resurrect_kl_forget0_{k}shot")
        o_mean, o_std = mean_std(oracle_vals)
        k_mean, k_std = mean_std(kl_vals)
        missing = f"{oracle_missing}/{kl_missing}"
        lines.append(
            f"| {k} | {fmt_pct(o_mean)} +/- {fmt_pct(o_std)} | {fmt_pct(k_mean)} +/- {fmt_pct(k_std)} | {missing} |"
        )

    # Low-shot controls (prompt length 5)
    lines.append("\n## Low-shot controls (prompt length 5)\n")
    lines.append("| K-shot | Oracle | KL | Missing (Oracle/KL) |")
    lines.append("|---|---|---|---|")
    for k in kshots_controls:
        oracle_vals, oracle_missing = collect(f"vpt_oracle_vit_cifar10_forget0_10shot_prompt5_kshot{k}")
        kl_vals, kl_missing = collect(f"vpt_resurrect_kl_forget0_10shot_prompt5_kshot{k}")
        o_mean, o_std = mean_std(oracle_vals)
        k_mean, k_std = mean_std(kl_vals)
        missing = f"{oracle_missing}/{kl_missing}"
        lines.append(
            f"| {k} | {fmt_pct(o_mean)} +/- {fmt_pct(o_std)} | {fmt_pct(k_mean)} +/- {fmt_pct(k_std)} | {missing} |"
        )

    lines.append("\n### Low-shot controls (prompt length 5) per-seed\n")
    lines.append("| Seed | Oracle k=1 | KL k=1 | Oracle k=5 | KL k=5 |")
    lines.append("|---|---|---|---|---|")
    for seed in seeds:
        o1 = get_val("vpt_oracle_vit_cifar10_forget0_10shot_prompt5_kshot1", seed)
        k1 = get_val("vpt_resurrect_kl_forget0_10shot_prompt5_kshot1", seed)
        o5 = get_val("vpt_oracle_vit_cifar10_forget0_10shot_prompt5_kshot5", seed)
        k5 = get_val("vpt_resurrect_kl_forget0_10shot_prompt5_kshot5", seed)
        lines.append(
            f"| {seed} | {fmt_pct(o1)} | {fmt_pct(k1)} | {fmt_pct(o5)} | {fmt_pct(k5)} |"
        )

    # Label controls (prompt length 5, k=10)
    lines.append("\n## Label controls (prompt length 5, k=10)\n")
    lines.append("| Control | Oracle | KL | Missing (Oracle/KL) |")
    lines.append("|---|---|---|---|")
    for mode in ["shufflelabels", "randomlabels"]:
        oracle_vals, oracle_missing = collect(f"vpt_oracle_vit_cifar10_forget0_10shot_prompt5_{mode}")
        kl_vals, kl_missing = collect(f"vpt_resurrect_kl_forget0_10shot_prompt5_{mode}")
        o_mean, o_std = mean_std(oracle_vals)
        k_mean, k_std = mean_std(kl_vals)
        missing = f"{oracle_missing}/{kl_missing}"
        lines.append(
            f"| {mode} | {fmt_pct(o_mean)} +/- {fmt_pct(o_std)} | {fmt_pct(k_mean)} +/- {fmt_pct(k_std)} | {missing} |"
        )

    lines.append("\n### Label controls per-seed (prompt length 5, k=10)\n")
    lines.append("| Seed | Oracle shuffle | KL shuffle | Oracle random | KL random |")
    lines.append("|---|---|---|---|---|")
    for seed in seeds:
        os_val = get_val("vpt_oracle_vit_cifar10_forget0_10shot_prompt5_shufflelabels", seed)
        ks_val = get_val("vpt_resurrect_kl_forget0_10shot_prompt5_shufflelabels", seed)
        or_val = get_val("vpt_oracle_vit_cifar10_forget0_10shot_prompt5_randomlabels", seed)
        kr_val = get_val("vpt_resurrect_kl_forget0_10shot_prompt5_randomlabels", seed)
        lines.append(
            f"| {seed} | {fmt_pct(os_val)} | {fmt_pct(ks_val)} | {fmt_pct(or_val)} | {fmt_pct(kr_val)} |"
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
