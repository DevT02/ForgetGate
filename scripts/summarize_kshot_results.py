#!/usr/bin/env python
import argparse
import json
import os


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


def select_suite_variant(runs, seeds, candidates):
    """Pick the most complete suite variant from a list of candidate names."""
    best_suite = None
    best_vals = []
    best_missing = list(seeds)

    for suite in candidates:
        vals = []
        missing = []
        for seed in seeds:
            key = (suite, seed)
            if key not in runs:
                missing.append(seed)
            else:
                vals.append(runs[key])

        if best_suite is None or len(missing) < len(best_missing):
            best_suite = suite
            best_vals = vals
            best_missing = missing

    return best_suite, best_vals, best_missing


def collect_suite_union(runs, seeds, candidates):
    """Collect values by taking the first available candidate for each seed."""
    vals = []
    missing = []
    chosen = {}
    for seed in seeds:
        selected_suite = None
        selected_value = None
        for suite in candidates:
            key = (suite, seed)
            if key in runs:
                selected_suite = suite
                selected_value = runs[key]
                break
        if selected_suite is None:
            missing.append(seed)
        else:
            vals.append(selected_value)
            chosen[seed] = selected_suite
    return vals, missing, chosen


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

    # SCRUB follow-up (default prompt length)
    lines.append("\n## SCRUB follow-up (default prompt length)\n")
    lines.append("| K-shot | Oracle | SCRUB | Missing (Oracle/SCRUB) |")
    lines.append("|---|---|---|---|")
    scrub_sources = {}
    for k in kshots_default:
        oracle_vals, oracle_missing = collect(f"vpt_oracle_vit_cifar10_forget0_{k}shot")
        scrub_vals, scrub_missing, chosen_suites = collect_suite_union(
            runs,
            seeds,
            [
                f"vpt_resurrect_scrub_forget0_{k}shot",
                f"vpt_resurrect_scrub_forget0_10shot_kshot{k}",
            ],
        )
        o_mean, o_std = mean_std(oracle_vals)
        s_mean, s_std = mean_std(scrub_vals)
        missing = f"{oracle_missing}/{scrub_missing}"
        scrub_sources[k] = chosen_suites
        lines.append(
            f"| {k} | {fmt_pct(o_mean)} +/- {fmt_pct(o_std)} | {fmt_pct(s_mean)} +/- {fmt_pct(s_std)} | {missing} |"
        )

    lines.append("")
    lines.append("Coverage note: SCRUB follow-up uses tracked logs for seeds 42/123/456 at k=10/25/50/100.")
    lines.append("For k=25, seed 42 comes from the canonical `vpt_resurrect_scrub_forget0_25shot` log and")
    lines.append("seeds 123/456 come from `vpt_resurrect_scrub_forget0_10shot_kshot25` override logs.")

    lines.append("\n### SCRUB follow-up per-seed\n")
    lines.append("| Seed | SCRUB k=10 | SCRUB k=25 | SCRUB k=50 | SCRUB k=100 |")
    lines.append("|---|---|---|---|---|")
    for seed in seeds:
        s10 = get_val(scrub_sources[10].get(seed), seed)
        s25 = get_val(scrub_sources[25].get(seed), seed)
        s50 = get_val(scrub_sources[50].get(seed), seed)
        s100 = get_val(scrub_sources[100].get(seed), seed)
        lines.append(
            f"| {seed} | {fmt_pct(s10)} | {fmt_pct(s25)} | {fmt_pct(s50)} | {fmt_pct(s100)} |"
        )

    # Low-shot controls (prompt length 5)
    lines.append("\n## Low-shot controls (prompt length 5)\n")
    lines.append("| K-shot | Oracle | KL | Missing (Oracle/KL) |")
    lines.append("|---|---|---|---|")
    lowshot_sources = {}
    for k in kshots_controls:
        oracle_suite, oracle_vals, oracle_missing = select_suite_variant(
            runs,
            seeds,
            [
                f"vpt_oracle_vit_cifar10_forget0_10shot_prompt5_kshot{k}",
                f"vpt_oracle_vit_cifar10_forget0_{k}shot_prompt5",
            ],
        )
        kl_suite, kl_vals, kl_missing = select_suite_variant(
            runs,
            seeds,
            [
                f"vpt_resurrect_kl_forget0_10shot_prompt5_kshot{k}",
                f"vpt_resurrect_kl_forget0_{k}shot_prompt5",
            ],
        )
        o_mean, o_std = mean_std(oracle_vals)
        k_mean, k_std = mean_std(kl_vals)
        missing = f"{oracle_missing}/{kl_missing}"
        lowshot_sources[k] = {"oracle": oracle_suite, "kl": kl_suite}
        lines.append(
            f"| {k} | {fmt_pct(o_mean)} +/- {fmt_pct(o_std)} | {fmt_pct(k_mean)} +/- {fmt_pct(k_std)} | {missing} |"
        )

    lines.append("")
    lines.append("Source note: prompt-length-5 low-shot controls prefer `*_10shot_prompt5_kshot{k}` logs and")
    lines.append("fall back to legacy `*_{k}shot_prompt5` logs when the newer filenames are incomplete.")

    lines.append("\n### Low-shot controls (prompt length 5) per-seed\n")
    lines.append("| Seed | Oracle k=1 | KL k=1 | Oracle k=5 | KL k=5 |")
    lines.append("|---|---|---|---|---|")
    for seed in seeds:
        o1 = get_val(lowshot_sources[1]["oracle"], seed)
        k1 = get_val(lowshot_sources[1]["kl"], seed)
        o5 = get_val(lowshot_sources[5]["oracle"], seed)
        k5 = get_val(lowshot_sources[5]["kl"], seed)
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

    # Stratified confidence buckets (prompt length 10, k=10)
    lines.append("\n## Stratified forget-set (prompt length 10, k=10)\n")
    lines.append("| Bucket | KL | Missing (KL) |")
    lines.append("|---|---|---|")
    for mode, suffix in [("high_conf", "highconf"), ("mid_conf", "midconf"), ("low_conf", "lowconf")]:
        kl_vals, kl_missing = collect(f"vpt_resurrect_kl_forget0_10shot_{suffix}")
        k_mean, k_std = mean_std(kl_vals)
        missing = f"{kl_missing}"
        lines.append(
            f"| {mode} | {fmt_pct(k_mean)} +/- {fmt_pct(k_std)} | {missing} |"
        )

    lines.append("\n### Stratified forget-set per-seed (prompt length 10, k=10)\n")
    lines.append("| Seed | KL high | KL mid | KL low |")
    lines.append("|---|---|---|---|")
    for seed in seeds:
        kh = get_val("vpt_resurrect_kl_forget0_10shot_highconf", seed)
        km = get_val("vpt_resurrect_kl_forget0_10shot_midconf", seed)
        kl = get_val("vpt_resurrect_kl_forget0_10shot_lowconf", seed)
        lines.append(
            f"| {seed} | {fmt_pct(kh)} | {fmt_pct(km)} | {fmt_pct(kl)} |"
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
