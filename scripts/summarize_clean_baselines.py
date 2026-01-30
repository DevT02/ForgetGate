#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path

METHOD_LABELS = {
    "base_vit_cifar10": "Base Model (No Unlearning)",
    "unlearn_kl_vit_cifar10_forget0": "Uniform KL",
    "unlearn_lora_vit_cifar10_forget0": "CE Ascent",
    "unlearn_salun_vit_cifar10_forget0": "SalUn (Fan et al. 2024)",
    "unlearn_scrub_distill_vit_cifar10_forget0": "SCRUB (Kurmanji et al. 2023)",
    "unlearn_noisy_retain_vit_cifar10_forget0": "Noisy Retain-Only (Proxy)",
}

ORDER = [
    "base_vit_cifar10",
    "unlearn_kl_vit_cifar10_forget0",
    "unlearn_lora_vit_cifar10_forget0",
    "unlearn_salun_vit_cifar10_forget0",
    "unlearn_scrub_distill_vit_cifar10_forget0",
    "unlearn_noisy_retain_vit_cifar10_forget0",
]


def mean_std(values):
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, var ** 0.5


def fmt_pct(x):
    if x is None:
        return "N/A"
    return f"{x * 100:.2f}"


def fmt_pp(x):
    if x is None:
        return "N/A"
    sign = "+" if x >= 0 else "-"
    return f"{sign}{abs(x):.2f}pp"


def load_eval(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="Summarize clean baseline evals")
    ap.add_argument("--log-dir", default="results/logs")
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--suite", default="eval_paper_baselines_vit_cifar10_forget0")
    ap.add_argument("--out-md", default="results/analysis/comparison_table_markdown.txt")
    ap.add_argument("--out-tex", default="results/analysis/comparison_table_latex.txt")
    ap.add_argument("--out-summary", default="results/analysis/clean_baselines_summary.md")
    args = ap.parse_args()

    rows = {k: {"forget": [], "retain": []} for k in ORDER}
    base_retain_by_seed = {}

    for seed in args.seeds:
        path = Path(args.log_dir) / f"{args.suite}_seed_{seed}_evaluation.json"
        if not path.exists():
            continue
        data = load_eval(path)
        for key in ORDER:
            if key not in data:
                continue
            clean = data[key].get("clean")
            if not clean:
                continue
            rows[key]["forget"].append(clean.get("forget_acc"))
            rows[key]["retain"].append(clean.get("retain_acc"))
            if key == "base_vit_cifar10":
                base_retain_by_seed[seed] = clean.get("retain_acc")

    # Delta utility: retain(method) - retain(base) per-seed
    delta_by_method = {k: [] for k in ORDER}
    for seed in args.seeds:
        if seed not in base_retain_by_seed:
            continue
        base_ret = base_retain_by_seed[seed]
        for key in ORDER:
            if key == "base_vit_cifar10":
                continue
            # We assume retain list order aligns with seeds, so compute directly by reloading
            path = Path(args.log_dir) / f"{args.suite}_seed_{seed}_evaluation.json"
            if not path.exists():
                continue
            data = load_eval(path)
            clean = data.get(key, {}).get("clean")
            if not clean:
                continue
            delta_by_method[key].append((clean.get("retain_acc") - base_ret) * 100)

    # Build markdown table
    lines = []
    lines.append("# Machine Unlearning Results - CIFAR-10 Class 0\n")
    lines.append("## Comparison of Unlearning Methods\n")
    lines.append("| Method | Forget Acc (%) [lower] | Retain Acc (%) [higher] | Delta Utility |")
    lines.append("|--------|------------------------|-------------------------|---------------|")

    summary_lines = []
    summary_lines.append("# Clean Baselines Summary\n")
    summary_lines.append(f"Seeds: {', '.join(map(str, args.seeds))}\n")

    for key in ORDER:
        f_mean, f_std = mean_std(rows[key]["forget"])
        r_mean, r_std = mean_std(rows[key]["retain"])
        if key == "base_vit_cifar10":
            delta = 0.0
        else:
            d_mean, _d_std = mean_std(delta_by_method[key])
            delta = d_mean if d_mean is not None else None
        lines.append(
            f"| {METHOD_LABELS[key]} | {fmt_pct(f_mean)} +/- {fmt_pct(f_std)} | {fmt_pct(r_mean)} +/- {fmt_pct(r_std)} | {fmt_pp(delta)} |"
        )
        summary_lines.append(
            f"- {METHOD_LABELS[key]}: forget={fmt_pct(f_mean)} +/- {fmt_pct(f_std)}, retain={fmt_pct(r_mean)} +/- {fmt_pct(r_std)}, delta={fmt_pp(delta)}"
        )

    Path(os.path.dirname(args.out_md)).mkdir(parents=True, exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # LaTeX table (ASCII only for delta)
    tex = []
    tex.append("\\begin{table}[t]")
    tex.append("\\centering")
    tex.append("\\caption{Comparison of machine unlearning methods on CIFAR-10. Forget Acc measures accuracy on forgotten class (lower is better). Retain Acc measures accuracy on remaining classes (higher is better).}")
    tex.append("\\label{tab:unlearning_results}")
    tex.append("\\begin{tabular}{lccc}")
    tex.append("\\toprule")
    tex.append("Method & Forget Acc (\\%) $\\downarrow$ & Retain Acc (\\%) $\\uparrow$ & $\\Delta$ Utility \\")
    tex.append("\\midrule")
    for key in ORDER:
        f_mean, f_std = mean_std(rows[key]["forget"])
        r_mean, r_std = mean_std(rows[key]["retain"])
        if key == "base_vit_cifar10":
            delta = 0.0
        else:
            d_mean, _d_std = mean_std(delta_by_method[key])
            delta = d_mean if d_mean is not None else None
        tex.append(
            f"{METHOD_LABELS[key]} & {fmt_pct(f_mean)} $\\pm$ {fmt_pct(f_std)} & {fmt_pct(r_mean)} $\\pm$ {fmt_pct(r_std)} & {fmt_pp(delta)} \\")
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("\\end{table}")

    with open(args.out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")

    with open(args.out_summary, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_tex}")
    print(f"Wrote {args.out_summary}")


if __name__ == "__main__":
    main()
