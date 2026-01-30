import argparse
import json
from pathlib import Path
from datetime import datetime
import statistics as stats


ORDER = [
    "base_vit_cifar10",
    "unlearn_lora_vit_cifar10_forget0",
    "unlearn_kl_vit_cifar10_forget0",
    "unlearn_salun_vit_cifar10_forget0",
    "unlearn_scrub_distill_vit_cifar10_forget0",
    "unlearn_noisy_retain_vit_cifar10_forget0",
]


def mean_std(values):
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return stats.mean(values), stats.stdev(values)


def load_eval(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description="Variance report for clean baselines")
    ap.add_argument("--results-dir", default="results/logs")
    ap.add_argument("--analysis-dir", default="results/analysis")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    ap.add_argument("--std-threshold", type=float, default=5.0,
                    help="Flag std >= threshold percentage points")
    args = ap.parse_args()

    per_method = {k: {"forget": [], "retain": []} for k in ORDER}

    for seed in args.seeds:
        path = Path(args.results_dir) / f"eval_paper_baselines_vit_cifar10_forget0_seed_{seed}_evaluation.json"
        if not path.exists():
            continue
        data = load_eval(path)
        for method in ORDER:
            if method not in data:
                continue
            clean = data[method].get("clean", {})
            f = clean.get("forget_acc")
            r = clean.get("retain_acc")
            if isinstance(f, (int, float)):
                per_method[method]["forget"].append(f * 100)
            if isinstance(r, (int, float)):
                per_method[method]["retain"].append(r * 100)

    rows = []
    flags = []
    for method, vals in per_method.items():
        f_mean, f_std = mean_std(vals["forget"])
        r_mean, r_std = mean_std(vals["retain"])
        rows.append({
            "method": method,
            "forget_mean": f_mean,
            "forget_std": f_std,
            "retain_mean": r_mean,
            "retain_std": r_std,
        })
        if f_std is not None and f_std >= args.std_threshold:
            flags.append(f"{method}: forget std {f_std:.2f}pp")
        if r_std is not None and r_std >= args.std_threshold:
            flags.append(f"{method}: retain std {r_std:.2f}pp")

    report = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seeds": args.seeds,
        "std_threshold_pp": args.std_threshold,
        "rows": rows,
        "flags": flags,
    }

    analysis_dir = Path(args.analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "variance_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    lines = ["# Variance Report", ""]
    lines.append(f"Generated: {report['generated_at']}")
    lines.append(f"Seeds: {', '.join(map(str, args.seeds))}")
    lines.append(f"Std threshold: {args.std_threshold:.2f}pp")
    lines.append("")
    if flags:
        lines.append("## Flags")
        lines.extend([f"- {f}" for f in flags])
    else:
        lines.append("## Flags")
        lines.append("- none")
    lines.append("")
    lines.append("## Summary")
    for row in rows:
        if row["forget_mean"] is None:
            continue
        lines.append(
            f"- {row['method']}: forget={row['forget_mean']:.2f} +/- {row['forget_std']:.2f}pp, "
            f"retain={row['retain_mean']:.2f} +/- {row['retain_std']:.2f}pp"
        )

    (analysis_dir / "variance_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
