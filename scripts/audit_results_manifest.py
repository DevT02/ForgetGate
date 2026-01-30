import re
import json
from pathlib import Path
from datetime import datetime
import argparse

PATTERNS = {
    "kshot_default": {
        "kshots": [10,25,50,100],
        "templates": [
            "vpt_oracle_vit_cifar10_forget0_{k}shot_seed_{seed}.jsonl",
            "vpt_resurrect_kl_forget0_{k}shot_seed_{seed}.jsonl",
        ],
    },
    "lowshot_controls": {
        "kshots": [1,5],
        "templates": [
            "vpt_oracle_vit_cifar10_forget0_10shot_prompt5_kshot{k}_seed_{seed}.jsonl",
            "vpt_resurrect_kl_forget0_10shot_prompt5_kshot{k}_seed_{seed}.jsonl",
        ],
    },
    "prompt_length_ablation": {
        "pls": [1,2,5],
        "templates": [
            "vpt_oracle_vit_cifar10_forget0_10shot_prompt{pl}_seed_{seed}.jsonl",
            "vpt_resurrect_kl_forget0_10shot_prompt{pl}_seed_{seed}.jsonl",
        ],
    },
    "classwise_10shot": {
        "classes": [1,2,5,9],
        "templates": [
            "vpt_oracle_vit_cifar10_forget{c}_10shot_seed_{seed}.jsonl",
            "vpt_resurrect_kl_forget{c}_10shot_seed_{seed}.jsonl",
        ],
    },
    "autoattack_eval": {
        "templates": [
            "eval_autoattack_vit_cifar10_forget0_seed_{seed}_evaluation.json",
        ],
    },
    "dependency_eval": {
        "templates": [
            "eval_dependency_vit_cifar10_forget0_seed_{seed}_evaluation.json",
        ],
    },
    "clean_baselines_eval": {
        "templates": [
            "eval_paper_baselines_vit_cifar10_forget0_seed_{seed}_evaluation.json",
        ],
    },
}


def list_logs(results_dir):
    logs = []
    for p in Path(results_dir).rglob("*.jsonl"):
        logs.append(str(p.as_posix()))
    for p in Path(results_dir).rglob("*_evaluation.json"):
        logs.append(str(p.as_posix()))
    return sorted(logs)


def find_missing(results_dir, seeds, profile, prompt_seeds=None):
    missing = []
    base = Path(results_dir)

    def exists(rel):
        return (base / rel).exists()

    if profile == "kshot_default":
        for seed in seeds:
            for k in PATTERNS[profile]["kshots"]:
                for tpl in PATTERNS[profile]["templates"]:
                    rel = tpl.format(seed=seed, k=k)
                    if not exists(rel):
                        missing.append(rel)

    elif profile == "lowshot_controls":
        for seed in seeds:
            for k in PATTERNS[profile]["kshots"]:
                for tpl in PATTERNS[profile]["templates"]:
                    rel = tpl.format(seed=seed, k=k)
                    if not exists(rel):
                        missing.append(rel)

    elif profile == "prompt_length_ablation":
        for seed in (prompt_seeds or seeds):
            for pl in PATTERNS[profile]["pls"]:
                for tpl in PATTERNS[profile]["templates"]:
                    rel = tpl.format(seed=seed, pl=pl)
                    if not exists(rel):
                        missing.append(rel)

    elif profile == "classwise_10shot":
        for seed in seeds:
            for c in PATTERNS[profile]["classes"]:
                for tpl in PATTERNS[profile]["templates"]:
                    rel = tpl.format(seed=seed, c=c)
                    if not exists(rel):
                        missing.append(rel)

    else:
        for seed in seeds:
            for tpl in PATTERNS[profile]["templates"]:
                rel = tpl.format(seed=seed)
                if not exists(rel):
                    missing.append(rel)

    return sorted(set(missing))


def main():
    ap = argparse.ArgumentParser(description="Audit results and emit a manifest")
    ap.add_argument("--results-dir", default="results/logs")
    ap.add_argument("--analysis-dir", default="results/analysis")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42,123,456])
    ap.add_argument("--prompt-seeds", nargs="+", type=int, default=[42,123],
                    help="Seeds used for prompt-length ablation")
    args = ap.parse_args()

    logs = list_logs(args.results_dir)

    missing = {
        "kshot_default": find_missing(args.results_dir, args.seeds, "kshot_default"),
        "lowshot_controls": find_missing(args.results_dir, args.seeds, "lowshot_controls"),
        "prompt_length_ablation": find_missing(args.results_dir, args.seeds, "prompt_length_ablation", args.prompt_seeds),
        "classwise_10shot": find_missing(args.results_dir, args.seeds, "classwise_10shot"),
        "autoattack_eval": find_missing(args.results_dir, args.seeds, "autoattack_eval"),
        "dependency_eval": find_missing(args.results_dir, args.seeds, "dependency_eval"),
        "clean_baselines_eval": find_missing(args.results_dir, args.seeds, "clean_baselines_eval"),
    }

    manifest = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results_dir": args.results_dir,
        "seeds": args.seeds,
        "prompt_seeds": args.prompt_seeds,
        "log_count": len(logs),
        "logs": logs,
        "missing": missing,
    }

    analysis_dir = Path(args.analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Simple report
    report_lines = ["# Results Manifest", "", f"Generated: {manifest['generated_at']}", ""]
    report_lines.append(f"Total logs found: {manifest['log_count']}")
    report_lines.append("")
    for key, items in missing.items():
        report_lines.append(f"## Missing: {key}")
        if items:
            report_lines.extend([f"- {p}" for p in items])
        else:
            report_lines.append("- none")
        report_lines.append("")
    (analysis_dir / "manifest_report.md").write_text("\n".join(report_lines))

    print("Manifest written to results/analysis/manifest.json")
    print("Report written to results/analysis/manifest_report.md")


if __name__ == "__main__":
    main()
