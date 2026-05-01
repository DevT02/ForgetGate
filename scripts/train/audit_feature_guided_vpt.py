#!/usr/bin/env python3
"""
Audit feature-guided VPT experiments and report missing runs / missing metrics.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_FINAL_KEYS = [
    "restored_best",
    "resurrection_acc",
    "test_clean_overall_acc",
    "test_clean_forget_acc",
    "test_clean_retain_acc",
    "target_test_clean_overall_acc",
    "target_test_clean_forget_acc",
    "target_test_clean_retain_acc",
]

REQUIRED_SWEEP_KEYS = [
    "resurrection_acc",
    "test_clean_overall_acc",
    "test_clean_retain_acc",
]


def fg_tag(feature_guide: str) -> str:
    return feature_guide.replace("_", "")


def read_last_jsonl(path: Path):
    last = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            last = json.loads(line)
    return last


def fmt_pct(value):
    if value is None:
        return "N/A"
    return f"{100.0 * value:.2f}%"


def build_train_command(suite: str, seed: int, feature_guide: str) -> str:
    return (
        "python scripts/3_train_vpt_resurrector.py "
        f"--config configs/experiment_suites.yaml --suite {suite} --seed {seed} "
        f"--feature-guide {feature_guide} --feature-guide-weight 1.0"
    )


def build_sweep_command(suite: str, seed: int, feature_guide: str, lambdas: str) -> str:
    return (
        "python scripts/11_vpt_tradeoff_sweep.py "
        f"--config configs/experiment_suites.yaml --suite {suite} --seed {seed} "
        f"--lambdas {lambdas} --feature-guide {feature_guide} --feature-guide-weight 1.0"
    )


def main():
    parser = argparse.ArgumentParser(description="Audit feature-guided VPT coverage")
    parser.add_argument("--log-dir", default="results/logs")
    parser.add_argument("--analysis-dir", default="results/analysis")
    parser.add_argument("--checkpoint-dir", default="checkpoints/vpt_resurrector")
    parser.add_argument(
        "--train-suites",
        nargs="+",
        default=[
            "vpt_oracle_vit_cifar10_forget0_10shot",
            "vpt_resurrect_kl_forget0_10shot",
            "vpt_resurrect_salun_forget0_10shot",
            "vpt_resurrect_scrub_forget0_10shot",
            "vpt_resurrect_orbit_forget0_10shot",
        ],
    )
    parser.add_argument(
        "--feature-guides",
        nargs="+",
        default=["linear_probe", "prototype_margin", "oracle_contrastive_probe"],
    )
    parser.add_argument(
        "--sweep-feature-guides",
        nargs="+",
        default=["linear_probe", "prototype_margin"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123])
    parser.add_argument("--sweep-suite", default="vpt_resurrect_kl_forget0_10shot")
    parser.add_argument("--sweep-seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--lambdas", default="0.0,0.1,0.5,1.0,2.0,5.0")
    parser.add_argument(
        "--transfer-evals",
        nargs="+",
        default=[
            "results/analysis/vpt_resurrect_kl_forget0_10shot_fgoraclecontrastiveprobe_seed_42_transfer_eval.json"
        ],
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    analysis_dir = Path(args.analysis_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    lambda_keys = [piece.strip() for piece in args.lambdas.split(",") if piece.strip()]
    report = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "train_runs": [],
        "missing_train_logs": [],
        "missing_prompt_dirs": [],
        "runs_missing_final_keys": [],
        "runs_with_eval_errors": [],
        "sweeps": [],
        "missing_sweep_summaries": [],
        "sweeps_missing_lambdas": [],
        "sweeps_missing_keys": [],
        "transfer_evals": [],
        "missing_transfer_evals": [],
        "recommended_commands": [],
    }

    for suite in args.train_suites:
        for feature_guide in args.feature_guides:
            tag = fg_tag(feature_guide)
            suite_name_for_io = f"{suite}_fg{tag}"
            for seed in args.seeds:
                log_path = log_dir / f"{suite_name_for_io}_seed_{seed}.jsonl"
                prompt_path = checkpoint_dir / f"{suite_name_for_io}_seed_{seed}"

                if not log_path.exists():
                    report["missing_train_logs"].append(str(log_path))
                    report["recommended_commands"].append(
                        build_train_command(suite, seed, feature_guide)
                    )
                    continue

                if not prompt_path.exists():
                    report["missing_prompt_dirs"].append(str(prompt_path))

                final_row = read_last_jsonl(log_path)
                if final_row is None:
                    report["runs_missing_final_keys"].append(
                        f"{log_path} :: empty_or_invalid_jsonl"
                    )
                    continue

                missing_keys = [key for key in REQUIRED_FINAL_KEYS if key not in final_row]
                if missing_keys:
                    for key in missing_keys:
                        report["runs_missing_final_keys"].append(f"{log_path} :: {key}")

                if "test_clean_eval_error" in final_row:
                    report["runs_with_eval_errors"].append(
                        f"{log_path} :: {final_row['test_clean_eval_error']}"
                    )

                report["train_runs"].append(
                    {
                        "suite": suite,
                        "feature_guide": feature_guide,
                        "seed": seed,
                        "log_path": str(log_path),
                        "prompt_path": str(prompt_path),
                        "resurrection_acc": final_row.get("resurrection_acc"),
                        "target_test_clean_forget_acc": final_row.get(
                            "target_test_clean_forget_acc"
                        ),
                        "test_clean_forget_acc": final_row.get("test_clean_forget_acc"),
                        "test_clean_forget_gain_vs_target": final_row.get(
                            "test_clean_forget_gain_vs_target"
                        ),
                        "target_test_clean_retain_acc": final_row.get(
                            "target_test_clean_retain_acc"
                        ),
                        "test_clean_retain_acc": final_row.get("test_clean_retain_acc"),
                        "test_clean_retain_drop_vs_target": final_row.get(
                            "test_clean_retain_drop_vs_target"
                        ),
                        "test_clean_overall_acc": final_row.get("test_clean_overall_acc"),
                    }
                )

    for feature_guide in args.sweep_feature_guides:
        tag = fg_tag(feature_guide)
        for seed in args.sweep_seeds:
            summary_path = analysis_dir / (
                f"{args.sweep_suite}_tradeoff_sweep_seed_{seed}_fg{tag}.json"
            )
            if not summary_path.exists():
                report["missing_sweep_summaries"].append(str(summary_path))
                report["recommended_commands"].append(
                    build_sweep_command(args.sweep_suite, seed, feature_guide, args.lambdas)
                )
                continue

            with summary_path.open("r", encoding="utf-8") as handle:
                sweep_data = json.load(handle)

            summary_row = {
                "feature_guide": feature_guide,
                "seed": seed,
                "summary_path": str(summary_path),
                "lambdas": {},
            }

            results_blob = sweep_data.get("results", {})
            for lam in lambda_keys:
                lam_blob = results_blob.get(lam)
                if lam_blob is None:
                    report["sweeps_missing_lambdas"].append(f"{summary_path} :: {lam}")
                    continue
                summary_row["lambdas"][lam] = {
                    "resurrection_acc": lam_blob.get("resurrection_acc"),
                    "test_clean_retain_acc": lam_blob.get("test_clean_retain_acc"),
                }
                for key in REQUIRED_SWEEP_KEYS:
                    if key not in lam_blob:
                        report["sweeps_missing_keys"].append(
                            f"{summary_path} :: lambda={lam} :: {key}"
                        )

            report["sweeps"].append(summary_row)

    for transfer_eval in args.transfer_evals:
        transfer_path = Path(transfer_eval)
        if not transfer_path.exists():
            report["missing_transfer_evals"].append(str(transfer_path))
            continue

        with transfer_path.open("r", encoding="utf-8") as handle:
            transfer_data = json.load(handle)

        transfer_rows = []
        for target_suite, metrics in sorted(transfer_data.get("targets", {}).items()):
            transfer_rows.append(
                {
                    "target_suite": target_suite,
                    "target_clean_forget_acc": metrics.get("target_clean_forget_acc"),
                    "prompted_clean_forget_acc": metrics.get("prompted_clean_forget_acc"),
                    "forget_gain_vs_target": metrics.get("forget_gain_vs_target"),
                    "target_clean_retain_acc": metrics.get("target_clean_retain_acc"),
                    "prompted_clean_retain_acc": metrics.get("prompted_clean_retain_acc"),
                    "retain_drop_vs_target": metrics.get("retain_drop_vs_target"),
                }
            )

        report["transfer_evals"].append(
            {
                "path": str(transfer_path),
                "prompt_dir": transfer_data.get("prompt_dir"),
                "prompt_seed": transfer_data.get("prompt_seed"),
                "targets": transfer_rows,
            }
        )

    out_json = analysis_dir / "feature_guided_vpt_audit.json"
    out_md = analysis_dir / "feature_guided_vpt_audit.md"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Feature-Guided VPT Audit",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Train Runs",
    ]
    if report["train_runs"]:
        lines.append(
            "| Suite | Guide | Seed | Resurrection | Target Forget | Prompted Forget | Forget Gain | Target Retain | Prompted Retain | Retain Drop | Test Overall |"
        )
        lines.append(
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        for row in sorted(
            report["train_runs"],
            key=lambda item: (item["suite"], item["feature_guide"], item["seed"]),
        ):
            lines.append(
                f"| {row['suite']} | {row['feature_guide']} | {row['seed']} | "
                f"{fmt_pct(row['resurrection_acc'])} | "
                f"{fmt_pct(row['target_test_clean_forget_acc'])} | "
                f"{fmt_pct(row['test_clean_forget_acc'])} | "
                f"{fmt_pct(row['test_clean_forget_gain_vs_target'])} | "
                f"{fmt_pct(row['target_test_clean_retain_acc'])} | "
                f"{fmt_pct(row['test_clean_retain_acc'])} | "
                f"{fmt_pct(row['test_clean_retain_drop_vs_target'])} | "
                f"{fmt_pct(row['test_clean_overall_acc'])} |"
            )
    else:
        lines.append("- none")
    lines.append("")

    def add_section(title, items):
        lines.append(f"## {title}")
        if items:
            lines.extend(f"- {item}" for item in items)
        else:
            lines.append("- none")
        lines.append("")

    add_section("Missing Train Logs", report["missing_train_logs"])
    add_section("Missing Prompt Directories", report["missing_prompt_dirs"])
    add_section("Runs Missing Final Keys", report["runs_missing_final_keys"])
    add_section("Runs With Eval Errors", report["runs_with_eval_errors"])
    add_section("Missing Sweep Summaries", report["missing_sweep_summaries"])
    add_section("Sweeps Missing Lambdas", report["sweeps_missing_lambdas"])
    add_section("Sweeps Missing Keys", report["sweeps_missing_keys"])
    lines.append("## Transfer Evaluations")
    if report["transfer_evals"]:
        for transfer_eval in report["transfer_evals"]:
            lines.append(
                f"### Prompt Seed {transfer_eval['prompt_seed']}: `{transfer_eval['prompt_dir']}`"
            )
            lines.append("")
            lines.append(
                "| Target Suite | Target Forget | Prompted Forget | Forget Gain | Target Retain | Prompted Retain | Retain Drop |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for row in transfer_eval["targets"]:
                lines.append(
                    f"| {row['target_suite']} | "
                    f"{fmt_pct(row['target_clean_forget_acc'])} | "
                    f"{fmt_pct(row['prompted_clean_forget_acc'])} | "
                    f"{fmt_pct(row['forget_gain_vs_target'])} | "
                    f"{fmt_pct(row['target_clean_retain_acc'])} | "
                    f"{fmt_pct(row['prompted_clean_retain_acc'])} | "
                    f"{fmt_pct(row['retain_drop_vs_target'])} |"
                )
            lines.append("")
    else:
        lines.append("- none")
        lines.append("")

    add_section("Missing Transfer Evaluations", report["missing_transfer_evals"])
    add_section("Recommended Commands", report["recommended_commands"])

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote audit JSON to {out_json}")
    print(f"Wrote audit report to {out_md}")


if __name__ == "__main__":
    main()
