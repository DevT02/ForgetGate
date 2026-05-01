#!/usr/bin/env python3
"""
Build a paper-facing summary of the current attack/defense story.
"""

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt_pct(value, signed=False):
    if value is None:
        return "N/A"
    scaled = 100.0 * float(value)
    if signed:
        return f"{scaled:+.2f}pp"
    return f"{scaled:.2f}%"


def fmt_pct_tex(value, signed=False):
    if value is None:
        return "N/A"
    scaled = 100.0 * float(value)
    if signed:
        return f"{scaled:+.2f}\\,pp"
    return f"{scaled:.2f}\\%"


def classify_row(forget_gain: float) -> str:
    if forget_gain is None:
        return "unknown"
    if forget_gain >= 0.02:
        return "vulnerable"
    return "resistant"


def build_main_rows(analysis_dir: Path):
    specs = [
        ("Uniform KL", "vpt_resurrect_kl_forget0_10shot", "oracle_contrastive_probe", [42, 123]),
        ("SalUn", "vpt_resurrect_salun_forget0_10shot", "oracle_contrastive_probe", [42, 123]),
        ("SCRUB", "vpt_resurrect_scrub_forget0_10shot", "oracle_contrastive_probe", [42, 123]),
        ("ORBIT", "vpt_resurrect_orbit_forget0_10shot", "oracle_contrastive_probe", [42, 123]),
    ]
    rows = []
    missing = []
    for method, suite, guide, seeds in specs:
        tag = guide.replace("_", "")
        for seed in seeds:
            path = analysis_dir / f"{suite}_fg{tag}_seed_{seed}_summary.json"
            if not path.exists():
                missing.append(str(path))
                continue
            data = load_json(path)
            metrics = data.get("final_metrics", {})
            row = {
                "method": method,
                "suite": suite,
                "guide": guide,
                "seed": seed,
                "resurrection_acc": metrics.get("resurrection_acc"),
                "target_forget": metrics.get("target_test_clean_forget_acc"),
                "prompted_forget": metrics.get("test_clean_forget_acc"),
                "forget_gain": metrics.get("test_clean_forget_gain_vs_target"),
                "target_retain": metrics.get("target_test_clean_retain_acc"),
                "prompted_retain": metrics.get("test_clean_retain_acc"),
                "retain_drop": metrics.get("test_clean_retain_drop_vs_target"),
                "target_overall": metrics.get("target_test_clean_overall_acc"),
                "prompted_overall": metrics.get("test_clean_overall_acc"),
                "overall_drop": metrics.get("test_clean_overall_drop_vs_target"),
                "status": classify_row(metrics.get("test_clean_forget_gain_vs_target")),
                "summary_path": str(path),
            }
            rows.append(row)
    return rows, missing


def build_transfer_rows(analysis_dir: Path):
    path = (
        analysis_dir
        / "vpt_resurrect_kl_forget0_10shot_fgoraclecontrastiveprobe_seed_42_transfer_eval.json"
    )
    if not path.exists():
        return [], [str(path)]

    data = load_json(path)
    rows = []
    for target_suite, metrics in sorted(data.get("targets", {}).items()):
        rows.append(
            {
                "target_suite": target_suite,
                "target_forget": metrics.get("target_clean_forget_acc"),
                "prompted_forget": metrics.get("prompted_clean_forget_acc"),
                "forget_gain": metrics.get("forget_gain_vs_target"),
                "target_retain": metrics.get("target_clean_retain_acc"),
                "prompted_retain": metrics.get("prompted_clean_retain_acc"),
                "retain_drop": metrics.get("retain_drop_vs_target"),
            }
        )
    return rows, []


def summarize_headlines(main_rows, transfer_rows):
    lines = []
    kl_rows = [row for row in main_rows if row["method"] == "Uniform KL"]
    positive_rows = sorted(
        [row for row in kl_rows if (row["forget_gain"] or 0.0) > 0.0],
        key=lambda row: row["forget_gain"],
        reverse=True,
    )
    if positive_rows:
        best = positive_rows[0]
        lines.append(
            "The strongest adaptive attack is still `oracle_contrastive_probe` on "
            f"`Uniform KL`, seed `{best['seed']}`: clean forget goes from "
            f"`{fmt_pct(best['target_forget'])}` to `{fmt_pct(best['prompted_forget'])}` "
            f"({fmt_pct(best['forget_gain'], signed=True)}) with retain drop "
            f"`{fmt_pct(best['retain_drop'], signed=True)}`."
        )

    resistant_methods = []
    for method in ["SalUn", "SCRUB", "ORBIT"]:
        rows = [row for row in main_rows if row["method"] == method]
        if rows and all((row["forget_gain"] or 0.0) <= 0.0 for row in rows):
            resistant_methods.append(method)
    if resistant_methods:
        lines.append(
            "Under the same attack family, "
            + ", ".join(f"`{name}`" for name in resistant_methods)
            + " stay in the resistant bucket on the tracked seeds."
        )

    kl_transfer = next(
        (row for row in transfer_rows if row["target_suite"] == "unlearn_kl_vit_cifar10_forget0"),
        None,
    )
    transfer_resistant = [
        row["target_suite"]
        for row in transfer_rows
        if row["target_suite"] != "unlearn_kl_vit_cifar10_forget0"
        and (row["forget_gain"] or 0.0) == 0.0
    ]
    if kl_transfer and transfer_resistant:
        lines.append(
            "The KL-trained prompt is not universal: it transfers back onto `KL` "
            f"({fmt_pct(kl_transfer['forget_gain'], signed=True)}) but not onto "
            + ", ".join(f"`{name}`" for name in transfer_resistant)
            + "."
        )
    return lines


def build_markdown(main_rows, transfer_rows, missing_main, missing_transfer):
    lines = [
        "# Paper Story Summary",
        "",
    ]
    headline_lines = summarize_headlines(main_rows, transfer_rows)
    if headline_lines:
        lines.append("## Headline")
        lines.extend(f"- {line}" for line in headline_lines)
        lines.append("")

    lines.append("## Main Attack Table")
    if main_rows:
        lines.append(
            "| Method | Seed | Status | Resurrection | Target Forget | Prompted Forget | Forget Gain | Target Retain | Prompted Retain | Retain Drop |"
        )
        lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in sorted(main_rows, key=lambda item: (item["method"], item["seed"])):
            lines.append(
                f"| {row['method']} | {row['seed']} | {row['status']} | "
                f"{fmt_pct(row['resurrection_acc'])} | "
                f"{fmt_pct(row['target_forget'])} | "
                f"{fmt_pct(row['prompted_forget'])} | "
                f"{fmt_pct(row['forget_gain'], signed=True)} | "
                f"{fmt_pct(row['target_retain'])} | "
                f"{fmt_pct(row['prompted_retain'])} | "
                f"{fmt_pct(row['retain_drop'], signed=True)} |"
            )
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Transfer Table")
    if transfer_rows:
        lines.append(
            "| Target Suite | Target Forget | Prompted Forget | Forget Gain | Target Retain | Prompted Retain | Retain Drop |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in transfer_rows:
            lines.append(
                f"| {row['target_suite']} | "
                f"{fmt_pct(row['target_forget'])} | "
                f"{fmt_pct(row['prompted_forget'])} | "
                f"{fmt_pct(row['forget_gain'], signed=True)} | "
                f"{fmt_pct(row['target_retain'])} | "
                f"{fmt_pct(row['prompted_retain'])} | "
                f"{fmt_pct(row['retain_drop'], signed=True)} |"
            )
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Notes")
    lines.append("- `Uniform KL` seed `456` is excluded from attack claims because the clean unlearning checkpoint failed.")
    lines.append("- Oracle resistance is currently supported by earlier oracle controls and by the transfer table, not by a dedicated `oracle_contrastive_probe` train run.")
    lines.append("- If you want a more aggressive red-team result, the next experiment should be a universal patch or multi-target ensemble prompt, not another single-model VPT rerun.")
    lines.append("")

    lines.append("## Missing Inputs")
    if missing_main or missing_transfer:
        for item in missing_main:
            lines.append(f"- missing main summary: `{item}`")
        for item in missing_transfer:
            lines.append(f"- missing transfer eval: `{item}`")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines) + "\n"


def build_latex(main_rows, transfer_rows):
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Adaptive prompt attack results on clean test data. Forget gain is relative to the attacked model before prompting.}",
        "\\label{tab:adaptive_attack_main}",
        "\\begin{tabular}{llrrrrrrr}",
        "\\toprule",
        "Method & Seed & Status & Res. Acc & Target Forget & Prompted Forget & Forget Gain & Target Retain & Retain Drop \\\\",
        "\\midrule",
    ]
    for row in sorted(main_rows, key=lambda item: (item["method"], item["seed"])):
        lines.append(
            f"{row['method']} & {row['seed']} & {row['status']} & "
            f"{fmt_pct_tex(row['resurrection_acc'])} & "
            f"{fmt_pct_tex(row['target_forget'])} & "
            f"{fmt_pct_tex(row['prompted_forget'])} & "
            f"{fmt_pct_tex(row['forget_gain'], signed=True)} & "
            f"{fmt_pct_tex(row['target_retain'])} & "
            f"{fmt_pct_tex(row['retain_drop'], signed=True)} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
            "",
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Zero-shot transfer of the KL-trained oracle-contrastive prompt from seed 42.}",
            "\\label{tab:adaptive_attack_transfer}",
            "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            "Target Suite & Target Forget & Prompted Forget & Forget Gain & Target Retain & Prompted Retain & Retain Drop \\\\",
            "\\midrule",
        ]
    )
    for row in transfer_rows:
        lines.append(
            f"{row['target_suite']} & "
            f"{fmt_pct_tex(row['target_forget'])} & "
            f"{fmt_pct_tex(row['prompted_forget'])} & "
            f"{fmt_pct_tex(row['forget_gain'], signed=True)} & "
            f"{fmt_pct_tex(row['target_retain'])} & "
            f"{fmt_pct_tex(row['prompted_retain'])} & "
            f"{fmt_pct_tex(row['retain_drop'], signed=True)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Build a paper-facing summary of current results.")
    parser.add_argument("--analysis-dir", default="results/analysis")
    parser.add_argument("--out-md", default="results/analysis/paper_story_summary.md")
    parser.add_argument("--out-tex", default="results/analysis/paper_story_tables.tex")
    parser.add_argument("--out-json", default="results/analysis/paper_story_summary.json")
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    main_rows, missing_main = build_main_rows(analysis_dir)
    transfer_rows, missing_transfer = build_transfer_rows(analysis_dir)

    payload = {
        "main_rows": main_rows,
        "transfer_rows": transfer_rows,
        "missing_main_summaries": missing_main,
        "missing_transfer_evals": missing_transfer,
        "headline": summarize_headlines(main_rows, transfer_rows),
    }

    Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    Path(args.out_md).write_text(
        build_markdown(main_rows, transfer_rows, missing_main, missing_transfer),
        encoding="utf-8",
    )
    Path(args.out_tex).write_text(build_latex(main_rows, transfer_rows), encoding="utf-8")

    print(f"Wrote story JSON to {args.out_json}")
    print(f"Wrote story summary to {args.out_md}")
    print(f"Wrote story tables to {args.out_tex}")


if __name__ == "__main__":
    main()
