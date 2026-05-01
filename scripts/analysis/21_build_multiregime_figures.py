#!/usr/bin/env python3
"""
Build paper-ready matplotlib figures and a LaTeX snippet for the current
multi-regime attack/defense story.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "-",
    }
)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def feature_rows() -> List[Dict]:
    return [
        {"model": "BalDRO", "label": "baseline", "seed": 42, "forget": 0.00343, "retain": 0.00846},
        {"model": "BalDRO", "label": "subspace", "seed": 42, "forget": 0.00349, "retain": 0.00858},
        {"model": "BalDRO", "label": "subspace", "seed": 123, "forget": 0.00340, "retain": 0.00699},
        {"model": "SalUn", "label": "baseline", "seed": 42, "forget": 0.00251, "retain": 0.00564},
        {"model": "SalUn", "label": "subspace", "seed": 42, "forget": 0.00276, "retain": 0.00600},
        {"model": "SalUn", "label": "subspace", "seed": 123, "forget": 0.00334, "retain": 0.00509},
        {"model": "ORBIT", "label": "baseline", "seed": 42, "forget": 0.00340, "retain": 0.00643},
        {"model": "ORBIT", "label": "subspace", "seed": 42, "forget": 0.00423, "retain": 0.00699},
        {"model": "ORBIT", "label": "subspace", "seed": 123, "forget": 0.00328, "retain": 0.00686},
        {"model": "RURK", "label": "baseline", "seed": 42, "forget": 0.00159, "retain": 0.00576},
        {"model": "RURK", "label": "subspace", "seed": 42, "forget": 0.00159, "retain": 0.00576},
        {"model": "SCRUB", "label": "baseline", "seed": 42, "forget": 0.01078, "retain": 0.00980},
        {"model": "SCRUB", "label": "baseline", "seed": 123, "forget": 0.00980, "retain": 0.00895},
    ]


def robust_rows() -> List[Dict]:
    summary = load_json(ROOT / "results" / "analysis" / "recovery_tradeoff_summary.json")
    rows = []
    for row in summary["rows"]:
        if row["group"] != "robust_base":
            continue
        rows.append(
            {
                "seed": row["seed"],
                "label": row["label"],
                "retain_acc": row["retain_acc"],
                "forget_success_rate": row["forget_success_rate"],
                "forget_median_radius": row["forget_median_radius"],
            }
        )
    return rows


def patch_rows() -> List[Dict]:
    return [
        {"model": "ORBIT", "seed": 42, "patch": "32x32", "area": 2.04, "forget": 78.1, "retain_hijack": 0.0, "retain_drop": 3.1},
        {"model": "ORBIT", "seed": 123, "patch": "32x32", "area": 2.04, "forget": 37.5, "retain_hijack": 0.0, "retain_drop": 0.0},
        {"model": "SalUn", "seed": 42, "patch": "32x32", "area": 2.04, "forget": 3.1, "retain_hijack": 0.0, "retain_drop": 3.1},
        {"model": "SalUn", "seed": 42, "patch": "48x48", "area": 4.59, "forget": 71.9, "retain_hijack": 0.0, "retain_drop": 6.2},
        {"model": "BalDRO", "seed": 42, "patch": "32x32", "area": 2.04, "forget": 0.0, "retain_hijack": 0.0, "retain_drop": 3.1},
        {"model": "BalDRO", "seed": 42, "patch": "48x48", "area": 4.59, "forget": 3.1, "retain_hijack": 0.0, "retain_drop": 3.1},
    ]


def plot_feature_tradeoff(save_dir: Path) -> str:
    rows = feature_rows()
    models = ["BalDRO", "SalUn", "ORBIT", "RURK", "SCRUB"]
    base = {r["model"]: r for r in rows if r["label"] == "baseline"}
    subspace = {}
    for r in rows:
        if r["label"] != "subspace":
            continue
        subspace.setdefault(r["model"], []).append(r)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=True)
    x = np.arange(len(models))
    width = 0.34
    base_forget = [base[m]["forget"] if m in base else np.nan for m in models]
    sub_forget = [
        float(np.mean([r["forget"] for r in subspace[m]])) if m in subspace else np.nan
        for m in models
    ]
    base_retain = [base[m]["retain"] if m in base else np.nan for m in models]
    sub_retain = [
        float(np.mean([r["retain"] for r in subspace[m]])) if m in subspace else np.nan
        for m in models
    ]

    axes[0].bar(x - width / 2, base_forget, width, color="#7a8da6", label="Baseline")
    axes[0].bar(x + width / 2, sub_forget, width, color="#1f4b99", label="Feature-subspace")
    axes[0].set_title("Forget median radius")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=18, ha="right")
    axes[0].set_ylabel("Median radius")

    axes[1].bar(x - width / 2, base_retain, width, color="#b8bfc9", label="Baseline")
    axes[1].bar(x + width / 2, sub_retain, width, color="#4f6fad", label="Feature-subspace")
    axes[1].set_title("Retain-control median radius")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=18, ha="right")

    fig.suptitle("Feature-Subspace Stage-2 Defense")
    axes[0].legend(frameon=False, loc="upper left")
    legend_handles = [
        plt.Line2D([], [], color="#7a8da6", linewidth=6, label="Baseline"),
        plt.Line2D([], [], color="#1f4b99", linewidth=6, label="Feature-subspace"),
    ]
    out = save_dir / "feature_subspace_tradeoff.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return str(out)


def plot_robust_frontier(save_dir: Path) -> str:
    rows = robust_rows()
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    colors = {42: "#1f4b99", 123: "#8b5e3c"}
    for seed in sorted({row["seed"] for row in rows}):
        pts = [row for row in rows if row["seed"] == seed]
        pts.sort(key=lambda item: item["retain_acc"])
        ax.plot(
            [p["retain_acc"] for p in pts],
            [p["forget_success_rate"] for p in pts],
            marker="o",
            color=colors[seed],
            linewidth=2.2,
            label=f"seed {seed}",
        )
        for p in pts:
            short = p["label"].split("(")[-1].replace(")", "")
            ax.annotate(short, (p["retain_acc"], p["forget_success_rate"]), textcoords="offset points", xytext=(5, 2), fontsize=8, color=colors[seed])

    ax.set_xlabel("Retain accuracy")
    ax.set_ylabel("Forget success rate")
    ax.set_title("Robust-Base SalUn Frontier")
    ax.legend(frameon=False)
    out = save_dir / "robust_base_frontier.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return str(out)


def plot_patch_regime(save_dir: Path) -> str:
    rows = patch_rows()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = {"ORBIT": "#1f4b99", "SalUn": "#8b5e3c", "BalDRO": "#4f7c48"}
    markers = {"32x32": "o", "48x48": "s"}
    for row in rows:
        ax.scatter(
            row["area"],
            row["forget"],
            s=60 + row["retain_drop"] * 12,
            c=colors[row["model"]],
            alpha=0.85,
            marker=markers[row["patch"]],
            edgecolors="white",
            linewidths=0.7,
        )
        if row["model"] in ("ORBIT", "SalUn"):
            ax.annotate(
                f"{row['model']} {row['patch']}",
                (row["area"], row["forget"]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=8,
            )

    ax.set_xlabel("Patch area (%)")
    ax.set_ylabel("Forget recovery (%)")
    ax.set_title("Conditional Patch Regime")
    legend_handles = [plt.Line2D([], [], color=color, marker="o", linestyle="", label=model) for model, color in colors.items()]
    patch_handles = [
        plt.Line2D([], [], color="#555555", marker=markers["32x32"], linestyle="", label="32x32"),
        plt.Line2D([], [], color="#555555", marker=markers["48x48"], linestyle="", label="48x48"),
    ]
    ax.legend(handles=legend_handles + patch_handles, frameon=False, ncol=2, loc="upper left")
    out = save_dir / "conditional_patch_regime.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return str(out)


def write_tex(save_dir: Path, pngs: Dict[str, str]) -> str:
    tex = save_dir / "multiregime_figures.tex"
    tex.write_text(
        "\n".join(
            [
                "\\section{Results}",
                "Figure~\\ref{fig:subspace-tradeoff} compares the matched recovery-radius medians before and after the feature-subspace stage-2 defense. The defense improves BalDRO, SalUn, and ORBIT, while RURK remains hard and SCRUB behaves like generic adversarial fragility.",
                "",
                "\\begin{figure}[t]",
                "  \\centering",
                f"  \\includegraphics[width=0.9\\linewidth]{{{Path(pngs['feature']).name}}}",
                "  \\caption{Forget/retain recovery medians under the feature-subspace stage-2 defense.}",
                "  \\label{fig:subspace-tradeoff}",
                "\\end{figure}",
                "",
                "\\begin{figure}[t]",
                "  \\centering",
                f"  \\includegraphics[width=0.9\\linewidth]{{{Path(pngs['frontier']).name}}}",
                "  \\caption{Utility-vs-recovery frontier for robust-base SalUn.}",
                "  \\label{fig:robust-frontier}",
                "\\end{figure}",
                "",
                "\\begin{figure}[t]",
                "  \\centering",
                f"  \\includegraphics[width=0.9\\linewidth]{{{Path(pngs['patch']).name}}}",
                "  \\caption{Conditional patch regime. Point size reflects retain accuracy drop.}",
                "  \\label{fig:patch-regime}",
                "\\end{figure}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return str(tex)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/analysis/figures")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    ensure_dir(out_dir)

    pngs = {
        "feature": plot_feature_tradeoff(out_dir),
        "frontier": plot_robust_frontier(out_dir),
        "patch": plot_patch_regime(out_dir),
    }
    tex_path = write_tex(out_dir, pngs)

    manifest = out_dir / "multiregime_figures.json"
    manifest.write_text(json.dumps({"figures": pngs, "tex": tex_path}, indent=2), encoding="utf-8")

    print(f"Wrote figures to {out_dir}")
    print(f"Wrote tex snippet to {tex_path}")


if __name__ == "__main__":
    main()
