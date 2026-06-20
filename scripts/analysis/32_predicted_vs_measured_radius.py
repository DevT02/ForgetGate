#!/usr/bin/env python3
"""
Predicted-vs-measured certified recovery radius figure (Theorem 4 of
results/analysis/figures/theory_appendix.tex).

Joins a smoothed-radius audit (per-sample Cohen-Rosenfeld-Kolter L_2
certificate produced by scripts/audits/27_smoothed_radius_audit.py) with
a measured recovery-radius audit (per-sample L_inf binary-search result
from scripts/audits/17_recovery_radius_audit.py) on the same checkpoint.

Outputs:
  - results/analysis/figures/predicted_vs_measured_radius.png
  - results/analysis/metrics/predicted_vs_measured_radius.json

This is a cross-norm diagnostic, not a direct certificate check. The certificate
is for the binary smoothed detector under Gaussian noise, while the measured
radius here attacks the deterministic base model in L_inf. Points below the
visual y=x guide are therefore not certificate violations; they show where the
converted L2 detector scale is larger than the deterministic L_inf radius found
by the separate audit. A direct certificate test would attack the same smoothed
binary detector in the same L2 norm.
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np


def load_smoothed(path: str, suite: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    forget = d["forget_recovery"][suite]["per_sample"]
    retain = d.get("retain_control", {}).get(suite, {}).get("per_sample", [])
    sigma = d["meta"]["sigma"]
    return {
        "sigma": sigma,
        "forget": {r["sample_index"]: r["certified_radius_l2"] for r in forget},
        "retain": {r["sample_index"]: r["certified_radius_l2"] for r in retain},
    }


def load_measured(path: str, suite: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    eps_max = d["meta"]["eps_max"]
    block = d.get("forget_recovery", d.get("splits", {}).get("forget", {}))
    forget_records = block[suite]["per_sample"]
    retain_records = (d.get("retain_control", d.get("splits", {}).get("retain", {}))
                      .get(suite, {}).get("per_sample", []))
    # Use radius if found, else eps_max as the binary search did not find one within budget.
    def _r(rec):
        v = rec.get("radius")
        return float(v) if v is not None else float(eps_max)
    return {
        "eps_max": eps_max,
        "forget": {rec["sample_index"]: _r(rec) for rec in forget_records},
        "retain": {rec["sample_index"]: _r(rec) for rec in retain_records},
    }


def join_and_plot(smoothed_path, measured_path, suite, input_dim, out_png, out_json):
    sm = load_smoothed(smoothed_path, suite)
    me = load_measured(measured_path, suite)

    # L_2 -> L_inf conversion: ||delta||_inf <= ||delta||_2 / sqrt(d) is wrong;
    # ||delta||_inf >= ||delta||_2 / sqrt(d). So predicted L_inf >= predicted_L2 / sqrt(d).
    # We compare predicted_linf_lb <= measured_linf (measured is the smallest L_inf that succeeds).
    sqrtd = math.sqrt(input_dim)

    rows = []
    for src, label in [("forget", "forget"), ("retain", "retain")]:
        s_per = sm[src]
        m_per = me[src]
        for idx in sorted(set(s_per) & set(m_per)):
            r_pred_l2 = s_per[idx]
            r_pred_linf_lb = r_pred_l2 / sqrtd
            r_meas_linf = m_per[idx]
            rows.append({
                "split": label,
                "sample_index": idx,
                "predicted_radius_l2": r_pred_l2,
                "predicted_radius_linf_lb": r_pred_linf_lb,
                "measured_radius_linf": r_meas_linf,
                "below_crossnorm_guide": r_pred_linf_lb > r_meas_linf,
            })

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({
            "input_dim": input_dim,
            "sigma": sm["sigma"],
            "eps_max": me["eps_max"],
            "rows": rows,
        }, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.style.use(os.path.join(os.path.dirname(__file__), "..", "..", "results", "analysis", "figures", "paper_style.mplstyle"))
    except ImportError:
        print("matplotlib not available; wrote JSON only.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    for label, color in [("forget", "C0"), ("retain", "C1")]:
        xs = [r["predicted_radius_linf_lb"] for r in rows if r["split"] == label]
        ys = [r["measured_radius_linf"] for r in rows if r["split"] == label]
        ax.scatter(xs, ys, c=color, alpha=0.7, label=label, edgecolor="black", linewidth=0.4, s=50)

    all_vals = [r["predicted_radius_linf_lb"] for r in rows] + [r["measured_radius_linf"] for r in rows]
    if all_vals:
        lo = max(min(all_vals) * 0.5, 1e-5)
        hi = max(all_vals) * 2.0
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x visual guide")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Converted detector scale  ($L_2$ radius / $\sqrt{d}$)")
    ax.set_ylabel(r"Deterministic $L_\infty$ recovery radius ($\hat r$)")
    ax.set_title(f"Cross-norm diagnostic only\nsuite={suite}, sigma={sm['sigma']}")
    n_below = sum(1 for r in rows if r["below_crossnorm_guide"])
    ax.text(0.02, 0.98,
            f"n={len(rows)}  below guide={n_below}\nnot a certificate test",
            transform=ax.transAxes, va="top", fontsize=8, alpha=0.7)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_json}")
    print(f"  joined n={len(rows)} (forget+retain), below_guide={n_below}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoothed", required=True, help="Path to smoothed_radius_*.json")
    p.add_argument("--measured", required=True, help="Path to recovery_radius_*.json")
    p.add_argument("--suite", required=True,
                   help="Suite key present in both JSONs (e.g. unlearn_smoothed_margin_vit_cifar10_forget0).")
    p.add_argument("--input-dim", type=int, default=3 * 224 * 224,
                   help="Input dimensionality d, used for L_2 -> L_inf conversion. 3*224*224 = 150528.")
    p.add_argument("--out-png", default="results/analysis/figures/predicted_vs_measured_radius.png")
    p.add_argument("--out-json", default="results/analysis/metrics/predicted_vs_measured_radius.json")
    args = p.parse_args()
    join_and_plot(args.smoothed, args.measured, args.suite, args.input_dim, args.out_png, args.out_json)


if __name__ == "__main__":
    main()
