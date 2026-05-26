#!/usr/bin/env python3
"""
Lipschitz-radius scatter (Theorem 1 of theory_appendix.tex).

For each available (method, seed) row, joins:
  - the median L_inf recovery radius from recovery_radius_*.json
  - a Lipschitz proxy of the margin function from one of:
      (a) input-Jacobian audit JSONs (jacobian_access_*.json) -- the
          existing audit reports `input_jacobian_trace` per sample,
          which equals sum_i sigma_i^2 of the logit-Jacobian; we take
          sqrt(trace) as an upper bound on the spectral norm, and treat
          this as the proxy L. The L_inf Lipschitz of the margin is
          itself bounded by sqrt(d) * spectral_norm in the worst case;
          we keep the spectral proxy for the scatter and label the axis
          accordingly so the bound stays one-sided in our favour
          (lower L => lower bound on r is sharper).
      (b) on-the-fly computation via src.theory.recovery_certification
          .lipschitz_proxy if a checkpoint is available and a
          --recompute flag is set.

Outputs:
  - results/analysis/figures/lipschitz_radius_scatter.png
  - results/analysis/metrics/lipschitz_radius_scatter.json (joined rows)

Theorem 1 line: r_t(x) >= -m_t / L. The scatter shows the empirical
(L, median_r) pairs; we overlay the line r = c / L for c = -median(m_t)
estimated from the available audit data (or a passed --gamma value). A
method that sits above the line has its radius explained purely by its
Lipschitz; a method below the line has *more* margin-headroom than the
bound predicts -- which is precisely when the bound is the binding
constraint on its certifiable epsilon_c (Cor 1.1).
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from src.theory.recovery_certification import (
    JsonAnalyzer,
    cert_budget_upper_bound,
)


METRICS_DIR = "results/analysis/metrics"
FIG_DIR = "results/analysis/figures"


_SUITE_LABEL = {
    "unlearn_kl_vit_cifar10_forget0": "Uniform-KL",
    "unlearn_salun_vit_cifar10_forget0": "SalUn",
    "unlearn_scrub_distill_vit_cifar10_forget0": "SCRUB",
    "unlearn_scrub_vit_cifar10_forget0": "SCRUB",
    "unlearn_orbit_vit_cifar10_forget0": "ORBIT",
    "unlearn_rurk_vit_cifar10_forget0": "RURK",
    "unlearn_ceu_vit_cifar10_forget0": "CE-U",
    "unlearn_sga_vit_cifar10_forget0": "SGA",
    "unlearn_baldro_vit_cifar10_forget0": "BalDRO",
}


def _short(suite: str) -> str:
    return _SUITE_LABEL.get(suite, suite)


# ---------------------------------------------------------------------------
# Lipschitz proxy from jacobian_access_*.json
# ---------------------------------------------------------------------------

def lipschitz_proxies_from_jacobian_audits(metrics_dir=METRICS_DIR):
    """Returns dict {(suite, seed): L_spectral_proxy} where
    L_spectral_proxy = sqrt(median input_jacobian_trace) of the logit map."""
    out = {}
    for p in glob.glob(os.path.join(metrics_dir, "jacobian_access_*.json")):
        with open(p) as f:
            d = json.load(f)
        meta = d.get("meta", {})
        seed = int(meta.get("seed", -1))
        splits = d.get("splits", {})
        forget = splits.get("forget", {})
        for suite, payload in forget.items():
            # prefer logit; fall back to feature
            for rep in ("logit", "feature"):
                if rep not in payload:
                    continue
                summary = payload[rep].get("summary", {})
                trace = summary.get("input_jacobian_trace", {}).get("median", None)
                if trace is None or trace <= 0:
                    continue
                out[(suite, seed)] = (rep, float(np.sqrt(trace)))
                break
    return out


# ---------------------------------------------------------------------------
# Median recovery radii from recovery_radius_*.json
# ---------------------------------------------------------------------------

def median_radii_from_audits(metrics_dir=METRICS_DIR):
    """Returns list of (suite, seed, median_radius, success_rate, attack_tag, file)."""
    rows = []
    for p in sorted(glob.glob(os.path.join(metrics_dir, "recovery_radius_*.json"))):
        try:
            ja = JsonAnalyzer(p)
        except Exception:
            continue
        attack_tag = os.path.basename(p).replace("recovery_radius_", "").replace(".json", "")
        for r in ja.summarize():
            if r.n_forget == 0:
                continue
            median_r = r.forget_median_r
            # If radius is NaN (no successful recoveries within eps_max), use eps_max as a
            # conservative *lower* bound on the true median radius.
            if not np.isfinite(median_r):
                median_r = r.eps_max
            rows.append({
                "suite": r.suite,
                "seed": r.seed,
                "median_radius": float(median_r),
                "success_rate": float(r.forget_p_at_eps),
                "n_F": r.n_forget,
                "file": attack_tag,
            })
    return rows


# ---------------------------------------------------------------------------
# Join and plot
# ---------------------------------------------------------------------------

def join_and_plot(args):
    L_map = lipschitz_proxies_from_jacobian_audits(args.metrics_dir)
    radii_rows = median_radii_from_audits(args.metrics_dir)

    if not L_map:
        print("No jacobian_access_*.json found; nothing to plot. "
              "Run scripts/audits/16_jacobian_access_audit.py per method first.")
        return

    joined = []
    for row in radii_rows:
        key = (row["suite"], row["seed"])
        if key not in L_map:
            continue
        rep, L = L_map[key]
        joined.append({**row, "rep": rep, "L_proxy": L})

    if not joined:
        print("No (method, seed) overlap between jacobian and recovery_radius audits.")
        print("Methods with Lipschitz proxies:", sorted({k[0] for k in L_map}))
        print("Methods with radius data:", sorted({r['suite'] for r in radii_rows}))
        return

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(args.metrics_dir, exist_ok=True)

    out_json = os.path.join(args.metrics_dir, "lipschitz_radius_scatter.json")
    with open(out_json, "w") as f:
        json.dump({"rows": joined}, f, indent=2)
    print(f"Wrote {len(joined)} joined rows -> {out_json}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figure.")
        return

    Ls = np.array([r["L_proxy"] for r in joined])
    Rs = np.array([r["median_radius"] for r in joined])
    labels = [_short(r["suite"]) for r in joined]
    seeds = [r["seed"] for r in joined]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    # Color by method label
    uniq = sorted(set(labels))
    cmap = plt.get_cmap("tab10")
    color_of = {lbl: cmap(i % 10) for i, lbl in enumerate(uniq)}
    for L, R, lbl, sd in zip(Ls, Rs, labels, seeds):
        ax.scatter([L], [R], color=color_of[lbl], s=80, edgecolor="black", linewidth=0.5,
                   label=lbl)
        ax.annotate(f"{lbl} (s{sd})", (L, R), fontsize=7, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points")
    # Theorem 1 bound: r >= -m_t / L. Overlay r = c / L for several c.
    L_grid = np.linspace(Ls.min() * 0.8, Ls.max() * 1.2, 100)
    for c, style in [(args.gamma, "-"), (args.gamma * 0.5, "--"), (args.gamma * 2.0, ":")]:
        ax.plot(L_grid, c / L_grid, style, alpha=0.4, color="grey",
                label=f"r = {c:.1f}/L (Thm 1, γ={c:.1f})")

    ax.set_xlabel("Lipschitz proxy L (sqrt of median input-Jacobian trace)")
    ax.set_ylabel("Median L_inf recovery radius  ($\\hat r$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Lipschitz-vs-recovery scatter (Thm 1 of theory_appendix.tex)")
    # Deduplicated legend
    handles, lbls = ax.get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, lbls):
        if l in seen: continue
        seen.add(l); h2.append(h); l2.append(l)
    ax.legend(h2, l2, fontsize=7, loc="best")
    fig.tight_layout()
    out_png = os.path.join(FIG_DIR, "lipschitz_radius_scatter.png")
    fig.savefig(out_png, dpi=160)
    print(f"Wrote {out_png}")

    # Print certifiable-eps_c upper bound for each row
    print("\nCorollary 1.1 certifiable-epsilon_c upper bound per row "
          "(assumes B_logits=10, gamma_oracle=γ):")
    for r in joined:
        eps_c = cert_budget_upper_bound(
            median_r=r["median_radius"],
            gamma_oracle=args.gamma,
            L_proxy=r["L_proxy"],
            B_logits=args.B_logits,
        )
        print(f"  {_short(r['suite']):12s} seed={r['seed']} "
              f"L={r['L_proxy']:.2f} median_r={r['median_radius']:.4f}  "
              f"=> max claimable eps_c = {eps_c:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", default=METRICS_DIR)
    ap.add_argument("--gamma", type=float, default=2.0,
                    help="Oracle expected margin separation (paper-default 2.0).")
    ap.add_argument("--B-logits", type=float, default=10.0,
                    help="Logit-magnitude bound for Cor 1.1.")
    args = ap.parse_args()
    join_and_plot(args)


if __name__ == "__main__":
    main()
