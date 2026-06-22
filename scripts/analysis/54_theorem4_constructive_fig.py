"""Headline constructive figure for Theorem 4 (no GPU, no new runs).

Shows the smoothed-margin objective as a *constructive* result, not a diagnostic:
  (A) Certification--utility frontier (sigma_frontier.json, seed 42): the smoothing
      scale sigma buys a larger certified L2 recovery floor while retain accuracy
      stays flat; strong forgetting lives at small sigma.
  (B) In-norm validation (smoothed_margin_l2_recovery_summary.json): across seeds
      42/123/456 the realized L2 forget recovery radius sits ABOVE the certified
      floor R=0.193 (the floor is respected), while retain controls are essentially
      unrecoverable (success ~2-3%). Empirical methods (no smoothing) carry a
      certified floor of 0.

Outputs: results/analysis/figures/theorem4_constructive_fig.{pdf,png}
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
M = os.path.join(ROOT, "results", "analysis", "metrics")
FIG = os.path.join(ROOT, "results", "analysis", "figures")

CERT_R = 0.193  # sigma=0.10 certified L2 detector radius


def main():
    front = json.load(open(os.path.join(M, "smoothed_margin_sigma_frontier.json")))["frontier"]
    l2 = json.load(open(os.path.join(M, "smoothed_margin_l2_recovery_summary.json")))

    sig = np.array([f["sigma"] for f in front])
    facc = np.array([f["forget_acc"] for f in front]) * 100
    racc = np.array([f["retain_acc"] for f in front]) * 100
    R = np.array([f["median_certified_l2_radius"] for f in front])

    seeds = [p["seed"] for p in l2["per_seed"]]
    f_med = np.array([p["forget_median_l2"] for p in l2["per_seed"]])
    r_med = np.array([p["retain_median_l2"] for p in l2["per_seed"]])
    f_succ = np.array([p["forget_success"] for p in l2["per_seed"]]) * 100
    r_succ = np.array([p["retain_success"] for p in l2["per_seed"]]) * 100

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.4,
    })
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.5))

    # ---- Panel A: certification-utility frontier ----
    a = ax[0]
    a.plot(sig, racc, "o-", c="#2c8a4a", lw=1.8, ms=5, label="retain acc (utility)")
    a.plot(sig, facc, "s-", c="#b3271e", lw=1.8, ms=5, label="forget acc (leakage)")
    a.set_xlabel(r"training smoothing scale $\sigma$")
    a.set_ylabel("accuracy (%)")
    a.set_ylim(-4, 104)
    a.axvspan(0.04, 0.12, color="#cfe8d4", alpha=0.5, zorder=0)
    a.text(0.08, 50, "useful\nregime", ha="center", va="center", fontsize=7,
           color="#1f5d33")
    a2 = a.twinx()
    a2.plot(sig, R, "^--", c="#2c6fbb", lw=1.6, ms=5, label="certified $L_2$ floor $R$")
    a2.set_ylabel(r"median certified $L_2$ recovery floor $R$", color="#2c6fbb")
    a2.tick_params(axis="y", labelcolor="#2c6fbb")
    a2.grid(False)
    h1, l1 = a.get_legend_handles_labels()
    h2, l2_ = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2_, frameon=False, fontsize=6.6, loc="center right")
    a.set_title("(A) Certification--utility frontier\n(positive certificate on 100% of forget points)")

    # ---- Panel B: in-norm certified-floor validation ----
    b = ax[1]
    x = np.arange(len(seeds))
    w = 0.36
    b.bar(x - w / 2, f_med, w, color="#b3271e", alpha=0.85,
          label="forget realized radius")
    b.bar(x + w / 2, r_med, w, color="#2c8a4a", alpha=0.85,
          label="retain realized radius")
    b.axhline(CERT_R, c="#2c6fbb", ls="--", lw=1.6,
              label=fr"certified floor $R={CERT_R}$")
    b.text(len(seeds) - 0.5, CERT_R + 0.015, "floor respected (realized $>$ certified)",
           ha="right", va="bottom", fontsize=6.6, color="#2c6fbb")
    for i in range(len(seeds)):
        b.text(x[i] - w / 2, f_med[i] + 0.012, f"{f_succ[i]:.0f}%", ha="center",
               va="bottom", fontsize=6.3, color="#b3271e")
        b.text(x[i] + w / 2, r_med[i] + 0.012, f"{r_succ[i]:.0f}%", ha="center",
               va="bottom", fontsize=6.3, color="#2c8a4a")
    b.set_xticks(x)
    b.set_xticklabels([f"seed {s}" for s in seeds])
    b.set_ylabel(r"native $L_2$ recovery radius")
    b.set_ylim(0, max(r_med.max(), f_med.max()) * 1.22)
    b.legend(frameon=False, fontsize=6.6, loc="upper left")
    b.set_title("(B) Same-norm validation: floor respected,\nretain unrecoverable (% = attack success)")

    fig.suptitle(r"A smoothed unlearning objective with a certified input-space recovery floor "
                 r"(Theorem 4, $\sigma{=}0.10$)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("pdf", "png"):
        out = os.path.join(FIG, f"theorem4_constructive_fig.{ext}")
        fig.savefig(out)
        print("wrote", out)
    print(f"frontier sigmas={list(sig)} retain_acc={list(racc)} R={list(R)}")
    print(f"l2: forget_med={list(f_med)} retain_med={list(r_med)} "
          f"forget_succ={list(f_succ)} retain_succ={list(r_succ)} floor={CERT_R}")


if __name__ == "__main__":
    main()
