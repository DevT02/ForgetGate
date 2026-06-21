"""Publication figures from existing audit metrics (no GPU, no new runs).

Panel A  forget-radius super-elasticity: log r_F vs log r_R across 132 audited
         checkpoints, OLS fit (b=1.31) against the single-scale null (b=1).
Panel B  fragility confound: signed selectivity S vs log fragility, with trend.
Panel C  where the fragility scale lives: eta^2 of log r_R by base seed vs by
         unlearning method (base-geometry / architecture-level signal).

Outputs: results/analysis/figures/elasticity_confound_fig.{pdf,png}
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
M = os.path.join(ROOT, "results", "analysis", "metrics")


def ols(x, y):
    b, a = np.polyfit(x, y, 1)
    return float(b), float(a)


def main():
    frag = json.load(open(os.path.join(M, "fragility_confound.json")))
    rows = [r for r in frag["rows"] if r["fragility"] > 0 and r["median_forget"] > 0]
    logrR = np.array([np.log(r["fragility"]) for r in rows])
    logrF = np.array([np.log(r["median_forget"]) for r in rows])
    S = np.array([r["selectivity"] for r in rows])

    base = json.load(open(os.path.join(M, "base_geometry_signal.json")))
    ctrl = json.load(open(os.path.join(M, "base_robustness_elasticity.json")))

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.4,
    })
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.4))
    ax = axes.ravel()

    # ---- Panel A: elasticity ----
    b, a = ols(logrR, logrF)
    ax[0].scatter(logrR, logrF, s=14, c="#2c6fbb", alpha=0.6, edgecolors="none")
    xs = np.linspace(logrR.min(), logrR.max(), 50)
    ax[0].plot(xs, a + b * xs, "-", c="#b3271e", lw=1.8,
               label=f"fit $b={b:.2f}$")
    # single-scale null b=1 through the data centroid
    x0, y0 = logrR.mean(), logrF.mean()
    ax[0].plot(xs, y0 + 1.0 * (xs - x0), "--", c="0.4", lw=1.3,
               label="single-scale null $b=1$")
    ax[0].set_xlabel(r"$\log r_R$ (retain-control radius)")
    ax[0].set_ylabel(r"$\log r_F$ (forget radius)")
    ax[0].set_title("(A) Forget-radius super-elasticity")
    ax[0].legend(frameon=False, fontsize=7.5, loc="upper left")

    # ---- Panel B: fragility confound ----
    c, c0 = ols(logrR, S)
    pear = float(np.corrcoef(logrR, S)[0, 1])
    ax[1].scatter(logrR, S, s=14, c="#2c8a4a", alpha=0.6, edgecolors="none")
    ax[1].plot(xs, c0 + c * xs, "-", c="#b3271e", lw=1.8,
               label=f"slope $={c:.2f}$, $r={pear:.2f}$")
    ax[1].axhline(0, c="0.4", ls="--", lw=1.0)
    ax[1].set_xlabel(r"$\log r_R$ (fragility scale)")
    ax[1].set_ylabel(r"selectivity $S=\log(r_R/r_F)$")
    ax[1].set_title("(B) Fragility confound")
    ax[1].legend(frameon=False, fontsize=7.5, loc="upper right")

    # ---- Panel C: where the scale lives ----
    labels = ["base seed\n(init)", "method\n(objective)", "adapter reg.\n(weight decay)"]
    vals = [base["eta2_logrR_by_base_seed"], base["eta2_logrR_by_method"], 0.0]
    colors = ["#888888", "#2c6fbb", "#cccccc"]
    ax[2].bar(labels, vals, color=colors, width=0.62)
    for i, v in enumerate(vals):
        ax[2].text(i, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax[2].set_ylabel(r"$\eta^2$ of $\log r_R$ explained")
    ax[2].set_ylim(0, max(vals) * 1.25 + 0.02)
    ax[2].set_title("(C) Recovery scale is architecture-level")

    # ---- Panel D: controlled causal elasticity (PGD-base instrument) ----
    pts = ctrl["points"]
    lrR = np.array([np.log(p["rR_mean"]) for p in pts])
    lrF = np.array([np.log(p["rF_mean"]) for p in pts])
    seeds = np.array([p["seed"] for p in pts])
    bc = ctrl["structural_b"]; cic = ctrl["structural_b_ci95"]
    for sd, mk, col in [(42, "o", "#6a3d9a"), (123, "^", "#ff7f00")]:
        m = seeds == sd
        ax[3].scatter(lrR[m], lrF[m], s=26, marker=mk, c=col, alpha=0.85,
                      edgecolors="none", label=f"seed {sd}")
    xs = np.linspace(lrR.min(), lrR.max(), 50)
    a_c = float(ctrl["intercept"])
    ax[3].plot(xs, a_c + bc * xs, "-", c="#b3271e", lw=1.8,
               label=f"causal $b={bc:.2f}$\n[{cic[0]:.2f},{cic[1]:.2f}]")
    x0, y0 = lrR.mean(), lrF.mean()
    ax[3].plot(xs, y0 + 1.0 * (xs - x0), "--", c="0.4", lw=1.3, label="null $b=1$")
    ax[3].set_xlabel(r"$\log r_R$ (base smoothness knob)")
    ax[3].set_ylabel(r"$\log r_F$ (forget radius)")
    ax[3].set_title("(D) Controlled causal elasticity")
    ax[3].legend(frameon=False, fontsize=7, loc="upper left")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(ROOT, "results", "analysis", "figures",
                           f"elasticity_confound_fig.{ext}")
        fig.savefig(out)
        print(f"wrote {out}")
    print(f"Panel A: b={b:.3f}  Panel B: r={pear:.3f} slope={c:.3f}  "
          f"Panel C: seed={vals[0]:.3f} method={vals[1]:.3f}")


if __name__ == "__main__":
    main()
