"""Cross-architecture replication figure (no GPU, no new runs).

Reads the pooled ResNet-18 + ViT-Small / CIFAR-100 cross-method audit
(crossarch_pooled.json) and shows that both halves of the mechanism replicate
on two NEW architectures:
  (A) forget-radius super-elasticity: log r_F vs log r_R, OLS fit (b>1) vs the
      single-scale null b=1, points marked by architecture.
  (B) fragility confound: selectivity S = log(r_R/r_F) vs log r_R, trend r<0.

Outputs: results/analysis/figures/crossarch_replication_fig.{pdf,png}
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
M = os.path.join(ROOT, "results", "analysis", "metrics")

ARCH_STYLE = {
    "resnet18":  ("o", "#2c6fbb", "ResNet-18 (CNN)"),
    "vit_small": ("^", "#ff7f00", "ViT-Small"),
}


def ols(x, y):
    b, a = np.polyfit(x, y, 1)
    return float(b), float(a)


def main():
    d = json.load(open(os.path.join(M, "crossarch_pooled.json")))
    pts = d["points"]
    lrR = np.array([p["logrR"] for p in pts])
    lrF = np.array([p["logrF"] for p in pts])
    S = np.array([p["S"] for p in pts])
    arch = np.array([p["arch"] for p in pts])

    b, a = d["elasticity_b"], None
    b, a = ols(lrR, lrF)
    cr = d["confound_r"]
    cslope, c0 = ols(lrR, S)
    b_ci = d["elasticity_b_ci95"]
    r_ci = d["confound_r_ci95"]

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.4,
    })
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.4))

    # ---- Panel A: elasticity replicates ----
    for af, (mk, col, lab) in ARCH_STYLE.items():
        m = arch == af
        ax[0].scatter(lrR[m], lrF[m], s=34, marker=mk, c=col, alpha=0.85,
                      edgecolors="none", label=lab)
    xs = np.linspace(lrR.min(), lrR.max(), 50)
    ax[0].plot(xs, a + b * xs, "-", c="#b3271e", lw=1.8,
               label=f"pooled fit $b={b:.2f}$\n95% CI [{b_ci[0]:.2f},{b_ci[1]:.2f}]")
    x0, y0 = lrR.mean(), lrF.mean()
    ax[0].plot(xs, y0 + 1.0 * (xs - x0), "--", c="0.4", lw=1.3,
               label="single-scale null $b=1$")
    ax[0].set_xlabel(r"$\log r_R$ (retain-control radius)")
    ax[0].set_ylabel(r"$\log r_F$ (forget radius)")
    ax[0].set_title("(A) Super-elasticity replicates")
    ax[0].legend(frameon=False, fontsize=7, loc="upper left")

    # ---- Panel B: confound replicates ----
    for af, (mk, col, lab) in ARCH_STYLE.items():
        m = arch == af
        ax[1].scatter(lrR[m], S[m], s=34, marker=mk, c=col, alpha=0.85,
                      edgecolors="none", label=lab)
    ax[1].plot(xs, c0 + cslope * xs, "-", c="#b3271e", lw=1.8,
               label=f"$r={cr:.2f}$\n95% CI [{r_ci[0]:.2f},{r_ci[1]:.2f}]")
    ax[1].axhline(0, c="0.4", ls="--", lw=1.0)
    ax[1].set_xlabel(r"$\log r_R$ (fragility scale)")
    ax[1].set_ylabel(r"selectivity $S=\log(r_R/r_F)$")
    ax[1].set_title("(B) Fragility confound replicates")
    ax[1].legend(frameon=False, fontsize=7, loc="upper right")

    fig.suptitle("Cross-architecture replication: ResNet-18 + ViT-Small / CIFAR-100 "
                 f"($n={d['n_points']}$, 6 LoRA methods)", fontsize=9.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(ROOT, "results", "analysis", "figures",
                           f"crossarch_replication_fig.{ext}")
        fig.savefig(out)
        print(f"wrote {out}")
    print(f"Panel A: pooled b={b:.3f}  Panel B: r={cr:.3f} slope={cslope:.3f}")


if __name__ == "__main__":
    main()
