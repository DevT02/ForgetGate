"""Cross-architecture check figure (no GPU, no new runs) -- an HONEST negative.

Reads the two-seed pool (crossarch_pooled_2seed.json) and shows that the
single-seed-per-cell agreement (ResNet-18 seed 42 + ViT-Small seed 42) does NOT
survive a second ResNet-18 seed:
  (A) log r_F vs log r_R: the single-seed pool fit (b=1.98) vs the full two-seed
      fit (b=0.51) and the single-scale null b=1.
  (B) selectivity S vs log r_R: single-seed pool (r=-0.74) vs full (r=+0.44).
The seed-123 ResNet-18 points (high r_R AND high selectivity) are the ones that
flip both fits -- marked distinctly.

Outputs: results/analysis/figures/crossarch_replication_fig.{pdf,png}
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
M = os.path.join(ROOT, "results", "analysis", "metrics")

# (arch, seed) -> (marker, color, label)
CELL_STYLE = {
    ("resnet18", 42):  ("o", "#2c6fbb", "ResNet-18 seed 42"),
    ("vit_small", 42): ("^", "#ff7f00", "ViT-Small seed 42"),
    ("resnet18", 123): ("X", "#b3271e", "ResNet-18 seed 123 (breaks it)"),
}


def ols(x, y):
    b, a = np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)
    return float(b), float(a)


def main():
    d = json.load(open(os.path.join(M, "crossarch_pooled_2seed.json")))
    pts = d["points"]
    lrR = np.array([p["logrR"] for p in pts])
    lrF = np.array([p["logrF"] for p in pts])
    S = np.array([p["S"] for p in pts])
    single = np.array([p["seed"] == 42 for p in pts])  # the original n=11 pool

    # fits: single-seed pool (seed 42 cells) vs full two-seed pool
    bA_s, aA_s = ols(lrR[single], lrF[single])
    bA_f, aA_f = ols(lrR, lrF)
    cB_s, c0_s = ols(lrR[single], S[single])
    cB_f, c0_f = ols(lrR, S)
    rB_s = float(np.corrcoef(lrR[single], S[single])[0, 1])
    rB_f = float(np.corrcoef(lrR, S)[0, 1])

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.4,
    })
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.5))
    xs = np.linspace(lrR.min(), lrR.max(), 50)

    # ---- Panel A: elasticity ----
    for (af, sd), (mk, col, lab) in CELL_STYLE.items():
        m = np.array([(p["arch"], p["seed"]) == (af, sd) for p in pts])
        ax[0].scatter(lrR[m], lrF[m], s=38, marker=mk, c=col, alpha=0.9,
                      edgecolors="none", label=lab)
    ax[0].plot(xs, aA_s + bA_s * xs, "-", c="#2c8a4a", lw=1.8,
               label=f"single-seed pool $b={bA_s:.2f}$")
    ax[0].plot(xs, aA_f + bA_f * xs, "-", c="#b3271e", lw=1.8,
               label=f"+2nd seed $b={bA_f:.2f}$")
    x0, y0 = lrR.mean(), lrF.mean()
    ax[0].plot(xs, y0 + 1.0 * (xs - x0), "--", c="0.4", lw=1.2, label="null $b=1$")
    ax[0].set_xlabel(r"$\log r_R$ (retain-control radius)")
    ax[0].set_ylabel(r"$\log r_F$ (forget radius)")
    ax[0].set_title("(A) Super-elasticity does not survive a 2nd seed")
    ax[0].legend(frameon=False, fontsize=6.3, loc="upper left")

    # ---- Panel B: confound ----
    for (af, sd), (mk, col, lab) in CELL_STYLE.items():
        m = np.array([(p["arch"], p["seed"]) == (af, sd) for p in pts])
        ax[1].scatter(lrR[m], S[m], s=38, marker=mk, c=col, alpha=0.9,
                      edgecolors="none", label=lab)
    ax[1].plot(xs, c0_s + cB_s * xs, "-", c="#2c8a4a", lw=1.8,
               label=f"single-seed $r={rB_s:.2f}$")
    ax[1].plot(xs, c0_f + cB_f * xs, "-", c="#b3271e", lw=1.8,
               label=f"+2nd seed $r={rB_f:.2f}$")
    ax[1].axhline(0, c="0.4", ls="--", lw=1.0)
    ax[1].set_xlabel(r"$\log r_R$ (fragility scale)")
    ax[1].set_ylabel(r"selectivity $S=\log(r_R/r_F)$")
    ax[1].set_title("(B) Confound flips sign with a 2nd seed")
    ax[1].legend(frameon=False, fontsize=6.3, loc="upper right")

    fig.suptitle("Cross-architecture check (CIFAR-100): single-seed agreement is "
                 "not robust", fontsize=9.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(ROOT, "results", "analysis", "figures",
                           f"crossarch_replication_fig.{ext}")
        fig.savefig(out)
        print(f"wrote {out}")
    print(f"single-seed: b={bA_s:.2f} r={rB_s:.2f}  |  +2nd seed: b={bA_f:.2f} r={rB_f:.2f}")


if __name__ == "__main__":
    main()
