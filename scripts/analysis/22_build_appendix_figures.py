from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def plot_variance():
    methods = ["Base", "CE Ascent", "Uniform KL", "SalUn", "SCRUB", "Retain-Only"]
    forget_std = [4.36, 46.56, 46.33, 50.47, 46.87, 33.75]
    retain_std = [3.77, 3.71, 3.90, 4.01, 3.71, 3.59]
    x = np.arange(len(methods))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - width / 2, forget_std, width, label="Forget std", color="#bc4b51")
    ax.bar(x + width / 2, retain_std, width, label="Retain std", color="#4a6fa5")
    ax.axhline(5.0, color="#444444", linestyle="--", linewidth=1.2, label="5pp threshold")
    ax.set_ylabel("Std. dev. across seeds (pp)")
    ax.set_title("Seed Variance of Clean Baselines")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "baseline_seed_variance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_kshot():
    ks = np.array([10, 25, 50, 100])
    oracle = np.array([0.0, 0.0, 0.0, 0.0])
    kl = np.array([0.47, 0.43, 0.43, 1.90])
    scrub = np.array([0.0, 0.0, 26.93, 28.20])

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharex=True)

    axes[0].plot(ks, oracle, marker="o", linewidth=2.0, color="#4a6fa5", label="Oracle")
    axes[0].plot(ks, kl, marker="o", linewidth=2.0, color="#bc4b51", label="KL")
    axes[0].set_title("Default Prompt Length")
    axes[0].set_ylabel("Resurrection rate (%)")
    axes[0].set_xlabel("K-shot")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(ks, oracle, marker="o", linewidth=2.0, color="#4a6fa5", label="Oracle")
    axes[1].plot(ks, scrub, marker="o", linewidth=2.0, color="#6c9a3b", label="SCRUB")
    axes[1].set_title("SCRUB Follow-up")
    axes[1].set_xlabel("K-shot")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle("Low-Shot and K-Shot Prompt Controls")
    fig.tight_layout()
    fig.savefig(OUT / "kshot_controls.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    plot_variance()
    plot_kshot()


if __name__ == "__main__":
    main()
