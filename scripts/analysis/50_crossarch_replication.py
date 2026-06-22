"""Cross-architecture replication: do the fragility confound and forget-radius
super-elasticity hold on ResNet18 / CIFAR-100 (vs the ViT-tiny / CIFAR-10
benchmark)? Reads the LoRA-unlearned method audits on ResNet18 and computes,
across methods:
  - selectivity S = log(r_R / r_F) and fragility F = r_R per method,
  - elasticity: OLS log r_F on log r_R (super-elastic if slope b > 1),
  - confound: Pearson r of S vs log F (negative on the benchmark).
This is the OBSERVATIONAL cross-method test (no PGD-adversarial ResNet bases
exist for the causal instrument). n is small (one method per point), so this is
a directional generalization check, reported with that caveat.

Output: results/analysis/metrics/crossarch_replication.json
"""
import json, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
METHODS = ["salun", "scrub", "ceascent", "uniformkl", "featurescrub", "smoothedmargin"]


def load(method):
    tag = f"{method}-resnet18-ca-forget0"
    p = os.path.join(ROOT, "results", "analysis",
                     f"recovery_radius_cifar100_resnet18_forget0_seed_42"
                     f"_test_with_retain_adam_margin_{tag}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    fr = list(d["forget_recovery"].values())[0]["summary"]
    rc = list(d["retain_control"].values())[0]["summary"]
    rF, rR = fr["median_radius"], rc["median_radius"]
    if rF is None or rR is None or rF <= 0 or rR <= 0:
        return {"method": method, "rR": rR, "rF": rF, "censored": True}
    return {"method": method, "rR": rR, "rF": rF,
            "S": float(np.log(rR / rF)), "logF": float(np.log(rR)),
            "censored": False}


def ols(x, y):
    b, a = np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)
    return float(b), float(a)


def main():
    pts = [p for m in METHODS if (p := load(m)) is not None]
    usable = [p for p in pts if not p.get("censored")]
    print(f"loaded {len(pts)}/{len(METHODS)} methods ({len(usable)} usable)")
    for p in pts:
        if p.get("censored"):
            print(f"  {p['method']:<14} CENSORED (rR={p['rR']}, rF={p['rF']})")
        else:
            print(f"  {p['method']:<14} r_R={p['rR']:.5f} r_F={p['rF']:.5f} "
                  f"S={p['S']:+.3f}")

    out = {"n_methods": len(usable), "points": pts,
           "dataset": "cifar100", "architecture": "resnet18"}
    if len(usable) >= 4:
        lrR = np.array([p["logF"] for p in usable])
        lrF = np.array([np.log(p["rF"]) for p in usable])
        S = np.array([p["S"] for p in usable])
        b, a = ols(lrR, lrF)
        conf_r = float(np.corrcoef(lrR, S)[0, 1]) if np.std(S) > 0 else float("nan")
        # leave-one-out robustness (honest for small n): does b>1 and r<0 hold?
        loo_b, loo_r = [], []
        for i in range(len(usable)):
            m = np.arange(len(usable)) != i
            loo_b.append(ols(lrR[m], lrF[m])[0])
            if np.std(S[m]) > 0:
                loo_r.append(float(np.corrcoef(lrR[m], S[m])[0, 1]))
        # case-resample bootstrap CI (wide at n=6; reported honestly)
        rng = np.random.default_rng(0)
        bs_b, bs_r = [], []
        for _ in range(10000):
            j = rng.choice(len(usable), len(usable), replace=True)
            if np.std(lrR[j]) > 0:
                bs_b.append(ols(lrR[j], lrF[j])[0])
                if np.std(S[j]) > 0:
                    bs_r.append(float(np.corrcoef(lrR[j], S[j])[0, 1]))
        b_ci = [float(np.percentile(bs_b, 2.5)), float(np.percentile(bs_b, 97.5))]
        r_ci = [float(np.percentile(bs_r, 2.5)), float(np.percentile(bs_r, 97.5))]
        out.update({
            "elasticity_b": b, "elasticity_intercept": a,
            "elasticity_b_ci95": b_ci,
            "elasticity_b_loo_range": [float(min(loo_b)), float(max(loo_b))],
            "confound_pearson_S_vs_logF": conf_r,
            "confound_r_ci95": r_ci,
            "confound_r_loo_range": [float(min(loo_r)), float(max(loo_r))],
            "benchmark_b": 1.31, "benchmark_confound_r": -0.25,
            "note": "observational cross-method, n=6 single seed; directional replication check",
        })
        print(f"\nELASTICITY  b = {b:.3f}  CI95 {b_ci}  LOO range "
              f"[{min(loo_b):.2f},{max(loo_b):.2f}]  (benchmark 1.31; b>1 all LOO: "
              f"{'YES' if min(loo_b) > 1 else 'no'})")
        print(f"CONFOUND    r(S,logF) = {conf_r:.3f}  CI95 {r_ci}  LOO range "
              f"[{min(loo_r):.2f},{max(loo_r):.2f}]  (benchmark -0.25; r<0 all LOO: "
              f"{'YES' if max(loo_r) < 0 else 'no'})")
    else:
        out["status"] = "UNDERPOWERED"
        print(f"\nonly {len(usable)} usable -- need >=4 for a fit")

    outp = os.path.join(ROOT, "results", "analysis", "metrics",
                        "crossarch_replication.json")
    json.dump(out, open(outp, "w"), indent=2)
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
