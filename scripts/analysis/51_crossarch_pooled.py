"""Pooled cross-ARCHITECTURE replication: combine the ResNet18/CIFAR-100 and
ViT-small/CIFAR-100 cross-method audits (six unlearning methods each) and test
whether the forget-radius super-elasticity (b>1) and the fragility confound
(selectivity falls with fragility) hold across architectures. Pooling two new
architectures (one CNN, one ViT) gives n=12, with leave-one-out and a
case-resample bootstrap reported honestly (still modest n).

Output: results/analysis/metrics/crossarch_pooled.json
"""
import json, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
METHODS = ["salun", "scrub", "ceascent", "uniformkl", "featurescrub", "smoothedmargin"]
ARCHS = [("resnet18", "resnet18"), ("vit_small", "vit-small")]


def load(method, arch_file, arch_tag):
    tag = f"{method}-{arch_tag}-ca-forget0"
    p = os.path.join(ROOT, "results", "analysis",
                     f"recovery_radius_cifar100_{arch_file}_forget0_seed_42"
                     f"_test_with_retain_adam_margin_{tag}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    fr = list(d["forget_recovery"].values())[0]["summary"]
    rc = list(d["retain_control"].values())[0]["summary"]
    rF, rR = fr["median_radius"], rc["median_radius"]
    if not rF or not rR or rF <= 0 or rR <= 0:
        return None
    return {"method": method, "arch": arch_file,
            "rR": rR, "rF": rF, "logrR": float(np.log(rR)),
            "logrF": float(np.log(rF)), "S": float(np.log(rR / rF))}


def ols(x, y):
    b, a = np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)
    return float(b), float(a)


def pear(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")


def main():
    pts = []
    for af, at in ARCHS:
        for m in METHODS:
            r = load(m, af, at)
            if r:
                pts.append(r)
    n_arch = len(set(p["arch"] for p in pts))
    print(f"pooled {len(pts)} points across {n_arch} architecture(s)")
    for p in pts:
        print(f"  {p['arch']:<10} {p['method']:<14} r_R={p['rR']:.5f} "
              f"r_F={p['rF']:.5f} S={p['S']:+.3f}")
    if len(pts) < 6:
        json.dump({"status": "INCOMPLETE", "n": len(pts), "points": pts},
                  open(os.path.join(ROOT, "results", "analysis", "metrics",
                                    "crossarch_pooled.json"), "w"), indent=2)
        print("incomplete -- run both arch batches first"); return

    lrR = np.array([p["logrR"] for p in pts])
    lrF = np.array([p["logrF"] for p in pts])
    S = np.array([p["S"] for p in pts])
    b, a = ols(lrR, lrF)
    cr = pear(lrR, S)
    loo_b = [ols(np.delete(lrR, i), np.delete(lrF, i))[0] for i in range(len(pts))]
    loo_r = [pear(np.delete(lrR, i), np.delete(S, i)) for i in range(len(pts))]
    rng = np.random.default_rng(0)
    bb, br = [], []
    for _ in range(10000):
        j = rng.choice(len(pts), len(pts), replace=True)
        if np.std(lrR[j]) > 0:
            bb.append(ols(lrR[j], lrF[j])[0])
            br.append(pear(lrR[j], S[j]))
    b_ci = [float(np.nanpercentile(bb, 2.5)), float(np.nanpercentile(bb, 97.5))]
    r_ci = [float(np.nanpercentile(br, 2.5)), float(np.nanpercentile(br, 97.5))]

    out = {"n_points": len(pts), "n_architectures": n_arch,
           "elasticity_b": b, "elasticity_b_ci95": b_ci,
           "elasticity_b_loo_range": [float(min(loo_b)), float(max(loo_b))],
           "confound_r": cr, "confound_r_ci95": r_ci,
           "confound_r_loo_range": [float(np.nanmin(loo_r)), float(np.nanmax(loo_r))],
           "benchmark_b": 1.31, "benchmark_confound_r": -0.25,
           "points": pts,
           "note": "observational, n=12 over 2 new architectures, single seed each"}
    json.dump(out, open(os.path.join(ROOT, "results", "analysis", "metrics",
                        "crossarch_pooled.json"), "w"), indent=2)
    print(f"\nELASTICITY  b = {b:.3f}  CI95 [{b_ci[0]:.2f},{b_ci[1]:.2f}]  "
          f"LOO [{min(loo_b):.2f},{max(loo_b):.2f}]  (b>1 all LOO: "
          f"{'YES' if min(loo_b) > 1 else 'no'})")
    print(f"CONFOUND    r = {cr:.3f}  CI95 [{r_ci[0]:.2f},{r_ci[1]:.2f}]  "
          f"LOO [{np.nanmin(loo_r):.2f},{np.nanmax(loo_r):.2f}]  (r<0 all LOO: "
          f"{'YES' if np.nanmax(loo_r) < 0 else 'no'})")


if __name__ == "__main__":
    main()
