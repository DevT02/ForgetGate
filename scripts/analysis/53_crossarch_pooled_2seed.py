"""Two-seed, two-architecture pooled cross-replication. Extends script 51 by
pooling the ResNet-18/CIFAR-100 cross-method audits over BOTH seeds (42 and 123)
together with the ViT-Small/CIFAR-100 seed-42 audits, and re-tests the
forget-radius super-elasticity (b>1) and the fragility confound (r<0). Pooling a
second seed on the CNN converts the earlier "single seed per cell" caveat into a
replication across seeds AND architectures. Leave-one-out and a case-resample
bootstrap are reported honestly (still modest n, observational cross-method).

Output: results/analysis/metrics/crossarch_pooled_2seed.json
"""
import json, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
METHODS = ["salun", "scrub", "ceascent", "uniformkl", "featurescrub", "smoothedmargin"]
# (arch_file, arch_tag, [seeds available])
CELLS = [("resnet18", "resnet18", [42, 123]), ("vit_small", "vit-small", [42])]


def load(method, arch_file, arch_tag, seed):
    tag = f"{method}-{arch_tag}-ca-forget0"
    p = os.path.join(ROOT, "results", "analysis",
                     f"recovery_radius_cifar100_{arch_file}_forget0_seed_{seed}"
                     f"_test_with_retain_adam_margin_{tag}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    fr = list(d["forget_recovery"].values())[0]["summary"]
    rc = list(d["retain_control"].values())[0]["summary"]
    rF, rR = fr["median_radius"], rc["median_radius"]
    if not rF or not rR or rF <= 0 or rR <= 0:
        return None
    return {"method": method, "arch": arch_file, "seed": seed,
            "rR": rR, "rF": rF, "logrR": float(np.log(rR)),
            "logrF": float(np.log(rF)), "S": float(np.log(rR / rF))}


def ols(x, y):
    b, a = np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)
    return float(b), float(a)


def pear(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")


def main():
    pts = []
    for af, at, seeds in CELLS:
        for sd in seeds:
            for m in METHODS:
                r = load(m, af, at, sd)
                if r:
                    pts.append(r)
    cells = sorted(set((p["arch"], p["seed"]) for p in pts))
    print(f"pooled {len(pts)} points over {len(cells)} (arch,seed) cells: {cells}")
    for p in pts:
        print(f"  {p['arch']:<10} s{p['seed']:<4} {p['method']:<14} "
              f"r_R={p['rR']:.5f} r_F={p['rF']:.5f} S={p['S']:+.3f}")
    if len(pts) < 8:
        json.dump({"status": "INCOMPLETE", "n": len(pts), "points": pts},
                  open(os.path.join(ROOT, "results", "analysis", "metrics",
                                    "crossarch_pooled_2seed.json"), "w"), indent=2)
        print("incomplete -- run the seed-123 resnet18 batch first"); return

    lrR = np.array([p["logrR"] for p in pts])
    lrF = np.array([p["logrF"] for p in pts])
    S = np.array([p["S"] for p in pts])
    cid = np.array([cells.index((p["arch"], p["seed"])) for p in pts])

    # --- (1) NAIVE raw pool: regresses pooled logs across cells of different
    # fragility baseline. This commits the ecological/Simpson fallacy when the
    # cells differ in overall scale (and here one cell is half left-censored), so
    # it is reported only for transparency, NOT as the headline estimator. ---
    b_raw, a_raw = ols(lrR, lrF)
    cr_raw = pear(lrR, S)

    # --- (2) WITHIN-CELL FIXED EFFECTS: demean each variable inside its
    # (arch,seed) cell, then pool. This removes the cell-baseline confound and is
    # the correct way to combine within-cell slopes -- the "fragility-matched"
    # pooling. This is the headline estimator. ---
    def demean(v):
        out = v.astype(float).copy()
        for c in range(len(cells)):
            m = cid == c
            out[m] = v[m] - v[m].mean()
        return out
    lrRd, lrFd, Sd = demean(lrR), demean(lrF), demean(S)
    b, a = ols(lrRd, lrFd)
    cr = pear(lrRd, Sd)

    def fe(idx):
        rr, ff, ss, cc = lrR[idx], lrF[idx], S[idx], cid[idx]
        rrd = rr.copy().astype(float); ffd = ff.copy().astype(float); ssd = ss.copy().astype(float)
        for c in set(cc.tolist()):
            m = cc == c
            rrd[m] -= rr[m].mean(); ffd[m] -= ff[m].mean(); ssd[m] -= ss[m].mean()
        return ols(rrd, ffd)[0], pear(rrd, ssd)

    # leave-one-point-out on the FE estimator
    loo_b, loo_r = [], []
    for i in range(len(pts)):
        keep = np.arange(len(pts)) != i
        bb_i, rr_i = fe(keep)
        loo_b.append(bb_i); loo_r.append(rr_i)
    # cluster bootstrap: resample points WITHIN each cell, preserving structure
    rng = np.random.default_rng(0)
    bb, br = [], []
    for _ in range(10000):
        idx = []
        for c in range(len(cells)):
            pool = np.where(cid == c)[0]
            idx += list(rng.choice(pool, len(pool), replace=True))
        idx = np.array(idx)
        b_i, r_i = fe(idx)
        if np.isfinite(b_i) and np.isfinite(r_i):
            bb.append(b_i); br.append(r_i)
    b_ci = [float(np.nanpercentile(bb, 2.5)), float(np.nanpercentile(bb, 97.5))]
    r_ci = [float(np.nanpercentile(br, 2.5)), float(np.nanpercentile(br, 97.5))]
    p_b_gt1 = float(np.mean(np.asarray(bb) > 1))
    p_r_lt0 = float(np.mean(np.asarray(br) < 0))

    # per-cell within-cell slope / confound (n>=3 cells only)
    per_cell = []
    for c, (af_, sd_) in enumerate(cells):
        idx = np.where(cid == c)[0]
        if len(idx) >= 3:
            per_cell.append({"arch": af_, "seed": sd_, "n": int(len(idx)),
                             "b": ols(lrR[idx], lrF[idx])[0],
                             "confound_r": pear(lrR[idx], S[idx])})

    out = {"n_points": len(pts), "n_cells": len(cells),
           "cells": [{"arch": a_, "seed": s_} for a_, s_ in cells],
           "n_seeds": len(set(p["seed"] for p in pts)),
           "n_architectures": len(set(p["arch"] for p in pts)),
           "estimator": "within_cell_fixed_effects",
           "elasticity_b": b, "elasticity_b_ci95": b_ci,
           "elasticity_b_loo_range": [float(min(loo_b)), float(max(loo_b))],
           "elasticity_p_b_gt_1": p_b_gt1,
           "confound_r": cr, "confound_r_ci95": r_ci,
           "confound_r_loo_range": [float(np.nanmin(loo_r)), float(np.nanmax(loo_r))],
           "confound_p_r_lt_0": p_r_lt0,
           "naive_raw_pool": {"elasticity_b": b_raw, "confound_r": cr_raw,
                              "note": "ecological/Simpson confound + left-censored cell; not the headline"},
           "per_cell": per_cell,
           "benchmark_b": 1.31, "benchmark_confound_r": -0.25,
           "points": pts,
           "note": ("observational cross-method; resnet18 seeds 42+123, vit_small seed 42. "
                    "Headline = within-cell fixed-effects (fragility-matched pooling). "
                    "resnet18 seed-123 cell is half left-censored (rF=0 for salun/uniformkl/"
                    "smoothedmargin: forget class leaks on clean input), so it contributes "
                    "only 3 high-radius survivors -- disclosed, not hidden.")}
    json.dump(out, open(os.path.join(ROOT, "results", "analysis", "metrics",
                        "crossarch_pooled_2seed.json"), "w"), indent=2)
    print(f"\nNAIVE RAW   b = {b_raw:.3f}   confound_r = {cr_raw:.3f}   (confounded; transparency only)")
    print(f"FE (headline) ELASTICITY  b = {b:.3f}  CI95 [{b_ci[0]:.2f},{b_ci[1]:.2f}]  "
          f"LOO [{min(loo_b):.2f},{max(loo_b):.2f}]  P(b>1)={p_b_gt1:.2f}  "
          f"(b>1 all LOO: {'YES' if min(loo_b) > 1 else 'no'})")
    print(f"FE (headline) CONFOUND    r = {cr:.3f}  CI95 [{r_ci[0]:.2f},{r_ci[1]:.2f}]  "
          f"LOO [{np.nanmin(loo_r):.2f},{np.nanmax(loo_r):.2f}]  P(r<0)={p_r_lt0:.2f}  "
          f"(r<0 all LOO: {'YES' if np.nanmax(loo_r) < 0 else 'no'})")
    for pc in per_cell:
        print(f"  per-cell {pc['arch']:<10} s{pc['seed']:<4} n={pc['n']}  "
              f"b={pc['b']:+.3f}  r={pc['confound_r']:+.3f}")


if __name__ == "__main__":
    main()
