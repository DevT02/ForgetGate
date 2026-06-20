"""
Controlled within-method elasticity: robust-base SalUn frontier.

The benchmark-wide elasticity (script 41, b=1.31) is observational -- it pools
different methods/seeds. Here is a CONTROLLED test that isolates the global
fragility knob while holding the method fixed: SalUn unlearning applied to the
SAME architecture trained at increasing adversarial-robustness levels
(smoke < mid < high < ultra). Dialing base robustness moves the global recovery
scale without changing the unlearning method, so the within-frontier slope of
log r_F on log r_R is a causal estimate of the forget-radius elasticity.

If b > 1 holds within this single-method controlled family, the super-elasticity
(forget class collapses faster than retain as the model gets globally more
fragile) is not a cross-method artifact.

Output: results/analysis/metrics/controlled_elasticity.json
"""
import json
import os
import glob
import re
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RNG = np.random.default_rng(3)
LEVEL_ORDER = {"smoke": 0, "mid": 1, "high": 2, "ultra": 3}


def med(rec):
    ps = rec.get("per_sample", [])
    r = [s["radius"] for s in ps if s.get("success") and s.get("radius")]
    return float(np.median(r)) if len(r) >= 5 else None


def main():
    files = glob.glob(os.path.join(
        ROOT, "results", "analysis", "metrics",
        "recovery_radius_*with_retain*salun-robust-base*.json"))
    pts = []
    for f in files:
        name = os.path.basename(f)
        seed = re.search(r"seed_(\d+)_", name).group(1)
        attack = "apgd" if "apgd" in name else "adam"
        lvl = re.search(r"forget0-(smoke|mid|high|ultra)", name).group(1)
        d = json.load(open(f))
        fr, rc = d["forget_recovery"], d["retain_control"]
        for suite in set(fr) & set(rc):
            mf, mr = med(fr[suite]), med(rc[suite])
            if mf and mr and mf > 0 and mr > 0:
                pts.append({"seed": seed, "attack": attack, "level": lvl,
                            "level_rank": LEVEL_ORDER[lvl],
                            "median_forget": mf, "median_retain": mr,
                            "logF": float(np.log(mf)),
                            "logR": float(np.log(mr)),
                            "selectivity": float(np.log(mr / mf))})

    # dedup
    seen = {}
    for p in pts:
        seen[(p["seed"], p["attack"], p["level"])] = p
    pts = list(seen.values())

    if len(pts) < 4:
        out = {"n_points": len(pts),
               "result": "UNDERPOWERED: robust-base frontier is too censored "
               "at high robustness to yield matched forget/retain medians "
               "(>=5 successful recoveries) -- controlled per-method elasticity "
               "needs a less-censored fragility sweep.",
               "usable_points": [(p["seed"], p["attack"], p["level"])
                                 for p in pts]}
        outp = os.path.join(ROOT, "results", "analysis", "metrics",
                            "controlled_elasticity.json")
        json.dump(out, open(outp, "w"), indent=2)
        print(f"controlled robust-base frontier: only {len(pts)} usable points "
              f"(too censored). {out['result']}")
        print(f"wrote {outp}")
        return

    x = np.array([p["logR"] for p in pts])
    y = np.array([p["logF"] for p in pts])
    b, a = np.polyfit(x, y, 1)

    # bootstrap CI for the controlled slope
    bs = []
    for _ in range(10000):
        j = RNG.choice(len(pts), len(pts), replace=True)
        if np.std(x[j]) > 0:
            bs.append(np.polyfit(x[j], y[j], 1)[0])
    b_ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))

    # within each (seed, attack) frontier separately -- strictest control
    sub = {}
    for key in {(p["seed"], p["attack"]) for p in pts}:
        g = sorted([p for p in pts if (p["seed"], p["attack"]) == key],
                   key=lambda p: p["level_rank"])
        if len(g) >= 3:
            gx = np.array([p["logR"] for p in g])
            gy = np.array([p["logF"] for p in g])
            if np.std(gx) > 0:
                sb = float(np.polyfit(gx, gy, 1)[0])
                sub["%s_%s" % key] = {
                    "n_levels": len(g),
                    "b": sb,
                    "levels": [(p["level"], round(p["median_forget"], 5),
                                round(p["median_retain"], 5)) for p in g],
                }

    out = {
        "n_points": len(pts),
        "pooled_controlled_slope_b": float(b),
        "b_ci95": b_ci,
        "intercept": float(a),
        "per_frontier": sub,
        "b_gt_1_all_frontiers": all(v["b"] > 1 for v in sub.values()),
    }
    outp = os.path.join(ROOT, "results", "analysis", "metrics",
                        "controlled_elasticity.json")
    json.dump(out, open(outp, "w"), indent=2)

    print(f"controlled robust-base SalUn frontier: {len(pts)} points")
    print(f"pooled controlled elasticity b = {b:.3f}  CI95 [{b_ci[0]:.3f},{b_ci[1]:.3f}]")
    print("per (seed,attack) frontier slopes:")
    for k, v in sub.items():
        levs = " ".join(f"{l}:rF{rf}/rR{rr}" for l, rf, rr in v["levels"])
        print(f"  {k}: b={v['b']:+.2f}  ({v['n_levels']} levels)  {levs}")
    print(f"b>1 on all frontiers: {out['b_gt_1_all_frontiers']}")
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
