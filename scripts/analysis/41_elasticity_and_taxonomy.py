"""
Two deeper questions raised by the fragility confound (script 40).

(A) MECHANISM. Under Theorem 1, r ~ -margin/L, so the Lipschitz scale L cancels
    in the ratio r_R/r_F and the naive theory predicts NO fragility confound.
    The confound exists, so forget and retain radii must respond *differently*
    to the global scale. Test it directly: fit
        log r_F = a + b * log r_R
    across checkpoints. b > 1 means the forget-class radius is MORE elastic to
    global fragility than the retain radius -- as a checkpoint gets brittle, the
    forget class collapses faster than retain controls, which is exactly what
    inflates apparent selectivity. (Algebraically corr(S, log r_R) < 0 iff
    b > 1, where S = log r_R - log r_F; we confirm and quantify b with a
    suite-clustered bootstrap CI.)

(B) FRAGILITY-MATCHED TAXONOMY. Residualize selectivity S on log fragility,
    S = a + c * log F + e; the residual e is fragility-adjusted selectivity.
    Average e per method family to rank which methods are selective BEYOND what
    their global brittleness explains -- the genuinely-leaky ones.

Output: results/analysis/metrics/elasticity_taxonomy.json
"""
import json
import os
import re
import numpy as np
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RNG = np.random.default_rng(0)

FAMILIES = ["orbit", "scrub", "rurk", "salun", "ceu", "baldro", "sga",
            "smoothed_margin", "kl", "oracle", "falw", "orthoreg", "prune"]


def family(suite):
    s = suite.lower()
    for fam in FAMILIES:
        if fam in s:
            return fam
    return "other"


def ols(x, y):
    """slope, intercept via least squares."""
    x = np.asarray(x); y = np.asarray(y)
    b, a = np.polyfit(x, y, 1)
    return float(b), float(a)


def main():
    d = json.load(open(os.path.join(ROOT, "results", "analysis", "metrics",
                                     "fragility_confound.json")))
    rows = d["rows"]
    for r in rows:
        r["family"] = family(r["suite"])
        r["logF"] = float(np.log(r["fragility"]))
        r["logrf"] = float(np.log(r["median_forget"]))

    # ---- (A) elasticity ----
    logrR = np.array([r["logF"] for r in rows])       # log retain radius
    logrF = np.array([r["logrf"] for r in rows])      # log forget radius
    b_hat, a_hat = ols(logrR, logrF)

    # suite-clustered bootstrap for the slope b
    by = defaultdict(list)
    for r in rows:
        by[r["suite"]].append(r)
    keys = list(by)
    bs = []
    for _ in range(10000):
        pick = RNG.choice(len(keys), len(keys), replace=True)
        xs, ys = [], []
        for i in pick:
            for r in by[keys[i]]:
                xs.append(r["logF"]); ys.append(r["logrf"])
        if len(xs) > 5 and np.std(xs) > 0:
            bb, _ = ols(xs, ys)
            bs.append(bb)
    b_ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))

    # ---- (B) fragility-matched taxonomy ----
    S = np.array([r["selectivity"] for r in rows])
    c_hat, c0 = ols(logrR, S)               # selectivity ~ log fragility
    resid = S - (c0 + c_hat * logrR)        # fragility-adjusted selectivity
    for r, e in zip(rows, resid):
        r["adj_selectivity"] = float(e)

    fam_stat = {}
    famgroups = defaultdict(list)
    for r in rows:
        famgroups[r["family"]].append(r)
    for fam, rs in famgroups.items():
        if len(rs) < 2:
            continue
        raw = np.array([x["selectivity"] for x in rs])
        adj = np.array([x["adj_selectivity"] for x in rs])
        # bootstrap CI on the adjusted-selectivity mean (resample rows in family)
        means = []
        for _ in range(10000):
            j = RNG.choice(len(adj), len(adj), replace=True)
            means.append(adj[j].mean())
        ci = (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))
        fam_stat[fam] = {
            "n": len(rs),
            "raw_selectivity_mean": float(raw.mean()),
            "adj_selectivity_mean": float(adj.mean()),
            "adj_selectivity_ci95": ci,
        }

    out = {
        "elasticity": {
            "slope_logrF_on_logrR": b_hat,
            "slope_ci95_clustered": b_ci,
            "intercept": a_hat,
            "interpretation": "b>1 => forget radius more fragility-elastic than retain",
        },
        "taxonomy": {
            "selectivity_on_logfragility_slope": c_hat,
            "by_family_sorted_by_adjusted": dict(sorted(
                fam_stat.items(),
                key=lambda kv: -kv[1]["adj_selectivity_mean"])),
        },
        "n_rows": len(rows),
    }
    outp = os.path.join(ROOT, "results", "analysis", "metrics",
                        "elasticity_taxonomy.json")
    json.dump(out, open(outp, "w"), indent=2)

    print(f"(A) elasticity: log r_F = {a_hat:+.3f} + {b_hat:.3f}*log r_R")
    print(f"    slope b = {b_hat:.3f}  clustered CI95 [{b_ci[0]:.3f},{b_ci[1]:.3f}]"
          f"  ({'b>1: forget more elastic' if b_ci[0] > 1 else 'CI includes 1'})")
    print(f"(B) fragility-matched taxonomy (adjusted selectivity, higher=leakier):")
    print(f"    {'family':<16}{'n':>4}{'raw S':>9}{'adj S':>9}{'adj CI95':>18}")
    for fam, st in out["taxonomy"]["by_family_sorted_by_adjusted"].items():
        ci = st["adj_selectivity_ci95"]
        sig = "*" if (ci[0] > 0 or ci[1] < 0) else " "
        print(f"    {fam:<16}{st['n']:>4}{st['raw_selectivity_mean']:>9.3f}"
              f"{st['adj_selectivity_mean']:>9.3f}  [{ci[0]:+.2f},{ci[1]:+.2f}]{sig}")
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
