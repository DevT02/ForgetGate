"""
Are the fragility confound (r) and the forget-radius elasticity (b) real, or
artifacts of a few rows / one method / outliers / chance?

Hard checks for BOTH statistics:
  1. Permutation null: shuffle fragility across rows, recompute; p = P(|stat| as
     extreme as observed) under the null of no association.
  2. Leave-one-method-out (LOMO): drop each method family, recompute; the sign
     must be stable across every deletion.
  3. Robust estimators: Theil-Sen slope (elasticity) and trimmed correlation,
     immune to a handful of outliers.
  4. Outlier trim: drop the most extreme 10% of fragility rows.

Note on the elasticity as the *clean* statement: corr(S, logF) shares the
variable r_R on both sides, so we treat the regression slope b of
log r_F on log r_R as primary. b=1 is the no-confound value; classic
errors-in-variables attenuation biases b TOWARD 1, so an observed b>1 is
conservative evidence (noise cannot manufacture b>1 from a true b=1).

Output: results/analysis/metrics/robustness_checks.json
"""
import json
import os
import numpy as np
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RNG = np.random.default_rng(7)


def theil_sen(x, y):
    x = np.asarray(x); y = np.asarray(y)
    n = len(x)
    if n > 200:
        idx = RNG.choice(n, 200, replace=False)
        x, y = x[idx], y[idx]
        n = 200
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if x[j] != x[i]:
                slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    return float(np.median(slopes))


def main():
    d = json.load(open(os.path.join(ROOT, "results", "analysis", "metrics",
                                     "fragility_confound.json")))
    rows = d["rows"]
    from re import search

    def fam(s):
        for f in ["orbit", "scrub", "rurk", "salun", "ceu", "baldro", "sga",
                  "smoothed_margin", "kl", "oracle", "falw", "orthoreg", "prune"]:
            if f in s.lower():
                return f
        return "other"

    S = np.array([r["selectivity"] for r in rows])
    logR = np.array([np.log(r["fragility"]) for r in rows])           # log retain radius
    logF_ = np.array([np.log(r["median_forget"]) for r in rows])      # log forget radius
    fams = [fam(r["suite"]) for r in rows]

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    def slope(x, y):
        return float(np.polyfit(x, y, 1)[0])

    r_obs = corr(S, logR)
    b_obs = slope(logR, logF_)

    # 1. permutation null.
    # Testing the confound corr(S, logR) = 0 is ALGEBRAICALLY identical to
    # testing the elasticity b = 1: since S = logR - logF_, we have
    # cov(S, logR) = var(logR) - cov(logF_, logR) = var(logR)(1 - b), so
    # cov(S, logR) = 0  <=>  b = 1. One permutation test covers both.
    N = 20000
    rc = 0
    for _ in range(N):
        perm = RNG.permutation(logR)
        if abs(corr(S, perm)) >= abs(r_obs):
            rc += 1
    p_r = (rc + 1) / (N + 1)
    p_b = p_r  # identical hypothesis (b=1 <=> confound=0)

    # 2. leave-one-method-out
    lomo_r, lomo_b = {}, {}
    for drop in sorted(set(fams)):
        keep = [i for i, f in enumerate(fams) if f != drop]
        if len(keep) > 10:
            lomo_r[drop] = corr(S[keep], logR[keep])
            lomo_b[drop] = slope(logR[keep], logF_[keep])

    # 3. robust estimators
    b_ts = theil_sen(logR, logF_)
    # trimmed correlation: drop |z|>2.5 in either axis
    z = lambda v: (v - v.mean()) / v.std()
    mask = (np.abs(z(S)) < 2.5) & (np.abs(z(logR)) < 2.5)
    r_trim = corr(S[mask], logR[mask])
    b_trim = slope(logR[mask], logF_[mask])

    # 4. outlier trim on fragility extremes (drop most extreme 10%)
    lo, hi = np.percentile(logR, [5, 95])
    m2 = (logR >= lo) & (logR <= hi)
    r_ftrim = corr(S[m2], logR[m2])
    b_ftrim = slope(logR[m2], logF_[m2])

    # 5. per-seed replication: does each independent seed reproduce sign?
    seeds = [r["seed"] for r in rows]
    per_seed = {}
    for sd in sorted(set(seeds)):
        idx = [i for i, s in enumerate(seeds) if s == sd]
        if len(idx) >= 6:
            per_seed[sd] = {
                "n": len(idx),
                "corr": corr(S[idx], logR[idx]),
                "elasticity_b": slope(logR[idx], logF_[idx]),
            }

    out = {
        "observed": {"corr_S_logF": r_obs, "elasticity_b": b_obs},
        "permutation_null": {
            "p_corr": p_r, "p_elasticity_b_vs_1": p_b, "n_perm": N,
            "note": "b=1 <=> confound=0 (S=logR-logF), so one test covers both"},
        "leave_one_method_out": {
            "corr": lomo_r, "elasticity_b": lomo_b,
            "corr_sign_stable": all(v < 0 for v in lomo_r.values()),
            "b_gt_1_stable": all(v > 1 for v in lomo_b.values())},
        "robust_estimators": {
            "theil_sen_b": b_ts, "trimmed_corr": r_trim, "trimmed_b": b_trim},
        "fragility_outlier_trim_5_95": {"corr": r_ftrim, "b": b_ftrim},
        "per_seed_replication": per_seed,
    }
    outp = os.path.join(ROOT, "results", "analysis", "metrics",
                        "robustness_checks.json")
    json.dump(out, open(outp, "w"), indent=2)

    print(f"observed: corr(S,logF)={r_obs:+.3f}  elasticity b={b_obs:.3f}")
    print(f"permutation null (n={N}): p={p_r:.4f}  "
          f"(tests both confound=0 and b=1; identical hypothesis)")
    print(f"LOMO corr range [{min(lomo_r.values()):+.3f},"
          f"{max(lomo_r.values()):+.3f}] sign-stable={out['leave_one_method_out']['corr_sign_stable']}")
    print(f"LOMO elasticity b range [{min(lomo_b.values()):.3f},"
          f"{max(lomo_b.values()):.3f}] b>1-stable={out['leave_one_method_out']['b_gt_1_stable']}")
    print(f"robust: Theil-Sen b={b_ts:.3f}  trimmed r={r_trim:+.3f} "
          f"trimmed b={b_trim:.3f}")
    print(f"fragility 5-95 trim: r={r_ftrim:+.3f}  b={b_ftrim:.3f}")
    print("per-seed replication (independent seeds):")
    for sd, st in per_seed.items():
        print(f"    seed {sd}: n={st['n']:3d}  corr={st['corr']:+.3f}  "
              f"b={st['elasticity_b']:.3f}")
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
