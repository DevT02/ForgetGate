"""
CONTROLLED within-method recovery-radius elasticity.

The cross-method elasticity (script 41, b=1.31) is observational: it pools
checkpoints that differ in method, so the fragility scale r_R is confounded with
unobserved method properties. Here we make it causal. We fix the method (SalUn),
the base model, and the forget class, and vary ONLY weight decay -- an exogenous
global-smoothness knob -- across {0, 1e-3, 1e-2, 1e-1, 5e-1} x seeds {42,123}.

Design (a dose-response / first-stage + structural fit):
  FIRST STAGE   does the knob move the global fragility scale r_R?
                regress log r_R on log(wd) (Spearman + OLS). If r_R responds,
                weight decay is a valid instrument for fragility.
  STRUCTURAL    fit log r_F = a + b * log r_R across the swept points. Because
                only wd varies, b is a WITHIN-method causal elasticity, directly
                comparable to the observational cross-method b.

Censoring: SalUn often already leaks the forget class at clean input, so the
forget MEDIAN radius can be exactly 0 (log-undefined). We therefore fit on the
mean radius (continuous, >0 whenever any sample resists) and report the median
descriptively. Points with r_F==0 or r_R==0 are dropped from the log-fit and
counted.

Output: results/analysis/metrics/controlled_wd_elasticity.json
"""
import json
import os
import glob
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RNG = np.random.default_rng(0)

# weight-decay value per suite tag
WD = {
    "salun-wd0": 0.0,
    "salun-wd1e3": 1e-3,
    "salun-wd1e2": 1e-2,
    "salun-wd1e1": 1e-1,
    "salun-wd5e1": 5e-1,
}
SEEDS = [42, 123]


def load_point(tag, seed):
    path = os.path.join(
        ROOT, "results", "analysis",
        f"recovery_radius_cifar10_vit_tiny_forget0_seed_{seed}"
        f"_test_with_retain_adam_margin_{tag}.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    fr = list(d["forget_recovery"].values())[0]["summary"]
    rc = list(d["retain_control"].values())[0]["summary"]
    return {
        "tag": tag, "seed": seed, "wd": WD[tag],
        "rF_mean": fr["mean_radius"], "rF_med": fr["median_radius"],
        "rR_mean": rc["mean_radius"], "rR_med": rc["median_radius"],
        "rF_clean_success": fr.get("clean_success_rate"),
        "rR_clean_success": rc.get("clean_success_rate"),
    }


def ols(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    b, a = np.polyfit(x, y, 1)
    return float(b), float(a)


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    points = []
    for tag in WD:
        for seed in SEEDS:
            p = load_point(tag, seed)
            if p is not None:
                points.append(p)

    n_total = len(WD) * len(SEEDS)
    print(f"loaded {len(points)}/{n_total} swept points")
    for p in points:
        print(f"  wd={p['wd']:<6} seed={p['seed']}  "
              f"r_R(mean/med)={p['rR_mean']:.5f}/{p['rR_med']:.5f}  "
              f"r_F(mean/med)={p['rF_mean']:.5f}/{p['rF_med']:.5f}")

    if len(points) < 4:
        out = {"status": "UNDERPOWERED",
               "reason": f"only {len(points)} swept points available",
               "points": points}
        outp = os.path.join(ROOT, "results", "analysis", "metrics",
                            "controlled_wd_elasticity.json")
        json.dump(out, open(outp, "w"), indent=2)
        print(f"\nUNDERPOWERED -- wrote {outp}")
        return

    # ---- FIRST STAGE: does weight decay move r_R? ----
    # use log(wd) with wd=0 mapped to a small floor for ranking; Spearman on raw wd
    wd = np.array([p["wd"] for p in points])
    rR = np.array([p["rR_mean"] for p in points])
    first_spear = spearman(wd, rR)
    # OLS of log r_R on log(wd) over wd>0 points
    pos = wd > 0
    fs_slope = fs_int = float("nan")
    if pos.sum() >= 3 and np.std(np.log(wd[pos])) > 0:
        fs_slope, fs_int = ols(np.log(wd[pos]), np.log(rR[pos]))

    # ---- STRUCTURAL: log r_F = a + b log r_R ----
    rF = np.array([p["rF_mean"] for p in points])
    use = (rF > 0) & (rR > 0)
    n_used, n_drop = int(use.sum()), int((~use).sum())
    b_hat = a_hat = float("nan")
    b_ci = [float("nan"), float("nan")]
    struct_spear = float("nan")
    if n_used >= 4 and np.std(np.log(rR[use])) > 0:
        lrR, lrF = np.log(rR[use]), np.log(rF[use])
        b_hat, a_hat = ols(lrR, lrF)
        struct_spear = spearman(lrR, lrF)
        # cluster bootstrap by weight-decay value (the exogenous unit)
        clusters = {}
        for i in np.where(use)[0]:
            clusters.setdefault(points[i]["wd"], []).append(i)
        keys = list(clusters)
        bs = []
        for _ in range(10000):
            pick = RNG.choice(len(keys), len(keys), replace=True)
            xs, ys = [], []
            for k in pick:
                for i in clusters[keys[k]]:
                    xs.append(np.log(rR[i])); ys.append(np.log(rF[i]))
            if len(xs) > 3 and np.std(xs) > 0:
                bb, _ = ols(xs, ys)
                bs.append(bb)
        if bs:
            b_ci = [float(np.percentile(bs, 2.5)),
                    float(np.percentile(bs, 97.5))]

    # Verdict: the structural elasticity is only identified if the first stage is
    # strong. If weight decay does not move r_R, the knob is invalid and no causal
    # b can be read (the apparent slope is then outlier-driven noise).
    knob_works = abs(first_spear) >= 0.5
    verdict = ("FIRST-STAGE NULL: weight decay does not move the recovery radius "
               "(|spearman| < 0.5); causal elasticity NOT identified. Likely cause: "
               "the recovery radius is set by base-model geometry and class margins, "
               "and a rank-8 LoRA adapter's weight norm (all WD controls) is too "
               "small a lever to shift input-space smoothness.") if not knob_works \
              else "First stage valid; structural elasticity is identified."

    out = {
        "status": "OK" if knob_works else "FIRST_STAGE_NULL",
        "verdict": verdict,
        "design": "SalUn fixed; weight decay varied as exogenous fragility knob",
        "n_points": len(points),
        "first_stage": {
            "spearman_wd_vs_rR": first_spear,
            "ols_logrR_on_logwd_slope": fs_slope,
            "ols_logrR_on_logwd_intercept": fs_int,
            "interpretation": "knob is valid if |spearman| is large / CI of structural fit usable",
        },
        "structural": {
            "slope_logrF_on_logrR": b_hat,
            "slope_ci95_clustered_by_wd": b_ci,
            "intercept": a_hat,
            "spearman_logrR_vs_logrF": struct_spear,
            "n_used": n_used,
            "n_dropped_censored": n_drop,
            "observational_cross_method_b": 1.31,
            "interpretation": "b>1 => forget radius super-elastic to fragility, within method",
        },
        "points": points,
    }
    outp = os.path.join(ROOT, "results", "analysis", "metrics",
                        "controlled_wd_elasticity.json")
    json.dump(out, open(outp, "w"), indent=2)

    print(f"\nFIRST STAGE  spearman(wd, r_R) = {first_spear:+.3f}"
          f"   log r_R = {fs_int:+.3f} + {fs_slope:+.3f} log wd")
    print(f"STRUCTURAL   log r_F = {a_hat:+.3f} + {b_hat:.3f} log r_R"
          f"   (n_used={n_used}, dropped={n_drop})")
    print(f"             clustered CI95 [{b_ci[0]:.3f}, {b_ci[1]:.3f}]"
          f"   spearman={struct_spear:+.3f}")
    print(f"             observational cross-method b = 1.31")
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
