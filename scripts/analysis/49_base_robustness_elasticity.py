"""CAUSAL within-method elasticity from the base-robustness sweep (STAGED).

Reads the SalUn audits over five adversarial-strength bases x two seeds and fits
log r_F = a + b log r_R. Because only the base smoothness varies (method fixed),
b is a within-method causal estimate of the forget-radius elasticity
(Proposition: super-elasticity), directly comparable to the observational
cross-method b=1.31. Reports the first stage (adv strength -> r_R) to confirm the
knob -- unlike weight decay -- actually moves the global fragility scale.

Output: results/analysis/metrics/base_robustness_elasticity.json
"""
import json, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RNG = np.random.default_rng(0)

# adversarial-training strength as an ordinal knob (0=clean .. 4=ultra)
ADV = {"salun-advclean": 0, "salun-advsmoke": 1, "salun-advmid": 2,
       "salun-advhigh": 3, "salun-advultra": 4}
SEEDS = [42, 123]


def load_point(tag, seed):
    p = os.path.join(ROOT, "results", "analysis",
                     f"recovery_radius_cifar10_vit_tiny_forget0_seed_{seed}"
                     f"_test_with_retain_adam_margin_{tag}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    fr = list(d["forget_recovery"].values())[0]["summary"]
    rc = list(d["retain_control"].values())[0]["summary"]
    return {"tag": tag, "seed": seed, "adv": ADV[tag],
            "rF_mean": fr["mean_radius"], "rF_med": fr["median_radius"],
            "rR_mean": rc["mean_radius"], "rR_med": rc["median_radius"]}


def ols(x, y):
    b, a = np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)
    return float(b), float(a)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    points = [p for tag in ADV for seed in SEEDS
              if (p := load_point(tag, seed)) is not None]
    print(f"loaded {len(points)}/{len(ADV)*len(SEEDS)} points")
    for p in points:
        print(f"  adv={p['adv']} ({p['tag']}) seed={p['seed']}  "
              f"r_R={p['rR_mean']:.5f}  r_F={p['rF_mean']:.5f}")
    if len(points) < 4:
        json.dump({"status": "STAGED_NOT_RUN", "n_points": len(points)},
                  open(os.path.join(ROOT, "results", "analysis", "metrics",
                                    "base_robustness_elasticity.json"), "w"), indent=2)
        print("STAGED -- run the sweep first (run_base_robustness_sweep.sh)")
        return

    adv = np.array([p["adv"] for p in points])
    rR = np.array([p["rR_mean"] for p in points])
    rF = np.array([p["rF_mean"] for p in points])
    # The identifying variation is that the bases produce widely different r_R;
    # the adv label need not be monotone in robustness (e.g. the smoke-test base
    # is an outlier). So judge the first stage by the induced RANGE of r_R, not
    # by monotonicity in the arbitrary ordinal.
    first = spearman(adv, rR)
    rR_range_over_mean = float((rR[np.isfinite(rR)].max() - rR[np.isfinite(rR)].min())
                               / (rR[np.isfinite(rR)].mean() + 1e-12))

    use = (rF > 0) & (rR > 0) & np.isfinite(rF) & np.isfinite(rR)
    b = a = float("nan"); ci = [float("nan"), float("nan")]
    if use.sum() >= 4 and np.std(np.log(rR[use])) > 0:
        b, a = ols(np.log(rR[use]), np.log(rF[use]))
        # cluster bootstrap by adversarial level
        clusters = {}
        for i in np.where(use)[0]:
            clusters.setdefault(points[i]["adv"], []).append(i)
        keys = list(clusters); bs = []
        for _ in range(10000):
            pick = RNG.choice(len(keys), len(keys), replace=True)
            xs, ys = [], []
            for k in pick:
                for i in clusters[keys[k]]:
                    xs.append(np.log(rR[i])); ys.append(np.log(rF[i]))
            if len(xs) > 3 and np.std(xs) > 0:
                bb, _ = ols(xs, ys); bs.append(bb)
        if bs:
            ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    knob_ok = rR_range_over_mean >= 0.5
    out = {"status": "OK" if knob_ok else "FIRST_STAGE_WEAK",
           "n_points": len(points),
           "first_stage_rR_range_over_mean": rR_range_over_mean,
           "first_stage_spearman_adv_vs_rR": first,
           "structural_b": b, "structural_b_ci95": ci, "intercept": a,
           "observational_cross_method_b": 1.31,
           "points": points}
    json.dump(out, open(os.path.join(ROOT, "results", "analysis", "metrics",
                        "base_robustness_elasticity.json"), "w"), indent=2)
    print(f"FIRST STAGE r_R range/mean = {rR_range_over_mean:.3f}  "
          f"({'valid: base smoothness moves r_R' if knob_ok else 'weak'})  "
          f"[spearman(adv,r_R)={first:+.2f}, non-monotone ordinal ok]")
    print(f"STRUCTURAL  log r_F = {a:+.3f} + {b:.3f} log r_R   CI95 [{ci[0]:.3f},{ci[1]:.3f}]")
    print(f"            (n_used={int(use.sum())})  observational cross-method b = 1.31")


if __name__ == "__main__":
    main()
