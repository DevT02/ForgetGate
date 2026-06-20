"""
SCRUB seed-dependence analysis.

The paper reports that SCRUB's selective recovery gap is seed-dependent:
selective on seed 42, at parity with the retain-only oracle null on 123/456.
This script quantifies that across all three seeds from the matched-attack
oracle-null audits, with bootstrap CIs, and cross-references the per-seed
representation geometry (inter-class cosine, forget-feature variance, CKA)
to look for a *predictive* correlate of when SCRUB leaks.

Inputs (committed JSON artifacts):
  results/analysis/recovery_radius_..._seed_{42,123,456}_..._scrub-distill__oracle.json
  results/analysis/metrics/scrub_deep_analysis_seed_42.json   (seed 42 only)

Output:
  results/analysis/metrics/scrub_seed_dependence.json
"""
import json
import os
import glob
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRUB = "unlearn_scrub_distill_vit_cifar10_forget0"
ORACLE = "oracle_vit_cifar10_forget0"
SEEDS = [42, 123, 456]
RNG = np.random.default_rng(0)


def oracle_file(seed):
    return os.path.join(
        ROOT, "results", "analysis",
        f"recovery_radius_cifar10_vit_tiny_forget0_seed_{seed}"
        f"_test_with_retain_adam_margin_scrub-distill__oracle.json",
    )


def radii(block, suite):
    """All-sample radii; censored failures -> eps_max (right-censor floor)."""
    rec = block[suite]
    ps = rec["per_sample"]
    eps_max = None
    out = []
    succ = []
    for s in ps:
        out.append(s["radius"] if s["success"] else np.nan)
        succ.append(bool(s["success"]))
    return np.array(out, dtype=float), np.array(succ, dtype=bool)


def median_finite(r):
    f = r[~np.isnan(r)]
    return float(np.median(f)) if len(f) else float("nan")


def boot_median_ratio(rf, rr, n=5000):
    """Bootstrap CI for median(forget)/median(retain) on finite successes."""
    ff = rf[~np.isnan(rf)]
    rrr = rr[~np.isnan(rr)]
    if len(ff) < 3 or len(rrr) < 3:
        return (float("nan"), float("nan"))
    ratios = []
    for _ in range(n):
        a = RNG.choice(ff, len(ff), replace=True)
        b = RNG.choice(rrr, len(rrr), replace=True)
        mb = np.median(b)
        if mb > 0:
            ratios.append(np.median(a) / mb)
    return (float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5)))


def rate_at(r, succ, eps):
    """Recovery rate at budget eps (success AND radius<=eps)."""
    ok = succ & (~np.isnan(r)) & (r <= eps)
    return float(ok.mean())


def main():
    eps_max = 0.03137254901960784
    # selectivity is sharpest at a tight budget; report at eps_max and a tight slice
    results = {"seeds": {}, "meta": {"eps_max": eps_max}}

    for seed in SEEDS:
        f = oracle_file(seed)
        d = json.load(open(f))
        fr, rc = d["forget_recovery"], d["retain_control"]

        s_f, s_fs = radii(fr, SCRUB)
        s_r, s_rs = radii(rc, SCRUB)
        o_f, o_fs = radii(fr, ORACLE)
        o_r, o_rs = radii(rc, ORACLE)

        mfo = median_finite(s_f)
        mro = median_finite(s_r)
        scrub_ratio = mfo / mro if mro > 0 else float("nan")
        ci = boot_median_ratio(s_f, s_r)

        omfo = median_finite(o_f)
        omro = median_finite(o_r)
        oracle_ratio = omfo / omro if omro > 0 else float("nan")

        # selective signal = how much MORE selective the unlearned model is
        # than the oracle null (ratio above the null baseline ratio)
        results["seeds"][str(seed)] = {
            "n_forget": int((~np.isnan(s_f)).sum()),
            "n_retain": int((~np.isnan(s_r)).sum()),
            "scrub_forget_median": mfo,
            "scrub_retain_median": mro,
            "scrub_ratio": scrub_ratio,
            "scrub_ratio_ci95": ci,
            "oracle_forget_median": omfo,
            "oracle_retain_median": omro,
            "oracle_ratio": oracle_ratio,
            "selective_excess_over_null": (
                scrub_ratio / oracle_ratio if oracle_ratio > 0 else float("nan")
            ),
            "pF_at_eps": rate_at(s_f, s_fs, eps_max),
            "pR_at_eps": rate_at(s_r, s_rs, eps_max),
        }

    # representation geometry (only seed 42 deep file exists)
    deep = os.path.join(ROOT, "results", "analysis", "metrics",
                        "scrub_deep_analysis_seed_42.json")
    if os.path.exists(deep):
        dd = json.load(open(deep))
        results["representation_seed_42"] = dd["models"]

    out = os.path.join(ROOT, "results", "analysis", "metrics",
                       "scrub_seed_dependence.json")
    json.dump(results, open(out, "w"), indent=2)

    # console summary
    print(f"{'seed':>5} {'scrubF':>9} {'scrubR':>9} {'ratio':>7} "
          f"{'ratio95CI':>16} {'nullR':>7} {'excess':>7} {'pF':>6} {'pR':>6}")
    for seed in SEEDS:
        s = results["seeds"][str(seed)]
        ci = s["scrub_ratio_ci95"]
        print(f"{seed:>5} {s['scrub_forget_median']:>9.5f} "
              f"{s['scrub_retain_median']:>9.5f} {s['scrub_ratio']:>7.2f} "
              f"[{ci[0]:>5.2f},{ci[1]:>5.2f}]   {s['oracle_ratio']:>7.2f} "
              f"{s['selective_excess_over_null']:>7.2f} "
              f"{s['pF_at_eps']:>6.2f} {s['pR_at_eps']:>6.2f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
