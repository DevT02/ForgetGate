"""
Is "selective leakage" a confound of global fragility?

Hypothesis (born from the SCRUB seed-dependence finding): the forget-vs-retain
recovery gap that the benchmark reads as residual-knowledge "selective leakage"
is partly an artifact of how globally fragile a checkpoint is. A checkpoint that
is easy to attack everywhere (small absolute recovery radius) may show an
apparent forget-vs-retain gap regardless of method.

Test, holistically, across EVERY (method, seed, config) audit row already on
disk that carries both a forget_recovery and a matched retain_control for the
same suite:

  selectivity  S = log( median_retain_radius / median_forget_radius )
                   ( S > 0  => forget easier to recover => "selective" )
  fragility    F = median_retain_radius            (oracle-free global scale)

If S correlates negatively with F (more fragile = smaller F = more apparent
selectivity), the gap is confounded by brittleness and single-checkpoint
selective claims are unsafe without controlling for F.

Outputs:
  results/analysis/metrics/fragility_confound.json
"""
import json
import glob
import os
import re
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RNG = np.random.default_rng(0)


def median_finite_radius(rec):
    ps = rec.get("per_sample", [])
    r = [s["radius"] for s in ps if s.get("success") and s.get("radius") is not None]
    return float(np.median(r)) if len(r) >= 5 else None


def parse_tags(path):
    name = os.path.basename(path)
    seed = re.search(r"seed_(\d+)_", name)
    dset = "cifar100" if "cifar100" in name else "cifar10"
    attack = "apgd" if "apgd" in name else ("adam" if "adam_margin" in name else "other")
    return (seed.group(1) if seed else "?"), dset, attack


def main():
    files = glob.glob(
        os.path.join(ROOT, "results", "analysis", "**",
                     "recovery_radius_*test_with_retain*.json"),
        recursive=True,
    )
    rows = []
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        fr = d.get("forget_recovery", {})
        rc = d.get("retain_control", {})
        seed, dset, attack = parse_tags(f)
        for suite in set(fr) & set(rc):
            mf = median_finite_radius(fr[suite])
            mr = median_finite_radius(rc[suite])
            if mf and mr and mf > 0 and mr > 0:
                rows.append({
                    "file": os.path.basename(f),
                    "suite": suite,
                    "seed": seed,
                    "dataset": dset,
                    "attack": attack,
                    "median_forget": mf,
                    "median_retain": mr,
                    "selectivity": float(np.log(mr / mf)),   # >0 => selective
                    "fragility": float(mr),                  # oracle-free scale
                })

    # dedup identical (suite, seed, medians) that appear in both / and /metrics
    seen = {}
    for r in rows:
        k = (r["suite"], r["seed"], round(r["median_forget"], 8),
             round(r["median_retain"], 8))
        seen[k] = r
    rows = list(seen.values())

    S = np.array([r["selectivity"] for r in rows])
    F = np.array([r["fragility"] for r in rows])
    logF = np.log(F)

    def pearson(a, b):
        if len(a) < 4:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    def boot_ci(a, b, n=10000):
        idx = np.arange(len(a))
        cs = []
        for _ in range(n):
            j = RNG.choice(idx, len(idx), replace=True)
            if np.std(a[j]) > 0 and np.std(b[j]) > 0:
                cs.append(np.corrcoef(a[j], b[j])[0, 1])
        return (float(np.percentile(cs, 2.5)), float(np.percentile(cs, 97.5)))

    r_SF = pearson(S, logF)
    ci_SF = boot_ci(S, logF)

    # Spearman (rank) as a robustness check
    def spearman(a, b):
        ra = np.argsort(np.argsort(a))
        rb = np.argsort(np.argsort(b))
        return pearson(ra.astype(float), rb.astype(float))

    rho_SF = spearman(S, logF)

    # cluster bootstrap by suite: rows sharing a suite are not independent,
    # so resample SUITES (not rows) for an honest CI.
    from collections import defaultdict

    def cluster_boot(subset, n=10000):
        by = defaultdict(list)
        for r in subset:
            by[r["suite"]].append(r)
        keys = list(by)
        cs = []
        for _ in range(n):
            pick = RNG.choice(len(keys), len(keys), replace=True)
            pooled = []
            for i in pick:
                pooled.extend(by[keys[i]])
            a = np.array([x["selectivity"] for x in pooled])
            b = np.log([x["fragility"] for x in pooled])
            if len(a) >= 5 and np.std(a) > 0 and np.std(b) > 0:
                cs.append(np.corrcoef(a, b)[0, 1])
        return (float(np.percentile(cs, 2.5)), float(np.percentile(cs, 97.5)),
                len(keys))

    cl_all = cluster_boot(rows)
    c10 = [r for r in rows if r["dataset"] == "cifar10"]
    cl_c10 = cluster_boot(c10)
    # stratified point estimates
    strata = {}
    for key, val in [("dataset", "cifar10"), ("attack", "adam"),
                     ("attack", "apgd"), ("seed", "42")]:
        sub = [r for r in rows if r[key] == val]
        Ss = np.array([r["selectivity"] for r in sub])
        Fs = np.log([r["fragility"] for r in sub]) if sub else np.array([])
        strata[f"{key}={val}"] = {"r": pearson(Ss, Fs), "n": len(sub)}

    # selective fraction split by fragility median
    medF = np.median(F)
    frac_sel_fragile = float((S[F <= medF] > 0).mean())
    frac_sel_robust = float((S[F > medF] > 0).mean())

    out = {
        "n_rows": len(rows),
        "n_unique_suites": len({r["suite"] for r in rows}),
        "seeds": sorted({r["seed"] for r in rows}),
        "selectivity_vs_log_fragility": {
            "pearson_r": r_SF,
            "pearson_ci95_row_boot": ci_SF,
            "spearman_rho": rho_SF,
            "cluster_boot_ci95_by_suite": cl_all[:2],
            "cluster_boot_n_suites": cl_all[2],
            "cifar10_only_cluster_ci95": cl_c10[:2],
            "stratified_r": strata,
        },
        "selective_fraction": {
            "fragile_half": frac_sel_fragile,
            "robust_half": frac_sel_robust,
            "fragility_split_at": float(medF),
        },
        "rows": sorted(rows, key=lambda r: r["fragility"]),
    }
    outp = os.path.join(ROOT, "results", "analysis", "metrics",
                        "fragility_confound.json")
    json.dump(out, open(outp, "w"), indent=2)

    print(f"rows={len(rows)}  unique suites={out['n_unique_suites']}  "
          f"seeds={out['seeds']}")
    print(f"corr(selectivity, log fragility): pearson r={r_SF:+.3f} "
          f"spearman={rho_SF:+.3f}")
    print(f"  cluster-boot by suite CI95 [{cl_all[0]:+.3f},{cl_all[1]:+.3f}] "
          f"(n_suites={cl_all[2]})")
    print(f"  cifar10-only cluster CI95 [{cl_c10[0]:+.3f},{cl_c10[1]:+.3f}]")
    print("  stratified: " + "  ".join(
        f"{k}:{v['r']:+.3f}(n{v['n']})" for k, v in strata.items()))
    print(f"P(selective | fragile half) = {frac_sel_fragile:.2f}   "
          f"P(selective | robust half) = {frac_sel_robust:.2f}   "
          f"(split at retain median={medF:.5f})")
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
