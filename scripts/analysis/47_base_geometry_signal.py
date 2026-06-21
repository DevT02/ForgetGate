"""Does the global recovery scale (retain-control radius r_R) track the BASE
model more than the unlearning METHOD? The weight-decay sweep (script 46) found
r_R invariant to adapter-side regularization across four orders of magnitude --
suggesting the radius is set by the frozen base geometry. Here we corroborate
that from the existing 132-row audit set, which shares one architecture
(vit_tiny) but three independently trained base seeds (42/123/456).

We decompose the variance of log r_R by base seed vs by method family
(one-way eta^2 each) and report the leave-method-out base spread. If the base
seed explains a comparable or larger share than the method, the global fragility
scale is a base-representation property, not an unlearning-method artifact --
which reframes the weight-decay null as positive evidence and motivates the
cross-architecture experiment.

Output: results/analysis/metrics/base_geometry_signal.json
"""
import json, os
import numpy as np
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FAMILIES = ["orbit", "scrub", "rurk", "salun", "ceu", "baldro", "sga",
            "smoothed_margin", "kl", "oracle", "falw", "orthoreg", "prune"]


def family(suite):
    s = suite.lower()
    for fam in FAMILIES:
        if fam in s:
            return fam
    return "other"


def eta2(groups):
    """One-way eta^2: between-group SS / total SS on the pooled values."""
    allv = np.concatenate([np.asarray(v, float) for v in groups.values()])
    grand = allv.mean()
    ss_tot = ((allv - grand) ** 2).sum()
    ss_between = sum(len(v) * (np.mean(v) - grand) ** 2 for v in groups.values())
    return float(ss_between / ss_tot) if ss_tot > 0 else float("nan")


def main():
    d = json.load(open(os.path.join(ROOT, "results", "analysis", "metrics",
                                     "fragility_confound.json")))
    rows = [r for r in d["rows"] if r["fragility"] > 0]
    logrR = {i: float(np.log(r["fragility"])) for i, r in enumerate(rows)}

    by_seed = defaultdict(list)
    by_method = defaultdict(list)
    for i, r in enumerate(rows):
        by_seed[str(r["seed"])].append(logrR[i])
        by_method[family(r["suite"])].append(logrR[i])

    eta_seed = eta2(by_seed)
    eta_method = eta2(by_method)

    # Within-method across-seed spread: for methods present in >=2 seeds, how much
    # does log r_R move between base seeds (sd of per-seed means), pooled.
    by_method_seed = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        by_method_seed[family(r["suite"])][str(r["seed"])].append(logrR[i])
    cross_seed_sds = []
    for fam, seeds in by_method_seed.items():
        means = [np.mean(v) for v in seeds.values() if len(v) >= 1]
        if len(means) >= 2:
            cross_seed_sds.append(float(np.std(means)))
    within_method_cross_seed_sd = float(np.mean(cross_seed_sds)) if cross_seed_sds else float("nan")

    # Within-seed across-method spread: for each seed, sd of per-method means.
    by_seed_method = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        by_seed_method[str(r["seed"])][family(r["suite"])].append(logrR[i])
    cross_method_sds = []
    for seed, methods in by_seed_method.items():
        means = [np.mean(v) for v in methods.values() if len(v) >= 1]
        if len(means) >= 2:
            cross_method_sds.append(float(np.std(means)))
    within_seed_cross_method_sd = float(np.mean(cross_method_sds)) if cross_method_sds else float("nan")

    out = {
        "n_rows": len(rows),
        "n_seeds": len(by_seed),
        "n_method_families": len(by_method),
        "eta2_logrR_by_base_seed": eta_seed,
        "eta2_logrR_by_method": eta_method,
        "within_method_cross_seed_sd_logrR": within_method_cross_seed_sd,
        "within_seed_cross_method_sd_logrR": within_seed_cross_method_sd,
        "seed_means_logrR": {s: float(np.mean(v)) for s, v in by_seed.items()},
        "note": ("higher eta2_by_base_seed vs eta2_by_method => base geometry "
                 "governs the global recovery scale more than the method"),
    }
    outp = os.path.join(ROOT, "results", "analysis", "metrics",
                        "base_geometry_signal.json")
    json.dump(out, open(outp, "w"), indent=2)

    print(f"n={len(rows)} rows, {len(by_seed)} base seeds, {len(by_method)} methods")
    print(f"eta^2(log r_R | base seed)  = {eta_seed:.3f}")
    print(f"eta^2(log r_R | method)     = {eta_method:.3f}")
    print(f"within-method cross-seed sd = {within_method_cross_seed_sd:.3f}")
    print(f"within-seed cross-method sd = {within_seed_cross_method_sd:.3f}")
    print(f"seed means log r_R: " +
          ", ".join(f"{s}:{m:.2f}" for s, m in out['seed_means_logrR'].items()))
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
