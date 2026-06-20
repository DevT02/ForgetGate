"""
Per-method forget-radius elasticity as a suppression-quality signature.

The pooled elasticity b=1.31 (script 41) says the forget radius is super-linearly
sensitive to global fragility. Does b vary BY METHOD, and does it mean something?

Interpretation: within a method, fragility varies across seeds/attack configs.
The within-method slope b_m of log r_F on log r_R measures whether that method's
forget-class suppression degrades UNIFORMLY with the global scale (b_m ~ 1, the
forget class behaves like any other input) or as a BRITTLE SPIKE that collapses
faster than retain controls when the model/attack gets harsher (b_m > 1). A
brittle spike is exactly what a targeted attacker exploits, so b_m is a candidate
mechanistic signature of genuine, exploitable leakage -- distinct from the LEVEL
of selectivity (adjusted selectivity e from script 41).

Tests:
  - estimability: per-method fragility spread (range of log r_R) and n.
  - per-method b_m with row bootstrap CI.
  - does b_m correlate with adjusted selectivity e_m (genuine-leaker ranking)?
  - does b_m separate the genuine leakers (salun, rurk, smoothed_margin) from the
    fragility-artifact rows (scrub, oracle)?

Output: results/analysis/metrics/per_method_elasticity.json
"""
import json
import os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RNG = np.random.default_rng(11)
FAMILIES = ["orbit", "scrub", "rurk", "salun", "ceu", "baldro", "sga",
            "smoothed_margin", "kl", "oracle"]


def fam(s):
    for f in FAMILIES:
        if f in s.lower():
            return f
    return "other"


def main():
    d = json.load(open(os.path.join(ROOT, "results", "analysis", "metrics",
                                     "fragility_confound.json")))
    rows = d["rows"]
    for r in rows:
        r["family"] = fam(r["suite"])
        r["logR"] = float(np.log(r["fragility"]))
        r["logF"] = float(np.log(r["median_forget"]))

    # adjusted selectivity per family from script 41 output
    tax = json.load(open(os.path.join(ROOT, "results", "analysis", "metrics",
                                       "elasticity_taxonomy.json")))
    adj = {k: v["adj_selectivity_mean"]
           for k, v in tax["taxonomy"]["by_family_sorted_by_adjusted"].items()}

    groups = {}
    for r in rows:
        groups.setdefault(r["family"], []).append(r)

    per = {}
    for f, rs in groups.items():
        if len(rs) < 8:
            continue
        x = np.array([r["logR"] for r in rs])
        y = np.array([r["logF"] for r in rs])
        spread = float(x.max() - x.min())
        if spread < 0.5 or np.std(x) == 0:   # not enough fragility range to fit
            per[f] = {"n": len(rs), "fragility_spread": spread,
                      "b": None, "note": "insufficient fragility range"}
            continue
        b = float(np.polyfit(x, y, 1)[0])
        bs = []
        for _ in range(10000):
            j = RNG.choice(len(rs), len(rs), replace=True)
            if np.std(x[j]) > 0:
                bs.append(np.polyfit(x[j], y[j], 1)[0])
        per[f] = {
            "n": len(rs),
            "fragility_spread": spread,
            "b": b,
            "b_ci95": [float(np.percentile(bs, 2.5)),
                       float(np.percentile(bs, 97.5))],
            "adj_selectivity": adj.get(f),
        }

    # correlate b_m with adjusted selectivity across well-estimated families
    est = {f: v for f, v in per.items()
           if v.get("b") is not None and v.get("adj_selectivity") is not None}
    bs_ = np.array([v["b"] for v in est.values()])
    es_ = np.array([v["adj_selectivity"] for v in est.values()])
    if len(bs_) >= 4:
        rho = float(np.corrcoef(
            np.argsort(np.argsort(bs_)).astype(float),
            np.argsort(np.argsort(es_)).astype(float))[0, 1])
        r_pear = float(np.corrcoef(bs_, es_)[0, 1])
    else:
        rho = r_pear = float("nan")

    out = {
        "per_method": dict(sorted(
            per.items(),
            key=lambda kv: -(kv[1]["b"] if kv[1].get("b") else -9))),
        "b_vs_adjusted_selectivity": {
            "pearson_r": r_pear, "spearman_rho": rho,
            "families": list(est.keys())},
    }
    outp = os.path.join(ROOT, "results", "analysis", "metrics",
                        "per_method_elasticity.json")
    json.dump(out, open(outp, "w"), indent=2)

    print(f"{'family':<16}{'n':>4}{'spread':>8}{'b_m':>8}{'b CI95':>16}{'adjS':>8}")
    for f, v in out["per_method"].items():
        if v.get("b") is None:
            print(f"{f:<16}{v['n']:>4}{v['fragility_spread']:>8.2f}   --  "
                  f"(low fragility range)")
            continue
        ci = v["b_ci95"]
        print(f"{f:<16}{v['n']:>4}{v['fragility_spread']:>8.2f}{v['b']:>8.2f}"
              f"  [{ci[0]:>5.2f},{ci[1]:>5.2f}]{v['adj_selectivity']:>8.2f}")
    print(f"\nb_m vs adjusted selectivity: pearson={r_pear:+.3f} "
          f"spearman={rho:+.3f}  (n_families={len(est)})")
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
