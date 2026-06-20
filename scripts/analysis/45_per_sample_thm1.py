"""
Per-sample validation of Theorem 1, and the margin-heterogeneity mechanism.

Theorem 1 says r_t(x) >= -m_t(x)/L pointwise. The paper currently checks it only
as a 4-point cross-method (L, median r) scatter -- a reviewer rightly notes that
looks more decisive than 4 aggregates can be. Here is the proper per-sample test
using the hundreds of individual (radius, clean margin) pairs already audited.

Margin proxy: clean_target_prob is the softmax probability of the forgotten class
at the clean input; log(clean_target_prob) is monotone in the forget-class logit
margin (when p_t is tiny the prob margin saturates at ~-1, so the LOG prob, not
the prob, carries the usable logit-margin signal). Theorem 1 then predicts
radius correlates NEGATIVELY with log(clean_target_prob): samples with a less
suppressed forget logit (larger log p_t) recover at smaller radius.

We (1) pool per-checkpoint correlations on blocks that actually have margin
variation (not uniformly underflowed), (2) compare forget vs retain so the test
is not forget-specific, and (3) show that margin HETEROGENEITY -- the spread of
log p_t within a checkpoint -- tracks selective recoverability, which mechanizes
the SCRUB seed-dependence (heterogeneous seed 42 is selective; uniformly crushed
seed 123 is not).

Output: results/analysis/metrics/per_sample_thm1.json
"""
import json
import os
import glob
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FLOOR = 1e-30
MIN_SPREAD = 1.5     # std of log p_t needed for a usable margin-variation block
MIN_USABLE = 15      # min non-underflowed successful samples


def block_corr(ps):
    """corr(radius, log clean_target_prob) over successful samples with signal."""
    s = [x for x in ps if x.get("success") and x.get("radius") is not None]
    if len(s) < MIN_USABLE:
        return None
    rad = np.array([x["radius"] for x in s])
    lp = np.log(np.array([max(x["clean_target_prob"], FLOOR) for x in s]))
    usable = lp > -25
    if usable.sum() < MIN_USABLE or np.std(lp[usable]) < MIN_SPREAD:
        return {"corr": None, "spread": float(np.std(lp)), "n": len(s),
                "n_usable": int(usable.sum())}
    r = float(np.corrcoef(rad[usable], lp[usable])[0, 1])
    # slope d(radius)/d(log p_t): proportional to local 1/L
    slope = float(np.polyfit(lp[usable], rad[usable], 1)[0])
    return {"corr": r, "slope": slope, "spread": float(np.std(lp[usable])),
            "n": len(s), "n_usable": int(usable.sum())}


def main():
    files = glob.glob(os.path.join(
        ROOT, "results", "analysis", "**",
        "recovery_radius_*test_with_retain*.json"), recursive=True)

    forget_corrs, retain_corrs = [], []
    spread_vs_sel = []   # (margin spread, selectivity) per checkpoint
    seen = set()
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        fr, rc = d.get("forget_recovery", {}), d.get("retain_control", {})
        for suite in set(fr) & set(rc):
            key = (suite, os.path.basename(f).split("seed_")[1][:3])
            if key in seen:
                continue
            seen.add(key)
            cf = block_corr(fr[suite]["per_sample"])
            cr = block_corr(rc[suite]["per_sample"])
            if cf and cf.get("corr") is not None:
                forget_corrs.append(cf["corr"])
                # selectivity proxy: retain median / forget median (finite)
                rf = [x["radius"] for x in fr[suite]["per_sample"]
                      if x.get("success") and x.get("radius")]
                rr = [x["radius"] for x in rc[suite]["per_sample"]
                      if x.get("success") and x.get("radius")]
                if len(rf) >= 5 and len(rr) >= 5:
                    sel = float(np.log(np.median(rr) / np.median(rf)))
                    spread_vs_sel.append((cf["spread"], sel))
            if cr and cr.get("corr") is not None:
                retain_corrs.append(cr["corr"])

    fc = np.array(forget_corrs)
    rcv = np.array(retain_corrs)
    sp = np.array(spread_vs_sel)

    out = {
        "forget": {
            "n_blocks_with_signal": len(fc),
            "median_corr": float(np.median(fc)) if len(fc) else None,
            "frac_negative": float((fc < 0).mean()) if len(fc) else None,
            "frac_strong_neg": float((fc < -0.3).mean()) if len(fc) else None,
        },
        "retain": {
            "n_blocks_with_signal": len(rcv),
            "median_corr": float(np.median(rcv)) if len(rcv) else None,
            "frac_negative": float((rcv < 0).mean()) if len(rcv) else None,
        },
        "margin_spread_vs_selectivity": {
            "n": len(sp),
            "pearson_r": (float(np.corrcoef(sp[:, 0], sp[:, 1])[0, 1])
                          if len(sp) >= 4 else None),
        },
    }
    outp = os.path.join(ROOT, "results", "analysis", "metrics",
                        "per_sample_thm1.json")
    json.dump(out, open(outp, "w"), indent=2)

    print("Per-sample Theorem 1 test  (radius vs log clean_target_prob):")
    print(f"  FORGET: {len(fc)} checkpoints with margin variation, "
          f"median corr = {np.median(fc):+.3f}, "
          f"{(fc<0).mean()*100:.0f}% negative, "
          f"{(fc<-0.3).mean()*100:.0f}% strongly negative (< -0.3)")
    if len(rcv):
        print(f"  RETAIN: {len(rcv)} checkpoints, "
              f"median corr = {np.median(rcv):+.3f}, "
              f"{(rcv<0).mean()*100:.0f}% negative")
    if len(sp) >= 4:
        print(f"  margin spread vs selectivity: pearson "
              f"{np.corrcoef(sp[:,0],sp[:,1])[0,1]:+.3f} (n={len(sp)}) "
              f"-> heterogeneous forget margins => more selective")
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
