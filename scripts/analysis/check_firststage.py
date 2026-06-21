"""Quick first-stage reader for a knob screen: given audit tags and their knob
values, print r_R/r_F per point and whether the knob moves r_R (the retain-control
radius). A knob is worth escalating only if it moves r_R with a clear monotone
trend. Usage: check_firststage.py seed tag1=val1 tag2=val2 ..."""
import json, os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load(tag, seed):
    p = os.path.join(ROOT, "results", "analysis",
                     f"recovery_radius_cifar10_vit_tiny_forget0_seed_{seed}"
                     f"_test_with_retain_adam_margin_{tag}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    fr = list(d["forget_recovery"].values())[0]["summary"]
    rc = list(d["retain_control"].values())[0]["summary"]
    return rc["mean_radius"], fr["mean_radius"], rc["median_radius"], fr["median_radius"]


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    seed = int(sys.argv[1])
    vals, rR, rF = [], [], []
    n_censored = 0
    print(f"{'knob':>10} {'r_R mean':>10} {'r_F mean':>10} {'r_R med':>10} {'r_F med':>10}")
    for arg in sys.argv[2:]:
        tag, v = arg.split("=")
        r = load(tag, seed)
        if r is None:
            print(f"{v:>10}  (missing: {tag})")
            continue
        rrm, rfm, rrmd, rfmd = r
        print(f"{v:>10} {rrm:>10.5f} {rfm:>10.5f} {rrmd:>10.5f} {rfmd:>10.5f}")
        # drop right-censored points (median NaN => attack never reached the radius)
        if not np.isfinite(rrm) or not np.isfinite(rfm):
            n_censored += 1
            continue
        vals.append(float(v)); rR.append(rrm); rF.append(rfm)
    if n_censored:
        print(f"\n[{n_censored} point(s) right-censored (r_R median NaN) -> dropped; "
              f"raise --eps-max to measure]")
    if len(rR) >= 3:
        rng = (max(rR) - min(rR)) / (np.mean(rR) + 1e-12)
        print(f"on {len(rR)} measurable points: r_R range/mean = {rng:.3f}   "
              f"spearman(knob, r_R) = {spearman(vals, rR):+.3f}   "
              f"spearman(knob, r_F) = {spearman(vals, rF):+.3f}")
        print("VERDICT:", "MOVES r_R (escalate)" if rng > 0.15 else "FLAT r_R (knob weak)")
    else:
        print(f"\nonly {len(rR)} measurable points -- cannot assess first stage")


if __name__ == "__main__":
    main()
