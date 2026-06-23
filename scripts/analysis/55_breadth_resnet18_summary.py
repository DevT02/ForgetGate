"""Consolidate the ResNet-18/CIFAR-100 breadth experiment for Theorem 4.

Reads, per seed, the three artifacts produced by the breadth pipeline:
  - L_inf recovery audit:  results/analysis/recovery_radius_cifar100_resnet18_forget0_seed_{S}_test_with_retain_adam_margin_smoothedmargin-resnet18-breadth-forget0.json
  - certified L2 audit:    results/analysis/metrics/smoothed_radius_seed_{S}_sigma_0p1_n256_smoothedmargin_resnet18_breadth_forget0.json
  - training history:      results/logs/unlearn_smoothedmargin_resnet18_breadth_forget0_seed_{S}_history.json
and writes a single summary JSON + prints a table.

Honest reporting: certified fraction is reported as measured (NOT assumed 1.0);
selectivity ratio = median retain recovery radius / median forget recovery radius.

Output: results/analysis/metrics/breadth_resnet18_cifar100_summary.json
"""
import json, os, glob

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SUITE = "unlearn_smoothedmargin_resnet18_breadth_forget0"
SEEDS = [123, 456]


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def _last_epoch(hist_path):
    h = _load(hist_path)
    if h is None:
        return None
    H = h if isinstance(h, list) else h.get("history", [])
    return H[-1] if H else None


def per_seed(seed):
    rec_p = os.path.join(
        ROOT, "results", "analysis",
        f"recovery_radius_cifar100_resnet18_forget0_seed_{seed}_test_with_retain_adam_margin_smoothedmargin-resnet18-breadth-forget0.json")
    cert_p = os.path.join(
        ROOT, "results", "analysis", "metrics",
        f"smoothed_radius_seed_{seed}_sigma_0p1_n256_smoothedmargin_resnet18_breadth_forget0.json")
    hist_p = os.path.join(
        ROOT, "results", "logs",
        f"unlearn_smoothedmargin_resnet18_breadth_forget0_seed_{seed}_history.json")

    rec, cert, last = _load(rec_p), _load(cert_p), _last_epoch(hist_p)
    if rec is None or cert is None:
        return None

    f = rec["forget_recovery"][SUITE]["summary"]
    r = rec["retain_control"][SUITE]["summary"]
    c = cert["forget_recovery"][SUITE]["summary"]
    sel = r["median_radius"] / f["median_radius"] if f["median_radius"] > 0 else None
    return {
        "seed": seed,
        "dataset": rec["meta"]["dataset"],
        "model": rec["meta"]["model"],
        "forget_acc_after": (last or {}).get("forget_acc"),
        "retain_acc_after": (last or {}).get("retain_acc"),
        "forget_recovery_success": f["success_rate"],
        "forget_clean_leak": f["clean_success_rate"],
        "forget_median_radius_linf": f["median_radius"],
        "retain_median_radius_linf": r["median_radius"],
        "selectivity_ratio_linf": sel,
        "certified_l2_median_R": c["median_certified_radius_l2"],
        "certified_frac_positive": c["frac_with_positive_certificate"],
    }


def main():
    rows = [x for x in (per_seed(s) for s in SEEDS) if x is not None]
    out = {
        "description": "ResNet-18/CIFAR-100 breadth check for the Theorem 4 constructive result (forget class 0, sigma=0.10). A different architecture FAMILY (convolutional) and harder dataset than the CIFAR-10/ViT-tiny core.",
        "n_seeds": len(rows),
        "per_seed": rows,
    }
    if rows:
        def agg(k):
            vals = [x[k] for x in rows if x[k] is not None]
            return {"mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals)} if vals else None
        out["aggregate"] = {k: agg(k) for k in [
            "forget_acc_after", "retain_acc_after", "forget_median_radius_linf",
            "retain_median_radius_linf", "selectivity_ratio_linf",
            "certified_l2_median_R", "certified_frac_positive"]}

    op = os.path.join(ROOT, "results", "analysis", "metrics", "breadth_resnet18_cifar100_summary.json")
    json.dump(out, open(op, "w"), indent=2)
    print("wrote", op, f"({len(rows)} seed(s))")
    print(f"\n{'seed':>5} {'fAcc':>6} {'rAcc':>6} {'fRad':>9} {'rRad':>9} {'sel x':>6} {'certR':>7} {'cert%':>6}")
    for x in rows:
        print(f"{x['seed']:>5} {x['forget_acc_after']:>6.3f} {x['retain_acc_after']:>6.3f} "
              f"{x['forget_median_radius_linf']:>9.5f} {x['retain_median_radius_linf']:>9.5f} "
              f"{x['selectivity_ratio_linf']:>6.2f} {x['certified_l2_median_R']:>7.3f} "
              f"{x['certified_frac_positive']:>6.2f}")


if __name__ == "__main__":
    main()
