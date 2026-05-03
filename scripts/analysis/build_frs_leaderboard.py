"""
Build a Forgetting Robustness Score (FRS) leaderboard.

The FRS aggregates the paper's multiple attack regimes into a single number
per method so practitioners can rank choices. It is intentionally simple and
auditable: explicit weights, an explicit retain-drop penalty, and a documented
fallback for missing regimes.

Input schema (JSON list of rows):

    [
      {
        "method": "ORBIT",
        "seed": 42,
        "cond_recovery":  0.78,   # forget recovery rate under per-example
                                  # conditional attack (range [0,1], lower is
                                  # better)
        "patch_recovery": 0.78,   # 32x32 conditional patch
        "frame_recovery": 0.62,   # border-frame conditional attack
        "multi_recovery": 0.78,   # multi-patch conditional attack
        "mia_auc_max":    0.74,   # max(loss-AUC, ent-AUC) from the MIA audit
                                  # (range [0.5, 1.0]; 0.5 is ideal)
        "retain_accuracy": 0.94,  # post-unlearning retain accuracy
        "retain_baseline": 0.95   # base-model retain accuracy on same split
      },
      ...
    ]

Any of the recovery / mia / retain_* fields may be omitted; the corresponding
weight is dropped and the remaining weights are renormalized (so methods with
fewer regimes still produce a comparable FRS, with a "n_regimes" column for
transparency).

Usage:

    python scripts/analysis/build_frs_leaderboard.py \\
        --input  results/analysis/metrics/frs_inputs.json \\
        --output_dir results/analysis

Outputs:
    results/analysis/metrics/frs_leaderboard.json
    results/analysis/figures/frs_leaderboard.tex
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional


# Default weights. Sum is 1.0 by convention; renormalization handles missing.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "cond_recovery":  0.30,
    "patch_recovery": 0.20,
    "frame_recovery": 0.15,
    "multi_recovery": 0.15,
    "mia_auc_max":    0.20,
}

# Maps an FRS-input field to the function that converts it into a [0,1]
# "robustness score" where 1.0 means "ideal".
def _robustness_term(field: str, value: float) -> float:
    if field == "mia_auc_max":
        # AUC is in [0.5, 1.0]; map 0.5 -> 1.0 and 1.0 -> 0.0 linearly.
        clamped = max(0.5, min(1.0, value))
        return max(0.0, 1.0 - 2.0 * (clamped - 0.5))
    # Recovery rates are in [0,1]; lower is better.
    clamped = max(0.0, min(1.0, value))
    return 1.0 - clamped


# Penalty applied per percentage-point of retain-accuracy regression below the
# baseline. 0.05 means: a 5pp drop subtracts 0.25 from the FRS.
DEFAULT_RETAIN_PENALTY_PER_PP: float = 0.05


def compute_row_frs(
    row: Dict,
    weights: Dict[str, float] = None,
    retain_penalty_per_pp: float = DEFAULT_RETAIN_PENALTY_PER_PP,
) -> Dict:
    """Compute FRS for a single (method, seed) row. Returns the row augmented
    with the score, the renormalized weights actually used, and diagnostic
    fields.
    """
    weights = dict(weights or DEFAULT_WEIGHTS)
    used_terms = {f: w for f, w in weights.items() if row.get(f) is not None}
    if not used_terms:
        return {**row, "frs": None, "n_regimes": 0,
                "renormalized_weights": {}, "retain_penalty": 0.0,
                "warning": "no recovery/mia fields present"}

    total_w = sum(used_terms.values())
    renorm = {f: w / total_w for f, w in used_terms.items()}

    base = sum(renorm[f] * _robustness_term(f, row[f]) for f in renorm)

    retain_penalty = 0.0
    if row.get("retain_accuracy") is not None and row.get("retain_baseline") is not None:
        drop_pp = max(0.0, (row["retain_baseline"] - row["retain_accuracy"]) * 100.0)
        retain_penalty = retain_penalty_per_pp * drop_pp

    score = max(0.0, base - retain_penalty)
    return {
        **row,
        "frs": float(score),
        "n_regimes": len(renorm),
        "renormalized_weights": renorm,
        "retain_penalty": float(retain_penalty),
    }


def aggregate_by_method(rows: List[Dict]) -> List[Dict]:
    """Average FRS across seeds within each method."""
    by_method: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        if row.get("frs") is None:
            continue
        by_method[row["method"]].append(row)

    out: List[Dict] = []
    for method, seed_rows in by_method.items():
        scores = [r["frs"] for r in seed_rows]
        n_regimes = max(r["n_regimes"] for r in seed_rows)
        out.append({
            "method": method,
            "n_seeds": len(seed_rows),
            "seeds": sorted(r["seed"] for r in seed_rows if "seed" in r),
            "n_regimes_max": n_regimes,
            "frs_mean": float(sum(scores) / len(scores)),
            "frs_min": float(min(scores)),
            "frs_max": float(max(scores)),
        })
    out.sort(key=lambda r: -r["frs_mean"])
    return out


def render_leaderboard(rows: List[Dict]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Forgetting Robustness Score (FRS) leaderboard. "
        r"FRS aggregates per-example, patch, frame, and multi-patch recovery "
        r"with MIA AUC, weighted as documented in "
        r"\texttt{scripts/analysis/build\_frs\_leaderboard.py}, "
        r"and penalized for retain-accuracy regression. "
        r"Higher is better; 1.0 is unattainable in practice.}",
        r"\label{tab:frs_leaderboard}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Seeds & Regimes & FRS (mean) & FRS (range) \\",
        r"\midrule",
    ]
    for r in rows:
        method = r["method"].replace("_", r"\_")
        rng = f"{r['frs_min']:.3f}--{r['frs_max']:.3f}"
        lines.append(
            f"{method} & {r['n_seeds']} & {r['n_regimes_max']} & "
            f"{r['frs_mean']:.3f} & {rng} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_leaderboard(
    rows: Iterable[Dict],
    weights: Optional[Dict[str, float]] = None,
    retain_penalty_per_pp: float = DEFAULT_RETAIN_PENALTY_PER_PP,
) -> Dict:
    """Pure function: rows -> {per_row, by_method, weights}. Used by tests."""
    rows = list(rows)
    per_row = [compute_row_frs(r, weights=weights,
                               retain_penalty_per_pp=retain_penalty_per_pp)
               for r in rows]
    by_method = aggregate_by_method(per_row)
    return {
        "per_row": per_row,
        "by_method": by_method,
        "weights": dict(weights or DEFAULT_WEIGHTS),
        "retain_penalty_per_pp": retain_penalty_per_pp,
    }


def _validate_input(rows: List[Dict]) -> None:
    if not isinstance(rows, list):
        raise ValueError("Input must be a JSON list of rows")
    required = {"method"}
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise ValueError(f"Row {i}: expected object, got {type(r).__name__}")
        missing = required - set(r.keys())
        if missing:
            raise ValueError(f"Row {i}: missing required field(s): {sorted(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to FRS input JSON")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join("results", "analysis"))
    parser.add_argument("--retain_penalty_per_pp", type=float,
                        default=DEFAULT_RETAIN_PENALTY_PER_PP)
    args = parser.parse_args()

    with open(args.input, "r") as f:
        rows = json.load(f)
    _validate_input(rows)

    result = build_leaderboard(
        rows, retain_penalty_per_pp=args.retain_penalty_per_pp
    )

    metrics_dir = os.path.join(args.output_dir, "metrics")
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    leaderboard_json = os.path.join(metrics_dir, "frs_leaderboard.json")
    with open(leaderboard_json, "w") as f:
        json.dump(result, f, indent=2)

    leaderboard_tex = os.path.join(figures_dir, "frs_leaderboard.tex")
    with open(leaderboard_tex, "w") as f:
        f.write(render_leaderboard(result["by_method"]))

    print(f"Wrote {leaderboard_json}")
    print(f"Wrote {leaderboard_tex}")
    for r in result["by_method"]:
        print(f"  {r['method']:>20s}  FRS={r['frs_mean']:.3f}  "
              f"(seeds={r['n_seeds']}, regimes={r['n_regimes_max']})")


if __name__ == "__main__":
    main()
