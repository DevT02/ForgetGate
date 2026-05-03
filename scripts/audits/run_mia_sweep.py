"""
Sweep the MIA audit over a set of unlearning checkpoints and aggregate the
results into one JSON and one LaTeX table.

Usage:

  # Auto-discover all checkpoints under checkpoints/unlearn/ and audit each
  python scripts/audits/run_mia_sweep.py

  # Restrict to specific suites (repeat --suite). Seeds inferred from dirnames.
  python scripts/audits/run_mia_sweep.py --suite unlearn_salun_vit_cifar10_forget0

  # Explicit (suite, seed) tuples
  python scripts/audits/run_mia_sweep.py --pair unlearn_orbit_vit_cifar10_forget0:42

The sweep is failure-isolating: if one suite raises, the others continue and
the failure is recorded in the summary.
"""

import argparse
import importlib.util
import json
import os
import re
import sys
import traceback
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

_AUDIT_PATH = os.path.join(os.path.dirname(__file__), "26_mia_audit.py")
_spec = importlib.util.spec_from_file_location("mia_audit", _AUDIT_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load {_AUDIT_PATH}")
_mia = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mia)
run_mia = _mia.run_mia


# Method-name → display label, in the order the paper presents them.
PAPER_METHOD_ORDER: List[Tuple[str, str]] = [
    ("salun", "SalUn"),
    ("scrub", "SCRUB"),
    ("rurk", "RURK"),
    ("ceu", "CE-U"),
    ("ce_u", "CE-U"),
    ("sga", "SGA"),
    ("baldro", "BalDRO"),
    ("orbit", "ORBIT"),
    ("kl", "Uniform KL"),
    ("salun_robust_base", "Robust-base SalUn"),
]

_SEED_RE = re.compile(r"_seed_(\d+)")


def discover_checkpoints(root: str) -> List[Tuple[str, int]]:
    """Walk checkpoints/unlearn/ and return (suite, seed) pairs."""
    out: List[Tuple[str, int]] = []
    base = os.path.join(root, "checkpoints", "unlearn")
    if not os.path.isdir(base):
        return out
    for name in sorted(os.listdir(base)):
        m = _SEED_RE.search(name)
        if not m:
            continue
        seed = int(m.group(1))
        suite = name[: m.start()]
        out.append((suite, seed))
    return out


def parse_pair(arg: str) -> Tuple[str, int]:
    if ":" not in arg:
        raise argparse.ArgumentTypeError(f"--pair must be SUITE:SEED, got {arg!r}")
    suite, seed = arg.rsplit(":", 1)
    return suite, int(seed)


def method_label(suite: str) -> str:
    s = suite.lower()
    # Match longest tag first so "salun_robust_base" wins over "salun".
    candidates = sorted(PAPER_METHOD_ORDER, key=lambda kv: -len(kv[0]))
    for tag, label in candidates:
        if tag in s:
            return label
    return suite


def aggregate(results: List[Dict]) -> Dict:
    """Group results by method label and average over seeds."""
    by_method: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        if "error" in r:
            continue
        by_method[method_label(r["suite"])].append(r)

    summary: Dict[str, Dict] = {}
    for label, runs in by_method.items():
        if not runs:
            continue
        vs_val_loss = [r["vs_val_heldout"]["auc_loss"] for r in runs]
        vs_val_ent = [r["vs_val_heldout"]["auc_entropy"] for r in runs]
        vs_test_loss = [r["vs_test"]["auc_loss"] for r in runs]
        vs_test_ent = [r["vs_test"]["auc_entropy"] for r in runs]
        seeds = sorted({r["seed"] for r in runs})
        summary[label] = {
            "n_seeds": len(seeds),
            "seeds": seeds,
            "vs_val_loss_mean": float(sum(vs_val_loss) / len(vs_val_loss)),
            "vs_val_ent_mean": float(sum(vs_val_ent) / len(vs_val_ent)),
            "vs_test_loss_mean": float(sum(vs_test_loss) / len(vs_test_loss)),
            "vs_test_ent_mean": float(sum(vs_test_ent) / len(vs_test_ent)),
            "memorization_gap_max": float(
                max(max(r["vs_val_heldout"]["memorization_gap"],
                        r["vs_test"]["memorization_gap"]) for r in runs)
            ),
        }
    return summary


def render_table(summary: Dict[str, Dict]) -> str:
    """Emit a booktabs LaTeX table sorted by descending memorization gap."""
    rows = sorted(summary.items(), key=lambda kv: -kv[1]["memorization_gap_max"])
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Membership inference attack AUC after unlearning. "
        r"Lower is better; 0.50 is perfect privacy. "
        r"\textit{vs val} compares forget-class samples the base model trained on "
        r"to forget-class samples held out from base training; "
        r"\textit{vs test} compares against the official test split.}",
        r"\label{tab:mia_audit}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & Seeds & vs val (loss) & vs val (ent) & vs test (loss) & vs test (ent) \\",
        r"\midrule",
    ]
    for label, s in rows:
        lines.append(
            f"{label} & {s['n_seeds']} & "
            f"{s['vs_val_loss_mean']:.3f} & {s['vs_val_ent_mean']:.3f} & "
            f"{s['vs_test_loss_mean']:.3f} & {s['vs_test_ent_mean']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="MIA sweep over unlearning checkpoints")
    parser.add_argument("--config", type=str, default="configs/experiment_suites.yaml")
    parser.add_argument("--suite", action="append", default=None,
                        help="Suite name. Repeatable. Seeds discovered from checkpoint dirs.")
    parser.add_argument("--pair", type=parse_pair, action="append", default=None,
                        help="Explicit SUITE:SEED tuples. Repeatable.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join("results", "analysis"))
    args = parser.parse_args()

    if args.pair:
        targets = list(args.pair)
    else:
        all_pairs = discover_checkpoints(ROOT)
        if args.suite:
            wanted = set(args.suite)
            targets = [(s, sd) for s, sd in all_pairs if s in wanted]
        else:
            targets = all_pairs

    if not targets:
        print("No (suite, seed) pairs to audit. Specify --suite, --pair, or place "
              "checkpoints under checkpoints/unlearn/.")
        sys.exit(1)

    print(f"Sweeping MIA over {len(targets)} (suite, seed) pairs...")
    results: List[Dict] = []
    for suite, seed in targets:
        try:
            r = run_mia(suite, config_path=args.config, batch_size=args.batch_size, seed=seed)
        except Exception as e:
            print(f"[mia] {suite} seed={seed}: FAILED -- {e}")
            traceback.print_exc()
            r = {"suite": suite, "seed": seed, "error": str(e)}
        results.append(r)

    summary = aggregate(results)

    metrics_dir = os.path.join(ROOT, args.out_dir, "metrics")
    figures_dir = os.path.join(ROOT, args.out_dir, "figures")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    summary_path = os.path.join(metrics_dir, "mia_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"per_run": results, "by_method": summary}, f, indent=2)

    table_path = os.path.join(figures_dir, "mia_audit_table.tex")
    with open(table_path, "w") as f:
        f.write(render_table(summary))

    print(f"\nWrote {summary_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
