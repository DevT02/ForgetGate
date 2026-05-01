#!/usr/bin/env python3
"""
Summarize utility-vs-recovery tradeoffs for trusted unlearning baselines.

This script collects utility metrics from unlearning history logs and recovery
metrics from per-example audit outputs, then writes JSON and Markdown summary
artifacts for the main trusted methods.
"""

import argparse
import glob
import json
import math
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils import ensure_dir


METHODS = [
    {
        "suite": "unlearn_orbit_vit_cifar10_forget0",
        "label": "ORBIT",
        "group": "non_robust",
    },
    {
        "suite": "unlearn_salun_vit_cifar10_forget0",
        "label": "SalUn",
        "group": "non_robust",
    },
    {
        "suite": "unlearn_baldro_vit_cifar10_forget0_smoke",
        "label": "BalDRO",
        "group": "non_robust",
    },
    {
        "suite": "unlearn_rurk_vit_cifar10_forget0_smoke",
        "label": "RURK",
        "group": "non_robust",
    },
    {
        "suite": "unlearn_salun_robust_base_forget0_smoke",
        "label": "SalUn + Robust Base (smoke)",
        "group": "robust_base",
    },
    {
        "suite": "unlearn_salun_robust_base_forget0_mid",
        "label": "SalUn + Robust Base (mid)",
        "group": "robust_base",
    },
    {
        "suite": "unlearn_salun_robust_base_forget0_high",
        "label": "SalUn + Robust Base (high)",
        "group": "robust_base",
    },
    {
        "suite": "unlearn_salun_robust_base_forget0_ultra",
        "label": "SalUn + Robust Base (ultra)",
        "group": "robust_base",
    },
]

ALLOWED_ATTACKS = {"adam_margin", "apgd_margin"}


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _history_metrics(suite: str, seed: int) -> Optional[Dict]:
    history_path = os.path.join("results", "logs", f"{suite}_seed_{seed}_history.json")
    if not os.path.exists(history_path):
        return None
    history = _load_json(history_path)
    if not history:
        return None
    last = history[-1]
    return {
        "history_path": history_path,
        "val_acc": last.get("val_acc"),
        "retain_acc": last.get("retain_acc"),
        "forget_acc": last.get("forget_acc"),
        "epoch": last.get("epoch"),
    }


def _suite_attack_candidates(suite: str, seed: int) -> List[Dict]:
    candidates = []
    pattern = os.path.join("results", "analysis", f"recovery_radius_*_seed_{seed}_*.json")
    for path in glob.glob(pattern):
        data = _load_json(path)
        meta = data.get("meta", {})
        attack_loss = meta.get("attack_loss")
        if attack_loss not in ALLOWED_ATTACKS:
            continue
        suites = meta.get("model_suites", [])
        if suite not in suites:
            continue

        forget_summary = data.get("forget_recovery", {}).get(suite, {}).get("summary", {})
        retain_summary = data.get("retain_control", {}).get(suite, {}).get("summary", {})
        if not forget_summary:
            continue

        attack_tag = attack_loss
        alpha_ratio = meta.get("alpha_ratio")
        if attack_loss == "adam_margin" and alpha_ratio is not None and abs(float(alpha_ratio) - 0.25) > 1e-12:
            attack_tag = f"{attack_tag}_ar{str(alpha_ratio).replace('.', 'p')}"

        candidates.append(
            {
                "path": path,
                "attack": attack_tag,
                "attack_loss": attack_loss,
                "forget_success_rate": forget_summary.get("success_rate"),
                "forget_median_radius": forget_summary.get("median_radius"),
                "retain_success_rate": retain_summary.get("success_rate"),
                "retain_median_radius": retain_summary.get("median_radius"),
                "n_forget": forget_summary.get("n_samples"),
                "n_retain": retain_summary.get("n_samples"),
            }
        )
    return candidates


def _is_nan(value) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _candidate_sort_key(candidate: Dict):
    forget_success = float(candidate.get("forget_success_rate") or 0.0)
    n_forget = int(candidate.get("n_forget") or 0)
    forget_median = candidate.get("forget_median_radius")
    if forget_median is None or _is_nan(forget_median):
        forget_median = float("inf")
    attack_pref = 1 if candidate.get("attack_loss") == "adam_margin" else 0
    return (forget_success, n_forget, -forget_median, attack_pref)


def _best_candidate(candidates: List[Dict]) -> Optional[Dict]:
    if not candidates:
        return None
    return max(candidates, key=_candidate_sort_key)


def _format_metric(value, precision: int = 4) -> str:
    if value is None:
        return "NA"
    if _is_nan(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def _write_markdown(rows: List[Dict], save_path: str):
    ensure_dir(os.path.dirname(save_path))
    lines = [
        "# Recovery Tradeoff Summary",
        "",
        "| Seed | Method | Group | Val Acc | Retain Acc | Forget Acc | Attack | Forget Success | Forget Median | Retain Success | Retain Median |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["seed"]),
                    row["label"],
                    row["group"],
                    _format_metric(row.get("val_acc")),
                    _format_metric(row.get("retain_acc")),
                    _format_metric(row.get("forget_acc")),
                    row.get("best_attack") or "NA",
                    _format_metric(row.get("forget_success_rate")),
                    _format_metric(row.get("forget_median_radius")),
                    _format_metric(row.get("retain_success_rate")),
                    _format_metric(row.get("retain_median_radius")),
                ]
            )
            + " |"
        )
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 123])
    args = parser.parse_args()

    rows: List[Dict] = []
    for seed in args.seeds:
        for method in METHODS:
            suite = method["suite"]
            history = _history_metrics(suite, seed)
            candidates = _suite_attack_candidates(suite, seed)
            best = _best_candidate(candidates)
            row = {
                "seed": seed,
                "suite": suite,
                "label": method["label"],
                "group": method["group"],
                "history_path": history["history_path"] if history else None,
                "val_acc": history["val_acc"] if history else None,
                "retain_acc": history["retain_acc"] if history else None,
                "forget_acc": history["forget_acc"] if history else None,
                "epoch": history["epoch"] if history else None,
                "best_attack": best["attack"] if best else None,
                "attack_path": best["path"] if best else None,
                "forget_success_rate": best["forget_success_rate"] if best else None,
                "forget_median_radius": best["forget_median_radius"] if best else None,
                "retain_success_rate": best["retain_success_rate"] if best else None,
                "retain_median_radius": best["retain_median_radius"] if best else None,
                "candidate_attacks": candidates,
            }
            rows.append(row)

    save_json_path = os.path.join("results", "analysis", "recovery_tradeoff_summary.json")
    save_md_path = os.path.join("results", "analysis", "recovery_tradeoff_summary.md")
    ensure_dir(os.path.dirname(save_json_path))
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2)
    _write_markdown(rows, save_md_path)

    print(f"Wrote JSON summary to: {save_json_path}")
    print(f"Wrote Markdown summary to: {save_md_path}")


if __name__ == "__main__":
    main()
