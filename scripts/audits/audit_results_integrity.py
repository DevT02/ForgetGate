import argparse
import json
import math
import re
from pathlib import Path
from datetime import datetime

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required for audit_results_integrity.py") from exc


EVAL_RE = re.compile(r"eval_(.+)_seed_(\d+)_evaluation\.json$")


def is_finite(val):
    return isinstance(val, (int, float)) and math.isfinite(val)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def audit_eval_files(results_dir, suites):
    issues = {
        "unknown_suite_files": [],
        "missing_models": [],
        "missing_attacks": [],
        "missing_metrics": [],
        "non_finite_metrics": [],
        "out_of_range_metrics": [],
        "error_fields": [],
        "json_decode_errors": [],
    }

    for p in Path(results_dir).rglob("*_evaluation.json"):
        m = EVAL_RE.search(p.name)
        if not m:
            continue
        suite_name = f"eval_{m.group(1)}"
        if suite_name not in suites:
            issues["unknown_suite_files"].append(str(p))
            continue

        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            issues["json_decode_errors"].append(str(p))
            continue

        suite_cfg = suites[suite_name]
        expected_models = suite_cfg.get("model_suites", [])
        expected_attacks = suite_cfg.get("evaluation", {}).get("attacks", [])
        expected_metrics = suite_cfg.get("evaluation", {}).get("metrics", [])

        for model in expected_models:
            if model not in data:
                issues["missing_models"].append(f"{p} :: {model}")
                continue

            model_blob = data[model]
            for attack in expected_attacks:
                if attack not in model_blob:
                    issues["missing_attacks"].append(f"{p} :: {model} :: {attack}")
                    continue

                attack_blob = model_blob.get(attack, {})
                if isinstance(attack_blob, dict) and "error" in attack_blob:
                    issues["error_fields"].append(f"{p} :: {model} :: {attack} :: {attack_blob['error']}")

                for metric in expected_metrics:
                    if metric not in attack_blob:
                        issues["missing_metrics"].append(f"{p} :: {model} :: {attack} :: {metric}")
                        continue
                    val = attack_blob.get(metric)
                    if not is_finite(val):
                        issues["non_finite_metrics"].append(f"{p} :: {model} :: {attack} :: {metric}={val}")
                        continue
                    if metric.endswith("_acc") and (val < 0.0 or val > 1.0):
                        issues["out_of_range_metrics"].append(f"{p} :: {model} :: {attack} :: {metric}={val}")

                # Sanity checks for optional metrics if present
                for metric_key, metric_val in attack_blob.items():
                    if not is_finite(metric_val):
                        continue
                    if metric_key.endswith("_acc") and (metric_val < 0.0 or metric_val > 1.0):
                        issues["out_of_range_metrics"].append(f"{p} :: {model} :: {attack} :: {metric_key}={metric_val}")
                    if metric_key.endswith("_confidence") and (metric_val < 0.0 or metric_val > 1.0):
                        issues["out_of_range_metrics"].append(f"{p} :: {model} :: {attack} :: {metric_key}={metric_val}")
                    if metric_key.endswith("_entropy") and metric_val < 0.0:
                        issues["out_of_range_metrics"].append(f"{p} :: {model} :: {attack} :: {metric_key}={metric_val}")

    return issues


def _checkpoint_exists(path):
    return Path(path).exists()


def audit_checkpoint_consistency(results_dir, suites):
    issues = {
        "missing_checkpoints": [],
        "missing_adapters": [],
    }

    for p in Path(results_dir).rglob("*_evaluation.json"):
        m = EVAL_RE.search(p.name)
        if not m:
            continue
        suite_name = f"eval_{m.group(1)}"
        if suite_name not in suites:
            continue

        seed = int(m.group(2))
        suite_cfg = suites[suite_name]
        expected_models = suite_cfg.get("model_suites", [])

        for model in expected_models:
            if model.startswith("base_"):
                ckpt = f"checkpoints/base/{model}_seed_{seed}_final.pt"
                alt = f"checkpoints/base/{model}_seed_{seed}_best.pt"
                if not (_checkpoint_exists(ckpt) or _checkpoint_exists(alt)):
                    issues["missing_checkpoints"].append(f"{p} :: {model} (seed {seed})")
            elif model.startswith("unlearn_"):
                adapter_dir = Path("checkpoints/unlearn_lora") / f"{model}_seed_{seed}"
                if not adapter_dir.exists():
                    issues["missing_adapters"].append(f"{p} :: {model} (seed {seed})")
            elif model.startswith("vpt_"):
                prompt_dir = Path("checkpoints/vpt_resurrector") / f"{model}_seed_{seed}"
                if not prompt_dir.exists():
                    issues["missing_adapters"].append(f"{p} :: {model} (seed {seed})")
            elif model.startswith("oracle_"):
                oracle_ckpt = f"checkpoints/oracle/{model}_seed_{seed}_final.pt"
                oracle_alt = f"checkpoints/oracle/{model}_seed_{seed}_best.pt"
                if not (_checkpoint_exists(oracle_ckpt) or _checkpoint_exists(oracle_alt)):
                    issues["missing_checkpoints"].append(f"{p} :: {model} (seed {seed})")

    return issues


def audit_jsonl_logs(results_dir, required_keys):
    issues = {
        "json_decode_errors": [],
        "missing_keys": [],
        "non_finite_metrics": [],
        "empty_files": [],
    }

    for p in Path(results_dir).rglob("*.jsonl"):
        try:
            lines = p.read_text(encoding="utf-8").strip().splitlines()
        except Exception:
            issues["json_decode_errors"].append(str(p))
            continue

        if not lines:
            issues["empty_files"].append(str(p))
            continue

        last = lines[-1]
        try:
            obj = json.loads(last)
        except json.JSONDecodeError:
            issues["json_decode_errors"].append(str(p))
            continue

        # Only enforce required keys on VPT-style logs
        if "vpt_" in p.name:
            for k in required_keys:
                if k not in obj:
                    issues["missing_keys"].append(f"{p} :: {k}")
            for k in required_keys:
                if k in obj and not is_finite(obj[k]):
                    issues["non_finite_metrics"].append(f"{p} :: {k}={obj[k]}")

    return issues


def write_report(analysis_dir, report):
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "integrity_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    lines = ["# Results Integrity Report", ""]
    lines.append(f"Generated: {report['generated_at']}")
    lines.append("")

    def section(title, items):
        lines.append(f"## {title}")
        if not items:
            lines.append("- none")
        else:
            lines.extend([f"- {i}" for i in items])
        lines.append("")

    section("Eval: unknown suite files", report["eval"]["unknown_suite_files"])
    section("Eval: JSON decode errors", report["eval"]["json_decode_errors"])
    section("Eval: missing model entries", report["eval"]["missing_models"])
    section("Eval: missing attacks", report["eval"]["missing_attacks"])
    section("Eval: missing metrics", report["eval"]["missing_metrics"])
    section("Eval: non-finite metrics", report["eval"]["non_finite_metrics"])
    section("Eval: out-of-range metrics", report["eval"]["out_of_range_metrics"])
    section("Eval: error fields", report["eval"]["error_fields"])

    section("Training logs: empty files", report["jsonl"]["empty_files"])
    section("Training logs: JSON decode errors", report["jsonl"]["json_decode_errors"])
    section("Training logs: missing keys", report["jsonl"]["missing_keys"])
    section("Training logs: non-finite metrics", report["jsonl"]["non_finite_metrics"])

    section("Checkpoints: missing base/oracle", report["checkpoints"]["missing_checkpoints"])
    section("Checkpoints: missing adapters/prompts", report["checkpoints"]["missing_adapters"])

    (analysis_dir / "integrity_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Integrity audit for eval + training logs")
    ap.add_argument("--results-dir", default="results/logs")
    ap.add_argument("--analysis-dir", default="results/analysis")
    ap.add_argument("--config", default="configs/experiment_suites.yaml")
    args = ap.parse_args()

    suites = load_yaml(args.config)
    eval_issues = audit_eval_files(args.results_dir, suites)
    jsonl_issues = audit_jsonl_logs(
        args.results_dir,
        required_keys=["train_loss", "loss_forget", "loss_retain", "resurrection_acc", "epoch"],
    )

    report = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "eval": eval_issues,
        "jsonl": jsonl_issues,
        "checkpoints": audit_checkpoint_consistency(args.results_dir, suites),
    }

    write_report(Path(args.analysis_dir), report)


if __name__ == "__main__":
    main()
