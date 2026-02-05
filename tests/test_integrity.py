import json
import math
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "logs"
CONFIG_PATH = ROOT / "configs" / "experiment_suites.yaml"


REQUIRED_EVAL_SEEDS = [42, 123, 456]

# Only enforce integrity for core suites we actually run.
REQUIRED_EVAL_SUITES = {
    "eval_paper_baselines_vit_cifar10_forget0": [42, 123, 456],
    "eval_autoattack_vit_cifar10_forget0": [42, 123, 456],
    "eval_dependency_vit_cifar10_forget0": [42, 123, 456],
    "eval_prune_kl_vit_cifar10_forget0": [42],
    "eval_backdoor_vit_cifar10_forget0": [42],
}

# Models that may be absent for a given suite/seed (e.g., missing VPT for seed 456)
OPTIONAL_MODELS = {
    "eval_autoattack_vit_cifar10_forget0": {"vpt_resurrect_vit_cifar10_forget0"},
    "eval_dependency_vit_cifar10_forget0": {"vpt_resurrect_vit_cifar10_forget0"},
}


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _is_finite(val):
    return isinstance(val, (int, float)) and math.isfinite(val)


def _eval_path(suite, seed):
    return RESULTS_DIR / f"{suite}_seed_{seed}_evaluation.json"


def _jsonl_path(name):
    return RESULTS_DIR / name


def _checkpoint_exists(path):
    return Path(path).exists()


def _get_eval_suites(cfg):
    return {k: v for k, v in cfg.items() if k in REQUIRED_EVAL_SUITES}


def _expected_eval_files(cfg):
    eval_suites = _get_eval_suites(cfg)
    expected = []
    for suite in eval_suites:
        for seed in REQUIRED_EVAL_SUITES[suite]:
            expected.append(_eval_path(suite, seed))
    return expected


def _required_vpt_jsonl(cfg):
    # Minimal coverage for core results
    kshots = [10, 25, 50, 100]
    files = []
    for seed in REQUIRED_EVAL_SEEDS:
        for k in kshots:
            files.append(_jsonl_path(f"vpt_oracle_vit_cifar10_forget0_{k}shot_seed_{seed}.jsonl"))
            files.append(_jsonl_path(f"vpt_resurrect_kl_forget0_{k}shot_seed_{seed}.jsonl"))
    return files


def _parse_eval_file(path, suite_cfg):
    data = _load_json(path)
    expected_models = suite_cfg.get("model_suites", [])
    expected_attacks = suite_cfg.get("evaluation", {}).get("attacks", [])
    expected_metrics = suite_cfg.get("evaluation", {}).get("metrics", [])
    return data, expected_models, expected_attacks, expected_metrics


def test_expected_eval_files_exist():
    cfg = _load_yaml(CONFIG_PATH)
    missing = [p for p in _expected_eval_files(cfg) if not p.exists()]
    assert not missing, f"Missing eval logs: {missing}"


def test_eval_files_have_expected_models_attacks_metrics():
    cfg = _load_yaml(CONFIG_PATH)
    eval_suites = _get_eval_suites(cfg)
    errors = []
    for suite, suite_cfg in eval_suites.items():
            for seed in REQUIRED_EVAL_SUITES[suite]:
                path = _eval_path(suite, seed)
                if not path.exists():
                    errors.append(f"{path} missing")
                    continue
                data, models, attacks, metrics = _parse_eval_file(path, suite_cfg)
                for model in models:
                    if model in OPTIONAL_MODELS.get(suite, set()) and model not in data:
                        continue
                    if model not in data:
                        errors.append(f"{path} missing model {model}")
                        continue
                    blob = data[model]
                    for attack in attacks:
                        # VPT-only attacks should only exist on VPT models
                        if attack in {"vpt_only", "vpt_plus_pgd"} and "vpt" not in model:
                            continue
                        if attack == "clean" and "vpt" in model and "vpt_only" in attacks:
                            # clean is optional when vpt_only is present
                            if attack not in blob:
                                continue
                        if attack not in blob:
                            errors.append(f"{path} {model} missing attack {attack}")
                            continue
                        attack_blob = blob.get(attack, {})
                        for metric in metrics:
                            # resurrection_success_rate only applies to VPT attacks
                            if metric == "resurrection_success_rate" and attack not in {"vpt_only", "vpt_plus_pgd"}:
                                continue
                            if metric not in attack_blob:
                                errors.append(f"{path} {model} {attack} missing metric {metric}")
    assert not errors, "Eval schema issues:\n" + "\n".join(errors)


def test_eval_metrics_are_finite_and_in_range():
    cfg = _load_yaml(CONFIG_PATH)
    eval_suites = _get_eval_suites(cfg)
    errors = []
    for suite, suite_cfg in eval_suites.items():
            for seed in REQUIRED_EVAL_SUITES[suite]:
                path = _eval_path(suite, seed)
                if not path.exists():
                    continue
            data, models, attacks, _metrics = _parse_eval_file(path, suite_cfg)
            for model in models:
                if model not in data:
                    continue
                blob = data[model]
                for attack in attacks:
                    if attack not in blob:
                        continue
                    attack_blob = blob.get(attack, {})
                    if isinstance(attack_blob, dict) and "error" in attack_blob:
                        errors.append(f"{path} {model} {attack} error: {attack_blob['error']}")
                        continue
                    for k, v in attack_blob.items():
                        if isinstance(v, (int, float)):
                            if not _is_finite(v):
                                errors.append(f"{path} {model} {attack} {k}={v}")
                            if k.endswith("_acc") and (v < 0.0 or v > 1.0):
                                errors.append(f"{path} {model} {attack} {k} out of range: {v}")
                            if k.endswith("_confidence") and (v < 0.0 or v > 1.0):
                                errors.append(f"{path} {model} {attack} {k} out of range: {v}")
                            if k.endswith("_entropy") and v < 0.0:
                                errors.append(f"{path} {model} {attack} {k} negative: {v}")
    assert not errors, "Eval metric issues:\n" + "\n".join(errors)


def test_vpt_jsonl_logs_have_required_keys():
    required_keys = {"train_loss", "loss_forget", "loss_retain", "resurrection_acc", "epoch"}
    missing = []
    for path in _required_vpt_jsonl(_load_yaml(CONFIG_PATH)):
        if not path.exists():
            missing.append(f"{path} missing")
            continue
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            missing.append(f"{path} empty")
            continue
        obj = json.loads(lines[-1])
        for key in required_keys:
            if key not in obj:
                missing.append(f"{path} missing {key}")
    assert not missing, "VPT log issues:\n" + "\n".join(missing)


def test_checkpoints_exist_for_eval_models():
    cfg = _load_yaml(CONFIG_PATH)
    eval_suites = _get_eval_suites(cfg)
    errors = []
    for suite, suite_cfg in eval_suites.items():
        for seed in REQUIRED_EVAL_SUITES[suite]:
            path = _eval_path(suite, seed)
            if not path.exists():
                continue
            for model in suite_cfg.get("model_suites", []):
                if model in OPTIONAL_MODELS.get(suite, set()):
                    # skip optional models when not present on disk
                    prompt_dir = ROOT / "checkpoints" / "vpt_resurrector" / f"{model}_seed_{seed}"
                    if not prompt_dir.exists():
                        continue
                if model.startswith("base_"):
                    ckpt = ROOT / "checkpoints" / "base" / f"{model}_seed_{seed}_final.pt"
                    alt = ROOT / "checkpoints" / "base" / f"{model}_seed_{seed}_best.pt"
                    if not (ckpt.exists() or alt.exists()):
                        errors.append(f"{model} seed {seed} missing base checkpoint")
                elif model.startswith("unlearn_"):
                    adapter_dir = ROOT / "checkpoints" / "unlearn_lora" / f"{model}_seed_{seed}"
                    if not adapter_dir.exists():
                        errors.append(f"{model} seed {seed} missing adapter dir")
                elif model.startswith("vpt_"):
                    prompt_dir = ROOT / "checkpoints" / "vpt_resurrector" / f"{model}_seed_{seed}"
                    if not prompt_dir.exists():
                        # VPT evals are not required for every seed
                        errors.append(f"{model} seed {seed} missing prompt dir")
                elif model.startswith("oracle_"):
                    ckpt = ROOT / "checkpoints" / "oracle" / f"{model}_seed_{seed}_final.pt"
                    alt = ROOT / "checkpoints" / "oracle" / f"{model}_seed_{seed}_best.pt"
                    if not (ckpt.exists() or alt.exists()):
                        errors.append(f"{model} seed {seed} missing oracle checkpoint")
    assert not errors, "Checkpoint issues:\n" + "\n".join(errors)
