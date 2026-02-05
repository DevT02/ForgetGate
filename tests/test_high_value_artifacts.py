import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "results" / "logs"
ANALYSIS = ROOT / "results" / "analysis"
CHECKPOINTS = ROOT / "checkpoints"

SEEDS = [42, 123, 456]

NOISE_STDS = ["0.0", "0.01", "0.02", "0.05"]
LAMBDAS = ["0.0", "0.1", "0.5", "1.0", "2.0", "5.0"]
LAYERS = ["early", "mid", "late", "final"]


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def _is_finite(val):
    return isinstance(val, (int, float)) and math.isfinite(val)


def test_benign_relearn_checkpoints_exist():
    missing = []
    for seed in SEEDS:
        ckpt = CHECKPOINTS / "benign_relearn" / f"benign_relearn_vit_cifar10_forget0_seed_{seed}_final.pt"
        if not ckpt.exists():
            missing.append(str(ckpt))
    assert not missing, "Missing benign relearn checkpoints:\n" + "\n".join(missing)


def test_feature_probe_logs_exist():
    missing = []
    for seed in SEEDS:
        base = LOGS / f"feature_probe_probe_feature_leakage_vit_cifar10_forget0_seed_{seed}.json"
        final = LOGS / f"feature_probe_probe_feature_leakage_vit_cifar10_forget0_seed_{seed}_layer_final.json"
        if not base.exists() and not final.exists():
            missing.append(str(base))
        for layer in LAYERS:
            path = LOGS / f"feature_probe_probe_feature_leakage_vit_cifar10_forget0_seed_{seed}_layer_{layer}.json"
            if not path.exists():
                missing.append(str(path))
    assert not missing, "Missing feature-probe logs:\n" + "\n".join(missing)


def test_feature_probe_logs_parse():
    bad = []
    for seed in SEEDS:
        for layer in LAYERS:
            path = LOGS / f"feature_probe_probe_feature_leakage_vit_cifar10_forget0_seed_{seed}_layer_{layer}.json"
            if not path.exists():
                continue
            try:
                _load_json(path)
            except Exception:
                bad.append(str(path))
    assert not bad, "Invalid feature-probe JSON:\n" + "\n".join(bad)


def test_feature_probe_schema_and_ranges():
    errors = []
    for seed in SEEDS:
        for layer in LAYERS:
            path = LOGS / f"feature_probe_probe_feature_leakage_vit_cifar10_forget0_seed_{seed}_layer_{layer}.json"
            if not path.exists():
                continue
            data = _load_json(path)
            if "models" not in data:
                errors.append(f"{path} missing models")
                continue
            for model, blob in data["models"].items():
                for k in ["train_acc", "val_acc", "val_auc"]:
                    if k not in blob or not _is_finite(blob[k]):
                        errors.append(f"{path} {model} missing/invalid {k}")
                        continue
                    if blob[k] < 0.0 or blob[k] > 1.0:
                        errors.append(f"{path} {model} {k} out of range: {blob[k]}")
                for k in ["n_samples", "n_train", "n_val", "feature_dim"]:
                    if k not in blob:
                        errors.append(f"{path} {model} missing {k}")
                        continue
                    if not isinstance(blob[k], int) or blob[k] <= 0:
                        errors.append(f"{path} {model} {k} invalid: {blob[k]}")
                if all(k in blob for k in ["n_samples", "n_train", "n_val"]):
                    if blob["n_train"] + blob["n_val"] != blob["n_samples"]:
                        errors.append(f"{path} {model} n_train+n_val != n_samples")
    assert not errors, "Feature-probe schema/range issues:\n" + "\n".join(errors)


def test_feature_probe_base_matches_final_when_both_exist():
    errors = []
    for seed in SEEDS:
        base = LOGS / f"feature_probe_probe_feature_leakage_vit_cifar10_forget0_seed_{seed}.json"
        final = LOGS / f"feature_probe_probe_feature_leakage_vit_cifar10_forget0_seed_{seed}_layer_final.json"
        if not (base.exists() and final.exists()):
            continue
        base_data = _load_json(base).get("models", {})
        final_data = _load_json(final).get("models", {})
        for model in base_data:
            if model not in final_data:
                errors.append(f"{final} missing model {model}")
                continue
            for k in ["val_acc", "val_auc"]:
                if k in base_data[model] and k in final_data[model]:
                    if abs(base_data[model][k] - final_data[model][k]) > 1e-6:
                        errors.append(f"{model} {k} differs between base and final")
    assert not errors, "Base vs final probe mismatch:\n" + "\n".join(errors)


def test_forget_neighborhood_eval_logs_exist():
    missing = []
    for seed in SEEDS:
        path = LOGS / f"eval_forget_neighborhood_vit_cifar10_forget0_seed_{seed}_evaluation.json"
        if not path.exists():
            missing.append(str(path))
    assert not missing, "Missing forget-neighborhood eval logs:\n" + "\n".join(missing)


def test_forget_neighborhood_eval_schema():
    errors = []
    for seed in SEEDS:
        path = LOGS / f"eval_forget_neighborhood_vit_cifar10_forget0_seed_{seed}_evaluation.json"
        if not path.exists():
            continue
        data = _load_json(path)
        if not data:
            errors.append(f"{path} empty")
            continue
        for model in [
            "base_vit_cifar10",
            "unlearn_lora_vit_cifar10_forget0",
            "vpt_resurrect_vit_cifar10_forget0",
        ]:
            if model not in data:
                errors.append(f"{path} missing model {model}")
                continue
            if "forget_neighborhood" not in data[model]:
                errors.append(f"{path} {model} missing forget_neighborhood")
                continue
            blob = data[model]["forget_neighborhood"]
            # Basic fields
            for k in ["forget_acc", "retain_acc", "overall_acc"]:
                if k not in blob or not _is_finite(blob[k]):
                    errors.append(f"{path} {model} missing/invalid {k}")
                    continue
                if blob[k] < 0.0 or blob[k] > 1.0:
                    errors.append(f"{path} {model} {k} out of range: {blob[k]}")
            # Forget-only eval should have retain_samples = 0 and retain_acc = 0.0
            if blob.get("retain_samples", 0) != 0:
                errors.append(f"{path} {model} retain_samples != 0")
            if blob.get("retain_acc", 0) != 0:
                errors.append(f"{path} {model} retain_acc != 0")
    assert not errors, "Forget-neighborhood schema issues:\n" + "\n".join(errors)


def test_forget_neighborhood_sweep_summaries():
    missing = []
    errors = []
    for seed in SEEDS:
        path = ANALYSIS / f"forget_neighborhood_sweep_seed_{seed}.json"
        if not path.exists():
            missing.append(str(path))
            continue
        data = _load_json(path)
        results = data.get("results", {})
        for noise in NOISE_STDS:
            if noise not in results:
                errors.append(f"{path} missing noise_std={noise}")
        # Cross-check against per-noise logs if present
        for noise in NOISE_STDS:
            tag = noise.replace(".", "p")
            log = LOGS / f"eval_forget_neighborhood_vit_cifar10_forget0_seed_{seed}_noise{tag}_evaluation.json"
            if log.exists() and noise in results:
                if results[noise] != _load_json(log):
                    errors.append(f"{path} does not match {log}")
    assert not missing, "Missing sweep summaries:\n" + "\n".join(missing)
    assert not errors, "Sweep summary schema issues:\n" + "\n".join(errors)


def test_vpt_tradeoff_sweep_summaries():
    missing = []
    errors = []
    for seed in SEEDS:
        path = ANALYSIS / f"vpt_tradeoff_sweep_seed_{seed}.json"
        if not path.exists():
            missing.append(str(path))
            continue
        data = _load_json(path)
        results = data.get("results", {})
        for lam in LAMBDAS:
            if lam not in results:
                errors.append(f"{path} missing lambda={lam}")
                continue
            if "resurrection_acc" not in results[lam]:
                errors.append(f"{path} lambda={lam} missing resurrection_acc")
            # Verify summary matches last log entry
            tag = lam.replace(".", "p")
            log = LOGS / f"vpt_resurrect_kl_forget0_10shot_lam{tag}_seed_{seed}.jsonl"
            if log.exists():
                lines = log.read_text(encoding="utf-8").strip().splitlines()
                if lines:
                    last = json.loads(lines[-1])
                    for k, v in results[lam].items():
                        if k in last and last[k] != v:
                            errors.append(f"{path} lambda={lam} {k} mismatch vs log")
    assert not missing, "Missing tradeoff sweep summaries:\n" + "\n".join(missing)
    assert not errors, "Tradeoff sweep schema issues:\n" + "\n".join(errors)
