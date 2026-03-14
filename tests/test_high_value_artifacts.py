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


def _load_last_jsonl(path: Path):
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines, f"{path} is empty"
    return json.loads(lines[-1])


def _mean_resurrection_acc(paths):
    values = [_load_last_jsonl(path)["resurrection_acc"] for path in paths]
    return round(sum(values) / len(values) * 100.0, 2)


def _scrub_log(k: int, seed: int) -> Path:
    canonical = LOGS / f"vpt_resurrect_scrub_forget0_{k}shot_seed_{seed}.jsonl"
    override = LOGS / f"vpt_resurrect_scrub_forget0_10shot_kshot{k}_seed_{seed}.jsonl"
    if canonical.exists():
        return canonical
    assert override.exists(), f"Missing SCRUB k={k} log for seed {seed}: {canonical} or {override}"
    return override


def _select_prompt5_variant(prefix: str, k: int):
    candidates = [
        [LOGS / f"{prefix}_10shot_prompt5_kshot{k}_seed_{seed}.jsonl" for seed in SEEDS],
        [LOGS / f"{prefix}_{k}shot_prompt5_seed_{seed}.jsonl" for seed in SEEDS],
    ]
    best_paths = None
    best_missing = None
    for paths in candidates:
        missing = [path for path in paths if not path.exists()]
        if best_paths is None or len(missing) < len(best_missing):
            best_paths = paths
            best_missing = missing
    assert best_paths is not None
    assert not best_missing, f"Incomplete prompt-length-5 k={k} bundle: {best_missing}"
    return best_paths


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


def test_headline_kshot_claims_match_tracked_logs():
    expected_default = {10: 0.47, 25: 0.43, 50: 0.43, 100: 1.90}
    expected_low_shot = {1: 4.27, 5: 4.53}

    for k, expected in expected_default.items():
        oracle_paths = [LOGS / f"vpt_oracle_vit_cifar10_forget0_{k}shot_seed_{seed}.jsonl" for seed in SEEDS]
        kl_paths = [LOGS / f"vpt_resurrect_kl_forget0_{k}shot_seed_{seed}.jsonl" for seed in SEEDS]
        assert _mean_resurrection_acc(oracle_paths) == 0.00
        assert _mean_resurrection_acc(kl_paths) == expected

    scrub_expected = {}
    for k in [10, 25, 50, 100]:
        scrub_paths = []
        for seed in SEEDS:
            try:
                scrub_paths.append(_scrub_log(k, seed))
            except AssertionError:
                pass
        if scrub_paths:
            scrub_expected[k] = _mean_resurrection_acc(scrub_paths)

    for k, expected in expected_low_shot.items():
        oracle_paths = _select_prompt5_variant("vpt_oracle_vit_cifar10_forget0", k)
        kl_paths = _select_prompt5_variant("vpt_resurrect_kl_forget0", k)
        assert _mean_resurrection_acc(oracle_paths) == 0.00
        assert _mean_resurrection_acc(kl_paths) == expected

    summary = (ANALYSIS / "kshot_summary.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    expected_lines = [
        "| 10 | 0.00% +/- 0.00% | 0.47% +/- 0.64% |",
        "| 25 | 0.00% +/- 0.00% | 0.43% +/- 0.40% |",
        "| 50 | 0.00% +/- 0.00% | 0.43% +/- 0.25% |",
        "| 100 | 0.00% +/- 0.00% | 1.90% +/- 0.75% |",
        "| 1 | 0.00% +/- 0.00% | 4.27% +/- 7.04% |",
        "| 5 | 0.00% +/- 0.00% | 4.53% +/- 7.68% |",
    ]
    for k, expected in scrub_expected.items():
        missing = []
        scrub_paths = []
        for seed in SEEDS:
            try:
                scrub_paths.append(_scrub_log(k, seed))
            except AssertionError:
                missing.append(seed)
        values = [_load_last_jsonl(path)["resurrection_acc"] for path in scrub_paths]
        if len(values) == 1:
            std = 0.0
        else:
            mean = sum(values) / len(values)
            std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
        expected_lines.append(
            f"| {k} | 0.00% +/- 0.00% | {expected:.2f}% +/- {std * 100.0:.2f}% | []/{missing} |"
        )
    for text in expected_lines:
        assert text in summary
    assert "SCRUB follow-up closes 10-shot recovery to 0.00%" in readme
    assert "k=1: 4.27%" in readme
    assert "k=5: 4.53%" in readme
