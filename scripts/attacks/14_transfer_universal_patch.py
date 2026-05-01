#!/usr/bin/env python3
"""
Evaluate a saved universal input attack zero-shot on multiple target checkpoints.
"""

import argparse
import json
import os
import sys
from typing import Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn

from src.attacks.fourier_perturbation import UniversalFourierAttack
from src.attacks.patch_lattice_warp import PatchLatticeWarpAttack
from src.attacks.universal_tile import UniversalTileAttack
from src.attacks.universal_patch import UniversalPatchAttack
from src.data import DataManager
from src.eval import create_evaluator
from src.models.cnn import create_cnn_model
from src.models.peft_lora import load_lora_adapter
from src.models.vit import create_vit_model
from src.utils import get_device, load_config, set_seed


def _build_model_for_suite(experiment_suites: Dict, suite_name: str, device: torch.device) -> nn.Module:
    data_config = load_config("configs/data.yaml")
    model_config = load_config("configs/model.yaml")

    suite_cfg = experiment_suites[suite_name]
    ref_cfg = experiment_suites[suite_cfg["base_model_suite"]] if "base_model_suite" in suite_cfg else suite_cfg
    dataset_name = ref_cfg["dataset"]
    model_type = ref_cfg["model"]
    num_classes = data_config[dataset_name]["num_classes"]

    if model_type.startswith("vit"):
        model_key = model_type.replace("vit_", "")
        return create_vit_model(model_config["vit"][model_key], num_classes=num_classes).to(device)
    return create_cnn_model(model_config["cnn"][model_type], num_classes=num_classes).to(device)


def load_target_model(experiment_suites: Dict, suite_name: str, seed: int, device: torch.device) -> nn.Module:
    if suite_name.startswith("oracle_"):
        checkpoint_path = f"checkpoints/oracle/{suite_name}_seed_{seed}_final.pt"
        fallback_path = f"checkpoints/oracle/{suite_name}_seed_{seed}_best.pt"
        if not os.path.exists(checkpoint_path):
            checkpoint_path = fallback_path
        model = _build_model_for_suite(experiment_suites, suite_name, device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    if suite_name.startswith("base_"):
        checkpoint_path = f"checkpoints/base/{suite_name}_seed_{seed}_final.pt"
        fallback_path = f"checkpoints/base/{suite_name}_seed_{seed}_best.pt"
        if not os.path.exists(checkpoint_path):
            checkpoint_path = fallback_path
        model = _build_model_for_suite(experiment_suites, suite_name, device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    if suite_name.startswith("unlearn_"):
        suite_cfg = experiment_suites[suite_name]
        base_model = load_target_model(experiment_suites, suite_cfg["base_model_suite"], seed, device)
        adapter_path = f"checkpoints/unlearn_lora/{suite_name}_seed_{seed}"
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
        return load_lora_adapter(base_model, adapter_path)

    raise ValueError(f"Unsupported suite type: {suite_name}")


def evaluate_clean(
    model: nn.Module,
    dataset_name: str,
    forget_class: int,
    num_classes: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
):
    data_manager = DataManager()
    test_loader = data_manager.get_dataloader(
        dataset_name,
        "test",
        batch_size=batch_size,
        num_workers=num_workers,
        use_pretrained=True,
        apply_imagenet_norm=False,
    )
    evaluator = create_evaluator(
        forget_class=forget_class,
        num_classes=num_classes,
        device=str(device),
    )
    return evaluator.evaluate_model(model, test_loader, attack_type="clean")


def evaluate_clean_best_tile_phase(
    model: nn.Module,
    dataset_name: str,
    forget_class: int,
    num_classes: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
):
    phase_grid = int(getattr(model, "phase_grid", 1))
    if phase_grid <= 1 or not hasattr(model, "eval_phase_offset_y"):
        return evaluate_clean(
            model,
            dataset_name,
            forget_class,
            num_classes,
            device,
            batch_size,
            num_workers,
        )

    best_metrics = None
    best_phase = (0, 0)
    best_key = (-1.0, -1.0, -1.0)
    for phase_y in range(phase_grid):
        for phase_x in range(phase_grid):
            model.eval_phase_offset_y = phase_y
            model.eval_phase_offset_x = phase_x
            metrics = evaluate_clean(
                model,
                dataset_name,
                forget_class,
                num_classes,
                device,
                batch_size,
                num_workers,
            )
            current_key = (
                float(metrics.get("forget_acc", 0.0)),
                float(metrics.get("retain_acc", 0.0)),
                float(metrics.get("overall_acc", 0.0)),
            )
            if best_metrics is None or current_key > best_key:
                best_key = current_key
                best_phase = (phase_y, phase_x)
                best_metrics = dict(metrics)

    model.eval_phase_offset_y = best_phase[0]
    model.eval_phase_offset_x = best_phase[1]
    if best_metrics is None:
        best_metrics = evaluate_clean(
            model,
            dataset_name,
            forget_class,
            num_classes,
            device,
            batch_size,
            num_workers,
        )
    best_metrics["eval_phase_offset_y"] = float(best_phase[0])
    best_metrics["eval_phase_offset_x"] = float(best_phase[1])
    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="Transfer a saved universal input attack across target models")
    parser.add_argument("--config", required=True)
    parser.add_argument("--patch-dir", required=True)
    parser.add_argument("--patch-seed", type=int, required=True)
    parser.add_argument("--target-suites", nargs="+", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    experiment_suites = load_config(args.config)

    info_path = os.path.join(args.patch_dir, "attack_info.json")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"attack_info.json not found in patch dir: {args.patch_dir}")

    with open(info_path, "r", encoding="utf-8") as handle:
        attack_info = json.load(handle)

    artifact_type = attack_info.get("artifact_type")
    if artifact_type == "universal_patch":
        attack_kind = "patch"
        attack_config = attack_info["patch_config"]
    elif artifact_type == "universal_fourier":
        attack_kind = "fourier"
        attack_config = attack_info["attack_config"]
    elif artifact_type == "universal_tile":
        attack_kind = "tile"
        attack_config = attack_info["attack_config"]
    elif artifact_type == "patch_lattice_warp":
        attack_kind = "warp"
        attack_config = attack_info["attack_config"]
    else:
        raise ValueError(f"{args.patch_dir} is not a supported universal attack artifact")

    forget_class = int(attack_info["forget_class"])

    results = {
        "patch_dir": args.patch_dir,
        "patch_seed": args.patch_seed,
        "patch_info": attack_info,
        "targets": {},
    }

    data_config = load_config("configs/data.yaml")

    for target_suite in args.target_suites:
        target_model = load_target_model(experiment_suites, target_suite, args.patch_seed, device)
        suite_cfg = experiment_suites[target_suite]
        ref_cfg = experiment_suites[suite_cfg["base_model_suite"]] if "base_model_suite" in suite_cfg else suite_cfg
        dataset_name = ref_cfg["dataset"]
        num_classes = int(data_config[dataset_name]["num_classes"])

        if attack_kind == "patch":
            attack = UniversalPatchAttack(
                target_models=target_model,
                oracle_model=None,
                forget_class=forget_class,
                patch_config=attack_config,
                device=str(device),
            )
            attack.load_attack_patch(args.patch_dir)
            attacked_model = attack.patched_target_model
        else:
            if attack_kind == "tile":
                attack = UniversalTileAttack(
                    target_models=target_model,
                    oracle_model=None,
                    forget_class=forget_class,
                    attack_config=attack_config,
                    device=str(device),
                )
                attack.load_attack_artifact(args.patch_dir)
                attacked_model = attack.perturbed_target_model
            elif attack_kind == "warp":
                attack = PatchLatticeWarpAttack(
                    target_models=target_model,
                    oracle_model=None,
                    forget_class=forget_class,
                    attack_config=attack_config,
                    device=str(device),
                )
                attack.load_attack_artifact(args.patch_dir)
                attacked_model = attack.perturbed_target_model
            else:
                attack = UniversalFourierAttack(
                    target_models=target_model,
                    oracle_model=None,
                    forget_class=forget_class,
                    attack_config=attack_config,
                    device=str(device),
                )
                attack.load_attack_artifact(args.patch_dir)
                attacked_model = attack.perturbed_target_model

        target_metrics = evaluate_clean(
            attack.target_clean_models[0],
            dataset_name,
            forget_class,
            num_classes,
            device,
            args.eval_batch_size,
            args.num_workers,
        )
        if attack_kind == "tile":
            patched_metrics = evaluate_clean_best_tile_phase(
                attacked_model,
                dataset_name,
                forget_class,
                num_classes,
                device,
                args.eval_batch_size,
                args.num_workers,
            )
        else:
            patched_metrics = evaluate_clean(
                attacked_model,
                dataset_name,
                forget_class,
                num_classes,
                device,
                args.eval_batch_size,
                args.num_workers,
            )

        results["targets"][target_suite] = {
            "target_clean_overall_acc": target_metrics.get("overall_acc", 0.0),
            "target_clean_forget_acc": target_metrics.get("forget_acc", 0.0),
            "target_clean_retain_acc": target_metrics.get("retain_acc", 0.0),
            "patched_clean_overall_acc": patched_metrics.get("overall_acc", 0.0),
            "patched_clean_forget_acc": patched_metrics.get("forget_acc", 0.0),
            "patched_clean_retain_acc": patched_metrics.get("retain_acc", 0.0),
            "forget_gain_vs_target": patched_metrics.get("forget_acc", 0.0)
            - target_metrics.get("forget_acc", 0.0),
            "retain_drop_vs_target": target_metrics.get("retain_acc", 0.0)
            - patched_metrics.get("retain_acc", 0.0),
            "overall_drop_vs_target": target_metrics.get("overall_acc", 0.0)
            - patched_metrics.get("overall_acc", 0.0),
        }

    if args.output is None:
        patch_name = os.path.basename(os.path.normpath(args.patch_dir))
        args.output = os.path.join("results", "analysis", f"{patch_name}_transfer_eval.json")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Wrote universal attack transfer evaluation to: {args.output}")
    for target_suite, blob in results["targets"].items():
        print(
            f"{target_suite}: "
            f"forget {100 * blob['target_clean_forget_acc']:.2f}% -> "
            f"{100 * blob['patched_clean_forget_acc']:.2f}% "
            f"(gain {100 * blob['forget_gain_vs_target']:.2f}pp), "
            f"retain drop {100 * blob['retain_drop_vs_target']:.2f}pp"
        )


if __name__ == "__main__":
    main()
