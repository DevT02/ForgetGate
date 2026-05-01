#!/usr/bin/env python3
"""
Train a universal input-space attack against an unlearned checkpoint.
"""

import argparse
import os
import random
import sys
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset

from src.attacks.fourier_perturbation import UniversalFourierAttack
from src.attacks.patch_lattice_warp import PatchLatticeWarpAttack
from src.attacks.universal_tile import UniversalTileAttack
from src.attacks.universal_patch import UniversalPatchAttack
from src.data import DataManager, create_forget_retain_splits
from src.eval import create_evaluator
from src.models.cnn import create_cnn_model
from src.models.peft_lora import load_lora_adapter
from src.models.vit import create_vit_model
from src.utils import (
    create_experiment_log,
    get_device,
    load_config,
    log_experiment,
    print_model_info,
    save_dict_to_json,
    set_seed,
)


def load_oracle_model(experiment_suites: Dict, suite_name: str, seed: int, device: torch.device) -> nn.Module:
    checkpoint_path = f"checkpoints/oracle/{suite_name}_seed_{seed}_final.pt"
    fallback_path = f"checkpoints/oracle/{suite_name}_seed_{seed}_best.pt"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = fallback_path
    checkpoint = torch.load(checkpoint_path, map_location=device)

    data_config = load_config("configs/data.yaml")
    model_config = load_config("configs/model.yaml")
    suite_cfg = experiment_suites[suite_name]
    dataset_name = suite_cfg["dataset"]
    model_type = suite_cfg["model"]
    num_classes = data_config[dataset_name]["num_classes"]

    if model_type.startswith("vit"):
        model_key = model_type.replace("vit_", "")
        model = create_vit_model(model_config["vit"][model_key], num_classes=num_classes)
    else:
        model = create_cnn_model(model_config["cnn"][model_type], num_classes=num_classes)

    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def load_target_model(experiment_suites: Dict, suite_config: Dict, seed: int, device: torch.device) -> nn.Module:
    if "oracle_model_suite" in suite_config and "unlearned_model_suite" not in suite_config:
        return load_oracle_model(experiment_suites, suite_config["oracle_model_suite"], seed, device)

    unlearned_suite_name = suite_config["unlearned_model_suite"]
    unlearned_suite_cfg = experiment_suites[unlearned_suite_name]
    base_suite_name = unlearned_suite_cfg["base_model_suite"]
    base_suite_cfg = experiment_suites[base_suite_name]

    data_config = load_config("configs/data.yaml")
    model_config = load_config("configs/model.yaml")
    num_classes = data_config[base_suite_cfg["dataset"]]["num_classes"]
    model_type = base_suite_cfg["model"]

    if model_type.startswith("vit"):
        model_key = model_type.replace("vit_", "")
        model = create_vit_model(model_config["vit"][model_key], num_classes=num_classes)
    else:
        model = create_cnn_model(model_config["cnn"][model_type], num_classes=num_classes)

    base_checkpoint = f"checkpoints/base/{base_suite_name}_seed_{seed}_final.pt"
    base_fallback = f"checkpoints/base/{base_suite_name}_seed_{seed}_best.pt"
    if not os.path.exists(base_checkpoint):
        base_checkpoint = base_fallback
    checkpoint = torch.load(base_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    adapter_path = f"checkpoints/unlearn_lora/{unlearned_suite_name}_seed_{seed}"
    if os.path.exists(adapter_path):
        model = load_lora_adapter(model, adapter_path)
        print(f"Loaded LoRA adapter from: {adapter_path}")
    else:
        print(f"Warning: adapter not found at {adapter_path}, using base model")

    return model


def load_target_model_by_suite_name(
    experiment_suites: Dict, suite_name: str, seed: int, device: torch.device
) -> nn.Module:
    if suite_name.startswith("oracle_"):
        return load_oracle_model(experiment_suites, suite_name, seed, device)
    if suite_name.startswith("unlearn_"):
        return load_target_model(
            experiment_suites,
            {"unlearned_model_suite": suite_name},
            seed,
            device,
        )
    if suite_name.startswith("base_"):
        suite_cfg = experiment_suites[suite_name]
        data_config = load_config("configs/data.yaml")
        model_config = load_config("configs/model.yaml")
        num_classes = data_config[suite_cfg["dataset"]]["num_classes"]
        model_type = suite_cfg["model"]
        if model_type.startswith("vit"):
            model_key = model_type.replace("vit_", "")
            model = create_vit_model(model_config["vit"][model_key], num_classes=num_classes)
        else:
            model = create_cnn_model(model_config["cnn"][model_type], num_classes=num_classes)
        checkpoint_path = f"checkpoints/base/{suite_name}_seed_{seed}_final.pt"
        fallback_path = f"checkpoints/base/{suite_name}_seed_{seed}_best.pt"
        if not os.path.exists(checkpoint_path):
            checkpoint_path = fallback_path
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(device)
    raise ValueError(f"Unsupported target suite type: {suite_name}")


def resolve_matching_oracle_suite(
    experiment_suites: Dict, dataset_name: str, model_type: str, forget_class: int
) -> str:
    for suite_name, suite_cfg in experiment_suites.items():
        if not suite_name.startswith("oracle_"):
            continue
        if suite_cfg.get("dataset") != dataset_name:
            continue
        if suite_cfg.get("model") != model_type:
            continue
        if int(suite_cfg.get("forget_class", -1)) != int(forget_class):
            continue
        return suite_name
    raise ValueError(
        f"No oracle suite for dataset={dataset_name}, model={model_type}, forget_class={forget_class}"
    )


def build_train_eval_view(data_manager: DataManager, dataset_name: str) -> Dataset:
    eval_transform = data_manager.get_transforms(
        dataset_name,
        split="test",
        use_pretrained=True,
        apply_imagenet_norm=False,
    )

    if dataset_name == "cifar10":
        try:
            return torchvision.datasets.CIFAR10(
                root=data_manager.data_dir,
                train=True,
                download=False,
                transform=eval_transform,
            )
        except RuntimeError:
            archive_path = os.path.join(data_manager.data_dir, "cifar-10-python.tar.gz")
            if os.path.exists(archive_path):
                from src.data import CIFAR10TarDataset

                return CIFAR10TarDataset(archive_path, train=True, transform=eval_transform)
            return torchvision.datasets.CIFAR10(
                root=data_manager.data_dir,
                train=True,
                download=True,
                transform=eval_transform,
            )

    raise ValueError(f"Unsupported dataset for train eval view: {dataset_name}")


def evaluate_clean_model(
    model: nn.Module,
    dataset_name: str,
    forget_class: int,
    num_classes: int,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 0,
) -> Dict[str, float]:
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
    return evaluator.evaluate_model(model=model, data_loader=test_loader, attack_type="clean")


def evaluate_clean_model_best_tile_phase(
    model: nn.Module,
    dataset_name: str,
    forget_class: int,
    num_classes: int,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 0,
) -> Dict[str, float]:
    phase_grid = int(getattr(model, "phase_grid", 1))
    if phase_grid <= 1 or not hasattr(model, "eval_phase_offset_y"):
        return evaluate_clean_model(
            model,
            dataset_name,
            forget_class,
            num_classes,
            device,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    best_metrics = None
    best_phase = (0, 0)
    best_key = (-1.0, -1.0, -1.0)
    for phase_y in range(phase_grid):
        for phase_x in range(phase_grid):
            model.eval_phase_offset_y = phase_y
            model.eval_phase_offset_x = phase_x
            metrics = evaluate_clean_model(
                model,
                dataset_name,
                forget_class,
                num_classes,
                device,
                batch_size=batch_size,
                num_workers=num_workers,
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
        best_metrics = evaluate_clean_model(
            model,
            dataset_name,
            forget_class,
            num_classes,
            device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    best_metrics["eval_phase_offset_y"] = float(best_phase[0])
    best_metrics["eval_phase_offset_x"] = float(best_phase[1])
    return best_metrics


def subset_to_base_indices(dataset_subset: Dataset) -> List[int]:
    if isinstance(dataset_subset, Subset):
        parent_indices = subset_to_base_indices(dataset_subset.dataset)
        return [parent_indices[int(idx)] for idx in dataset_subset.indices]
    return list(range(len(dataset_subset)))


def main():
    parser = argparse.ArgumentParser(description="Train a universal input-space attack")
    parser.add_argument("--config", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--k-shot", type=int, default=None)
    parser.add_argument("--oracle-contrast-weight", type=float, default=None)
    parser.add_argument("--lambda-retain", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--append-log", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    experiment_suites = load_config(args.config)
    if args.suite not in experiment_suites:
        raise ValueError(f"Unknown suite: {args.suite}")

    suite_config = experiment_suites[args.suite]
    device = get_device(args.device)

    if "patch_attack" in suite_config:
        attack_kind = "patch"
        attack_label = "Universal Patch"
        attack_root = "checkpoints/universal_patch"
        attack_params = dict(suite_config.get("patch_attack", {}))
    elif "fourier_attack" in suite_config:
        attack_kind = "fourier"
        attack_label = "Universal Fourier"
        attack_root = "checkpoints/fourier_attack"
        attack_params = dict(suite_config.get("fourier_attack", {}))
    elif "tile_attack" in suite_config:
        attack_kind = "tile"
        attack_label = "Universal Tile"
        attack_root = "checkpoints/tile_attack"
        attack_params = dict(suite_config.get("tile_attack", {}))
    elif "warp_attack" in suite_config:
        attack_kind = "warp"
        attack_label = "Patch Lattice Warp"
        attack_root = "checkpoints/warp_attack"
        attack_params = dict(suite_config.get("warp_attack", {}))
    else:
        raise ValueError(f"Suite {args.suite} is missing patch_attack/fourier_attack/tile_attack/warp_attack config")

    print(f"Run: Training {attack_label} - {args.suite}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")

    forget_class = int(attack_params.get("target_class", 0))
    if attack_kind == "patch" and args.patch_size is not None:
        attack_params["patch_size"] = args.patch_size
    if attack_kind == "fourier" and args.epsilon is not None:
        attack_params["epsilon"] = args.epsilon
    if args.lambda_retain is not None:
        attack_params["lambda_retain"] = args.lambda_retain
    if args.oracle_contrast_weight is not None:
        attack_params["oracle_contrast_weight"] = args.oracle_contrast_weight

    print(f"Target class: {forget_class}")
    if attack_kind == "patch":
        print(f"Patch size: {attack_params.get('patch_size', 32)}")
        print(f"Patch position: {attack_params.get('position', 'bottom_right')}")
    elif attack_kind == "fourier":
        print(f"Epsilon: {attack_params.get('epsilon', 8.0 / 255.0)}")
        print(f"Patch stride: {attack_params.get('patch_stride', 16)}")
        print(f"Harmonics: {attack_params.get('harmonics', [1, 2, 3])}")
    elif attack_kind == "tile":
        print(f"Epsilon: {attack_params.get('epsilon', 8.0 / 255.0)}")
        print(f"Tile size: {attack_params.get('tile_size', attack_params.get('patch_stride', 16))}")
        print(f"Phase jitter: {attack_params.get('phase_jitter', attack_params.get('tile_size', 16) - 1)}")
    else:
        print(f"Patch stride: {attack_params.get('patch_stride', 16)}")
        print(f"Flow grid size: {attack_params.get('flow_grid_size', 14)}")
        print(f"Max flow pixels: {attack_params.get('max_flow_pixels', 3.0)}")

    target_suite_names: List[str]
    if "target_model_suites" in suite_config:
        target_suite_names = list(suite_config["target_model_suites"])
        target_models = [
            load_target_model_by_suite_name(experiment_suites, suite_name, args.seed, device)
            for suite_name in target_suite_names
        ]
        for suite_name, target_model in zip(target_suite_names, target_models):
            print_model_info(target_model, f"Target model: {suite_name}")
    else:
        target_models = load_target_model(experiment_suites, suite_config, args.seed, device)
        target_suite_names = [
            suite_config.get("unlearned_model_suite", suite_config.get("oracle_model_suite", args.suite))
        ]
        print_model_info(target_models, "Target model")

    if "unlearned_model_suite" in suite_config:
        unlearned_suite_cfg = experiment_suites[suite_config["unlearned_model_suite"]]
        base_suite_cfg = experiment_suites[unlearned_suite_cfg["base_model_suite"]]
        dataset_name = base_suite_cfg["dataset"]
        model_type = base_suite_cfg["model"]
        oracle_suite_name = suite_config.get("oracle_model_suite")
        if not oracle_suite_name:
            oracle_suite_name = resolve_matching_oracle_suite(
                experiment_suites, dataset_name, model_type, forget_class
            )
    elif "target_model_suites" in suite_config:
        first_target_suite = suite_config["target_model_suites"][0]
        first_target_cfg = experiment_suites[first_target_suite]
        base_suite_cfg = experiment_suites[first_target_cfg["base_model_suite"]]
        dataset_name = base_suite_cfg["dataset"]
        model_type = base_suite_cfg["model"]
        for target_suite in suite_config["target_model_suites"][1:]:
            target_cfg = experiment_suites[target_suite]
            target_base_cfg = experiment_suites[target_cfg["base_model_suite"]]
            if target_base_cfg["dataset"] != dataset_name or target_base_cfg["model"] != model_type:
                raise ValueError("All target_model_suites must share the same dataset and model architecture.")
        oracle_suite_name = suite_config.get("oracle_model_suite")
        if not oracle_suite_name:
            oracle_suite_name = resolve_matching_oracle_suite(
                experiment_suites, dataset_name, model_type, forget_class
            )
    else:
        oracle_suite_name = suite_config.get("oracle_model_suite")
        oracle_suite_cfg = experiment_suites[oracle_suite_name]
        dataset_name = oracle_suite_cfg["dataset"]
        model_type = oracle_suite_cfg["model"]

    oracle_model = None
    oracle_contrast_weight = float(attack_params.get("oracle_contrast_weight", 1.0))
    if oracle_suite_name and oracle_contrast_weight > 0.0:
        oracle_model = load_oracle_model(experiment_suites, oracle_suite_name, args.seed, device)
        print(f"Loaded oracle model for patch contrast: {oracle_suite_name}")

    data_config = load_config("configs/data.yaml")
    num_classes = int(data_config[dataset_name]["num_classes"])
    data_manager = DataManager()
    train_dataset = data_manager.load_dataset(
        dataset_name,
        "train",
        use_pretrained=True,
        apply_imagenet_norm=False,
    )
    forget_train, retain_train, forget_val, _ = create_forget_retain_splits(
        train_dataset, forget_class, train_ratio=0.8
    )
    full_forget_train = forget_train
    full_retain_train = retain_train

    eval_train_dataset = build_train_eval_view(data_manager, dataset_name)
    if isinstance(forget_val, Subset):
        forget_val = Subset(eval_train_dataset, subset_to_base_indices(forget_val))

    def to_eval_subset(dataset_subset):
        if isinstance(dataset_subset, Subset):
            return Subset(eval_train_dataset, subset_to_base_indices(dataset_subset))
        return dataset_subset

    k_shot = attack_params.get("k_shot")
    if args.k_shot is not None:
        k_shot = args.k_shot
    if k_shot is not None:
        print(f"\nK-shot mode: Using only {k_shot} samples from forget class")
        random.seed(args.seed)
        indices = list(range(len(forget_train)))
        random.shuffle(indices)
        forget_train = Subset(forget_train, indices[:k_shot])
        print(f"Forget train samples: {len(forget_train)} (original: {len(indices)})")

    if bool(attack_params.get("attack_use_eval_view", False)):
        forget_train = to_eval_subset(forget_train)
        retain_train = to_eval_subset(retain_train)
        print("Attack training view: deterministic eval-view tensors")

    batch_size = int(attack_params.get("batch_size", 64))
    eval_batch_size = int(attack_params.get("eval_batch_size", 256))
    num_workers = args.num_workers if args.num_workers is not None else 0
    print(f"DataLoader workers: {num_workers}")

    forget_train_loader = DataLoader(
        forget_train,
        batch_size=min(batch_size, len(forget_train)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    retain_train_loader = DataLoader(
        retain_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    forget_val_loader = DataLoader(
        forget_val,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    guidance_forget_loader = None
    guidance_retain_loader = None
    if str(attack_params.get("feature_mode", "")).lower().strip() == "harp":
        guidance_use_full_forget = bool(attack_params.get("guidance_use_full_forget", False))
        guidance_batch_size = int(attack_params.get("guidance_batch_size", eval_batch_size))
        if guidance_use_full_forget:
            guidance_forget_dataset = to_eval_subset(full_forget_train)
            guidance_retain_dataset = to_eval_subset(full_retain_train)
            guidance_forget_loader = DataLoader(
                guidance_forget_dataset,
                batch_size=min(guidance_batch_size, len(guidance_forget_dataset)),
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=(num_workers > 0),
            )
            guidance_retain_loader = DataLoader(
                guidance_retain_dataset,
                batch_size=guidance_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=(num_workers > 0),
            )
            print(
                f"Guidance references: forget={len(guidance_forget_dataset)} retain={len(guidance_retain_dataset)} "
                f"(eval-view, full forget train)"
            )

    if attack_kind == "patch":
        attack = UniversalPatchAttack(
            target_models=target_models,
            oracle_model=oracle_model,
            forget_class=forget_class,
            patch_config=attack_params,
            device=str(device),
        )
    elif attack_kind == "fourier":
        attack = UniversalFourierAttack(
            target_models=target_models,
            oracle_model=oracle_model,
            forget_class=forget_class,
            attack_config=attack_params,
            device=str(device),
        )
    else:
        if attack_kind == "tile":
            attack = UniversalTileAttack(
                target_models=target_models,
                oracle_model=oracle_model,
                forget_class=forget_class,
                attack_config=attack_params,
                device=str(device),
            )
        else:
            attack = PatchLatticeWarpAttack(
                target_models=target_models,
                oracle_model=oracle_model,
                forget_class=forget_class,
                attack_config=attack_params,
                device=str(device),
            )

    suite_name_for_io = args.suite
    if args.k_shot is not None:
        suite_name_for_io = f"{suite_name_for_io}_kshot{args.k_shot}"
    if attack_kind == "patch" and args.patch_size is not None:
        suite_name_for_io = f"{suite_name_for_io}_ps{args.patch_size}"
    if attack_kind == "fourier" and args.epsilon is not None:
        eps_tag = str(args.epsilon).replace(".", "p")
        suite_name_for_io = f"{suite_name_for_io}_eps{eps_tag}"
    if attack_kind == "tile" and args.epsilon is not None:
        eps_tag = str(args.epsilon).replace(".", "p")
        suite_name_for_io = f"{suite_name_for_io}_eps{eps_tag}"
    if attack_kind == "warp" and args.epsilon is not None:
        eps_tag = str(args.epsilon).replace(".", "p")
        suite_name_for_io = f"{suite_name_for_io}_eps{eps_tag}"

    log_path = create_experiment_log(f"{suite_name_for_io}_seed_{args.seed}", {})
    if os.path.exists(log_path) and not args.append_log:
        print(f"Overwriting existing log: {log_path}")
        os.remove(log_path)

    if attack_kind == "patch":
        history = attack.train_patch(
            forget_loader=forget_train_loader,
            retain_loader=retain_train_loader,
            val_forget_loader=forget_val_loader,
            guidance_forget_loader=guidance_forget_loader,
            guidance_retain_loader=guidance_retain_loader,
            epochs=int(attack_params.get("epochs", 100)),
            lr=float(attack_params.get("lr", 5e-2)),
            lambda_retain=float(attack_params.get("lambda_retain", 1.0)),
            temperature=float(attack_params.get("T", 1.0)),
            eval_every=int(attack_params.get("eval_every", 10)),
            oracle_contrast_weight=oracle_contrast_weight,
        )
    elif attack_kind == "fourier":
        history = attack.train_attack(
            forget_loader=forget_train_loader,
            retain_loader=retain_train_loader,
            val_forget_loader=forget_val_loader,
            guidance_forget_loader=guidance_forget_loader,
            guidance_retain_loader=guidance_retain_loader,
            epochs=int(attack_params.get("epochs", 100)),
            lr=float(attack_params.get("lr", 5e-2)),
            lambda_retain=float(attack_params.get("lambda_retain", 1.0)),
            temperature=float(attack_params.get("T", 1.0)),
            eval_every=int(attack_params.get("eval_every", 10)),
            oracle_contrast_weight=oracle_contrast_weight,
        )
    else:
        history = attack.train_attack(
            forget_loader=forget_train_loader,
            retain_loader=retain_train_loader,
            val_forget_loader=forget_val_loader,
            guidance_forget_loader=guidance_forget_loader,
            guidance_retain_loader=guidance_retain_loader,
            epochs=int(attack_params.get("epochs", 100)),
            lr=float(attack_params.get("lr", 5e-2)),
            lambda_retain=float(attack_params.get("lambda_retain", 1.0)),
            temperature=float(attack_params.get("T", 1.0)),
            eval_every=int(attack_params.get("eval_every", 10)),
            oracle_contrast_weight=oracle_contrast_weight,
        )

    for row in history:
        log_experiment(log_path, row)

    print(f"\nRunning clean test evaluation on restored {attack_kind} model...")
    per_target_metrics = {}
    attacked_models = attack.patched_target_models if attack_kind == "patch" else attack.perturbed_target_models
    for target_suite_name, clean_model, attacked_model in zip(
        target_suite_names, attack.target_clean_models, attacked_models
    ):
        target_metrics = evaluate_clean_model(
            clean_model,
            dataset_name,
            forget_class,
            num_classes,
            device,
            batch_size=eval_batch_size,
            num_workers=num_workers,
        )
        if attack_kind == "tile":
            patched_metrics = evaluate_clean_model_best_tile_phase(
                attacked_model,
                dataset_name,
                forget_class,
                num_classes,
                device,
                batch_size=eval_batch_size,
                num_workers=num_workers,
            )
        else:
            patched_metrics = evaluate_clean_model(
                attacked_model,
                dataset_name,
                forget_class,
                num_classes,
                device,
                batch_size=eval_batch_size,
                num_workers=num_workers,
            )
        per_target_metrics[target_suite_name] = {
            "target_test_clean_overall_acc": target_metrics.get("overall_acc", 0.0),
            "target_test_clean_forget_acc": target_metrics.get("forget_acc", 0.0),
            "target_test_clean_retain_acc": target_metrics.get("retain_acc", 0.0),
            "test_clean_overall_acc": patched_metrics.get("overall_acc", 0.0),
            "test_clean_forget_acc": patched_metrics.get("forget_acc", 0.0),
            "test_clean_retain_acc": patched_metrics.get("retain_acc", 0.0),
            "test_clean_overall_drop_vs_target": (
                target_metrics.get("overall_acc", 0.0) - patched_metrics.get("overall_acc", 0.0)
            ),
            "test_clean_retain_drop_vs_target": (
                target_metrics.get("retain_acc", 0.0) - patched_metrics.get("retain_acc", 0.0)
            ),
            "test_clean_forget_gain_vs_target": (
                patched_metrics.get("forget_acc", 0.0) - target_metrics.get("forget_acc", 0.0)
            ),
        }

    primary_target_metrics = per_target_metrics[target_suite_names[0]]

    final_metrics = dict(history[-1]) if history else {}
    final_metrics.update(primary_target_metrics)

    save_dir = f"{attack_root}/{suite_name_for_io}_seed_{args.seed}"
    if attack_kind == "patch":
        attack.save_attack_patch(save_dir)
        attack_stats = attack.get_patch_statistics()
    else:
        attack.save_attack_artifact(save_dir)
        attack_stats = attack.get_attack_statistics()
    summary_path = f"results/analysis/{suite_name_for_io}_seed_{args.seed}_summary.json"
    save_dict_to_json(
        {
            "suite": args.suite,
            "suite_name_for_io": suite_name_for_io,
            "seed": args.seed,
            "dataset": dataset_name,
            "forget_class": forget_class,
            "target_suites": target_suite_names,
            "attack_kind": attack_kind,
            "attack_save_path": save_dir,
            "log_path": log_path,
            "attack_config": attack_params,
            "attack_stats": attack_stats,
            "final_metrics": final_metrics,
            "per_target_metrics": per_target_metrics,
        },
        summary_path,
    )

    print(f"\n{attack_label} training complete!")
    print(f"{attack_label} saved to: {save_dir}")
    print("\nFinal attack metrics:")
    for key, value in final_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    if len(per_target_metrics) > 1:
        print("\nPer-target attacked clean metrics:")
        for target_suite_name, blob in per_target_metrics.items():
            print(
                f"  {target_suite_name}: "
                f"forget {blob['target_test_clean_forget_acc']:.4f} -> {blob['test_clean_forget_acc']:.4f} | "
                f"retain drop {blob['test_clean_retain_drop_vs_target']:.4f}"
            )

    print("\nAttack statistics:")
    for key, value in attack_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
