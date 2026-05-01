#!/usr/bin/env python3
"""
Script 3: Train VPT resurrector against unlearned model
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import random
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from typing import Dict
import yaml
import torchvision

from src.data import DataManager, create_forget_retain_splits
from src.models.vit import create_vit_model, add_vpt_to_model
from src.models.cnn import create_cnn_model, add_vpt_to_cnn_model
from src.models.peft_lora import load_lora_adapter
from src.models.normalize import create_imagenet_normalizer
from src.attacks.vpt_resurrection import VPTResurrectionAttack
from src.eval import create_evaluator
from src.utils import set_seed, ensure_dir, load_config, print_model_info, get_device
from src.utils import log_experiment, create_experiment_log, save_dict_to_json


def load_oracle_model(experiment_suites: Dict, suite_config: Dict, device: torch.device, seed: int) -> nn.Module:
    """Load oracle model (trained without forget class)."""
    oracle_suite_name = suite_config.get("oracle_model_suite", "")
    if not oracle_suite_name:
        raise ValueError("No oracle_model_suite specified in VPT suite")

    oracle_checkpoint = f"checkpoints/oracle/{oracle_suite_name}_seed_{seed}_final.pt"

    if not os.path.exists(oracle_checkpoint):
        raise FileNotFoundError(f"Oracle model checkpoint not found: {oracle_checkpoint}")

    checkpoint = torch.load(oracle_checkpoint, map_location=device)

    # Recreate oracle model using oracle suite config
    data_config = load_config("configs/data.yaml")
    model_config = load_config("configs/model.yaml")

    oracle_suite_cfg = experiment_suites[oracle_suite_name]
    dataset_name = oracle_suite_cfg["dataset"]
    dataset_info = data_config[dataset_name]
    model_type = oracle_suite_cfg["model"]

    if model_type.startswith("vit"):
        model_config_name = model_type.replace("vit_", "")
        oracle_model = create_vit_model(
            model_config["vit"][model_config_name],
            num_classes=dataset_info["num_classes"],
        )
    else:
        oracle_model = create_cnn_model(
            model_config["cnn"][model_type],
            num_classes=dataset_info["num_classes"],
        )

    oracle_model.load_state_dict(checkpoint["model_state_dict"])
    oracle_model = oracle_model.to(device)

    return oracle_model


def load_unlearned_model(experiment_suites: Dict, suite_config: Dict, device: torch.device, seed: int) -> nn.Module:
    """Load unlearned model with LoRA adapter."""
    unlearned_suite_name = suite_config.get("unlearned_model_suite", "")
    if not unlearned_suite_name:
        raise ValueError("No unlearned_model_suite specified in VPT suite")

    # Resolve the suite chain: VPT -> unlearned -> base
    unlearned_suite_config = experiment_suites.get(unlearned_suite_name, {})
    base_suite_name = unlearned_suite_config.get("base_model_suite", "")
    if not base_suite_name:
        raise ValueError("No base_model_suite found in unlearned suite")

    base_checkpoint = f"checkpoints/base/{base_suite_name}_seed_{seed}_final.pt"
    base_fallback = f"checkpoints/base/{base_suite_name}_seed_{seed}_best.pt"

    if not os.path.exists(base_checkpoint):
        if os.path.exists(base_fallback):
            base_checkpoint = base_fallback
        else:
            raise FileNotFoundError(
                f"Base model checkpoint not found: {base_checkpoint} (or fallback {base_fallback})"
            )

    checkpoint = torch.load(base_checkpoint, map_location=device)

    # Recreate base model using resolved base suite config
    data_config = load_config("configs/data.yaml")
    model_config = load_config("configs/model.yaml")

    # Resolve dataset and model from base suite
    base_suite_name = unlearned_suite_config["base_model_suite"]
    base_suite_cfg = experiment_suites[base_suite_name]
    dataset_name = base_suite_cfg["dataset"]
    dataset_info = data_config[dataset_name]
    model_type = base_suite_cfg["model"]

    if model_type.startswith("vit"):
        model_config_name = model_type.replace("vit_", "")
        base_model = create_vit_model(
            model_config["vit"][model_config_name],
            num_classes=dataset_info["num_classes"],
        )
    else:
        base_model = create_cnn_model(
            model_config["cnn"][model_type],
            num_classes=dataset_info["num_classes"],
        )

    base_model.load_state_dict(checkpoint["model_state_dict"])
    base_model = base_model.to(device)

    # Load LoRA adapter
    adapter_path = f"checkpoints/unlearn_lora/{unlearned_suite_name}_seed_{seed}"
    if os.path.exists(adapter_path):
        unlearned_model = load_lora_adapter(base_model, adapter_path)
        print(f"Loaded LoRA adapter from: {adapter_path}")
    else:
        print(f"Warning: LoRA adapter not found at {adapter_path}, using base model")
        unlearned_model = base_model

    return unlearned_model


def resolve_matching_oracle_suite(experiment_suites: Dict,
                                  dataset_name: str,
                                  model_type: str,
                                  forget_class: int) -> str:
    """Find the oracle suite matching dataset/model/forget_class."""
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
        f"No matching oracle suite found for dataset={dataset_name}, model={model_type}, forget_class={forget_class}"
    )


class LabelOverrideDataset(Dataset):
    def __init__(self, base_dataset, new_labels):
        self.base_dataset = base_dataset
        self.new_labels = new_labels

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, _ = self.base_dataset[idx]
        return x, self.new_labels[idx]


def build_train_eval_view(data_manager: DataManager,
                          dataset_name: str,
                          use_pretrained: bool = True,
                          apply_imagenet_norm: bool = True) -> Dataset:
    """Load the training split with evaluation transforms for deterministic validation."""
    eval_transform = data_manager.get_transforms(
        dataset_name,
        split="test",
        use_pretrained=use_pretrained,
        apply_imagenet_norm=apply_imagenet_norm,
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
    if dataset_name == "mnist":
        return torchvision.datasets.MNIST(
            root=data_manager.data_dir,
            train=True,
            download=True,
            transform=eval_transform,
        )

    raise ValueError(f"Unsupported dataset for train eval view: {dataset_name}")


def evaluate_clean_test_metrics(model: nn.Module,
                                target_model: nn.Module,
                                dataset_name: str,
                                forget_class: int,
                                num_classes: int,
                                device: torch.device,
                                batch_size: int = 256,
                                num_workers: int = 0) -> Dict[str, float]:
    """Evaluate restored prompted model and target model on clean test data."""
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

    # Match scripts/4_adv_evaluate.py: attacks and clean eval operate in [0,1] space,
    # with ImageNet normalization applied inside the model.
    target_eval_model = nn.Sequential(create_imagenet_normalizer(), target_model).to(device)
    prompted_eval_model = nn.Sequential(create_imagenet_normalizer(), model).to(device)

    target_metrics = evaluator.evaluate_model(
        model=target_eval_model,
        data_loader=test_loader,
        attack_type="clean",
    )
    prompted_metrics = evaluator.evaluate_model(
        model=prompted_eval_model,
        data_loader=test_loader,
        attack_type="clean",
    )

    return {
        "target_test_clean_overall_acc": target_metrics.get("overall_acc", 0.0),
        "target_test_clean_forget_acc": target_metrics.get("forget_acc", 0.0),
        "target_test_clean_retain_acc": target_metrics.get("retain_acc", 0.0),
        "test_clean_overall_acc": prompted_metrics.get("overall_acc", 0.0),
        "test_clean_forget_acc": prompted_metrics.get("forget_acc", 0.0),
        "test_clean_retain_acc": prompted_metrics.get("retain_acc", 0.0),
        "test_clean_overall_drop_vs_target": (
            target_metrics.get("overall_acc", 0.0) - prompted_metrics.get("overall_acc", 0.0)
        ),
        "test_clean_retain_drop_vs_target": (
            target_metrics.get("retain_acc", 0.0) - prompted_metrics.get("retain_acc", 0.0)
        ),
        "test_clean_forget_gain_vs_target": (
            prompted_metrics.get("forget_acc", 0.0) - target_metrics.get("forget_acc", 0.0)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Train VPT resurrector")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment suites config")
    parser.add_argument("--suite", type=str, required=True, help="Experiment suite name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=None,
        help="Override prompt length from config (for ablations)",
    )
    parser.add_argument(
        "--label-mode",
        type=str,
        default=None,
        choices=["true", "shuffle", "random"],
        help="Override label mode for forget data: true, shuffle, or random",
    )
    parser.add_argument(
        "--k-shot",
        type=int,
        default=None,
        help="Override k-shot count for forget data (controls)",
    )
    parser.add_argument(
        "--lambda-retain",
        type=float,
        default=None,
        help="Override lambda_retain for VPT training (tradeoff sweeps)",
    )
    parser.add_argument(
        "--stratify",
        type=str,
        default=None,
        choices=["high_conf", "mid_conf", "low_conf"],
        help="Stratify forget data by model confidence before k-shot selection",
    )
    parser.add_argument(
        "--stratify-fraction",
        type=float,
        default=None,
        help="Optional fraction of forget data to keep after stratification",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override DataLoader worker count. Defaults to 0 on Windows to avoid CUDA spawn issues.",
    )
    parser.add_argument(
        "--feature-guide",
        type=str,
        default=None,
        choices=["linear_probe", "centroid_margin", "prototype_margin", "oracle_contrastive_probe"],
        help="Enable feature-guided prompt optimization with the selected frozen-feature objective.",
    )
    parser.add_argument(
        "--feature-guide-weight",
        type=float,
        default=None,
        help="Weight for the feature-guidance loss term during VPT training.",
    )
    parser.add_argument(
        "--feature-contrast-weight",
        type=float,
        default=None,
        help="Oracle-contrastive penalty weight for oracle_contrastive_probe mode.",
    )
    parser.add_argument(
        "--feature-contrast-margin",
        type=float,
        default=None,
        help="Required target-minus-oracle probe-score margin for oracle_contrastive_probe mode.",
    )
    parser.add_argument(
        "--append-log",
        action="store_true",
        help="Append to an existing JSONL log instead of overwriting the log for this run name.",
    )

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Load configs
    experiment_suites = load_config(args.config)
    if args.suite not in experiment_suites:
        raise ValueError(f"Experiment suite '{args.suite}' not found in config")

    suite_config = experiment_suites[args.suite]
    data_config = load_config("configs/data.yaml")
    vpt_config_file = load_config("configs/vpt_attack.yaml")

    # Setup device
    device = get_device(args.device)
    print(f"Run: Training VPT Resurrector - {args.suite}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")

    # Get VPT parameters
    vpt_params = suite_config.get("vpt_attack", {})
    forget_class = vpt_params.get("target_class", 0)
    prompt_length = vpt_params.get("prompt_length", 5)
    if args.prompt_length is not None:
        prompt_length = args.prompt_length
    init_strategy = vpt_params.get("init_strategy", "random")
    label_mode = vpt_params.get("label_mode", "true")
    if args.label_mode is not None:
        label_mode = args.label_mode

    print(f"Target class: {forget_class}")
    print(f"Prompt length: {prompt_length}")
    print(f"Init strategy: {init_strategy}")
    print(f"Label mode: {label_mode}")

    # Create VPT configuration
    vpt_config = {
        "prompt_type": vpt_params.get("prompt_type", "prefix"),
        "prompt_length": prompt_length,
        "init_strategy": init_strategy,
        "dropout": vpt_config_file["vpt_prompt"].get("dropout", 0.1),
    }

    # Determine if this is an oracle or unlearned suite
    is_oracle = "oracle_model_suite" in suite_config
    is_unlearned = "unlearned_model_suite" in suite_config

    if is_oracle:
        target_model = load_oracle_model(experiment_suites, suite_config, device, args.seed)
        print_model_info(target_model, "Oracle model")
        # Resolve dataset from oracle suite
        oracle_suite_name = suite_config.get("oracle_model_suite", "")
        oracle_suite_config = experiment_suites.get(oracle_suite_name, {})
        dataset_name = oracle_suite_config.get("dataset", "cifar10")
        model_type = oracle_suite_config.get("model", "vit_tiny")
    elif is_unlearned:
        target_model = load_unlearned_model(experiment_suites, suite_config, device, args.seed)
        print_model_info(target_model, "Unlearned model")
        # Resolve dataset from suite chain (VPT -> unlearned -> base)
        unlearned_suite_name = suite_config.get("unlearned_model_suite", "")
        unlearned_suite_config = experiment_suites.get(unlearned_suite_name, {})
        base_suite_name = unlearned_suite_config.get("base_model_suite", "")
        if base_suite_name:
            base_suite_config = experiment_suites.get(base_suite_name, {})
            dataset_name = base_suite_config.get("dataset", "cifar10")
            model_type = base_suite_config.get("model", "vit_tiny")
        else:
            dataset_name = suite_config.get("dataset", "cifar10")
            model_type = suite_config.get("model", "vit_tiny")
    else:
        raise ValueError("Suite must specify either 'oracle_model_suite' or 'unlearned_model_suite'")

    # Create data loader for forget class samples
    data_manager = DataManager()

    # Load train dataset and split into forget/retain for training + validation
    train_dataset = data_manager.load_dataset(dataset_name, "train")
    forget_train, retain_train, forget_val, retain_val = create_forget_retain_splits(
        train_dataset, forget_class, train_ratio=0.8
    )

    # Deterministic eval view: same training indices, evaluation transforms.
    # This avoids stochastic val metrics from RandomResizedCrop/flip on held-out train data.
    eval_train_dataset = build_train_eval_view(
        data_manager,
        dataset_name,
        use_pretrained=True,
        apply_imagenet_norm=True,
    )
    if isinstance(forget_val, Subset):
        forget_val = Subset(eval_train_dataset, list(forget_val.indices))

    # Optional stratification by model confidence on forget samples
    stratify_mode = args.stratify or vpt_params.get("stratify", None)
    stratify_fraction = (
        args.stratify_fraction
        if args.stratify_fraction is not None
        else vpt_params.get("stratify_fraction", None)
    )
    if stratify_mode:
        print(f"\nStratifying forget samples by confidence: {stratify_mode}")
        target_model.eval()
        conf_loader = DataLoader(
            forget_train,
            batch_size=256,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        conf_scores = []
        with torch.no_grad():
            for inputs, labels in conf_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = target_model(inputs)
                probs = torch.softmax(outputs, dim=-1)
                conf = probs.gather(1, labels.view(-1, 1)).squeeze(1)
                conf_scores.extend(conf.detach().cpu().tolist())

        # Rank indices by confidence
        idx_conf = list(enumerate(conf_scores))
        # Sort by confidence (high -> low)
        idx_conf.sort(key=lambda x: x[1], reverse=True)
        ranked_indices = [i for i, _ in idx_conf]

        if stratify_mode == "low_conf":
            ranked_indices = list(reversed(ranked_indices))

        if stratify_mode == "mid_conf":
            if stratify_fraction is None:
                stratify_fraction = 0.5

        if stratify_fraction is not None:
            if not (0 < stratify_fraction <= 1.0):
                raise ValueError("stratify_fraction must be in (0,1]")
            keep_n = max(1, int(len(ranked_indices) * stratify_fraction))
            if stratify_mode == "mid_conf":
                start = max(0, (len(ranked_indices) - keep_n) // 2)
                ranked_indices = ranked_indices[start:start + keep_n]
                print(f"Keeping middle {keep_n}/{len(conf_scores)} forget samples after stratification")
            else:
                ranked_indices = ranked_indices[:keep_n]
                print(f"Keeping top {keep_n}/{len(conf_scores)} forget samples after stratification")

        forget_train = Subset(forget_train, ranked_indices)

    # K-shot sampling - limit training data if specified
    k_shot = vpt_params.get('k_shot', None)
    if args.k_shot is not None:
        k_shot = args.k_shot
    if k_shot is not None:
        print(f"\nK-shot mode: Using only {k_shot} samples from forget class")
        # Use torch.utils.data.Subset to limit samples
        # Create deterministic k-shot subset
        random.seed(args.seed)
        indices = list(range(len(forget_train)))
        random.shuffle(indices)
        k_shot_indices = indices[:k_shot]
        forget_train = Subset(forget_train, k_shot_indices)
        print(f"Forget train samples: {len(forget_train)} (original: {len(indices)})")

    # Optional label controls for forget data
    if label_mode != "true":
        def _get_labels(ds):
            # Fast path for datasets with targets
            if hasattr(ds, "targets"):
                return list(ds.targets)
            if hasattr(ds, "labels"):
                return list(ds.labels)
            # Subset support
            if hasattr(ds, "dataset") and hasattr(ds, "indices"):
                base = ds.dataset
                if hasattr(base, "targets"):
                    return [base.targets[i] for i in ds.indices]
                if hasattr(base, "labels"):
                    return [base.labels[i] for i in ds.indices]
            # Fallback (may be slower)
            return [ds[i][1] for i in range(len(ds))]

        # Determine num_classes from dataset config
        num_classes = data_config.get(dataset_name, {}).get("num_classes", None)
        if num_classes is None:
            raise ValueError(f"num_classes not found for dataset '{dataset_name}' in configs/data.yaml")

        labels = _get_labels(forget_train)
        if label_mode == "shuffle":
            random.seed(args.seed)
            shuffled = labels.copy()
            random.shuffle(shuffled)
            # Ensure labels differ from original
            fixed = [(lbl + 1) % num_classes if lbl == orig else lbl
                     for orig, lbl in zip(labels, shuffled)]
            new_labels = fixed
        elif label_mode == "random":
            random.seed(args.seed)
            new_labels = [random.randrange(num_classes) for _ in labels]
            # Ensure labels differ from original
            new_labels = [(lbl + 1) % num_classes if lbl == orig else lbl
                          for orig, lbl in zip(labels, new_labels)]
        else:
            new_labels = labels

        forget_train = LabelOverrideDataset(forget_train, new_labels)
        print(f"Applied label mode '{label_mode}' to forget train data")

    # Explicit loaders (preferred): avoids CPU caching/subsampling inside the attack
    if args.num_workers is not None:
        num_workers = max(int(args.num_workers), 0)
    elif os.name == "nt":
        num_workers = 0
    else:
        num_workers = 0 if label_mode != "true" else 4
    persistent_workers = num_workers > 0
    prefetch_factor = 4 if num_workers > 0 else None
    print(f"DataLoader workers: {num_workers}")
    training_params = vpt_params
    if args.k_shot is not None:
        training_params = dict(training_params)
        training_params["k_shot"] = args.k_shot
    if args.lambda_retain is not None:
        training_params = dict(training_params)
        training_params["lambda_retain"] = args.lambda_retain
    feature_guidance = dict(training_params.get("feature_guidance", {}))
    if args.feature_guide is not None:
        feature_guidance["enabled"] = True
        feature_guidance["mode"] = args.feature_guide
        if args.feature_guide == "oracle_contrastive_probe":
            feature_guidance.setdefault("contrast_weight", 1.0)
            feature_guidance.setdefault("contrast_margin", 0.0)
    if args.feature_guide_weight is not None:
        feature_guidance["enabled"] = True
        feature_guidance["weight"] = args.feature_guide_weight
    if args.feature_contrast_weight is not None:
        feature_guidance["enabled"] = True
        feature_guidance["contrast_weight"] = args.feature_contrast_weight
    if args.feature_contrast_margin is not None:
        feature_guidance["enabled"] = True
        feature_guidance["contrast_margin"] = args.feature_contrast_margin
    if not feature_guidance.get("enabled", False):
        feature_guidance = None
    oracle_model_for_guidance = None
    if feature_guidance is not None and feature_guidance.get("mode") == "oracle_contrastive_probe":
        if is_oracle:
            raise ValueError("oracle_contrastive_probe is only meaningful for unlearned model suites.")
        oracle_suite_name = resolve_matching_oracle_suite(
            experiment_suites,
            dataset_name=dataset_name,
            model_type=model_type,
            forget_class=forget_class,
        )
        oracle_guidance_cfg = {"oracle_model_suite": oracle_suite_name}
        oracle_model_for_guidance = load_oracle_model(
            experiment_suites,
            oracle_guidance_cfg,
            device,
            args.seed,
        )
        feature_guidance["oracle_model_suite"] = oracle_suite_name
        print(f"Loaded oracle model for contrastive guidance: {oracle_suite_name}")
    train_batch_size = training_params.get("train_batch_size", 64)
    retain_batch_size = training_params.get("retain_batch_size", 64)
    eval_batch_size = training_params.get("eval_batch_size", 256)
    forget_train_loader = DataLoader(
        forget_train,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    retain_train_loader = DataLoader(
        retain_train,
        batch_size=retain_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    # Use held-out TRAIN split for validation (no test set leakage)
    forget_val_loader = DataLoader(
        forget_val,
        batch_size=eval_batch_size,  # Large batch for evaluation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    # Create VPT resurrection attack
    vpt_attack = VPTResurrectionAttack(
        target_model=target_model,
        oracle_model=oracle_model_for_guidance,
        forget_class=forget_class,
        prompt_config=vpt_config,
        device=device,
    )

    # Train resurrection prompt
    epochs = training_params.get("epochs", 100)
    lr = training_params.get("lr", 1e-2)

    print(f"\nTraining VPT resurrector for {epochs} epochs with lr={lr}...")

    # Debug: check trainable parameters
    print("Trainable parameters in VPT model:")
    trainable_count = 0
    for name, param in vpt_attack.vpt_model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.shape}")
            trainable_count += 1
    print(f"Total trainable parameters: {trainable_count}")

    attack_history = vpt_attack.train_resurrection_prompt(
        train_forget_loader=forget_train_loader,
        train_retain_loader=retain_train_loader,
        val_forget_loader=forget_val_loader,  # held-out forget split
        epochs=epochs,
        lr=lr,
        weight_decay=training_params.get("weight_decay", 0.0),
        patience=vpt_config_file["vpt_training"]["early_stopping"].get("patience", 20),
        eval_every=training_params.get("eval_every", 10),
        lambda_retain=training_params.get("lambda_retain", 1.0),
        T=training_params.get("T", 1.0),
        feature_guidance=feature_guidance,
    )

    # Save trained VPT prompt
    suite_name_for_io = args.suite
    if args.prompt_length is not None:
        suite_name_for_io = f"{args.suite}_prompt{prompt_length}"
    if args.k_shot is not None:
        suite_name_for_io = f"{suite_name_for_io}_kshot{args.k_shot}"
    if args.lambda_retain is not None:
        lambda_tag = str(args.lambda_retain).replace(".", "p")
        suite_name_for_io = f"{suite_name_for_io}_lam{lambda_tag}"
    if feature_guidance is not None:
        fg_tag = feature_guidance.get("mode", "linear_probe").replace("_", "")
        suite_name_for_io = f"{suite_name_for_io}_fg{fg_tag}"
    if stratify_mode:
        if stratify_mode == "high_conf":
            suffix = "highconf"
        elif stratify_mode == "mid_conf":
            suffix = "midconf"
        else:
            suffix = "lowconf"
        suite_name_for_io = f"{suite_name_for_io}_{suffix}"
    if label_mode != "true":
        suite_name_for_io = f"{suite_name_for_io}_{label_mode}labels"

    prompt_save_path = f"checkpoints/vpt_resurrector/{suite_name_for_io}_seed_{args.seed}"
    vpt_attack.save_attack_prompt(prompt_save_path)

    num_classes = data_config.get(dataset_name, {}).get("num_classes", 10)
    clean_eval_metrics = {}
    if attack_history:
        print("\nRunning clean test evaluation on restored prompted model...")
        try:
            clean_eval_metrics = evaluate_clean_test_metrics(
                model=vpt_attack.vpt_model,
                target_model=target_model,
                dataset_name=dataset_name,
                forget_class=forget_class,
                num_classes=num_classes,
                device=device,
                batch_size=eval_batch_size,
                num_workers=0,
            )
            attack_history[-1].update(clean_eval_metrics)
        except Exception as exc:
            clean_eval_metrics = {"test_clean_eval_error": str(exc)}
            attack_history[-1].update(clean_eval_metrics)
            print(f"Warning: clean test evaluation failed: {exc}")

    # Log results
    log_path = create_experiment_log(f"{suite_name_for_io}_seed_{args.seed}", suite_config)
    if os.path.exists(log_path) and not args.append_log:
        print(f"Overwriting existing log: {log_path}")
        os.remove(log_path)

    for entry in attack_history:
        log_experiment(log_path, entry)

    print("\nVPT resurrector training complete!")
    print(f"VPT prompt saved to: {prompt_save_path}")

    # Print final metrics
    if attack_history:
        final_metrics = attack_history[-1]
        print("\nFinal resurrection metrics:")
        for key, value in final_metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                print(f"  {key}: {value:.4f}")

    # Get prompt statistics
    prompt_stats = vpt_attack.get_prompt_statistics()
    print("\nPrompt statistics:")
    for key, value in prompt_stats.items():
        print(f"  {key}: {value}")

    summary_path = f"results/analysis/{suite_name_for_io}_seed_{args.seed}_summary.json"
    save_dict_to_json(
        {
            "suite": args.suite,
            "suite_name_for_io": suite_name_for_io,
            "seed": args.seed,
            "dataset": dataset_name,
            "forget_class": forget_class,
            "prompt_save_path": prompt_save_path,
            "log_path": log_path,
            "feature_guidance": feature_guidance,
            "prompt_stats": prompt_stats,
            "final_metrics": attack_history[-1] if attack_history else {},
        },
        summary_path,
    )
    print(f"Summary written to: {summary_path}")

    print("\nNext: Run adversarial evaluation with: python scripts/4_adv_evaluate.py")


if __name__ == "__main__":
    main()
