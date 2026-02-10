#!/usr/bin/env python3
"""
Script 2: Train LoRA unlearning adapter
"""

import argparse
import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from typing import Dict
import yaml

from src.data import DataManager
from src.data import CombinedDataLoader, create_forget_retain_splits
from src.models.vit import create_vit_model
from src.models.cnn import create_cnn_model
from src.models.peft_lora import create_lora_config, apply_lora_to_model, save_lora_adapter
from src.models.pruning import apply_global_pruning
from src.unlearning.trainer import create_unlearning_trainer
from src.utils import set_seed, ensure_dir, load_config, print_model_info, get_device
from src.utils import log_experiment, create_experiment_log


def load_base_model(experiment_suites: Dict, suite_config: Dict, device: torch.device, seed: int) -> nn.Module:
    """Load pretrained base model."""
    base_suite = suite_config.get("base_model_suite", "")

    checkpoint_path = f"checkpoints/base/{base_suite}_seed_{seed}_final.pt"
    fallback_path = f"checkpoints/base/{base_suite}_seed_{seed}_best.pt"

    if not os.path.exists(checkpoint_path):
        if os.path.exists(fallback_path):
            checkpoint_path = fallback_path
        else:
            raise FileNotFoundError(
                f"Base model checkpoint not found: {checkpoint_path} (or fallback {fallback_path})"
            )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Recreate model architecture using base suite config
    data_config = load_config("configs/data.yaml")
    model_config = load_config("configs/model.yaml")

    # Get base suite config to read model/dataset info
    base_suite_config = experiment_suites.get(base_suite, {})
    dataset_name = base_suite_config.get("dataset", "cifar10")
    dataset_info = data_config[dataset_name]
    model_type = base_suite_config.get("model", "vit_tiny")

    if model_type.startswith("vit"):
        model_config_name = model_type.replace("vit_", "")
        model = create_vit_model(
            model_config["vit"][model_config_name],
            num_classes=dataset_info["num_classes"],
        )
    else:
        model = create_cnn_model(
            model_config["cnn"][model_type],
            num_classes=dataset_info["num_classes"],
        )

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print(f"Loaded base model from: {checkpoint_path}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train LoRA unlearning adapter")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment suites config")
    parser.add_argument("--suite", type=str, required=True, help="Experiment suite name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device to use")

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Load configs
    experiment_suites = load_config(args.config)
    if args.suite not in experiment_suites:
        raise ValueError(f"Experiment suite '{args.suite}' not found in config")

    suite_config = experiment_suites[args.suite]
    data_config = load_config("configs/data.yaml")
    model_config = load_config("configs/model.yaml")
    unlearning_config = load_config("configs/unlearning.yaml")

    # Setup device
    device = get_device(args.device)
    print(f"Run: Training LoRA Unlearning - {args.suite}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")

    # Get unlearning parameters
    unlearning_params = suite_config.get("unlearning", {})
    forget_class = unlearning_params.get("forget_class", 0)
    objective_name = unlearning_params.get("objective", "ce_ascent")
    lora_rank = unlearning_params.get("lora_rank", 8)
    unlearning_method = unlearning_params.get("method", "lora")

    # Get objective-specific config from unlearning.yaml
    objective_config = unlearning_config.get("objectives", {}).get(objective_name, {})

    # Extract all parameters except 'name' and 'loss_type' (metadata only)
    objective_kwargs = {
        k: v
        for k, v in objective_config.items()
        if k not in ["name", "loss_type", "normal_source", "scrub_full_schedule", "scrub_max_steps"]
    }

    print(f"Forget class: {forget_class}")
    print(f"Unlearning objective: {objective_name}")
    print(f"LoRA rank: {lora_rank}")
    print(f"Unlearning method: {unlearning_method}")

    # Load base model
    base_model = load_base_model(experiment_suites, suite_config, device, args.seed)

    # Optional pruning before unlearning
    pruning_defaults = unlearning_config.get("pruning", {})
    pruning_cfg = dict(pruning_defaults)
    pruning_cfg.update(unlearning_params.get('pruning', {}) or {})
    if pruning_cfg.get("enabled", False):
        pruned = apply_global_pruning(
            base_model,
            amount=pruning_cfg.get("amount", 0.2),
            module_types=pruning_cfg.get("module_types", ["Linear"]),
            make_permanent=pruning_cfg.get("make_permanent", True),
            strategy=pruning_cfg.get("strategy", "global_unstructured"),
        )
        print(f"Applied pruning before unlearning: pruned_params={pruned}")

    # For SCRUB, we need a separate teacher model (frozen copy of base)
    # Load it BEFORE applying LoRA to avoid reference issues
    teacher_model = None
    if "scrub" in objective_name and "feature" not in objective_name:
        print("\nLoading separate teacher model for SCRUB...")
        teacher_model = load_base_model(experiment_suites, suite_config, device, args.seed)
        if pruning_cfg.get("enabled", False) and pruning_cfg.get("apply_to_teacher", True):
            pruned_t = apply_global_pruning(
                teacher_model,
                amount=pruning_cfg.get("amount", 0.2),
                module_types=pruning_cfg.get("module_types", ["Linear"]),
                make_permanent=pruning_cfg.get("make_permanent", True),
                strategy=pruning_cfg.get("strategy", "global_unstructured"),
            )
            print(f"Applied pruning to teacher model: pruned_params={pruned_t}")
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        print(f"Teacher model loaded and frozen: {type(teacher_model).__name__}")

    # Get model type from base suite for LoRA target selection
    base_suite = suite_config.get("base_model_suite", "")
    base_suite_config = experiment_suites.get(base_suite, {})
    model_type = base_suite_config.get("model", "vit_tiny")

    def _unwrap_core_model(m: nn.Module) -> nn.Module:
        if hasattr(m, "model"):
            return m.model
        return m

    def _freeze_all(m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def _unfreeze_names(m: nn.Module, name_keywords):
        for name, param in m.named_parameters():
            if any(k in name for k in name_keywords):
                param.requires_grad = True

    def _unfreeze_last_block(m: nn.Module):
        core = _unwrap_core_model(m)
        # ViT: blocks[-1]
        if hasattr(core, "blocks") and len(core.blocks) > 0:
            for p in core.blocks[-1].parameters():
                p.requires_grad = True
            # also unfreeze head
            if hasattr(core, "head"):
                for p in core.head.parameters():
                    p.requires_grad = True
            if hasattr(core, "fc"):
                for p in core.fc.parameters():
                    p.requires_grad = True
            return
        # ResNet: layer4 + fc
        if hasattr(core, "layer4"):
            for p in core.layer4.parameters():
                p.requires_grad = True
        if hasattr(core, "fc"):
            for p in core.fc.parameters():
                p.requires_grad = True

    # Apply LoRA or configure finetuning method
    if unlearning_method == "lora":
        if model_type.startswith("vit"):
            target_modules = model_config["lora"]["vit_target_modules"]
        else:
            target_modules = model_config["lora"]["cnn_target_modules"]

        lora_config = create_lora_config(
            r=lora_rank,
            lora_alpha=unlearning_config["lora_unlearn"].get("lora_alpha", 16),
            target_modules=target_modules
        )
        lora_model = apply_lora_to_model(base_model, lora_config)
        print_model_info(lora_model, f"LoRA-{lora_rank} {objective_name}")
    else:
        lora_model = base_model
        _freeze_all(lora_model)
        if unlearning_method == "head_only":
            _unfreeze_names(lora_model, ["head", "fc", "classifier"])
        elif unlearning_method == "last_block":
            _unfreeze_last_block(lora_model)
        elif unlearning_method == "full_finetune":
            for p in lora_model.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"Unknown unlearning method: {unlearning_method}")
        print_model_info(lora_model, f"{unlearning_method} {objective_name}")

    # Create data manager and loaders
    data_manager = DataManager()
    base_suite_config = experiment_suites.get(suite_config.get("base_model_suite", ""), {})
    dataset_name = base_suite_config.get("dataset", "cifar10")

    # Create forget/retain split
    full_train_dataset = data_manager.load_dataset(dataset_name, "train")

    # Objective-specific data stats (FaLW-style)
    forget_train, retain_train, forget_test, retain_test = create_forget_retain_splits(
        full_train_dataset, forget_class, train_ratio=0.8
    )
    if objective_name in ("falw", "fa_lw"):
        # FaLW balance factor should be computed from the forget set distribution.
        if hasattr(forget_train, "dataset") and hasattr(forget_train, "indices"):
            base_dataset = forget_train.dataset
            indices = forget_train.indices
            if hasattr(base_dataset, "targets") and base_dataset.targets is not None:
                labels = [int(base_dataset.targets[i]) for i in indices]
            else:
                labels = [int(base_dataset[i][1]) for i in indices]
        else:
            labels = [int(forget_train[i][1]) for i in range(len(forget_train))]
        class_counts = {}
        for y in labels:
            class_counts[y] = class_counts.get(y, 0) + 1
        objective_kwargs["class_counts"] = class_counts
        objective_kwargs["total_samples"] = len(labels)

    # Create data loaders
    default_workers = unlearning_config["lora_unlearn"].get("num_workers")
    if default_workers is None:
        default_workers = 0 if os.name == "nt" else 4
    pin_memory = unlearning_config["lora_unlearn"].get("pin_memory")
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    forget_loader = DataLoader(
        forget_train,
        batch_size=unlearning_config["lora_unlearn"].get("batch_size", 128),
        shuffle=True,
        num_workers=default_workers,
        pin_memory=pin_memory
    )

    retain_loader = DataLoader(
        retain_train,
        batch_size=unlearning_config["lora_unlearn"].get("batch_size", 128),
        shuffle=True,
        num_workers=default_workers,
        pin_memory=pin_memory
    )

    normal_loader = None
    if objective_name in ("sga", "smoothed_ga"):
        normal_source = unlearning_params.get("normal_source", objective_config.get("normal_source", "retain"))
        normal_cfg = unlearning_params.get("normal_data")
        if normal_cfg:
            normal_dataset = normal_cfg.get("dataset", dataset_name)
            normal_split = normal_cfg.get("split", "train")
            exclude_class = normal_cfg.get("exclude_class")
            include_classes = normal_cfg.get("include_classes")
            normal_loader = data_manager.get_dataloader(
                normal_dataset,
                split=normal_split,
                batch_size=unlearning_config["lora_unlearn"].get("batch_size", 128),
                include_classes=include_classes,
                exclude_classes=exclude_class,
                num_workers=default_workers,
                use_pretrained=True,
                apply_imagenet_norm=True
            )
            print(f"SGA normal data loader: {normal_dataset}:{normal_split}")
        elif normal_source == "retain":
            normal_loader = retain_loader

    print(f"Forget class {forget_class} samples: {len(forget_train)}")
    print(f"Retain samples: {len(retain_train)}")

    # Create combined validation dataset (forget_test + retain_test for proper validation metrics)
    val_dataset = ConcatDataset([forget_test, retain_test])
    val_loader = DataLoader(
        val_dataset,
        batch_size=unlearning_config["lora_unlearn"].get("batch_size", 128),
        shuffle=False,  # No shuffle for validation
        num_workers=default_workers,
        pin_memory=pin_memory
    )

    # Create combined loader for training (alternates forget/retain batches)
    combined_train_loader = CombinedDataLoader(
        forget_loader,
        retain_loader,
        steps_per_epoch=len(retain_loader)  # full pass over retain each epoch
    )

    # Create unlearning trainer
    retain_lambda = unlearning_config["lora_unlearn"].get("retain_lambda", 1.0)
    grad_noise_std = unlearning_params.get(
        "grad_noise_std",
        unlearning_config["lora_unlearn"].get("grad_noise_std", 0.0),
    )
    gu_projection = unlearning_params.get("gu_projection", False)
    grad_surgery = unlearning_params.get("grad_surgery", False)
    orthogonal_reg = float(unlearning_params.get("orthogonal_reg", 0.0))
    gu_beta = float(unlearning_params.get("gu_beta", 1.0))
    scrub_rewind = bool(unlearning_params.get("scrub_rewind", False))
    scrub_rewind_on_train = bool(unlearning_params.get("scrub_rewind_on_train", True))
    scrub_full_schedule = bool(unlearning_params.get("scrub_full_schedule", False))
    scrub_max_steps = unlearning_params.get("scrub_max_steps")
    scrub_max_steps = None if scrub_max_steps is None else int(scrub_max_steps)
    trainer = create_unlearning_trainer(
        model=lora_model,
        objective_name=objective_name,
        forget_class=forget_class,
        train_loader=combined_train_loader,  # Use combined loader
        val_loader=val_loader,  # Use combined validation set
        normal_loader=normal_loader,
        num_classes=data_config[dataset_name]["num_classes"],
        device=device,
        retain_lambda=retain_lambda,
        objective_kwargs=objective_kwargs,
        grad_noise_std=grad_noise_std,
        gu_projection=gu_projection,
        grad_surgery=grad_surgery,
        orthogonal_reg=orthogonal_reg,
        gu_beta=gu_beta,
        scrub_rewind=scrub_rewind,
        scrub_rewind_on_train=scrub_rewind_on_train,
        scrub_full_schedule=scrub_full_schedule,
        scrub_max_steps=scrub_max_steps
    )

    # Set teacher model for SCRUB (distillation-based unlearning)
    if "scrub" in objective_name and "feature" not in objective_name:
        print("\nSetting teacher model in SCRUB objective...")
        # Use the separately loaded teacher model
        trainer.set_teacher_model(teacher_model)
        print(f"Teacher model set successfully!")

    # Configure optimizer and scheduler
    trainer.configure_optimizer(
        lr=unlearning_params.get("lr", 1e-3),
        weight_decay=unlearning_config["lora_unlearn"].get("weight_decay", 0.01),
    )

    # Training parameters
    max_epochs = unlearning_params.get("epochs", 50)

    trainer.configure_scheduler(
        scheduler_name="cosine",
        max_epochs=max_epochs,
        min_lr=unlearning_config["lora_unlearn"].get("min_lr", 1e-6)
    )
    checkpoint_dir = f"checkpoints/unlearn_lora/{args.suite}_seed_{args.seed}"

    # Train
    print(f"\nStarting unlearning training for {max_epochs} epochs...")
    training_history = trainer.train(
        max_epochs=max_epochs,
        early_stopping_patience=unlearning_config["early_stopping"].get("patience", 3),
        save_every=10,
        checkpoint_dir=checkpoint_dir,
        monitor_metric="forget_acc",
        mode="min"
    )

    # Save training history
    history_path = f"results/logs/{args.suite}_seed_{args.seed}_history.json"
    trainer.save_training_history(history_path)

    if unlearning_method == "lora":
        # Save final LoRA adapter
        adapter_save_path = f"checkpoints/unlearn_lora/{args.suite}_seed_{args.seed}"
        save_lora_adapter(lora_model, adapter_save_path)

        # If a best model was saved during training, overwrite with it
        best_model_path = os.path.join(checkpoint_dir, "best_model")
        if os.path.exists(best_model_path):
            print(f"Copying best model from {best_model_path} to {adapter_save_path}")
            # Remove the final adapter files
            for filename in os.listdir(adapter_save_path):
                if filename.endswith(('.bin', '.safetensors', '.json')):
                    os.remove(os.path.join(adapter_save_path, filename))
            # Copy best model files
            for filename in os.listdir(best_model_path):
                if filename.endswith(('.bin', '.safetensors', '.json')):
                    shutil.copy2(os.path.join(best_model_path, filename), adapter_save_path)

        print("\nUnlearning training complete!")
        print(f"LoRA adapter saved to: {adapter_save_path}")
    else:
        full_save_path = f"checkpoints/unlearn_full/{args.suite}_seed_{args.seed}_final.pt"
        os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
        torch.save({"model_state_dict": lora_model.state_dict()}, full_save_path)
        print("\nUnlearning training complete!")
        print(f"Full model saved to: {full_save_path}")
    print(f"Training history saved to: {history_path}")

    # Print final metrics
    if training_history:
        final_metrics = training_history[-1]
        print("Final metrics:")
        for key, value in final_metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                print(f"  {key}: {value:.4f}")

    print("\nNext: Train VPT resurrector with: python scripts/3_train_vpt_resurrector.py")


if __name__ == "__main__":
    main()
