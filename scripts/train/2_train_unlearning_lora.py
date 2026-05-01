#!/usr/bin/env python3
"""
Script 2: Train LoRA unlearning adapter
"""

import argparse
import os
import sys
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from typing import Dict, Optional
import yaml

from src.data import DataManager
from src.data import CombinedDataLoader, create_forget_retain_splits
from src.models.vit import create_vit_model
from src.models.cnn import create_cnn_model
from src.models.peft_lora import create_lora_config, apply_lora_to_model, save_lora_adapter, load_lora_adapter
from src.models.pruning import apply_global_pruning
from src.unlearning.trainer import create_unlearning_trainer
from src.unlearning.objectives import _extract_features
from src.utils import set_seed, ensure_dir, load_config, print_model_info, get_device
from src.utils import log_experiment, create_experiment_log


def _build_model_from_suite(experiment_suites: Dict, suite_name: str, device: torch.device) -> nn.Module:
    """Recreate the model architecture for a named experiment suite."""
    data_config = load_config("configs/data.yaml")
    model_config = load_config("configs/model.yaml")

    suite_info = experiment_suites.get(suite_name, {})
    if "base_model_suite" in suite_info:
        reference_info = experiment_suites.get(suite_info["base_model_suite"], {})
    else:
        reference_info = suite_info

    dataset_name = reference_info.get("dataset", suite_info.get("dataset", "cifar10"))
    dataset_info = data_config[dataset_name]
    model_type = reference_info.get("model", suite_info.get("model", "vit_tiny"))

    if model_type.startswith("vit"):
        model_config_name = model_type.replace("vit_", "")
        vit_cfg = dict(model_config["vit"][model_config_name])
        vit_cfg["pretrained"] = False
        model = create_vit_model(
            vit_cfg,
            num_classes=dataset_info["num_classes"],
        )
    else:
        model = create_cnn_model(
            model_config["cnn"][model_type],
            num_classes=dataset_info["num_classes"],
        )

    return model.to(device)


def load_checkpointed_model(experiment_suites: Dict,
                            suite_name: str,
                            checkpoint_root: str,
                            device: torch.device,
                            seed: int) -> nn.Module:
    """Load a checkpointed model from checkpoints/<root>/<suite>_seed_<seed>_*.pt."""
    checkpoint_path = f"checkpoints/{checkpoint_root}/{suite_name}_seed_{seed}_final.pt"
    fallback_path = f"checkpoints/{checkpoint_root}/{suite_name}_seed_{seed}_best.pt"

    if not os.path.exists(checkpoint_path):
        if os.path.exists(fallback_path):
            checkpoint_path = fallback_path
        else:
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path} (or fallback {fallback_path})"
            )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = _build_model_from_suite(experiment_suites, suite_name, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from: {checkpoint_path}")
    return model


def resolve_unlearn_artifact_path(experiment_suites: Dict, suite_name: str, seed: int) -> str:
    """Resolve a saved artifact path for an unlearn suite."""
    suite_info = experiment_suites.get(suite_name, {})
    explicit_path = suite_info.get("path")
    if explicit_path:
        return explicit_path.format(seed=seed, suite_name=suite_name)

    method = suite_info.get("unlearning", {}).get("method", "lora")
    if method == "lora":
        return f"checkpoints/unlearn_lora/{suite_name}_seed_{seed}"
    return f"checkpoints/unlearn_full/{suite_name}_seed_{seed}_final.pt"


def load_unlearned_model_from_suite(experiment_suites: Dict,
                                    suite_name: str,
                                    device: torch.device,
                                    seed: int,
                                    is_trainable: bool = False) -> nn.Module:
    """Load an unlearned model from a named suite, handling both LoRA and full checkpoints."""
    suite_info = experiment_suites.get(suite_name)
    if suite_info is None:
        raise ValueError(f"Experiment suite '{suite_name}' not found in config")

    base_model = _build_model_from_suite(experiment_suites, suite_name, device)
    artifact_path = resolve_unlearn_artifact_path(experiment_suites, suite_name, seed)
    artifact_path = os.path.normpath(artifact_path)

    if os.path.isdir(artifact_path):
        adapter_cfg = os.path.join(artifact_path, "adapter_config.json")
        adapter_weights = os.path.join(artifact_path, "adapter_model.safetensors")
        model_file = os.path.join(artifact_path, "model.pt")
        if os.path.exists(adapter_cfg) and os.path.exists(adapter_weights):
            model = load_lora_adapter(base_model, artifact_path, is_trainable=is_trainable)
            print(f"Loaded LoRA model from suite '{suite_name}': {artifact_path}")
            return model
        if os.path.exists(model_file):
            checkpoint = torch.load(model_file, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            base_model.load_state_dict(state_dict)
            print(f"Loaded checkpoint model from suite '{suite_name}': {model_file}")
            return base_model
        raise FileNotFoundError(f"No adapter or model.pt found in {artifact_path}")

    if os.path.isfile(artifact_path):
        checkpoint = torch.load(artifact_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        base_model.load_state_dict(state_dict)
        print(f"Loaded checkpoint model from suite '{suite_name}': {artifact_path}")
        return base_model

    raise FileNotFoundError(
        f"Artifact for suite '{suite_name}' not found at {artifact_path}"
    )


def find_resume_checkpoint(checkpoint_dir: str) -> str:
    """Return the most recent resumable checkpoint directory, if any."""
    candidates = []

    def _add_candidate(path: str):
        state_path = os.path.join(path, "training_state.pt")
        adapter_path = os.path.join(path, "adapter_config.json")
        if os.path.exists(state_path) and os.path.exists(adapter_path):
            try:
                state = torch.load(state_path, map_location="cpu")
                epoch = int(state.get("epoch", -1))
            except Exception:
                epoch = -1
            candidates.append((epoch, path))

    if os.path.exists(checkpoint_dir):
        _add_candidate(checkpoint_dir)

        for name in os.listdir(checkpoint_dir):
            full = os.path.join(checkpoint_dir, name)
            if os.path.isdir(full) and (name.startswith("checkpoint_epoch_") or name == "best_model"):
                _add_candidate(full)

    if not candidates:
        return ""

    candidates.sort(key=lambda item: (item[0], item[1].endswith("best_model")), reverse=True)
    return candidates[0][1]


def resolve_init_adapter_dir(experiment_suites: Dict, suite_config: Dict, seed: int) -> str:
    """Resolve an optional LoRA adapter initialization directory."""
    direct_path = suite_config.get("init_adapter_path")
    if direct_path and os.path.exists(direct_path):
        return direct_path

    init_suite = suite_config.get("init_adapter_suite")
    if not init_suite:
        return ""

    init_seed = int(suite_config.get("init_adapter_seed", seed))
    candidate = f"checkpoints/unlearn_lora/{init_suite}_seed_{init_seed}"
    if os.path.exists(candidate):
        return candidate

    best_candidate = os.path.join(candidate, "best_model")
    if os.path.exists(best_candidate):
        return best_candidate

    raise FileNotFoundError(
        f"Init adapter for suite '{init_suite}' not found at {candidate} or {best_candidate}"
    )


def resolve_suite_adapter_dir(suite_name: str, seed: int) -> str:
    """Resolve a saved LoRA adapter directory for a named suite."""
    candidate = f"checkpoints/unlearn_lora/{suite_name}_seed_{seed}"
    if os.path.exists(candidate):
        return candidate

    best_candidate = os.path.join(candidate, "best_model")
    if os.path.exists(best_candidate):
        return best_candidate

    raise FileNotFoundError(
        f"Adapter for suite '{suite_name}' not found at {candidate} or {best_candidate}"
    )


def load_lora_model_from_suite(experiment_suites: Dict,
                               suite_name: str,
                               device: torch.device,
                               seed: int,
                               is_trainable: bool = False) -> nn.Module:
    """Load a LoRA model from a named experiment suite."""
    suite_info = experiment_suites.get(suite_name)
    if suite_info is None:
        raise ValueError(f"Experiment suite '{suite_name}' not found in config")

    base_suite = suite_info.get("base_model_suite", "")
    if not base_suite:
        raise ValueError(f"Suite '{suite_name}' does not define base_model_suite")

    base_model = load_checkpointed_model(experiment_suites, base_suite, "base", device, seed)
    adapter_dir = resolve_suite_adapter_dir(suite_name, seed)
    model = load_lora_adapter(base_model, adapter_dir, is_trainable=is_trainable)
    return model.to(device)


def compute_feature_subspace(model: nn.Module,
                             dataset,
                             device: torch.device,
                             batch_size: int,
                             num_workers: int,
                             pin_memory: bool,
                             max_samples: Optional[int],
                             rank: int,
                             normalize_features: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a PCA basis for forget features from a frozen teacher model."""
    if max_samples is not None and len(dataset) > int(max_samples):
        generator = torch.Generator().manual_seed(0)
        perm = torch.randperm(len(dataset), generator=generator)[:int(max_samples)].tolist()
        dataset = Subset(dataset, perm)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    features = []
    model.eval()
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            feats = _extract_features(model, inputs)
            if normalize_features:
                feats = torch.nn.functional.normalize(feats, dim=-1)
            features.append(feats.detach().cpu())

    if not features:
        raise ValueError("Could not compute feature subspace: no feature batches found.")

    feats = torch.cat(features, dim=0)
    center = feats.mean(dim=0)
    centered = feats - center.unsqueeze(0)
    if centered.shape[0] < 2:
        basis = torch.nn.functional.normalize(centered, dim=-1)
        return basis, center

    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    q = max(1, min(int(rank), vh.shape[0], vh.shape[1]))
    basis = vh[:q].contiguous()
    return basis, center


def main():
    parser = argparse.ArgumentParser(description="Train LoRA unlearning adapter")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment suites config")
    parser.add_argument("--suite", type=str, required=True, help="Experiment suite name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument("--auto-resume", action="store_true", help="Resume from the latest saved checkpoint if available")

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
    extra_forget_classes = unlearning_params.get("forget_classes", [])
    forget_classes = [int(forget_class), *[int(c) for c in extra_forget_classes if int(c) != int(forget_class)]]
    objective_name = unlearning_params.get("objective", "ce_ascent")
    lora_rank = unlearning_params.get("lora_rank", 8)
    unlearning_method = unlearning_params.get("method", "lora")
    checkpoint_dir = f"checkpoints/unlearn_lora/{args.suite}_seed_{args.seed}"
    resume_checkpoint_dir = find_resume_checkpoint(checkpoint_dir) if args.auto_resume else ""
    init_adapter_dir = resolve_init_adapter_dir(experiment_suites, suite_config, args.seed) if not resume_checkpoint_dir else ""

    # Get objective-specific config from unlearning.yaml
    objective_config = unlearning_config.get("objectives", {}).get(objective_name, {})

    # Extract all parameters except 'name' and 'loss_type' (metadata only)
    objective_kwargs = {
        k: v
        for k, v in objective_config.items()
        if k not in ["name", "loss_type", "normal_source", "scrub_full_schedule", "scrub_max_steps"]
    }
    objective_meta_keys = {
        "method", "objective", "forget_class", "lora_rank", "epochs", "lr",
        "batch_size", "val_batch_size", "grad_noise_std", "gu_projection",
        "grad_surgery", "orthogonal_reg", "gu_beta", "scrub_rewind",
        "scrub_rewind_on_train", "scrub_full_schedule", "scrub_max_steps",
        "normal_source", "normal_data", "pruning", "max_forget_samples",
        "max_retain_samples", "robust_retain", "robust_retain_eps",
        "robust_retain_alpha", "robust_retain_steps", "basis_rank",
        "basis_max_samples", "forget_classes", "forget_class_counts"
    }
    for key, value in unlearning_params.items():
        if key in objective_meta_keys:
            continue
        if key in objective_config or key not in objective_kwargs:
            objective_kwargs[key] = value

    if len(forget_classes) == 1:
        print(f"Forget class: {forget_class}")
    else:
        print(f"Forget classes: {forget_classes} (primary target: {forget_class})")
    print(f"Unlearning objective: {objective_name}")
    print(f"LoRA rank: {lora_rank}")
    print(f"Unlearning method: {unlearning_method}")
    if init_adapter_dir:
        print(f"Init adapter: {init_adapter_dir}")

    # Load base model
    base_suite = suite_config.get("base_model_suite", "")
    base_model = load_checkpointed_model(experiment_suites, base_suite, "base", device, args.seed)

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
    teacher_suite = suite_config.get("teacher_model_suite") or suite_config.get("teacher_adapter_suite")
    if teacher_suite:
        print(f"\nLoading teacher model from suite: {teacher_suite}")
        teacher_model = load_unlearned_model_from_suite(
            experiment_suites, teacher_suite, device, args.seed, is_trainable=False
        )
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        print(f"Teacher model loaded and frozen: {type(teacher_model).__name__}")
    elif "scrub" in objective_name and "feature" not in objective_name:
        print("\nLoading separate teacher model for SCRUB...")
        teacher_model = load_checkpointed_model(experiment_suites, base_suite, "base", device, args.seed)
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

    oracle_model = None
    if objective_name == "orbit":
        oracle_suite = unlearning_params.get("oracle_model_suite", f"oracle_vit_cifar10_forget{forget_class}")
        print(f"\nLoading oracle model for ORBIT: {oracle_suite}")
        oracle_model = load_checkpointed_model(experiment_suites, oracle_suite, "oracle", device, args.seed)
        oracle_model.eval()
        for param in oracle_model.parameters():
            param.requires_grad = False
        print(f"Oracle model loaded and frozen: {type(oracle_model).__name__}")

    # Get model type from base suite for LoRA target selection
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
        if resume_checkpoint_dir:
            print(f"Resuming LoRA adapter from: {resume_checkpoint_dir}")
            lora_model = load_lora_adapter(base_model, resume_checkpoint_dir, is_trainable=True)
        elif init_adapter_dir:
            print(f"Initializing LoRA adapter from: {init_adapter_dir}")
            lora_model = load_lora_adapter(base_model, init_adapter_dir, is_trainable=True)
        else:
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
        init_model_suite = suite_config.get("init_model_suite")
        init_model_path = suite_config.get("init_model_path")
        if init_model_path and os.path.exists(init_model_path):
            checkpoint = torch.load(init_model_path, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            base_model.load_state_dict(state_dict)
            print(f"Initialized model weights from: {init_model_path}")
        elif init_model_suite:
            print(f"Initializing model weights from suite: {init_model_suite}")
            base_model = load_unlearned_model_from_suite(
                experiment_suites, init_model_suite, device, args.seed, is_trainable=False
            )

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
        full_train_dataset, forget_classes, train_ratio=0.8
    )

    def _limit_subset(dataset, max_samples):
        if max_samples is None:
            return dataset
        max_samples = int(max_samples)
        if max_samples <= 0 or len(dataset) <= max_samples:
            return dataset
        generator = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
        return Subset(dataset, perm)

    def _limit_subset_per_class(dataset, class_caps):
        if not class_caps:
            return dataset
        normalized_caps = {int(k): int(v) for k, v in class_caps.items() if int(v) > 0}
        if not normalized_caps:
            return dataset

        if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
            base_dataset = dataset.dataset
            indices = list(dataset.indices)
        else:
            base_dataset = dataset
            indices = list(range(len(dataset)))

        generator = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(indices), generator=generator).tolist()
        shuffled_indices = [indices[i] for i in perm]

        kept = []
        seen = {cls: 0 for cls in normalized_caps}
        for idx in shuffled_indices:
            if hasattr(base_dataset, "targets") and base_dataset.targets is not None:
                label = int(base_dataset.targets[idx])
            else:
                label = int(base_dataset[idx][1])
            if label not in normalized_caps:
                continue
            if seen[label] >= normalized_caps[label]:
                continue
            kept.append(idx)
            seen[label] += 1
            if all(seen[cls] >= normalized_caps[cls] for cls in normalized_caps):
                break
        if not kept:
            return dataset
        return Subset(base_dataset, kept)

    max_forget_samples = unlearning_params.get("max_forget_samples")
    max_retain_samples = unlearning_params.get("max_retain_samples")
    forget_class_counts = unlearning_params.get("forget_class_counts")
    forget_train = _limit_subset_per_class(forget_train, forget_class_counts)
    forget_train = _limit_subset(forget_train, max_forget_samples)
    retain_train = _limit_subset(retain_train, max_retain_samples)
    if objective_name in ("falw", "fa_lw"):
        # FaLW's balancing term is defined over the forget-set class distribution.
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
        objective_kwargs["forget_classes"] = forget_classes
        if len(class_counts) <= 1:
            print(
                "Warning: FaLW is being run on a single-class forget benchmark. "
                "The long-tail class-balance factor is neutralized, so this run "
                "only exercises FaLW's per-sample forgetting-aware reweighting."
            )
        else:
            print(f"FaLW forget-set class counts: {class_counts}")

    # Create data loaders
    train_batch_size = int(unlearning_params.get(
        "batch_size",
        unlearning_config["lora_unlearn"].get("batch_size", 128),
    ))
    val_batch_size = int(unlearning_params.get("val_batch_size", train_batch_size))
    default_workers = unlearning_config["lora_unlearn"].get("num_workers")
    if default_workers is None:
        default_workers = 0 if os.name == "nt" else 4
    pin_memory = unlearning_config["lora_unlearn"].get("pin_memory")
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    print(f"Train batch size: {train_batch_size}")
    print(f"Validation batch size: {val_batch_size}")

    forget_loader = DataLoader(
        forget_train,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=default_workers,
        pin_memory=pin_memory
    )

    retain_loader = DataLoader(
        retain_train,
        batch_size=train_batch_size,
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
                batch_size=train_batch_size,
                include_classes=include_classes,
                exclude_classes=exclude_class,
                num_workers=default_workers,
                use_pretrained=True,
                apply_imagenet_norm=True
            )
            print(f"SGA normal data loader: {normal_dataset}:{normal_split}")
        elif normal_source == "retain":
            normal_loader = retain_loader

    print(f"Forget set samples: {len(forget_train)}")
    print(f"Retain samples: {len(retain_train)}")

    # Create combined validation dataset (forget_test + retain_test for proper validation metrics)
    val_dataset = ConcatDataset([forget_test, retain_test])
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
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
    if objective_name == "ce_ascent" and orthogonal_reg > 0:
        print(
            "Warning: orthogonal_reg in this repo is a projected-gradient proxy "
            "on top of CE ascent, not a standalone paper-faithful OrthoReg loss."
        )
    gu_beta = float(unlearning_params.get("gu_beta", 1.0))
    robust_retain = bool(unlearning_params.get("robust_retain", False))
    robust_retain_eps = float(unlearning_params.get("robust_retain_eps", 8 / 255))
    robust_retain_alpha = float(unlearning_params.get("robust_retain_alpha", 2 / 255))
    robust_retain_steps = int(unlearning_params.get("robust_retain_steps", 5))
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
        robust_retain=robust_retain,
        robust_retain_eps=robust_retain_eps,
        robust_retain_alpha=robust_retain_alpha,
        robust_retain_steps=robust_retain_steps,
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
    if teacher_model is not None and hasattr(trainer.objective, "set_teacher_model"):
        print("\nSetting teacher model in objective...")
        trainer.set_teacher_model(teacher_model)
        print("Teacher model set successfully!")
    elif objective_name == "orbit":
        print("\nSetting oracle model in ORBIT objective...")
        trainer.set_teacher_model(oracle_model)
        print("Oracle model set successfully!")

    if hasattr(trainer.objective, "set_projection_basis"):
        basis_teacher = teacher_model
        if basis_teacher is None:
            raise ValueError(
                f"Objective {objective_name} requires a frozen teacher adapter to compute the feature subspace."
            )
        basis_rank = int(unlearning_params.get("basis_rank", 8))
        basis_max_samples = unlearning_params.get("basis_max_samples")
        normalize_features = bool(objective_kwargs.get("normalize_features", True))
        print(
            f"\nComputing feature subspace from teacher forget features "
            f"(rank={basis_rank}, max_samples={basis_max_samples})..."
        )
        basis, center = compute_feature_subspace(
            model=basis_teacher,
            dataset=forget_train,
            device=device,
            batch_size=val_batch_size,
            num_workers=default_workers,
            pin_memory=pin_memory,
            max_samples=basis_max_samples,
            rank=basis_rank,
            normalize_features=normalize_features,
        )
        trainer.objective.set_projection_basis(basis.to(device), center.to(device))
        print(f"Feature subspace ready: basis_shape={tuple(basis.shape)}")

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
    # Train
    print(f"\nStarting unlearning training for {max_epochs} epochs...")
    if resume_checkpoint_dir:
        trainer.load_checkpoint(resume_checkpoint_dir)
        print(f"Loaded optimizer/scheduler state from: {resume_checkpoint_dir}")

    try:
        training_history = trainer.train(
            max_epochs=max_epochs,
            early_stopping_patience=unlearning_config["early_stopping"].get("patience", 3),
            save_every=10,
            checkpoint_dir=checkpoint_dir,
            monitor_metric="forget_acc",
            mode="min"
        )
    except KeyboardInterrupt:
        interrupted_path = os.path.join(checkpoint_dir, "interrupted_model")
        print("\nTraining interrupted. Saving current checkpoint...")
        trainer.save_checkpoint(interrupted_path)
        history_path = f"results/logs/{args.suite}_seed_{args.seed}_history.json"
        trainer.save_training_history(history_path)
        print(f"Interrupted checkpoint saved to: {interrupted_path}")
        print(f"Training history saved to: {history_path}")
        raise

    # Save training history
    history_path = f"results/logs/{args.suite}_seed_{args.seed}_history.json"
    trainer.save_training_history(history_path)

    if unlearning_method == "lora":
        # Save final LoRA adapter
        adapter_save_path = f"checkpoints/unlearn_lora/{args.suite}_seed_{args.seed}"
        save_lora_adapter(lora_model, adapter_save_path)

        if scrub_rewind:
            rewound_model_path = os.path.join(checkpoint_dir, "rewound_model")
            trainer.save_checkpoint(rewound_model_path)
            print(f"Saved SCRUB rewound checkpoint to: {rewound_model_path}")

        # For SCRUB+R, the in-memory model has already been rewound and should
        # remain the exported adapter. For other methods, keep exporting the
        # best validation checkpoint when available.
        best_model_path = os.path.join(checkpoint_dir, "best_model")
        if scrub_rewind:
            print("SCRUB rewind enabled: preserving the rewound adapter as the exported checkpoint.")
        elif os.path.exists(best_model_path):
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
