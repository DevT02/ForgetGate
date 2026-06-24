#!/usr/bin/env python3
"""
Script 2: Train LoRA unlearning adapter for ForgetGate-V
"""

import argparse
import os
import sys
import shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict
import yaml

from src.data import DataManager
from src.models.vit import create_vit_model
from src.models.cnn import create_cnn_model
from src.models.peft_lora import create_lora_config, apply_lora_to_model, save_lora_adapter
from src.unlearning.trainer import create_unlearning_trainer
from src.utils import set_seed, ensure_dir, load_config, print_model_info, get_device
from src.utils import log_experiment, create_experiment_log


def load_base_model(experiment_suites: Dict, suite_config: Dict, device: torch.device, seed: int) -> nn.Module:
    """Load pretrained base model"""
    base_suite = suite_config.get('base_model_suite', '')

    checkpoint_path = f"checkpoints/base/{base_suite}_seed_{seed}_final.pt"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Base model checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Recreate model architecture using base suite config
    data_config = load_config("configs/data.yaml")
    model_config = load_config("configs/model.yaml")

    # Get base suite config to read model/dataset info
    base_suite_config = experiment_suites.get(base_suite, {})
    dataset_name = base_suite_config.get('dataset', 'cifar10')
    dataset_info = data_config[dataset_name]
    model_type = base_suite_config.get('model', 'vit_tiny')

    if model_type.startswith('vit'):
        model_config_name = model_type.replace('vit_', '')
        model = create_vit_model(
            model_config['vit'][model_config_name],
            num_classes=dataset_info['num_classes']
        )
    else:
        model = create_cnn_model(
            model_config['cnn'][model_type],
            num_classes=dataset_info['num_classes']
        )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"Loaded base model from: {checkpoint_path}")
    return model


def load_oracle_model(experiment_suites: Dict, suite_config: Dict, device: torch.device,
                      seed: int, forget_class: int) -> nn.Module:
    """Load the frozen retain-only ORACLE for ORBIT (same architecture as the base).

    Oracle checkpoints follow checkpoints/oracle/oracle_<arch_dataset>_forget<c>_seed_<s>_final.pt;
    the suite name is derived from the base suite (base_X -> oracle_X_forget<c>).
    """
    base_suite = suite_config.get('base_model_suite', '')
    oracle_suite = base_suite.replace('base_', 'oracle_', 1) + f"_forget{forget_class}"
    checkpoint_path = f"checkpoints/oracle/{oracle_suite}_seed_{seed}_final.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"ORBIT oracle checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    data_config = load_config("configs/data.yaml")
    model_config = load_config("configs/model.yaml")
    base_suite_config = experiment_suites.get(base_suite, {})
    dataset_info = data_config[base_suite_config.get('dataset', 'cifar10')]
    model_type = base_suite_config.get('model', 'vit_tiny')

    if model_type.startswith('vit'):
        model = create_vit_model(model_config['vit'][model_type.replace('vit_', '')],
                                 num_classes=dataset_info['num_classes'])
    else:
        model = create_cnn_model(model_config['cnn'][model_type],
                                 num_classes=dataset_info['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"Loaded ORBIT oracle from: {checkpoint_path}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train LoRA unlearning adapter for ForgetGate-V")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to experiment suites config")
    parser.add_argument("--suite", type=str, required=True,
                       help="Experiment suite name")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use")

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

    print("=" * 50)
    print(f"ForgetGate-V: Training LoRA Unlearning - {args.suite}")
    print("=" * 50)
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")

    # Get unlearning parameters
    unlearning_params = suite_config.get('unlearning', {})
    forget_class = unlearning_params.get('forget_class', 0)
    objective_name = unlearning_params.get('objective', 'ce_ascent')
    lora_rank = unlearning_params.get('lora_rank', 8)

    # Get objective-specific config from unlearning.yaml
    objective_config = unlearning_config.get('objectives', {}).get(objective_name, {})

    # Extract all parameters except 'name' and 'loss_type' (metadata only)
    objective_kwargs = {
        k: v for k, v in objective_config.items()
        if k not in ['name', 'loss_type']
    }

    print(f"Forget class: {forget_class}")
    print(f"Unlearning objective: {objective_name}")
    print(f"LoRA rank: {lora_rank}")

    # Load base model
    base_model = load_base_model(experiment_suites, suite_config, device, args.seed)

    # For SCRUB, we need a separate teacher model (frozen copy of base)
    # Load it BEFORE applying LoRA to avoid reference issues
    teacher_model = None
    if objective_name == "scrub":
        print("\nLoading separate teacher model for SCRUB...")
        teacher_model = load_base_model(experiment_suites, suite_config, device, args.seed)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        print(f"Teacher model loaded and frozen: {type(teacher_model).__name__}")
    elif objective_name == "orbit":
        print("\nLoading frozen retain-only ORACLE as ORBIT teacher...")
        teacher_model = load_oracle_model(experiment_suites, suite_config, device,
                                          args.seed, forget_class)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        print(f"ORBIT oracle loaded and frozen: {type(teacher_model).__name__}")

    # Get model type from base suite for LoRA target selection
    base_suite = suite_config.get('base_model_suite', '')
    base_suite_config = experiment_suites.get(base_suite, {})
    model_type = base_suite_config.get('model', 'vit_tiny')

    # Apply LoRA - select target modules based on model type
    if model_type.startswith('vit'):
        target_modules = model_config['lora']['vit_target_modules']
    else:
        target_modules = model_config['lora']['cnn_target_modules']

    lora_config = create_lora_config(
        r=lora_rank,
        lora_alpha=unlearning_config['lora_unlearn'].get('lora_alpha', 16),
        target_modules=target_modules
    )
    lora_model = apply_lora_to_model(base_model, lora_config)
    print_model_info(lora_model, f"LoRA-{lora_rank} {objective_name}")

    # Create data manager and loaders
    data_manager = DataManager()
    base_suite_config = experiment_suites.get(suite_config.get('base_model_suite', ''), {})
    dataset_name = base_suite_config.get('dataset', 'cifar10')

    # Create forget/retain split
    full_train_dataset = data_manager.load_dataset(dataset_name, "train")
    from src.data import create_forget_retain_splits

    forget_train, retain_train, forget_test, retain_test = create_forget_retain_splits(
        full_train_dataset, forget_class, train_ratio=0.8
    )

    # Create data loaders. num_workers is env-configurable: objectives that run a
    # nested model forward inside the loss (e.g. RURK) can deadlock with Windows
    # DataLoader worker processes, so set FG_NUM_WORKERS=0 for those.
    _nw = int(os.environ.get("FG_NUM_WORKERS", "4"))
    # batch_size is suite-overridable so heavier architectures (e.g. vit_small)
    # can use a smaller batch that fits the GPU.
    _bs = unlearning_params.get(
        'batch_size', unlearning_config['lora_unlearn'].get('batch_size', 128))
    forget_loader = DataLoader(
        forget_train,
        batch_size=_bs,
        shuffle=True,
        num_workers=_nw,
        pin_memory=True
    )

    retain_loader = DataLoader(
        retain_train,
        batch_size=_bs,
        shuffle=True,
        num_workers=_nw,
        pin_memory=True
    )

    print(f"Forget class {forget_class} samples: {len(forget_train)}")
    print(f"Retain samples: {len(retain_train)}")

    # Create combined validation dataset (forget_test + retain_test for proper validation metrics)
    from torch.utils.data import ConcatDataset
    val_dataset = ConcatDataset([forget_test, retain_test])
    val_loader = DataLoader(
        val_dataset,
        batch_size=_bs,
        shuffle=False,  # No shuffle for validation
        num_workers=_nw,
        pin_memory=True
    )

    # Create combined loader for training (alternates forget/retain batches)
    from src.data import CombinedDataLoader
    combined_train_loader = CombinedDataLoader(
        forget_loader,
        retain_loader,
        steps_per_epoch=len(retain_loader)  # full pass over retain each epoch
    )

    # Create unlearning trainer. retain_lambda is suite-overridable so it can be
    # used as a controlled robustness knob (forget/retain margin tradeoff).
    retain_lambda = unlearning_params.get(
        'retain_lambda',
        unlearning_config['lora_unlearn'].get('retain_lambda', 1.0))
    trainer = create_unlearning_trainer(
        model=lora_model,
        objective_name=objective_name,
        forget_class=forget_class,
        train_loader=combined_train_loader,  # Use combined loader
        val_loader=val_loader,  # Use combined validation set
        num_classes=data_config[dataset_name]['num_classes'],
        device=device,
        retain_lambda=retain_lambda,
        objective_kwargs=objective_kwargs
    )

    # Set teacher model for distillation-based objectives (SCRUB: frozen base;
    # ORBIT: frozen retain-only oracle).
    if objective_name in ("scrub", "orbit") and teacher_model is not None:
        print(f"\nSetting teacher model in {objective_name} objective...")
        trainer.set_teacher_model(teacher_model)
        print(f"Teacher model set successfully!")

    # Configure optimizer and scheduler. weight_decay is suite-overridable so a
    # controlled robustness-knob sweep can vary global smoothness while holding
    # the unlearning method fixed (falls back to the global default).
    wd = unlearning_params.get(
        'weight_decay',
        unlearning_config['lora_unlearn'].get('weight_decay', 0.01))
    print(f"Weight decay: {wd}")
    trainer.configure_optimizer(
        lr=unlearning_params.get('lr', 1e-3),
        weight_decay=wd
    )

    # Training parameters
    max_epochs = unlearning_params.get('epochs', 50)

    trainer.configure_scheduler(
        scheduler_name="cosine",
        max_epochs=max_epochs,
        min_lr=unlearning_config['lora_unlearn'].get('min_lr', 1e-6)
    )
    checkpoint_dir = f"checkpoints/unlearn_lora/{args.suite}_seed_{args.seed}"

    # Train
    print(f"\nStarting unlearning training for {max_epochs} epochs...")
    training_history = trainer.train(
        max_epochs=max_epochs,
        early_stopping_patience=unlearning_config['early_stopping'].get('patience', 3),
        save_every=10,
        checkpoint_dir=checkpoint_dir,
        monitor_metric="forget_acc",
        mode="min"
    )

    # Save training history
    history_path = f"results/logs/{args.suite}_seed_{args.seed}_history.json"
    trainer.save_training_history(history_path)

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
