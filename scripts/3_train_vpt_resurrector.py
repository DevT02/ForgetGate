#!/usr/bin/env python3
"""
Script 3: Train VPT resurrector against unlearned model for ForgetGate-V
"""

import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, ConcatDataset
from typing import Dict
import yaml

from src.data import DataManager
from src.models.vit import create_vit_model, add_vpt_to_model
from src.models.cnn import create_cnn_model, add_vpt_to_cnn_model
from src.models.peft_lora import load_lora_adapter
from src.attacks.vpt_resurrection import VPTResurrectionAttack
from src.utils import set_seed, ensure_dir, load_config, print_model_info, get_device
from src.utils import log_experiment, create_experiment_log


def load_oracle_model(experiment_suites: Dict, suite_config: Dict, device: torch.device, seed: int) -> nn.Module:
    """Load oracle model (trained without forget class)"""
    oracle_suite_name = suite_config.get('oracle_model_suite', '')
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

    if model_type.startswith('vit'):
        model_config_name = model_type.replace('vit_', '')
        oracle_model = create_vit_model(
            model_config['vit'][model_config_name],
            num_classes=dataset_info['num_classes']
        )
    else:
        oracle_model = create_cnn_model(
            model_config['cnn'][model_type],
            num_classes=dataset_info['num_classes']
        )

    oracle_model.load_state_dict(checkpoint['model_state_dict'])
    oracle_model = oracle_model.to(device)

    return oracle_model


def load_unlearned_model(experiment_suites: Dict, suite_config: Dict, device: torch.device, seed: int) -> nn.Module:
    """Load unlearned model with LoRA adapter"""
    unlearned_suite_name = suite_config.get('unlearned_model_suite', '')
    if not unlearned_suite_name:
        raise ValueError("No unlearned_model_suite specified in VPT suite")

    # Resolve the suite chain: VPT -> unlearned -> base
    unlearned_suite_config = experiment_suites.get(unlearned_suite_name, {})
    base_suite_name = unlearned_suite_config.get('base_model_suite', '')
    if not base_suite_name:
        raise ValueError("No base_model_suite found in unlearned suite")

    base_checkpoint = f"checkpoints/base/{base_suite_name}_seed_{seed}_final.pt"

    if not os.path.exists(base_checkpoint):
        raise FileNotFoundError(f"Base model checkpoint not found: {base_checkpoint}")

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

    if model_type.startswith('vit'):
        model_config_name = model_type.replace('vit_', '')
        base_model = create_vit_model(
            model_config['vit'][model_config_name],
            num_classes=dataset_info['num_classes']
        )
    else:
        base_model = create_cnn_model(
            model_config['cnn'][model_type],
            num_classes=dataset_info['num_classes']
        )

    base_model.load_state_dict(checkpoint['model_state_dict'])
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


def main():
    parser = argparse.ArgumentParser(description="Train VPT resurrector for ForgetGate-V")
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
    vpt_config_file = load_config("configs/vpt_attack.yaml")

    # Setup device
    device = get_device(args.device)

    print("=" * 50)
    print(f"ForgetGate-V: Training VPT Resurrector - {args.suite}")
    print("=" * 50)
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")

    # Get VPT parameters
    vpt_params = suite_config.get('vpt_attack', {})
    forget_class = vpt_params.get('target_class', 0)
    prompt_length = vpt_params.get('prompt_length', 5)
    init_strategy = vpt_params.get('init_strategy', 'random')

    print(f"Target class: {forget_class}")
    print(f"Prompt length: {prompt_length}")
    print(f"Init strategy: {init_strategy}")

    # Create VPT configuration
    vpt_config = {
        'prompt_type': vpt_params.get('prompt_type', 'prefix'),
        'prompt_length': prompt_length,
        'init_strategy': init_strategy,
        'dropout': vpt_config_file['vpt_prompt'].get('dropout', 0.1)
    }

    # Determine if this is an oracle or unlearned suite
    is_oracle = 'oracle_model_suite' in suite_config
    is_unlearned = 'unlearned_model_suite' in suite_config

    if is_oracle:
        target_model = load_oracle_model(experiment_suites, suite_config, device, args.seed)
        print_model_info(target_model, "Oracle model")
        # Resolve dataset from oracle suite
        oracle_suite_name = suite_config.get('oracle_model_suite', '')
        oracle_suite_config = experiment_suites.get(oracle_suite_name, {})
        dataset_name = oracle_suite_config.get('dataset', 'cifar10')
    elif is_unlearned:
        target_model = load_unlearned_model(experiment_suites, suite_config, device, args.seed)
        print_model_info(target_model, "Unlearned model")
        # Resolve dataset from suite chain (VPT -> unlearned -> base)
        unlearned_suite_name = suite_config.get('unlearned_model_suite', '')
        unlearned_suite_config = experiment_suites.get(unlearned_suite_name, {})
        base_suite_name = unlearned_suite_config.get('base_model_suite', '')
        if base_suite_name:
            base_suite_config = experiment_suites.get(base_suite_name, {})
            dataset_name = base_suite_config.get('dataset', 'cifar10')
        else:
            dataset_name = suite_config.get('dataset', 'cifar10')
    else:
        raise ValueError("Suite must specify either 'oracle_model_suite' or 'unlearned_model_suite'")

    # Create data loader for forget class samples
    data_manager = DataManager()

    # Load train dataset and split into forget/retain for training + validation
    train_dataset = data_manager.load_dataset(dataset_name, "train")
    from src.data import create_forget_retain_splits
    forget_train, retain_train, forget_val, retain_val = create_forget_retain_splits(
        train_dataset, forget_class, train_ratio=0.8
    )

    # K-shot sampling - limit training data if specified
    k_shot = vpt_params.get('k_shot', None)
    if k_shot is not None:
        print(f"\nK-shot mode: Using only {k_shot} samples from forget class")
        # Use torch.utils.data.Subset to limit samples
        from torch.utils.data import Subset
        import random

        # Create deterministic k-shot subset
        random.seed(args.seed)
        indices = list(range(len(forget_train)))
        random.shuffle(indices)
        k_shot_indices = indices[:k_shot]
        forget_train = Subset(forget_train, k_shot_indices)
        print(f"Forget train samples: {len(forget_train)} (original: {len(indices)})")

    # Explicit loaders (preferred): avoids CPU caching/subsampling inside the attack
    forget_train_loader = DataLoader(
        forget_train,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    retain_train_loader = DataLoader(
        retain_train,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Use held-out TRAIN split for validation (no test set leakage)
    forget_val_loader = DataLoader(
        forget_val,
        batch_size=256,  # Large batch for evaluation
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create VPT resurrection attack
    vpt_attack = VPTResurrectionAttack(
        target_model=target_model,
        forget_class=forget_class,
        prompt_config=vpt_config,
        device=device
    )

    # Train resurrection prompt
    training_params = vpt_params
    epochs = training_params.get('epochs', 100)
    lr = training_params.get('lr', 1e-2)

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
        weight_decay=training_params.get('weight_decay', 0.0),
        patience=vpt_config_file['vpt_training']['early_stopping'].get('patience', 20),
        eval_every=training_params.get('eval_every', 10),
        lambda_retain=training_params.get('lambda_retain', 1.0),
        T=training_params.get('T', 1.0)
    )

    # Save trained VPT prompt
    prompt_save_path = f"checkpoints/vpt_resurrector/{args.suite}_seed_{args.seed}"
    vpt_attack.save_attack_prompt(prompt_save_path)

    # Log results
    log_path = create_experiment_log(f"{args.suite}_seed_{args.seed}", suite_config)

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

    print("\nNext: Run adversarial evaluation with: python scripts/4_adv_evaluate.py")


if __name__ == "__main__":
    main()
