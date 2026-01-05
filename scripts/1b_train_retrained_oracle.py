#!/usr/bin/env python3
"""
Script 1b: Train Retraining Oracle for ForgetGate-V
Trains models from scratch WITHOUT the forget class - the gold standard for unlearning

This creates the "oracle" baseline that represents perfect unlearning:
if we never trained on the forget class at all, what would the model look like?
"""

import argparse
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from src.data import DataManager, create_forget_retain_splits
from src.models.vit import create_vit_model
from src.models.cnn import create_cnn_model
from src.utils import set_seed, ensure_dir, load_config, print_model_info, get_device
from src.utils import log_experiment, create_experiment_log, save_dict_to_json

logger = logging.getLogger(__name__)


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer, criterion,
                device: torch.device, epoch: int) -> float:
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}: "
                  f"Loss={loss.item():.4f}, Acc={100.*correct/total:.2f}%")

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model: nn.Module, val_loader: DataLoader, criterion,
             device: torch.device) -> tuple:
    """Validate model"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc


def main():
    parser = argparse.ArgumentParser(description="Train Retraining Oracle for ForgetGate-V")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to experiment suites config")
    parser.add_argument("--suite", type=str, required=True,
                       help="Experiment suite name (e.g., oracle_vit_cifar10_forget0)")
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

    # Setup device
    device = get_device(args.device)

    print("=" * 50)
    print(f"ForgetGate-V: Training Retraining Oracle - {args.suite}")
    print("=" * 50)
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")

    # Get oracle parameters (prefer top-level forget_class; fall back to oracle block)
    oracle_params = suite_config.get('oracle', {})
    forget_class = suite_config.get('forget_class', oracle_params.get('forget_class', 0))

    print(f"Oracle mode: Train WITHOUT class {forget_class}")
    print("This represents the gold standard for perfect unlearning")

    # Create data manager
    data_manager = DataManager()

    # Get dataset info
    dataset_name = suite_config.get('dataset', 'cifar10')
    dataset_info = data_config[dataset_name]

    print(f"Dataset: {dataset_name}")
    print(f"Number of classes: {dataset_info['num_classes']}")

    # Load full train dataset and filter out forget class
    full_train_dataset = data_manager.load_dataset(dataset_name, "train")

    # Use create_forget_retain_splits to get retain-only data
    _, retain_train, _, retain_val = create_forget_retain_splits(
        full_train_dataset, forget_class, train_ratio=0.8
    )

    print(f"Retain training samples (excluding class {forget_class}): {len(retain_train)}")
    print(f"Retain validation samples: {len(retain_val)}")

    # Create data loaders
    batch_size = suite_config.get('training', {}).get('batch_size', 128)

    train_loader = DataLoader(
        retain_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        retain_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create model
    model_type = suite_config.get('model', 'vit_tiny')
    if 'vit' in model_type:
        model_config_name = model_type.split('_')[1]
        model = create_vit_model(
            model_config['vit'][model_config_name],
            num_classes=dataset_info['num_classes']
        )
    else:
        model = create_cnn_model(
            model_config['cnn'][model_type],
            num_classes=dataset_info['num_classes']
        )

    model = model.to(device)
    print_model_info(model, f"{model_type} (Oracle)")

    # Setup optimizer
    training_config = suite_config.get('training', {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.get('lr', 1e-3),
        weight_decay=training_config.get('weight_decay', 0.01)
    )

    # Setup scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_config.get('epochs', 100)
    )

    # Setup loss
    criterion = nn.CrossEntropyLoss()

    # Training loop
    max_epochs = training_config.get('epochs', 100)
    best_val_acc = 0.0
    log_path = create_experiment_log(f"{args.suite}_seed_{args.seed}", suite_config)

    print("\nStarting oracle training (retain classes only)...")

    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch+1}/{max_epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Log metrics
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc / 100.0,  # Convert to [0,1]
            'val_loss': val_loss,
            'val_acc': val_acc / 100.0
        }
        log_experiment(log_path, epoch_metrics)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = f"checkpoints/oracle/{args.suite}_seed_{args.seed}_best.pt"
            ensure_dir(os.path.dirname(save_path))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_acc': val_acc
            }, save_path)
            print(f"[BEST] New best model saved: {val_acc:.2f}%")

        # Update scheduler
        if scheduler:
            scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # Save final model
    final_save_path = f"checkpoints/oracle/{args.suite}_seed_{args.seed}_final.pt"
    ensure_dir(os.path.dirname(final_save_path))
    torch.save({
        'epoch': max_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }, final_save_path)

    print("\nOracle training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Models saved to: checkpoints/oracle/")
    print("\nThis oracle represents the gold standard for unlearning:")
    print(f"  - Trained on all classes EXCEPT class {forget_class}")
    print(f"  - Forget accuracy should be ~random (10% for CIFAR-10)")
    print(f"  - Retain accuracy should match base model on other classes")
    print("\nNext: Compare with unlearning methods using: python scripts/4_adv_evaluate.py")


if __name__ == "__main__":
    main()
