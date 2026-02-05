#!/usr/bin/env python3
"""
Script 1: Train baseline model
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import yaml
import torchvision
from tqdm import tqdm

from src.data import DataManager
from src.models.vit import create_vit_model
from src.models.cnn import create_cnn_model
from src.attacks.pgd import PGDAttack
from src.utils import set_seed, ensure_dir, load_config, print_model_info, get_device
from src.utils import AverageMeter, ProgressMeter, log_experiment, create_experiment_log
from src.models.normalize import create_imagenet_normalizer


def train_epoch(model, train_loader, optimizer, scheduler, device, epoch):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = nn.functional.cross_entropy(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track stats
        batch_size = inputs.size(0)
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()

        total_loss += loss.item() * batch_size
        total_correct += correct
        total_samples += batch_size

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/batch_size:.4f}"})

    if scheduler:
        scheduler.step()

    return {
        "train_loss": total_loss / total_samples,
        "train_acc": total_correct / total_samples,
    }


def train_epoch_adversarial(model, train_loader, optimizer, scheduler, device, epoch, adv_config):
    """Train one epoch with adversarial training (Madry et al.)."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_adv_correct = 0
    total_samples = 0

    # Create PGD attack for adversarial training
    attack = PGDAttack(
        model=model,
        eps=adv_config.get("eps", 8 / 255),
        alpha=adv_config.get("alpha", 2 / 255),
        steps=adv_config.get("steps", 10),
        random_start=True,
        norm='l_inf'
    )

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} (Adversarial)")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        # Generate adversarial examples
        adv_inputs = attack(inputs, labels)

        # Train on adversarial examples
        outputs = model(adv_inputs)
        loss = nn.functional.cross_entropy(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track stats
        batch_size = inputs.size(0)

        # Adversarial accuracy
        _, predicted = outputs.max(1)
        adv_correct = predicted.eq(labels).sum().item()

        # Clean accuracy (for monitoring)
        with torch.no_grad():
            clean_outputs = model(inputs)
            _, clean_predicted = clean_outputs.max(1)
            clean_correct = clean_predicted.eq(labels).sum().item()

        total_loss += loss.item() * batch_size
        total_correct += clean_correct
        total_adv_correct += adv_correct
        total_samples += batch_size

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "clean_acc": f"{clean_correct/batch_size:.4f}",
                "adv_acc": f"{adv_correct/batch_size:.4f}",
            }
        )

    if scheduler:
        scheduler.step()

    return {
        "train_loss": total_loss / total_samples,
        "train_acc": total_correct / total_samples,  # Clean accuracy
        "train_adv_acc": total_adv_correct / total_samples,  # Adversarial accuracy
    }


def validate(model, val_loader, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = nn.functional.cross_entropy(outputs, labels)

            _, predicted = outputs.max(1)
            correct = predicted.eq(labels).sum().item()

            total_loss += loss.item() * inputs.size(0)
            total_correct += correct
            total_samples += inputs.size(0)

    return {
        "val_loss": total_loss / total_samples,
        "val_acc": total_correct / total_samples,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    save_path: str,
):
    """Save model checkpoint."""
    ensure_dir(os.path.dirname(save_path))

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train baseline model")
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

    # Setup device
    device = get_device(args.device)
    print(f"Run: Training Baseline Model - {args.suite}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")

    # Create data manager
    data_manager = DataManager()

    # Get dataset info
    dataset_name = suite_config.get("dataset", "cifar10")
    dataset_info = data_config[dataset_name]

    # Create data loaders (train/val split from training set to avoid test leakage)
    training_config = suite_config.get("training", {})
    batch_size = training_config.get("batch_size", 128)
    num_workers = training_config.get("num_workers", 4)
    env_workers = os.getenv("FG_NUM_WORKERS")
    if env_workers is not None:
        try:
            num_workers = int(env_workers)
        except ValueError:
            raise ValueError(f"Invalid FG_NUM_WORKERS='{env_workers}', must be int")
    val_ratio = training_config.get("val_ratio", 0.1)

    # Check if adversarial training is enabled
    adv_train_config = training_config.get("adv_train", {})
    use_adv_train = adv_train_config.get("enabled", False)

    full_train_dataset = data_manager.load_dataset(
        dataset_name,
        "train",
        use_pretrained=True,
        apply_imagenet_norm=not use_adv_train
    )

    n_total = len(full_train_dataset)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n_total, generator=generator).tolist()
    train_indices = perm[:n_train]
    val_indices = perm[n_train:]

    train_dataset = Subset(full_train_dataset, train_indices)

    eval_transform = data_manager.get_transforms(
        dataset_name,
        split="test",
        use_pretrained=True,
        apply_imagenet_norm=not use_adv_train
    )
    if dataset_name == "cifar10":
        val_full_dataset = torchvision.datasets.CIFAR10(
            root=data_manager.data_dir, train=True, download=True, transform=eval_transform
        )
    elif dataset_name == "mnist":
        val_full_dataset = torchvision.datasets.MNIST(
            root=data_manager.data_dir, train=True, download=True, transform=eval_transform
        )
    else:
        raise ValueError(f"Unsupported dataset for base training split: {dataset_name}")

    val_dataset = Subset(val_full_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    print(f"Dataset: {dataset_name}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Number of classes: {dataset_info['num_classes']}")

    # Create model
    model_type = suite_config.get("model", "vit_tiny")
    if 'vit' in model_type:
        # Extract model size from name (tiny, small, base, etc.)
        model_config_name = model_type.split('_')[1]
        model_config_key = f"vit_{model_config_name}"
        base_model = create_vit_model(
        model_config["vit"][model_config_name],
        num_classes=dataset_info["num_classes"],
        )
    else:
        # CNN model
        model_config_key = model_type
        base_model = create_cnn_model(
        model_config["cnn"][model_config_key],
        num_classes=dataset_info["num_classes"],
        )

    base_model = base_model.to(device)
    print_model_info(base_model, f"{model_type}")

    # For adversarial training, keep inputs in [0,1] and normalize inside the model
    if use_adv_train:
        norm_layer = create_imagenet_normalizer().to(device)
        model_for_train = nn.Sequential(norm_layer, base_model).to(device)
    else:
        model_for_train = base_model

    # Setup optimizer
    optimizer = optim.AdamW(
        base_model.parameters(),
        lr=training_config.get("lr", 1e-3),
        weight_decay=training_config.get("weight_decay", 0.01),
    )

    # Setup scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_config.get("epochs", 100)
    )

    # Training loop
    max_epochs = training_config.get("epochs", 100)
    best_val_acc = 0.0
    log_path = create_experiment_log(f"{args.suite}_seed_{args.seed}", suite_config)

    if use_adv_train:
        print("\n[Adversarial Training ENABLED]")
        print(
            f"   eps={adv_train_config.get('eps', 8 / 255):.6f}, "
            f"alpha={adv_train_config.get('alpha', 2 / 255):.6f}, "
            f"steps={adv_train_config.get('steps', 10)}"
        )
    else:
        print("\nStandard Training")

    print("\nStarting training...")

    for epoch in range(max_epochs):
        # Train (adversarial or standard)
        if use_adv_train:
            train_metrics = train_epoch_adversarial(
                model_for_train, train_loader, optimizer, scheduler, device, epoch, adv_train_config
            )
        else:
            train_metrics = train_epoch(
                model_for_train, train_loader, optimizer, scheduler, device, epoch
            )

        # Validate
        val_metrics = validate(model_for_train, val_loader, device)

        # Combine metrics
        epoch_metrics = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
        }

        # Log metrics
        log_experiment(log_path, epoch_metrics)

        # Save best model
        if val_metrics['val_acc'] > best_val_acc:
            best_val_acc = val_metrics['val_acc']
            save_path = f"checkpoints/base/{args.suite}_seed_{args.seed}_best.pt"
            save_checkpoint(base_model, optimizer, scheduler, epoch, save_path)

        # Print progress
        if use_adv_train and 'train_adv_acc' in train_metrics:
            print(f"Epoch {epoch}: Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Clean Acc: {train_metrics['train_acc']:.4f}, "
                  f"Adv Acc: {train_metrics['train_adv_acc']:.4f}, "
                  f"Val Acc: {val_metrics['val_acc']:.4f}")
        else:
            print(f"Epoch {epoch}: Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Train Acc: {train_metrics['train_acc']:.4f}, "
                  f"Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Val Acc: {val_metrics['val_acc']:.4f}")

    # Save final model
    final_save_path = f"checkpoints/base/{args.suite}_seed_{args.seed}_final.pt"
    save_checkpoint(base_model, optimizer, scheduler, max_epochs-1, final_save_path)

    print("\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Models saved to: checkpoints/base/")
    print("\nNext: Run unlearning with: python scripts/2_train_unlearning_lora.py")


if __name__ == "__main__":
    main()
