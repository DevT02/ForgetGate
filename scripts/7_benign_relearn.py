#!/usr/bin/env python3
"""
Script 7: Benign relearning probe (retain-only finetune)
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data import DataManager
from src.models.vit import create_vit_model
from src.models.cnn import create_cnn_model
from src.models.peft_lora import load_lora_adapter
from src.utils import set_seed, ensure_dir, load_config, print_model_info, get_device


def train_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = nn.functional.cross_entropy(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()

        total_loss += loss.item() * batch_size
        total_correct += correct
        total_samples += batch_size

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{correct/batch_size:.4f}"
        })

    return {
        "train_loss": total_loss / max(1, total_samples),
        "train_acc": total_correct / max(1, total_samples)
    }


def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = nn.functional.cross_entropy(outputs, labels)
            _, predicted = outputs.max(1)
            correct = predicted.eq(labels).sum().item()
            total_loss += loss.item() * inputs.size(0)
            total_correct += correct
            total_samples += inputs.size(0)

    return {
        "val_loss": total_loss / max(1, total_samples),
        "val_acc": total_correct / max(1, total_samples)
    }


def save_checkpoint_state_dict(state_dict, optimizer, epoch, path):
    ensure_dir(os.path.dirname(path))
    torch.save({
        "epoch": epoch,
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Benign relearning probe (retain-only finetune)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--suite", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    experiment_suites = load_config(args.config)
    if args.suite not in experiment_suites:
        raise ValueError(f"Experiment suite '{args.suite}' not found in config")

    suite_cfg = experiment_suites[args.suite]
    data_config = load_config("configs/data.yaml")
    model_config = load_config("configs/model.yaml")

    device = get_device(args.device)

    print(f"Benign relearning: {args.suite}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")

    dataset_name = suite_cfg.get("dataset", "cifar10")
    dataset_info = data_config[dataset_name]
    model_type = suite_cfg.get("model", "vit_tiny")

    # Build base model
    if model_type.startswith("vit"):
        model_cfg_name = model_type.replace("vit_", "")
        vit_cfg = dict(model_config["vit"][model_cfg_name])
        vit_cfg["pretrained"] = False
        model = create_vit_model(vit_cfg, num_classes=dataset_info["num_classes"]).to(device)
    else:
        model = create_cnn_model(model_config["cnn"][model_type], num_classes=dataset_info["num_classes"]).to(device)

    # Load unlearned adapter on top of base weights
    unlearned_suite = suite_cfg.get("unlearned_model_suite")
    if not unlearned_suite:
        raise ValueError("benign_relearn requires unlearned_model_suite in config")

    base_suite = experiment_suites[unlearned_suite].get("base_model_suite", None)
    if not base_suite:
        raise ValueError("unlearned_model_suite missing base_model_suite")

    base_ckpt = f"checkpoints/base/{base_suite}_seed_{args.seed}_final.pt"
    if not os.path.exists(base_ckpt):
        raise FileNotFoundError(f"Base model checkpoint not found: {base_ckpt}")

    checkpoint = torch.load(base_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    adapter_path = f"checkpoints/unlearn_lora/{unlearned_suite}_seed_{args.seed}"
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Unlearned adapter not found: {adapter_path}")

    model = load_lora_adapter(model, adapter_path).to(device)

    print_model_info(model, "Unlearned model (before benign relearn)")

    # Enable gradients for benign relearning (LoRA adapters may be frozen by default)
    for _, param in model.named_parameters():
        param.requires_grad = True

    # Data: retain-only
    data_manager = DataManager()
    forget_class = experiment_suites[unlearned_suite].get("unlearning", {}).get("forget_class", 0)
    retain_fraction = float(suite_cfg.get("benign_relearn", {}).get("retain_fraction", 1.0))
    batch_size = int(suite_cfg.get("benign_relearn", {}).get("batch_size", 128))
    epochs = int(suite_cfg.get("benign_relearn", {}).get("epochs", 5))
    lr = float(suite_cfg.get("benign_relearn", {}).get("lr", 1e-4))

    retain_dataset = data_manager.load_dataset(
        dataset_name, "train",
        exclude_classes=[forget_class],
        use_pretrained=True,
        apply_imagenet_norm=True
    )

    if retain_fraction < 1.0:
        n = len(retain_dataset)
        keep = max(1, int(n * retain_fraction))
        g = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(n, generator=g).tolist()[:keep]
        retain_dataset = Subset(retain_dataset, indices)

    # Simple train/val split
    n = len(retain_dataset)
    val_size = max(1, int(0.1 * n))
    train_size = n - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        retain_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    history = []
    best_acc = -1.0
    final_path = f"checkpoints/benign_relearn/{args.suite}_seed_{args.seed}_final.pt"

    for epoch in range(epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        val_metrics = validate(model, val_loader, device)
        row = {**train_metrics, **val_metrics, "epoch": epoch}
        history.append(row)
        print(f"Epoch {epoch}: Train Loss {train_metrics['train_loss']:.4f}, "
              f"Train Acc {train_metrics['train_acc']:.4f}, "
              f"Val Loss {val_metrics['val_loss']:.4f}, Val Acc {val_metrics['val_acc']:.4f}")

        if val_metrics["val_acc"] > best_acc:
            best_acc = val_metrics["val_acc"]

    # Save merged base weights for evaluation compatibility.
    to_save = model
    if hasattr(to_save, "merge_and_unload"):
        to_save = to_save.merge_and_unload()
    if hasattr(to_save, "base_model"):
        to_save = to_save.base_model
    if hasattr(to_save, "model"):
        to_save = to_save.model
    save_checkpoint_state_dict(to_save.state_dict(), optimizer, epochs - 1, final_path)

    ensure_dir("results/logs")
    hist_path = f"results/logs/benign_relearn_{args.suite}_seed_{args.seed}_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Saved checkpoints to {final_path}")
    print(f"Saved history to {hist_path}")


if __name__ == "__main__":
    main()
