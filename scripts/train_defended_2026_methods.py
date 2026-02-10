#!/usr/bin/env python3
"""
Train 2026 unlearning methods WITH distillation defense.
Then probe them to verify the defense closes the feature leakage gap.
"""

import argparse
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.data import DataManager
from src.models.vit import create_vit_model
from src.models.peft_lora import apply_lora_to_model, create_lora_config, save_lora_adapter
from src.models.normalize import create_imagenet_normalizer
from src.unlearning.objectives import (
    BalDROUnlearning, 
    FaLWUnlearning,
    wrap_with_distillation_defense
)
from src.unlearning.trainer import UnlearningTrainer
from src.utils import set_seed, ensure_dir, load_config, get_device


def main():
    parser = argparse.ArgumentParser(description="Train 2026 methods with distillation defense")
    parser.add_argument("--method", type=str, default="baldro", choices=["baldro", "falw"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--forget-class", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    
    # Load configs
    data_cfg = load_config("configs/data.yaml")
    model_cfg = load_config("configs/model.yaml")
    
    dataset_name = "cifar10"
    dataset_info = data_cfg[dataset_name]
    num_classes = dataset_info["num_classes"]
    forget_class = args.forget_class
    
    # Load data
    data_manager = DataManager()
    train_dataset = data_manager.load_dataset(dataset_name, "train", use_pretrained=True, apply_imagenet_norm=False)
    val_dataset = data_manager.load_dataset(dataset_name, "test", use_pretrained=True, apply_imagenet_norm=False)
    
    # Split into forget and retain
    forget_indices = [i for i, (_, y) in enumerate(train_dataset) if y == forget_class]
    retain_indices = [i for i, (_, y) in enumerate(train_dataset) if y != forget_class]
    
    forget_subset = torch.utils.data.Subset(train_dataset, forget_indices)
    retain_subset = torch.utils.data.Subset(train_dataset, retain_indices)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    # Build student model (from base checkpoint)
    vit_cfg = dict(model_cfg["vit"]["tiny"])
    vit_cfg["pretrained"] = False
    student = create_vit_model(vit_cfg, num_classes=num_classes).to(device)
    
    # Load base checkpoint
    base_ckpt_path = f"checkpoints/base/base_vit_cifar10_seed_{args.seed}_final.pt"
    base_ckpt = torch.load(base_ckpt_path, map_location=device)
    student.load_state_dict(base_ckpt["model_state_dict"])
    
    # Build teacher model (frozen copy of base)
    teacher = create_vit_model(vit_cfg, num_classes=num_classes).to(device)
    teacher.load_state_dict(base_ckpt["model_state_dict"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    
    # Apply LoRA to student
    lora_config = create_lora_config(r=8, lora_alpha=16)
    student = apply_lora_to_model(student, lora_config).to(device)
    
    # Create base objective
    if args.method == "baldro":
        base_objective = BalDROUnlearning(forget_class, num_classes)
    else:
        base_objective = FaLWUnlearning(forget_class, num_classes)
    
    # Wrap with distillation defense
    defended_objective = wrap_with_distillation_defense(
        base_objective, 
        distill_weight=args.distill_weight,
        temperature=args.temperature
    )
    defended_objective.set_teacher(teacher)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr
    )
    
    # Training loop
    normalizer = create_imagenet_normalizer().to(device)
    
    print(f"[Defense] Training {args.method.upper()} with distillation defense")
    print(f"          distill_weight={args.distill_weight}, temperature={args.temperature}")
    print(f"          epochs={args.epochs}, seed={args.seed}")
    
    for epoch in range(args.epochs):
        student.train()
        total_loss = 0
        num_batches = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = normalizer(inputs)
            
            outputs = student(inputs)
            loss = defended_objective(outputs, labels, inputs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}")
    
    # Save checkpoint
    output_name = f"unlearn_{args.method}_defended_vit_cifar10_forget{forget_class}_seed_{args.seed}"
    save_dir = f"checkpoints/unlearn_lora/{output_name}"
    ensure_dir(save_dir)
    
    # Save LoRA adapter using PEFT format
    save_lora_adapter(student, save_dir)
    
    print(f"Saved defended model to {save_dir}")
    
    # Quick validation
    student.eval()
    correct = 0
    total = 0
    forget_correct = 0
    forget_total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = normalizer(inputs)
            outputs = student(inputs)
            preds = outputs.argmax(dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            forget_mask = (labels == forget_class)
            forget_correct += (preds[forget_mask] == labels[forget_mask]).sum().item()
            forget_total += forget_mask.sum().item()
    
    overall_acc = correct / total
    forget_acc = forget_correct / forget_total if forget_total > 0 else 0
    
    print(f"Validation: overall_acc={overall_acc:.4f}, forget_acc={forget_acc:.4f}")


if __name__ == "__main__":
    main()
