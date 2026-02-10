#!/usr/bin/env python3
"""
SCRUB-style defense for BalDRO.
Applies SCRUB's alternating push/pull mechanism:
- Forget phase: BalDRO + gradient ASCENT on KL(student||teacher)
- Retain phase: CE + gradient DESCENT on KL(student||teacher)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.data import DataManager
from src.models.vit import create_vit_model
from src.models.peft_lora import apply_lora_to_model, create_lora_config, save_lora_adapter
from src.models.normalize import create_imagenet_normalizer
from src.utils import set_seed, ensure_dir, load_config, get_device


def kl_divergence(student_logits, teacher_logits, temperature=4.0):
    """KL(student || teacher) with temperature scaling."""
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)


def main():
    parser = argparse.ArgumentParser(description="SCRUB-style defense for BalDRO")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)  # Match SCRUB
    parser.add_argument("--lr", type=float, default=1e-3)  # Match SCRUB
    parser.add_argument("--forget-kl-weight", type=float, default=1.0)
    parser.add_argument("--retain-kl-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=2.0)  # Match SCRUB
    parser.add_argument("--forget-class", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    
    data_cfg = load_config("configs/data.yaml")
    model_cfg = load_config("configs/model.yaml")
    
    dataset_name = "cifar10"
    num_classes = data_cfg[dataset_name]["num_classes"]
    forget_class = args.forget_class
    
    # Load data
    data_manager = DataManager()
    train_dataset = data_manager.load_dataset(dataset_name, "train", use_pretrained=True, apply_imagenet_norm=False)
    val_dataset = data_manager.load_dataset(dataset_name, "test", use_pretrained=True, apply_imagenet_norm=False)
    
    # Split into forget and retain subsets
    forget_indices = [i for i, (_, y) in enumerate(train_dataset) if y == forget_class]
    retain_indices = [i for i, (_, y) in enumerate(train_dataset) if y != forget_class]
    
    forget_subset = Subset(train_dataset, forget_indices)
    retain_subset = Subset(train_dataset, retain_indices)
    
    forget_loader = DataLoader(forget_subset, batch_size=64, shuffle=True, num_workers=2)
    retain_loader = DataLoader(retain_subset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)
    normalizer = create_imagenet_normalizer().to(device)
    
    # Build student model
    vit_cfg = dict(model_cfg["vit"]["tiny"])
    vit_cfg["pretrained"] = False
    student = create_vit_model(vit_cfg, num_classes=num_classes).to(device)
    
    # Load base checkpoint
    base_ckpt = torch.load(f"checkpoints/base/base_vit_cifar10_seed_{args.seed}_final.pt", 
                           map_location=device)
    student.load_state_dict(base_ckpt["model_state_dict"])
    
    # Build teacher (frozen copy of base)
    teacher = create_vit_model(vit_cfg, num_classes=num_classes).to(device)
    teacher.load_state_dict(base_ckpt["model_state_dict"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    
    # Apply LoRA to student
    lora_config = create_lora_config(r=8, lora_alpha=16)
    student = apply_lora_to_model(student, lora_config).to(device)
    
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr
    )
    
    print(f"[Defense] SCRUB-style BalDRO Defense")
    print(f"          forget_kl_weight={args.forget_kl_weight}, retain_kl_weight={args.retain_kl_weight}")
    print(f"          temperature={args.temperature}, epochs={args.epochs}")
    
    forget_iter = iter(forget_loader)
    retain_iter = iter(retain_loader)
    
    for epoch in range(args.epochs):
        student.train()
        total_forget_loss = 0
        total_retain_loss = 0
        num_batches = 0
        
        # Alternating batches
        steps_per_epoch = max(len(forget_loader), len(retain_loader))
        
        for step in range(steps_per_epoch):
            # === FORGET PHASE: Gradient ASCENT on KL ===
            try:
                forget_inputs, forget_labels = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forget_loader)
                forget_inputs, forget_labels = next(forget_iter)
            
            forget_inputs = normalizer(forget_inputs.to(device))
            
            optimizer.zero_grad()
            
            student_logits = student(forget_inputs)
            with torch.no_grad():
                teacher_logits = teacher(forget_inputs)
            
            # Gradient ASCENT on KL = minimize -KL
            forget_kl = kl_divergence(student_logits, teacher_logits, args.temperature)
            forget_loss = -args.forget_kl_weight * forget_kl  # Negative for ascent
            
            forget_loss.backward()
            optimizer.step()
            
            total_forget_loss += forget_kl.item()
            
            # === RETAIN PHASE: Gradient DESCENT on KL + CE ===
            try:
                retain_inputs, retain_labels = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                retain_inputs, retain_labels = next(retain_iter)
            
            retain_inputs = normalizer(retain_inputs.to(device))
            retain_labels = retain_labels.to(device)
            
            optimizer.zero_grad()
            
            student_logits = student(retain_inputs)
            with torch.no_grad():
                teacher_logits = teacher(retain_inputs)
            
            # Gradient DESCENT on KL + CE
            retain_kl = kl_divergence(student_logits, teacher_logits, args.temperature)
            retain_ce = F.cross_entropy(student_logits, retain_labels)
            retain_loss = args.retain_kl_weight * retain_kl + retain_ce
            
            retain_loss.backward()
            optimizer.step()
            
            total_retain_loss += retain_loss.item()
            num_batches += 1
        
        if (epoch + 1) % 10 == 0:
            avg_forget = total_forget_loss / num_batches
            avg_retain = total_retain_loss / num_batches
            print(f"  Epoch {epoch+1}/{args.epochs}: forget_kl={avg_forget:.4f}, retain_loss={avg_retain:.4f}")
    
    # Save checkpoint
    output_name = f"unlearn_baldro_scrub_vit_cifar10_forget{forget_class}_seed_{args.seed}"
    save_dir = f"checkpoints/unlearn_lora/{output_name}"
    ensure_dir(save_dir)
    save_lora_adapter(student, save_dir)
    
    print(f"Saved to {save_dir}")
    
    # Validation
    student.eval()
    correct = 0
    total = 0
    forget_correct = 0
    forget_total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = student(normalizer(inputs))
            preds = outputs.argmax(dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            forget_mask = (labels == forget_class)
            forget_correct += (preds[forget_mask] == labels[forget_mask]).sum().item()
            forget_total += forget_mask.sum().item()
    
    print(f"Validation: overall={correct/total:.4f}, forget={forget_correct/forget_total:.4f}")


if __name__ == "__main__":
    main()
