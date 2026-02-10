#!/usr/bin/env python3
"""
Correct SCRUB-style training with FULL EPOCH schedule.
Key differences from our previous attempt:
1. Full epoch of max (forget) THEN full epoch of min (retain) - not per-batch alternating
2. Max phase stops after scrub_max_epochs, then only min phase continues
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


def kl_divergence(student_logits, teacher_logits, temperature=2.0):
    """KL(student || teacher) with temperature scaling."""
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)


def main():
    parser = argparse.ArgumentParser(description="Correct SCRUB-style defense with full epoch schedule")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--total-epochs", type=int, default=50)
    parser.add_argument("--max-epochs", type=int, default=25)  # Stop max phase after this
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=2.0)
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
    
    print(f"[Defense] SCRUB Full Epoch Schedule")
    print(f"          total_epochs={args.total_epochs}, max_epochs={args.max_epochs}")
    print(f"          temperature={args.temperature}")
    print(f"          Phase 1 (epochs 0-{args.max_epochs-1}): MAX on forget + MIN on retain")
    print(f"          Phase 2 (epochs {args.max_epochs}-{args.total_epochs-1}): MIN on retain only")
    
    for epoch in range(args.total_epochs):
        student.train()
        
        # === PHASE 1: MAX epoch on forget (only if epoch < max_epochs) ===
        if epoch < args.max_epochs:
            max_loss = 0.0
            max_batches = 0
            for inputs, labels in forget_loader:
                inputs = normalizer(inputs.to(device))
                
                optimizer.zero_grad()
                student_logits = student(inputs)
                with torch.no_grad():
                    teacher_logits = teacher(inputs)
                
                # Gradient ASCENT on KL = minimize -KL
                kl = kl_divergence(student_logits, teacher_logits, args.temperature)
                loss = -kl  # Negative for ascent
                
                loss.backward()
                optimizer.step()
                
                max_loss += kl.item()
                max_batches += 1
            
            avg_max = max_loss / max(1, max_batches)
        else:
            avg_max = 0.0
        
        # === PHASE 2: MIN epoch on retain (always) ===
        min_loss = 0.0
        min_batches = 0
        for inputs, labels in retain_loader:
            inputs = normalizer(inputs.to(device))
            labels = labels.to(device)
            
            optimizer.zero_grad()
            student_logits = student(inputs)
            with torch.no_grad():
                teacher_logits = teacher(inputs)
            
            # Gradient DESCENT on KL + CE
            kl = kl_divergence(student_logits, teacher_logits, args.temperature)
            ce = F.cross_entropy(student_logits, labels)
            loss = kl + ce  # Minimize both
            
            loss.backward()
            optimizer.step()
            
            min_loss += loss.item()
            min_batches += 1
        
        avg_min = min_loss / max(1, min_batches)
        
        if (epoch + 1) % 10 == 0:
            phase = "MAX+MIN" if epoch < args.max_epochs else "MIN only"
            print(f"  Epoch {epoch+1}/{args.total_epochs} [{phase}]: max_kl={avg_max:.2f}, min_loss={avg_min:.4f}")
    
    # Save checkpoint
    output_name = f"unlearn_scrub_full_vit_cifar10_forget{forget_class}_seed_{args.seed}"
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
