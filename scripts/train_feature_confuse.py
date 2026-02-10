#!/usr/bin/env python3
"""
Feature Confusion Defense: Make forget features look like retain features.
Instead of compressing, we SHIFT forget features toward retain distribution.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import DataManager
from src.models.vit import create_vit_model
from src.models.peft_lora import apply_lora_to_model, create_lora_config, save_lora_adapter
from src.models.normalize import create_imagenet_normalizer
from src.utils import set_seed, ensure_dir, load_config, get_device


def get_feature_model(model):
    """Get model for feature extraction, handles PEFT-wrapped models."""
    for name, module in model.named_modules():
        if hasattr(module, 'forward_features'):
            return module
    return None


def main():
    parser = argparse.ArgumentParser(description="Train with feature confusion defense")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--confuse-weight", type=float, default=0.5)
    parser.add_argument("--retain-weight", type=float, default=1.0)
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
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)
    normalizer = create_imagenet_normalizer().to(device)
    
    # Build model
    vit_cfg = dict(model_cfg["vit"]["tiny"])
    vit_cfg["pretrained"] = False
    student = create_vit_model(vit_cfg, num_classes=num_classes).to(device)
    
    # Load base checkpoint
    base_ckpt = torch.load(f"checkpoints/base/base_vit_cifar10_seed_{args.seed}_final.pt", 
                           map_location=device)
    student.load_state_dict(base_ckpt["model_state_dict"])
    
    # Apply LoRA
    lora_config = create_lora_config(r=8, lora_alpha=16)
    student = apply_lora_to_model(student, lora_config).to(device)
    
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr
    )
    
    print(f"[Defense] Feature Confusion Defense")
    print(f"          confuse_weight={args.confuse_weight}, retain_weight={args.retain_weight}")
    print(f"          epochs={args.epochs}, seed={args.seed}")
    
    for epoch in range(args.epochs):
        student.train()
        total_loss = 0
        num_batches = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            norm_inputs = normalizer(inputs)
            
            forget_mask = (labels == forget_class)
            retain_mask = ~forget_mask
            
            if not forget_mask.any() or not retain_mask.any():
                continue
            
            # Get features
            feature_model = get_feature_model(student)
            features = feature_model.forward_features(norm_inputs)
            if features.dim() == 3:
                features = features[:, 0, :]
            
            outputs = student(norm_inputs)
            
            loss = torch.tensor(0.0, device=device)
            
            # 1. Retain CE loss
            retain_ce = F.cross_entropy(outputs[retain_mask], labels[retain_mask])
            loss = loss + args.retain_weight * retain_ce
            
            # 2. Feature confusion: push forget features toward retain CENTROID
            # This makes forget features look like "average" retain features
            forget_feats = features[forget_mask]
            retain_centroid = features[retain_mask].mean(dim=0, keepdim=True).detach()
            
            # MSE distance to retain centroid (minimize)
            confuse_loss = ((forget_feats - retain_centroid) ** 2).mean()
            loss = loss + args.confuse_weight * confuse_loss
            
            # 3. Also match variance - forget should have similar spread to retain
            # Only if we have enough samples
            if forget_feats.size(0) > 1 and features[retain_mask].size(0) > 1:
                forget_var = forget_feats.var(dim=0).mean()
                retain_var = features[retain_mask].var(dim=0).mean().detach()
                if not (torch.isnan(forget_var) or torch.isnan(retain_var)):
                    var_match_loss = (forget_var - retain_var) ** 2
                    loss = loss + 0.1 * var_match_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}")
    
    # Save checkpoint
    output_name = f"unlearn_confuse_vit_cifar10_forget{forget_class}_seed_{args.seed}"
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
