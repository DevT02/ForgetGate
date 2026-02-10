#!/usr/bin/env python3
"""
Train with VARIANCE COMPRESSION defense.
Inspired by SCRUB analysis: SCRUB compresses forget-class variance (13.9 vs base 22.5).
This defense minimizes forget-class feature variance to make probing harder.
"""

import argparse
import os
import sys
from datetime import datetime

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
    # Try various possible locations
    candidates = [
        model,
        getattr(model, 'module', None),
        getattr(model, 'base_model', None),
        getattr(model, 'model', None),
    ]
    
    # For PEFT models, the structure is model.base_model.model
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        candidates.append(model.base_model.model)
    
    for cand in candidates:
        if cand is not None and hasattr(cand, 'forward_features'):
            return cand
    
    # Last resort: try to find any module with forward_features
    for name, module in model.named_modules():
        if hasattr(module, 'forward_features'):
            return module
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Train with variance compression defense")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--compress-weight", type=float, default=0.1)
    parser.add_argument("--retain-weight", type=float, default=1.0)
    parser.add_argument("--forget-class", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    
    # Load configs
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
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr
    )
    
    print(f"[Defense] Variance Compression Defense")
    print(f"          compress_weight={args.compress_weight}, retain_weight={args.retain_weight}")
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
            
            # Get features
            feature_model = get_feature_model(student)
            features = feature_model.forward_features(norm_inputs)
            if features.dim() == 3:
                features = features[:, 0, :]
            
            # Get logits
            outputs = student(norm_inputs)
            
            loss = torch.tensor(0.0, device=device)
            
            # 1. Retain CE loss - keep retain accuracy high
            if retain_mask.any():
                retain_ce = F.cross_entropy(outputs[retain_mask], labels[retain_mask])
                loss = loss + args.retain_weight * retain_ce
            
            # 2. Forget variance compression - compress forget features
            if forget_mask.any():
                forget_feats = features[forget_mask]
                # Minimize variance = features cluster together
                forget_centroid = forget_feats.mean(dim=0, keepdim=True)
                variance_loss = ((forget_feats - forget_centroid) ** 2).mean()
                loss = loss + args.compress_weight * variance_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}")
    
    # Save checkpoint
    output_name = f"unlearn_compress_vit_cifar10_forget{forget_class}_seed_{args.seed}"
    save_dir = f"checkpoints/unlearn_lora/{output_name}"
    ensure_dir(save_dir)
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
            outputs = student(normalizer(inputs))
            preds = outputs.argmax(dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            forget_mask = (labels == forget_class)
            forget_correct += (preds[forget_mask] == labels[forget_mask]).sum().item()
            forget_total += forget_mask.sum().item()
    
    print(f"Validation: overall={correct/total:.4f}, forget={forget_correct/forget_total:.4f}")
    
    # Measure forget variance
    student.eval()
    forget_features = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            norm_inputs = normalizer(inputs)
            features = feature_model.forward_features(norm_inputs)
            if features.dim() == 3:
                features = features[:, 0, :]
            forget_mask = (labels == forget_class)
            if forget_mask.any():
                forget_features.append(features[forget_mask].cpu())
    
    forget_feats = torch.cat(forget_features, dim=0)
    variance = ((forget_feats - forget_feats.mean(dim=0)) ** 2).mean().item()
    print(f"Forget variance: {variance:.4f} (target: < 15)")


if __name__ == "__main__":
    main()
