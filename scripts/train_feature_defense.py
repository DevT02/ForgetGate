#!/usr/bin/env python3
"""
Train 2026 unlearning methods with FEATURE-LEVEL defenses.
- FeatureOrthogonalizationDefense: push forget features orthogonal to centroid
- AdversarialFeatureMasking: add noise to discriminative dimensions
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
from src.unlearning.objectives import (
    BalDROUnlearning, 
    FeatureOrthogonalizationDefense,
    AdversarialFeatureMasking
)
from src.utils import set_seed, ensure_dir, load_config, get_device


def get_feature_model(model):
    """Get model for feature extraction."""
    for cand in [model, getattr(model, 'module', None), 
                 getattr(model, 'base_model', None), getattr(model, 'model', None)]:
        if cand is not None and hasattr(cand, 'forward_features'):
            return cand
    return None


def extract_features_batch(model, normalizer, inputs, device):
    """Extract CLS token features from a batch."""
    feature_model = get_feature_model(model)
    inputs = normalizer(inputs.to(device))
    
    with torch.no_grad():
        if feature_model:
            feats = feature_model.forward_features(inputs)
        else:
            feats = model(inputs)
        
        if feats.dim() == 3:
            feats = feats[:, 0, :]
    
    return feats


def compute_forget_centroid(model, normalizer, loader, device, forget_class):
    """Compute centroid of forget-class features."""
    model.eval()
    forget_features = []
    retain_features = []
    
    for inputs, labels in loader:
        feats = extract_features_batch(model, normalizer, inputs, device)
        forget_mask = (labels == forget_class)
        
        if forget_mask.any():
            forget_features.append(feats[forget_mask].cpu())
        if (~forget_mask).any():
            retain_features.append(feats[~forget_mask].cpu())
    
    forget_feats = torch.cat(forget_features, dim=0)
    retain_feats = torch.cat(retain_features, dim=0)
    
    forget_centroid = forget_feats.mean(dim=0)
    return forget_centroid, forget_feats, retain_feats


def main():
    parser = argparse.ArgumentParser(description="Train with feature-level defenses")
    parser.add_argument("--defense", type=str, default="ortho", choices=["ortho", "mask", "both"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ortho-weight", type=float, default=2.0)
    parser.add_argument("--mask-weight", type=float, default=1.0)
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
    
    # Compute forget-class centroid from base model
    print(f"[Defense] Computing forget-class centroid...")
    forget_centroid, forget_feats, retain_feats = compute_forget_centroid(
        student, normalizer, train_loader, device, forget_class
    )
    
    # Apply LoRA
    lora_config = create_lora_config(r=8, lora_alpha=16)
    student = apply_lora_to_model(student, lora_config).to(device)
    
    # Create base objective
    base_objective = BalDROUnlearning(forget_class, num_classes)
    
    # Wrap with feature-level defense
    if args.defense == "ortho":
        print(f"[Defense] Using FeatureOrthogonalization (weight={args.ortho_weight})")
        defended_objective = FeatureOrthogonalizationDefense(
            base_objective, ortho_weight=args.ortho_weight
        )
        defended_objective.set_centroid(forget_centroid.to(device))
    elif args.defense == "mask":
        print(f"[Defense] Using AdversarialFeatureMasking (weight={args.mask_weight})")
        defended_objective = AdversarialFeatureMasking(
            base_objective, mask_weight=args.mask_weight
        )
        defended_objective.compute_forget_directions(forget_feats, retain_feats)
    else:
        print(f"[Defense] Using BOTH defenses")
        ortho_obj = FeatureOrthogonalizationDefense(
            base_objective, ortho_weight=args.ortho_weight
        )
        ortho_obj.set_centroid(forget_centroid.to(device))
        defended_objective = AdversarialFeatureMasking(
            ortho_obj, mask_weight=args.mask_weight
        )
        defended_objective.compute_forget_directions(forget_feats, retain_feats)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr
    )
    
    print(f"[Defense] Training BalDRO + {args.defense} defense")
    print(f"          epochs={args.epochs}, seed={args.seed}")
    
    for epoch in range(args.epochs):
        student.train()
        total_loss = 0
        num_batches = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            norm_inputs = normalizer(inputs)
            
            # Get features for defense
            feature_model = get_feature_model(student)
            if feature_model:
                features = feature_model.forward_features(norm_inputs)
                if features.dim() == 3:
                    features = features[:, 0, :]
            else:
                features = None
            
            outputs = student(norm_inputs)
            loss = defended_objective(outputs, labels, features)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}")
    
    # Save checkpoint
    output_name = f"unlearn_baldro_{args.defense}_vit_cifar10_forget{forget_class}_seed_{args.seed}"
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


if __name__ == "__main__":
    main()
