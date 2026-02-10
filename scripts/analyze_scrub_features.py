#!/usr/bin/env python3
"""
Deep analysis of why SCRUB is probe-resistant.
Compare feature-level differences between SCRUB and other unlearning methods.
"""

import argparse
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import DataManager
from src.models.vit import create_vit_model
from src.models.peft_lora import load_lora_adapter
from src.models.normalize import create_imagenet_normalizer
from src.utils import set_seed, ensure_dir, load_config, get_device


def extract_features(model, normalizer, loader, device, max_samples=2000):
    """Extract CLS token features from model."""
    model.eval()
    features = []
    labels = []
    
    # Get feature extractor
    feature_model = None
    for cand in [model, getattr(model, 'module', None), getattr(model, 'base_model', None)]:
        if cand is not None and hasattr(cand, 'forward_features'):
            feature_model = cand
            break
    
    samples_collected = 0
    with torch.no_grad():
        for inputs, y in loader:
            if samples_collected >= max_samples:
                break
            inputs = inputs.to(device)
            inputs = normalizer(inputs)
            
            if feature_model:
                feats = feature_model.forward_features(inputs)
            else:
                feats = model(inputs)
            
            if feats.dim() == 3:
                feats = feats[:, 0, :]  # CLS token
            
            features.append(feats.cpu())
            labels.append(y)
            samples_collected += len(y)
    
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def compute_cka(X, Y):
    """Compute Centered Kernel Alignment between feature matrices."""
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    
    XX = X @ X.T
    YY = Y @ Y.T
    
    # Frobenius inner product
    hsic_xy = (XX * YY).sum()
    hsic_xx = (XX * XX).sum()
    hsic_yy = (YY * YY).sum()
    
    cka = hsic_xy / (torch.sqrt(hsic_xx) * torch.sqrt(hsic_yy) + 1e-8)
    return cka.item()


def compute_class_separability(features, labels, forget_class):
    """Compute how separable forget class is from retain classes."""
    forget_mask = (labels == forget_class)
    retain_mask = ~forget_mask
    
    if not forget_mask.any() or not retain_mask.any():
        return 0.0, 0.0, 0.0
    
    forget_feats = features[forget_mask]
    retain_feats = features[retain_mask]
    
    # Compute centroids
    forget_centroid = forget_feats.mean(dim=0)
    retain_centroid = retain_feats.mean(dim=0)
    
    # Inter-class distance (between centroids)
    inter_dist = F.cosine_similarity(forget_centroid.unsqueeze(0), 
                                      retain_centroid.unsqueeze(0)).item()
    
    # Intra-class variance (within forget class)
    forget_var = ((forget_feats - forget_centroid) ** 2).mean().item()
    retain_var = ((retain_feats - retain_centroid) ** 2).mean().item()
    
    return inter_dist, forget_var, retain_var


def main():
    parser = argparse.ArgumentParser(description="Deep SCRUB analysis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
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
    test_dataset = data_manager.load_dataset(dataset_name, "test", use_pretrained=True, apply_imagenet_norm=False)
    loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    normalizer = create_imagenet_normalizer().to(device)
    
    # Models to analyze
    models_to_analyze = {
        "base": ("checkpoints/base", "base_vit_cifar10", "base"),
        "oracle": ("checkpoints/oracle", "oracle_vit_cifar10_forget0", "base"),
        "scrub": ("checkpoints/unlearn_lora", "unlearn_scrub_distill_vit_cifar10_forget0", "lora"),
        "kl": ("checkpoints/unlearn_lora", "unlearn_kl_vit_cifar10_forget0", "lora"),
        "sga": ("checkpoints/unlearn_lora", "unlearn_sga_vit_cifar10_forget0_smoke", "lora"),
        "baldro": ("checkpoints/unlearn_lora", "unlearn_baldro_vit_cifar10_forget0_smoke", "lora"),
    }
    
    # Build base model template
    vit_cfg = dict(model_cfg["vit"]["tiny"])
    vit_cfg["pretrained"] = False
    
    results = {}
    base_features = None
    
    print(f"[SCRUB Analysis] Extracting features, forget_class={forget_class}, seed={args.seed}")
    print("=" * 60)
    
    for name, (ckpt_dir, suite_name, model_type) in models_to_analyze.items():
        print(f"\n[{name.upper()}]")
        
        # Load model
        model = create_vit_model(vit_cfg, num_classes=num_classes).to(device)
        
        try:
            if model_type == "base":
                ckpt_path = f"{ckpt_dir}/{suite_name}_seed_{args.seed}_final.pt"
                if not os.path.exists(ckpt_path):
                    ckpt_path = f"{ckpt_dir}/{suite_name}_seed_{args.seed}_best.pt"
                ckpt = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                # Load base first
                base_ckpt = torch.load(f"checkpoints/base/base_vit_cifar10_seed_{args.seed}_final.pt", 
                                       map_location=device)
                model.load_state_dict(base_ckpt["model_state_dict"])
                # Load LoRA adapter
                adapter_path = f"{ckpt_dir}/{suite_name}_seed_{args.seed}"
                model = load_lora_adapter(model, adapter_path).to(device)
        except Exception as e:
            print(f"  Skip: {e}")
            continue
        
        # Extract features
        features, labels = extract_features(model, normalizer, loader, device)
        
        # Store base features for CKA comparison
        if name == "base":
            base_features = features
        
        # Compute metrics
        inter_dist, forget_var, retain_var = compute_class_separability(features, labels, forget_class)
        
        result = {
            "inter_class_cosine": inter_dist,
            "forget_variance": forget_var,
            "retain_variance": retain_var,
        }
        
        # CKA with base model
        if base_features is not None and name != "base":
            cka = compute_cka(features, base_features)
            result["cka_vs_base"] = cka
            print(f"  CKA vs base: {cka:.4f}")
        
        print(f"  Inter-class cosine (forget vs retain): {inter_dist:.4f}")
        print(f"  Forget class variance: {forget_var:.6f}")
        print(f"  Retain class variance: {retain_var:.6f}")
        
        results[name] = result
    
    # Save results
    ensure_dir("results/analysis")
    out_path = f"results/analysis/scrub_deep_analysis_seed_{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump({
            "seed": args.seed,
            "forget_class": forget_class,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "models": results
        }, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"Saved analysis to {out_path}")
    
    # Summary
    print(f"\n[SUMMARY]")
    print(f"Method      | CKA vs Base | Inter-Cosine | Forget Var")
    print("-" * 55)
    for name, r in results.items():
        cka = r.get("cka_vs_base", 1.0)
        inter = r.get("inter_class_cosine", 0)
        fvar = r.get("forget_variance", 0)
        print(f"{name:11s} | {cka:10.4f} | {inter:12.4f} | {fvar:.6f}")


if __name__ == "__main__":
    main()
