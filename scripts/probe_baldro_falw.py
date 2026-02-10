#!/usr/bin/env python3
"""Probe BalDRO and FaLW to add to our attack results."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.data import DataManager
from src.models.vit import create_vit_model
from src.models.peft_lora import load_lora_adapter
from src.models.normalize import create_imagenet_normalizer
from src.utils import set_seed, load_config, get_device


def get_feature_model(model):
    """Get the model that has forward_features method."""
    candidates = [model]
    if hasattr(model, "module"):
        candidates.append(model.module)
    if hasattr(model, "get_base_model"):
        try:
            candidates.append(model.get_base_model())
        except:
            pass
    if hasattr(model, "base_model"):
        candidates.append(model.base_model)
    if hasattr(model, "model"):
        candidates.append(model.model)
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        candidates.append(model.base_model.model)
    
    for cand in candidates:
        if hasattr(cand, "forward_features"):
            return cand
    return None


def extract_features(model, loader, normalizer, device):
    """Extract CLS features from model."""
    model.eval()
    feature_model = get_feature_model(model)
    features = []
    labels = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = normalizer(inputs.to(device))
            
            if feature_model is not None:
                feats = feature_model.forward_features(inputs)
            else:
                feats = model(inputs)
            
            # ViT: use CLS token
            if feats.dim() == 3:
                feats = feats[:, 0, :]
            
            features.append(feats.cpu())
            labels.append(targets)
    
    return torch.cat(features), torch.cat(labels)


def train_probe(features, labels, forget_class, device):
    """Train binary probe: forget vs retain."""
    binary_labels = (labels == forget_class).long()
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        features.numpy(), binary_labels.numpy(), 
        test_size=0.2, random_state=42, stratify=binary_labels.numpy()
    )
    
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)
    
    # Train logistic regression
    probe = nn.Linear(features.shape[1], 2).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(100):
        probe.train()
        optimizer.zero_grad()
        out = probe(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
    
    # Evaluate
    probe.eval()
    with torch.no_grad():
        val_out = probe(X_val)
        val_preds = val_out.argmax(dim=1)
        val_acc = (val_preds == y_val).float().mean().item()
    
    return val_acc


def main():
    set_seed(42)
    device = get_device(None)
    
    data_cfg = load_config("configs/data.yaml")
    model_cfg = load_config("configs/model.yaml")
    
    # Load test data
    data_manager = DataManager()
    test_dataset = data_manager.load_dataset("cifar10", "test", use_pretrained=True, apply_imagenet_norm=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    normalizer = create_imagenet_normalizer().to(device)
    
    forget_class = 0
    
    vit_cfg = dict(model_cfg["vit"]["tiny"])
    vit_cfg["pretrained"] = False
    
    print("=" * 60)
    print("PROBE ATTACK: BalDRO and FaLW")
    print("=" * 60)
    
    results = {}
    
    # Oracle first (baseline)
    print("\n[ORACLE] Loading...")
    oracle = create_vit_model(vit_cfg, num_classes=10).to(device)
    ckpt = torch.load("checkpoints/oracle/oracle_vit_cifar10_forget0_seed_42_final.pt", map_location=device)
    oracle.load_state_dict(ckpt["model_state_dict"])
    features, labels = extract_features(oracle, test_loader, normalizer, device)
    results["oracle"] = train_probe(features, labels, forget_class, device)
    print(f"  Probe accuracy: {results['oracle']*100:.1f}%")
    
    # BalDRO
    print("\n[BALDRO] Loading...")
    baldro = create_vit_model(vit_cfg, num_classes=10).to(device)
    base_ckpt = torch.load("checkpoints/base/base_vit_cifar10_seed_42_final.pt", map_location=device)
    baldro.load_state_dict(base_ckpt["model_state_dict"])
    load_lora_adapter(baldro, "checkpoints/unlearn_lora/unlearn_baldro_vit_cifar10_forget0_smoke_seed_42")
    features, labels = extract_features(baldro, test_loader, normalizer, device)
    results["baldro"] = train_probe(features, labels, forget_class, device)
    print(f"  Probe accuracy: {results['baldro']*100:.1f}%")
    
    # FaLW
    print("\n[FALW] Loading...")
    falw = create_vit_model(vit_cfg, num_classes=10).to(device)
    falw.load_state_dict(base_ckpt["model_state_dict"])
    load_lora_adapter(falw, "checkpoints/unlearn_lora/unlearn_falw_vit_cifar10_forget0_smoke_seed_42")
    features, labels = extract_features(falw, test_loader, normalizer, device)
    results["falw"] = train_probe(features, labels, forget_class, device)
    print(f"  Probe accuracy: {results['falw']*100:.1f}%")
    
    # Summary
    oracle_acc = results["oracle"]
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Method':<15} | {'Probe Acc':>10} | {'Gap vs Oracle':>15}")
    print("-" * 45)
    for name, acc in results.items():
        gap = (acc - oracle_acc) * 100
        status = "OK" if abs(gap) < 2 else "LEAK"
        print(f"{name:<15} | {acc*100:>9.1f}% | {gap:>+14.1f}pp {status}")


if __name__ == "__main__":
    main()
