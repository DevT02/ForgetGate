#!/usr/bin/env python3
"""Quick probe of the variance compression model."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.data import DataManager
from src.models.vit import create_vit_model
from src.models.peft_lora import load_lora_adapter
from src.models.normalize import create_imagenet_normalizer
from src.utils import set_seed, load_config, get_device


def extract_features(model, normalizer, loader, device, forget_class):
    """Extract features and binary labels."""
    model.eval()
    features = []
    binary_labels = []
    
    # Get feature model
    feature_model = None
    for name, module in model.named_modules():
        if hasattr(module, 'forward_features'):
            feature_model = module
            break
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = normalizer(inputs.to(device))
            if feature_model:
                feats = feature_model.forward_features(inputs)
            else:
                feats = model(inputs)
            if feats.dim() == 3:
                feats = feats[:, 0, :]
            
            features.append(feats.cpu())
            binary_labels.append((labels == forget_class).float())
    
    return torch.cat(features, dim=0), torch.cat(binary_labels, dim=0)


def train_probe(features, labels, seed=42, epochs=50):
    """Train linear probe."""
    import torch.nn as nn
    
    n = features.size(0)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    split = int(0.8 * n)
    
    x_train, y_train = features[idx[:split]], labels[idx[:split]]
    x_val, y_val = features[idx[split:]], labels[idx[split:]]
    
    probe = nn.Linear(features.size(1), 1)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    
    for _ in range(epochs):
        probe.train()
        optimizer.zero_grad()
        loss = criterion(probe(x_train).squeeze(), y_train)
        loss.backward()
        optimizer.step()
    
    probe.eval()
    with torch.no_grad():
        preds = (torch.sigmoid(probe(x_val).squeeze()) > 0.5).float()
        acc = (preds == y_val).float().mean().item()
    
    return acc


def main():
    set_seed(42)
    device = get_device(None)
    
    data_cfg = load_config("configs/data.yaml")
    model_cfg = load_config("configs/model.yaml")
    
    # Load data
    data_manager = DataManager()
    test_dataset = data_manager.load_dataset("cifar10", "test", use_pretrained=True, apply_imagenet_norm=False)
    loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    normalizer = create_imagenet_normalizer().to(device)
    
    forget_class = 0
    vit_cfg = dict(model_cfg["vit"]["tiny"])
    vit_cfg["pretrained"] = False
    
    results = {}
    
    # Models to probe
    models = {
        "scrub_full": ("checkpoints/unlearn_lora/unlearn_scrub_full_vit_cifar10_forget0_seed_42", "lora"),
        "oracle": ("checkpoints/oracle/oracle_vit_cifar10_forget0_seed_42_final.pt", "base"),
        "scrub": ("checkpoints/unlearn_lora/unlearn_scrub_distill_vit_cifar10_forget0_seed_42", "lora"),
    }
    
    for name, (path, model_type) in models.items():
        print(f"[{name.upper()}] Probing...")
        
        model = create_vit_model(vit_cfg, num_classes=10).to(device)
        
        try:
            if model_type == "base":
                ckpt = torch.load(path, map_location=device)
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                base_ckpt = torch.load("checkpoints/base/base_vit_cifar10_seed_42_final.pt", map_location=device)
                model.load_state_dict(base_ckpt["model_state_dict"])
                model = load_lora_adapter(model, path).to(device)
        except Exception as e:
            print(f"  Skip: {e}")
            continue
        
        features, labels = extract_features(model, normalizer, loader, device, forget_class)
        acc = train_probe(features, labels)
        
        print(f"  Probe accuracy: {acc*100:.1f}%")
        results[name] = acc
    
    print("\n" + "=" * 50)
    print("RESULTS:")
    print("=" * 50)
    oracle_acc = results.get("oracle", 0)
    for name, acc in results.items():
        gap = (acc - oracle_acc) * 100
        status = "✅" if abs(gap) < 1.5 else "❌"
        print(f"{name:10s}: {acc*100:.1f}%  (gap: {gap:+.1f}pp {status})")


if __name__ == "__main__":
    main()
