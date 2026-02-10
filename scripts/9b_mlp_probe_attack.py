#!/usr/bin/env python3
"""
Stronger feature probe attack using MLP (nonlinear) probe.
Designed to break SCRUB which resists linear probes.
"""

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import DataManager
from src.models.vit import create_vit_model
from src.models.peft_lora import load_lora_adapter
from src.models.normalize import create_imagenet_normalizer
from src.utils import set_seed, ensure_dir, load_config, get_device


def resolve_base_suite_info(experiment_suites, suite_name):
    suite = experiment_suites[suite_name]
    if "base_model_suite" in suite:
        return resolve_base_suite_info(experiment_suites, suite["base_model_suite"])
    if "unlearned_model_suite" in suite:
        return resolve_base_suite_info(experiment_suites, suite["unlearned_model_suite"])
    return suite_name, suite


def build_model(model_type, dataset_info, device):
    model_config = load_config("configs/model.yaml")
    model_config_name = model_type.replace("vit_", "")
    vit_cfg = dict(model_config["vit"][model_config_name])
    vit_cfg["pretrained"] = False
    from src.models.vit import create_vit_model
    model = create_vit_model(vit_cfg, num_classes=dataset_info["num_classes"])
    return model.to(device)


def load_checkpoint_or_best(base_dir, name, seed):
    final_path = os.path.join(base_dir, f"{name}_seed_{seed}_final.pt")
    best_path = os.path.join(base_dir, f"{name}_seed_{seed}_best.pt")
    if os.path.exists(final_path):
        return final_path
    if os.path.exists(best_path):
        return best_path
    raise FileNotFoundError(f"Checkpoint not found: {final_path}")


def get_feature_model(model):
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
    
    for cand in candidates:
        if hasattr(cand, "forward_features"):
            return cand
    return None


def extract_features(model, normalizer, loader, device, forget_class):
    model.eval()
    feature_model = get_feature_model(model)
    features = []
    labels = []

    with torch.no_grad():
        for inputs, y in loader:
            inputs = inputs.to(device)
            y = y.to(device)
            inputs = normalizer(inputs)
            
            if feature_model is not None:
                feats = feature_model.forward_features(inputs)
                if isinstance(feats, (tuple, list)):
                    feats = feats[0]
            else:
                feats = model(inputs)
            
            if feats.dim() == 3:
                feats = feats[:, 0, :]
            
            features.append(feats.cpu())
            labels.append(y.cpu())

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    binary = (labels == forget_class).float().unsqueeze(1)
    return features, binary


class MLPProbe(nn.Module):
    """Nonlinear MLP probe - stronger than linear"""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x)


def train_mlp_probe(features, labels, seed, device, epochs=100, lr=1e-3, hidden_dim=256):
    n = features.size(0)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    split = int(0.8 * n)
    train_idx = idx[:split]
    val_idx = idx[split:]

    x_train = features[train_idx].to(device)
    y_train = labels[train_idx].to(device)
    x_val = features[val_idx].to(device)
    y_val = labels[val_idx].to(device)

    probe = MLPProbe(x_train.size(1), hidden_dim).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    best_auc = 0.0
    
    for epoch in range(epochs):
        probe.train()
        logits = probe(x_train)
        loss = loss_fn(logits, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        
        # Eval every 10 epochs
        if (epoch + 1) % 10 == 0:
            probe.eval()
            with torch.no_grad():
                val_logits = probe(x_val)
                val_pred = (torch.sigmoid(val_logits) >= 0.5).float()
                val_acc = val_pred.eq(y_val).float().mean().item()
                val_auc = compute_auc(y_val, torch.sigmoid(val_logits))
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_auc = val_auc

    probe.eval()
    with torch.no_grad():
        train_logits = probe(x_train)
        val_logits = probe(x_val)
        train_pred = (torch.sigmoid(train_logits) >= 0.5).float()
        val_pred = (torch.sigmoid(val_logits) >= 0.5).float()
        train_acc = train_pred.eq(y_train).float().mean().item()
        val_acc = val_pred.eq(y_val).float().mean().item()
        val_auc = compute_auc(y_val, torch.sigmoid(val_logits))

    return {
        "train_acc": train_acc,
        "val_acc": max(val_acc, best_val_acc),
        "val_auc": max(val_auc, best_auc),
        "n_train": int(x_train.size(0)),
        "n_val": int(x_val.size(0)),
        "probe_type": "mlp",
        "hidden_dim": hidden_dim,
    }


def compute_auc(y_true, y_score):
    y_true = y_true.view(-1).cpu()
    y_score = y_score.view(-1).cpu()
    pos = (y_true == 1).sum().item()
    neg = (y_true == 0).sum().item()
    if pos == 0 or neg == 0:
        return 0.0
    scores, order = torch.sort(y_score)
    y_true = y_true[order]
    rank_sum = torch.arange(1, len(y_true) + 1, dtype=torch.float32)[y_true == 1].sum().item()
    auc = (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def main():
    parser = argparse.ArgumentParser(description="MLP probe attack (stronger than linear)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--suite", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=256)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    experiment_suites = load_config(args.config)
    suite_cfg = experiment_suites[args.suite]
    model_suites = suite_cfg.get("model_suites", [])

    base_suite_name, base_suite = resolve_base_suite_info(experiment_suites, model_suites[0])
    dataset_name = base_suite.get("dataset", "cifar10")
    model_type = base_suite.get("model", "vit_tiny")

    data_cfg = load_config("configs/data.yaml")
    dataset_info = data_cfg[dataset_name]
    forget_class = suite_cfg.get("forget_class", 0)

    data_manager = DataManager()
    dataset = data_manager.load_dataset(dataset_name, "test", use_pretrained=True, apply_imagenet_norm=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    normalizer = create_imagenet_normalizer().to(device)

    results = {
        "suite": args.suite,
        "seed": args.seed,
        "probe_type": "mlp",
        "hidden_dim": args.hidden_dim,
        "forget_class": forget_class,
        "run_info": {"started_utc": datetime.utcnow().isoformat() + "Z"},
        "models": {},
    }

    for model_suite in model_suites:
        print(f"[MLP Probe] {model_suite}")
        model = build_model(model_type, dataset_info, device)

        if model_suite.startswith("base_"):
            ckpt_path = load_checkpoint_or_best("checkpoints/base", model_suite, args.seed)
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
        elif model_suite.startswith("oracle_"):
            ckpt_path = load_checkpoint_or_best("checkpoints/oracle", model_suite, args.seed)
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
        elif model_suite.startswith("unlearn_"):
            base_suite = experiment_suites[model_suite].get("base_model_suite", None)
            if not base_suite:
                continue
            ckpt_path = load_checkpoint_or_best("checkpoints/base", base_suite, args.seed)
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            adapter_path = f"checkpoints/unlearn_lora/{model_suite}_seed_{args.seed}"
            model = load_lora_adapter(model, adapter_path).to(device)
        else:
            continue

        features, labels = extract_features(model, normalizer, loader, device, forget_class)
        probe_stats = train_mlp_probe(
            features, labels, seed=args.seed, device=device,
            epochs=args.epochs, hidden_dim=args.hidden_dim
        )
        results["models"][model_suite] = {"n_samples": int(features.size(0)), **probe_stats}

    ensure_dir("results/logs")
    out_path = f"results/logs/mlp_probe_{args.suite}_seed_{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved MLP probe results to {out_path}")


if __name__ == "__main__":
    main()
