#!/usr/bin/env python3
"""
Script 25: Adversarial Weight Perturbation (AWP) Audit
Tests if the model's unlearning is merely a "shallow valley" by applying a highly constrained, 
imperceptible $L_\infty$ perturbation directly to the model parameters to maximize forget class recovery.
"""

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data import DataManager
from src.models.vit import create_vit_model
from src.models.cnn import create_cnn_model
from src.models.peft_lora import load_lora_adapter
from src.utils import set_seed, ensure_dir, load_config, get_device

def evaluate(model, loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)
    return total_correct / max(1, total_samples)

def main():
    parser = argparse.ArgumentParser(description="Adversarial Weight Perturbation (AWP) Audit")
    parser.add_argument("--config", type=str, default="configs/suites")
    parser.add_argument("--suite", type=str, required=True, help="Unlearned suite to audit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eps", type=float, default=1e-3, help="L_infinity bound for weight perturbation")
    parser.add_argument("--steps", type=int, default=10, help="Number of PGD steps on parameters")
    parser.add_argument("--samples", type=int, default=500, help="Number of forget samples to use for extraction")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    experiment_suites = load_config(args.config)
    if args.suite not in experiment_suites:
        raise ValueError(f"Suite '{args.suite}' not found")
        
    suite_cfg = experiment_suites[args.suite]
    base_suite_name = suite_cfg.get("base_model_suite", args.suite) # Fallback to itself if it's already a base suite
    dataset_name = experiment_suites[base_suite_name].get("dataset", "cifar10")
    model_type = experiment_suites[base_suite_name].get("model", "vit_tiny")
    forget_class = suite_cfg.get("unlearning", {}).get("forget_class", 0)

    # 1. Load Data
    data_cfg = load_config("configs/data.yaml")
    dataset_info = data_cfg[dataset_name]
    data_manager = DataManager()
    
    # We use Test set for robust evaluation, Train set for the attack (to simulate adversary using a few known forgotten samples)
    train_dataset = data_manager.load_dataset(dataset_name, "train", use_pretrained=True, apply_imagenet_norm=True)
    test_dataset = data_manager.load_dataset(dataset_name, "test", use_pretrained=True, apply_imagenet_norm=True)

    forget_train_idx = [i for i, (_, y) in enumerate(train_dataset) if y == forget_class][:args.samples]
    forget_test_idx = [i for i, (_, y) in enumerate(test_dataset) if y == forget_class]
    retain_test_idx = [i for i, (_, y) in enumerate(test_dataset) if y != forget_class]

    forget_train_loader = DataLoader(Subset(train_dataset, forget_train_idx), batch_size=128, shuffle=True)
    forget_test_loader = DataLoader(Subset(test_dataset, forget_test_idx), batch_size=128, shuffle=False)
    retain_test_loader = DataLoader(Subset(test_dataset, retain_test_idx), batch_size=128, shuffle=False)

    # 2. Build Model
    if model_type.startswith("vit"):
        model_config = load_config("configs/model.yaml")
        model_config_name = model_type.replace("vit_", "")
        vit_cfg = dict(model_config["vit"][model_config_name])
        vit_cfg["pretrained"] = False
        model = create_vit_model(vit_cfg, num_classes=dataset_info["num_classes"]).to(device)
    else:
        model_config = load_config("configs/model.yaml")
        model = create_cnn_model(model_config["cnn"][model_type], num_classes=dataset_info["num_classes"]).to(device)

    # Load Unlearned weights
    if "unlearn" in args.suite:
        base_ckpt = f"checkpoints/base/{base_suite_name}_seed_{args.seed}_final.pt"
        checkpoint = torch.load(base_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load adapter or full ft
        adapter_path = f"checkpoints/unlearn_lora/{args.suite}_seed_{args.seed}"
        if os.path.exists(adapter_path):
            model = load_lora_adapter(model, adapter_path).to(device)
        else:
             # Look for full ft checkpoint
             full_ft_ckpt = f"checkpoints/unlearning/{args.suite}_seed_{args.seed}_final.pt"
             if os.path.exists(full_ft_ckpt):
                 ckpt = torch.load(full_ft_ckpt, map_location=device)
                 model.load_state_dict(ckpt["model_state_dict"])
    else:
        # Just loading a base/oracle model
        ckpt = f"checkpoints/base/{args.suite}_seed_{args.seed}_final.pt"
        checkpoint = torch.load(ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    # 3. Clean Evaluation
    clean_forget_acc = evaluate(model, forget_test_loader, device)
    clean_retain_acc = evaluate(model, retain_test_loader, device)
    print(f"[{args.suite}] Clean Forget Acc: {clean_forget_acc:.4f} | Clean Retain Acc: {clean_retain_acc:.4f}")

    # 4. Adversarial Weight Perturbation (AWP)
    print(f"\nRunning AWP (eps={args.eps}, steps={args.steps}) to reconstruct forget class...")
    
    # Store origin parameters
    orig_params = {n: p.clone().detach() for n, p in model.named_parameters()}
    
    # Enable gradients for all parameters mathematically
    for p in model.parameters():
        p.requires_grad = True

    alpha = args.eps / max(1, (args.steps * 0.5))  # Step size ensuring we easily reach epsilon
    
    for step in range(args.steps):
        model.zero_grad()
        total_loss = 0.0
        
        # We process the whole extraction batch
        for x, y in forget_train_loader:
            x, y = x.to(device), y.to(device)
            # Use enable_grad since model is in eval mode (keeps BN / Dropout frozen but computes gradients)
            with torch.enable_grad():
                outputs = model(x)
                # Minimize loss on the forget class (which pushes model to re-learn it)
                loss = nn.functional.cross_entropy(outputs, y)
            
            loss.backward()
            total_loss += loss.item()
            
        # Update parameters via PGD
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.grad is not None:
                    # Gradient descent step to MINIMIZE loss on forget class
                    p.data = p.data - alpha * torch.sign(p.grad)
                    
                    # Project back to L_infinity epsilon ball around original parameters
                    orig_p = orig_params[n]
                    p.data = torch.max(orig_p - args.eps, torch.min(orig_p + args.eps, p.data))
                    
        print(f"  Step {step+1}/{args.steps} - Extraction Loss: {total_loss:.4f}")

    # 5. Perturbed Evaluation
    model.eval()
    pert_forget_acc = evaluate(model, forget_test_loader, device)
    pert_retain_acc = evaluate(model, retain_test_loader, device)
    print(f"\n[{args.suite}] Perturbed Forget Acc: {pert_forget_acc:.4f} | Perturbed Retain Acc: {pert_retain_acc:.4f}")
    
    # 6. Save Report
    recovery_gap = pert_forget_acc - clean_forget_acc
    print(f"--> Catastrophic Re-emergence Gap: +{recovery_gap*100:.2f}% (eps={args.eps})")

    results = {
        "suite": args.suite,
        "seed": args.seed,
        "eps": args.eps,
        "steps": args.steps,
        "samples": args.samples,
        "clean_forget_acc": clean_forget_acc,
        "clean_retain_acc": clean_retain_acc,
        "pert_forget_acc": pert_forget_acc,
        "pert_retain_acc": pert_retain_acc,
        "recovery_gap": recovery_gap,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    ensure_dir("results/analysis")
    out_path = f"results/analysis/awp_audit_{args.suite}_seed_{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved AWP audit results to {out_path}")

if __name__ == "__main__":
    main()
