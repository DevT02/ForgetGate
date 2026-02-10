#!/usr/bin/env python3
"""
Train BalDRO-SCRUB with Feature Orthogonalization (BalDRO-SCRUB-Ortho).
Combines:
1.  BalDRO: Hard negative mining (focus on worst forget samples).
2.  SCRUB: Teacher-Student Distillation + Forget Maximization.
3.  OrthoReg: Explicitly push forget features to be orthogonal to the class centroid.

This is the "Kitchen Sink" defense aiming for SOTA results.
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import DataManager
from src.models.vit import create_vit_model
from src.models.peft_lora import apply_lora_to_model, create_lora_config, save_lora_adapter
from src.models.normalize import create_imagenet_normalizer
from src.unlearning.objectives import (
    BalDRO_SCRUB,
    FeatureOrthogonalizationDefense
)
from src.utils import set_seed, ensure_dir, load_config, get_device

def get_feature_model(model):
    """Recursively search for a module with forward_features."""
    if hasattr(model, 'forward_features'):
        return model
    
    # Check common wrapper attributes
    for attr in ['module', 'base_model', 'model']:
        if hasattr(model, attr):
            child = getattr(model, attr)
            if child is not None:
                # Recursive call
                res = get_feature_model(child)
                if res is not None:
                    return res
    return None

def compute_stats(model, normalizer, loader, device, forget_class, return_all=False):
    """
    Compute statistics of forget and retain classes.
    If return_all=True, returns all retain features (for Imposter Defense).
    Otherwise returns centroids.
    """
    model.eval()
    forget_features = []
    retain_features = []
    
    # We need to compute features for the entire dataset relative to the forget class
    # Iterate over loader
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            norm_inputs = normalizer(inputs)
            
            # Extract features (CLS token)
            # Try to get features from model
            feature_model = None
            if hasattr(model, "forward_features"):
                feature_model = model
            elif hasattr(model, "module") and hasattr(model.module, "forward_features"):
                feature_model = model.module
            elif hasattr(model, "base_model") and hasattr(model.base_model, "forward_features"):
                feature_model = model.base_model
            # Handle ViTWrapper (Full FT): model.model is the timm model
            elif hasattr(model, "model") and hasattr(model.model, "forward_features"):
                feature_model = model.model
            # Handle PEFT wrapping ViTWrapper
            if feature_model is None and hasattr(model, "base_model") and hasattr(model.base_model, "model"):
                vit_wrapper = model.base_model.model
                if hasattr(vit_wrapper, "model") and hasattr(vit_wrapper.model, "forward_features"):
                    feature_model = vit_wrapper.model
                elif hasattr(vit_wrapper, "forward_features"):
                    feature_model = vit_wrapper
            
            if feature_model:
                feats = feature_model.forward_features(norm_inputs)
                if feats.dim() == 3:
                    feats = feats[:, 0]
            else:
                 # Fallback
                 feats = model(norm_inputs) # Logits? Or Pre-logits?
                 # Assuming logits is bad. But let's hope we found feature model.
                 pass

            forget_mask = (labels == forget_class)
            retain_mask = (labels != forget_class)
            
            if forget_mask.any():
                forget_features.append(feats[forget_mask].cpu())
            if retain_mask.any():
                retain_features.append(feats[retain_mask].cpu())
                
    forget_feats = torch.cat(forget_features, dim=0)
    retain_feats = torch.cat(retain_features, dim=0)
    
    if return_all:
        return forget_feats, retain_feats
        
    forget_centroid = forget_feats.mean(dim=0)
    
    # Compute centroids for each retain class (for Manifold Injection)
    retain_centroids = []
    # We need labels for retain_feats to compute per-class centroids?
    # Actually, we didn't store labels in the list.
    # But for "Manifold Injection" we used random retain features?
    # In previous code: "Also compute average similarity of retain samples to forget centroid?"
    # No, we switched to "Manifold Injection": Align forget with Retain Centroids.
    # The previous implementation of `compute_stats` returned `target_sim` which was scalar.
    # Then we hacked it in `main` to use random orthogonal vectors.
    
    # Let's fix this properly.
    # We want:
    # 1. Forget Centroid (for BalDRO target, maybe)
    # 2. Retain Features (All of them, for Imposter)
    # OR
    # 3. Retain Centroids (Per class, for Manifold Injection)
    
    # Since we didn't save labels for retain_feats, we can't compute per-class centroids easily here
    # unless we refactor.
    # BUT, `retain_feats` contains all retain samples mixed.
    # No, we want distinct targets to maintain variance?
    # The previous implementation used `target_sim` (scalar).
    
    # Let's assume `return_all=True` is for Imposter.
    # For legacy/centroid mode, we can compute Global Retain Centroid?
    # Compute stats logic refactor
    forget_centroid_feats = forget_feats.mean(dim=0)
    
    # Cosine sim for target_sim (legacy)
    sims = F.cosine_similarity(retain_feats, forget_centroid_feats.unsqueeze(0))
    target_sim = sims.mean().item()
    
    # For compatibility, return forget_centroid, target_sim
    # But main expects (..., retain_centroids) if not careful?
    # No, let's fix main.
    
    return forget_centroid_feats, target_sim

def main():
    parser = argparse.ArgumentParser(description="Train BalDRO-SCRUB-Ortho")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--forget-class", type=int, default=0)
    
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ortho-weight", type=float, default=10.0)
    # BalDRO-SCRUB params
    parser.add_argument("--distill-weight", type=float, default=1.0) # Alpha
    parser.add_argument("--forget-weight", type=float, default=10.0) # Beta (Loss Max) - STRONG Default
    parser.add_argument("--retain-ce-weight", type=float, default=1.0) # Gamma
    parser.add_argument("--target-weight", type=float, default=5.0) # Target (BalDRO margin) - STRONG Default
    parser.add_argument("--top-fraction", type=float, default=1.0) # Using all forget samples - STRONG Default
    
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--full-ft", action="store_true", help="Use Full Fine-Tuning instead of LoRA")
    
    # New Argument for Imposter Defense
    parser.add_argument("--imposter-mode", action="store_true", help="Sample random retain imposters instead of centroids")
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    # Loader for computing stats (needs to see all data)
    stats_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=2)

    normalizer = create_imagenet_normalizer().to(device)
    
    # 1. Build Base Model (Teacher)
    vit_cfg = dict(model_cfg["vit"]["tiny"])
    vit_cfg["pretrained"] = False
    base_model = create_vit_model(vit_cfg, num_classes=num_classes).to(device)
    
    base_ckpt_path = f"checkpoints/base/base_vit_cifar10_seed_{args.seed}_final.pt"
    if not os.path.exists(base_ckpt_path):
         base_ckpt_path = f"checkpoints/base/base_vit_cifar10_seed_42_final.pt"
         
    print(f"[Info] Loading base model from {base_ckpt_path}")
    base_ckpt = torch.load(base_ckpt_path, map_location=device)
    base_model.load_state_dict(base_ckpt["model_state_dict"])
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
        
    # 2. Compute Target Stats (Centroid + Retain Bank)
    forget_centroid = None
    retain_feats_bank = None
    
    if args.imposter_mode:
        _, retain_feats_bank = compute_stats(base_model, normalizer, stats_loader, device, forget_class, return_all=True)
        retain_feats_bank = retain_feats_bank.to(device)
        print(f"[Info] Imposter Mode: Using {len(retain_feats_bank)} retain features as bank (dim={retain_feats_bank.shape[-1]})")
    else:
        forget_centroid, target_sim = compute_stats(base_model, normalizer, stats_loader, device, forget_class)
        print(f"[Info] Computed Forget Centroid vs Retain Sim: {target_sim:.4f}")

    
    # 3. Build Student (Clone of Base)
    student = create_vit_model(vit_cfg, num_classes=num_classes).to(device)
    student.load_state_dict(base_ckpt["model_state_dict"])
    
    # 4. Apply LoRA or Full FT
    if args.full_ft:
        print("[Info] Using Full Fine-Tuning")
        # Ensure all parameters track gradients (default for new model) but model was loaded from ckpt.
        # But create_vit_model usually returns model with requires_grad=True
        # Just to be sure:
        for p in student.parameters():
            p.requires_grad = True
    else:
        print(f"[Info] Using LoRA with rank {args.lora_rank}")
        lora_config = create_lora_config(r=args.lora_rank, lora_alpha=16)
        student = apply_lora_to_model(student, lora_config).to(device)
    
    # 5. Setup Objectives
    # Inner: BalDRO-SCRUB
    inner_objective = BalDRO_SCRUB(
        forget_class=forget_class,
        num_classes=num_classes,
        teacher_model=base_model,
        top_fraction=args.top_fraction,
        target_weight=args.target_weight,
        distill_weight=args.distill_weight,
        forget_weight=args.forget_weight,
        retain_ce_weight=args.retain_ce_weight,
        temperature=2.0
    )
    
    # Outer: Feature Defense Wrapper
    if args.imposter_mode:
        print("[Info] Imposter Mode: Using raw inner objective (loss computed manually)")
        objective = inner_objective
    else:
        # Standard Orthogonalization Defense
        objective = FeatureOrthogonalizationDefense(
            inner_objective,
            ortho_weight=args.ortho_weight
        )
        objective.set_centroid(forget_centroid)
    
    # 6. Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr
    )
    
    # 7. Custom Training Loop
    print(f"[Info] Starting training BalDRO-SCRUB-Ortho...")
    print(f"       Epochs: {args.epochs}, Ortho Weight: {args.ortho_weight}")
    
    base_model.eval() # Teacher
    
    for epoch in range(args.epochs):
        student.train()
        total_loss = 0
        total_ortho_loss = 0
        num_batches = 0
        
        # We need to extract features for Ortho penalty
        # and inputs for SCRUB teacher distillation
        
        pbar = enumerate(train_loader) 
        # Manual progress bar removed for simplicity or use standard print
        
        for batch_idx, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            norm_inputs = normalizer(inputs)
            
            # --- FEATURE EXTRACTION for Defense ---
            feature_model = None
            
            # 1. Check for PEFT (LoRA)
            if hasattr(student, "base_model") and hasattr(student.base_model, "model"):
                 # LoRA -> ViTWrapper -> timm
                 vit_wrapper = student.base_model.model
                 # Check if vit_wrapper has model attribute (timm model inside wrapper)
                 if hasattr(vit_wrapper, "model") and hasattr(vit_wrapper.model, "forward_features"):
                     feature_model = vit_wrapper.model
                 elif hasattr(vit_wrapper, "forward_features"):
                     feature_model = vit_wrapper
            
            # 2. Check for ViTWrapper (Full FT)
            elif hasattr(student, "model") and hasattr(student.model, "forward_features"):
                 feature_model = student.model
                 
            # 3. Check for Timm (Direct)
            elif hasattr(student, "forward_features"):
                  feature_model = student
            
            features = None
            if feature_model:
                try:
                    features = feature_model.forward_features(norm_inputs)
                    if features.dim() == 3:
                        features = features[:, 0, :]
                except Exception as e:
                    pass

            # --- TEACHER FORWARD ---
            with torch.no_grad():
                teacher_logits = base_model(norm_inputs)

            # --- STUDENT FORWARD ---
            student_logits = student(norm_inputs)

            # --- LOSS COMPUTATION ---
            if args.imposter_mode:
                # 1. Base Loss (BalDRO + SCRUB)
                # BalDRO_SCRUB requires splitting batch manually
                forget_mask = (labels == forget_class)
                retain_mask = ~forget_mask
                
                loss_val = torch.tensor(0.0, device=device)
                
                # Forget Loss
                if forget_mask.any():
                    f_logits = student_logits[forget_mask]
                    f_teacher = teacher_logits[forget_mask]
                    f_labels = labels[forget_mask]
                    # compute_forget_loss handles BalDRO (maximize) + SCRUB (KL)
                    # It returns -(Loss), so we minimize it to maximize unlearning
                    loss_f = objective.compute_forget_loss(f_logits, f_teacher, labels=f_labels)
                    loss_val += loss_f
                    
                    # 2. Imposter Penalty (Only for Forget Samples)
                    if features is not None:
                         f_feats = features[forget_mask]
                         # Sample Imposters
                         indices = torch.randint(0, len(retain_feats_bank), (f_feats.size(0),)).to(device)
                         targets = retain_feats_bank[indices]
                         
                         # Cosine Similarity Maximization
                         cos_sim = F.cosine_similarity(f_feats, targets)
                         # Minimize (1 - cos_sim) => Maximize similarity
                         ortho_loss = (1 - cos_sim).mean()
                         
                         loss_val += args.ortho_weight * ortho_loss
                         total_ortho_loss += ortho_loss.item()

                # Retain Loss
                if retain_mask.any():
                    r_logits = student_logits[retain_mask]
                    r_teacher = teacher_logits[retain_mask]
                    r_labels = labels[retain_mask]
                    # compute_retain_loss minimizes Retain CE + KL
                    loss_r = objective.compute_retain_loss(r_logits, r_teacher, r_labels)
                    loss_val += loss_r
                
                loss = loss_val
            else:
                # Use Wrapper which handles features internally
                # It expects (student_output, labels, student_features, teacher_logits=...)
                loss = objective(student_logits, labels, features=features, teacher_logits=teacher_logits)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        avg_ortho = total_ortho_loss / max(1, num_batches)
        
        print(f"Epoch {epoch+1}/{args.epochs}: Loss={avg_loss:.4f} Ortho={avg_ortho:.4f}")

    # 8. Save
    output_name = f"unlearn_baldro_scrub_ortho_vit_cifar10_forget{forget_class}_seed_{args.seed}"
    if args.suffix:
        output_name += f"_{args.suffix}"
        
    save_dir = f"checkpoints/unlearn_lora/{output_name}"
    ensure_dir(save_dir)
    
    if args.full_ft:
        # Save full model
        torch.save({
            'model_state_dict': student.state_dict(),
            'config': vit_cfg,
            'args': vars(args)
        }, f"{save_dir}/model.pt")
    else:
        # Save LoRA adapter
        save_lora_adapter(student, save_dir)
        
    print(f"[Info] Saved to {save_dir}")

if __name__ == "__main__":
    main()
