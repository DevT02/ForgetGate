import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import DataManager
from src.models.vit import create_vit_model
from src.models.cnn import create_cnn_model
from src.models.peft_lora import load_lora_adapter
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
    if model_type.startswith("vit"):
        model_config_name = model_type.replace("vit_", "")
        vit_cfg = dict(model_config["vit"][model_config_name])
        vit_cfg["pretrained"] = False
        model = create_vit_model(vit_cfg, num_classes=dataset_info["num_classes"])
    else:
        model = create_cnn_model(
            model_config["cnn"][model_type], num_classes=dataset_info["num_classes"]
        )
    return model.to(device)

def load_checkpoint_or_best(base_dir, name, seed):
    final_path = os.path.join(base_dir, f"{name}_seed_{seed}_final.pt")
    best_path = os.path.join(base_dir, f"{name}_seed_{seed}_best.pt")
    if os.path.exists(final_path):
        return final_path
    if os.path.exists(best_path):
        return best_path
    raise FileNotFoundError(f"Checkpoint not found: {final_path} (or {best_path})")

def extract_features(model, loader, device):
    model.eval()
    features_list = []
    labels_list = []
    
    # Try to hook the feature layer or use forward_features
    feature_model = None
    if hasattr(model, "forward_features"):
        feature_model = model
    elif hasattr(model, "module") and hasattr(model.module, "forward_features"):
        feature_model = model.module
    elif hasattr(model, "base_model") and hasattr(model.base_model, "forward_features"):
        feature_model = model.base_model
    elif hasattr(model, "model") and hasattr(model.model, "forward_features"):
         # ViTWrapper -> timm model
        feature_model = model.model
    # Handle PEFT wrapping ViTWrapper
    # PeftModel -> LoraModel -> ViTWrapper -> timm model
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        # This is likely ViTWrapper
        vit_wrapper = model.base_model.model
        if hasattr(vit_wrapper, "model") and hasattr(vit_wrapper.model, "forward_features"):
            feature_model = vit_wrapper.model
        elif hasattr(vit_wrapper, "forward_features"):
            feature_model = vit_wrapper
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            if feature_model:
                feats = feature_model.forward_features(inputs)
                # ViT forward_features returns [B, N, D], take CLS token [B, 0, D]
                if feats.ndim == 3:
                     feats = feats[:, 0]
            else:
                 # Fallback: Forward full model and hope we can get pre-logits
                 # This is tricky without hooks. Start with simple assumption for ViT
                 # If we can't get features easily, we might just get logits which is not ideal
                 # but for visualization often people use penultimate layer.
                 # Let's assume ViTWrapper has forward_features
                 print(f"Failed to find forward_features. Model type: {type(model)}")
                 print(f"Model attributes: {dir(model)}")
                 if hasattr(model, "base_model"):
                     print(f"Base model type: {type(model.base_model)}")
                     print(f"Base model attributes: {dir(model.base_model)}")
                 raise ValueError("Model does not support forward_features extraction")
            
            features_list.append(feats.cpu().numpy())
            labels_list.append(targets.numpy())
            
    return np.concatenate(features_list), np.concatenate(labels_list)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--suite", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=1000)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    experiment_suites = load_config(args.config)
    suite_config = experiment_suites[args.suite]
    
    # Setup data
    dataset_name = suite_config.get("dataset", "cifar10") # Default or from suite
    # Look for dataset in model_suites if not in top level
    if "model_suites" in suite_config:
        # Just grab first model suite to resolve dataset? 
        # Actually dataset is usually defined nicely. Let's assume cifar10 for now as per this repo
        pass
        
    dm = DataManager()
    train_loader = dm.get_dataloader(dataset_name, "train", batch_size=128)
    val_loader = dm.get_dataloader(dataset_name, "test", batch_size=128)
    
    models_to_plot = {}
    
    # Identify models to load
    model_suites = suite_config.get("model_suites", [])
    if not model_suites and "unlearned_model_suite" in suite_config:
        model_suites = [suite_config["unlearned_model_suite"]]
        
    print(f"Visualizing models: {model_suites}")
    
    plt.figure(figsize=(15, 5 * len(model_suites)))
    
    for idx, model_suite_name in enumerate(model_suites):
        print(f"Processing {model_suite_name}...")
        
        # Resolve model config
        base_suite_name, base_suite = resolve_base_suite_info(experiment_suites, model_suite_name)
        model_type = base_suite.get("model", "vit_tiny")
        
        # Build Model
        model = build_model(model_type, {"num_classes": 10}, device)
        
        # Load weights
        is_base = "base_model_suite" not in experiment_suites[model_suite_name] and "unlearned_model_suite" not in experiment_suites[model_suite_name]
        
        if is_base:
             print(f"Loading base model {model_suite_name}")
             if "oracle" in model_suite_name:
                 ckpt_dir = "checkpoints/oracle"
             else:
                 ckpt_dir = "checkpoints/base"
             ckpt = load_checkpoint_or_best(ckpt_dir, model_suite_name, args.seed)
             # Handle ViTWrapper loading - usually checkpoints are just state_dicts
             checkpoint = torch.load(ckpt, map_location=device)
             if "model_state_dict" in checkpoint:
                 state_dict = checkpoint["model_state_dict"]
             else:
                 state_dict = checkpoint
             model.load_state_dict(state_dict)
        else:
             print(f"Loading unlearned model {model_suite_name}")
             spec = experiment_suites[model_suite_name]
             base_name = spec.get("base_model_suite")
             if not base_name:
                 print(f"Warning: Could not determine base model for {model_suite_name}, skipping")
                 continue
            
             base_ckpt = load_checkpoint_or_best("checkpoints/base", base_name, args.seed)
             checkpoint = torch.load(base_ckpt, map_location=device)
             
             # Load base structure/weights first
             if "model_state_dict" in checkpoint:
                 state_dict = checkpoint["model_state_dict"]
             else:
                 state_dict = checkpoint
             model.load_state_dict(state_dict)
             
             # Determine path to unlearned weights
             custom_path = spec.get("path", None)
             if custom_path:
                 tgt_path = custom_path
             else:
                 tgt_path = f"checkpoints/unlearn_lora/{model_suite_name}_seed_{args.seed}"
                 
             # Check for Full FT
             is_full_ft = False
             if os.path.isfile(tgt_path):
                 is_full_ft = True
             elif os.path.isdir(tgt_path) and not os.path.exists(os.path.join(tgt_path, "adapter_config.json")):
                  if os.path.exists(os.path.join(tgt_path, "model.pt")):
                      tgt_path = os.path.join(tgt_path, "model.pt")
                      is_full_ft = True

             if is_full_ft:
                 print(f"[Info] Loading Full FT model from {tgt_path}")
                 ft_ckpt = torch.load(tgt_path, map_location=device)
                 if "model_state_dict" in ft_ckpt:
                     model.load_state_dict(ft_ckpt["model_state_dict"])
                 else:
                     model.load_state_dict(ft_ckpt)
             else:
                 print(f"[Info] Loading LoRA adapter from {tgt_path}")
                 model = load_lora_adapter(model, tgt_path).to(device)

        model.eval()
        model.to(device)
        
        # Extract features
        print(f"Extracting features from {model_suite_name}...")
        feats, labels = extract_features(model, val_loader, device)
        
        # t-SNE
        print("Running t-SNE...")
        tsne = TSNE(n_components=2, random_state=args.seed, max_iter=1000)
        # Subsample for speed
        if len(feats) > 2000:
            indices = np.random.choice(len(feats), 2000, replace=False)
            feats_sub = feats[indices]
            labels_sub = labels[indices]
        else:
            feats_sub = feats
            labels_sub = labels
            
        projections = tsne.fit_transform(feats_sub)
        
        # Plot
        ax = plt.subplot(len(model_suites), 1, idx + 1)
        
        # Plot Retain (Grey)
        retain_mask = labels_sub != 0
        ax.scatter(projections[retain_mask, 0], projections[retain_mask, 1], c='lightgrey', label='Retain', alpha=0.3, s=10)
        
        # Plot Forget (Red)
        forget_mask = labels_sub == 0
        ax.scatter(projections[forget_mask, 0], projections[forget_mask, 1], c='red', label='Forget (Class 0)', alpha=0.8, s=15)
        
        # Metrics Calculation
        retain_mask = labels_sub != 0
        forget_mask = labels_sub == 0
        
        retain_proj = projections[retain_mask]
        forget_proj = projections[forget_mask]
        
        dist = 0.0
        var_ratio = 1.0
        
        if len(retain_proj) > 0 and len(forget_proj) > 0:
            retain_center = np.mean(retain_proj, axis=0)
            forget_center = np.mean(forget_proj, axis=0)
            dist = np.linalg.norm(retain_center - forget_center)
            
            retain_var = np.var(retain_proj, axis=0).mean()
            forget_var = np.var(forget_proj, axis=0).mean()
            if retain_var > 0:
                var_ratio = forget_var / retain_var
            
            print(f"[{model_suite_name}] Metrics:")
            print(f"  Centroid Dist: {dist:.4f}")
            print(f"  Retain Var:    {retain_var:.4f}")
            print(f"  Forget Var:    {forget_var:.4f}")
            print(f"  Var Ratio:     {var_ratio:.4f}")
        
        ax.set_title(f"{model_suite_name} (Dist={dist:.2f}, VarRatio={var_ratio:.2f})")
        ax.legend()
        
    save_path = f"results/plots/tsne_comparison_{args.suite}.png"
    ensure_dir(os.path.dirname(save_path))
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved t-SNE plot to {save_path}")

if __name__ == "__main__":
    main()
