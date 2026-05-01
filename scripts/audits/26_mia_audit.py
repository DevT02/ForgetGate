import argparse
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import numpy as np
import json
import sys

# Ensure we can import from src regardless of where script is run
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.vit import create_vit_model
from src.models.cnn import create_cnn_model
from src.utils import load_config
from src.data import DataManager, create_forget_retain_splits

def compute_entropy(logits):
    """Compute Shannon entropy from logits."""
    probs = F.softmax(logits, dim=-1)
    # add small epsilon to prevent log(0)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    return entropy

def gather_metrics(model, dataloader, device):
    """Gathers cross entropy loss and prediction entropy for exactly one class."""
    model.eval()
    losses = []
    entropies = []
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                images, labels = batch
            else:
                images, labels = batch[0], batch[1]
                
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            ent = compute_entropy(logits)
            
            losses.extend(loss.cpu().numpy())
            entropies.extend(ent.cpu().numpy())
            
    return np.array(losses), np.array(entropies)

def main():
    parser = argparse.ArgumentParser(description="Membership Inference Attack (MIA) Audit")
    parser.add_argument("--config", type=str, default="configs/experiment_suites.yaml", help="Path to config file or dir")
    parser.add_argument("--suite", type=str, required=True, help="Suite name to evaluate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for extracting statistics")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running MIA Audit on {args.suite} with {device}")

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    import glob
    search_pattern = os.path.join(root, 'checkpoints', '**', f"{args.suite}*")
    matches = glob.glob(search_pattern, recursive=True)
    
    if not matches:
        print(f"Cannot find checkpoint directory for suite: {args.suite}")
        return
    checkpoint_dir = matches[0]
    suite_name = args.suite
    
    forget_class = 0
    if "forget" in suite_name:
        parts = suite_name.split("forget")
        if len(parts) > 1 and parts[1][0].isdigit():
            forget_class = int(parts[1][0])

    dataset_name = "cifar100" if "cifar100" in suite_name else "cifar10"
    is_resnet = "resnet" in suite_name
    model_name = "resnet18" if is_resnet else ("vit_small" if "vit_small" in suite_name else "vit_tiny")
    
    num_classes = 100 if dataset_name == "cifar100" else 10
    model_config = load_config("configs/model.yaml")
    
    # Recreate base model
    if is_resnet:
        model = create_cnn_model(model_config["cnn"][model_name], num_classes=num_classes)
    else:
        model_config_name = "small" if "small" in model_name else "tiny"
        model = create_vit_model(model_config["vit"][model_config_name], num_classes=num_classes)
    
    # Parse seed from directory
    seed = "42"
    if "_seed_" in checkpoint_dir:
        seed = checkpoint_dir.split("_seed_")[-1].split(os.sep)[0]
        
    experiment_suites = load_config(args.config)
    suite_cfg = experiment_suites.get(args.suite, {})
    base_suite_name = suite_cfg.get("base_model_suite", args.suite)
    
    # Load base weights
    base_ckpt = os.path.join(root, "checkpoints", "base", f"{base_suite_name}_seed_{seed}_final.pt")
    if os.path.exists(base_ckpt):
        try:
            checkpoint = torch.load(base_ckpt, map_location='cpu')
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        except Exception as e:
            pass
            
    # Load Adapter or Full ft weights
    from src.models.peft_lora import load_lora_adapter
    if os.path.exists(os.path.join(checkpoint_dir, "adapter_model.safetensors")) or os.path.exists(os.path.join(checkpoint_dir, "final_model", "adapter_model.safetensors")):
        # Path is either the root folder or final_model inside it
        adapter_path = checkpoint_dir
        if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            adapter_path = os.path.join(checkpoint_dir, "final_model")
        model = load_lora_adapter(model, adapter_path)
    else:
        final_pt = os.path.join(checkpoint_dir, 'final_model.pt')
        if os.path.exists(final_pt):
            ckpt = torch.load(final_pt, map_location='cpu')
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            
    model = model.to(device)

    # Load Data
    data_manager = DataManager()
    base_dataset = data_manager.load_dataset(dataset_name, split="train", use_pretrained=True)
    
    # We want to split the forget class to test MIA (seen vs unseen)
    forget_train, _, forget_test, _ = create_forget_retain_splits(
        base_dataset, forget_class=forget_class, train_ratio=0.8
    )
    
    # We also need the actual test set for forget unseen
    test_dataset = data_manager.load_dataset(dataset_name, split="test", include_classes=[forget_class], use_pretrained=True)
    
    forget_train_loader = torch.utils.data.DataLoader(forget_train, batch_size=args.batch_size, shuffle=False)
    forget_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Extracted {len(forget_train)} FORGET_TRAIN (seen) and {len(test_dataset)} FORGET_TEST (unseen) samples")

    print("Gathering metrics for FORGET_TRAIN...")
    train_loss, train_ent = gather_metrics(model, forget_train_loader, device)
    
    print("Gathering metrics for FORGET_TEST...")
    test_loss, test_ent = gather_metrics(model, forget_test_loader, device)
    
    # Membership Inference Attack (Thresholding Loss and Entropy)
    # The attacker predicts 1 (seen) if loss is lower, or entropy is lower.
    # Therefore, negative loss correlates with being in the training set.
    y_true = np.concatenate([np.ones_like(train_loss), np.zeros_like(test_loss)])
    y_pred_loss = np.concatenate([-train_loss, -test_loss])
    y_pred_ent = np.concatenate([-train_ent, -test_ent])
    
    auc_loss = roc_auc_score(y_true, y_pred_loss)
    auc_ent = roc_auc_score(y_true, y_pred_ent)
    
    print(f"--- MIA Results for {suite_name} ---")
    print(f"Loss-based MIA ROC-AUC:    {auc_loss:.4f}  (0.50 is perfect unlearning)")
    print(f"Entropy-based MIA ROC-AUC: {auc_ent:.4f}  (0.50 is perfect unlearning)")
    
    gap = max(auc_loss, auc_ent) - 0.50
    if gap > 0.15:
        print(f"WARNING: Severe data extraction vulnerability detected! (+{gap*100:.1f}% over random guessing)")
    elif gap <= 0.05:
        print(f"SUCCESS: The model successfully protects instance-level membership (+{gap*100:.1f}% over random guessing)")
        
    res_dir = os.path.join(root, 'results', 'analysis', 'metrics')
    os.makedirs(res_dir, exist_ok=True)
    out_file = os.path.join(res_dir, f'mia_audit_{suite_name}.json')
    
    with open(out_file, 'w') as f:
        json.dump({
            "suite": suite_name,
            "target_class": forget_class,
            "auc_loss": float(auc_loss),
            "auc_entropy": float(auc_ent),
            "memorization_gap": float(gap)
        }, f, indent=4)
        
    print(f"Saved MIA results to {out_file}")

if __name__ == "__main__":
    main()
