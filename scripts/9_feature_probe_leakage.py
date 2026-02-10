#!/usr/bin/env python3
"""
Feature-probe leakage test (linear probe on frozen features).
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
from src.models.cnn import create_cnn_model
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


def get_feature_model(model):
    candidates = []
    candidates.append(model)
    if hasattr(model, "module"):
        candidates.append(model.module)
    if hasattr(model, "get_base_model"):
        try:
            candidates.append(model.get_base_model())
        except Exception:
            pass
    if hasattr(model, "base_model"):
        candidates.append(model.base_model)
    if hasattr(model, "model"):
        candidates.append(model.model)
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        candidates.append(model.base_model.model)
    if hasattr(model, "get_base_model"):
        try:
            bm = model.get_base_model()
            if hasattr(bm, "model"):
                candidates.append(bm.model)
        except Exception:
            pass

    for cand in candidates:
        if hasattr(cand, "forward_features"):
            return cand
    return None


def resolve_vit_blocks(feature_model):
    if hasattr(feature_model, "blocks"):
        return feature_model.blocks
    if hasattr(feature_model, "model") and hasattr(feature_model.model, "blocks"):
        return feature_model.model.blocks
    return None


def select_block_index(num_blocks, probe_layer):
    if probe_layer is None or probe_layer == "final":
        return None
    if probe_layer.startswith("block"):
        try:
            return int(probe_layer.replace("block", ""))
        except ValueError:
            return None
    if probe_layer == "early":
        return max(0, num_blocks // 4)
    if probe_layer == "mid":
        return max(0, num_blocks // 2)
    if probe_layer == "late":
        return max(0, num_blocks - 1)
    return None


def extract_features(model, normalizer, loader, device, forget_class, max_samples_per_class=None, probe_layer=None):
    model.eval()
    feature_model = get_feature_model(model)
    blocks = resolve_vit_blocks(feature_model) if feature_model is not None else None
    block_idx = select_block_index(len(blocks) if blocks is not None else 0, probe_layer)
    features = []
    labels = []
    class_counts = {}

    with torch.no_grad():
        for inputs, y in loader:
            inputs = inputs.to(device)
            y = y.to(device)

            if max_samples_per_class is not None:
                keep = []
                for i, yi in enumerate(y.tolist()):
                    count = class_counts.get(yi, 0)
                    if count < max_samples_per_class:
                        keep.append(i)
                        class_counts[yi] = count + 1
                if not keep:
                    continue
                keep_idx = torch.tensor(keep, device=device)
                inputs = inputs.index_select(0, keep_idx)
                y = y.index_select(0, keep_idx)

            inputs = normalizer(inputs)

            feats = None
            hook_handle = None
            captured = {}
            if feature_model is not None and blocks is not None and block_idx is not None:
                def _hook(_mod, _inp, out):
                    captured["feat"] = out
                if block_idx < len(blocks):
                    hook_handle = blocks[block_idx].register_forward_hook(_hook)

            if feature_model is not None:
                feats = feature_model.forward_features(inputs)
                if isinstance(feats, (tuple, list)):
                    feats = feats[0]
            else:
                feats = model(inputs)

            if hook_handle is not None:
                hook_handle.remove()
                if "feat" in captured:
                    feats = captured["feat"]

            # Reduce to (batch, dim) for probing
            if feats.dim() == 4:
                # CNN-like features: global average pool
                feats = feats.mean(dim=(2, 3))
            elif feats.dim() == 3:
                # ViT-like features: use CLS token when available
                feats = feats[:, 0, :]

            feats = feats.detach().cpu()
            y = y.detach().cpu()
            features.append(feats)
            labels.append(y)

    if not features:
        return None, None

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    binary = (labels == forget_class).float().unsqueeze(1)
    return features, binary


def train_linear_probe(features, labels, seed, device, epochs=50, lr=1e-2):
    n = features.size(0)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    split = int(0.8 * n)
    train_idx = idx[:split]
    val_idx = idx[split:]

    x_train = features[train_idx].to(device)
    y_train = labels[train_idx].to(device)
    x_val = features[val_idx].to(device)
    y_val = labels[val_idx].to(device)

    probe = nn.Linear(x_train.size(1), 1).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        probe.train()
        logits = probe(x_train)
        loss = loss_fn(logits, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        train_logits = probe(x_train)
        val_logits = probe(x_val)
        train_pred = (torch.sigmoid(train_logits) >= 0.5).float()
        val_pred = (torch.sigmoid(val_logits) >= 0.5).float()
        train_acc = (train_pred.eq(y_train).float().mean().item())
        val_acc = (val_pred.eq(y_val).float().mean().item())
        val_auc = compute_auc(y_val, torch.sigmoid(val_logits))

    return {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "val_auc": val_auc,
        "n_train": int(x_train.size(0)),
        "n_val": int(x_val.size(0)),
    }


def compute_auc(y_true, y_score):
    y_true = y_true.view(-1).cpu()
    y_score = y_score.view(-1).cpu()
    pos = (y_true == 1).sum().item()
    neg = (y_true == 0).sum().item()
    if pos == 0 or neg == 0:
        return 0.0

    # Rank-based AUC
    scores, order = torch.sort(y_score)
    y_true = y_true[order]
    rank_sum = torch.arange(1, len(y_true) + 1, dtype=torch.float32)[y_true == 1].sum().item()
    auc = (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def main():
    parser = argparse.ArgumentParser(description="Feature-probe leakage test")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--suite", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--probe-layer", type=str, default="final",
                        help="Feature layer to probe: final, early, mid, late, or block<N>")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    experiment_suites = load_config(args.config)
    if args.suite not in experiment_suites:
        raise ValueError(f"Suite '{args.suite}' not found in config")

    suite_cfg = experiment_suites[args.suite]
    model_suites = suite_cfg.get("model_suites", [])
    if not model_suites:
        raise ValueError("feature-probe suite must define model_suites")

    base_suite_name, base_suite = resolve_base_suite_info(experiment_suites, model_suites[0])
    dataset_name = base_suite.get("dataset", "cifar10")
    model_type = base_suite.get("model", "vit_tiny")

    data_cfg = load_config("configs/data.yaml")
    dataset_info = data_cfg[dataset_name]
    forget_class = suite_cfg.get("forget_class", base_suite.get("forget_class", 0))

    data_manager = DataManager()
    dataset = data_manager.load_dataset(
        dataset_name, "test",
        use_pretrained=True,
        apply_imagenet_norm=False
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    normalizer = create_imagenet_normalizer().to(device)

    results = {
        "suite": args.suite,
        "seed": args.seed,
        "dataset": dataset_name,
        "model": model_type,
        "forget_class": forget_class,
        "run_info": {
            "started_utc": datetime.utcnow().isoformat() + "Z",
            "argv": sys.argv,
            "probe_layer": args.probe_layer,
        },
        "models": {},
    }

    for model_suite in model_suites:
        model = build_model(model_type, dataset_info, device)

        if model_suite.startswith("base_"):
            ckpt_path = load_checkpoint_or_best("checkpoints/base", model_suite, args.seed)
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
        elif model_suite.startswith("oracle_"):
            ckpt_path = load_checkpoint_or_best("checkpoints/oracle", model_suite, args.seed)
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
        elif model_suite.startswith("benign_relearn_"):
            final_path = os.path.join("checkpoints/benign_relearn", f"{model_suite}_seed_{args.seed}_final.pt")
            best_path = os.path.join("checkpoints/benign_relearn", f"{model_suite}_seed_{args.seed}_best.pt")
            if os.path.exists(final_path):
                ckpt_path = final_path
            elif os.path.exists(best_path):
                ckpt_path = best_path
            else:
                print(f"Warning: benign relearn checkpoint not found: {final_path} (or {best_path}); skipping")
                continue
            checkpoint = torch.load(ckpt_path, map_location=device)
            state = checkpoint.get("model_state_dict", checkpoint)
            if all(not k.startswith("model.") for k in state.keys()):
                state = {f"model.{k}": v for k, v in state.items()}
            model.load_state_dict(state)
        elif model_suite.startswith("unlearn_"):
            spec = experiment_suites[model_suite]
            base_suite = spec.get("base_model_suite", None)
            if not base_suite:
                raise ValueError(f"Missing base_model_suite for {model_suite}")
            
            ckpt_path = load_checkpoint_or_best("checkpoints/base", base_suite, args.seed)
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            
            # Determine path to unlearned weights
            custom_path = spec.get("path", None)
            if custom_path:
                tgt_path = custom_path
            else:
                tgt_path = f"checkpoints/unlearn_lora/{model_suite}_seed_{args.seed}"
                
            # Check for Full FT (File or explicit model.pt)
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
                model.load_state_dict(ft_ckpt["model_state_dict"])
            else:
                print(f"[Info] Loading LoRA adapter from {tgt_path}")
                model = load_lora_adapter(model, tgt_path).to(device)
        else:
            continue

        features, labels = extract_features(
            model, normalizer, loader, device,
            forget_class=forget_class,
            max_samples_per_class=args.max_samples_per_class,
            probe_layer=args.probe_layer
        )
        if features is None:
            continue

        probe_stats = train_linear_probe(
            features,
            labels,
            seed=args.seed,
            device=device,
            epochs=args.epochs,
            lr=args.lr
        )

        results["models"][model_suite] = {
            "n_samples": int(features.size(0)),
            "feature_dim": int(features.size(1)),
            **probe_stats
        }

    ensure_dir("results/logs")
    layer_tag = args.probe_layer.replace(".", "p") if args.probe_layer else "final"
    out_path = f"results/logs/feature_probe_{args.suite}_seed_{args.seed}_layer_{layer_tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved feature-probe results to {out_path}")


if __name__ == "__main__":
    main()
