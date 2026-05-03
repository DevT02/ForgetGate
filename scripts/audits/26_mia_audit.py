"""
Membership Inference Attack (MIA) audit for unlearning checkpoints.

Reports loss-threshold and entropy-threshold MIA AUC against two reference
"unseen" sets:

  - val_heldout: the forget-class samples held out from base training
    (reconstructs the same seeded torch.randperm split used in
    scripts/1_train_base.py).
  - test:        the forget-class samples in the dataset's official test split.

A perfectly-unlearned model should yield AUC ~ 0.50 against both. A surviving
gap indicates residual memorization that clean forget accuracy alone misses.
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data import DataManager
from src.models.cnn import create_cnn_model
from src.models.normalize import create_imagenet_normalizer
from src.models.vit import create_vit_model
from src.utils import load_config, set_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_forget_class(suite_name: str) -> int:
    """Parse forget class from a suite name.

    Suite names use the convention `..._forgetN` where N is an integer (one or
    more digits). The previous implementation only consumed a single digit,
    misparsing `forget10` as `forget1`.
    """
    if "forget" not in suite_name:
        return 0
    tail = suite_name.split("forget", 1)[1]
    digits = ""
    for ch in tail:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits) if digits else 0


def reconstruct_base_train_val_indices(
    n_total: int, seed: int, val_ratio: float = 0.1
) -> Tuple[List[int], List[int]]:
    """Reproduce the (train, val) index split used by scripts/1_train_base.py.

    Must mirror that file's logic exactly:
        n_val = max(1, int(n_total * val_ratio))
        n_train = n_total - n_val
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n_total, generator=generator).tolist()
        train = perm[:n_train]
        val   = perm[n_train:]
    """
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator).tolist()
    return perm[:n_train], perm[n_train:]


def detect_adv_train(experiment_suites: Dict, base_suite: str) -> bool:
    base_cfg = experiment_suites.get(base_suite, {}) or {}
    return bool(
        base_cfg.get("training", {})
        .get("adv_train", {})
        .get("enabled", False)
    )


def build_model(suite_name: str, num_classes: int, model_config: Dict) -> nn.Module:
    is_resnet = "resnet" in suite_name or "cnn" in suite_name
    if is_resnet:
        return create_cnn_model(model_config["cnn"]["resnet18"], num_classes=num_classes)
    if "vit_small" in suite_name:
        return create_vit_model(model_config["vit"]["small"], num_classes=num_classes)
    return create_vit_model(model_config["vit"]["tiny"], num_classes=num_classes)


def load_state_dict_with_diagnostics(model: nn.Module, state_dict: Dict, label: str) -> None:
    result = model.load_state_dict(state_dict, strict=False)
    missing = len(result.missing_keys)
    unexpected = len(result.unexpected_keys)
    if missing or unexpected:
        print(
            f"  [{label}] load_state_dict: missing={missing}, unexpected={unexpected}"
        )
        if missing:
            print(f"    first missing: {result.missing_keys[:3]}")
        if unexpected:
            print(f"    first unexpected: {result.unexpected_keys[:3]}")


def find_checkpoint_dir(root: str, suite: str) -> Optional[str]:
    pattern = os.path.join(root, "checkpoints", "**", f"{suite}*")
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None


def load_unlearned_model(
    suite: str,
    seed: int,
    base_suite: str,
    checkpoint_dir: str,
    root: str,
    num_classes: int,
    model_config: Dict,
    device: torch.device,
) -> nn.Module:
    model = build_model(suite, num_classes, model_config)

    base_ckpt = os.path.join(root, "checkpoints", "base", f"{base_suite}_seed_{seed}_final.pt")
    if os.path.exists(base_ckpt):
        try:
            payload = torch.load(base_ckpt, map_location="cpu")
            sd = payload.get("model_state_dict", payload)
            load_state_dict_with_diagnostics(model, sd, "base")
        except Exception as e:
            print(f"  [base] load failed: {e}")

    from src.models.peft_lora import load_lora_adapter

    has_adapter_root = os.path.exists(os.path.join(checkpoint_dir, "adapter_model.safetensors"))
    has_adapter_nested = os.path.exists(
        os.path.join(checkpoint_dir, "final_model", "adapter_model.safetensors")
    )
    if has_adapter_root or has_adapter_nested:
        adapter_path = checkpoint_dir
        if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            adapter_path = os.path.join(checkpoint_dir, "final_model")
        model = load_lora_adapter(model, adapter_path)
    else:
        final_pt = os.path.join(checkpoint_dir, "final_model.pt")
        if os.path.exists(final_pt):
            payload = torch.load(final_pt, map_location="cpu")
            sd = payload.get("model_state_dict", payload)
            load_state_dict_with_diagnostics(model, sd, "final_model")

    return model.to(device)


def gather_metrics(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    losses: List[np.ndarray] = []
    entropies: List[np.ndarray] = []
    ce = nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        for batch in loader:
            images, labels = batch[0], batch[1]
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            losses.append(ce(logits, labels).cpu().numpy())
            probs = F.softmax(logits, dim=-1)
            ent = -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=-1)
            entropies.append(ent.cpu().numpy())
    return np.concatenate(losses), np.concatenate(entropies)


def auc_pair(seen_loss, seen_ent, unseen_loss, unseen_ent) -> Dict[str, float]:
    y_true = np.concatenate([np.ones_like(seen_loss), np.zeros_like(unseen_loss)])
    auc_loss = float(
        roc_auc_score(y_true, np.concatenate([-seen_loss, -unseen_loss]))
    )
    auc_ent = float(
        roc_auc_score(y_true, np.concatenate([-seen_ent, -unseen_ent]))
    )
    return {
        "auc_loss": auc_loss,
        "auc_entropy": auc_ent,
        "memorization_gap": max(auc_loss, auc_ent) - 0.5,
        "n_seen": int(len(seen_loss)),
        "n_unseen": int(len(unseen_loss)),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_mia(
    suite: str,
    *,
    config_path: str = "configs/experiment_suites.yaml",
    batch_size: int = 128,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    val_ratio: float = 0.1,
) -> Dict:
    """Run MIA audit for a single suite. Returns the results dict that would
    also be written to disk by the CLI entry point."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = find_checkpoint_dir(root, suite)
    if checkpoint_dir is None:
        raise FileNotFoundError(f"No checkpoint directory found for suite: {suite}")

    if seed is None:
        seed = 42
        if "_seed_" in checkpoint_dir:
            try:
                seed = int(checkpoint_dir.split("_seed_")[-1].split(os.sep)[0])
            except ValueError:
                pass
    set_seed(seed)

    forget_class = parse_forget_class(suite)
    dataset_name = "cifar100" if "cifar100" in suite else "cifar10"
    num_classes = 100 if dataset_name == "cifar100" else 10

    experiment_suites = load_config(config_path)
    suite_cfg = experiment_suites.get(suite, {}) or {}
    base_suite = suite_cfg.get("base_model_suite", suite)
    is_adv_base = detect_adv_train(experiment_suites, base_suite)

    model_config = load_config("configs/model.yaml")
    model = load_unlearned_model(
        suite=suite,
        seed=seed,
        base_suite=base_suite,
        checkpoint_dir=checkpoint_dir,
        root=root,
        num_classes=num_classes,
        model_config=model_config,
        device=device,
    )

    if is_adv_base:
        # Adv-trained base expects un-normalized [0,1] inputs and applies
        # normalization inside its forward pass. Ensure the dataloader does
        # not double-normalize.
        normalizer = create_imagenet_normalizer().to(device)
        model = nn.Sequential(normalizer, model).to(device)

    apply_norm = not is_adv_base

    data_manager = DataManager()
    train_full = data_manager.load_dataset(
        dataset_name, split="train", use_pretrained=True, apply_imagenet_norm=apply_norm
    )
    n_total = len(train_full)
    train_idx, val_idx = reconstruct_base_train_val_indices(n_total, seed, val_ratio)

    base_targets = train_full.targets if hasattr(train_full, "targets") else None
    if base_targets is None:
        raise RuntimeError(
            "Underlying dataset does not expose .targets; cannot reconstruct splits"
        )

    train_idx_set = set(train_idx)
    val_idx_set = set(val_idx)

    forget_train_indices = [
        i for i, lbl in enumerate(base_targets) if int(lbl) == forget_class and i in train_idx_set
    ]
    forget_val_indices = [
        i for i, lbl in enumerate(base_targets) if int(lbl) == forget_class and i in val_idx_set
    ]

    seen_set = Subset(train_full, forget_train_indices)
    val_heldout_set = Subset(train_full, forget_val_indices)
    test_set = data_manager.load_dataset(
        dataset_name,
        split="test",
        include_classes=[forget_class],
        use_pretrained=True,
        apply_imagenet_norm=apply_norm,
    )

    seen_loader = DataLoader(seen_set, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_heldout_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    seen_loss, seen_ent = gather_metrics(model, seen_loader, device)
    val_loss, val_ent = gather_metrics(model, val_loader, device)
    test_loss, test_ent = gather_metrics(model, test_loader, device)

    vs_val = auc_pair(seen_loss, seen_ent, val_loss, val_ent)
    vs_test = auc_pair(seen_loss, seen_ent, test_loss, test_ent)

    out = {
        "suite": suite,
        "base_suite": base_suite,
        "seed": seed,
        "forget_class": forget_class,
        "dataset": dataset_name,
        "is_adv_base": is_adv_base,
        "vs_val_heldout": vs_val,
        "vs_test": vs_test,
    }

    out_dir = os.path.join(root, "results", "analysis", "metrics")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"mia_audit_{suite}_seed_{seed}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[mia] {suite} seed={seed} -> {out_path}")
    print(f"  vs val:  loss-AUC={vs_val['auc_loss']:.4f}  ent-AUC={vs_val['auc_entropy']:.4f}")
    print(f"  vs test: loss-AUC={vs_test['auc_loss']:.4f}  ent-AUC={vs_test['auc_entropy']:.4f}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Membership Inference Attack (MIA) Audit")
    parser.add_argument("--config", type=str, default="configs/experiment_suites.yaml")
    parser.add_argument("--suite", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed (defaults to the seed parsed from the checkpoint dir name)")
    args = parser.parse_args()
    run_mia(
        args.suite,
        config_path=args.config,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
