#!/usr/bin/env python3
"""
Standalone trainer for the SmoothedMarginUnlearning objective (Theorem 4
of `results/analysis/figures/theory_appendix.tex`).

Why standalone: the main `scripts/train/2_train_unlearning_lora.py` has
accumulated keyword arguments (gradient noise, projection, surgery,
SCRUB rewind, etc.) that `src/unlearning/trainer.create_unlearning_trainer`
does not currently accept — a pre-existing version drift. Rather than
extend the older trainer signature on the same commit as a theory result,
we wire the smoothed-margin objective through a minimal, transparent
training loop that is exactly what Theorem 4 needs and nothing more.

The output adapter is saved at the same path the audits already look for:
  checkpoints/unlearn_lora/{suite_tag}_seed_{seed}/adapter_*

Default suite tag: unlearn_smoothed_margin_vit_cifar10_forget0
"""

import argparse
import json
import os
import sys
from typing import Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.data import DataManager
from src.models.normalize import create_imagenet_normalizer
from src.models.peft_lora import (
    apply_lora_to_model,
    create_lora_config,
    save_lora_adapter,
)
from src.models.vit import create_vit_model
from src.unlearning.objectives import SmoothedMarginUnlearning
from src.utils import get_device, load_config, set_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_vit(num_classes: int, vit_size: str = "tiny") -> nn.Module:
    model_cfg = load_config("configs/model.yaml")
    vit_cfg = dict(model_cfg["vit"][vit_size])
    vit_cfg["pretrained"] = False
    return create_vit_model(vit_cfg, num_classes=num_classes)


def _load_base(checkpoint_path: str, num_classes: int, device: torch.device,
               vit_size: str = "tiny") -> nn.Module:
    model = _build_vit(num_classes, vit_size).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    return model


def _build_split_loaders(
    dataset_name: str,
    forget_class: int,
    batch_size: int,
    num_workers: int,
    max_retain: int,
    max_forget: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Returns (forget_loader, retain_loader, val_loader)."""
    dm = DataManager()
    train_kwargs = {"use_pretrained": True, "apply_imagenet_norm": False}
    val_kwargs = train_kwargs

    forget_ds = dm.load_dataset(dataset_name, "train", include_classes=[forget_class], **train_kwargs)
    retain_ds = dm.load_dataset(dataset_name, "train", exclude_classes=[forget_class], **train_kwargs)
    val_ds = dm.load_dataset(dataset_name, "test", **val_kwargs)

    if max_forget > 0 and len(forget_ds) > max_forget:
        forget_ds = Subset(forget_ds, list(range(max_forget)))
    if max_retain > 0 and len(retain_ds) > max_retain:
        retain_ds = Subset(retain_ds, list(range(max_retain)))

    forget_loader = DataLoader(forget_ds, batch_size=min(batch_size, len(forget_ds)),
                               shuffle=True, num_workers=num_workers, pin_memory=True,
                               drop_last=False)
    retain_loader = DataLoader(retain_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=num_workers,
                            pin_memory=True)
    return forget_loader, retain_loader, val_loader


@torch.no_grad()
def _evaluate(model: nn.Module, normalizer: nn.Module, loader: DataLoader,
              forget_class: int, device: torch.device) -> dict:
    model.eval()
    f_correct = f_total = r_correct = r_total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(normalizer(x)).argmax(dim=1)
        f_mask = (y == forget_class)
        r_mask = ~f_mask
        f_correct += int((preds[f_mask] == y[f_mask]).sum().item())
        f_total += int(f_mask.sum().item())
        r_correct += int((preds[r_mask] == y[r_mask]).sum().item())
        r_total += int(r_mask.sum().item())
    return {
        "forget_acc": f_correct / max(f_total, 1),
        "retain_acc": r_correct / max(r_total, 1),
        "n_forget": f_total,
        "n_retain": r_total,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-suite", default="base_vit_cifar10",
                   help="Base model suite name (provides the dataset and the checkpoint).")
    p.add_argument("--arch", default="tiny", choices=["tiny", "small", "base"],
                   help="ViT size to build (must match the base checkpoint).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--forget-class", type=int, default=0)
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--max-retain", type=int, default=0, help="Cap retain set (0 == all).")
    p.add_argument("--max-forget", type=int, default=0, help="Cap forget set (0 == all).")
    # smoothed-margin hyperparameters (theory_appendix.tex Thm 4)
    p.add_argument("--sigma", type=float, default=0.10)
    p.add_argument("--n-noise", type=int, default=4)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--forget-weight", type=float, default=1.0)
    p.add_argument("--retain-weight", type=float, default=1.0)
    # output
    p.add_argument("--suite-tag", default="unlearn_smoothed_margin_vit_cifar10_forget0",
                   help="Used to name the output adapter dir.")
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}, seed: {args.seed}")

    data_cfg = load_config("configs/data.yaml")
    num_classes = int(data_cfg[args.dataset]["num_classes"])

    base_path = f"checkpoints/base/{args.base_suite}_seed_{args.seed}_final.pt"
    if not os.path.exists(base_path):
        base_path = f"checkpoints/base/{args.base_suite}_seed_{args.seed}_best.pt"
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base checkpoint missing: {base_path}")
    print(f"Loading base from {base_path}")

    base = _load_base(base_path, num_classes, device, vit_size=args.arch)

    # Apply LoRA to the base
    lora_cfg = create_lora_config(
        r=args.lora_rank,
        target_modules=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
    )
    model = apply_lora_to_model(base, lora_cfg)
    model.to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA rank={args.lora_rank} | trainable params: {n_trainable:,}")

    normalizer = create_imagenet_normalizer().to(device)

    # The objective re-runs the model on noisy inputs internally; it expects
    # the model to take *normalized* inputs (so noise sigma is in the same
    # space as the audit). Wrap the LoRA model so the objective sees a
    # callable that handles normalization.
    class _NormalizedModel(nn.Module):
        def __init__(self, inner, norm):
            super().__init__()
            self.inner = inner
            self.norm = norm
        def forward(self, x):
            return self.inner(self.norm(x))
    wrapped = _NormalizedModel(model, normalizer).to(device)

    objective = SmoothedMarginUnlearning(
        forget_class=args.forget_class,
        num_classes=num_classes,
        sigma=args.sigma,
        n_noise=args.n_noise,
        gamma=args.gamma,
        forget_weight=args.forget_weight,
        retain_weight=args.retain_weight,
        clip_min=0.0,
        clip_max=1.0,
    )
    objective.set_model(wrapped)

    forget_loader, retain_loader, val_loader = _build_split_loaders(
        dataset_name=args.dataset,
        forget_class=args.forget_class,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_retain=args.max_retain,
        max_forget=args.max_forget,
    )
    print(f"forget train n={len(forget_loader.dataset)} retain train n={len(retain_loader.dataset)} "
          f"val n={len(val_loader.dataset)}")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Pre-eval
    m0 = _evaluate(model, normalizer, val_loader, args.forget_class, device)
    print(f"[epoch -1 / before] forget_acc={m0['forget_acc']:.4f} retain_acc={m0['retain_acc']:.4f}")

    history = [{"epoch": -1, **m0}]

    for epoch in range(args.epochs):
        model.train()
        wrapped.train()
        forget_iter = iter(forget_loader)
        running = 0.0
        n_steps = 0
        for retain_x, retain_y in retain_loader:
            try:
                forget_x, forget_y = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forget_loader)
                forget_x, forget_y = next(forget_iter)

            x = torch.cat([forget_x, retain_x], dim=0).to(device)
            y = torch.cat([forget_y, retain_y], dim=0).to(device)
            logits = wrapped(x)  # normalized + LoRA forward
            loss = objective(logits, y, inputs=x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_steps += 1
        scheduler.step()

        m = _evaluate(model, normalizer, val_loader, args.forget_class, device)
        avg_loss = running / max(n_steps, 1)
        print(f"[epoch {epoch:02d}] loss={avg_loss:.4f} forget_acc={m['forget_acc']:.4f} "
              f"retain_acc={m['retain_acc']:.4f}")
        history.append({"epoch": epoch, "train_loss": avg_loss, **m})

    # Save LoRA adapter where the audits look for it
    out_dir = f"checkpoints/unlearn_lora/{args.suite_tag}_seed_{args.seed}"
    os.makedirs(out_dir, exist_ok=True)
    save_lora_adapter(model, out_dir)
    print(f"\nSaved LoRA adapter to {out_dir}")

    # Save training history
    os.makedirs("results/logs", exist_ok=True)
    hist_path = f"results/logs/{args.suite_tag}_seed_{args.seed}_history.json"
    with open(hist_path, "w") as f:
        json.dump({
            "args": vars(args),
            "history": history,
            "base_checkpoint": base_path,
        }, f, indent=2)
    print(f"Saved training history -> {hist_path}")


if __name__ == "__main__":
    main()
