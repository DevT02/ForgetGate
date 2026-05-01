#!/usr/bin/env python3
"""
Conditional small-area patch attack audit.

Learns image-conditioned patch content and coarse-grid placement, then evaluates
whether a bounded patch blend can recover forget samples while minimally
affecting retain samples.
"""

import argparse
import importlib.util
import itertools
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.normalize import create_imagenet_normalizer
from src.utils import get_device, load_config, save_dict_to_json, set_seed


_AUDIT_PATH = os.path.join(os.path.dirname(__file__), "17_recovery_radius_audit.py")
_AUDIT_SPEC = importlib.util.spec_from_file_location("recovery_radius_audit", _AUDIT_PATH)
if _AUDIT_SPEC is None or _AUDIT_SPEC.loader is None:
    raise ImportError(f"Unable to load recovery audit helpers from {_AUDIT_PATH}")
_AUDIT = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(_AUDIT)

DEFAULT_MODEL_SUITES = list(getattr(_AUDIT, "DEFAULT_MODEL_SUITES"))
_load_model = getattr(_AUDIT, "_load_model")
_build_loader = getattr(_AUDIT, "_build_loader")
_predict = getattr(_AUDIT, "_predict")
_margin_objective = getattr(_AUDIT, "_margin_objective")
_suite_tag = getattr(_AUDIT, "_suite_tag")


class PatchAttackNet(nn.Module):
    def __init__(self, image_size: int, patch_size: int, stride: int, width: int = 64):
        super().__init__()
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.positions = self._build_positions()
        self.num_positions = len(self.positions)

        self.backbone = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        hidden = width * 4 * 4
        self.patch_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 3 * self.patch_size * self.patch_size),
        )
        self.loc_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(inplace=True),
            nn.Linear(hidden // 2, self.num_positions),
        )

    def _build_positions(self) -> List[Tuple[int, int]]:
        positions: List[Tuple[int, int]] = []
        max_y = self.image_size - self.patch_size
        max_x = self.image_size - self.patch_size
        ys = list(range(0, max_y + 1, self.stride))
        xs = list(range(0, max_x + 1, self.stride))
        if ys[-1] != max_y:
            ys.append(max_y)
        if xs[-1] != max_x:
            xs.append(max_x)
        for y in ys:
            for x in xs:
                positions.append((y, x))
        return positions

    def forward(self, x_small: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x_small)
        patch = self.patch_head(feat).view(-1, 3, self.patch_size, self.patch_size)
        patch = torch.sigmoid(patch)
        loc_logits = self.loc_head(feat)
        return patch, loc_logits


def _default_model_suites(experiment_suites: Dict) -> List[str]:
    return [suite for suite in DEFAULT_MODEL_SUITES if suite in experiment_suites]


def _reshuffle_loader(loader, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(loader.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def _soft_patch_overlay(
    inputs: torch.Tensor,
    patch: torch.Tensor,
    loc_logits: torch.Tensor,
    positions: Sequence[Tuple[int, int]],
    image_size: int,
    patch_size: int,
    alpha: float,
    hard: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if hard:
        loc_probs = F.one_hot(loc_logits.argmax(dim=1), num_classes=len(positions)).float()
    else:
        loc_probs = F.softmax(loc_logits, dim=1)

    canvas = torch.zeros_like(inputs)
    mask = torch.zeros(inputs.size(0), 1, image_size, image_size, device=inputs.device, dtype=inputs.dtype)
    ones_patch = torch.ones(inputs.size(0), 1, patch_size, patch_size, device=inputs.device, dtype=inputs.dtype)
    for idx, (y, x) in enumerate(positions):
        weight = loc_probs[:, idx].view(-1, 1, 1, 1)
        padded_patch = F.pad(
            patch,
            (x, image_size - x - patch_size, y, image_size - y - patch_size),
        )
        padded_mask = F.pad(
            ones_patch,
            (x, image_size - x - patch_size, y, image_size - y - patch_size),
        )
        canvas = canvas + weight * padded_patch
        mask = mask + weight * padded_mask

    mask = mask.clamp(0.0, 1.0)
    adv = torch.clamp(inputs + (alpha * mask * (canvas - inputs)), 0.0, 1.0)
    return adv, mask, loc_probs


def _retain_target_calibration_loss(
    clean_logits: torch.Tensor,
    adv_logits: torch.Tensor,
    target_class: int,
    logit_buffer: float,
) -> torch.Tensor:
    clean_target = clean_logits[:, target_class]
    adv_target = adv_logits[:, target_class]
    target_delta = adv_target - clean_target
    return F.softplus(target_delta - logit_buffer).mean()


def _retain_clean_anchor_loss(
    clean_logits: torch.Tensor,
    adv_logits: torch.Tensor,
    target_class: int,
    margin_buffer: float,
    kl_temperature: float,
) -> torch.Tensor:
    clean_pred = clean_logits.argmax(dim=1)
    clean_adv_logit = adv_logits.gather(1, clean_pred.view(-1, 1)).squeeze(1)
    target_adv_logit = adv_logits[:, target_class]
    clean_margin_penalty = F.softplus((target_adv_logit - clean_adv_logit) + margin_buffer).mean()

    temp = max(float(kl_temperature), 1e-6)
    clean_probs = F.softmax(clean_logits / temp, dim=1)
    adv_log_probs = F.log_softmax(adv_logits / temp, dim=1)
    kl = F.kl_div(adv_log_probs, clean_probs, reduction="batchmean") * (temp * temp)
    return clean_margin_penalty + kl


def _patch_area_penalty(mask: torch.Tensor, patch_fraction: float) -> torch.Tensor:
    return F.relu(mask.mean() - patch_fraction)


def _train_patch_attack(
    attack_net: PatchAttackNet,
    model: nn.Module,
    normalizer: nn.Module,
    forget_loader,
    retain_loader,
    target_class: int,
    image_size: int,
    patch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    retain_weight: float,
    retain_target_calib_weight: float,
    retain_target_calib_buffer: float,
    retain_clean_anchor_weight: float,
    retain_clean_anchor_buffer: float,
    retain_clean_anchor_temp: float,
    forget_ce_weight: float,
    forget_margin_weight: float,
    patch_area_weight: float,
) -> List[Dict[str, float]]:
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    optimizer = torch.optim.AdamW(attack_net.parameters(), lr=lr, weight_decay=weight_decay)
    history: List[Dict[str, float]] = []
    retain_iter = itertools.cycle(retain_loader) if retain_loader is not None else None
    patch_fraction = (patch_size * patch_size) / float(image_size * image_size)

    for epoch_idx in range(epochs):
        epoch_loss = 0.0
        epoch_forget_margin = 0.0
        epoch_retain_margin = 0.0
        epoch_area = 0.0
        n_steps = 0

        for forget_inputs, _ in forget_loader:
            forget_inputs = forget_inputs.to(next(attack_net.parameters()).device)
            retain_inputs = None
            if retain_iter is not None:
                retain_inputs, _ = next(retain_iter)
                retain_inputs = retain_inputs.to(forget_inputs.device)

            x_small = F.interpolate(forget_inputs, size=(32, 32), mode="bilinear", align_corners=False)
            patch, loc_logits = attack_net(x_small)
            forget_adv, forget_mask, _ = _soft_patch_overlay(
                forget_inputs,
                patch,
                loc_logits,
                attack_net.positions,
                image_size=image_size,
                patch_size=patch_size,
                alpha=1.0,
                hard=False,
            )
            forget_logits = model(normalizer(forget_adv))
            forget_margin = _margin_objective(forget_logits, target_class)
            target_labels = torch.full(
                (forget_inputs.size(0),),
                target_class,
                dtype=torch.long,
                device=forget_inputs.device,
            )
            forget_ce = F.cross_entropy(forget_logits, target_labels)
            forget_loss = (forget_ce_weight * forget_ce) - (forget_margin_weight * forget_margin.mean())

            retain_loss = forget_inputs.new_zeros(())
            retain_margin_mean = forget_inputs.new_zeros(())
            if retain_inputs is not None and retain_weight > 0.0:
                retain_small = F.interpolate(retain_inputs, size=(32, 32), mode="bilinear", align_corners=False)
                retain_patch, retain_loc_logits = attack_net(retain_small)
                retain_adv, retain_mask, _ = _soft_patch_overlay(
                    retain_inputs,
                    retain_patch,
                    retain_loc_logits,
                    attack_net.positions,
                    image_size=image_size,
                    patch_size=patch_size,
                    alpha=1.0,
                    hard=False,
                )
                with torch.no_grad():
                    retain_clean_logits = model(normalizer(retain_inputs)).detach()
                retain_logits = model(normalizer(retain_adv))
                retain_margin = _margin_objective(retain_logits, target_class)
                retain_margin_mean = retain_margin.mean().detach()
                retain_loss = F.softplus(retain_margin).mean()
                if retain_target_calib_weight > 0.0:
                    retain_loss = retain_loss + (
                        retain_target_calib_weight
                        * _retain_target_calibration_loss(
                            clean_logits=retain_clean_logits,
                            adv_logits=retain_logits,
                            target_class=target_class,
                            logit_buffer=retain_target_calib_buffer,
                        )
                    )
                if retain_clean_anchor_weight > 0.0:
                    retain_loss = retain_loss + (
                        retain_clean_anchor_weight
                        * _retain_clean_anchor_loss(
                            clean_logits=retain_clean_logits,
                            adv_logits=retain_logits,
                            target_class=target_class,
                            margin_buffer=retain_clean_anchor_buffer,
                            kl_temperature=retain_clean_anchor_temp,
                        )
                    )
                area_penalty = 0.5 * (
                    _patch_area_penalty(forget_mask, patch_fraction) + _patch_area_penalty(retain_mask, patch_fraction)
                )
            else:
                area_penalty = _patch_area_penalty(forget_mask, patch_fraction)

            loss = forget_loss + (retain_weight * retain_loss) + (patch_area_weight * area_penalty)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(attack_net.parameters(), max_norm=5.0)
            optimizer.step()

            n_steps += 1
            epoch_loss += float(loss.item())
            epoch_forget_margin += float(forget_margin.mean().item())
            epoch_retain_margin += float(retain_margin_mean.item())
            epoch_area += float(forget_mask.mean().item())

        history.append(
            {
                "epoch": float(epoch_idx),
                "loss": epoch_loss / max(n_steps, 1),
                "forget_margin": epoch_forget_margin / max(n_steps, 1),
                "retain_margin": epoch_retain_margin / max(n_steps, 1),
                "mask_area_mean": epoch_area / max(n_steps, 1),
            }
        )
        print(
            f"Epoch {epoch_idx}: "
            f"loss={history[-1]['loss']:.4f} "
            f"forget_margin={history[-1]['forget_margin']:.4f} "
            f"retain_margin={history[-1]['retain_margin']:.4f} "
            f"mask_area={history[-1]['mask_area_mean']:.4f}"
        )

    attack_net.eval()
    return history


def _patch_recovery_search(
    model: nn.Module,
    normalizer: nn.Module,
    attack_net: PatchAttackNet,
    inputs: torch.Tensor,
    target_class: int,
    search_steps: int,
    image_size: int,
    patch_size: int,
) -> Dict[str, Optional[float]]:
    clean_logits, clean_preds = _predict(model, normalizer, inputs)
    clean_target_prob = float(F.softmax(clean_logits, dim=1)[0, target_class].item())
    clean_pred_int = int(clean_preds.item())
    clean_success = bool((clean_preds == target_class).all().item())
    if clean_success:
        return {
            "success": True,
            "clean_success": True,
            "radius": 0.0,
            "clean_target_prob": clean_target_prob,
            "adv_target_prob": clean_target_prob,
            "clean_pred": clean_pred_int,
            "adv_pred": clean_pred_int,
        }

    with torch.no_grad():
        x_small = F.interpolate(inputs, size=(32, 32), mode="bilinear", align_corners=False)
        patch, loc_logits = attack_net(x_small)

    low = 0.0
    high = None
    high_logits = None
    probe_schedule = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]
    best_mask_area = None
    best_location = None

    for alpha in probe_schedule:
        with torch.no_grad():
            adv_inputs, mask, loc_probs = _soft_patch_overlay(
                inputs,
                patch,
                loc_logits,
                attack_net.positions,
                image_size=image_size,
                patch_size=patch_size,
                alpha=float(alpha),
                hard=True,
            )
            adv_logits, adv_preds = _predict(model, normalizer, adv_inputs)
            if best_mask_area is None:
                best_mask_area = float(mask.mean().item())
                best_location = int(loc_probs.argmax(dim=1).item())
        if bool((adv_preds == target_class).all().item()):
            high = float(alpha)
            high_logits = adv_logits
            break
        low = float(alpha)

    if high is None:
        with torch.no_grad():
            adv_inputs, mask, loc_probs = _soft_patch_overlay(
                inputs,
                patch,
                loc_logits,
                attack_net.positions,
                image_size=image_size,
                patch_size=patch_size,
                alpha=1.0,
                hard=True,
            )
            adv_logits, adv_preds = _predict(model, normalizer, adv_inputs)
            best_mask_area = float(mask.mean().item())
            best_location = int(loc_probs.argmax(dim=1).item())
            if not bool((adv_preds == target_class).all().item()):
                adv_target_prob = float(F.softmax(adv_logits, dim=1)[0, target_class].item())
                return {
                    "success": False,
                    "clean_success": False,
                    "radius": None,
                    "clean_target_prob": clean_target_prob,
                    "adv_target_prob": adv_target_prob,
                    "clean_pred": clean_pred_int,
                    "adv_pred": int(adv_preds.item()),
                    "mask_area": best_mask_area,
                    "location_index": best_location,
                }
            high = 1.0
            high_logits = adv_logits

    for _ in range(search_steps):
        mid = 0.5 * (low + high)
        with torch.no_grad():
            adv_inputs, _mask, _loc_probs = _soft_patch_overlay(
                inputs,
                patch,
                loc_logits,
                attack_net.positions,
                image_size=image_size,
                patch_size=patch_size,
                alpha=float(mid),
                hard=True,
            )
            adv_logits, adv_preds = _predict(model, normalizer, adv_inputs)
        if bool((adv_preds == target_class).all().item()):
            high = float(mid)
            high_logits = adv_logits
        else:
            low = float(mid)

    if high_logits is None:
        raise RuntimeError("Patch binary search ended without a successful state.")

    adv_probs = F.softmax(high_logits, dim=1)
    return {
        "success": True,
        "clean_success": False,
        "radius": float(high),
        "clean_target_prob": clean_target_prob,
        "adv_target_prob": float(adv_probs[0, target_class].item()),
        "clean_pred": clean_pred_int,
        "adv_pred": int(high_logits.argmax(dim=1).item()),
        "mask_area": best_mask_area,
        "location_index": best_location,
    }


def _summarize_radius(records: Sequence[Dict[str, Optional[float]]]) -> Dict[str, float]:
    radii = [float(r["radius"]) for r in records if r["radius"] is not None]
    success_rate = (sum(1 for r in records if r["success"]) / len(records)) if records else 0.0
    clean_success_rate = (sum(1 for r in records if r["clean_success"]) / len(records)) if records else 0.0
    target_prob_gain = [float(r["adv_target_prob"] - r["clean_target_prob"]) for r in records]
    mask_areas = [float(r["mask_area"]) for r in records if r.get("mask_area") is not None]
    clean_correct = [float(r.get("clean_correct", 0.0)) for r in records]
    adv_correct = [float(r.get("adv_correct", 0.0)) for r in records]

    def _stat(vals: Sequence[float], fn: str) -> float:
        if not vals:
            return float("nan")
        tensor = torch.tensor(list(vals), dtype=torch.float32)
        if fn == "mean":
            return float(tensor.mean().item())
        if fn == "median":
            return float(tensor.median().item())
        if fn == "min":
            return float(tensor.min().item())
        if fn == "max":
            return float(tensor.max().item())
        if fn == "p25":
            return float(torch.quantile(tensor, 0.25).item())
        if fn == "p75":
            return float(torch.quantile(tensor, 0.75).item())
        raise ValueError(fn)

    return {
        "n_samples": len(records),
        "success_rate": success_rate,
        "clean_success_rate": clean_success_rate,
        "median_radius": _stat(radii, "median"),
        "mean_radius": _stat(radii, "mean"),
        "min_radius": _stat(radii, "min"),
        "max_radius": _stat(radii, "max"),
        "p25_radius": _stat(radii, "p25"),
        "p75_radius": _stat(radii, "p75"),
        "mean_target_prob_gain": _stat(target_prob_gain, "mean"),
        "median_target_prob_gain": _stat(target_prob_gain, "median"),
        "mean_mask_area": _stat(mask_areas, "mean"),
        "clean_accuracy": _stat(clean_correct, "mean"),
        "attacked_accuracy": _stat(adv_correct, "mean"),
        "accuracy_drop": _stat(clean_correct, "mean") - _stat(adv_correct, "mean"),
    }


def main():
    parser = argparse.ArgumentParser(description="Conditional patch attack audit")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--train-split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--eval-split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model-suites", nargs="*", default=None)
    parser.add_argument("--max-train-forget-samples", type=int, default=512)
    parser.add_argument("--max-train-retain-samples", type=int, default=512)
    parser.add_argument("--max-eval-forget-samples", type=int, default=32)
    parser.add_argument("--max-eval-retain-samples", type=int, default=32)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--patch-stride", type=int, default=32)
    parser.add_argument("--generator-width", type=int, default=64)
    parser.add_argument("--retain-weight", type=float, default=1.0)
    parser.add_argument("--retain-target-calib-weight", type=float, default=1.0)
    parser.add_argument("--retain-target-calib-buffer", type=float, default=0.0)
    parser.add_argument("--retain-clean-anchor-weight", type=float, default=1.0)
    parser.add_argument("--retain-clean-anchor-buffer", type=float, default=0.0)
    parser.add_argument("--retain-clean-anchor-temp", type=float, default=1.0)
    parser.add_argument("--forget-ce-weight", type=float, default=1.0)
    parser.add_argument("--forget-margin-weight", type=float, default=0.75)
    parser.add_argument("--patch-area-weight", type=float, default=10.0)
    parser.add_argument("--search-steps", type=int, default=6)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    experiment_suites = load_config(args.config)
    model_suites = args.model_suites or _default_model_suites(experiment_suites)
    if not model_suites:
        raise ValueError("No model suites provided and no default suites found in config")
    if int(args.eval_batch_size) != 1:
        raise ValueError("Patch audit is defined per sample. Use --eval-batch-size 1.")

    first_suite_cfg = experiment_suites[model_suites[0]]
    base_suite_name = first_suite_cfg["base_model_suite"]
    base_suite_cfg = experiment_suites[base_suite_name]
    dataset_name = base_suite_cfg["dataset"]
    model_type = base_suite_cfg["model"]
    forget_class = int(first_suite_cfg.get("unlearning", {}).get("forget_class", first_suite_cfg.get("forget_class", 0)))
    normalizer = create_imagenet_normalizer().to(device)
    image_size = 224

    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_type}")
    print(f"Forget class: {forget_class}")
    print(f"Model suites: {model_suites}")
    print(f"Patch size: {args.patch_size}")
    print(f"Patch stride: {args.patch_stride}")

    results = {
        "meta": {
            "seed": int(args.seed),
            "train_split": args.train_split,
            "eval_split": args.eval_split,
            "dataset": dataset_name,
            "model": model_type,
            "forget_class": int(forget_class),
            "model_suites": model_suites,
            "attack_family": "conditional_patch",
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "patch_size": int(args.patch_size),
            "patch_stride": int(args.patch_stride),
            "generator_width": int(args.generator_width),
            "retain_weight": float(args.retain_weight),
            "retain_target_calib_weight": float(args.retain_target_calib_weight),
            "retain_clean_anchor_weight": float(args.retain_clean_anchor_weight),
            "patch_area_weight": float(args.patch_area_weight),
        },
        "forget_recovery": {},
        "retain_control": {},
        "training": {},
    }

    for suite_name in model_suites:
        print(f"\nTraining conditional patch attacker for {suite_name}...")
        model, suite_dataset, suite_model = _load_model(experiment_suites, suite_name, args.seed, device)
        if suite_dataset != dataset_name or suite_model != model_type:
            raise ValueError(f"Suite {suite_name} resolved to {suite_dataset}/{suite_model}, expected {dataset_name}/{model_type}")
        model.eval()

        train_forget_loader = _reshuffle_loader(
            _build_loader(
                dataset_name=dataset_name,
                split=args.train_split,
                batch_size=args.train_batch_size,
                num_workers=args.num_workers,
                forget_class=forget_class,
                retain=False,
                max_samples=args.max_train_forget_samples,
            ),
            batch_size=int(args.train_batch_size),
            num_workers=int(args.num_workers),
        )
        train_retain_loader = _reshuffle_loader(
            _build_loader(
                dataset_name=dataset_name,
                split=args.train_split,
                batch_size=args.train_batch_size,
                num_workers=args.num_workers,
                forget_class=forget_class,
                retain=True,
                max_samples=args.max_train_retain_samples,
            ),
            batch_size=int(args.train_batch_size),
            num_workers=int(args.num_workers),
        )
        eval_forget_loader = _build_loader(
            dataset_name=dataset_name,
            split=args.eval_split,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            forget_class=forget_class,
            retain=False,
            max_samples=args.max_eval_forget_samples,
        )
        eval_retain_loader = _build_loader(
            dataset_name=dataset_name,
            split=args.eval_split,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            forget_class=forget_class,
            retain=True,
            max_samples=args.max_eval_retain_samples,
        )

        attack_net = PatchAttackNet(
            image_size=image_size,
            patch_size=int(args.patch_size),
            stride=int(args.patch_stride),
            width=int(args.generator_width),
        ).to(device)

        history = _train_patch_attack(
            attack_net=attack_net,
            model=model,
            normalizer=normalizer,
            forget_loader=train_forget_loader,
            retain_loader=train_retain_loader,
            target_class=forget_class,
            image_size=image_size,
            patch_size=int(args.patch_size),
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            retain_weight=float(args.retain_weight),
            retain_target_calib_weight=float(args.retain_target_calib_weight),
            retain_target_calib_buffer=float(args.retain_target_calib_buffer),
            retain_clean_anchor_weight=float(args.retain_clean_anchor_weight),
            retain_clean_anchor_buffer=float(args.retain_clean_anchor_buffer),
            retain_clean_anchor_temp=float(args.retain_clean_anchor_temp),
            forget_ce_weight=float(args.forget_ce_weight),
            forget_margin_weight=float(args.forget_margin_weight),
            patch_area_weight=float(args.patch_area_weight),
        )

        ckpt_dir = os.path.join("checkpoints", "conditional_attack")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"conditional_patch_{suite_name}_seed_{args.seed}.pt")
        torch.save({"state_dict": attack_net.state_dict(), "training_history": history, "meta": results["meta"]}, ckpt_path)

        print(f"Evaluating conditional patch attacker for {suite_name}...")
        forget_records: List[Dict[str, Optional[float]]] = []
        for idx, (inputs, labels) in enumerate(eval_forget_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            record = _patch_recovery_search(
                model=model,
                normalizer=normalizer,
                attack_net=attack_net,
                inputs=inputs,
                target_class=forget_class,
                search_steps=int(args.search_steps),
                image_size=image_size,
                patch_size=int(args.patch_size),
            )
            record["sample_index"] = idx
            record["label"] = int(labels.item())
            record["clean_correct"] = float(record["clean_pred"] == int(labels.item()))
            record["adv_correct"] = float(record["adv_pred"] == int(labels.item()))
            forget_records.append(record)

        retain_records: List[Dict[str, Optional[float]]] = []
        for idx, (inputs, labels) in enumerate(eval_retain_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            record = _patch_recovery_search(
                model=model,
                normalizer=normalizer,
                attack_net=attack_net,
                inputs=inputs,
                target_class=forget_class,
                search_steps=int(args.search_steps),
                image_size=image_size,
                patch_size=int(args.patch_size),
            )
            record["sample_index"] = idx
            record["label"] = int(labels.item())
            record["clean_correct"] = float(record["clean_pred"] == int(labels.item()))
            record["adv_correct"] = float(record["adv_pred"] == int(labels.item()))
            retain_records.append(record)

        results["training"][suite_name] = {"history": history, "checkpoint_path": ckpt_path}
        results["forget_recovery"][suite_name] = {"summary": _summarize_radius(forget_records), "per_sample": forget_records}
        results["retain_control"][suite_name] = {"summary": _summarize_radius(retain_records), "per_sample": retain_records}

    attack_tag = (
        f"conditional_patch_ps{int(args.patch_size)}_st{int(args.patch_stride)}"
        f"_rw{str(args.retain_weight).replace('.', 'p')}"
        f"_rtc{str(args.retain_target_calib_weight).replace('.', 'p')}"
        f"_rca{str(args.retain_clean_anchor_weight).replace('.', 'p')}"
    )
    save_path = (
        f"results/analysis/conditional_patch_{dataset_name}_{model_type}_forget{forget_class}"
        f"_seed_{args.seed}_{args.eval_split}_{attack_tag}_{_suite_tag(model_suites)}.json"
    )
    save_dict_to_json(results, save_path)
    print(f"\nSaved conditional patch audit to: {save_path}")

    print("\nForget recovery summary")
    print("Model                                   | success | median alpha | mean alpha | mask area | acc drop")
    print("-" * 110)
    for suite_name in model_suites:
        summary = results["forget_recovery"][suite_name]["summary"]
        print(
            f"{suite_name:<39} | "
            f"{summary['success_rate']:.3f} | "
            f"{summary['median_radius']:.5f} | "
            f"{summary['mean_radius']:.5f} | "
            f"{summary['mean_mask_area']:.4f} | "
            f"{summary['accuracy_drop']:.3f}"
        )

    print("\nRetain-to-forget control summary")
    print("Model                                   | success | median alpha | mean alpha | acc drop")
    print("-" * 102)
    for suite_name in model_suites:
        summary = results["retain_control"][suite_name]["summary"]
        print(
            f"{suite_name:<39} | "
            f"{summary['success_rate']:.3f} | "
            f"{summary['median_radius']:.5f} | "
            f"{summary['mean_radius']:.5f} | "
            f"{summary['accuracy_drop']:.3f}"
        )


if __name__ == "__main__":
    main()
