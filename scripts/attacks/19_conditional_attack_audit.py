#!/usr/bin/env python3
"""
Learned conditional attacker audit.

This script trains a small image-conditioned perturbation generator G(x) for a
target model, then evaluates whether the generated perturbation direction can
recover forgotten examples more easily than retain controls.

Unlike the per-example optimization in 17_recovery_radius_audit.py, this attack
learns a shared conditional mapping across many samples. It is therefore a
better test of whether the residual forget signal is conditionally accessible in
a systematic way, rather than only by running a heavy optimizer per sample.
"""

import argparse
import importlib.util
import itertools
import json
import math
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
_resolve_matching_oracle_suite = getattr(_AUDIT, "_resolve_matching_oracle_suite")
_build_loader = getattr(_AUDIT, "_build_loader")
_predict = getattr(_AUDIT, "_predict")
_extract_normalized_features = getattr(_AUDIT, "_extract_normalized_features")
_margin_objective = getattr(_AUDIT, "_margin_objective")
_suite_tag = getattr(_AUDIT, "_suite_tag")
_actual_delta_norm = getattr(_AUDIT, "_actual_delta_norm")
_run_adam_margin_attack = getattr(_AUDIT, "_run_adam_margin_attack")


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.block(x), inplace=True)


class ConditionalPerturbationNet(nn.Module):
    """
    Tiny low-resolution perturbation generator.

    The generator sees a downsampled view of the input and outputs a low-res
    perturbation map that is bilinearly upsampled back to image resolution.
    """

    def __init__(self, in_channels: int = 3, width: int = 64, depth: int = 3):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        ]
        for _ in range(max(depth - 1, 0)):
            layers.append(ResidualBlock(width))
        layers.extend(
            [
                nn.Conv2d(width, width, kernel_size=3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(width, in_channels, kernel_size=1),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x_small: torch.Tensor) -> torch.Tensor:
        return self.net(x_small)


def _default_model_suites(experiment_suites: Dict) -> List[str]:
    return [suite for suite in DEFAULT_MODEL_SUITES if suite in experiment_suites]


def _normalize_direction(delta: torch.Tensor, norm: str) -> torch.Tensor:
    if norm == "l_inf":
        denom = delta.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
        return delta / denom
    if norm == "l2":
        flat = delta.view(delta.size(0), -1)
        denom = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        return (flat / denom).view_as(delta)
    raise ValueError(f"Unsupported norm: {norm}")


def _generator_direction(
    generator: nn.Module,
    inputs: torch.Tensor,
    lowres_size: int,
    norm: str,
) -> torch.Tensor:
    x_small = F.interpolate(inputs, size=(lowres_size, lowres_size), mode="bilinear", align_corners=False)
    delta_small = generator(x_small)
    delta = F.interpolate(delta_small, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
    delta = torch.tanh(delta)
    return _normalize_direction(delta, norm)


def _reshuffle_loader(loader, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        loader.dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )


def _apply_direction(
    inputs: torch.Tensor,
    direction: torch.Tensor,
    eps: float,
    norm: str,
) -> torch.Tensor:
    if norm not in {"l_inf", "l2"}:
        raise ValueError(f"Unsupported norm: {norm}")
    adv_inputs = torch.clamp(inputs + (eps * direction), 0.0, 1.0)
    return adv_inputs


def _total_variation(x: torch.Tensor) -> torch.Tensor:
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw


def _prepare_oracle_feats(
    oracle_model: Optional[nn.Module],
    normalizer: nn.Module,
    inputs: torch.Tensor,
    oracle_feature_weight: float,
) -> Optional[torch.Tensor]:
    if oracle_model is None or oracle_feature_weight <= 0.0:
        return None
    with torch.no_grad():
        return _extract_normalized_features(oracle_model, normalizer, inputs).detach()


def _feature_penalty(
    model: nn.Module,
    normalizer: nn.Module,
    adv_inputs: torch.Tensor,
    oracle_feats: Optional[torch.Tensor],
    oracle_feature_weight: float,
) -> torch.Tensor:
    if oracle_feats is None or oracle_feature_weight <= 0.0:
        return adv_inputs.new_zeros(())
    feats = _extract_normalized_features(model, normalizer, adv_inputs)
    return oracle_feature_weight * (1.0 - F.cosine_similarity(feats, oracle_feats, dim=1)).mean()


def _direction_alignment_loss(pred_dir: torch.Tensor, teacher_dir: torch.Tensor) -> torch.Tensor:
    pred_flat = pred_dir.view(pred_dir.size(0), -1)
    teacher_flat = teacher_dir.view(teacher_dir.size(0), -1)
    cosine = F.cosine_similarity(pred_flat, teacher_flat, dim=1)
    mse = (pred_dir - teacher_dir).pow(2).mean(dim=(1, 2, 3))
    return (1.0 - cosine + 0.5 * mse).mean()


def _teacher_dirs_by_target_groups(
    model: nn.Module,
    normalizer: nn.Module,
    inputs: torch.Tensor,
    target_classes: torch.Tensor,
    eps: float,
    norm: str,
    teacher_attack_steps: int,
    teacher_attack_restarts: int,
    teacher_alpha_ratio: float,
    teacher_alpha_min: float,
) -> torch.Tensor:
    teacher_dir = torch.zeros_like(inputs)
    unique_targets = torch.unique(target_classes.detach())
    for cls_tensor in unique_targets:
        cls = int(cls_tensor.item())
        mask = target_classes == cls
        if not bool(mask.any().item()):
            continue
        subset_inputs = inputs[mask]
        with torch.enable_grad():
            teacher_adv, _teacher_logits, _teacher_success = _run_adam_margin_attack(
                model=model,
                normalizer=normalizer,
                inputs=subset_inputs,
                target_class=cls,
                eps=eps,
                steps=teacher_attack_steps,
                restarts=teacher_attack_restarts,
                norm=norm,
                alpha_ratio=teacher_alpha_ratio,
                alpha_min=teacher_alpha_min,
            )
        teacher_dir[mask] = _normalize_direction((teacher_adv - subset_inputs).detach(), norm)
    return teacher_dir


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


def _evaluate_conditional_attack(
    model: nn.Module,
    normalizer: nn.Module,
    inputs: torch.Tensor,
    target_class: int,
    direction: torch.Tensor,
    eps: float,
    norm: str,
    refine_steps: int,
    refine_restarts: int,
    refine_alpha_ratio: float,
    refine_alpha_min: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    warm_adv = _apply_direction(inputs, direction, eps, norm)
    if refine_steps <= 0:
        logits, preds = _predict(model, normalizer, warm_adv)
        success = bool((preds == target_class).all().item())
        return warm_adv, logits, preds, success
    adv_inputs, adv_logits, success = _run_adam_margin_attack(
        model=model,
        normalizer=normalizer,
        inputs=inputs,
        target_class=target_class,
        eps=eps,
        steps=refine_steps,
        restarts=refine_restarts,
        norm=norm,
        alpha_ratio=refine_alpha_ratio,
        alpha_min=refine_alpha_min,
        initial_advs=[warm_adv],
    )
    adv_preds = adv_logits.argmax(dim=1)
    return adv_inputs, adv_logits, adv_preds, success


def _directional_radius(
    model: nn.Module,
    normalizer: nn.Module,
    inputs: torch.Tensor,
    target_class: int,
    direction: torch.Tensor,
    eps_max: float,
    search_steps: int,
    norm: str,
    refine_steps: int,
    refine_restarts: int,
    refine_alpha_ratio: float,
    refine_alpha_min: float,
) -> Dict[str, Optional[float]]:
    clean_logits, clean_preds = _predict(model, normalizer, inputs)
    clean_target_prob = float(F.softmax(clean_logits, dim=1)[0, target_class].item())
    clean_success = bool((clean_preds == target_class).all().item())
    if clean_success:
        return {
            "success": True,
            "clean_success": True,
            "radius": 0.0,
            "actual_delta": 0.0,
            "clean_target_prob": clean_target_prob,
            "adv_target_prob": clean_target_prob,
            "adv_pred": int(clean_preds.item()),
        }

    low = 0.0
    high = None
    high_adv = None
    high_logits = None
    probe_schedule = [eps_max * frac for frac in (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)]

    for eps in probe_schedule:
        adv_inputs, adv_logits, adv_preds, success = _evaluate_conditional_attack(
            model=model,
            normalizer=normalizer,
            inputs=inputs,
            target_class=target_class,
            direction=direction,
            eps=eps,
            norm=norm,
            refine_steps=refine_steps,
            refine_restarts=refine_restarts,
            refine_alpha_ratio=refine_alpha_ratio,
            refine_alpha_min=refine_alpha_min,
        )
        if success:
            high = eps
            high_adv = adv_inputs
            high_logits = adv_logits
            break
        low = eps

    if high is None:
        adv_inputs, adv_logits, adv_preds, success = _evaluate_conditional_attack(
            model=model,
            normalizer=normalizer,
            inputs=inputs,
            target_class=target_class,
            direction=direction,
            eps=eps_max,
            norm=norm,
            refine_steps=refine_steps,
            refine_restarts=refine_restarts,
            refine_alpha_ratio=refine_alpha_ratio,
            refine_alpha_min=refine_alpha_min,
        )
        if not success:
            adv_target_prob = float(F.softmax(adv_logits, dim=1)[0, target_class].item())
            return {
                "success": False,
                "clean_success": False,
                "radius": None,
                "actual_delta": _actual_delta_norm(inputs, adv_inputs, norm),
                "clean_target_prob": clean_target_prob,
                "adv_target_prob": adv_target_prob,
                "adv_pred": int(adv_preds.item()),
            }
        high = eps_max
        high_adv = adv_inputs
        high_logits = adv_logits

    for _ in range(search_steps):
        mid = 0.5 * (low + high)
        adv_inputs, adv_logits, adv_preds, success = _evaluate_conditional_attack(
            model=model,
            normalizer=normalizer,
            inputs=inputs,
            target_class=target_class,
            direction=direction,
            eps=mid,
            norm=norm,
            refine_steps=refine_steps,
            refine_restarts=refine_restarts,
            refine_alpha_ratio=refine_alpha_ratio,
            refine_alpha_min=refine_alpha_min,
        )
        if success:
            high = mid
            high_adv = adv_inputs
            high_logits = adv_logits
        else:
            low = mid

    if high_adv is None or high_logits is None:
        raise RuntimeError("Binary search finished without a successful adversarial state.")

    adv_probs = F.softmax(high_logits, dim=1)
    return {
        "success": True,
        "clean_success": False,
        "radius": float(high),
        "actual_delta": _actual_delta_norm(inputs, high_adv, norm),
        "clean_target_prob": clean_target_prob,
        "adv_target_prob": float(adv_probs[0, target_class].item()),
        "adv_pred": int(high_logits.argmax(dim=1).item()),
    }


def _summarize_radius(records: Sequence[Dict[str, Optional[float]]]) -> Dict[str, float]:
    radii = [float(r["radius"]) for r in records if r["radius"] is not None]
    success_rate = (sum(1 for r in records if r["success"]) / len(records)) if records else 0.0
    clean_success_rate = (sum(1 for r in records if r["clean_success"]) / len(records)) if records else 0.0
    target_prob_gain = [
        float(r["adv_target_prob"] - r["clean_target_prob"])
        for r in records
    ]

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
    }


def _train_conditional_generator(
    generator: nn.Module,
    model: nn.Module,
    normalizer: nn.Module,
    forget_loader,
    retain_loader,
    target_class: int,
    norm: str,
    eps_max: float,
    lowres_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    retain_weight: float,
    retain_margin_buffer: float,
    train_scale_min: float,
    tv_weight: float,
    delta_l2_weight: float,
    forget_ce_weight: float,
    forget_margin_weight: float,
    retain_target_calib_weight: float,
    retain_target_calib_buffer: float,
    retain_clean_anchor_weight: float,
    retain_clean_anchor_buffer: float,
    retain_clean_anchor_temp: float,
    use_retain_teacher_distill: bool,
    retain_distill_weight: float,
    use_teacher_distill: bool,
    distill_weight: float,
    teacher_attack_steps: int,
    teacher_attack_restarts: int,
    teacher_alpha_ratio: float,
    teacher_alpha_min: float,
    oracle_model: Optional[nn.Module] = None,
    oracle_feature_weight: float = 0.0,
) -> List[Dict[str, float]]:
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    if oracle_model is not None:
        for param in oracle_model.parameters():
            param.requires_grad_(False)
        oracle_model.eval()

    generator.train()
    optimizer = torch.optim.AdamW(generator.parameters(), lr=lr, weight_decay=weight_decay)
    history: List[Dict[str, float]] = []
    retain_iter = itertools.cycle(retain_loader) if retain_loader is not None else None

    for epoch_idx in range(epochs):
        epoch_forget_margin = 0.0
        epoch_retain_margin = 0.0
        epoch_loss = 0.0
        epoch_distill = 0.0
        epoch_retain_distill = 0.0
        n_steps = 0

        for forget_inputs, _ in forget_loader:
            forget_inputs = forget_inputs.to(next(generator.parameters()).device)
            retain_inputs = None
            if retain_iter is not None:
                retain_inputs, _ = next(retain_iter)
                retain_inputs = retain_inputs.to(forget_inputs.device)

            forget_scale = eps_max * (train_scale_min + (1.0 - train_scale_min) * torch.rand(1, device=forget_inputs.device))
            forget_dir = _generator_direction(generator, forget_inputs, lowres_size, norm)
            forget_adv = _apply_direction(forget_inputs, forget_dir, float(forget_scale.item()), norm)
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

            oracle_feats = _prepare_oracle_feats(
                oracle_model=oracle_model,
                normalizer=normalizer,
                inputs=forget_inputs,
                oracle_feature_weight=oracle_feature_weight,
            )
            feat_penalty = _feature_penalty(
                model=model,
                normalizer=normalizer,
                adv_inputs=forget_adv,
                oracle_feats=oracle_feats,
                oracle_feature_weight=oracle_feature_weight,
            )
            distill_loss = forget_inputs.new_zeros(())
            if use_teacher_distill and distill_weight > 0.0:
                with torch.enable_grad():
                    teacher_adv, _teacher_logits, _teacher_success = _run_adam_margin_attack(
                        model=model,
                        normalizer=normalizer,
                        inputs=forget_inputs,
                        target_class=target_class,
                        eps=float(forget_scale.item()),
                        steps=teacher_attack_steps,
                        restarts=teacher_attack_restarts,
                        norm=norm,
                        alpha_ratio=teacher_alpha_ratio,
                        alpha_min=teacher_alpha_min,
                    )
                teacher_dir = _normalize_direction((teacher_adv - forget_inputs).detach(), norm)
                distill_loss = _direction_alignment_loss(forget_dir, teacher_dir)

            retain_loss = forget_inputs.new_zeros(())
            retain_margin_mean = forget_inputs.new_zeros(())
            retain_distill_loss = forget_inputs.new_zeros(())
            reg_tv = _total_variation(forget_dir)
            reg_l2 = forget_dir.pow(2).mean()

            if retain_inputs is not None and retain_weight > 0.0:
                retain_scale = eps_max * (train_scale_min + (1.0 - train_scale_min) * torch.rand(1, device=retain_inputs.device))
                retain_dir = _generator_direction(generator, retain_inputs, lowres_size, norm)
                retain_adv = _apply_direction(retain_inputs, retain_dir, float(retain_scale.item()), norm)
                with torch.no_grad():
                    retain_clean_logits = model(normalizer(retain_inputs)).detach()
                retain_logits = model(normalizer(retain_adv))
                retain_margin = _margin_objective(retain_logits, target_class)
                retain_margin_mean = retain_margin.mean().detach()
                retain_loss = F.softplus(retain_margin + retain_margin_buffer).mean()
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
                if use_retain_teacher_distill and retain_distill_weight > 0.0:
                    retain_clean_preds = retain_clean_logits.argmax(dim=1)
                    teacher_retain_dir = _teacher_dirs_by_target_groups(
                        model=model,
                        normalizer=normalizer,
                        inputs=retain_inputs,
                        target_classes=retain_clean_preds,
                        eps=float(retain_scale.item()),
                        norm=norm,
                        teacher_attack_steps=teacher_attack_steps,
                        teacher_attack_restarts=teacher_attack_restarts,
                        teacher_alpha_ratio=teacher_alpha_ratio,
                        teacher_alpha_min=teacher_alpha_min,
                    )
                    retain_distill_loss = _direction_alignment_loss(retain_dir, teacher_retain_dir)
                reg_tv = 0.5 * (reg_tv + _total_variation(retain_dir))
                reg_l2 = 0.5 * (reg_l2 + retain_dir.pow(2).mean())

            loss = forget_loss + (retain_weight * retain_loss) + feat_penalty + (distill_weight * distill_loss)
            loss = loss + (retain_distill_weight * retain_distill_loss)
            loss = loss + (tv_weight * reg_tv) + (delta_l2_weight * reg_l2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
            optimizer.step()

            n_steps += 1
            epoch_loss += float(loss.item())
            epoch_forget_margin += float(forget_margin.mean().item())
            epoch_retain_margin += float(retain_margin_mean.item())
            epoch_distill += float(distill_loss.item())
            epoch_retain_distill += float(retain_distill_loss.item())

        history.append(
            {
                "epoch": float(epoch_idx),
                "loss": epoch_loss / max(n_steps, 1),
                "forget_margin": epoch_forget_margin / max(n_steps, 1),
                "retain_margin": epoch_retain_margin / max(n_steps, 1),
                "distill_loss": epoch_distill / max(n_steps, 1),
                "retain_distill_loss": epoch_retain_distill / max(n_steps, 1),
            }
        )
        print(
            f"Epoch {epoch_idx}: "
            f"loss={history[-1]['loss']:.4f} "
            f"forget_margin={history[-1]['forget_margin']:.4f} "
            f"retain_margin={history[-1]['retain_margin']:.4f} "
            f"distill={history[-1]['distill_loss']:.4f} "
            f"retain_distill={history[-1]['retain_distill_loss']:.4f}"
        )

    generator.eval()
    return history


def main():
    parser = argparse.ArgumentParser(description="Learned conditional attacker audit")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--train-split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--eval-split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model-suites", nargs="*", default=None)
    parser.add_argument("--max-train-forget-samples", type=int, default=256)
    parser.add_argument("--max-train-retain-samples", type=int, default=512)
    parser.add_argument("--max-eval-forget-samples", type=int, default=32)
    parser.add_argument("--max-eval-retain-samples", type=int, default=32)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eps-max", type=float, default=8 / 255)
    parser.add_argument("--norm", type=str, default="l_inf", choices=["l_inf", "l2"])
    parser.add_argument("--search-steps", type=int, default=6)
    parser.add_argument("--generator-width", type=int, default=64)
    parser.add_argument("--generator-depth", type=int, default=3)
    parser.add_argument("--lowres-size", type=int, default=32)
    parser.add_argument("--retain-weight", type=float, default=0.5)
    parser.add_argument("--retain-margin-buffer", type=float, default=0.0)
    parser.add_argument("--retain-target-calib-weight", type=float, default=0.0)
    parser.add_argument("--retain-target-calib-buffer", type=float, default=0.0)
    parser.add_argument("--retain-clean-anchor-weight", type=float, default=0.0)
    parser.add_argument("--retain-clean-anchor-buffer", type=float, default=0.0)
    parser.add_argument("--retain-clean-anchor-temp", type=float, default=1.0)
    parser.add_argument("--use-retain-teacher-distill", action="store_true")
    parser.add_argument("--retain-distill-weight", type=float, default=0.0)
    parser.add_argument("--train-scale-min", type=float, default=0.5)
    parser.add_argument("--tv-weight", type=float, default=1e-2)
    parser.add_argument("--delta-l2-weight", type=float, default=1e-3)
    parser.add_argument("--forget-ce-weight", type=float, default=1.0)
    parser.add_argument("--forget-margin-weight", type=float, default=0.25)
    parser.add_argument("--use-teacher-distill", action="store_true")
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--teacher-attack-steps", type=int, default=6)
    parser.add_argument("--teacher-attack-restarts", type=int, default=1)
    parser.add_argument("--teacher-alpha-ratio", type=float, default=0.2)
    parser.add_argument("--teacher-alpha-min", type=float, default=1 / 255)
    parser.add_argument("--refine-steps", type=int, default=0)
    parser.add_argument("--refine-restarts", type=int, default=1)
    parser.add_argument("--refine-alpha-ratio", type=float, default=0.2)
    parser.add_argument("--refine-alpha-min", type=float, default=1 / 255)
    parser.add_argument("--oracle-feature-weight", type=float, default=0.0)
    parser.add_argument("--use-oracle-guidance", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    experiment_suites = load_config(args.config)
    model_suites = args.model_suites or _default_model_suites(experiment_suites)
    if not model_suites:
        raise ValueError("No model suites provided and no default suites found in config")
    if int(args.eval_batch_size) != 1:
        raise ValueError("Directional radius audit is defined per sample. Use --eval-batch-size 1.")

    first_suite_cfg = experiment_suites[model_suites[0]]
    base_suite_name = first_suite_cfg["base_model_suite"]
    base_suite_cfg = experiment_suites[base_suite_name]
    dataset_name = base_suite_cfg["dataset"]
    model_type = base_suite_cfg["model"]
    forget_class = int(first_suite_cfg.get("unlearning", {}).get("forget_class", first_suite_cfg.get("forget_class", 0)))
    normalizer = create_imagenet_normalizer().to(device)

    oracle_model = None
    if args.use_oracle_guidance and args.oracle_feature_weight > 0.0:
        oracle_suite = _resolve_matching_oracle_suite(
            experiment_suites,
            dataset_name=dataset_name,
            model_type=model_type,
            forget_class=forget_class,
        )
        oracle_model, _, _ = _load_model(experiment_suites, oracle_suite, args.seed, device)
        oracle_model = oracle_model.eval()

    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_type}")
    print(f"Forget class: {forget_class}")
    print(f"Model suites: {model_suites}")
    print(f"Attack: conditional_generator")
    print(f"Norm: {args.norm}")
    print(f"eps_max: {args.eps_max}")

    results = {
        "meta": {
            "seed": int(args.seed),
            "train_split": args.train_split,
            "eval_split": args.eval_split,
            "dataset": dataset_name,
            "model": model_type,
            "forget_class": int(forget_class),
            "model_suites": model_suites,
            "attack_family": "conditional_generator",
            "norm": args.norm,
            "eps_max": float(args.eps_max),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "generator_width": int(args.generator_width),
            "generator_depth": int(args.generator_depth),
            "lowres_size": int(args.lowres_size),
            "retain_weight": float(args.retain_weight),
            "retain_margin_buffer": float(args.retain_margin_buffer),
            "retain_target_calib_weight": float(args.retain_target_calib_weight),
            "retain_target_calib_buffer": float(args.retain_target_calib_buffer),
            "retain_clean_anchor_weight": float(args.retain_clean_anchor_weight),
            "retain_clean_anchor_buffer": float(args.retain_clean_anchor_buffer),
            "retain_clean_anchor_temp": float(args.retain_clean_anchor_temp),
            "use_retain_teacher_distill": bool(args.use_retain_teacher_distill),
            "retain_distill_weight": float(args.retain_distill_weight),
            "train_scale_min": float(args.train_scale_min),
            "tv_weight": float(args.tv_weight),
            "delta_l2_weight": float(args.delta_l2_weight),
            "forget_ce_weight": float(args.forget_ce_weight),
            "forget_margin_weight": float(args.forget_margin_weight),
            "use_teacher_distill": bool(args.use_teacher_distill),
            "distill_weight": float(args.distill_weight),
            "teacher_attack_steps": int(args.teacher_attack_steps),
            "teacher_attack_restarts": int(args.teacher_attack_restarts),
            "teacher_alpha_ratio": float(args.teacher_alpha_ratio),
            "teacher_alpha_min": float(args.teacher_alpha_min),
            "refine_steps": int(args.refine_steps),
            "refine_restarts": int(args.refine_restarts),
            "refine_alpha_ratio": float(args.refine_alpha_ratio),
            "refine_alpha_min": float(args.refine_alpha_min),
            "use_oracle_guidance": bool(args.use_oracle_guidance),
            "oracle_feature_weight": float(args.oracle_feature_weight),
        },
        "forget_recovery": {},
        "retain_control": {},
        "training": {},
    }

    for suite_name in model_suites:
        print(f"\nTraining conditional attacker for {suite_name}...")
        model, suite_dataset, suite_model = _load_model(experiment_suites, suite_name, args.seed, device)
        if suite_dataset != dataset_name or suite_model != model_type:
            raise ValueError(f"Suite {suite_name} resolved to {suite_dataset}/{suite_model}, expected {dataset_name}/{model_type}")
        model.eval()

        train_forget_loader = _build_loader(
            dataset_name=dataset_name,
            split=args.train_split,
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            forget_class=forget_class,
            retain=False,
            max_samples=args.max_train_forget_samples,
        )
        train_forget_loader = _reshuffle_loader(
            train_forget_loader,
            batch_size=int(args.train_batch_size),
            num_workers=int(args.num_workers),
        )
        train_retain_loader = _build_loader(
            dataset_name=dataset_name,
            split=args.train_split,
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            forget_class=forget_class,
            retain=True,
            max_samples=args.max_train_retain_samples,
        )
        train_retain_loader = _reshuffle_loader(
            train_retain_loader,
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

        generator = ConditionalPerturbationNet(
            in_channels=3,
            width=int(args.generator_width),
            depth=int(args.generator_depth),
        ).to(device)

        history = _train_conditional_generator(
            generator=generator,
            model=model,
            normalizer=normalizer,
            forget_loader=train_forget_loader,
            retain_loader=train_retain_loader,
            target_class=forget_class,
            norm=args.norm,
            eps_max=float(args.eps_max),
            lowres_size=int(args.lowres_size),
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            retain_weight=float(args.retain_weight),
            retain_margin_buffer=float(args.retain_margin_buffer),
            retain_target_calib_weight=float(args.retain_target_calib_weight),
            retain_target_calib_buffer=float(args.retain_target_calib_buffer),
            retain_clean_anchor_weight=float(args.retain_clean_anchor_weight),
            retain_clean_anchor_buffer=float(args.retain_clean_anchor_buffer),
            retain_clean_anchor_temp=float(args.retain_clean_anchor_temp),
            use_retain_teacher_distill=bool(args.use_retain_teacher_distill),
            retain_distill_weight=float(args.retain_distill_weight),
            train_scale_min=float(args.train_scale_min),
            tv_weight=float(args.tv_weight),
            delta_l2_weight=float(args.delta_l2_weight),
            forget_ce_weight=float(args.forget_ce_weight),
            forget_margin_weight=float(args.forget_margin_weight),
            use_teacher_distill=bool(args.use_teacher_distill),
            distill_weight=float(args.distill_weight),
            teacher_attack_steps=int(args.teacher_attack_steps),
            teacher_attack_restarts=int(args.teacher_attack_restarts),
            teacher_alpha_ratio=float(args.teacher_alpha_ratio),
            teacher_alpha_min=float(args.teacher_alpha_min),
            oracle_model=oracle_model,
            oracle_feature_weight=float(args.oracle_feature_weight) if args.use_oracle_guidance else 0.0,
        )

        checkpoint_dir = os.path.join("checkpoints", "conditional_attack")
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(
            checkpoint_dir,
            f"conditional_generator_{suite_name}_seed_{args.seed}.pt",
        )
        torch.save(
            {
                "state_dict": generator.state_dict(),
                "meta": results["meta"],
                "training_history": history,
            },
            ckpt_path,
        )

        print(f"Evaluating learned direction recovery for {suite_name}...")
        forget_records: List[Dict[str, Optional[float]]] = []
        for idx, (inputs, _labels) in enumerate(eval_forget_loader):
            inputs = inputs.to(device)
            with torch.no_grad():
                direction = _generator_direction(generator, inputs, int(args.lowres_size), args.norm)
            record = _directional_radius(
                model=model,
                normalizer=normalizer,
                inputs=inputs,
                target_class=forget_class,
                direction=direction,
                eps_max=float(args.eps_max),
                search_steps=int(args.search_steps),
                norm=args.norm,
                refine_steps=int(args.refine_steps),
                refine_restarts=int(args.refine_restarts),
                refine_alpha_ratio=float(args.refine_alpha_ratio),
                refine_alpha_min=float(args.refine_alpha_min),
            )
            record["sample_index"] = idx
            record["direction_mean_abs"] = float(direction.abs().mean().item())
            record["direction_max_abs"] = float(direction.abs().max().item())
            forget_records.append(record)

        retain_records: List[Dict[str, Optional[float]]] = []
        for idx, (inputs, _labels) in enumerate(eval_retain_loader):
            inputs = inputs.to(device)
            with torch.no_grad():
                direction = _generator_direction(generator, inputs, int(args.lowres_size), args.norm)
            record = _directional_radius(
                model=model,
                normalizer=normalizer,
                inputs=inputs,
                target_class=forget_class,
                direction=direction,
                eps_max=float(args.eps_max),
                search_steps=int(args.search_steps),
                norm=args.norm,
                refine_steps=int(args.refine_steps),
                refine_restarts=int(args.refine_restarts),
                refine_alpha_ratio=float(args.refine_alpha_ratio),
                refine_alpha_min=float(args.refine_alpha_min),
            )
            record["sample_index"] = idx
            record["direction_mean_abs"] = float(direction.abs().mean().item())
            record["direction_max_abs"] = float(direction.abs().max().item())
            retain_records.append(record)

        results["training"][suite_name] = {
            "history": history,
            "checkpoint_path": ckpt_path,
        }
        results["forget_recovery"][suite_name] = {
            "summary": _summarize_radius(forget_records),
            "per_sample": forget_records,
        }
        results["retain_control"][suite_name] = {
            "summary": _summarize_radius(retain_records),
            "per_sample": retain_records,
        }

    attack_tag = "conditional_generator"
    if bool(args.use_teacher_distill):
        attack_tag = f"{attack_tag}_distill"
        if abs(float(args.distill_weight) - 1.0) > 1e-12:
            distill_tag = str(args.distill_weight).replace("-", "m").replace(".", "p")
            attack_tag = f"{attack_tag}_dw{distill_tag}"
        if int(args.teacher_attack_steps) != 6:
            attack_tag = f"{attack_tag}_ts{int(args.teacher_attack_steps)}"
    if int(args.refine_steps) > 0:
        attack_tag = f"{attack_tag}_adamrefine_s{int(args.refine_steps)}"
    if int(args.epochs) != 8:
        attack_tag = f"{attack_tag}_ep{int(args.epochs)}"
    if abs(float(args.retain_weight) - 0.5) > 1e-12:
        retain_tag_value = str(args.retain_weight).replace("-", "m").replace(".", "p")
        attack_tag = f"{attack_tag}_rw{retain_tag_value}"
    if abs(float(args.retain_target_calib_weight)) > 1e-12:
        calib_tag = str(args.retain_target_calib_weight).replace("-", "m").replace(".", "p")
        attack_tag = f"{attack_tag}_rtc{calib_tag}"
    if abs(float(args.retain_clean_anchor_weight)) > 1e-12:
        anchor_tag = str(args.retain_clean_anchor_weight).replace("-", "m").replace(".", "p")
        attack_tag = f"{attack_tag}_rca{anchor_tag}"
    if bool(args.use_retain_teacher_distill) and abs(float(args.retain_distill_weight)) > 1e-12:
        retain_distill_tag = str(args.retain_distill_weight).replace("-", "m").replace(".", "p")
        attack_tag = f"{attack_tag}_rtd{retain_distill_tag}"
    if int(args.max_train_forget_samples) != 256:
        attack_tag = f"{attack_tag}_tf{int(args.max_train_forget_samples)}"
    if int(args.max_train_retain_samples) != 512:
        attack_tag = f"{attack_tag}_tr{int(args.max_train_retain_samples)}"
    if args.use_oracle_guidance and args.oracle_feature_weight > 0.0:
        weight_tag = str(args.oracle_feature_weight).replace(".", "p")
        attack_tag = f"{attack_tag}_oracle_w{weight_tag}"
    save_path = (
        f"results/analysis/conditional_attack_{dataset_name}_{model_type}_forget{forget_class}"
        f"_seed_{args.seed}_{args.eval_split}_{attack_tag}_{_suite_tag(model_suites)}.json"
    )
    save_dict_to_json(results, save_path)
    print(f"\nSaved conditional attacker audit to: {save_path}")

    print("\nForget recovery summary")
    print("Model                                   | success | median eps | mean eps | clean-success")
    print("-" * 92)
    for suite_name in model_suites:
        summary = results["forget_recovery"][suite_name]["summary"]
        print(
            f"{suite_name:<39} | "
            f"{summary['success_rate']:.3f} | "
            f"{summary['median_radius']:.5f} | "
            f"{summary['mean_radius']:.5f} | "
            f"{summary['clean_success_rate']:.3f}"
        )

    print("\nRetain-to-forget control summary")
    print("Model                                   | success | median eps | mean eps")
    print("-" * 84)
    for suite_name in model_suites:
        summary = results["retain_control"][suite_name]["summary"]
        print(
            f"{suite_name:<39} | "
            f"{summary['success_rate']:.3f} | "
            f"{summary['median_radius']:.5f} | "
            f"{summary['mean_radius']:.5f}"
        )


if __name__ == "__main__":
    main()
