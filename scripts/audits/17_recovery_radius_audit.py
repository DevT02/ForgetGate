#!/usr/bin/env python3
"""
Per-example recovery-radius audit.

For each forget sample, find the smallest targeted perturbation budget that makes
the unlearned model predict the forgotten class again. This tests conditional
recoverability even when universal attacks fail.
"""

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.attacks.pgd import PGDAttack
from src.data import DataManager
from src.models.cnn import create_cnn_model
from src.models.normalize import create_imagenet_normalizer
from src.models.peft_lora import load_lora_adapter
from src.models.vit import create_vit_model
from src.unlearning.objectives import _resolve_feature_model
from src.utils import get_device, load_config, save_dict_to_json, set_seed


DEFAULT_MODEL_SUITES = [
    "unlearn_kl_vit_cifar10_forget0",
    "unlearn_salun_vit_cifar10_forget0",
    "unlearn_scrub_distill_vit_cifar10_forget0",
    "unlearn_orbit_vit_cifar10_forget0",
]


def _checkpoint_or_best(base_dir: str, suite_name: str, seed: int) -> str:
    final_path = os.path.join(base_dir, f"{suite_name}_seed_{seed}_final.pt")
    best_path = os.path.join(base_dir, f"{suite_name}_seed_{seed}_best.pt")
    if os.path.exists(final_path):
        return final_path
    if os.path.exists(best_path):
        return best_path
    raise FileNotFoundError(f"Checkpoint not found for {suite_name}: {final_path} / {best_path}")


def _build_model(model_type: str, dataset_name: str, device: torch.device) -> torch.nn.Module:
    data_cfg = load_config("configs/data.yaml")
    model_cfg = load_config("configs/model.yaml")
    num_classes = int(data_cfg[dataset_name]["num_classes"])

    if model_type.startswith("vit"):
        model_key = model_type.replace("vit_", "")
        vit_cfg = dict(model_cfg["vit"][model_key])
        vit_cfg["pretrained"] = False
        model = create_vit_model(vit_cfg, num_classes=num_classes)
    else:
        model = create_cnn_model(model_cfg["cnn"][model_type], num_classes=num_classes)
    return model.to(device)


def _load_checkpoint_path_model(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return model.to(device)


def _resolve_unlearn_artifact_path(suite_cfg: Dict, suite_name: str, seed: int) -> str:
    explicit_path = suite_cfg.get("path")
    if explicit_path:
        return explicit_path.format(seed=seed, suite_name=suite_name)
    method = suite_cfg.get("unlearning", {}).get("method", "lora")
    if method == "lora":
        return os.path.join("checkpoints", "unlearn_lora", f"{suite_name}_seed_{seed}")
    return os.path.join("checkpoints", "unlearn_full", f"{suite_name}_seed_{seed}_final.pt")


def _load_unlearn_artifact(
    model: torch.nn.Module,
    artifact_path: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, str]:
    if os.path.isdir(artifact_path):
        adapter_cfg = os.path.join(artifact_path, "adapter_config.json")
        adapter_weights = os.path.join(artifact_path, "adapter_model.safetensors")
        model_file = os.path.join(artifact_path, "model.pt")
        if os.path.exists(adapter_cfg) and os.path.exists(adapter_weights):
            return load_lora_adapter(model, artifact_path).to(device), "adapter"
        if os.path.exists(model_file):
            return _load_checkpoint_path_model(model, model_file, device), "checkpoint"
    elif os.path.isfile(artifact_path):
        return _load_checkpoint_path_model(model, artifact_path, device), "checkpoint"
    raise FileNotFoundError(f"Unlearned artifact not found: {artifact_path}")


def _load_model(
    experiment_suites: Dict,
    suite_name: str,
    seed: int,
    device: torch.device,
) -> Tuple[torch.nn.Module, str, str]:
    if suite_name.startswith("oracle_"):
        suite_cfg = experiment_suites[suite_name]
        model = _build_model(suite_cfg["model"], suite_cfg["dataset"], device)
        ckpt_path = _checkpoint_or_best("checkpoints/oracle", suite_name, seed)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(device), suite_cfg["dataset"], suite_cfg["model"]

    if suite_name.startswith("unlearn_"):
        suite_cfg = experiment_suites[suite_name]
        base_suite_name = suite_cfg["base_model_suite"]
        base_suite_cfg = experiment_suites[base_suite_name]
        model = _build_model(base_suite_cfg["model"], base_suite_cfg["dataset"], device)
        base_ckpt_path = _checkpoint_or_best("checkpoints/base", base_suite_name, seed)
        checkpoint = torch.load(base_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        artifact_path = _resolve_unlearn_artifact_path(suite_cfg, suite_name, seed)
        if os.path.exists(artifact_path):
            model, _artifact_kind = _load_unlearn_artifact(model, artifact_path, device)
        else:
            print(f"Warning: adapter/checkpoint not found for {suite_name}, using base model weights only")
        return model.to(device), base_suite_cfg["dataset"], base_suite_cfg["model"]

    if suite_name.startswith("base_"):
        suite_cfg = experiment_suites[suite_name]
        model = _build_model(suite_cfg["model"], suite_cfg["dataset"], device)
        ckpt_path = _checkpoint_or_best("checkpoints/base", suite_name, seed)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(device), suite_cfg["dataset"], suite_cfg["model"]

    raise ValueError(f"Unsupported suite type: {suite_name}")


def _resolve_matching_oracle_suite(
    experiment_suites: Dict,
    dataset_name: str,
    model_type: str,
    forget_class: int,
) -> str:
    for suite_name, suite_cfg in experiment_suites.items():
        if not suite_name.startswith("oracle_"):
            continue
        if suite_cfg.get("dataset") != dataset_name:
            continue
        if suite_cfg.get("model") != model_type:
            continue
        if int(suite_cfg.get("forget_class", -1)) != int(forget_class):
            continue
        return suite_name
    raise ValueError(
        f"No oracle suite found for dataset={dataset_name}, model={model_type}, forget_class={forget_class}"
    )


def _build_loader(
    dataset_name: str,
    split: str,
    batch_size: int,
    num_workers: int,
    forget_class: int,
    retain: bool,
    max_samples: int,
) -> DataLoader:
    data_manager = DataManager()
    kwargs = {
        "use_pretrained": True,
        "apply_imagenet_norm": False,
    }
    if retain:
        dataset = data_manager.load_dataset(
            dataset_name,
            split,
            exclude_classes=[forget_class],
            **kwargs,
        )
    else:
        dataset = data_manager.load_dataset(
            dataset_name,
            split,
            include_classes=[forget_class],
            **kwargs,
        )
    if max_samples > 0 and len(dataset) > max_samples:
        dataset = torch.utils.data.Subset(dataset, list(range(max_samples)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


@torch.no_grad()
def _predict(
    model: torch.nn.Module,
    normalizer: torch.nn.Module,
    inputs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = model(normalizer(inputs))
    probs = F.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    return logits, preds


def _extract_normalized_features(
    model: torch.nn.Module,
    normalizer: torch.nn.Module,
    inputs: torch.Tensor,
) -> torch.Tensor:
    feature_model = _resolve_feature_model(model)
    if feature_model is None:
        raise ValueError("Oracle-feature guidance requires forward_features support.")
    feats = feature_model.forward_features(normalizer(inputs))
    if isinstance(feats, (tuple, list)):
        feats = feats[0]
    if feats.ndim == 4:
        feats = feats.mean(dim=(2, 3))
    elif feats.ndim == 3:
        feats = feats[:, 0]
    elif feats.ndim > 2:
        feats = feats.flatten(start_dim=1)
    return F.normalize(feats, dim=1)


def _project_ball(
    adv_inputs: torch.Tensor,
    clean_inputs: torch.Tensor,
    eps: float,
    norm: str,
) -> torch.Tensor:
    if norm == "l_inf":
        delta = torch.clamp(adv_inputs - clean_inputs, -eps, eps)
        return torch.clamp(clean_inputs + delta, 0.0, 1.0)
    if norm == "l2":
        delta = adv_inputs - clean_inputs
        flat = delta.view(delta.size(0), -1)
        delta_norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        factor = torch.minimum(torch.ones_like(delta_norm), torch.full_like(delta_norm, eps) / delta_norm)
        delta = (flat * factor).view_as(adv_inputs)
        return torch.clamp(clean_inputs + delta, 0.0, 1.0)
    raise ValueError(f"Unsupported norm: {norm}")


def _random_init(
    clean_inputs: torch.Tensor,
    eps: float,
    norm: str,
) -> torch.Tensor:
    if norm == "l_inf":
        noise = torch.empty_like(clean_inputs).uniform_(-eps, eps)
        return torch.clamp(clean_inputs + noise, 0.0, 1.0)
    if norm == "l2":
        noise = torch.randn_like(clean_inputs)
        flat = noise.view(noise.size(0), -1)
        norm_val = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        noise = (flat / norm_val).view_as(clean_inputs) * eps
        return torch.clamp(clean_inputs + noise, 0.0, 1.0)
    raise ValueError(f"Unsupported norm: {norm}")


def _atanh(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _box_bounds(clean_inputs: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    lower = torch.clamp(clean_inputs - eps, 0.0, 1.0)
    upper = torch.clamp(clean_inputs + eps, 0.0, 1.0)
    return lower, upper


def _box_parameterize(latent: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    mid = 0.5 * (upper + lower)
    half = 0.5 * (upper - lower)
    return mid + half * torch.tanh(latent)


def _box_inverse(adv_inputs: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    width = (upper - lower).clamp_min(1e-6)
    scaled = ((adv_inputs - lower) / width) * 2.0 - 1.0
    return _atanh(scaled)


def _margin_objective(logits: torch.Tensor, target_class: int) -> torch.Tensor:
    target_logits = logits[:, target_class]
    other_logits = logits.clone()
    other_logits[:, target_class] = float("-inf")
    max_other = other_logits.max(dim=1).values
    return target_logits - max_other


def _margin_loss(logits: torch.Tensor, target_class: int) -> torch.Tensor:
    return -_margin_objective(logits, target_class).mean()


def _oracle_guided_objective(
    model: torch.nn.Module,
    normalizer: torch.nn.Module,
    adv_inputs: torch.Tensor,
    logits: torch.Tensor,
    target_class: int,
    oracle_feats: Optional[torch.Tensor],
    oracle_feature_weight: float,
) -> torch.Tensor:
    objective = _margin_objective(logits, target_class)
    if oracle_feats is None or oracle_feature_weight <= 0.0:
        return objective
    feats = _extract_normalized_features(model, normalizer, adv_inputs)
    feat_penalty = 1.0 - F.cosine_similarity(feats, oracle_feats, dim=1)
    return objective - (oracle_feature_weight * feat_penalty)


def _run_margin_attack(
    model: torch.nn.Module,
    normalizer: torch.nn.Module,
    inputs: torch.Tensor,
    target_class: int,
    eps: float,
    steps: int,
    restarts: int,
    norm: str,
    alpha_ratio: float,
    alpha_min: float,
    oracle_model: Optional[torch.nn.Module] = None,
    oracle_feature_weight: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    if eps <= 0.0:
        logits, preds = _predict(model, normalizer, inputs)
        success = bool((preds == target_class).all().item())
        return inputs.detach().clone(), logits, success

    alpha = min(max(alpha_min, eps * alpha_ratio), eps)
    best_adv = inputs.detach().clone()
    best_logits, _ = _predict(model, normalizer, best_adv)
    best_margin = float("-inf")

    oracle_feats = None
    if oracle_model is not None and oracle_feature_weight > 0.0:
        with torch.no_grad():
            oracle_feats = _extract_normalized_features(oracle_model, normalizer, inputs).detach()

    for _ in range(restarts):
        adv_inputs = _random_init(inputs.detach(), eps, norm)
        for _step in range(steps):
            adv_inputs.requires_grad_(True)
            logits = model(normalizer(adv_inputs))
            loss = _margin_loss(logits, target_class)
            if oracle_feats is not None:
                feats = _extract_normalized_features(model, normalizer, adv_inputs)
                feat_loss = (1.0 - F.cosine_similarity(feats, oracle_feats, dim=1)).mean()
                loss = loss + (oracle_feature_weight * feat_loss)
            grad = torch.autograd.grad(loss, adv_inputs, only_inputs=True)[0]
            with torch.no_grad():
                if norm == "l_inf":
                    direction = grad.sign()
                elif norm == "l2":
                    flat = grad.view(grad.size(0), -1)
                    grad_norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
                    direction = (flat / grad_norm).view_as(grad)
                else:
                    raise ValueError(f"Unsupported norm: {norm}")
                adv_inputs = adv_inputs - alpha * direction
                adv_inputs = _project_ball(adv_inputs, inputs, eps, norm)
            adv_inputs = adv_inputs.detach()

        with torch.no_grad():
            logits, preds = _predict(model, normalizer, adv_inputs)
            target_logits = logits[:, target_class]
            other_logits = logits.clone()
            other_logits[:, target_class] = float("-inf")
            margin = float((target_logits - other_logits.max(dim=1).values).mean().item())
            if margin > best_margin:
                best_margin = margin
                best_adv = adv_inputs.detach().clone()
                best_logits = logits.detach().clone()

    success = bool((best_logits.argmax(dim=1) == target_class).all().item())
    return best_adv, best_logits, success


def _run_apgd_margin_attack(
    model: torch.nn.Module,
    normalizer: torch.nn.Module,
    inputs: torch.Tensor,
    target_class: int,
    eps: float,
    steps: int,
    restarts: int,
    norm: str,
    alpha_ratio: float,
    alpha_min: float,
    oracle_model: Optional[torch.nn.Module] = None,
    oracle_feature_weight: float = 0.0,
    apgd_rho: float = 0.5,
    apgd_momentum: float = 0.75,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    if eps <= 0.0:
        logits, preds = _predict(model, normalizer, inputs)
        success = bool((preds == target_class).all().item())
        return inputs.detach().clone(), logits, success

    batch_size = inputs.size(0)
    base_alpha = max(alpha_min, eps * alpha_ratio)
    base_alpha = min(base_alpha, eps)
    check_interval = max(4, steps // 6)
    best_adv = inputs.detach().clone()
    best_logits, _ = _predict(model, normalizer, best_adv)
    best_objective = torch.full((batch_size,), float("-inf"), device=inputs.device)

    oracle_feats = None
    if oracle_model is not None and oracle_feature_weight > 0.0:
        with torch.no_grad():
            oracle_feats = _extract_normalized_features(oracle_model, normalizer, inputs).detach()

    for _ in range(restarts):
        adv_inputs = _random_init(inputs.detach(), eps, norm)
        prev_adv = adv_inputs.detach().clone()
        step_size = torch.full((batch_size, 1, 1, 1), base_alpha, device=inputs.device)
        checkpoint_objective = torch.full((batch_size,), float("-inf"), device=inputs.device)

        with torch.no_grad():
            logits = model(normalizer(adv_inputs))
            objective = _oracle_guided_objective(
                model=model,
                normalizer=normalizer,
                adv_inputs=adv_inputs,
                logits=logits,
                target_class=target_class,
                oracle_feats=oracle_feats,
                oracle_feature_weight=oracle_feature_weight,
            )
            improved = objective > best_objective
            if improved.any():
                best_objective[improved] = objective[improved]
                best_adv[improved] = adv_inputs[improved].detach().clone()
                best_logits[improved] = logits[improved].detach().clone()

        for step_idx in range(steps):
            adv_inputs.requires_grad_(True)
            logits = model(normalizer(adv_inputs))
            objective = _oracle_guided_objective(
                model=model,
                normalizer=normalizer,
                adv_inputs=adv_inputs,
                logits=logits,
                target_class=target_class,
                oracle_feats=oracle_feats,
                oracle_feature_weight=oracle_feature_weight,
            )
            grad = torch.autograd.grad(objective.sum(), adv_inputs, only_inputs=True)[0]

            with torch.no_grad():
                if norm == "l_inf":
                    direction = grad.sign()
                elif norm == "l2":
                    flat = grad.view(grad.size(0), -1)
                    grad_norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
                    direction = (flat / grad_norm).view_as(grad)
                else:
                    raise ValueError(f"Unsupported norm: {norm}")

                candidate = adv_inputs + step_size * direction
                candidate = _project_ball(candidate, inputs, eps, norm)
                if step_idx > 0 and apgd_momentum > 0.0:
                    candidate = candidate + apgd_momentum * (candidate - prev_adv)
                    candidate = _project_ball(candidate, inputs, eps, norm)

                prev_adv = adv_inputs.detach().clone()
                adv_inputs = candidate.detach()

                candidate_logits = model(normalizer(adv_inputs))
                candidate_objective = _oracle_guided_objective(
                    model=model,
                    normalizer=normalizer,
                    adv_inputs=adv_inputs,
                    logits=candidate_logits,
                    target_class=target_class,
                    oracle_feats=oracle_feats,
                    oracle_feature_weight=oracle_feature_weight,
                )
                improved = candidate_objective > best_objective
                if improved.any():
                    best_objective[improved] = candidate_objective[improved]
                    best_adv[improved] = adv_inputs[improved].detach().clone()
                    best_logits[improved] = candidate_logits[improved].detach().clone()

                if ((step_idx + 1) % check_interval == 0) and (step_idx + 1 < steps):
                    stalled = best_objective <= (checkpoint_objective + 1e-4)
                    if stalled.any():
                        step_size[stalled] = torch.clamp(step_size[stalled] * apgd_rho, min=alpha_min)
                        adv_inputs[stalled] = best_adv[stalled].detach().clone()
                        prev_adv[stalled] = best_adv[stalled].detach().clone()
                    checkpoint_objective = torch.maximum(checkpoint_objective, best_objective)

    success = bool((best_logits.argmax(dim=1) == target_class).all().item())
    return best_adv, best_logits, success


def _run_adam_margin_attack(
    model: torch.nn.Module,
    normalizer: torch.nn.Module,
    inputs: torch.Tensor,
    target_class: int,
    eps: float,
    steps: int,
    restarts: int,
    norm: str,
    alpha_ratio: float,
    alpha_min: float,
    oracle_model: Optional[torch.nn.Module] = None,
    oracle_feature_weight: float = 0.0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    initial_advs: Optional[Sequence[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    if eps <= 0.0:
        logits, preds = _predict(model, normalizer, inputs)
        success = bool((preds == target_class).all().item())
        return inputs.detach().clone(), logits, success

    base_lr = min(max(alpha_min, eps * alpha_ratio), eps)
    batch_size = inputs.size(0)
    best_adv = inputs.detach().clone()
    best_logits, _ = _predict(model, normalizer, best_adv)
    best_objective = torch.full((batch_size,), float("-inf"), device=inputs.device)

    oracle_feats = None
    if oracle_model is not None and oracle_feature_weight > 0.0:
        with torch.no_grad():
            oracle_feats = _extract_normalized_features(oracle_model, normalizer, inputs).detach()

    init_list = list(initial_advs) if initial_advs is not None else []
    total_restarts = max(restarts, len(init_list))

    for restart_idx in range(total_restarts):
        if restart_idx < len(init_list):
            adv_inputs = _project_ball(init_list[restart_idx].detach().clone(), inputs, eps, norm)
        else:
            adv_inputs = _random_init(inputs.detach(), eps, norm)
        m = torch.zeros_like(adv_inputs)
        v = torch.zeros_like(adv_inputs)

        for step_idx in range(steps):
            adv_inputs.requires_grad_(True)
            logits = model(normalizer(adv_inputs))
            objective = _oracle_guided_objective(
                model=model,
                normalizer=normalizer,
                adv_inputs=adv_inputs,
                logits=logits,
                target_class=target_class,
                oracle_feats=oracle_feats,
                oracle_feature_weight=oracle_feature_weight,
            )
            grad = torch.autograd.grad(objective.sum(), adv_inputs, only_inputs=True)[0]

            with torch.no_grad():
                m = adam_beta1 * m + (1.0 - adam_beta1) * grad
                v = adam_beta2 * v + (1.0 - adam_beta2) * grad.pow(2)
                m_hat = m / (1.0 - adam_beta1 ** (step_idx + 1))
                v_hat = v / (1.0 - adam_beta2 ** (step_idx + 1))
                update = m_hat / (v_hat.sqrt() + 1e-8)

                if norm == "l_inf":
                    adv_inputs = adv_inputs + base_lr * update.sign()
                elif norm == "l2":
                    flat = update.view(update.size(0), -1)
                    update_norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
                    direction = (flat / update_norm).view_as(update)
                    adv_inputs = adv_inputs + base_lr * direction
                else:
                    raise ValueError(f"Unsupported norm: {norm}")

                adv_inputs = _project_ball(adv_inputs, inputs, eps, norm).detach()

                candidate_logits = model(normalizer(adv_inputs))
                candidate_objective = _oracle_guided_objective(
                    model=model,
                    normalizer=normalizer,
                    adv_inputs=adv_inputs,
                    logits=candidate_logits,
                    target_class=target_class,
                    oracle_feats=oracle_feats,
                    oracle_feature_weight=oracle_feature_weight,
                )
                improved = candidate_objective > best_objective
                if improved.any():
                    best_objective[improved] = candidate_objective[improved]
                    best_adv[improved] = adv_inputs[improved].detach().clone()
                    best_logits[improved] = candidate_logits[improved].detach().clone()

    success = bool((best_logits.argmax(dim=1) == target_class).all().item())
    return best_adv, best_logits, success


def _run_box_adam_margin_attack(
    model: torch.nn.Module,
    normalizer: torch.nn.Module,
    inputs: torch.Tensor,
    target_class: int,
    eps: float,
    steps: int,
    restarts: int,
    norm: str,
    alpha_ratio: float,
    alpha_min: float,
    oracle_model: Optional[torch.nn.Module] = None,
    oracle_feature_weight: float = 0.0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    if eps <= 0.0:
        logits, preds = _predict(model, normalizer, inputs)
        success = bool((preds == target_class).all().item())
        return inputs.detach().clone(), logits, success
    if norm != "l_inf":
        raise ValueError("adam_margin_box currently supports only l_inf norm")

    step_ratio = min(max(alpha_min / max(eps, 1e-12), alpha_ratio), 1.0)
    batch_size = inputs.size(0)
    lower, upper = _box_bounds(inputs.detach(), eps)
    best_adv = inputs.detach().clone()
    best_logits, _ = _predict(model, normalizer, best_adv)
    best_objective = torch.full((batch_size,), float("-inf"), device=inputs.device)

    oracle_feats = None
    if oracle_model is not None and oracle_feature_weight > 0.0:
        with torch.no_grad():
            oracle_feats = _extract_normalized_features(oracle_model, normalizer, inputs).detach()

    for restart_idx in range(restarts):
        if restart_idx == 0:
            adv_init = inputs.detach().clone()
        else:
            adv_init = lower + torch.rand_like(inputs) * (upper - lower)
        latent = _box_inverse(adv_init, lower, upper).detach()
        m = torch.zeros_like(latent)
        v = torch.zeros_like(latent)

        for step_idx in range(steps):
            latent = latent.detach().clone().requires_grad_(True)
            adv_inputs = _box_parameterize(latent, lower, upper)
            logits = model(normalizer(adv_inputs))
            objective = _oracle_guided_objective(
                model=model,
                normalizer=normalizer,
                adv_inputs=adv_inputs,
                logits=logits,
                target_class=target_class,
                oracle_feats=oracle_feats,
                oracle_feature_weight=oracle_feature_weight,
            )
            grad_latent = torch.autograd.grad(objective.sum(), latent, only_inputs=True)[0]

            with torch.no_grad():
                m = adam_beta1 * m + (1.0 - adam_beta1) * grad_latent
                v = adam_beta2 * v + (1.0 - adam_beta2) * grad_latent.pow(2)
                m_hat = m / (1.0 - adam_beta1 ** (step_idx + 1))
                v_hat = v / (1.0 - adam_beta2 ** (step_idx + 1))
                latent = (latent + step_ratio * (m_hat / (v_hat.sqrt() + 1e-8))).detach()

                candidate_adv = _box_parameterize(latent, lower, upper).detach()
                candidate_logits = model(normalizer(candidate_adv))
                candidate_objective = _oracle_guided_objective(
                    model=model,
                    normalizer=normalizer,
                    adv_inputs=candidate_adv,
                    logits=candidate_logits,
                    target_class=target_class,
                    oracle_feats=oracle_feats,
                    oracle_feature_weight=oracle_feature_weight,
                )
                improved = candidate_objective > best_objective
                if improved.any():
                    best_objective[improved] = candidate_objective[improved]
                    best_adv[improved] = candidate_adv[improved].detach().clone()
                    best_logits[improved] = candidate_logits[improved].detach().clone()

    success = bool((best_logits.argmax(dim=1) == target_class).all().item())
    return best_adv, best_logits, success


def _run_bundle_warmstart_adam_attack(
    model: torch.nn.Module,
    normalizer: torch.nn.Module,
    inputs: torch.Tensor,
    target_class: int,
    eps: float,
    steps: int,
    restarts: int,
    norm: str,
    alpha_ratio: float,
    alpha_min: float,
    oracle_model: Optional[torch.nn.Module] = None,
    oracle_feature_weight: float = 0.0,
    bundle_margin_gate: Optional[float] = -3.0,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    if eps <= 0.0:
        logits, preds = _predict(model, normalizer, inputs)
        success = bool((preds == target_class).all().item())
        return inputs.detach().clone(), logits, success

    warm_steps = max(4, steps // 3)
    refine_steps = max(steps - warm_steps, 1)
    warm_adv, _warm_logits, _warm_success = _run_logit_bundle_hybrid_attack(
        model=model,
        normalizer=normalizer,
        inputs=inputs,
        target_class=target_class,
        eps=eps,
        steps=warm_steps,
        restarts=1,
        norm=norm,
        alpha_ratio=alpha_ratio,
        alpha_min=alpha_min,
        oracle_model=oracle_model,
        orthogonal_bundle=True,
        adaptive_bundle=True,
        topk_competitors=3,
        coordinatewise_precondition=True,
        accepted_step=True,
        accepted_margin_gate=bundle_margin_gate,
        debug_context=None,
    )
    return _run_adam_margin_attack(
        model=model,
        normalizer=normalizer,
        inputs=inputs,
        target_class=target_class,
        eps=eps,
        steps=refine_steps,
        restarts=restarts,
        norm=norm,
        alpha_ratio=alpha_ratio,
        alpha_min=alpha_min,
        oracle_model=oracle_model,
        oracle_feature_weight=oracle_feature_weight,
        initial_advs=[warm_adv],
    )


def _run_targeted_attack(
    model: torch.nn.Module,
    normalizer: torch.nn.Module,
    inputs: torch.Tensor,
    target_class: int,
    eps: float,
    steps: int,
    restarts: int,
    norm: str,
    alpha_ratio: float,
    alpha_min: float,
    attack_loss: str,
    oracle_model: Optional[torch.nn.Module] = None,
    oracle_feature_weight: float = 0.0,
    apgd_rho: float = 0.5,
    apgd_momentum: float = 0.75,
    bundle_margin_gate: Optional[float] = None,
    attack_state: Optional[Dict[str, torch.Tensor]] = None,
    debug_context: Optional[Dict[str, object]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    if attack_loss == "margin":
        return _run_margin_attack(
            model=model,
            normalizer=normalizer,
            inputs=inputs,
            target_class=target_class,
            eps=eps,
            steps=steps,
            restarts=restarts,
            norm=norm,
            alpha_ratio=alpha_ratio,
            alpha_min=alpha_min,
        )

    if attack_loss == "margin_oracle":
        if oracle_model is None:
            raise ValueError("margin_oracle attack requires an oracle model")
        return _run_margin_attack(
            model=model,
            normalizer=normalizer,
            inputs=inputs,
            target_class=target_class,
            eps=eps,
            steps=steps,
            restarts=restarts,
            norm=norm,
            alpha_ratio=alpha_ratio,
            alpha_min=alpha_min,
            oracle_model=oracle_model,
            oracle_feature_weight=oracle_feature_weight,
        )

    if attack_loss == "apgd_margin":
        return _run_apgd_margin_attack(
            model=model,
            normalizer=normalizer,
            inputs=inputs,
            target_class=target_class,
            eps=eps,
            steps=steps,
            restarts=restarts,
            norm=norm,
            alpha_ratio=alpha_ratio,
            alpha_min=alpha_min,
            apgd_rho=apgd_rho,
            apgd_momentum=apgd_momentum,
        )

    if attack_loss == "adam_margin":
        return _run_adam_margin_attack(
            model=model,
            normalizer=normalizer,
            inputs=inputs,
            target_class=target_class,
            eps=eps,
            steps=steps,
            restarts=restarts,
            norm=norm,
            alpha_ratio=alpha_ratio,
            alpha_min=alpha_min,
        )

    if attack_loss == "adam_margin_box":
        return _run_box_adam_margin_attack(
            model=model,
            normalizer=normalizer,
            inputs=inputs,
            target_class=target_class,
            eps=eps,
            steps=steps,
            restarts=restarts,
            norm=norm,
            alpha_ratio=alpha_ratio,
            alpha_min=alpha_min,
        )

    if attack_loss == "adam_margin_bundle_warm":
        return _run_bundle_warmstart_adam_attack(
            model=model,
            normalizer=normalizer,
            inputs=inputs,
            target_class=target_class,
            eps=eps,
            steps=steps,
            restarts=restarts,
            norm=norm,
            alpha_ratio=alpha_ratio,
            alpha_min=alpha_min,
            bundle_margin_gate=bundle_margin_gate,
        )

    if attack_loss == "logit_subspace_adam":
        if attack_state is None:
            raise ValueError("logit_subspace_adam attack requires a prepared attack_state")
        return _run_logit_subspace_adam_attack(
            model=model,
            normalizer=normalizer,
            inputs=inputs,
            target_class=target_class,
            eps=eps,
            steps=steps,
            restarts=restarts,
            norm=norm,
            alpha_ratio=alpha_ratio,
            alpha_min=alpha_min,
            attack_state=attack_state,
            oracle_guidance=False,
        )

    if attack_loss == "logit_gap_gn_oracle":
        if oracle_model is None:
            raise ValueError("logit_gap_gn_oracle attack requires an oracle model")
        return _run_logit_gap_gn_attack(
            model=model,
            normalizer=normalizer,
            inputs=inputs,
            target_class=target_class,
            eps=eps,
            steps=steps,
            restarts=restarts,
            norm=norm,
            alpha_ratio=alpha_ratio,
            alpha_min=alpha_min,
            oracle_model=oracle_model,
        )

    if attack_loss == "logit_bundle_gn":
        return _run_logit_bundle_gn_attack(
            model=model,
            normalizer=normalizer,
            inputs=inputs,
            target_class=target_class,
            eps=eps,
            steps=steps,
            restarts=restarts,
            norm=norm,
            alpha_ratio=alpha_ratio,
            alpha_min=alpha_min,
            oracle_model=None,
            debug_context=debug_context,
        )

    if attack_loss == "logit_bundle_gn_oracle":
        if oracle_model is None:
            raise ValueError("logit_bundle_gn_oracle attack requires an oracle model")
        return _run_logit_bundle_gn_attack(
            model=model,
            normalizer=normalizer,
            inputs=inputs,
            target_class=target_class,
            eps=eps,
            steps=steps,
            restarts=restarts,
            norm=norm,
            alpha_ratio=alpha_ratio,
            alpha_min=alpha_min,
            oracle_model=oracle_model,
            debug_context=debug_context,
        )

    if attack_loss == "logit_bundle_hybrid":
        return _run_logit_bundle_hybrid_attack(
            model=model,
            normalizer=normalizer,
            inputs=inputs,
            target_class=target_class,
            eps=eps,
            steps=steps,
            restarts=restarts,
            norm=norm,
            alpha_ratio=alpha_ratio,
            alpha_min=alpha_min,
            oracle_model=None,
            debug_context=debug_context,
        )

    if attack_loss == "logit_bundle_orth":
        return _run_logit_bundle_hybrid_attack(
            model=model,
            normalizer=normalizer,
            inputs=inputs,
            target_class=target_class,
            eps=eps,
            steps=steps,
            restarts=restarts,
            norm=norm,
            alpha_ratio=alpha_ratio,
            alpha_min=alpha_min,
            oracle_model=None,
            orthogonal_bundle=True,
            debug_context=debug_context,
        )

    if attack_loss == "logit_bundle_adaptive":
        return _run_logit_bundle_hybrid_attack(
            model=model,
            normalizer=normalizer,
            inputs=inputs,
            target_class=target_class,
            eps=eps,
            steps=steps,
            restarts=restarts,
            norm=norm,
            alpha_ratio=alpha_ratio,
            alpha_min=alpha_min,
            oracle_model=None,
            orthogonal_bundle=True,
            adaptive_bundle=True,
            debug_context=debug_context,
        )

    if attack_loss == "logit_bundle_topk":
        return _run_logit_bundle_hybrid_attack(
            model=model,
            normalizer=normalizer,
            inputs=inputs,
            target_class=target_class,
            eps=eps,
            steps=steps,
            restarts=restarts,
            norm=norm,
            alpha_ratio=alpha_ratio,
            alpha_min=alpha_min,
            oracle_model=None,
            orthogonal_bundle=True,
            adaptive_bundle=True,
            topk_competitors=3,
            coordinatewise_precondition=True,
            debug_context=debug_context,
        )

    if attack_loss == "logit_bundle_accept":
        return _run_logit_bundle_hybrid_attack(
            model=model,
            normalizer=normalizer,
            inputs=inputs,
            target_class=target_class,
            eps=eps,
            steps=steps,
            restarts=restarts,
            norm=norm,
            alpha_ratio=alpha_ratio,
            alpha_min=alpha_min,
            oracle_model=None,
            orthogonal_bundle=True,
            adaptive_bundle=True,
            topk_competitors=3,
            coordinatewise_precondition=True,
            accepted_step=True,
            debug_context=debug_context,
        )

    if attack_loss == "logit_bundle_boundary":
        return _run_logit_bundle_hybrid_attack(
            model=model,
            normalizer=normalizer,
            inputs=inputs,
            target_class=target_class,
            eps=eps,
            steps=steps,
            restarts=restarts,
            norm=norm,
            alpha_ratio=alpha_ratio,
            alpha_min=alpha_min,
            oracle_model=None,
            orthogonal_bundle=True,
            adaptive_bundle=True,
            topk_competitors=3,
            coordinatewise_precondition=True,
            accepted_step=True,
            accepted_margin_gate=bundle_margin_gate,
            debug_context=debug_context,
        )

    if attack_loss == "logit_bundle_hybrid_oracle":
        if oracle_model is None:
            raise ValueError("logit_bundle_hybrid_oracle attack requires an oracle model")
        return _run_logit_bundle_hybrid_attack(
            model=model,
            normalizer=normalizer,
            inputs=inputs,
            target_class=target_class,
            eps=eps,
            steps=steps,
            restarts=restarts,
            norm=norm,
            alpha_ratio=alpha_ratio,
            alpha_min=alpha_min,
            oracle_model=oracle_model,
            debug_context=debug_context,
        )

    if attack_loss == "logit_bundle_orth_oracle":
        if oracle_model is None:
            raise ValueError("logit_bundle_orth_oracle attack requires an oracle model")
        return _run_logit_bundle_hybrid_attack(
            model=model,
            normalizer=normalizer,
            inputs=inputs,
            target_class=target_class,
            eps=eps,
            steps=steps,
            restarts=restarts,
            norm=norm,
            alpha_ratio=alpha_ratio,
            alpha_min=alpha_min,
            oracle_model=oracle_model,
            orthogonal_bundle=True,
            debug_context=debug_context,
        )

    if attack_loss == "logit_bundle_topk_oracle":
        if oracle_model is None:
            raise ValueError("logit_bundle_topk_oracle attack requires an oracle model")
        return _run_logit_bundle_hybrid_attack(
            model=model,
            normalizer=normalizer,
            inputs=inputs,
            target_class=target_class,
            eps=eps,
            steps=steps,
            restarts=restarts,
            norm=norm,
            alpha_ratio=alpha_ratio,
            alpha_min=alpha_min,
            oracle_model=oracle_model,
            orthogonal_bundle=True,
            adaptive_bundle=True,
            topk_competitors=3,
            coordinatewise_precondition=True,
            debug_context=debug_context,
        )

    if attack_loss == "logit_bundle_accept_oracle":
        if oracle_model is None:
            raise ValueError("logit_bundle_accept_oracle attack requires an oracle model")
        return _run_logit_bundle_hybrid_attack(
            model=model,
            normalizer=normalizer,
            inputs=inputs,
            target_class=target_class,
            eps=eps,
            steps=steps,
            restarts=restarts,
            norm=norm,
            alpha_ratio=alpha_ratio,
            alpha_min=alpha_min,
            oracle_model=oracle_model,
            orthogonal_bundle=True,
            adaptive_bundle=True,
            topk_competitors=3,
            coordinatewise_precondition=True,
            accepted_step=True,
            debug_context=debug_context,
        )

    if attack_loss == "logit_bundle_boundary_oracle":
        if oracle_model is None:
            raise ValueError("logit_bundle_boundary_oracle attack requires an oracle model")
        return _run_logit_bundle_hybrid_attack(
            model=model,
            normalizer=normalizer,
            inputs=inputs,
            target_class=target_class,
            eps=eps,
            steps=steps,
            restarts=restarts,
            norm=norm,
            alpha_ratio=alpha_ratio,
            alpha_min=alpha_min,
            oracle_model=oracle_model,
            orthogonal_bundle=True,
            adaptive_bundle=True,
            topk_competitors=3,
            coordinatewise_precondition=True,
            accepted_step=True,
            accepted_margin_gate=bundle_margin_gate,
            debug_context=debug_context,
        )

    if attack_loss == "apgd_margin_oracle":
        if oracle_model is None:
            raise ValueError("apgd_margin_oracle attack requires an oracle model")
        return _run_apgd_margin_attack(
            model=model,
            normalizer=normalizer,
            inputs=inputs,
            target_class=target_class,
            eps=eps,
            steps=steps,
            restarts=restarts,
            norm=norm,
            alpha_ratio=alpha_ratio,
            alpha_min=alpha_min,
            oracle_model=oracle_model,
            oracle_feature_weight=oracle_feature_weight,
            apgd_rho=apgd_rho,
            apgd_momentum=apgd_momentum,
        )

    if attack_loss == "adam_margin_oracle":
        if oracle_model is None:
            raise ValueError("adam_margin_oracle attack requires an oracle model")
        return _run_adam_margin_attack(
            model=model,
            normalizer=normalizer,
            inputs=inputs,
            target_class=target_class,
            eps=eps,
            steps=steps,
            restarts=restarts,
            norm=norm,
            alpha_ratio=alpha_ratio,
            alpha_min=alpha_min,
            oracle_model=oracle_model,
            oracle_feature_weight=oracle_feature_weight,
        )

    if attack_loss == "adam_margin_box_oracle":
        if oracle_model is None:
            raise ValueError("adam_margin_box_oracle attack requires an oracle model")
        return _run_box_adam_margin_attack(
            model=model,
            normalizer=normalizer,
            inputs=inputs,
            target_class=target_class,
            eps=eps,
            steps=steps,
            restarts=restarts,
            norm=norm,
            alpha_ratio=alpha_ratio,
            alpha_min=alpha_min,
            oracle_model=oracle_model,
            oracle_feature_weight=oracle_feature_weight,
        )

    if attack_loss == "adam_margin_bundle_warm_oracle":
        if oracle_model is None:
            raise ValueError("adam_margin_bundle_warm_oracle attack requires an oracle model")
        return _run_bundle_warmstart_adam_attack(
            model=model,
            normalizer=normalizer,
            inputs=inputs,
            target_class=target_class,
            eps=eps,
            steps=steps,
            restarts=restarts,
            norm=norm,
            alpha_ratio=alpha_ratio,
            alpha_min=alpha_min,
            oracle_model=oracle_model,
            oracle_feature_weight=oracle_feature_weight,
            bundle_margin_gate=bundle_margin_gate,
        )

    if attack_loss == "logit_subspace_adam_oracle":
        if attack_state is None:
            raise ValueError("logit_subspace_adam_oracle attack requires a prepared attack_state")
        return _run_logit_subspace_adam_attack(
            model=model,
            normalizer=normalizer,
            inputs=inputs,
            target_class=target_class,
            eps=eps,
            steps=steps,
            restarts=restarts,
            norm=norm,
            alpha_ratio=alpha_ratio,
            alpha_min=alpha_min,
            attack_state=attack_state,
            oracle_guidance=True,
        )

    if attack_loss != "ce":
        raise ValueError(f"Unsupported attack_loss: {attack_loss}")

    if eps <= 0.0:
        logits, preds = _predict(model, normalizer, inputs)
        success = bool((preds == target_class).all().item())
        return inputs.detach().clone(), logits, success

    wrapped_model = torch.nn.Sequential(normalizer, model)
    alpha = max(alpha_min, eps * alpha_ratio)
    attack = PGDAttack(
        wrapped_model,
        eps=eps,
        alpha=alpha,
        steps=steps,
        random_start=True,
        norm=norm,
        restarts=restarts,
    )
    attack._debug_logged = True
    labels = torch.zeros(inputs.size(0), dtype=torch.long, device=inputs.device)
    target_labels = torch.full_like(labels, target_class)
    adv_inputs = attack(inputs, labels, targeted=True, target_labels=target_labels)
    logits, preds = _predict(model, normalizer, adv_inputs)
    success = bool((preds == target_class).all().item())
    return adv_inputs.detach(), logits.detach(), success


def _actual_delta_norm(inputs: torch.Tensor, adv_inputs: torch.Tensor, norm: str) -> float:
    delta = (adv_inputs - inputs).detach()
    if norm == "l_inf":
        return float(delta.abs().max().item())
    if norm == "l2":
        return float(delta.view(delta.size(0), -1).norm(p=2, dim=1).max().item())
    raise ValueError(f"Unsupported norm: {norm}")


def _vector_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    denom = a_flat.norm() * b_flat.norm()
    if float(denom.item()) < 1e-12:
        return float("nan")
    return float(torch.dot(a_flat, b_flat).div(denom).item())


def _direction_step(direction: torch.Tensor, step_size: float, norm: str) -> torch.Tensor:
    if norm == "l_inf":
        return step_size * direction.sign()
    if norm == "l2":
        flat = direction.view(direction.size(0), -1)
        direction_norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        return step_size * (flat / direction_norm).view_as(direction)
    raise ValueError(f"Unsupported norm: {norm}")


def _predicted_margin_gain(margin_grad: torch.Tensor, delta: torch.Tensor) -> float:
    return float((margin_grad.detach() * delta.detach()).sum().item())


def _effective_rank_from_gram(gram: torch.Tensor) -> float:
    eigvals = torch.linalg.eigvalsh(gram.float()).clamp_min(0.0)
    total = float(eigvals.sum().item())
    if total <= 0.0:
        return 0.0
    probs = eigvals / total
    probs = probs[probs > 0]
    entropy = -(probs * probs.log()).sum()
    return float(entropy.exp().item())


def _spectrum_mass_from_gram(gram: torch.Tensor, topk: int) -> float:
    eigvals = torch.linalg.eigvalsh(gram.float()).clamp_min(0.0)
    total = float(eigvals.sum().item())
    if total <= 0.0:
        return 0.0
    k = min(int(topk), int(eigvals.numel()))
    return float(eigvals.topk(k).values.sum().div(total).item())


def _topk_logit_payload(logits_vec: torch.Tensor, topk: int) -> Dict[str, List[float]]:
    k = min(int(topk), int(logits_vec.numel()))
    vals, idxs = torch.topk(logits_vec.detach().float(), k)
    return {
        "topk_indices": [int(v) for v in idxs.tolist()],
        "topk_logits": [float(v) for v in vals.tolist()],
    }


def _topk_competitor_shift(
    logits_vec: torch.Tensor,
    target_class: int,
    target_margin_bias: float,
    topk_competitors: int,
) -> Tuple[torch.Tensor, List[int], List[float], List[float]]:
    competitor_logits = logits_vec.clone()
    competitor_logits[target_class] = float("-inf")
    k = min(int(topk_competitors), int(logits_vec.numel()) - 1)
    vals, idxs = torch.topk(competitor_logits, k)
    target_logit = logits_vec[target_class]
    margins = target_logit - vals
    needed = torch.clamp(target_margin_bias - margins, min=0.0)

    if float(needed.sum().item()) <= 1e-12:
        weights = torch.softmax(vals.detach().float(), dim=0)
        weighted_needed = weights * 0.0
    else:
        weights = torch.softmax(vals.detach().float(), dim=0)
        weighted_needed = weights * needed.float()

    desired_shift = torch.zeros_like(logits_vec)
    desired_shift[target_class] = weighted_needed.sum()
    for comp_idx, amount in zip(idxs.tolist(), weighted_needed.tolist()):
        desired_shift[int(comp_idx)] -= float(amount)

    return (
        desired_shift,
        [int(v) for v in idxs.tolist()],
        [float(v) for v in weights.tolist()],
        [float(v) for v in margins.tolist()],
    )


def _maybe_append_geometry_row(debug_context: Optional[Dict[str, object]], row: Dict[str, object]) -> None:
    if not debug_context or not bool(debug_context.get("enabled", False)):
        return
    rows = debug_context.get("rows")
    if not isinstance(rows, list):
        return
    payload = {
        "suite_name": debug_context.get("suite_name"),
        "sample_type": debug_context.get("sample_type"),
        "sample_index": int(debug_context.get("sample_index", -1)),
        "attack_loss": debug_context.get("attack_loss"),
        "search_phase": debug_context.get("search_phase"),
        "search_iter": int(debug_context.get("search_iter", -1)),
        "eps": float(debug_context.get("eps", 0.0)),
    }
    payload.update(row)
    rows.append(payload)


def _summarize_geometry_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (str(row.get("suite_name")), str(row.get("sample_type")))
        grouped.setdefault(key, []).append(row)

    def _mean(vals: Sequence[float]) -> float:
        vals = [float(v) for v in vals if v == v]
        if not vals:
            return float("nan")
        return float(sum(vals) / len(vals))

    summary: Dict[str, Dict[str, float]] = {}
    for (suite_name, sample_type), subset in grouped.items():
        switch_total = 0
        switch_denom = 0
        traces: Dict[Tuple[int, str, int], List[Dict[str, object]]] = {}
        for row in subset:
            trace_key = (
                int(row.get("sample_index", -1)),
                str(row.get("search_phase")),
                int(row.get("search_iter", -1)),
            )
            traces.setdefault(trace_key, []).append(row)
        for trace_rows in traces.values():
            trace_rows = sorted(trace_rows, key=lambda r: (int(r.get("restart", 0)), int(r.get("step", 0))))
            prev_comp = None
            for row in trace_rows:
                comp = row.get("competitor")
                if prev_comp is not None:
                    switch_denom += 1
                    if comp != prev_comp:
                        switch_total += 1
                prev_comp = comp

        margin_wins = sum(1 for row in subset if row.get("selected_candidate") == "margin")
        margin_forced_wins = sum(1 for row in subset if row.get("selected_candidate") == "margin_forced")
        hybrid_wins = sum(1 for row in subset if row.get("selected_candidate") == "hybrid")
        total_wins = margin_wins + margin_forced_wins + hybrid_wins

        summary[f"{suite_name}::{sample_type}"] = {
            "n_rows": float(len(subset)),
            "mean_margin": _mean([row.get("margin", float("nan")) for row in subset]),
            "mean_target_prob": _mean([row.get("target_prob", float("nan")) for row in subset]),
            "mean_cos_margin_bundle": _mean([row.get("cos_margin_bundle", float("nan")) for row in subset]),
            "mean_pred_gain_margin": _mean([row.get("pred_gain_margin", float("nan")) for row in subset]),
            "mean_pred_gain_bundle": _mean([row.get("pred_gain_bundle", float("nan")) for row in subset]),
            "mean_pred_gain_hybrid": _mean([row.get("pred_gain_hybrid", float("nan")) for row in subset]),
            "mean_pred_gain_selected": _mean([row.get("pred_gain_selected", float("nan")) for row in subset]),
            "mean_actual_margin_gain": _mean([row.get("actual_margin_gain", float("nan")) for row in subset]),
            "mean_preproj_postproj_cos": _mean([row.get("preproj_postproj_cos", float("nan")) for row in subset]),
            "mean_projection_ratio": _mean([row.get("projection_ratio", float("nan")) for row in subset]),
            "mean_jacobian_eff_rank": _mean([row.get("jacobian_eff_rank", float("nan")) for row in subset]),
            "mean_jacobian_top1_mass": _mean([row.get("jacobian_top1_mass", float("nan")) for row in subset]),
            "mean_jacobian_top3_mass": _mean([row.get("jacobian_top3_mass", float("nan")) for row in subset]),
            "boundary_gate_active_rate": _mean([1.0 if row.get("boundary_gate_active") else 0.0 for row in subset]),
            "hybrid_considered_rate": _mean([1.0 if row.get("hybrid_considered") else 0.0 for row in subset]),
            "competitor_switch_rate": float(switch_total / switch_denom) if switch_denom > 0 else float("nan"),
            "margin_candidate_win_rate": float(margin_wins / total_wins) if total_wins > 0 else float("nan"),
            "margin_forced_win_rate": float(margin_forced_wins / total_wins) if total_wins > 0 else float("nan"),
            "hybrid_candidate_win_rate": float(hybrid_wins / total_wins) if total_wins > 0 else float("nan"),
        }
    return summary


def _compute_logit_jacobian(
    model: torch.nn.Module,
    normalizer: torch.nn.Module,
    inputs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = inputs.detach().clone().requires_grad_(True)
    logits = model(normalizer(x)).squeeze(0)
    rows: List[torch.Tensor] = []
    for idx in range(logits.numel()):
        grad_i = torch.autograd.grad(
            logits[idx],
            x,
            retain_graph=(idx + 1) < logits.numel(),
            create_graph=False,
            allow_unused=False,
        )[0]
        rows.append(grad_i.reshape(-1).detach())
    jac = torch.stack(rows, dim=0)
    return logits.detach(), jac


def _logit_jacobian_state(
    model: torch.nn.Module,
    normalizer: torch.nn.Module,
    inputs: torch.Tensor,
    oracle_model: Optional[torch.nn.Module] = None,
) -> Dict[str, torch.Tensor]:
    logits, jac = _compute_logit_jacobian(model, normalizer, inputs)
    gram = (jac @ jac.T).float()

    eigvals, eigvecs = torch.linalg.eigh(gram)
    keep = eigvals > 1e-8
    basis_vectors: List[torch.Tensor] = []
    for eigval, eigvec in zip(eigvals[keep], eigvecs[:, keep].T):
        direction = jac.T @ eigvec
        direction = direction / eigval.sqrt().clamp_min(1e-8)
        direction = direction / direction.norm().clamp_min(1e-8)
        basis_vectors.append(direction)

    if basis_vectors:
        basis = torch.stack(basis_vectors, dim=1)
    else:
        basis = torch.zeros(jac.size(1), 0, device=jac.device, dtype=jac.dtype)

    state: Dict[str, torch.Tensor] = {
        "basis": basis.detach(),
    }

    if oracle_model is not None:
        with torch.no_grad():
            oracle_logits = oracle_model(normalizer(inputs)).squeeze(0).detach()
        gap = (oracle_logits - logits.detach()).float()
        coeff_gap = torch.linalg.pinv(gram, hermitian=True) @ gap
        delta_gap = jac.T @ coeff_gap
        alpha0 = basis.T @ delta_gap if basis.numel() > 0 else torch.zeros(0, device=jac.device, dtype=jac.dtype)
        state["oracle_logits"] = oracle_logits
        state["alpha0"] = alpha0.detach()
    else:
        state["alpha0"] = torch.zeros(basis.size(1), device=jac.device, dtype=jac.dtype)

    return state


def _run_logit_gap_gn_attack(
    model: torch.nn.Module,
    normalizer: torch.nn.Module,
    inputs: torch.Tensor,
    target_class: int,
    eps: float,
    steps: int,
    restarts: int,
    norm: str,
    alpha_ratio: float,
    alpha_min: float,
    oracle_model: torch.nn.Module,
    gn_reg: float = 1e-3,
    target_margin_bias: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    if eps <= 0.0:
        logits, preds = _predict(model, normalizer, inputs)
        success = bool((preds == target_class).all().item())
        return inputs.detach().clone(), logits, success

    with torch.no_grad():
        oracle_logits = oracle_model(normalizer(inputs)).squeeze(0).detach()

    step_size = min(max(alpha_min, eps * alpha_ratio), eps)
    best_adv = inputs.detach().clone()
    best_logits, _ = _predict(model, normalizer, best_adv)
    best_objective = torch.full((1,), float("-inf"), device=inputs.device)

    eye = torch.eye(oracle_logits.numel(), device=inputs.device)
    target_bias = eye[target_class] * target_margin_bias

    for restart_idx in range(restarts):
        adv_inputs = inputs.detach().clone() if restart_idx == 0 else _random_init(inputs.detach(), eps, norm)
        velocity = torch.zeros_like(adv_inputs)

        for _step in range(steps):
            logits_vec, jac = _compute_logit_jacobian(model, normalizer, adv_inputs)
            gap = (oracle_logits - logits_vec).float() + target_bias
            gram = (jac @ jac.T).float()
            coeff = torch.linalg.solve(gram + (gn_reg * eye), gap)
            direction_flat = jac.T @ coeff
            direction = direction_flat.view_as(inputs)

            with torch.no_grad():
                if norm == "l_inf":
                    velocity = 0.75 * velocity + 0.25 * direction.sign()
                    adv_inputs = adv_inputs + step_size * velocity.sign()
                elif norm == "l2":
                    velocity = 0.75 * velocity + 0.25 * direction
                    flat = velocity.view(velocity.size(0), -1)
                    vel_norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
                    adv_inputs = adv_inputs + step_size * (flat / vel_norm).view_as(velocity)
                else:
                    raise ValueError(f"Unsupported norm: {norm}")

                adv_inputs = _project_ball(adv_inputs, inputs, eps, norm).detach()
                adv_logits = model(normalizer(adv_inputs)).detach()
                objective = _margin_objective(adv_logits, target_class) - 0.05 * (
                    adv_logits.squeeze(0) - oracle_logits
                ).pow(2).mean()
                if objective.item() > best_objective.item():
                    best_objective[:] = objective
                    best_adv = adv_inputs.clone()
                    best_logits = adv_logits.clone()

    success = bool((best_logits.argmax(dim=1) == target_class).all().item())
    return best_adv, best_logits, success


def _run_logit_bundle_gn_attack(
    model: torch.nn.Module,
    normalizer: torch.nn.Module,
    inputs: torch.Tensor,
    target_class: int,
    eps: float,
    steps: int,
    restarts: int,
    norm: str,
    alpha_ratio: float,
    alpha_min: float,
    oracle_model: Optional[torch.nn.Module] = None,
    bundle_reg: float = 1e-3,
    target_margin_bias: float = 1.0,
    oracle_mix: float = 0.2,
    debug_context: Optional[Dict[str, object]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    if eps <= 0.0:
        logits, preds = _predict(model, normalizer, inputs)
        success = bool((preds == target_class).all().item())
        return inputs.detach().clone(), logits, success

    oracle_logits = None
    if oracle_model is not None:
        with torch.no_grad():
            oracle_logits = oracle_model(normalizer(inputs)).squeeze(0).detach()

    step_size = min(max(alpha_min, eps * alpha_ratio), eps)
    best_adv = inputs.detach().clone()
    best_logits, _ = _predict(model, normalizer, best_adv)
    best_objective = torch.full((1,), float("-inf"), device=inputs.device)
    n_classes = int(best_logits.size(1))
    eye = torch.eye(n_classes, device=inputs.device)

    for restart_idx in range(restarts):
        adv_inputs = inputs.detach().clone() if restart_idx == 0 else _random_init(inputs.detach(), eps, norm)
        velocity = torch.zeros_like(adv_inputs)

        for _step in range(steps):
            logits_vec, jac = _compute_logit_jacobian(model, normalizer, adv_inputs)
            competitor_logits = logits_vec.clone()
            competitor_logits[target_class] = float("-inf")
            competitor = int(competitor_logits.argmax().item())
            current_margin = float((logits_vec[target_class] - logits_vec[competitor]).item())
            needed_margin = max(0.0, target_margin_bias - current_margin)

            desired_shift = torch.zeros_like(logits_vec)
            desired_shift[target_class] += 0.5 * needed_margin
            desired_shift[competitor] -= 0.5 * needed_margin
            if oracle_logits is not None:
                desired_shift = desired_shift + (oracle_mix * (oracle_logits - logits_vec))

            gram = (jac @ jac.T).float()
            coeff = torch.linalg.solve(gram + (bundle_reg * eye), desired_shift.float())
            direction_flat = jac.T @ coeff
            direction = direction_flat.view_as(inputs)

            with torch.no_grad():
                if norm == "l_inf":
                    velocity = 0.75 * velocity + 0.25 * direction.sign()
                    adv_inputs = adv_inputs + step_size * velocity.sign()
                elif norm == "l2":
                    velocity = 0.75 * velocity + 0.25 * direction
                    flat = velocity.view(velocity.size(0), -1)
                    vel_norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
                    adv_inputs = adv_inputs + step_size * (flat / vel_norm).view_as(velocity)
                else:
                    raise ValueError(f"Unsupported norm: {norm}")

                adv_inputs = _project_ball(adv_inputs, inputs, eps, norm).detach()
                adv_logits = model(normalizer(adv_inputs)).detach()
                objective = _margin_objective(adv_logits, target_class)
                if oracle_logits is not None:
                    objective = objective - 0.02 * (adv_logits.squeeze(0) - oracle_logits).pow(2).mean()
                if objective.item() > best_objective.item():
                    best_objective[:] = objective
                    best_adv = adv_inputs.clone()
                    best_logits = adv_logits.clone()

    success = bool((best_logits.argmax(dim=1) == target_class).all().item())
    return best_adv, best_logits, success


def _run_logit_bundle_hybrid_attack(
    model: torch.nn.Module,
    normalizer: torch.nn.Module,
    inputs: torch.Tensor,
    target_class: int,
    eps: float,
    steps: int,
    restarts: int,
    norm: str,
    alpha_ratio: float,
    alpha_min: float,
    oracle_model: Optional[torch.nn.Module] = None,
    bundle_reg: float = 1e-3,
    target_margin_bias: float = 1.0,
    bundle_mix: float = 0.35,
    oracle_mix: float = 0.1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    orthogonal_bundle: bool = False,
    adaptive_bundle: bool = False,
    topk_competitors: int = 1,
    coordinatewise_precondition: bool = False,
    accepted_step: bool = False,
    accepted_margin_gate: Optional[float] = None,
    debug_context: Optional[Dict[str, object]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    if eps <= 0.0:
        logits, preds = _predict(model, normalizer, inputs)
        success = bool((preds == target_class).all().item())
        return inputs.detach().clone(), logits, success

    oracle_logits = None
    if oracle_model is not None:
        with torch.no_grad():
            oracle_logits = oracle_model(normalizer(inputs)).squeeze(0).detach()

    base_lr = min(max(alpha_min, eps * alpha_ratio), eps)
    best_adv = inputs.detach().clone()
    best_logits, _ = _predict(model, normalizer, best_adv)
    best_objective = torch.full((1,), float("-inf"), device=inputs.device)
    n_classes = int(best_logits.size(1))
    eye = torch.eye(n_classes, device=inputs.device)

    for restart_idx in range(restarts):
        adv_inputs = inputs.detach().clone() if restart_idx == 0 else _random_init(inputs.detach(), eps, norm)
        m = torch.zeros_like(adv_inputs)
        v = torch.zeros_like(adv_inputs)

        for step_idx in range(steps):
            grad_inputs = adv_inputs.detach().clone().requires_grad_(True)
            logits_live = model(normalizer(grad_inputs))
            margin_obj = _margin_objective(logits_live, target_class)
            margin_grad = torch.autograd.grad(margin_obj.sum(), grad_inputs, only_inputs=True)[0].detach()

            logits_vec, jac = _compute_logit_jacobian(model, normalizer, adv_inputs)
            desired_shift, competitor_list, competitor_weights, competitor_margins = _topk_competitor_shift(
                logits_vec=logits_vec,
                target_class=target_class,
                target_margin_bias=target_margin_bias,
                topk_competitors=topk_competitors,
            )
            competitor = int(competitor_list[0])
            current_margin = float(competitor_margins[0])
            boundary_gate_active = accepted_margin_gate is None or current_margin > float(accepted_margin_gate)
            if oracle_logits is not None:
                desired_shift = desired_shift + (oracle_mix * (oracle_logits - logits_vec))

            gram = (jac @ jac.T).float()
            coeff = torch.linalg.solve(gram + (bundle_reg * eye), desired_shift.float())
            bundle_dir = (jac.T @ coeff).view_as(inputs).detach()

            margin_unit = margin_grad / margin_grad.view(margin_grad.size(0), -1).norm(p=2, dim=1, keepdim=True).view(-1, 1, 1, 1).clamp_min(1e-12)
            bundle_flat_raw = bundle_dir.view(bundle_dir.size(0), -1)
            margin_flat = margin_unit.view(margin_unit.size(0), -1)
            bundle_raw_norm = bundle_flat_raw.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            cos_raw = (bundle_flat_raw * margin_flat).sum(dim=1, keepdim=True) / bundle_raw_norm.clamp_min(1e-12)
            if orthogonal_bundle:
                bundle_flat = bundle_flat_raw - (cos_raw * margin_flat)
                bundle_dir = bundle_flat.view_as(bundle_dir)
            bundle_unit = bundle_dir / bundle_dir.view(bundle_dir.size(0), -1).norm(p=2, dim=1, keepdim=True).view(-1, 1, 1, 1).clamp_min(1e-12)
            bundle_mix_eff = bundle_mix
            if adaptive_bundle:
                bundle_mix_eff = bundle_mix * (1.0 - cos_raw.abs().view(-1, 1, 1, 1)).clamp(0.0, 1.0)
            if coordinatewise_precondition:
                bundle_scale = bundle_dir.detach().abs().mean().clamp_min(1e-12)
                bundle_gate = torch.tanh(bundle_dir / bundle_scale)
                hybrid_dir = margin_grad * (1.0 + bundle_mix_eff * bundle_gate)
            else:
                hybrid_dir = (1.0 - bundle_mix_eff) * margin_unit + bundle_mix_eff * bundle_unit

            with torch.no_grad():
                adv_before = adv_inputs.clone()
                delta_margin_pre = _direction_step(margin_unit, base_lr, norm)
                delta_bundle_pre = _direction_step(bundle_unit, base_lr, norm)

                def _candidate_from_state(direction: torch.Tensor, state_m: torch.Tensor, state_v: torch.Tensor):
                    cand_m = adam_beta1 * state_m + (1.0 - adam_beta1) * direction
                    cand_v = adam_beta2 * state_v + (1.0 - adam_beta2) * direction.pow(2)
                    cand_m_hat = cand_m / (1.0 - adam_beta1 ** (step_idx + 1))
                    cand_v_hat = cand_v / (1.0 - adam_beta2 ** (step_idx + 1))
                    cand_update = cand_m_hat / (cand_v_hat.sqrt() + 1e-8)
                    if norm == "l_inf":
                        adv_candidate = adv_before + base_lr * cand_update.sign()
                    elif norm == "l2":
                        flat = cand_update.view(cand_update.size(0), -1)
                        update_norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
                        adv_candidate = adv_before + base_lr * (flat / update_norm).view_as(cand_update)
                    else:
                        raise ValueError(f"Unsupported norm: {norm}")
                    adv_pre = adv_candidate.clone()
                    adv_post = _project_ball(adv_candidate, inputs, eps, norm).detach()
                    logits_post = model(normalizer(adv_post)).detach()
                    obj_post = _margin_objective(logits_post, target_class)
                    if oracle_logits is not None:
                        obj_post = obj_post - 0.01 * (logits_post.squeeze(0) - oracle_logits).pow(2).mean()
                    return cand_m, cand_v, adv_pre, adv_post, logits_post, obj_post

                if accepted_step:
                    margin_m, margin_v, adv_margin_pre, adv_margin_post, logits_margin_post, obj_margin_post = _candidate_from_state(
                        margin_grad, m, v
                    )
                    hybrid_considered = bool(boundary_gate_active)
                    if hybrid_considered:
                        hybrid_m, hybrid_v, adv_hybrid_pre, adv_hybrid_post, logits_hybrid_post, obj_hybrid_post = _candidate_from_state(
                            hybrid_dir, m, v
                        )
                    else:
                        hybrid_m = hybrid_v = None
                        adv_hybrid_pre = adv_before.clone()
                        adv_hybrid_post = adv_margin_post
                        logits_hybrid_post = logits_margin_post
                        obj_hybrid_post = torch.full_like(obj_margin_post, float("-inf"))
                    if (not hybrid_considered) or obj_margin_post.item() >= obj_hybrid_post.item():
                        selected_candidate = "margin_forced" if not hybrid_considered else "margin"
                        m, v = margin_m, margin_v
                        adv_preproj = adv_margin_pre
                        adv_inputs = adv_margin_post
                        adv_logits = logits_margin_post
                        objective = obj_margin_post
                    else:
                        selected_candidate = "hybrid"
                        m, v = hybrid_m, hybrid_v
                        adv_preproj = adv_hybrid_pre
                        adv_inputs = adv_hybrid_post
                        adv_logits = logits_hybrid_post
                        objective = obj_hybrid_post
                    delta_hybrid_pre = adv_hybrid_pre - adv_before
                else:
                    hybrid_considered = False
                    selected_candidate = "hybrid"
                    m = adam_beta1 * m + (1.0 - adam_beta1) * hybrid_dir
                    v = adam_beta2 * v + (1.0 - adam_beta2) * hybrid_dir.pow(2)
                    m_hat = m / (1.0 - adam_beta1 ** (step_idx + 1))
                    v_hat = v / (1.0 - adam_beta2 ** (step_idx + 1))
                    update = m_hat / (v_hat.sqrt() + 1e-8)
                    if norm == "l_inf":
                        adv_inputs = adv_before + base_lr * update.sign()
                    elif norm == "l2":
                        flat = update.view(update.size(0), -1)
                        update_norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
                        adv_inputs = adv_before + base_lr * (flat / update_norm).view_as(update)
                    else:
                        raise ValueError(f"Unsupported norm: {norm}")

                    adv_preproj = adv_inputs.clone()
                    adv_inputs = _project_ball(adv_inputs, inputs, eps, norm).detach()
                    adv_logits = model(normalizer(adv_inputs)).detach()
                    objective = _margin_objective(adv_logits, target_class)
                    if oracle_logits is not None:
                        objective = objective - 0.01 * (adv_logits.squeeze(0) - oracle_logits).pow(2).mean()
                    delta_hybrid_pre = adv_preproj - adv_before

                if objective.item() > best_objective.item():
                    best_objective[:] = objective
                    best_adv = adv_inputs.clone()
                    best_logits = adv_logits.clone()

                if debug_context and bool(debug_context.get("enabled", False)) and step_idx < int(debug_context.get("max_steps", steps)):
                    delta_selected_pre = adv_preproj - adv_before
                    delta_post = adv_inputs - adv_before
                    next_margin = float(_margin_objective(adv_logits, target_class).item())
                    target_prob = float(F.softmax(logits_vec.unsqueeze(0), dim=1)[0, target_class].item())
                    payload = {
                        "restart": int(restart_idx),
                        "step": int(step_idx),
                        "competitor": int(competitor),
                        "competitor_weights": competitor_weights,
                        "competitor_margins": competitor_margins,
                        "margin": float(current_margin),
                        "target_prob": target_prob,
                        "cos_margin_bundle": _vector_cosine(margin_grad, bundle_dir),
                        "pred_gain_margin": _predicted_margin_gain(margin_grad, delta_margin_pre),
                        "pred_gain_bundle": _predicted_margin_gain(margin_grad, delta_bundle_pre),
                        "pred_gain_hybrid": _predicted_margin_gain(margin_grad, delta_hybrid_pre),
                        "pred_gain_selected": _predicted_margin_gain(margin_grad, delta_selected_pre),
                        "actual_margin_gain": float(next_margin - current_margin),
                        "preproj_postproj_cos": _vector_cosine(delta_selected_pre, delta_post),
                        "projection_ratio": float(
                            delta_post.reshape(-1).norm().div(delta_selected_pre.reshape(-1).norm().clamp_min(1e-12)).item()
                        ),
                        "jacobian_eff_rank": _effective_rank_from_gram(gram),
                        "jacobian_top1_mass": _spectrum_mass_from_gram(gram, 1),
                        "jacobian_top3_mass": _spectrum_mass_from_gram(gram, 3),
                        "bundle_mix_eff": float(bundle_mix_eff.reshape(-1)[0].item()) if torch.is_tensor(bundle_mix_eff) else float(bundle_mix_eff),
                        "selected_candidate": selected_candidate,
                        "boundary_gate_active": bool(boundary_gate_active),
                        "boundary_gate_threshold": float(accepted_margin_gate) if accepted_margin_gate is not None else float("nan"),
                        "hybrid_considered": bool(hybrid_considered),
                    }
                    if accepted_step:
                        payload["candidate_margin_obj"] = float(obj_margin_post.item())
                        payload["candidate_hybrid_obj"] = float(obj_hybrid_post.item())
                    payload.update(_topk_logit_payload(logits_vec, int(debug_context.get("topk", 5))))
                    _maybe_append_geometry_row(debug_context, payload)

    success = bool((best_logits.argmax(dim=1) == target_class).all().item())
    return best_adv, best_logits, success


def _run_logit_subspace_adam_attack(
    model: torch.nn.Module,
    normalizer: torch.nn.Module,
    inputs: torch.Tensor,
    target_class: int,
    eps: float,
    steps: int,
    restarts: int,
    norm: str,
    alpha_ratio: float,
    alpha_min: float,
    attack_state: Dict[str, torch.Tensor],
    oracle_guidance: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    if eps <= 0.0:
        logits, preds = _predict(model, normalizer, inputs)
        success = bool((preds == target_class).all().item())
        return inputs.detach().clone(), logits, success

    basis = attack_state["basis"]
    k = int(basis.size(1))
    if k == 0:
        logits, preds = _predict(model, normalizer, inputs)
        success = bool((preds == target_class).all().item())
        return inputs.detach().clone(), logits, success

    base_lr = min(max(alpha_min, eps * alpha_ratio), eps)
    best_adv = inputs.detach().clone()
    best_logits, _ = _predict(model, normalizer, best_adv)
    best_objective = torch.full((1,), float("-inf"), device=inputs.device)
    init_alpha = attack_state["alpha0"]
    oracle_logits = attack_state.get("oracle_logits")

    for restart_idx in range(restarts):
        if restart_idx == 0 and init_alpha.numel() == k:
            alpha = init_alpha.clone().detach()
        else:
            alpha = torch.empty(k, device=inputs.device).uniform_(-eps, eps)
        m = torch.zeros_like(alpha)
        v = torch.zeros_like(alpha)

        for step_idx in range(steps):
            alpha = alpha.detach().requires_grad_(True)
            delta_flat = basis @ alpha
            delta = delta_flat.view_as(inputs)
            adv_inputs = torch.clamp(inputs + torch.clamp(delta, -eps, eps), 0.0, 1.0)
            logits = model(normalizer(adv_inputs))
            objective = _margin_objective(logits, target_class)
            if oracle_guidance and oracle_logits is not None:
                objective = objective - 0.1 * (logits.squeeze(0) - oracle_logits).pow(2).mean()
            grad = torch.autograd.grad(objective.sum(), alpha, only_inputs=True)[0]

            with torch.no_grad():
                m = 0.9 * m + 0.1 * grad
                v = 0.999 * v + 0.001 * grad.pow(2)
                m_hat = m / (1.0 - 0.9 ** (step_idx + 1))
                v_hat = v / (1.0 - 0.999 ** (step_idx + 1))
                alpha = alpha + base_lr * (m_hat / (v_hat.sqrt() + 1e-8))

                delta_flat = basis @ alpha
                delta = delta_flat.view_as(inputs)
                adv_inputs = torch.clamp(inputs + torch.clamp(delta, -eps, eps), 0.0, 1.0)
                adv_inputs = _project_ball(adv_inputs, inputs, eps, norm).detach()
                logits = model(normalizer(adv_inputs)).detach()
                objective_val = _margin_objective(logits, target_class)
                if oracle_guidance and oracle_logits is not None:
                    objective_val = objective_val - 0.1 * (logits.squeeze(0) - oracle_logits).pow(2).mean()
                if objective_val.item() > best_objective.item():
                    best_objective[:] = objective_val
                    best_adv = adv_inputs.clone()
                    best_logits = logits.clone()

    success = bool((best_logits.argmax(dim=1) == target_class).all().item())
    return best_adv, best_logits, success


def _find_recovery_radius(
    model: torch.nn.Module,
    normalizer: torch.nn.Module,
    inputs: torch.Tensor,
    target_class: int,
    eps_max: float,
    steps: int,
    restarts: int,
    search_steps: int,
    norm: str,
    alpha_ratio: float,
    alpha_min: float,
    attack_loss: str,
    oracle_model: Optional[torch.nn.Module] = None,
    oracle_feature_weight: float = 0.0,
    apgd_rho: float = 0.5,
    apgd_momentum: float = 0.75,
    bundle_margin_gate: Optional[float] = None,
    attack_state: Optional[Dict[str, torch.Tensor]] = None,
    debug_context: Optional[Dict[str, object]] = None,
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

    probe_schedule = [eps_max * frac for frac in (1/16, 1/8, 1/4, 1/2, 1.0)]
    for probe_idx, eps in enumerate(probe_schedule):
        child_debug = None
        if debug_context and bool(debug_context.get("enabled", False)):
            child_debug = dict(debug_context)
            child_debug["search_phase"] = "probe"
            child_debug["search_iter"] = int(probe_idx)
            child_debug["eps"] = float(eps)
        adv_inputs, adv_logits, success = _run_targeted_attack(
            model=model,
            normalizer=normalizer,
            inputs=inputs,
            target_class=target_class,
            eps=eps,
            steps=steps,
            restarts=restarts,
            norm=norm,
            alpha_ratio=alpha_ratio,
            alpha_min=alpha_min,
            attack_loss=attack_loss,
            oracle_model=oracle_model,
            oracle_feature_weight=oracle_feature_weight,
            apgd_rho=apgd_rho,
            apgd_momentum=apgd_momentum,
            bundle_margin_gate=bundle_margin_gate,
            attack_state=attack_state,
            debug_context=child_debug,
        )
        if success:
            high = eps
            high_adv = adv_inputs
            high_logits = adv_logits
            break
        low = eps

    if high is None:
        child_debug = None
        if debug_context and bool(debug_context.get("enabled", False)):
            child_debug = dict(debug_context)
            child_debug["search_phase"] = "fallback"
            child_debug["search_iter"] = -1
            child_debug["eps"] = float(eps_max)
        adv_inputs, adv_logits, success = _run_targeted_attack(
            model=model,
            normalizer=normalizer,
            inputs=inputs,
            target_class=target_class,
            eps=eps_max,
            steps=steps,
            restarts=restarts,
            norm=norm,
            alpha_ratio=alpha_ratio,
            alpha_min=alpha_min,
            attack_loss=attack_loss,
            oracle_model=oracle_model,
            oracle_feature_weight=oracle_feature_weight,
            apgd_rho=apgd_rho,
            apgd_momentum=apgd_momentum,
            bundle_margin_gate=bundle_margin_gate,
            attack_state=attack_state,
            debug_context=child_debug,
        )
        if not success:
            adv_pred = int(adv_logits.argmax(dim=1).item())
            adv_target_prob = float(F.softmax(adv_logits, dim=1)[0, target_class].item())
            return {
                "success": False,
                "clean_success": False,
                "radius": None,
                "actual_delta": _actual_delta_norm(inputs, adv_inputs, norm),
                "clean_target_prob": clean_target_prob,
                "adv_target_prob": adv_target_prob,
                "adv_pred": adv_pred,
            }
        high = eps_max
        high_adv = adv_inputs
        high_logits = adv_logits

    for search_idx in range(search_steps):
        mid = 0.5 * (low + high)
        child_debug = None
        if debug_context and bool(debug_context.get("enabled", False)):
            child_debug = dict(debug_context)
            child_debug["search_phase"] = "binary"
            child_debug["search_iter"] = int(search_idx)
            child_debug["eps"] = float(mid)
        adv_inputs, adv_logits, success = _run_targeted_attack(
            model=model,
            normalizer=normalizer,
            inputs=inputs,
            target_class=target_class,
            eps=mid,
            steps=steps,
            restarts=restarts,
            norm=norm,
            alpha_ratio=alpha_ratio,
            alpha_min=alpha_min,
            attack_loss=attack_loss,
            oracle_model=oracle_model,
            oracle_feature_weight=oracle_feature_weight,
            apgd_rho=apgd_rho,
            apgd_momentum=apgd_momentum,
            bundle_margin_gate=bundle_margin_gate,
            attack_state=attack_state,
            debug_context=child_debug,
        )
        if success:
            high = mid
            high_adv = adv_inputs
            high_logits = adv_logits
        else:
            low = mid

    if high_adv is None or high_logits is None:
        raise RuntimeError("Binary search ended without a successful attack state.")

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


def _default_model_suites(experiment_suites: Dict) -> List[str]:
    return [suite for suite in DEFAULT_MODEL_SUITES if suite in experiment_suites]


def _suite_tag(model_suites: Sequence[str]) -> str:
    if list(model_suites) == DEFAULT_MODEL_SUITES:
        return "default"
    short = []
    for suite in model_suites:
        token = suite
        if token.startswith("unlearn_"):
            token = token[len("unlearn_"):]
        if token.endswith("_vit_cifar10_forget0_smoke"):
            token = token[:-len("_vit_cifar10_forget0_smoke")]
        elif token.endswith("_vit_cifar10_forget0"):
            token = token[:-len("_vit_cifar10_forget0")]
        short.append(token.replace("_", "-"))
    return "__".join(short)


def _safe_result_stem(prefix: str, suffix: str, max_len: int = 180) -> str:
    stem = f"{prefix}_{suffix}"
    if len(stem) <= max_len:
        return stem
    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:10]
    keep = max_len - len(prefix) - len(digest) - 2
    trimmed_suffix = suffix[: max(keep, 16)].rstrip("_-")
    return f"{prefix}_{trimmed_suffix}_{digest}"


def main():
    parser = argparse.ArgumentParser(description="Per-example recovery-radius audit")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-forget-samples", type=int, default=32)
    parser.add_argument("--max-retain-samples", type=int, default=32)
    parser.add_argument("--model-suites", nargs="*", default=None)
    parser.add_argument("--eps-max", type=float, default=8/255)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--search-steps", type=int, default=6)
    parser.add_argument("--norm", type=str, default="l_inf", choices=["l_inf", "l2"])
    parser.add_argument("--alpha-ratio", type=float, default=0.25)
    parser.add_argument("--alpha-min", type=float, default=1/255)
    parser.add_argument("--bundle-margin-gate", type=float, default=-3.0)
    parser.add_argument(
        "--attack-loss",
        type=str,
        default="ce",
        choices=[
            "ce",
            "margin",
            "margin_oracle",
            "apgd_margin",
            "apgd_margin_oracle",
            "adam_margin",
            "adam_margin_oracle",
            "adam_margin_box",
            "adam_margin_box_oracle",
            "adam_margin_bundle_warm",
            "adam_margin_bundle_warm_oracle",
            "logit_subspace_adam",
            "logit_subspace_adam_oracle",
            "logit_gap_gn_oracle",
            "logit_bundle_gn",
            "logit_bundle_gn_oracle",
            "logit_bundle_hybrid",
            "logit_bundle_hybrid_oracle",
            "logit_bundle_orth",
            "logit_bundle_adaptive",
            "logit_bundle_topk",
            "logit_bundle_accept",
            "logit_bundle_boundary",
            "logit_bundle_orth_oracle",
            "logit_bundle_topk_oracle",
            "logit_bundle_accept_oracle",
            "logit_bundle_boundary_oracle",
        ],
    )
    parser.add_argument("--oracle-feature-weight", type=float, default=0.5)
    parser.add_argument("--apgd-rho", type=float, default=0.5)
    parser.add_argument("--apgd-momentum", type=float, default=0.75)
    parser.add_argument("--debug-attack-geometry", action="store_true")
    parser.add_argument("--debug-max-samples", type=int, default=2)
    parser.add_argument("--debug-max-steps", type=int, default=20)
    parser.add_argument("--debug-topk", type=int, default=5)
    parser.add_argument("--skip-retain-control", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    experiment_suites = load_config(args.config)
    model_suites = args.model_suites or _default_model_suites(experiment_suites)
    if not model_suites:
        raise ValueError("No model suites provided and no default suites found in config")
    if int(args.batch_size) != 1:
        raise ValueError("Recovery-radius audit is defined per sample. Use --batch-size 1.")

    first_suite_cfg = experiment_suites[model_suites[0]]
    base_suite_name = first_suite_cfg["base_model_suite"]
    base_suite_cfg = experiment_suites[base_suite_name]
    dataset_name = base_suite_cfg["dataset"]
    model_type = base_suite_cfg["model"]
    forget_class = int(first_suite_cfg.get("unlearning", {}).get("forget_class", first_suite_cfg.get("forget_class", 0)))

    models: Dict[str, torch.nn.Module] = {}
    for suite_name in model_suites:
        model, suite_dataset, suite_model = _load_model(experiment_suites, suite_name, args.seed, device)
        if suite_dataset != dataset_name or suite_model != model_type:
            raise ValueError(f"Suite {suite_name} resolved to {suite_dataset}/{suite_model}, expected {dataset_name}/{model_type}")
        models[suite_name] = model.eval()

    oracle_model = None
    if args.attack_loss in {
        "margin_oracle",
        "apgd_margin_oracle",
        "adam_margin_oracle",
        "adam_margin_box_oracle",
        "adam_margin_bundle_warm_oracle",
        "logit_subspace_adam_oracle",
        "logit_gap_gn_oracle",
        "logit_bundle_gn_oracle",
        "logit_bundle_hybrid_oracle",
        "logit_bundle_orth_oracle",
        "logit_bundle_topk_oracle",
        "logit_bundle_accept_oracle",
        "logit_bundle_boundary_oracle",
    }:
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
    print(f"Norm: {args.norm}")
    print(f"eps_max: {args.eps_max}")
    print(f"attack_loss: {args.attack_loss}")

    forget_loader = _build_loader(
        dataset_name=dataset_name,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        forget_class=forget_class,
        retain=False,
        max_samples=args.max_forget_samples,
    )
    retain_loader = None
    if not args.skip_retain_control:
        retain_loader = _build_loader(
            dataset_name=dataset_name,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            forget_class=forget_class,
            retain=True,
            max_samples=args.max_retain_samples,
        )

    normalizer = create_imagenet_normalizer().to(device)
    geometry_rows: List[Dict[str, object]] = []

    results = {
        "meta": {
            "seed": int(args.seed),
            "split": args.split,
            "dataset": dataset_name,
            "model": model_type,
            "forget_class": int(forget_class),
            "model_suites": model_suites,
            "norm": args.norm,
            "eps_max": float(args.eps_max),
            "steps": int(args.steps),
            "restarts": int(args.restarts),
            "search_steps": int(args.search_steps),
            "attack_loss": args.attack_loss,
            "oracle_feature_weight": float(args.oracle_feature_weight),
            "apgd_rho": float(args.apgd_rho),
            "apgd_momentum": float(args.apgd_momentum),
            "bundle_margin_gate": float(args.bundle_margin_gate),
            "debug_attack_geometry": bool(args.debug_attack_geometry),
            "debug_max_samples": int(args.debug_max_samples),
            "debug_max_steps": int(args.debug_max_steps),
            "debug_topk": int(args.debug_topk),
        },
        "forget_recovery": {},
        "retain_control": {},
    }

    for suite_name in model_suites:
        model = models[suite_name]
        print(f"\nAuditing forget recovery radius for {suite_name}...")
        records: List[Dict[str, Optional[float]]] = []
        for idx, (inputs, _labels) in enumerate(forget_loader):
            inputs = inputs.to(device)
            attack_state = None
            if args.attack_loss in {"logit_subspace_adam", "logit_subspace_adam_oracle"}:
                attack_state = _logit_jacobian_state(
                    model=model,
                    normalizer=normalizer,
                    inputs=inputs,
                    oracle_model=oracle_model if args.attack_loss.endswith("_oracle") else None,
                )
            debug_context = None
            if args.debug_attack_geometry and idx < int(args.debug_max_samples):
                debug_context = {
                    "enabled": True,
                    "rows": geometry_rows,
                    "suite_name": suite_name,
                    "sample_type": "forget",
                    "sample_index": int(idx),
                    "attack_loss": args.attack_loss,
                    "max_steps": int(args.debug_max_steps),
                    "topk": int(args.debug_topk),
                }
            record = _find_recovery_radius(
                model=model,
                normalizer=normalizer,
                inputs=inputs,
                target_class=forget_class,
                eps_max=float(args.eps_max),
                steps=int(args.steps),
                restarts=int(args.restarts),
                search_steps=int(args.search_steps),
                norm=args.norm,
                alpha_ratio=float(args.alpha_ratio),
                alpha_min=float(args.alpha_min),
                attack_loss=args.attack_loss,
                oracle_model=oracle_model,
                oracle_feature_weight=float(args.oracle_feature_weight),
                apgd_rho=float(args.apgd_rho),
                apgd_momentum=float(args.apgd_momentum),
                bundle_margin_gate=float(args.bundle_margin_gate),
                attack_state=attack_state,
                debug_context=debug_context,
            )
            record["sample_index"] = idx
            records.append(record)
        results["forget_recovery"][suite_name] = {
            "summary": _summarize_radius(records),
            "per_sample": records,
        }

        if retain_loader is not None:
            print(f"Auditing retain-to-forget control radius for {suite_name}...")
            control_records: List[Dict[str, Optional[float]]] = []
            for idx, (inputs, _labels) in enumerate(retain_loader):
                inputs = inputs.to(device)
                attack_state = None
                if args.attack_loss in {"logit_subspace_adam", "logit_subspace_adam_oracle"}:
                    attack_state = _logit_jacobian_state(
                        model=model,
                        normalizer=normalizer,
                        inputs=inputs,
                        oracle_model=oracle_model if args.attack_loss.endswith("_oracle") else None,
                    )
                debug_context = None
                if args.debug_attack_geometry and idx < int(args.debug_max_samples):
                    debug_context = {
                        "enabled": True,
                        "rows": geometry_rows,
                        "suite_name": suite_name,
                        "sample_type": "retain",
                        "sample_index": int(idx),
                        "attack_loss": args.attack_loss,
                        "max_steps": int(args.debug_max_steps),
                        "topk": int(args.debug_topk),
                    }
                record = _find_recovery_radius(
                    model=model,
                    normalizer=normalizer,
                    inputs=inputs,
                    target_class=forget_class,
                    eps_max=float(args.eps_max),
                    steps=int(args.steps),
                    restarts=int(args.restarts),
                    search_steps=int(args.search_steps),
                    norm=args.norm,
                    alpha_ratio=float(args.alpha_ratio),
                    alpha_min=float(args.alpha_min),
                    attack_loss=args.attack_loss,
                    oracle_model=oracle_model,
                    oracle_feature_weight=float(args.oracle_feature_weight),
                    apgd_rho=float(args.apgd_rho),
                    apgd_momentum=float(args.apgd_momentum),
                    bundle_margin_gate=float(args.bundle_margin_gate),
                    attack_state=attack_state,
                    debug_context=debug_context,
                )
                record["sample_index"] = idx
                control_records.append(record)
            results["retain_control"][suite_name] = {
                "summary": _summarize_radius(control_records),
                "per_sample": control_records,
            }

    attack_tag = args.attack_loss
    if args.attack_loss == "margin_oracle":
        weight_tag = str(args.oracle_feature_weight).replace(".", "p")
        attack_tag = f"{attack_tag}_w{weight_tag}"
    if args.attack_loss in {"logit_bundle_boundary", "logit_bundle_boundary_oracle"}:
        gate_tag = str(args.bundle_margin_gate).replace("-", "m").replace(".", "p")
        attack_tag = f"{attack_tag}_g{gate_tag}"
    if abs(float(args.alpha_ratio) - 0.25) > 1e-12:
        alpha_ratio_tag = str(args.alpha_ratio).replace("-", "m").replace(".", "p")
        attack_tag = f"{attack_tag}_ar{alpha_ratio_tag}"
    if abs(float(args.alpha_min) - (1 / 255)) > 1e-12:
        alpha_min_tag = str(args.alpha_min).replace("-", "m").replace(".", "p")
        attack_tag = f"{attack_tag}_amin{alpha_min_tag}"
    if int(args.steps) != 20:
        attack_tag = f"{attack_tag}_s{int(args.steps)}"
    if int(args.restarts) != 2:
        attack_tag = f"{attack_tag}_r{int(args.restarts)}"
    if int(args.search_steps) != 6:
        attack_tag = f"{attack_tag}_bs{int(args.search_steps)}"
    retain_tag = "forget_only" if args.skip_retain_control else "with_retain"
    result_prefix = (
        f"results/analysis/recovery_radius_{dataset_name}_{model_type}_forget{forget_class}"
        f"_seed_{args.seed}_{args.split}_{retain_tag}"
    )
    result_suffix = f"{attack_tag}_{_suite_tag(model_suites)}"
    save_path = f"{_safe_result_stem(result_prefix, result_suffix)}.json"
    if args.debug_attack_geometry and geometry_rows:
        geometry_summary = _summarize_geometry_rows(geometry_rows)
        results["geometry_debug_summary"] = geometry_summary
        geometry_prefix = (
            f"results/logs/attack_geometry_{dataset_name}_{model_type}_forget{forget_class}"
            f"_seed_{args.seed}_{args.split}_{retain_tag}"
        )
        geometry_suffix = f"{attack_tag}_{_suite_tag(model_suites)}"
        geometry_log_path = f"{_safe_result_stem(geometry_prefix, geometry_suffix)}.jsonl"
        os.makedirs(os.path.dirname(geometry_log_path), exist_ok=True)
        with open(geometry_log_path, "w", encoding="utf-8") as f:
            for row in geometry_rows:
                f.write(json.dumps(row) + "\n")
        results["meta"]["geometry_log_path"] = geometry_log_path
    save_dict_to_json(results, save_path)
    print(f"\nSaved recovery-radius audit to: {save_path}")

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

    if retain_loader is not None:
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
