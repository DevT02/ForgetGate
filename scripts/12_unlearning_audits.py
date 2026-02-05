#!/usr/bin/env python3
"""
Script 12: Lightweight audits inspired by recent LLM unlearning evaluations,
adapted for vision models (CIFAR/MNIST).

Audits:
1) Loss-landscape variance (REMIND-like): variance of CE loss under small input noise.
2) Stochastic leakage (leak@-like): any-perturbation recovery rate for forget samples.
"""

import argparse
import os
import sys
from typing import Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import DataManager
from src.models.vit import create_vit_model
from src.models.cnn import create_cnn_model
from src.models.peft_lora import load_lora_adapter
from src.models.normalize import create_imagenet_normalizer
from src.utils import set_seed, load_config, get_device, save_dict_to_json


def resolve_base_suite_info(experiment_suites: Dict, suite_name: str) -> Tuple[str, Dict]:
    suite = experiment_suites[suite_name]
    if "base_model_suite" in suite:
        return resolve_base_suite_info(experiment_suites, suite["base_model_suite"])
    if "unlearned_model_suite" in suite:
        return resolve_base_suite_info(experiment_suites, suite["unlearned_model_suite"])
    return suite_name, suite


def get_eval_config(experiment_suites: Dict, eval_suite_name: str) -> Dict:
    eval_suite = experiment_suites[eval_suite_name]
    model_suites = eval_suite.get("model_suites", [])
    if model_suites:
        base_suite_name, base_suite = resolve_base_suite_info(experiment_suites, model_suites[0])
    else:
        base_suite_name, base_suite = resolve_base_suite_info(experiment_suites, eval_suite_name)

    forget_class = None
    for suite_name in model_suites:
        if "unlearn" in suite_name:
            unlearn_suite = experiment_suites.get(suite_name, {})
            forget_class = unlearn_suite.get("unlearning", {}).get("forget_class")
            break
    if forget_class is None:
        forget_class = eval_suite.get("unlearning", {}).get("forget_class", 0)

    return {
        "dataset": base_suite.get("dataset", "cifar10"),
        "model": base_suite.get("model", "vit_tiny"),
        "forget_class": forget_class,
        "model_suites": model_suites,
        "eval_suite_name": eval_suite_name,
        "base_suite_name": base_suite_name,
    }


def build_model_from_eval_config(eval_config: Dict, device: torch.device) -> torch.nn.Module:
    data_config = load_config("configs/data.yaml")
    model_config = load_config("configs/model.yaml")

    dataset_name = eval_config["dataset"]
    dataset_info = data_config[dataset_name]
    model_type = eval_config["model"]

    if model_type.startswith("vit"):
        model_config_name = model_type.replace("vit_", "")
        vit_cfg = dict(model_config["vit"][model_config_name])
        vit_cfg["pretrained"] = False
        model = create_vit_model(vit_cfg, num_classes=dataset_info["num_classes"])
    else:
        model = create_cnn_model(model_config["cnn"][model_type], num_classes=dataset_info["num_classes"])

    return model.to(device)


def load_model_for_eval(experiment_suites: Dict,
                        eval_config: Dict,
                        model_suite_name: str,
                        device: torch.device,
                        seed: int) -> torch.nn.Module:
    if "base" in model_suite_name and "unlearn" not in model_suite_name:
        ckpt = f"checkpoints/base/{model_suite_name}_seed_{seed}_final.pt"
        if not os.path.exists(ckpt):
            return None
        checkpoint = torch.load(ckpt, map_location=device)
        model = build_model_from_eval_config(eval_config, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(device)

    if "unlearn" in model_suite_name:
        unlearn_cfg = experiment_suites.get(model_suite_name, {})
        base_suite = unlearn_cfg.get("base_model_suite")
        if not base_suite:
            return None
        base_ckpt = f"checkpoints/base/{base_suite}_seed_{seed}_final.pt"
        if not os.path.exists(base_ckpt):
            return None
        checkpoint = torch.load(base_ckpt, map_location=device)
        model = build_model_from_eval_config(eval_config, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        adapter_path = f"checkpoints/unlearn_lora/{model_suite_name}_seed_{seed}"
        if os.path.exists(adapter_path):
            model = load_lora_adapter(model, adapter_path)
        return model.to(device)

    return None


def sample_subset(loader: DataLoader, max_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
        if sum(t.size(0) for t in xs) >= max_samples:
            break
    x_all = torch.cat(xs, dim=0)[:max_samples]
    y_all = torch.cat(ys, dim=0)[:max_samples]
    return x_all, y_all


@torch.no_grad()
def loss_landscape_variance(model: torch.nn.Module,
                            inputs: torch.Tensor,
                            labels: torch.Tensor,
                            n_perturb: int,
                            noise_std: float) -> Dict[str, float]:
    losses = []
    for _ in range(n_perturb):
        noise = torch.randn_like(inputs) * noise_std
        x = torch.clamp(inputs + noise, 0.0, 1.0)
        logits = model(x)
        loss = F.cross_entropy(logits, labels, reduction="none")
        losses.append(loss)
    losses = torch.stack(losses, dim=0)
    return {
        "loss_mean": float(losses.mean().item()),
        "loss_var": float(losses.var(unbiased=False).mean().item()),
    }


@torch.no_grad()
def stochastic_leakage(model: torch.nn.Module,
                       inputs: torch.Tensor,
                       labels: torch.Tensor,
                       forget_class: int,
                       n_perturb: int,
                       noise_std: float) -> Dict[str, float]:
    any_correct = torch.zeros(labels.size(0), device=labels.device, dtype=torch.bool)
    any_forget = torch.zeros_like(any_correct)
    for _ in range(n_perturb):
        noise = torch.randn_like(inputs) * noise_std
        x = torch.clamp(inputs + noise, 0.0, 1.0)
        preds = model(x).argmax(dim=1)
        any_correct |= preds.eq(labels)
        any_forget |= preds.eq(forget_class)

    return {
        "any_correct_rate": float(any_correct.float().mean().item()),
        "any_forget_rate": float(any_forget.float().mean().item()),
    }


def main():
    parser = argparse.ArgumentParser(description="Run unlearning audits (REMIND/leak@ analogs)")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment suites config")
    parser.add_argument("--suite", type=str, required=True, help="Audit suite name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument("--n-perturb", type=int, default=8, help="Number of perturbations per sample")
    parser.add_argument("--noise-std", type=float, default=0.01, help="Gaussian noise std")
    parser.add_argument("--max-samples", type=int, default=512, help="Max samples per split")
    args = parser.parse_args()

    set_seed(args.seed)
    experiment_suites = load_config(args.config)
    eval_cfg = get_eval_config(experiment_suites, args.suite)

    device = get_device(args.device)

    data_manager = DataManager()
    dataset_name = eval_cfg["dataset"]
    forget_class = eval_cfg["forget_class"]

    forget_ds = data_manager.load_dataset(
        dataset_name, "test", include_classes=[forget_class],
        use_pretrained=True, apply_imagenet_norm=False
    )
    retain_ds = data_manager.load_dataset(
        dataset_name, "test", exclude_classes=[forget_class],
        use_pretrained=True, apply_imagenet_norm=False
    )

    forget_loader = DataLoader(forget_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    retain_loader = DataLoader(retain_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    forget_x, forget_y = sample_subset(forget_loader, args.max_samples)
    retain_x, retain_y = sample_subset(retain_loader, args.max_samples)

    forget_x = forget_x.to(device)
    forget_y = forget_y.to(device)
    retain_x = retain_x.to(device)
    retain_y = retain_y.to(device)

    norm_layer = create_imagenet_normalizer().to(device)

    results = {
        "audit": {
            "n_perturb": args.n_perturb,
            "noise_std": args.noise_std,
            "max_samples": args.max_samples,
        },
        "models": {}
    }

    for model_suite in eval_cfg["model_suites"]:
        model = load_model_for_eval(experiment_suites, eval_cfg, model_suite, device, args.seed)
        if model is None:
            continue

        model = torch.nn.Sequential(norm_layer, model).to(device)
        model.eval()

        forget_stats = loss_landscape_variance(model, forget_x, forget_y, args.n_perturb, args.noise_std)
        retain_stats = loss_landscape_variance(model, retain_x, retain_y, args.n_perturb, args.noise_std)

        forget_leak = stochastic_leakage(model, forget_x, forget_y, forget_class, args.n_perturb, args.noise_std)
        retain_leak = stochastic_leakage(model, retain_x, retain_y, forget_class, args.n_perturb, args.noise_std)

        results["models"][model_suite] = {
            "forget_loss_mean": forget_stats["loss_mean"],
            "forget_loss_var": forget_stats["loss_var"],
            "retain_loss_mean": retain_stats["loss_mean"],
            "retain_loss_var": retain_stats["loss_var"],
            "forget_any_correct": forget_leak["any_correct_rate"],
            "forget_any_forget": forget_leak["any_forget_rate"],
            "retain_any_correct": retain_leak["any_correct_rate"],
            "retain_any_forget": retain_leak["any_forget_rate"],
        }

    out_path = f"results/logs/{args.suite}_seed_{args.seed}_audit.json"
    save_dict_to_json(results, out_path)
    print(f"Audit saved to: {out_path}")


if __name__ == "__main__":
    main()
