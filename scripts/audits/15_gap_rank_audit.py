#!/usr/bin/env python3
"""
Audit whether the oracle-minus-unlearned gap is low-rank enough for a universal attack.

The key object is the per-example gap:
    g(x) = phi_oracle(x) - phi_model(x)

If g(x) is strongly aligned across forget examples, a universal perturbation has a
plausible shared direction to exploit. If g(x) is high-rank / heterogeneous, a
static universal attack is much less likely to succeed even when leakage exists.
"""

import argparse
import os
import sys
from typing import Dict, Iterable, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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

    if suite_name.startswith("base_"):
        suite_cfg = experiment_suites[suite_name]
        model = _build_model(suite_cfg["model"], suite_cfg["dataset"], device)
        ckpt_path = _checkpoint_or_best("checkpoints/base", suite_name, seed)
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
        adapter_path = os.path.join("checkpoints", "unlearn_lora", f"{suite_name}_seed_{seed}")
        if os.path.exists(adapter_path):
            model = load_lora_adapter(model, adapter_path).to(device)
        else:
            print(f"Warning: adapter not found for {suite_name}, using base model weights only")
        return model.to(device), base_suite_cfg["dataset"], base_suite_cfg["model"]

    raise ValueError(f"Unsupported suite type: {suite_name}")


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
def _extract_features_and_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    normalizer: torch.nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    feature_model = _resolve_feature_model(model)
    feature_list: List[torch.Tensor] = []
    logit_list: List[torch.Tensor] = []

    for inputs, _labels in loader:
        inputs = inputs.to(device)
        norm_inputs = normalizer(inputs)
        logits = model(norm_inputs)
        if feature_model is None:
            feats = logits
        else:
            feats = feature_model.forward_features(norm_inputs)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
        if feats.ndim == 4:
            feats = feats.mean(dim=(2, 3))
        elif feats.ndim == 3:
            feats = feats[:, 0]
        elif feats.ndim > 2:
            feats = feats.flatten(start_dim=1)

        feature_list.append(feats.detach().cpu())
        logit_list.append(logits.detach().cpu())

    return torch.cat(feature_list, dim=0), torch.cat(logit_list, dim=0)


def _gap_geometry(gaps: torch.Tensor, ks: Iterable[int]) -> Dict[str, float]:
    gaps = gaps.float()
    n, dim = gaps.shape
    total_energy = float((gaps * gaps).sum().item())
    mean_gap = gaps.mean(dim=0, keepdim=True)
    mean_gap_norm = float(mean_gap.norm().item())

    result: Dict[str, float] = {
        "n_samples": int(n),
        "dim": int(dim),
        "mean_gap_norm": mean_gap_norm,
        "gap_mean_abs": float(gaps.abs().mean().item()),
        "gap_l2_per_sample": float(gaps.norm(dim=1).mean().item()),
    }

    if mean_gap_norm > 0.0:
        mean_unit = mean_gap / mean_gap.norm(dim=1, keepdim=True).clamp_min(1e-8)
        cos = F.cosine_similarity(gaps, mean_unit.expand_as(gaps), dim=1)
        proj = gaps @ mean_unit.squeeze(0)
        mean_dir_energy = float((proj * proj).sum().item())
        result["mean_cos_to_gap_mean"] = float(cos.mean().item())
        result["mean_abs_cos_to_gap_mean"] = float(cos.abs().mean().item())
        result["mean_direction_energy_ratio"] = mean_dir_energy / max(total_energy, 1e-12)
    else:
        result["mean_cos_to_gap_mean"] = 0.0
        result["mean_abs_cos_to_gap_mean"] = 0.0
        result["mean_direction_energy_ratio"] = 0.0

    centered = gaps - mean_gap
    centered_energy = float((centered * centered).sum().item())
    result["centered_gap_energy_ratio"] = centered_energy / max(total_energy, 1e-12)

    if n < 2 or centered_energy <= 1e-12:
        result["centered_effective_rank"] = 0.0
        result["centered_stable_rank"] = 0.0
        for k in ks:
            result[f"centered_top{k}_energy_ratio"] = 0.0
        return result

    singular_vals = torch.linalg.svdvals(centered)
    spectral_energy = singular_vals.pow(2)
    energy_sum = spectral_energy.sum().clamp_min(1e-12)
    probs = spectral_energy / energy_sum
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
    result["centered_effective_rank"] = float(torch.exp(entropy).item())
    result["centered_stable_rank"] = float((energy_sum / spectral_energy[0].clamp_min(1e-12)).item())

    cumulative = torch.cumsum(spectral_energy, dim=0) / energy_sum
    for k in ks:
        idx = min(k, cumulative.numel()) - 1
        result[f"centered_top{k}_energy_ratio"] = float(cumulative[idx].item())

    return result


def _analyze_pair(
    oracle_feats: torch.Tensor,
    oracle_logits: torch.Tensor,
    model_feats: torch.Tensor,
    model_logits: torch.Tensor,
) -> Dict[str, Dict[str, float]]:
    oracle_feats = F.normalize(oracle_feats.float(), dim=1)
    model_feats = F.normalize(model_feats.float(), dim=1)
    feat_gaps = oracle_feats - model_feats
    logit_gaps = oracle_logits.float() - model_logits.float()

    ks = (1, 2, 4, 8, 16)
    feature_stats = _gap_geometry(feat_gaps, ks)
    logit_stats = _gap_geometry(logit_gaps, ks)
    return {
        "feature_gap": feature_stats,
        "logit_gap": logit_stats,
    }


def _default_model_suites(experiment_suites: Dict) -> List[str]:
    return [suite for suite in DEFAULT_MODEL_SUITES if suite in experiment_suites]


def main():
    parser = argparse.ArgumentParser(description="Audit oracle-gap rank for unlearning methods")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-forget-samples", type=int, default=1000)
    parser.add_argument("--max-retain-samples", type=int, default=2000)
    parser.add_argument("--model-suites", nargs="*", default=None)
    parser.add_argument("--oracle-suite", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    experiment_suites = load_config(args.config)

    model_suites = args.model_suites or _default_model_suites(experiment_suites)
    if not model_suites:
        raise ValueError("No model suites provided and no default suites found in config")

    first_suite = experiment_suites[model_suites[0]]
    base_suite_name = first_suite.get("base_model_suite")
    if not base_suite_name:
        raise ValueError(f"Suite {model_suites[0]} is not an unlearned suite")
    base_suite_cfg = experiment_suites[base_suite_name]
    dataset_name = base_suite_cfg["dataset"]
    model_type = base_suite_cfg["model"]
    forget_class = int(first_suite.get("unlearning", {}).get("forget_class", first_suite.get("forget_class", 0)))

    oracle_suite = args.oracle_suite or _resolve_matching_oracle_suite(
        experiment_suites,
        dataset_name=dataset_name,
        model_type=model_type,
        forget_class=forget_class,
    )

    suites_to_load = [base_suite_name, oracle_suite] + model_suites
    models: Dict[str, torch.nn.Module] = {}
    for suite_name in suites_to_load:
        model, suite_dataset, suite_model = _load_model(experiment_suites, suite_name, args.seed, device)
        if suite_dataset != dataset_name or suite_model != model_type:
            raise ValueError(f"Suite {suite_name} resolved to {suite_dataset}/{suite_model}, expected {dataset_name}/{model_type}")
        models[suite_name] = model

    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_type}")
    print(f"Forget class: {forget_class}")
    print(f"Oracle suite: {oracle_suite}")
    print(f"Model suites: {model_suites}")

    forget_loader = _build_loader(
        dataset_name=dataset_name,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        forget_class=forget_class,
        retain=False,
        max_samples=args.max_forget_samples,
    )
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
    extracted: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}
    for split_name, loader in [("forget", forget_loader), ("retain", retain_loader)]:
        extracted[split_name] = {}
        for suite_name in suites_to_load:
            print(f"Extracting {split_name} features for {suite_name}...")
            feats, logits = _extract_features_and_logits(models[suite_name], loader, device, normalizer)
            extracted[split_name][suite_name] = (feats, logits)

    results: Dict[str, Dict] = {
        "meta": {
            "seed": int(args.seed),
            "split": args.split,
            "dataset": dataset_name,
            "model": model_type,
            "forget_class": int(forget_class),
            "base_suite": base_suite_name,
            "oracle_suite": oracle_suite,
            "model_suites": model_suites,
            "max_forget_samples": int(args.max_forget_samples),
            "max_retain_samples": int(args.max_retain_samples),
        },
        "splits": {},
    }

    comparisons = [base_suite_name] + model_suites
    for split_name in ["forget", "retain"]:
        split_results: Dict[str, Dict] = {}
        oracle_feats, oracle_logits = extracted[split_name][oracle_suite]
        for suite_name in comparisons:
            feats, logits = extracted[split_name][suite_name]
            split_results[suite_name] = _analyze_pair(oracle_feats, oracle_logits, feats, logits)
        results["splits"][split_name] = split_results

    save_path = (
        f"results/analysis/gap_rank_{dataset_name}_{model_type}_forget{forget_class}"
        f"_seed_{args.seed}_{args.split}.json"
    )
    save_dict_to_json(results, save_path)

    print(f"\nSaved gap-rank audit to: {save_path}")
    print("\nFeature-gap summary (forget split)")
    print("Model                                   | mean-cos | mean-dir | top1 | top4 | eff-rank")
    print("-" * 88)
    for suite_name in comparisons:
        stats = results["splits"]["forget"][suite_name]["feature_gap"]
        print(
            f"{suite_name:<39} | "
            f"{stats['mean_cos_to_gap_mean']:.3f} | "
            f"{stats['mean_direction_energy_ratio']:.3f} | "
            f"{stats['centered_top1_energy_ratio']:.3f} | "
            f"{stats['centered_top4_energy_ratio']:.3f} | "
            f"{stats['centered_effective_rank']:.2f}"
        )

    print("\nLogit-gap summary (forget split)")
    print("Model                                   | mean-cos | mean-dir | top1 | top4 | eff-rank")
    print("-" * 88)
    for suite_name in comparisons:
        stats = results["splits"]["forget"][suite_name]["logit_gap"]
        print(
            f"{suite_name:<39} | "
            f"{stats['mean_cos_to_gap_mean']:.3f} | "
            f"{stats['mean_direction_energy_ratio']:.3f} | "
            f"{stats['centered_top1_energy_ratio']:.3f} | "
            f"{stats['centered_top4_energy_ratio']:.3f} | "
            f"{stats['centered_effective_rank']:.2f}"
        )


if __name__ == "__main__":
    main()
