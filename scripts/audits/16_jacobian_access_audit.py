#!/usr/bin/env python3
"""
Audit how much of the oracle-minus-unlearned gap is reachable from input space
under a first-order linearization of the unlearned model.

For a representation phi(x) and gap g(x) = phi_oracle(x) - phi_model(x), we
measure the accessible-energy ratio:

    access(x) = ||P_range(J_x) g(x)||^2 / ||g(x)||^2

where J_x is the Jacobian of phi_model at x with respect to the input.

If this ratio is high, the missing oracle direction is locally reachable from
small input perturbations. If it is low, a static input-space attack is fighting
the wrong interface even when a gap exists.
"""

import argparse
import os
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

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


def _representation_vector(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    normalizer: torch.nn.Module,
    kind: str,
) -> torch.Tensor:
    norm_inputs = normalizer(inputs)
    if kind == "logit":
        logits = model(norm_inputs)
        return logits.reshape(logits.size(0), -1)

    if kind != "feature":
        raise ValueError(f"Unsupported representation kind: {kind}")

    feature_model = _resolve_feature_model(model)
    if feature_model is None:
        raise ValueError("Feature representation requires a model with forward_features support.")

    feats = feature_model.forward_features(norm_inputs)
    if isinstance(feats, (tuple, list)):
        feats = feats[0]
    if feats.ndim == 4:
        feats = feats.mean(dim=(2, 3))
    elif feats.ndim == 3:
        feats = feats[:, 0]
    elif feats.ndim > 2:
        feats = feats.flatten(start_dim=1)
    feats = F.normalize(feats, dim=1)
    return feats


@torch.no_grad()
def _gap_vectors(
    model: torch.nn.Module,
    oracle_model: torch.nn.Module,
    inputs: torch.Tensor,
    normalizer: torch.nn.Module,
    kind: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    oracle_vec = _representation_vector(oracle_model, inputs, normalizer, kind).detach()
    model_vec = _representation_vector(model, inputs, normalizer, kind).detach()
    gap = (oracle_vec - model_vec).squeeze(0).float().cpu()
    return model_vec.squeeze(0).float().cpu(), oracle_vec.squeeze(0).float().cpu(), gap


def _jacobian_gram(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    normalizer: torch.nn.Module,
    kind: str,
) -> torch.Tensor:
    model.eval()
    x = inputs.detach().clone().requires_grad_(True)
    vec = _representation_vector(model, x, normalizer, kind).squeeze(0)
    grads: List[torch.Tensor] = []
    for i in range(vec.numel()):
        grad_i = torch.autograd.grad(
            vec[i],
            x,
            retain_graph=(i + 1) < vec.numel(),
            create_graph=False,
            allow_unused=False,
        )[0]
        grads.append(grad_i.reshape(-1))
    jac = torch.stack(grads, dim=0)
    gram = jac @ jac.T
    return gram.detach().cpu().float()


def _projected_gap_stats(gap: torch.Tensor, gram: torch.Tensor) -> Dict[str, float]:
    gap = gap.float()
    gram = gram.float()
    gap_energy = float(torch.dot(gap, gap).item())
    if gap_energy <= 1e-12:
        return {
            "gap_norm": 0.0,
            "projected_gap_norm": 0.0,
            "accessible_energy_ratio": 0.0,
            "residual_energy_ratio": 0.0,
            "input_jacobian_rank": 0.0,
            "input_jacobian_trace": 0.0,
            "input_jacobian_top1_ratio": 0.0,
            "input_jacobian_top4_ratio": 0.0,
            "min_delta_norm_for_projection": 0.0,
        }

    projector = gram @ torch.linalg.pinv(gram, hermitian=True)
    projected_gap = projector @ gap
    projected_energy = float(torch.dot(projected_gap, projected_gap).item())
    residual_energy = max(gap_energy - projected_energy, 0.0)

    eigvals = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    eig_sum = eigvals.sum().clamp_min(1e-12)
    positive = eigvals[eigvals > 1e-8]
    rank = float(positive.numel())
    top1 = float((eigvals[-1] / eig_sum).item()) if eigvals.numel() > 0 else 0.0
    top4 = float((eigvals[-min(4, eigvals.numel()):].sum() / eig_sum).item()) if eigvals.numel() > 0 else 0.0

    gram_pinv = torch.linalg.pinv(gram, hermitian=True)
    min_delta_norm_sq = float(torch.dot(gap, gram_pinv @ gap).item())

    return {
        "gap_norm": float(gap.norm().item()),
        "projected_gap_norm": float(projected_gap.norm().item()),
        "accessible_energy_ratio": projected_energy / gap_energy,
        "residual_energy_ratio": residual_energy / gap_energy,
        "input_jacobian_rank": rank,
        "input_jacobian_trace": float(eig_sum.item()),
        "input_jacobian_top1_ratio": top1,
        "input_jacobian_top4_ratio": top4,
        "min_delta_norm_for_projection": max(min_delta_norm_sq, 0.0) ** 0.5,
    }


def _summarize(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p25": 0.0,
            "p75": 0.0,
        }
    tensor = torch.tensor(list(values), dtype=torch.float32)
    return {
        "mean": float(tensor.mean().item()),
        "median": float(tensor.median().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "p25": float(torch.quantile(tensor, 0.25).item()),
        "p75": float(torch.quantile(tensor, 0.75).item()),
    }


def _default_model_suites(experiment_suites: Dict) -> List[str]:
    return [suite for suite in DEFAULT_MODEL_SUITES if suite in experiment_suites]


def _run_split(
    split_name: str,
    model_suite_names: Sequence[str],
    models: Dict[str, torch.nn.Module],
    oracle_suite: str,
    loader: DataLoader,
    device: torch.device,
    normalizer: torch.nn.Module,
    representation_kinds: Sequence[str],
) -> Dict[str, Dict]:
    split_results: Dict[str, Dict] = {}
    oracle_model = models[oracle_suite]

    for suite_name in model_suite_names:
        print(f"Auditing {split_name} Jacobian access for {suite_name}...")
        model = models[suite_name]
        suite_result: Dict[str, Dict] = {}
        for kind in representation_kinds:
            sample_stats: List[Dict[str, float]] = []
            for inputs, _labels in loader:
                inputs = inputs.to(device)
                gram = _jacobian_gram(model, inputs, normalizer, kind)
                _, _, gap = _gap_vectors(model, oracle_model, inputs, normalizer, kind)
                stats = _projected_gap_stats(gap, gram)
                sample_stats.append(stats)

            summary: Dict[str, Dict[str, float]] = {}
            metric_keys = sample_stats[0].keys() if sample_stats else []
            for metric in metric_keys:
                summary[metric] = _summarize([sample[metric] for sample in sample_stats])
            suite_result[kind] = {
                "summary": summary,
                "per_sample": sample_stats,
            }
        split_results[suite_name] = suite_result
    return split_results


def main():
    parser = argparse.ArgumentParser(description="Audit local Jacobian access to the oracle gap")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-forget-samples", type=int, default=24)
    parser.add_argument("--max-retain-samples", type=int, default=24)
    parser.add_argument("--model-suites", nargs="*", default=None)
    parser.add_argument("--oracle-suite", type=str, default=None)
    parser.add_argument("--representations", nargs="*", default=["feature", "logit"], choices=["feature", "logit"])
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    experiment_suites = load_config(args.config)

    model_suites = args.model_suites or _default_model_suites(experiment_suites)
    if not model_suites:
        raise ValueError("No model suites provided and no default suites found in config")

    first_suite_cfg = experiment_suites[model_suites[0]]
    base_suite_name = first_suite_cfg.get("base_model_suite")
    if not base_suite_name:
        raise ValueError(f"Suite {model_suites[0]} is not an unlearned suite")
    base_suite_cfg = experiment_suites[base_suite_name]
    dataset_name = base_suite_cfg["dataset"]
    model_type = base_suite_cfg["model"]
    forget_class = int(first_suite_cfg.get("unlearning", {}).get("forget_class", first_suite_cfg.get("forget_class", 0)))

    oracle_suite = args.oracle_suite or _resolve_matching_oracle_suite(
        experiment_suites,
        dataset_name=dataset_name,
        model_type=model_type,
        forget_class=forget_class,
    )

    suites_to_load = [oracle_suite] + model_suites
    models: Dict[str, torch.nn.Module] = {}
    for suite_name in suites_to_load:
        model, suite_dataset, suite_model = _load_model(experiment_suites, suite_name, args.seed, device)
        if suite_dataset != dataset_name or suite_model != model_type:
            raise ValueError(f"Suite {suite_name} resolved to {suite_dataset}/{suite_model}, expected {dataset_name}/{model_type}")
        models[suite_name] = model.eval()

    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_type}")
    print(f"Forget class: {forget_class}")
    print(f"Oracle suite: {oracle_suite}")
    print(f"Model suites: {model_suites}")
    print(f"Representations: {list(args.representations)}")

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

    results = {
        "meta": {
            "seed": int(args.seed),
            "split": args.split,
            "dataset": dataset_name,
            "model": model_type,
            "forget_class": int(forget_class),
            "oracle_suite": oracle_suite,
            "model_suites": list(model_suites),
            "representations": list(args.representations),
            "max_forget_samples": int(args.max_forget_samples),
            "max_retain_samples": int(args.max_retain_samples),
        },
        "splits": {},
    }

    results["splits"]["forget"] = _run_split(
        split_name="forget",
        model_suite_names=model_suites,
        models=models,
        oracle_suite=oracle_suite,
        loader=forget_loader,
        device=device,
        normalizer=normalizer,
        representation_kinds=args.representations,
    )
    results["splits"]["retain"] = _run_split(
        split_name="retain",
        model_suite_names=model_suites,
        models=models,
        oracle_suite=oracle_suite,
        loader=retain_loader,
        device=device,
        normalizer=normalizer,
        representation_kinds=args.representations,
    )

    save_path = (
        f"results/analysis/jacobian_access_{dataset_name}_{model_type}_forget{forget_class}"
        f"_seed_{args.seed}_{args.split}.json"
    )
    save_dict_to_json(results, save_path)

    print(f"\nSaved Jacobian-access audit to: {save_path}")
    for split_name in ["forget", "retain"]:
        print(f"\n{split_name.upper()} split accessible-energy ratios")
        for kind in args.representations:
            print(f"  [{kind}]")
            for suite_name in model_suites:
                stats = results["splits"][split_name][suite_name][kind]["summary"]["accessible_energy_ratio"]
                print(
                    f"    {suite_name}: mean={stats['mean']:.3f} median={stats['median']:.3f} "
                    f"p25={stats['p25']:.3f} p75={stats['p75']:.3f}"
                )


if __name__ == "__main__":
    main()
