#!/usr/bin/env python3
"""
Smoothed certified-recovery audit (Theorem 4 of theory_appendix.tex).

For each forget / retain-control sample x, draws n_noise Gaussian
perturbations and counts how many noisy forwards land *outside* the
forget class. The Clopper-Pearson lower confidence bound on that
non-forget probability gives a Cohen-Rosenfeld-Kolter L_2 certified
radius

    R = sigma * Phi^{-1}(p_lower)

inside which the binary Gaussian-smoothed forget detector provably does not
predict the forgotten class. By default the noisy certification forwards are
unclipped, matching the Cohen-Rosenfeld-Kolter Gaussian certificate. Pass
--clip-noisy-inputs only for a clipped-distribution implementation diagnostic.
Pair training-time sigma in configs/unlearning.yaml::smoothed_margin with
audit-time sigma here.

Output JSON has the same forget_recovery / retain_control shape as
17_recovery_radius_audit.py so downstream analysis scripts can consume
both without forking schemas.

Usage:
    python scripts/audits/27_smoothed_radius_audit.py \
        --model-suites unlearn_orbit_vit_cifar10_forget0 \
        --seed 42 --sigma 0.1 --n-noise 256 --alpha 0.001
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

# Project root on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import DataManager
from src.models.cnn import create_cnn_model
from src.models.normalize import create_imagenet_normalizer
from src.models.peft_lora import load_lora_adapter
from src.models.vit import create_vit_model
from src.theory.recovery_certification import smoothed_certified_radius
from src.utils import get_device, load_config, set_seed


# ---------------------------------------------------------------------------
# Minimal model/loader helpers (kept local to avoid coupling to other audit
# scripts whose imports drift). They mirror the conventions used in
# scripts/audits/17_recovery_radius_audit.py.
# ---------------------------------------------------------------------------

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


def _load_checkpoint_path_model(model, ckpt, device):
    checkpoint = torch.load(ckpt, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return model.to(device)


def _resolve_unlearn_artifact_path(suite_cfg: Dict, suite_name: str, seed: int) -> str:
    explicit = suite_cfg.get("path")
    if explicit:
        return explicit.format(seed=seed, suite_name=suite_name)
    method = suite_cfg.get("unlearning", {}).get("method", "lora")
    if method == "lora":
        return os.path.join("checkpoints", "unlearn_lora", f"{suite_name}_seed_{seed}")
    return os.path.join("checkpoints", "unlearn_full", f"{suite_name}_seed_{seed}_final.pt")


def _load_unlearn_artifact(model, artifact_path, device):
    if os.path.isdir(artifact_path):
        adapter_cfg = os.path.join(artifact_path, "adapter_config.json")
        adapter_weights = os.path.join(artifact_path, "adapter_model.safetensors")
        model_file = os.path.join(artifact_path, "model.pt")
        if os.path.exists(adapter_cfg) and os.path.exists(adapter_weights):
            return load_lora_adapter(model, artifact_path).to(device)
        if os.path.exists(model_file):
            return _load_checkpoint_path_model(model, model_file, device)
    elif os.path.isfile(artifact_path):
        return _load_checkpoint_path_model(model, artifact_path, device)
    raise FileNotFoundError(f"Unlearned artifact not found: {artifact_path}")


def _load_model(experiment_suites, suite_name, seed, device):
    if suite_name.startswith("oracle_"):
        cfg = experiment_suites[suite_name]
        model = _build_model(cfg["model"], cfg["dataset"], device)
        ckpt = _checkpoint_or_best("checkpoints/oracle", suite_name, seed)
        checkpoint = torch.load(ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(device), cfg["dataset"], cfg["model"]
    if suite_name.startswith("unlearn_"):
        cfg = experiment_suites[suite_name]
        base = cfg["base_model_suite"]
        base_cfg = experiment_suites[base]
        model = _build_model(base_cfg["model"], base_cfg["dataset"], device)
        base_ckpt = _checkpoint_or_best("checkpoints/base", base, seed)
        checkpoint = torch.load(base_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        artifact = _resolve_unlearn_artifact_path(cfg, suite_name, seed)
        if os.path.exists(artifact):
            model = _load_unlearn_artifact(model, artifact, device)
        else:
            print(f"Warning: artifact missing for {suite_name}; using base weights.")
        return model.to(device), base_cfg["dataset"], base_cfg["model"]
    if suite_name.startswith("base_"):
        cfg = experiment_suites[suite_name]
        model = _build_model(cfg["model"], cfg["dataset"], device)
        ckpt = _checkpoint_or_best("checkpoints/base", suite_name, seed)
        checkpoint = torch.load(ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(device), cfg["dataset"], cfg["model"]
    raise ValueError(f"Unsupported suite type: {suite_name}")


def _build_loader(dataset_name, split, batch_size, num_workers,
                  forget_class, retain, max_samples) -> DataLoader:
    dm = DataManager()
    kwargs = {"use_pretrained": True, "apply_imagenet_norm": False}
    if retain:
        ds = dm.load_dataset(dataset_name, split, exclude_classes=[forget_class], **kwargs)
    else:
        ds = dm.load_dataset(dataset_name, split, include_classes=[forget_class], **kwargs)
    if max_samples > 0 and len(ds) > max_samples:
        ds = torch.utils.data.Subset(ds, list(range(max_samples)))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# ---------------------------------------------------------------------------
# Clopper-Pearson lower bound (one-sided)
# ---------------------------------------------------------------------------

def cp_lower_bound(k: int, n: int, alpha: float) -> float:
    """One-sided Clopper-Pearson lower bound on the success probability.

    Returns p_L such that P[Bin(n, p_L) >= k] = alpha (the standard
    Cohen-Rosenfeld-Kolter choice). Closed form via the inverse Beta CDF.
    Falls back to a normal approximation if scipy is unavailable.
    """
    if k == 0:
        return 0.0
    if k == n:
        # one-sided: alpha**(1/n) for upper end of the lower bound
        return float(alpha ** (1.0 / n))
    try:
        from scipy.stats import beta as _beta
        return float(_beta.ppf(alpha, k, n - k + 1))
    except Exception:
        p_hat = k / n
        se = np.sqrt(p_hat * (1 - p_hat) / n)
        # one-sided normal lower bound at level alpha
        from math import erf, sqrt
        # crude inverse normal at alpha; for alpha ~ 1e-3 this is roughly -3.09
        z = 3.09 if alpha < 5e-3 else 1.96
        return max(0.0, p_hat - z * se)


# ---------------------------------------------------------------------------
# Per-sample certified-radius routine
# ---------------------------------------------------------------------------

@torch.no_grad()
def _smoothed_predict_counts(
    model: torch.nn.Module,
    normalizer: torch.nn.Module,
    inputs: torch.Tensor,
    sigma: float,
    n_noise: int,
    forget_class: int,
    batch_chunk: int = 64,
    clip_noisy_inputs: bool = False,
) -> Tuple[int, int]:
    """Returns (n_nonforget, n_noise): number of noisy forwards that
    predicted any class other than forget_class. Inputs assumed [1, ...]."""
    assert inputs.shape[0] == 1, "single-sample certification path"
    n_remaining = n_noise
    n_nonforget = 0
    while n_remaining > 0:
        m = min(batch_chunk, n_remaining)
        x = inputs.expand(m, *inputs.shape[1:]).contiguous()
        noise = torch.randn_like(x) * sigma
        x_noisy = x + noise
        if clip_noisy_inputs:
            x_noisy = x_noisy.clamp(0.0, 1.0)
        logits = model(normalizer(x_noisy))
        preds = logits.argmax(dim=1)
        n_nonforget += int((preds != forget_class).sum().item())
        n_remaining -= m
    return n_nonforget, n_noise


def _certify_one(
    model: torch.nn.Module,
    normalizer: torch.nn.Module,
    inputs: torch.Tensor,
    sigma: float,
    n_noise: int,
    alpha: float,
    forget_class: int,
    batch_chunk: int,
    clip_noisy_inputs: bool,
) -> Dict:
    k, n = _smoothed_predict_counts(
        model=model,
        normalizer=normalizer,
        inputs=inputs,
        sigma=sigma,
        n_noise=n_noise,
        forget_class=forget_class,
        batch_chunk=batch_chunk,
        clip_noisy_inputs=clip_noisy_inputs,
    )
    p_hat = k / n
    p_lower = cp_lower_bound(k, n, alpha)
    R = smoothed_certified_radius(p_lower, sigma)
    return {
        "n": int(n),
        "k_nonforget": int(k),
        "p_hat_nonforget": float(p_hat),
        "p_lower_nonforget": float(p_lower),
        "certified_radius_l2": float(R),
        "sigma": float(sigma),
        "alpha": float(alpha),
        "clip_noisy_inputs": bool(clip_noisy_inputs),
    }


def _summarize(records: List[Dict]) -> Dict:
    Rs = np.array([r["certified_radius_l2"] for r in records], dtype=float)
    p_hats = np.array([r["p_hat_nonforget"] for r in records], dtype=float)
    p_lowers = np.array([r["p_lower_nonforget"] for r in records], dtype=float)
    return {
        "n_samples": int(len(records)),
        "mean_certified_radius_l2": float(Rs.mean()) if len(Rs) else float("nan"),
        "median_certified_radius_l2": float(np.median(Rs)) if len(Rs) else float("nan"),
        "frac_with_positive_certificate": float((Rs > 0).mean()) if len(Rs) else float("nan"),
        "mean_p_hat_nonforget": float(p_hats.mean()) if len(p_hats) else float("nan"),
        "mean_p_lower_nonforget": float(p_lowers.mean()) if len(p_lowers) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-suites", nargs="+", required=True,
                   help="Unlearn suite names, e.g. unlearn_orbit_vit_cifar10_forget0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--forget-class", type=int, default=0)
    p.add_argument("--max-forget", type=int, default=64)
    p.add_argument("--max-retain", type=int, default=64)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--n-noise", type=int, default=256)
    p.add_argument("--alpha", type=float, default=1e-3, help="Clopper-Pearson confidence level")
    p.add_argument("--batch-chunk", type=int, default=64,
                   help="GPU minibatch size for noisy forward passes per sample")
    p.add_argument("--clip-noisy-inputs", action="store_true",
                   help="Clip x+noise into [0,1]. Omit for the unclipped Gaussian certificate.")
    p.add_argument("--skip-retain-control", action="store_true")
    p.add_argument("--output", default=None,
                   help="Optional explicit output JSON path; else auto-named under results/analysis/metrics")
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    experiment_suites = load_config("configs/experiment_suites.yaml")

    results = {
        "meta": {
            "seed": args.seed,
            "split": args.split,
            "forget_class": args.forget_class,
            "model_suites": args.model_suites,
            "sigma": args.sigma,
            "n_noise": args.n_noise,
            "alpha": args.alpha,
            "clip_noisy_inputs": bool(args.clip_noisy_inputs),
            "audit_type": "smoothed_certified_radius_l2",
            "theorem": "Theorem 4 of theory_appendix.tex",
        },
        "forget_recovery": {},
        "retain_control": {},
    }

    normalizer = create_imagenet_normalizer().to(device)

    for suite_name in args.model_suites:
        print(f"\n=== Suite: {suite_name} ===")
        model, dataset_name, model_type = _load_model(
            experiment_suites, suite_name, args.seed, device,
        )
        model.eval()

        forget_loader = _build_loader(
            dataset_name=dataset_name,
            split=args.split,
            batch_size=1,
            num_workers=0,
            forget_class=args.forget_class,
            retain=False,
            max_samples=args.max_forget,
        )

        print(f"Certifying forget radii for {suite_name} (sigma={args.sigma}, n={args.n_noise})...")
        records = []
        for idx, (inputs, _labels) in enumerate(forget_loader):
            inputs = inputs.to(device)
            rec = _certify_one(
                model=model,
                normalizer=normalizer,
                inputs=inputs,
                sigma=args.sigma,
                n_noise=args.n_noise,
                alpha=args.alpha,
                forget_class=args.forget_class,
                batch_chunk=args.batch_chunk,
                clip_noisy_inputs=args.clip_noisy_inputs,
            )
            rec["sample_index"] = idx
            records.append(rec)
            if (idx + 1) % 16 == 0:
                print(f"  forget {idx+1}/{len(forget_loader)} median R = "
                      f"{np.median([r['certified_radius_l2'] for r in records]):.4f}")

        results["forget_recovery"][suite_name] = {
            "summary": _summarize(records),
            "per_sample": records,
        }

        if not args.skip_retain_control:
            retain_loader = _build_loader(
                dataset_name=dataset_name,
                split=args.split,
                batch_size=1,
                num_workers=0,
                forget_class=args.forget_class,
                retain=True,
                max_samples=args.max_retain,
            )
            print(f"Certifying retain-control radii for {suite_name}...")
            control_records = []
            for idx, (inputs, _labels) in enumerate(retain_loader):
                inputs = inputs.to(device)
                rec = _certify_one(
                    model=model,
                    normalizer=normalizer,
                    inputs=inputs,
                    sigma=args.sigma,
                    n_noise=args.n_noise,
                    alpha=args.alpha,
                    forget_class=args.forget_class,
                    batch_chunk=args.batch_chunk,
                    clip_noisy_inputs=args.clip_noisy_inputs,
                )
                rec["sample_index"] = idx
                control_records.append(rec)
                if (idx + 1) % 16 == 0:
                    print(f"  retain {idx+1}/{len(retain_loader)} median R = "
                          f"{np.median([r['certified_radius_l2'] for r in control_records]):.4f}")

            results["retain_control"][suite_name] = {
                "summary": _summarize(control_records),
                "per_sample": control_records,
            }

        # Cleanup before next suite to keep GPU memory steady
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save
    if args.output:
        save_path = args.output
    else:
        os.makedirs("results/analysis/metrics", exist_ok=True)
        sig_tag = str(args.sigma).replace(".", "p")
        suite_tag = "_".join(s.replace("unlearn_", "") for s in args.model_suites)[:80]
        save_path = (
            f"results/analysis/metrics/smoothed_radius_seed_{args.seed}"
            f"_sigma_{sig_tag}_n{args.n_noise}_{suite_tag}.json"
        )
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved smoothed certified-radius audit -> {save_path}")


if __name__ == "__main__":
    main()
