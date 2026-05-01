#!/usr/bin/env python3
"""
Evaluate a saved VPT prompt zero-shot on multiple target checkpoints.
"""

import argparse
import json
import os
import sys
from typing import Dict, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn

from src.data import DataManager
from src.models.vit import create_vit_model
from src.models.cnn import create_cnn_model
from src.models.normalize import create_imagenet_normalizer
from src.models.peft_lora import load_lora_adapter
from src.attacks.vpt_resurrection import VPTResurrectionAttack
from src.eval import create_evaluator
from src.utils import load_config, get_device, set_seed


def _build_model_for_suite(experiment_suites: Dict, suite_name: str, device: torch.device) -> nn.Module:
    data_config = load_config("configs/data.yaml")
    model_config = load_config("configs/model.yaml")

    suite_cfg = experiment_suites[suite_name]
    if "base_model_suite" in suite_cfg:
        ref_cfg = experiment_suites[suite_cfg["base_model_suite"]]
    else:
        ref_cfg = suite_cfg

    dataset_name = ref_cfg.get("dataset", suite_cfg.get("dataset", "cifar10"))
    model_type = ref_cfg.get("model", suite_cfg.get("model", "vit_tiny"))
    dataset_info = data_config[dataset_name]

    if model_type.startswith("vit"):
        model_config_name = model_type.replace("vit_", "")
        model = create_vit_model(
            model_config["vit"][model_config_name],
            num_classes=dataset_info["num_classes"],
        )
    else:
        model = create_cnn_model(
            model_config["cnn"][model_type],
            num_classes=dataset_info["num_classes"],
        )

    return model.to(device)


def load_target_model(experiment_suites: Dict, suite_name: str, seed: int, device: torch.device) -> nn.Module:
    suite_cfg = experiment_suites[suite_name]

    if suite_name.startswith("oracle_"):
        checkpoint_path = f"checkpoints/oracle/{suite_name}_seed_{seed}_final.pt"
        fallback_path = f"checkpoints/oracle/{suite_name}_seed_{seed}_best.pt"
        if not os.path.exists(checkpoint_path):
            checkpoint_path = fallback_path
        model = _build_model_for_suite(experiment_suites, suite_name, device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    if suite_name.startswith("base_"):
        checkpoint_path = f"checkpoints/base/{suite_name}_seed_{seed}_final.pt"
        fallback_path = f"checkpoints/base/{suite_name}_seed_{seed}_best.pt"
        if not os.path.exists(checkpoint_path):
            checkpoint_path = fallback_path
        model = _build_model_for_suite(experiment_suites, suite_name, device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    if suite_name.startswith("unlearn_"):
        suite_cfg = experiment_suites[suite_name]
        base_suite = suite_cfg.get("base_model_suite")
        if not base_suite:
            raise ValueError(f"Unlearned suite {suite_name} is missing base_model_suite")
        base_model = load_target_model(experiment_suites, base_suite, seed, device)
        adapter_path = f"checkpoints/unlearn_lora/{suite_name}_seed_{seed}"
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
        return load_lora_adapter(base_model, adapter_path)

    raise ValueError(f"Unsupported target suite type: {suite_name}")


def evaluate_clean(model: nn.Module,
                   dataset_name: str,
                   forget_class: int,
                   num_classes: int,
                   device: torch.device,
                   batch_size: int,
                   num_workers: int) -> Dict[str, float]:
    data_manager = DataManager()
    test_loader = data_manager.get_dataloader(
        dataset_name,
        "test",
        batch_size=batch_size,
        num_workers=num_workers,
        use_pretrained=True,
        apply_imagenet_norm=False,
    )
    evaluator = create_evaluator(
        forget_class=forget_class,
        num_classes=num_classes,
        device=str(device),
    )
    eval_model = nn.Sequential(create_imagenet_normalizer(), model).to(device)
    return evaluator.evaluate_model(eval_model, test_loader, attack_type="clean")


def main():
    parser = argparse.ArgumentParser(description="Transfer a saved VPT prompt across target models")
    parser.add_argument("--config", required=True, help="Experiment suite config path")
    parser.add_argument("--prompt-dir", required=True, help="Path to saved VPT prompt directory")
    parser.add_argument("--prompt-seed", type=int, required=True, help="Seed used for the saved prompt")
    parser.add_argument("--target-suites", nargs="+", required=True, help="Target model suites to evaluate")
    parser.add_argument("--device", default=None, help="Device to use")
    parser.add_argument("--eval-batch-size", type=int, default=256, help="Test batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic loading")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    experiment_suites = load_config(args.config)

    prompt_dir = args.prompt_dir
    info_path = os.path.join(prompt_dir, "attack_info.json")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"attack_info.json not found in prompt dir: {prompt_dir}")

    with open(info_path, "r", encoding="utf-8") as handle:
        attack_info = json.load(handle)

    prompt_config = attack_info["prompt_config"]
    forget_class = int(attack_info["forget_class"])

    results = {
        "prompt_dir": prompt_dir,
        "prompt_seed": args.prompt_seed,
        "prompt_info": attack_info,
        "targets": {},
    }

    for target_suite in args.target_suites:
        target_model = load_target_model(experiment_suites, target_suite, args.prompt_seed, device)
        suite_cfg = experiment_suites[target_suite]
        if "base_model_suite" in suite_cfg:
            ref_cfg = experiment_suites[suite_cfg["base_model_suite"]]
        else:
            ref_cfg = suite_cfg

        dataset_name = ref_cfg.get("dataset", suite_cfg.get("dataset", "cifar10"))
        num_classes = load_config("configs/data.yaml")[dataset_name]["num_classes"]

        attack = VPTResurrectionAttack(
            target_model=target_model,
            oracle_model=None,
            forget_class=forget_class,
            prompt_config=prompt_config,
            device=str(device),
        )
        attack.load_attack_prompt(prompt_dir)

        target_metrics = evaluate_clean(
            target_model,
            dataset_name,
            forget_class,
            num_classes,
            device,
            args.eval_batch_size,
            args.num_workers,
        )
        prompted_metrics = evaluate_clean(
            attack.vpt_model,
            dataset_name,
            forget_class,
            num_classes,
            device,
            args.eval_batch_size,
            args.num_workers,
        )

        results["targets"][target_suite] = {
            "target_clean_overall_acc": target_metrics.get("overall_acc", 0.0),
            "target_clean_forget_acc": target_metrics.get("forget_acc", 0.0),
            "target_clean_retain_acc": target_metrics.get("retain_acc", 0.0),
            "prompted_clean_overall_acc": prompted_metrics.get("overall_acc", 0.0),
            "prompted_clean_forget_acc": prompted_metrics.get("forget_acc", 0.0),
            "prompted_clean_retain_acc": prompted_metrics.get("retain_acc", 0.0),
            "forget_gain_vs_target": prompted_metrics.get("forget_acc", 0.0) - target_metrics.get("forget_acc", 0.0),
            "retain_drop_vs_target": target_metrics.get("retain_acc", 0.0) - prompted_metrics.get("retain_acc", 0.0),
            "overall_drop_vs_target": target_metrics.get("overall_acc", 0.0) - prompted_metrics.get("overall_acc", 0.0),
        }

    if args.output is None:
        prompt_name = os.path.basename(os.path.normpath(prompt_dir))
        args.output = os.path.join("results", "analysis", f"{prompt_name}_transfer_eval.json")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Wrote transfer evaluation to: {args.output}")
    for target_suite, blob in results["targets"].items():
        print(
            f"{target_suite}: "
            f"forget {100*blob['target_clean_forget_acc']:.2f}% -> {100*blob['prompted_clean_forget_acc']:.2f}% "
            f"(gain {100*blob['forget_gain_vs_target']:.2f}pp), "
            f"retain drop {100*blob['retain_drop_vs_target']:.2f}pp"
        )


if __name__ == "__main__":
    main()
