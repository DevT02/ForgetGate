#!/usr/bin/env python3
"""
Script 4: Run adversarial evaluation for ForgetGate-V
Tests clean accuracy, PGD, AutoAttack, and VPT resurrection
"""

import argparse
import logging
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Tuple, Optional
import yaml

from src.data import DataManager
from src.models.vit import create_vit_model, add_vpt_to_model
from src.models.cnn import create_cnn_model, add_vpt_to_cnn_model
from src.models.peft_lora import load_lora_adapter
from src.models.normalize import create_imagenet_normalizer
from src.attacks.vpt_resurrection import VPTResurrectionAttack
from src.attacks.pgd import PGDAttack, evaluate_adversarial_robustness
from src.attacks.autoattack_wrapper import create_autoattack
from src.eval import create_evaluator
from src.utils import set_seed, ensure_dir, load_config, print_model_info, get_device
from src.utils import log_experiment, create_experiment_log, save_dict_to_json


LOGGER = logging.getLogger("forgetgate.adv_eval")


def resolve_base_suite_info(experiment_suites: Dict, suite_name: str) -> Tuple[str, Dict]:
    """Resolve the base suite that contains dataset/model info, following suite references"""
    suite = experiment_suites[suite_name]

    if "base_model_suite" in suite:
        return resolve_base_suite_info(experiment_suites, suite["base_model_suite"])
    elif "unlearned_model_suite" in suite:
        return resolve_base_suite_info(experiment_suites, suite["unlearned_model_suite"])
    else:
        return suite_name, suite


def get_evaluation_config(experiment_suites: Dict, eval_suite_name: str) -> Dict:
    """Extract dataset, model, forget_class, and model_suites from the appropriate suites"""
    eval_suite = experiment_suites[eval_suite_name]
    model_suites = eval_suite.get('model_suites', [])

    # Resolve base suite info (contains dataset and model)
    # Prefer resolving from the first listed model suite if present
    if model_suites:
        base_suite_name, base_suite = resolve_base_suite_info(experiment_suites, model_suites[0])
    else:
        base_suite_name, base_suite = resolve_base_suite_info(experiment_suites, eval_suite_name)

    # Get forget_class from unlearned suite if available
    forget_class = None
    for suite_name in model_suites:
        if 'unlearn' in suite_name:
            unlearn_suite = experiment_suites.get(suite_name, {})
            forget_class = unlearn_suite.get('unlearning', {}).get('forget_class')
            break

    # Fallback to eval suite
    if forget_class is None:
        forget_class = eval_suite.get('unlearning', {}).get('forget_class', 0)

    return {
        'dataset': base_suite.get('dataset', 'cifar10'),
        'model': base_suite.get('model', 'vit_tiny'),
        'forget_class': forget_class,
        'seeds': eval_suite.get('seeds', [42]),
        'model_suites': model_suites,   # Include model_suites
        'eval_suite_name': eval_suite_name,
        'base_suite_name': base_suite_name
    }


def build_model_from_eval_config(eval_config: Dict, device: torch.device) -> nn.Module:
    """Instantiate the correct model architecture for the dataset/model in eval_config."""
    data_config = load_config("configs/data.yaml")
    model_config = load_config("configs/model.yaml")

    dataset_name = eval_config['dataset']
    dataset_info = data_config[dataset_name]
    model_type = eval_config['model']

    if model_type.startswith('vit'):
        model_config_name = model_type.replace('vit_', '')
        vit_cfg = dict(model_config['vit'][model_config_name])
        vit_cfg['pretrained'] = False  # IMPORTANT: we're loading from checkpoint, not using pretrained
        model = create_vit_model(
            vit_cfg,
            num_classes=dataset_info['num_classes']
        )
    else:
        model = create_cnn_model(
            model_config['cnn'][model_type],
            num_classes=dataset_info['num_classes']
        )

    return model.to(device)


class ForgetCuePatchDataset(Dataset):
    """Overlay a small forget-class patch onto retain images (dependency-aware eval)."""

    def __init__(self, retain_dataset: Dataset, forget_dataset: Dataset,
                 patch_size: int = 8, alpha: float = 0.7,
                 position: str = "bottom_right", seed: int = 0):
        self.retain_dataset = retain_dataset
        self.forget_dataset = forget_dataset
        self.patch_size = patch_size
        self.alpha = alpha
        self.position = position
        self.seed = seed

    def __len__(self):
        return len(self.retain_dataset)

    def _sample_forget_index(self, idx: int) -> int:
        # Deterministic sampling per index for reproducibility
        import random
        rng = random.Random(self.seed + idx * 9973)
        return rng.randrange(len(self.forget_dataset))

    def _get_patch_slice(self, h: int, w: int) -> Tuple[slice, slice]:
        ps = self.patch_size
        if self.position == "top_left":
            return slice(0, ps), slice(0, ps)
        if self.position == "top_right":
            return slice(0, ps), slice(w - ps, w)
        if self.position == "bottom_left":
            return slice(h - ps, h), slice(0, ps)
        # default: bottom_right
        return slice(h - ps, h), slice(w - ps, w)

    def __getitem__(self, idx):
        x_retain, y_retain = self.retain_dataset[idx]
        fidx = self._sample_forget_index(idx)
        x_forget, _ = self.forget_dataset[fidx]

        # Ensure tensor shape is [C,H,W]
        if not torch.is_tensor(x_retain):
            x_retain = torch.tensor(x_retain)
        if not torch.is_tensor(x_forget):
            x_forget = torch.tensor(x_forget)

        c, h, w = x_retain.shape
        ps = min(self.patch_size, h, w)
        hs, ws = self._get_patch_slice(h, w)
        # Use a patch from the forget image (top-left crop)
        patch = x_forget[:, :ps, :ps]

        x = x_retain.clone()
        x[:, hs, ws] = (1 - self.alpha) * x[:, hs, ws] + self.alpha * patch
        x = torch.clamp(x, 0.0, 1.0)

        return x, y_retain


def load_model_for_evaluation(
    experiment_suites: Dict,
    eval_config: Dict,
    model_suite_name: str,  # Changed from model_key to actual suite name
    device: torch.device,
    seed: int,
    prompt_seed: int = None
) -> nn.Module:
    """Load model for evaluation (base, unlearned, or vpt)."""

    model_suites = eval_config.get('model_suites', [])
    if not model_suites:
        raise ValueError(
            f"Eval suite '{eval_config.get('eval_suite_name','<unknown>')}' has no model_suites. "
            "Add model_suites: [<base_suite>, <unlearn_suite>, <vpt_suite>] to your experiment_suites.yaml."
        )

    # Determine model type from suite name
    if 'base' in model_suite_name and 'unlearn' not in model_suite_name:
        # Base model
        checkpoint_path = f"checkpoints/base/{model_suite_name}_seed_{seed}_final.pt"
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Base checkpoint not found: {checkpoint_path}")
            return None

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = build_model_from_eval_config(eval_config, device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded base model from: {checkpoint_path}")
        return model

    elif 'unlearn' in model_suite_name:
        # Unlearned model - load base weights + LoRA adapter
        unlearned_suite_config = experiment_suites.get(model_suite_name, {})
        base_suite = unlearned_suite_config.get("base_model_suite", None)
        if not base_suite:
            # Fallback to first suite when base_model_suite is not available
            base_suite = model_suites[0]
            print(f"Warning: No base_model_suite in {model_suite_name}; "
                  f"falling back to first model_suites entry: {base_suite}")

        base_checkpoint = f"checkpoints/base/{base_suite}_seed_{seed}_final.pt"
        if not os.path.exists(base_checkpoint):
            print(f"Warning: Base checkpoint not found: {base_checkpoint}")
            return None

        checkpoint = torch.load(base_checkpoint, map_location=device)
        model = build_model_from_eval_config(eval_config, device)
        model.load_state_dict(checkpoint['model_state_dict'])

        adapter_path = f"checkpoints/unlearn_lora/{model_suite_name}_seed_{seed}"
        if os.path.exists(adapter_path):
            model = load_lora_adapter(model, adapter_path)
            print(f"Loaded unlearned (LoRA) model from: {adapter_path}")
        else:
            # Still return the base-weight model (useful for debugging),
            # but warn loudly.
            print(f"Warning: LoRA adapter not found at {adapter_path}. "
                  f"Evaluating base weights without unlearning adapter.")
        return model.to(device)

    elif 'vpt' in model_suite_name:
        # VPT resurrection model
        # First, find the unlearned suite (predecessor in model_suites list)
        unlearned_suite = None
        for suite in model_suites:
            if 'unlearn' in suite and 'vpt' not in suite:
                unlearned_suite = suite
                break

        if unlearned_suite is None:
            print("Warning: No unlearned suite found to base VPT on.")
            return None

        # Load unlearned model first
        unlearned_model = load_model_for_evaluation(experiment_suites, eval_config, unlearned_suite, device, seed)
        if unlearned_model is None:
            return None

        # Build VPT config from the VPT suite
        vpt_config_file = load_config("configs/vpt_attack.yaml")
        vpt_params = experiment_suites[model_suite_name].get('vpt_attack', {})

        vpt_config = {
            'prompt_type': vpt_params.get('prompt_type', 'prefix'),
            'prompt_length': vpt_params.get('prompt_length', 5),
            'init_strategy': vpt_params.get('init_strategy', 'random'),
            'dropout': vpt_config_file['vpt_prompt'].get('dropout', 0.1)
        }

        # Load trained prompt (supports cross-seed transfer)
        prompt_seed = prompt_seed if prompt_seed is not None else seed
        prompt_path = f"checkpoints/vpt_resurrector/{model_suite_name}_seed_{prompt_seed}"
        if not os.path.exists(prompt_path):
            print(f"Warning: VPT prompt not found: {prompt_path}")
            return None

        # Create attack wrapper and load prompt
        forget_class = eval_config.get('forget_class', 0)
        vpt_attack = VPTResurrectionAttack(unlearned_model, forget_class, vpt_config, device)
        vpt_attack.load_attack_prompt(prompt_path)
        print(f"Loaded VPT prompt from: {prompt_path}")

        return vpt_attack.vpt_model.to(device)

    else:
        raise ValueError(f"Unknown model suite type: {model_suite_name}")


def run_evaluation(suite_config: Dict, model: nn.Module, model_name: str,
                  data_loader: DataLoader, evaluator: 'ForgetGateEvaluator',
                  attack_configs: Dict, device: torch.device,
                  dependency_loader: Optional[DataLoader] = None) -> Dict[str, Dict]:
    """Run evaluation for a model under different attacks"""
    results = {}

    print(f"\nEvaluating {model_name}...")

    for attack_type, attack_config in attack_configs.items():
        print(f"  Running {attack_type} attack...")
        LOGGER.debug(
            "[run_evaluation] model=%s attack_type=%s attack_config=%s",
            model_name, attack_type, attack_config
        )

        try:
            # Map attack names to evaluator's expected format
            if attack_type.startswith("pgd"):
                attack_type_for_evaluator = "pgd"
            elif attack_type.startswith("autoattack"):
                attack_type_for_evaluator = "autoattack"
            elif attack_type == "vpt_only":
                attack_type_for_evaluator = "clean"
            elif attack_type == "vpt_plus_pgd":
                attack_type_for_evaluator = "pgd"
            elif attack_type == "dependency_forget_patch":
                attack_type_for_evaluator = "clean"
            elif attack_type == "backdoor_patch":
                attack_type_for_evaluator = "backdoor_patch"
            else:
                attack_type_for_evaluator = "clean"

            LOGGER.debug(
                "[run_evaluation] attack_type_for_evaluator=%s (from %s)",
                attack_type_for_evaluator, attack_type
            )

            eval_loader = data_loader
            if attack_type == "dependency_forget_patch":
                if dependency_loader is None:
                    raise ValueError("dependency_loader is required for dependency_forget_patch")
                eval_loader = dependency_loader

            metrics = evaluator.evaluate_model(
                model=model,
                data_loader=eval_loader,
                attack_type=attack_type_for_evaluator,
                attack_config=attack_config if attack_type_for_evaluator != "clean" else None
            )

            results[attack_type] = metrics

            # Print key metrics
            forget_acc = metrics.get('forget_acc', 0.0)
            retain_acc = metrics.get('retain_acc', 0.0)
            print(f"    Forget Acc: {forget_acc:.4f}, Retain Acc: {retain_acc:.4f}")

        except Exception as e:
            print(f"    Error in {attack_type}: {e}")
            results[attack_type] = {'error': str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(description="Run adversarial evaluation for ForgetGate-V")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to experiment suites config")
    parser.add_argument("--suite", type=str, required=True,
                       help="Experiment suite name")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use")
    parser.add_argument("--prompt-seed", type=int, default=None,
                       help="Seed to load VPT prompt from (default: same as --seed)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable verbose debug logging")
    parser.add_argument("--dump-configs", action="store_true",
                       help="Print loaded YAML configs and derived attack configs, then continue")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Set seed
    set_seed(args.seed)

    # Load configs
    experiment_suites = load_config(args.config)
    if args.suite not in experiment_suites:
        raise ValueError(f"Experiment suite '{args.suite}' not found in config")

    suite_config = experiment_suites[args.suite]
    adv_eval_config = load_config("configs/adv_eval.yaml")

    # IMPORTANT: show what we loaded
    LOGGER.info("Loaded suite: %s", args.suite)
    LOGGER.info("Loaded adv_eval.yaml pgd section: %s", adv_eval_config.get("pgd", {}))
    LOGGER.info("Loaded adv_eval.yaml compute section: %s", adv_eval_config.get("compute", {}))

    # Use adv_eval.yaml as the single source of truth for PGD defaults
    base_pgd = adv_eval_config.get("pgd", {})
    def make_pgd_cfg(*, eps: float, targeted: bool, apply_to: str,
                     target_class: int = None,
                     alpha: float = None, steps: int = None) -> Dict:
        _eps = float(eps)
        _alpha = float(alpha if alpha is not None else (_eps / 4.0))  # scale alpha with eps by default
        _steps = int(steps if steps is not None else base_pgd.get("steps", 10))
        cfg = {
            "eps": _eps,
            "alpha": _alpha,
            "steps": _steps,
            "norm": str(base_pgd.get("norm", "l_inf")).lower(),
            "random_start": bool(base_pgd.get("random_start", True)),
            "targeted": bool(targeted),
            "apply_to": apply_to,
        }
        if targeted:
            cfg["target_class"] = int(target_class)
        return cfg

    # Resolve evaluation configuration from base/unlearned suites
    eval_config = get_evaluation_config(experiment_suites, args.suite)

    # Setup device
    device = get_device(args.device)

    print("=" * 50)
    print(f"ForgetGate-V: Adversarial Evaluation - {args.suite}")
    print("=" * 50)
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"Dataset: {eval_config['dataset']}")
    print(f"Model type: {eval_config['model']}")
    print(f"Forget class: {eval_config['forget_class']}")

    # Get evaluation parameters
    forget_class = eval_config['forget_class']

    # Create data loader
    data_manager = DataManager()
    dataset_name = eval_config['dataset']

    test_loader = data_manager.get_dataloader(
        dataset_name, "test",
        batch_size=adv_eval_config['compute'].get('eval_batch_size', 256),
        num_workers=0,  # Set to 0 for Windows debugging
        use_pretrained=True,  # Match training resolution (224x224 for CIFAR-10)
        apply_imagenet_norm=False  # Don't apply ImageNet norm in transforms - do it in model
    )

    # Dataloader sanity
    try:
        LOGGER.info("Test loader: batches=%s batch_size=%s", len(test_loader), test_loader.batch_size)
    except Exception:
        LOGGER.info("Test loader: (len not available) batch_size=%s", getattr(test_loader, "batch_size", None))

    # Create evaluator
    evaluator = create_evaluator(forget_class=forget_class, device=str(device))

    # Define attacks to run
    attacks_to_run = suite_config.get('evaluation', {}).get('attacks', ['clean', 'pgd_linf_8'])

    attack_configs = {}
    LOGGER.info("Attacks to run (from suite): %s", attacks_to_run)

    for attack in attacks_to_run:
        if attack == "clean":
            attack_configs[attack] = {}
        elif attack == "pgd":
            pgd_cfg = adv_eval_config.get("pgd", {})
            attack_configs[attack] = {
                "eps": float(pgd_cfg.get("eps", 8/255)),
                "alpha": float(pgd_cfg.get("alpha", 2/255)),
                "steps": int(pgd_cfg.get("steps", 10)),
                "norm": "l_inf",  # or make it configurable
                "random_start": bool(pgd_cfg.get("random_start", True)),
            }
            LOGGER.debug("Built attack_config for %s: %s", attack, attack_configs[attack])
        elif attack == "pgd_linf_8":
            attack_configs[attack] = make_pgd_cfg(
                eps=base_pgd.get("eps", 8/255),
                alpha=base_pgd.get("alpha", None),
                steps=base_pgd.get("steps", None),
                targeted=False,
                apply_to="all",
            )
            LOGGER.debug("Built attack_config for %s: %s", attack, attack_configs[attack])
        elif attack == "pgd_linf_4":
            attack_configs[attack] = make_pgd_cfg(
                eps=4/255,
                targeted=False,
                apply_to="all",
            )
            LOGGER.debug("Built attack_config for %s: %s", attack, attack_configs[attack])
        elif attack == "pgd_linf_8_retain":
            attack_configs[attack] = make_pgd_cfg(
                eps=base_pgd.get("eps", 8/255),
                alpha=base_pgd.get("alpha", None),
                steps=base_pgd.get("steps", None),
                targeted=False,
                apply_to="retain",
            )
            LOGGER.debug("Built attack_config for %s: %s", attack, attack_configs[attack])
        elif attack == "pgd_linf_8_forget":
            attack_configs[attack] = make_pgd_cfg(
                eps=base_pgd.get("eps", 8/255),
                alpha=base_pgd.get("alpha", None),
                steps=base_pgd.get("steps", None),
                targeted=False,
                apply_to="forget",
            )
            LOGGER.debug("Built attack_config for %s: %s", attack, attack_configs[attack])
        elif attack == "pgd_strong":
            # Stronger PGD: 20 steps, 5 restarts, eps/10 step size
            strong_cfg = adv_eval_config.get("pgd_strong", {})
            attack_configs[attack] = {
                "eps": float(strong_cfg.get("eps", 8/255)),
                "alpha": float(strong_cfg.get("alpha", 8/255/10)),
                "steps": int(strong_cfg.get("steps", 20)),
                "norm": "l_inf",
                "random_start": bool(strong_cfg.get("random_start", True)),
                "restarts": int(strong_cfg.get("restarts", 5)),
                "targeted": False,
                "apply_to": "all",
            }
            LOGGER.debug("Built attack_config for %s: %s", attack, attack_configs[attack])
        elif attack == "pgd_very_strong":
            # Very strong PGD: 50 steps, 10 restarts
            vs_cfg = adv_eval_config.get("pgd_very_strong", {})
            attack_configs[attack] = {
                "eps": float(vs_cfg.get("eps", 8/255)),
                "alpha": float(vs_cfg.get("alpha", 8/255/10)),
                "steps": int(vs_cfg.get("steps", 50)),
                "norm": "l_inf",
                "random_start": bool(vs_cfg.get("random_start", True)),
                "restarts": int(vs_cfg.get("restarts", 10)),
                "targeted": False,
                "apply_to": "all",
            }
            LOGGER.debug("Built attack_config for %s: %s", attack, attack_configs[attack])
        elif attack.startswith("autoattack"):
            # Build AutoAttack configs from adv_eval.yaml
            aa_cfg = adv_eval_config.get("autoattack", {})
            if attack == "autoattack_linf_8":
                # Legacy hardcoded attack - use default eps
                eps = 8/255
            else:
                # Try to parse eps from attack name, fallback to first eps in config
                eps = aa_cfg.get("epsilons", [8/255])[0]
                if "_linf_" in attack or "_l2_" in attack:
                    try:
                        eps_val = int(attack.split("_")[-1]) / 255.0
                        eps = eps_val
                    except (ValueError, IndexError):
                        pass

            attack_configs[attack] = {
                "norm": aa_cfg.get("norm", "Linf"),
                "eps": float(eps),
                "version": aa_cfg.get("version", "standard"),
                "attacks_to_run": aa_cfg.get("attacks_to_run", None),
                "batch_size": int(aa_cfg.get("batch_size", 128)),
            }
            LOGGER.debug("Built attack_config for %s: %s", attack, attack_configs[attack])
        elif attack == "vpt_only":
            attack_configs[attack] = {}
        elif attack == "vpt_plus_pgd":
            # Adaptive attacker: optimize perturbation to RESURRECT the forgotten class
            attack_configs[attack] = make_pgd_cfg(
                eps=base_pgd.get("eps", 8/255),
                alpha=base_pgd.get("alpha", None),
                steps=base_pgd.get("steps", None),
                targeted=True,
                target_class=int(forget_class),
                apply_to="forget",
            )
            LOGGER.debug("Built attack_config for %s: %s", attack, attack_configs[attack])
        elif attack == "dependency_forget_patch":
            dep_cfg = adv_eval_config.get("dependency_patch", {})
            attack_configs[attack] = {
                "patch_size": int(dep_cfg.get("patch_size", 8)),
                "alpha": float(dep_cfg.get("alpha", 0.7)),
                "position": str(dep_cfg.get("position", "bottom_right")),
            }
            LOGGER.debug("Built attack_config for %s: %s", attack, attack_configs[attack])
        elif attack == "backdoor_patch":
            bd_cfg = adv_eval_config.get("backdoor_patch", {})
            bd_target = bd_cfg.get("target_class", forget_class)
            if bd_target is None:
                bd_target = forget_class
            attack_configs[attack] = {
                "type": str(bd_cfg.get("type", "patch")),
                "patch_size": int(bd_cfg.get("patch_size", 4)),
                "position": str(bd_cfg.get("position", "bottom_right")),
                "pattern": str(bd_cfg.get("pattern", "checkerboard")),
                "intensity": float(bd_cfg.get("intensity", 1.0)),
                "target_class": int(bd_target),
            }
            LOGGER.debug("Built attack_config for %s: %s", attack, attack_configs[attack])

    if args.dump_configs:
        LOGGER.info("Final derived attack_configs: %s", attack_configs)

    # Optional dependency-aware loader (retain-only with forget cue patch)
    dependency_loader = None
    if "dependency_forget_patch" in attacks_to_run:
        dep_cfg = adv_eval_config.get("dependency_patch", {})
        retain_dataset = data_manager.load_dataset(
            dataset_name, "test",
            exclude_classes=[forget_class],
            use_pretrained=True,
            apply_imagenet_norm=False
        )
        forget_dataset = data_manager.load_dataset(
            dataset_name, "test",
            include_classes=[forget_class],
            use_pretrained=True,
            apply_imagenet_norm=False
        )
        dep_dataset = ForgetCuePatchDataset(
            retain_dataset,
            forget_dataset,
            patch_size=int(dep_cfg.get("patch_size", 8)),
            alpha=float(dep_cfg.get("alpha", 0.7)),
            position=str(dep_cfg.get("position", "bottom_right")),
            seed=args.seed,
        )
        dependency_loader = DataLoader(
            dep_dataset,
            batch_size=adv_eval_config['compute'].get('eval_batch_size', 256),
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    # Evaluate each model
    model_suites = suite_config.get('model_suites', [])
    all_results = {}

    LOGGER.info("Model suites: %s", model_suites)

    try:
        for model_suite in model_suites:
            # Determine model type for display
            if 'base' in model_suite and 'unlearn' not in model_suite:
                model_key = "base"
            elif 'unlearn' in model_suite and 'vpt' not in model_suite:
                model_key = "unlearned"
            elif 'vpt' in model_suite:
                model_key = "vpt"
            else:
                continue

            # Load model using actual suite name
            model = load_model_for_evaluation(
                experiment_suites, eval_config, model_suite, device, args.seed, prompt_seed=args.prompt_seed
            )
            if model is None:
                continue

            # Wrap model with ImageNet normalization for adversarial evaluation
            # (attacks work in [0,1] pixel space, normalization happens inside model)
            norm_layer = create_imagenet_normalizer()
            model = torch.nn.Sequential(norm_layer, model).to(device)

            print_model_info(model, f"{model_key} model (with normalization)")

            # Filter attack configs based on model type
            # VPT-specific attacks should only run on VPT models
            relevant_attacks = {}
            for attack_name, attack_config in attack_configs.items():
                if attack_name in ["vpt_only", "vpt_plus_pgd"]:
                    # Only evaluate VPT attacks on VPT models
                    if model_key == "vpt":
                        relevant_attacks[attack_name] = attack_config
                elif attack_name == "clean" and model_key == "vpt" and "vpt_only" in attack_configs:
                    # Skip clean evaluation for VPT models when vpt_only is available (duplicate)
                    continue
                else:
                    # Regular attacks run on all models
                    relevant_attacks[attack_name] = attack_config

            # Run evaluation
            model_results = run_evaluation(
                suite_config=suite_config,
                model=model,
                model_name=model_key,
                data_loader=test_loader,
                evaluator=evaluator,
                attack_configs=relevant_attacks,
                device=device,
                dependency_loader=dependency_loader
            )

            # Save results with actual suite name to distinguish between methods
            all_results[model_suite] = model_results

    except KeyboardInterrupt:
        print("\nInterrupted - saving partial results...")

    finally:
        # Derived README metrics. Used for analysis.
        try:
            # Find first unlearning method for VPT comparison
            unlearn_clean_forget = 0.0
            for suite_name in all_results:
                if 'unlearn' in suite_name and 'vpt' not in suite_name:
                    if 'clean' in all_results[suite_name]:
                        unlearn_clean_forget = all_results[suite_name]['clean'].get('forget_acc', 0.0)
                        break

            # VPT resurrection metrics
            for suite_name in all_results:
                if 'vpt' in suite_name:
                    for atk in ['vpt_only', 'vpt_plus_pgd']:
                        if atk in all_results[suite_name] and 'forget_acc' in all_results[suite_name][atk]:
                            rsr = all_results[suite_name][atk]['forget_acc']
                            all_results[suite_name][atk]['resurrection_success_rate'] = rsr
                            all_results[suite_name][atk]['resurrection_delta_vs_unlearned'] = rsr - unlearn_clean_forget

            # Add robust_retain_acc for all models under robust attacks
            for suite_name, mr in all_results.items():
                for atk in ["pgd_linf_4", "pgd_linf_8"]:
                    if atk in mr and isinstance(mr[atk], dict):
                        ra = mr[atk].get('retain_acc', None)
                        if ra is not None:
                            mr[atk]['robust_retain_acc'] = ra
        except Exception:
            pass

        # Save results
        results_file = f"results/logs/{args.suite}_seed_{args.seed}_evaluation.json"
        save_dict_to_json(all_results, results_file)

        # Log to experiment log
        log_path = create_experiment_log(f"{args.suite}_seed_{args.seed}", suite_config)

        # Flatten results for logging
        for model_key, model_results in all_results.items():
            for attack_type, metrics in model_results.items():
                log_entry = {
                    'model': model_key,
                    'attack': attack_type,
                    **metrics
                }
                log_experiment(log_path, log_entry)

    print("\nEvaluation complete!")
    print(f"Results saved to: {results_file}")

    # Print summary
    print("\nSummary:")
    for model_key, model_results in all_results.items():
        print(f"\n{model_key.upper()} Model:")
        for attack_type, metrics in model_results.items():
            if 'error' not in metrics:
                forget_acc = metrics.get('forget_acc', 0.0)
                retain_acc = metrics.get('retain_acc', 0.0)
                print(f"  {attack_type}: Forget={forget_acc:.4f}, Retain={retain_acc:.4f}")

    print("\nNext: Analyze results with: python scripts/6_analyze_results.py")


if __name__ == "__main__":
    main()
