"""
Evaluation metrics and utilities for ForgetGate-V
Computes forget/retain metrics and resurrection success
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Tuple, Union
import numpy as np
import json
import os
from datetime import datetime
from tqdm import tqdm


class ForgetGateEvaluator:
    """Evaluator for forget/retain metrics and resurrection success"""

    def __init__(self,
                 forget_class: int,
                 num_classes: int = 10,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Args:
            forget_class: Class that should be forgotten
            num_classes: Total number of classes
            device: Device to run evaluation on
        """
        self.forget_class = forget_class
        self.num_classes = num_classes
        self.device = device

        # Results storage
        self.results_history = []

    def evaluate_model(self,
                      model: nn.Module,
                      data_loader: DataLoader,
                      attack_type: str = "clean",
                      attack_config: Optional[Dict] = None) -> Dict[str, float]:
        """
        Evaluate model on forget/retain metrics

        Args:
            model: Model to evaluate
            data_loader: Data loader
            attack_type: Type of attack ("clean", "pgd", "autoattack", "vpt", "joint")
            attack_config: Attack configuration

        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        attack_config = attack_config or {}

        # Initialize attack once outside the loop
        attack = None
        if attack_type == "pgd":
            from .attacks.pgd import PGDAttack, LowRankPGDAttack

            pgd_kind = attack_config.get("pgd_kind", "standard")  # "standard" | "lowrank"
            if pgd_kind == "lowrank":
                attack = LowRankPGDAttack(
                    model,
                    eps=attack_config.get("eps", 8/255),
                    alpha=attack_config.get("alpha", 2/255),
                    steps=attack_config.get("steps", 10),
                    norm=attack_config.get("norm", "l_inf"),
                    random_start=attack_config.get("random_start", True),
                    rank=attack_config.get("rank", 5),
                    downsample=attack_config.get("downsample", 8),
                )
            else:
                attack = PGDAttack(
                    model,
                    eps=attack_config.get("eps", 8/255),
                    alpha=attack_config.get("alpha", 2/255),
                    steps=attack_config.get("steps", 10),
                    norm=attack_config.get("norm", "l_inf"),
                    random_start=attack_config.get("random_start", True),
                    restarts=attack_config.get("restarts", 1),  # Support multiple restarts
                )
        elif attack_type == "autoattack":
            from .attacks.autoattack_wrapper import create_autoattack
            attack = create_autoattack(
                model,
                norm=attack_config.get("norm", "Linf"),
                eps=attack_config.get("eps", 8/255),
                version=attack_config.get("version", "standard"),
                attacks_to_run=attack_config.get("attacks_to_run", None),
            )

        # Initialize counters
        total_correct = 0
        forget_correct = 0
        retain_correct = 0
        total_samples = 0
        forget_samples = 0
        retain_samples = 0

        all_logits = []
        all_labels = []

        for inputs, labels in tqdm(data_loader, desc=f"Eval ({attack_type})"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            batch_size = inputs.size(0)

            # Sanity check: inputs should be in [0,1] for adversarial attacks (first batch only)
            if total_samples == 0:
                mn, mx = inputs.min().item(), inputs.max().item()
                print(f"[sanity] input range: min={mn:.4f} max={mx:.4f}")
                print(f"[sanity] input shape: {inputs.shape}")
                assert mn >= -1e-3 and mx <= 1.0 + 1e-3, \
                    "Adversarial eval expects inputs in [0,1]. Did you normalize in transforms?"

            # Apply attack if specified (needs gradients)
            if attack_type == "pgd":
                targeted = bool(attack_config.get("targeted", False))
                target_class = int(attack_config.get("target_class", self.forget_class))
                apply_to = str(attack_config.get("apply_to", "all"))  # "all" | "forget" | "retain"

                if apply_to not in {"all", "forget", "retain"}:
                    raise ValueError(f"Unknown apply_to='{apply_to}'. Expected all/forget/retain")

                def _run_pgd(x, y):
                    if targeted:
                        t = torch.full_like(y, target_class)
                        return attack(x, y, targeted=True, target_labels=t)
                    return attack(x, y, targeted=False, target_labels=None)

                if apply_to == "all":
                    inputs = _run_pgd(inputs, labels)
                else:
                    mask = (labels == self.forget_class) if apply_to == "forget" else (labels != self.forget_class)
                    if mask.any():
                        adv = inputs.clone()
                        adv[mask] = _run_pgd(inputs[mask], labels[mask])
                        inputs = adv
            elif attack_type == "autoattack":
                adv_inputs, _ = attack.run_attack(
                    inputs, labels,
                    batch_size=int(attack_config.get("batch_size", 128))
                )
                inputs = adv_inputs
            elif attack_type != "clean":
                inputs = self._apply_attack(inputs, labels, attack_type, attack_config, model)

            # Forward pass (under no_grad for efficiency)
            with torch.no_grad():
                outputs = model(inputs)
                _, predicted = outputs.max(1)

                # Quick sanity check for PGD (first batch only)
                if attack_type == "pgd" and total_samples == 0:
                    print("first batch predicted unique:", predicted.unique().numel())
                    print("first batch correct:", predicted.eq(labels).sum().item(), "/", labels.numel())

                # Store for detailed analysis
                all_logits.append(outputs.cpu())
                all_labels.append(labels.cpu())

                # Overall accuracy
                total_correct += predicted.eq(labels).sum().item()
                total_samples += batch_size

                # Forget class accuracy
                forget_mask = (labels == self.forget_class)
                forget_correct += (predicted[forget_mask] == labels[forget_mask]).sum().item()
                forget_samples += forget_mask.sum().item()

                # Retain class accuracy
                retain_mask = ~forget_mask
                retain_correct += (predicted[retain_mask] == labels[retain_mask]).sum().item()
                retain_samples += retain_mask.sum().item()

        # Calculate metrics
        overall_acc = total_correct / total_samples if total_samples > 0 else 0.0
        forget_acc = forget_correct / forget_samples if forget_samples > 0 else 0.0
        retain_acc = retain_correct / retain_samples if retain_samples > 0 else 0.0

        # Concatenate logits and labels
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Additional metrics
        metrics = {
            'overall_acc': overall_acc,
            'forget_acc': forget_acc,
            'retain_acc': retain_acc,
            'forget_samples': forget_samples,
            'retain_samples': retain_samples,
            'total_samples': total_samples,
            'attack_type': attack_type
        }

        # Confidence metrics
        confidence_metrics = self._compute_confidence_metrics(all_logits, all_labels)
        metrics.update(confidence_metrics)

        # Entropy metrics
        entropy_metrics = self._compute_entropy_metrics(all_logits)
        metrics.update(entropy_metrics)

        return metrics

    def _apply_attack(self, inputs: torch.Tensor, labels: torch.Tensor,
                     attack_type: str, attack_config: Dict, model: nn.Module) -> torch.Tensor:
        """Apply specified attack to inputs"""
        if attack_type == "vpt":
            # VPT attack would be applied externally
            # For now, assume inputs are already attacked
            return inputs

        elif attack_type == "joint":
            # Joint VPT + adversarial attack
            # Would need VPT attack instance
            return inputs

        else:
            return inputs

    def _compute_confidence_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Compute confidence-related metrics"""
        probs = F.softmax(logits, dim=-1)

        # Confidence of predictions
        confidences, _ = probs.max(dim=-1)

        # Correct prediction confidence
        correct_mask = (logits.argmax(dim=-1) == labels)
        correct_confidence = confidences[correct_mask].mean().item() if correct_mask.any() else 0.0

        # Incorrect prediction confidence
        incorrect_confidence = confidences[~correct_mask].mean().item() if (~correct_mask).any() else 0.0

        # Forget class specific confidence
        forget_mask = (labels == self.forget_class)
        if forget_mask.any():
            forget_confidence = confidences[forget_mask].mean().item()
            forget_correct_confidence = confidences[forget_mask & correct_mask].mean().item() if (forget_mask & correct_mask).any() else 0.0
        else:
            forget_confidence = 0.0
            forget_correct_confidence = 0.0

        return {
            'mean_confidence': confidences.mean().item(),
            'correct_confidence': correct_confidence,
            'incorrect_confidence': incorrect_confidence,
            'forget_confidence': forget_confidence,
            'forget_correct_confidence': forget_correct_confidence
        }

    def _compute_entropy_metrics(self, logits: torch.Tensor) -> Dict[str, float]:
        """Compute entropy-related metrics"""
        probs = F.softmax(logits, dim=-1)

        # Prediction entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)

        return {
            'mean_entropy': entropy.mean().item(),
            'max_entropy': entropy.max().item(),
            'min_entropy': entropy.min().item()
        }

    def evaluate_resurrection_success(self,
                                    clean_metrics: Dict[str, float],
                                    attacked_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluate resurrection success rate

        Args:
            clean_metrics: Metrics on clean inputs
            attacked_metrics: Metrics on attacked inputs

        Returns:
            Resurrection success metrics
        """
        # Resurrection success rate: fraction of forget samples that become
        # correctly classified after attack
        resurrection_success_rate = attacked_metrics['forget_acc']

        # Relative improvement
        clean_forget_acc = clean_metrics.get('forget_acc', 0.0)
        relative_improvement = resurrection_success_rate - clean_forget_acc

        # Absolute improvement
        absolute_improvement = max(0, resurrection_success_rate - clean_forget_acc)

        return {
            'resurrection_success_rate': resurrection_success_rate,
            'relative_improvement': relative_improvement,
            'absolute_improvement': absolute_improvement,
            'clean_forget_acc': clean_forget_acc,
            'attacked_forget_acc': resurrection_success_rate
        }

    def evaluate_unlearning_quality(self,
                                  baseline_metrics: Dict[str, float],
                                  unlearned_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluate unlearning quality metrics

        Args:
            baseline_metrics: Metrics from baseline model
            unlearned_metrics: Metrics from unlearned model

        Returns:
            Unlearning quality metrics
        """
        # Forget quality: how much forget accuracy decreased
        forget_quality = max(0, baseline_metrics.get('forget_acc', 1.0) - unlearned_metrics.get('forget_acc', 0.0))

        # Retain utility drop: how much retain accuracy decreased
        retain_utility_drop = max(0, baseline_metrics.get('retain_acc', 1.0) - unlearned_metrics.get('retain_acc', 0.0))

        # Overall utility drop
        overall_utility_drop = max(0, baseline_metrics.get('overall_acc', 1.0) - unlearned_metrics.get('overall_acc', 0.0))

        return {
            'forget_quality': forget_quality,
            'retain_utility_drop': retain_utility_drop,
            'overall_utility_drop': overall_utility_drop,
            'baseline_forget_acc': baseline_metrics.get('forget_acc', 0.0),
            'baseline_retain_acc': baseline_metrics.get('retain_acc', 0.0),
            'unlearned_forget_acc': unlearned_metrics.get('forget_acc', 0.0),
            'unlearned_retain_acc': unlearned_metrics.get('retain_acc', 0.0)
        }

    def save_results(self, results: Dict, save_path: str, experiment_name: str = ""):
        """Save evaluation results"""
        os.makedirs(save_path, exist_ok=True)

        # Add timestamp and experiment info
        results_with_meta = {
            'timestamp': datetime.now().isoformat(),
            'experiment': experiment_name,
            'forget_class': self.forget_class,
            **results
        }

        self.results_history.append(results_with_meta)

        # Save individual result
        result_file = os.path.join(save_path, f"eval_{experiment_name}_{len(self.results_history)}.json")
        with open(result_file, 'w') as f:
            json.dump(results_with_meta, f, indent=2)

        # Save complete history
        history_file = os.path.join(save_path, "evaluation_history.json")
        with open(history_file, 'w') as f:
            json.dump(self.results_history, f, indent=2)

    def aggregate_results(self, results_list: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Aggregate results across multiple runs/seeds"""
        if not results_list:
            return {}

        # Extract numeric metrics
        metric_keys = []
        for result in results_list:
            for key, value in result.items():
                if isinstance(value, (int, float)) and key not in ['timestamp', 'forget_class']:
                    if key not in metric_keys:
                        metric_keys.append(key)

        # Aggregate each metric
        aggregated = {}
        for key in metric_keys:
            values = [r.get(key, 0.0) for r in results_list if isinstance(r.get(key), (int, float))]
            if values:
                aggregated[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }

        return aggregated


def create_evaluator(forget_class: int, num_classes: int = 10, device: str = None) -> ForgetGateEvaluator:
    """Factory function to create evaluator"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return ForgetGateEvaluator(forget_class, num_classes, device)


if __name__ == "__main__":
    # Test evaluator
    evaluator = create_evaluator(forget_class=0, num_classes=10)
    print(f"Created evaluator for forget class {evaluator.forget_class}")

    # Test metrics computation
    print("Evaluator class defined successfully")
    print("Run evaluation with: python scripts/4_adv_evaluate.py")
