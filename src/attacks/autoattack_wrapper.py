"""
AutoAttack wrapper for comprehensive adversarial evaluation
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple, Union
import numpy as np

try:
    from autoattack import AutoAttack
    AUTOATTACK_AVAILABLE = True
except ImportError:
    AUTOATTACK_AVAILABLE = False
    print("Warning: AutoAttack not available. Install with: pip install autoattack")


class AutoAttackWrapper:
    """Wrapper for AutoAttack evaluation"""

    def __init__(self,
                 model: nn.Module,
                 norm: str = "Linf",
                 eps: float = 8/255,
                 version: str = "standard",
                 attacks_to_run: Optional[List[str]] = None):
        """
        Args:
            model: Target model
            norm: Attack norm ("Linf", "L2")
            eps: Perturbation budget
            version: AutoAttack version
            attacks_to_run: List of attacks to run
        """
        if not AUTOATTACK_AVAILABLE:
            raise ImportError("AutoAttack not installed. Install with: pip install autoattack")

        self.model = model
        self.norm = norm
        self.eps = eps
        self.version = version

        self.attacks_to_run = attacks_to_run

        # Initialize AutoAttack
        aa_version = self.version
        aa_kwargs = dict(
            log_path=None  # Disable logging
        )

        if self.attacks_to_run is not None:
            aa_version = "custom"
            aa_kwargs["attacks_to_run"] = self.attacks_to_run

        self.adversary = AutoAttack(
            self.model,
            norm=self.norm,
            eps=self.eps,
            version=aa_version,
            **aa_kwargs
        )

        # Try to enable verbosity if supported
        if hasattr(self.adversary, 'verbose'):
            self.adversary.verbose = True

    def run_attack(self, inputs: torch.Tensor, labels: torch.Tensor,
                  batch_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run AutoAttack on inputs

        Args:
            inputs: Input images
            labels: True labels
            batch_size: Batch size for attack

        Returns:
            Tuple of (adversarial inputs, attack success indicators)
        """
        # Set batch size
        self.adversary.batch_size = batch_size

        # Run attack
        adv_inputs = self.adversary.run_standard_evaluation(
            inputs, labels, return_labels=False
        )

        # Get attack success (1 = successful attack, 0 = failed)
        with torch.no_grad():
            adv_outputs = self.model(adv_inputs)
            _, adv_predicted = adv_outputs.max(1)
            attack_success = (adv_predicted != labels).float()

        return adv_inputs, attack_success

    def evaluate_robustness(self, data_loader: torch.utils.data.DataLoader,
                           batch_size: int = 128,
                           device: str = "cuda") -> Dict[str, float]:
        """
        Evaluate robustness on a dataset

        Args:
            data_loader: Data loader
            batch_size: Batch size for evaluation
            device: Device to run on

        Returns:
            Dictionary with robustness metrics
        """
        self.model.eval()
        all_labels = []
        all_attack_success = []

        print(f"Running AutoAttack evaluation with {self.norm} norm, eps={self.eps}")

        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Run attack
            _, attack_success = self.run_attack(inputs, labels, batch_size)

            all_labels.extend(labels.cpu().numpy())
            all_attack_success.extend(attack_success.cpu().numpy())

        all_labels = np.array(all_labels)
        all_attack_success = np.array(all_attack_success)

        # Calculate metrics
        robust_acc = 1.0 - np.mean(all_attack_success)

        # Per-class robustness
        classes = np.unique(all_labels)
        per_class_robust_acc = {}
        for cls in classes:
            cls_mask = (all_labels == cls)
            if np.sum(cls_mask) > 0:
                cls_robust_acc = 1.0 - np.mean(all_attack_success[cls_mask])
                per_class_robust_acc[f'class_{cls}_robust_acc'] = cls_robust_acc

        results = {
            'robust_acc': robust_acc,
            'attack_success_rate': np.mean(all_attack_success),
            **per_class_robust_acc
        }

        return results


class FallbackAutoAttack:
    """Fallback implementation when AutoAttack is not available"""

    def __init__(self, model, norm="Linf", eps=8/255, **kwargs):
        print("AutoAttack not available, using PGD fallback")
        from .pgd import PGDAttack
        # Only pass parameters that PGDAttack accepts
        pgd_norm = "l_inf" if norm.lower() == "linf" else "l2"
        self.fallback_attack = PGDAttack(
            model=model,
            eps=eps,
            alpha=eps/4,  # Default alpha
            steps=50,     # Default steps
            norm=pgd_norm
        )

    def run_attack(self, inputs: torch.Tensor, labels: torch.Tensor, **kwargs):
        """Fallback to PGD attack"""
        adv_inputs = self.fallback_attack(inputs, labels)
        return adv_inputs, None

    def evaluate_robustness(self, data_loader, **kwargs):
        """Fallback robustness evaluation"""
        from .pgd import evaluate_adversarial_robustness
        return evaluate_adversarial_robustness(
            self.fallback_attack.model, data_loader, self.fallback_attack
        )


def create_autoattack(model: nn.Module, norm: str = "Linf", eps: float = 8/255,
                     **kwargs) -> Union[AutoAttackWrapper, FallbackAutoAttack]:
    """
    Factory function to create AutoAttack instance

    Args:
        model: Target model
        norm: Attack norm
        eps: Perturbation budget
        **kwargs: Additional arguments

    Returns:
        AutoAttack instance (or fallback)
    """
    if AUTOATTACK_AVAILABLE:
        return AutoAttackWrapper(model, norm, eps, **kwargs)
    else:
        # Fallback to PGD - don't pass AutoAttack-specific kwargs
        print("Using PGD fallback for AutoAttack")
        return FallbackAutoAttack(model, norm=norm, eps=eps)


if __name__ == "__main__":
    # Test AutoAttack wrapper
    if AUTOATTACK_AVAILABLE:
        print("AutoAttack is available")
    else:
        print("AutoAttack not available - will use PGD fallback")

    print("AutoAttackWrapper class defined successfully")
    print("Run evaluation with: python scripts/4_adv_evaluate.py")
