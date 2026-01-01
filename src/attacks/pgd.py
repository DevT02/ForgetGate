"""
PGD (Projected Gradient Descent) adversarial attack implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple, Union
import numpy as np


class PGDAttack:
    """PGD adversarial attack"""

    def __init__(self,
                 model: nn.Module,
                 eps: float = 8/255,
                 alpha: float = 2/255,
                 steps: int = 10,
                 random_start: bool = True,
                 norm: str = "l_inf",
                 restarts: int = 1):
        """
        Args:
            model: Target model
            eps: Maximum perturbation
            alpha: Step size
            steps: Number of PGD steps
            random_start: Random initialization
            norm: Norm for perturbation ("l_inf", "l2")
            restarts: Number of random restarts (for stronger attacks)
        """
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.norm = norm
        self.restarts = restarts

    def __call__(self, inputs: torch.Tensor, labels: torch.Tensor,
                 targeted: bool = False, target_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate adversarial examples with optional restarts

        Args:
            inputs: Clean inputs
            labels: True labels
            targeted: Whether to perform targeted attack
            target_labels: Target labels for targeted attack

        Returns:
            Adversarial examples
        """
        # If using restarts, run PGD multiple times and keep the best
        if self.restarts > 1:
            return self._pgd_with_restarts(inputs, labels, targeted, target_labels)
        else:
            return self._pgd_single(inputs, labels, targeted, target_labels)

    def _pgd_single(self, inputs: torch.Tensor, labels: torch.Tensor,
                    targeted: bool = False, target_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Single PGD run without restarts"""
        # Clone inputs to avoid modifying originals
        adv_inputs = inputs.clone().detach()

        # Random initialization
        if self.random_start:
            adv_inputs = self._random_init(adv_inputs)

        # PGD iterations
        for step in range(self.steps):
            adv_inputs.requires_grad_(True)

            # Forward pass
            outputs = self.model(adv_inputs)

            # Compute loss
            if targeted and target_labels is not None:
                loss = F.cross_entropy(outputs, target_labels)
            else:
                loss = F.cross_entropy(outputs, labels)

            # Backward pass (only compute grad w.r.t. adv_inputs for speed)
            grad = torch.autograd.grad(loss, adv_inputs, only_inputs=True)[0]

            # Update adversarial examples
            with torch.no_grad():
                # Compute update direction based on norm
                if self.norm == "l_inf":
                    direction = grad.sign()
                elif self.norm == "l2":
                    # For L2, normalize gradient per sample
                    g = grad.view(grad.size(0), -1)
                    g_norm = g.norm(p=2, dim=1, keepdim=True) + 1e-12
                    direction = (g / g_norm).view_as(grad)
                else:
                    raise ValueError(f"Unsupported norm: {self.norm}")

                if targeted:
                    # Targeted: minimize loss to target class
                    adv_inputs = adv_inputs - self.alpha * direction
                else:
                    # Untargeted: maximize loss
                    adv_inputs = adv_inputs + self.alpha * direction

                # Project back to epsilon ball
                adv_inputs = self._project(adv_inputs, inputs)

            adv_inputs = adv_inputs.detach()

        # Debug logging for first batch only
        if not hasattr(self, '_debug_logged'):
            self._debug_logged = True
            with torch.no_grad():
                # Log input ranges to verify [0,1] clamping is appropriate
                input_min, input_max = inputs.min().item(), inputs.max().item()

                # Compute perturbation statistics
                delta = adv_inputs - inputs
                max_delta = delta.abs().max().item()
                l2_delta = torch.norm(delta.view(delta.size(0), -1), p=2, dim=1).mean().item()

                # Compute accuracies
                clean_outputs = self.model(inputs)
                adv_outputs = self.model(adv_inputs)
                clean_preds = clean_outputs.argmax(dim=1)
                adv_preds = adv_outputs.argmax(dim=1)
                clean_acc = (clean_preds == labels).float().mean().item()
                adv_acc = (adv_preds == labels).float().mean().item()

                print("\n--- PGD Debug (first batch only) ---")
                print(f"Input range: [{input_min:.3f}, {input_max:.3f}] (should be ~[0,1] for PGD)")
                print(f"PGD Config: eps={self.eps:.6f}, alpha={self.alpha:.6f}, steps={self.steps}, norm={self.norm}")
                print(f"Perturbation: max|δ|={max_delta:.6f}, ||δ||₂_avg={l2_delta:.6f}")
                print(f"Accuracy: clean={clean_acc:.4f}, adv={adv_acc:.4f}")
                print("--- End PGD Debug ---\n")

        return adv_inputs

    def _pgd_with_restarts(self, inputs: torch.Tensor, labels: torch.Tensor,
                           targeted: bool = False, target_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """PGD with multiple random restarts - keeps the best adversarial example"""
        best_adv = inputs.clone()
        best_loss = torch.zeros(inputs.size(0), device=inputs.device)
        if not targeted:
            best_loss.fill_(float('-inf'))  # For untargeted, we want to maximize loss
        else:
            best_loss.fill_(float('inf'))   # For targeted, we want to minimize loss

        for restart in range(self.restarts):
            # Run a single PGD iteration
            adv_inputs = self._pgd_single(inputs, labels, targeted, target_labels)

            # Evaluate this restart's performance
            with torch.no_grad():
                outputs = self.model(adv_inputs)
                if targeted and target_labels is not None:
                    # Targeted: minimize CE loss to target
                    loss = F.cross_entropy(outputs, target_labels, reduction='none')
                else:
                    # Untargeted: maximize CE loss
                    loss = F.cross_entropy(outputs, labels, reduction='none')

                # Update best adversarial examples
                if not targeted:
                    # Untargeted: keep examples with higher loss
                    update_mask = loss > best_loss
                else:
                    # Targeted: keep examples with lower loss
                    update_mask = loss < best_loss

                best_loss = torch.where(update_mask, loss, best_loss)
                best_adv = torch.where(update_mask.view(-1, 1, 1, 1), adv_inputs, best_adv)

        return best_adv

    def _random_init(self, inputs: torch.Tensor) -> torch.Tensor:
        """Random initialization within epsilon ball"""
        if self.norm == "l_inf":
            noise = torch.empty_like(inputs).uniform_(-self.eps, self.eps)
        elif self.norm == "l2":
            # Generate random direction and scale to eps
            noise = torch.randn_like(inputs)
            noise_flat = noise.view(noise.size(0), -1)
            noise_norm = noise_flat.norm(p=2, dim=1, keepdim=True) + 1e-12
            noise = (noise_flat / noise_norm).view(inputs.shape) * self.eps
        else:
            raise ValueError(f"Unsupported norm: {self.norm}")

        return torch.clamp(inputs + noise, 0, 1)

    def _project(self, adv_inputs: torch.Tensor, clean_inputs: torch.Tensor) -> torch.Tensor:
        """Project back to epsilon ball around clean inputs"""
        if self.norm == "l_inf":
            perturbation = torch.clamp(adv_inputs - clean_inputs, -self.eps, self.eps)
            return torch.clamp(clean_inputs + perturbation, 0, 1)
        elif self.norm == "l2":
            perturbation = adv_inputs - clean_inputs
            perturbation_norm = torch.norm(perturbation.view(perturbation.size(0), -1), p=2, dim=1, keepdim=True)

            # Project to L2 ball of radius eps
            factor = torch.min(torch.ones_like(perturbation_norm), self.eps / (perturbation_norm + 1e-8))
            perturbation = perturbation * factor
            perturbation = perturbation.view(adv_inputs.shape)

            return torch.clamp(clean_inputs + perturbation, 0, 1)
        else:
            raise ValueError(f"Unsupported norm: {self.norm}")


class LowRankPGDAttack(PGDAttack):
    """
    PGD where each step's update is projected onto a low-rank subspace
    (approx via downsample + svd_lowrank per sample/channel).
    """

    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, random_start=True,
                 norm="l_inf", rank: int = 5, downsample: int = 8, niter: int = 2):
        super().__init__(model, eps, alpha, steps, random_start, norm)
        self.rank = int(rank)
        self.downsample = int(downsample)
        self.niter = int(niter)

    def _lowrank_project_update(self, g: torch.Tensor) -> torch.Tensor:
        # g: [B,C,H,W] gradient/update tensor
        B, C, H, W = g.shape

        # Downsample to make low-rank projection cheap
        if self.downsample > 1:
            h2, w2 = max(1, H // self.downsample), max(1, W // self.downsample)
            gd = F.interpolate(g, size=(h2, w2), mode="bilinear", align_corners=False)
        else:
            gd = g
            h2, w2 = H, W

        out = torch.zeros_like(gd)

        # Project each (sample, channel) matrix to rank-k
        # NOTE: loops are OK because h2,w2 are small after downsample (e.g., 28x28)
        for b in range(B):
            for c in range(C):
                M = gd[b, c]  # [h2, w2]
                # svd_lowrank returns U:[h2,q], S:[q], V:[w2,q]
                # Works well for small q (rank).
                U, S, V = torch.svd_lowrank(M, q=self.rank, niter=self.niter)
                Mk = (U * S.unsqueeze(0)) @ V.t()
                out[b, c] = Mk

        # Upsample back
        if self.downsample > 1:
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

        return out

    def __call__(self, inputs, labels, targeted=False, target_labels=None):
        adv_inputs = inputs.clone().detach()
        if self.random_start:
            adv_inputs = self._random_init(adv_inputs)

        for _ in range(self.steps):
            adv_inputs.requires_grad_(True)
            outputs = self.model(adv_inputs)
            if targeted and target_labels is not None:
                loss = F.cross_entropy(outputs, target_labels)
            else:
                loss = F.cross_entropy(outputs, labels)

            grad = torch.autograd.grad(loss, adv_inputs, only_inputs=True)[0]

            # Low-rank projection of the update direction
            upd = self._lowrank_project_update(grad)

            with torch.no_grad():
                # Compute update direction based on norm
                if self.norm == "l_inf":
                    direction = upd.sign()
                elif self.norm == "l2":
                    # For L2, normalize gradient per sample
                    g = upd.view(upd.size(0), -1)
                    g_norm = g.norm(p=2, dim=1, keepdim=True) + 1e-12
                    direction = (g / g_norm).view_as(upd)
                else:
                    raise ValueError(f"Unsupported norm: {self.norm}")

                if targeted:
                    adv_inputs = adv_inputs - self.alpha * direction
                else:
                    adv_inputs = adv_inputs + self.alpha * direction
                adv_inputs = self._project(adv_inputs, inputs)

            adv_inputs = adv_inputs.detach()

        return adv_inputs


class APGDAttack:
    """Auto-PGD attack (improved PGD with automatic step size)"""

    def __init__(self,
                 model: nn.Module,
                 eps: float = 8/255,
                 steps: int = 10,
                 norm: str = "l_inf",
                 n_restarts: int = 1):
        """
        Args:
            model: Target model
            eps: Maximum perturbation
            steps: Number of steps
            norm: Norm type
            n_restarts: Number of random restarts
        """
        self.model = model
        self.eps = eps
        self.steps = steps
        self.norm = norm
        self.n_restarts = n_restarts

    def __call__(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples using Auto-PGD"""
        # For simplicity, fall back to PGD for now
        # In practice, would implement the full APGD algorithm
        pgd = PGDAttack(
            self.model,
            eps=self.eps,
            alpha=self.alpha,
            steps=self.steps,
            random_start=True,
            norm=self.norm,
        )

        best_adv = inputs.clone()
        best_loss = float('-inf')  # Start with lowest possible loss

        for restart in range(self.n_restarts):
            adv_inputs = pgd(inputs, labels)

            # Evaluate loss (for untargeted attack, we want to maximize loss)
            with torch.no_grad():
                outputs = self.model(adv_inputs)
                loss = F.cross_entropy(outputs, labels, reduction='none').mean()

            if loss > best_loss:  # Maximize loss for untargeted attack
                best_loss = loss
                best_adv = adv_inputs

        return best_adv

    @property
    def alpha(self):
        """Auto-computed step size"""
        return self.eps / 4


def evaluate_adversarial_robustness(model: nn.Module,
                                   data_loader: torch.utils.data.DataLoader,
                                   attack: Union[PGDAttack, APGDAttack],
                                   device: str = "cuda") -> Dict[str, float]:
    """
    Evaluate model robustness under adversarial attack

    Args:
        model: Model to evaluate
        data_loader: Data loader
        attack: Adversarial attack
        device: Device to run on

    Returns:
        Dictionary with clean and robust accuracy
    """
    model.eval()
    clean_correct = 0
    adv_correct = 0
    total = 0

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        batch_size = inputs.size(0)

        # Clean accuracy
        with torch.no_grad():
            clean_outputs = model(inputs)
            _, clean_predicted = clean_outputs.max(1)
            clean_correct += clean_predicted.eq(labels).sum().item()

        # Adversarial accuracy
        adv_inputs = attack(inputs, labels)
        with torch.no_grad():
            adv_outputs = model(adv_inputs)
            _, adv_predicted = adv_outputs.max(1)
            adv_correct += adv_predicted.eq(labels).sum().item()

        total += batch_size

    return {
        'clean_acc': clean_correct / total,
        'robust_acc': adv_correct / total,
        'robustness_drop': (clean_correct - adv_correct) / total
    }


if __name__ == "__main__":
    # Test PGD attack
    print("PGDAttack and APGDAttack classes defined successfully")

    # Example usage:
    # model = load_your_model()
    # attack = PGDAttack(model, eps=8/255, alpha=2/255, steps=10)
    # adv_inputs = attack(inputs, labels)
    # robust_metrics = evaluate_adversarial_robustness(model, test_loader, attack)
