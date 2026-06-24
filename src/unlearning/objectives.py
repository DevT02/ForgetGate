"""
Unlearning objectives for ForgetGate-V
Implements various loss functions for machine unlearning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Re-exports kept for backward compatibility with scripts that import
# `_resolve_feature_model` / `_extract_features` from this module. The
# canonical definitions live in src.attacks.universal_patch (BFS through
# wrapped modules to find a `forward_features`-capable backbone), and we
# expose a thin wrapper here so the existing training and audit scripts
# (`scripts/train/2_train_unlearning_lora.py`,
# `scripts/audits/17_recovery_radius_audit.py`) can keep their import lines
# unchanged.
# ---------------------------------------------------------------------------
from src.attacks.universal_patch import _resolve_feature_model  # noqa: E402,F401


def _extract_features(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Return per-sample features from the first `forward_features`-capable
    submodule of `model`. CLS-token for 3-D outputs, mean-pool for 4-D, flatten
    otherwise. Used by feature-subspace PCA computation in the LoRA trainer.
    """
    feature_model = _resolve_feature_model(model)
    if feature_model is None:
        raise ValueError("Model does not expose forward_features anywhere.")
    feats = feature_model.forward_features(inputs)
    if isinstance(feats, (tuple, list)):
        feats = feats[0]
    if feats.ndim == 4:
        feats = feats.mean(dim=(2, 3))
    elif feats.ndim == 3:
        feats = feats[:, 0]
    elif feats.ndim > 2:
        feats = feats.flatten(start_dim=1)
    return feats


class UnlearningObjective(nn.Module):
    """Base class for unlearning objectives"""

    def __init__(self, forget_class: int, num_classes: int = 10):
        super().__init__()
        self.forget_class = forget_class
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute unlearning loss

        Args:
            logits: Model outputs [batch_size, num_classes]
            labels: Ground truth labels [batch_size]

        Returns:
            Loss value
        """
        raise NotImplementedError

    def get_target_distribution(self, batch_size: int) -> torch.Tensor:
        """Get target distribution for unlearning"""
        raise NotImplementedError


class CrossEntropyAscent(UnlearningObjective):
    """Cross-entropy ascent on forget class samples"""

    def __init__(self, forget_class: int, num_classes: int = 10,
                 retain_weight: float = 0.1, target_weight: float = 1.0):
        super().__init__(forget_class, num_classes)
        self.retain_weight = retain_weight
        self.target_weight = target_weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Maximize loss on forget class, minimize on retain classes
        """
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(logits, labels, reduction='none')

        # Modify loss based on class
        forget_mask = (labels == self.forget_class)
        retain_mask = ~forget_mask

        # For forget class: maximize loss (negative CE)
        # For retain classes: minimize loss (positive CE)
        modified_loss = torch.where(
            forget_mask,
            -self.target_weight * ce_loss,  # Weighted ascent on forget class
            self.retain_weight * ce_loss  # Weighted descent on retain classes
        )

        return modified_loss.mean()


# --- Restored adapted-method objectives (BalDRO/FaLW/RURK/SGA) ---
# Implementation-fidelity caveats are in the paper Adaptation appendix;
# these are "X-style" adaptations, not faithful reproductions.
class SmoothedGradientAscent(UnlearningObjective):
    """SGA (Pang et al. 2025): smooth GA by mixing forget loss with normal data losses."""

    def __init__(self, forget_class: int, num_classes: int = 10,
                 smoothing_rate: float = 0.5, num_normal_samples: int = 1,
                 retain_weight: float = 0.1, target_weight: float = 1.0):
        super().__init__(forget_class, num_classes)
        self.smoothing_rate = smoothing_rate
        self.num_normal_samples = max(int(num_normal_samples), 1)
        self.retain_weight = retain_weight
        self.target_weight = target_weight

    def compute_forget_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, labels, reduction='mean')
        return ce_loss

    def compute_normal_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, labels, reduction='mean')
        return ce_loss

    def combine_losses(self, forget_loss: torch.Tensor, normal_loss: torch.Tensor) -> torch.Tensor:
        # L_SGA = (1 - r + r/K) * L_f + (r/K) * sum_k L_p^k  (Paper Eq. 4)
        r = self.smoothing_rate
        k = self.num_normal_samples
        forget_term = (1.0 - r + r / k) * (-forget_loss)  # ascent on forget
        normal_term = (r / k) * normal_loss  # descent on normal
        return forget_term + normal_term

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Default to ascent loss only (trainer combines with normal loss).
        return -F.cross_entropy(logits, labels, reduction='mean')


class BalDROUnlearning(UnlearningObjective):
    """BalDRO (Shao et al. 2026): log-sum-exp reweighting over hard forget samples."""

    def __init__(self, forget_class: int, num_classes: int = 10,
                 retain_weight: float = 0.1, target_weight: float = 1.0,
                 beta: float = 1.0, top_fraction: float = 1.0):
        super().__init__(forget_class, num_classes)
        self.retain_weight = retain_weight
        self.target_weight = target_weight
        self.beta = max(beta, 1e-6)
        self.top_fraction = float(top_fraction)

    def _log_mean_exp(self, losses: torch.Tensor) -> torch.Tensor:
        # beta * log( (1/N) * sum exp(loss / beta) )
        scaled = losses / self.beta
        return self.beta * (torch.logsumexp(scaled, dim=0) - torch.log(torch.tensor(losses.numel(), device=losses.device)))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        forget_mask = (labels == self.forget_class)
        retain_mask = ~forget_mask

        total = 0.0
        parts = 0

        if forget_mask.any():
            forget_losses = ce_loss[forget_mask]
            if self.top_fraction < 1.0:
                k = max(1, int(forget_losses.numel() * self.top_fraction))
                forget_losses = torch.topk(forget_losses, k=k, largest=True).values
            forget_term = self._log_mean_exp(forget_losses)
            total += -self.target_weight * forget_term
            parts += 1

        if retain_mask.any():
            retain_term = ce_loss[retain_mask].mean()
            total += self.retain_weight * retain_term
            parts += 1

        if parts == 0:
            return torch.tensor(0.0, device=logits.device)

        return total / parts


class FaLWUnlearning(UnlearningObjective):
    """FaLW (Yu et al. 2026): forgetting-aware loss reweighting for long-tailed data."""

    def __init__(self, forget_class: int, num_classes: int = 10,
                 retain_weight: float = 0.1, target_weight: float = 1.0,
                 class_counts: Optional[Dict[int, int]] = None,
                 total_samples: Optional[int] = None,
                 tau: float = 1.0):
        super().__init__(forget_class, num_classes)
        self.retain_weight = retain_weight
        self.target_weight = target_weight
        self.class_counts = class_counts or {}
        self.total_samples = total_samples
        self.tau = tau
        self.class_mu = {}
        self.class_sigma = {}

    def set_class_stats(self, class_mu: Dict[int, float], class_sigma: Dict[int, float]):
        self.class_mu = class_mu
        self.class_sigma = class_sigma

    def _balance_factor(self, label: torch.Tensor) -> torch.Tensor:
        if not self.class_counts or self.total_samples is None:
            return torch.ones_like(label, dtype=torch.float32)
        counts = torch.tensor([self.class_counts.get(int(y), 1) for y in label],
                              device=label.device, dtype=torch.float32)
        num_classes = max(self.num_classes, 1)
        nf = float(sum(self.class_counts.values()))
        b = (nf / (num_classes * counts)).pow(self.tau)
        return b

    def _falw_weight(self, probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Paper Eq. 7: w_i = 1 + sign(z_i) * (tanh(|z_i|))^(1/B_i)
        # where B_i is the balance factor from Eq. 6
        mu = torch.tensor([self.class_mu.get(int(y), 0.0) for y in labels],
                          device=labels.device, dtype=torch.float32)
        sigma = torch.tensor([self.class_sigma.get(int(y), 1.0) for y in labels],
                             device=labels.device, dtype=torch.float32)
        z = (probs - mu) / (sigma + 1e-8)
        b = self._balance_factor(labels)  # B_i from Eq. 6
        w = 1.0 + torch.sign(z) * torch.tanh(torch.abs(z)).pow(1.0 / (b + 1e-8))
        return w

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        p_true = probs.gather(1, labels.view(-1, 1)).squeeze(1)
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        weights = self._falw_weight(p_true, labels)

        forget_mask = (labels == self.forget_class)
        retain_mask = ~forget_mask

        total = 0.0
        parts = 0

        if forget_mask.any():
            forget_loss = (ce_loss[forget_mask] * weights[forget_mask]).mean()
            total += -self.target_weight * forget_loss
            parts += 1

        if retain_mask.any():
            retain_loss = (ce_loss[retain_mask] * weights[retain_mask]).mean()
            total += self.retain_weight * retain_loss
            parts += 1

        if parts == 0:
            return torch.tensor(0.0, device=logits.device)

        return total / parts


class RURKUnlearning(UnlearningObjective):
    """Robust unlearning via perturbation loss on forget samples (RURK-style)."""

    def __init__(self, forget_class: int, num_classes: int = 10,
                 retain_weight: float = 0.1, target_weight: float = 1.0,
                 adv_eps: float = 8/255, adv_alpha: float = 2/255,
                 adv_steps: int = 3, adv_samples: int = 1,
                 adv_lambda: float = 1.0, random_start: bool = True):
        super().__init__(forget_class, num_classes)
        self.retain_weight = retain_weight
        self.target_weight = target_weight
        self.adv_eps = adv_eps
        self.adv_alpha = adv_alpha
        self.adv_steps = adv_steps
        self.adv_samples = max(int(adv_samples), 1)
        self.adv_lambda = adv_lambda
        self.random_start = random_start
        self.model = None

    def set_model(self, model: nn.Module):
        self.model = model

    def _pgd_minimize_loss(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """PGD to minimize CE loss (keep correct classification) within L_inf ball."""
        if self.adv_steps <= 0 or self.adv_eps <= 0:
            return inputs.detach()

        x = inputs.detach()
        best_adv = x.clone()
        best_loss = torch.full((x.size(0),), float("inf"), device=x.device)

        for _ in range(self.adv_samples):
            if self.random_start:
                adv = x + torch.empty_like(x).uniform_(-self.adv_eps, self.adv_eps)
                adv = torch.clamp(adv, 0.0, 1.0)
            else:
                adv = x.clone()

            for _ in range(self.adv_steps):
                adv = adv.detach().requires_grad_(True)
                logits = self.model(adv)
                loss = F.cross_entropy(logits, labels, reduction='sum')
                grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
                adv = adv - self.adv_alpha * torch.sign(grad)
                delta = torch.clamp(adv - x, min=-self.adv_eps, max=self.adv_eps)
                adv = torch.clamp(x + delta, 0.0, 1.0)

            with torch.no_grad():
                logits = self.model(adv)
                per_loss = F.cross_entropy(logits, labels, reduction='none')
                better = per_loss < best_loss
                if better.any():
                    best_loss = torch.where(better, per_loss, best_loss)
                    better_view = better.view(-1, *([1] * (adv.ndim - 1)))
                    best_adv = torch.where(better_view, adv, best_adv)

        return best_adv.detach()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.model is None:
            raise ValueError("RURKUnlearning requires model to be set via set_model().")
        if inputs is None:
            raise ValueError("RURKUnlearning requires inputs for perturbation loss.")

        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        forget_mask = (labels == self.forget_class)
        retain_mask = ~forget_mask

        total = 0.0
        parts = 0

        if forget_mask.any():
            forget_inputs = inputs[forget_mask]
            forget_labels = labels[forget_mask]
            if torch.is_grad_enabled():
                adv_inputs = self._pgd_minimize_loss(forget_inputs, forget_labels)
                adv_logits = self.model(adv_inputs)
                adv_loss = F.cross_entropy(adv_logits, forget_labels, reduction='mean')
            else:
                # Validation runs under no_grad; skip adversarial search.
                adv_loss = torch.tensor(0.0, device=logits.device)
            clean_loss = ce_loss[forget_mask].mean()
            forget_term = clean_loss + (self.adv_lambda * adv_loss)
            total += -self.target_weight * forget_term
            parts += 1

        if retain_mask.any():
            retain_term = ce_loss[retain_mask].mean()
            total += self.retain_weight * retain_term
            parts += 1

        if parts == 0:
            return torch.tensor(0.0, device=logits.device)

        return total / parts


class UniformKL(UnlearningObjective):
    """KL divergence to uniform distribution on forget class"""

    def __init__(self, forget_class: int, num_classes: int = 10,
                 temperature: float = 1.0, retain_weight: float = 0.1, target_weight: float = 1.0):
        super().__init__(forget_class, num_classes)
        self.temperature = temperature
        self.retain_weight = retain_weight
        self.target_weight = target_weight
        self.register_buffer("uniform_dist", torch.ones(num_classes) / num_classes)

    def get_target_distribution(self, batch_size: int, device: str = "cpu") -> torch.Tensor:
        """Uniform distribution for forget class"""
        return self.uniform_dist.to(device).unsqueeze(0).repeat(batch_size, 1)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        KL divergence to uniform on forget class, CE on retain classes
        """
        # Temperature scaling
        scaled_logits = logits / self.temperature

        # Get predicted probabilities
        probs = F.softmax(scaled_logits, dim=-1)

        # KL divergence to uniform for forget class samples
        forget_mask = (labels == self.forget_class)
        retain_mask = ~forget_mask

        # Target distribution (uniform for forget class, true label for retain)
        targets = torch.zeros_like(probs)

        # For forget class: uniform distribution (ensure same device)
        targets[forget_mask] = self.uniform_dist.to(targets.device)

        # For retain class: one-hot encoding
        targets[retain_mask] = F.one_hot(labels[retain_mask], num_classes=self.num_classes).float()

        # KL divergence loss
        kl_loss = F.kl_div(
            F.log_softmax(scaled_logits, dim=-1),
            targets,
            reduction='none'
        ).sum(dim=-1)

        # Weight the losses
        weighted_loss = torch.where(
            forget_mask,
            self.target_weight * kl_loss,  # Weighted loss for forget class
            self.retain_weight * kl_loss  # Weighted loss for retain classes
        )

        return weighted_loss.mean()


class FeatureScrub(UnlearningObjective):
    """Feature scrubbing objective (gradient ascent on features)"""

    def __init__(self, forget_class: int, num_classes: int = 10,
                 layer_name: str = "head", retain_weight: float = 0.1, target_weight: float = 1.0):
        super().__init__(forget_class, num_classes)
        self.layer_name = layer_name
        self.retain_weight = retain_weight
        self.target_weight = target_weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Gradient ascent on intermediate features for forget class

        Args:
            logits: Model outputs
            labels: Ground truth labels
            features: Intermediate features from target layer
        """
        if features is None:
            # Fallback to logits if no features provided
            return self._logits_based_loss(logits, labels)

        # L2 norm of features (to be maximized for forget class)
        feature_norm = torch.norm(features, p=2, dim=-1)

        forget_mask = (labels == self.forget_class)
        retain_mask = ~forget_mask

        # Maximize feature norm for forget class, minimize for retain classes
        feature_loss = torch.where(
            forget_mask,
            -self.target_weight * feature_norm,  # Weighted ascent (maximize norm)
            self.retain_weight * feature_norm  # Weighted descent (minimize norm)
        )

        return feature_loss.mean()

    def _logits_based_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Fallback loss based on logits when features not available"""
        # Similar to CE ascent but on logits magnitude
        logits_norm = torch.norm(logits, p=2, dim=-1)

        forget_mask = (labels == self.forget_class)
        retain_mask = ~forget_mask

        loss = torch.where(
            forget_mask,
            -self.target_weight * logits_norm,  # Weighted maximize logits norm for forget class
            self.retain_weight * logits_norm  # Weighted minimize for retain classes
        )

        return loss.mean()


class NegativeGradient(UnlearningObjective):
    """Negative gradient descent (amplified forgetting)"""

    def __init__(self, forget_class: int, num_classes: int = 10,
                 amplification: float = 2.0):
        super().__init__(forget_class, num_classes)
        self.amplification = amplification

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Amplify gradients in the wrong direction for forget class
        """
        # Standard cross-entropy
        ce_loss = F.cross_entropy(logits, labels, reduction='none')

        forget_mask = (labels == self.forget_class)

        # Amplify loss for forget class (negative gradients)
        amplified_loss = torch.where(
            forget_mask,
            self.amplification * ce_loss,
            ce_loss
        )

        return amplified_loss.mean()


class SalUn(UnlearningObjective):
    """
    Saliency-based Unlearning (Fan et al., ICLR 2024)
    Uses gradient-based saliency to identify important LoRA parameters,
    then applies random labeling to forget samples.
    """

    def __init__(self,
                 forget_class: int,
                 num_classes: int = 10,
                 saliency_threshold: float = 0.5,
                 random_label_weight: float = 1.0,
                 retain_weight: float = 1.0,
                 compute_saliency_every: int = 10):
        """
        Args:
            forget_class: Class to forget
            num_classes: Total number of classes
            saliency_threshold: Top k% of parameters by gradient magnitude (0-1)
            random_label_weight: Weight for random labeling loss on forget samples
            retain_weight: Weight for retain loss
            compute_saliency_every: Steps between saliency recomputation
        """
        super().__init__(forget_class, num_classes)
        self.saliency_threshold = saliency_threshold
        self.random_label_weight = random_label_weight
        self.retain_weight = retain_weight
        self.compute_saliency_every = compute_saliency_every

        # Storage for saliency masks (will be computed during training)
        self.saliency_mask = None
        self.step_counter = 0

    def compute_saliency(self, model: nn.Module, forget_loader) -> Dict[str, torch.Tensor]:
        """
        Compute gradient-based saliency for LoRA parameters.
        Called periodically during training by the trainer.

        Args:
            model: Model with LoRA parameters
            forget_loader: DataLoader for forget set

        Returns:
            Dictionary mapping parameter names to saliency masks
        """
        model.eval()
        saliency_scores = {}

        # Accumulate gradients over forget set
        for inputs, labels in forget_loader:
            inputs, labels = inputs.to(next(model.parameters()).device), labels.to(next(model.parameters()).device)

            # Forward + backward to get gradients
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()

            # Accumulate gradient magnitudes for LoRA parameters only
            for name, param in model.named_parameters():
                if param.requires_grad and ('lora' in name.lower() or 'adapter' in name.lower()):
                    if param.grad is not None:
                        if name not in saliency_scores:
                            saliency_scores[name] = torch.zeros_like(param.data)
                        saliency_scores[name] += param.grad.abs()

            model.zero_grad()

        # Normalize and create binary masks based on threshold
        saliency_masks = {}
        all_scores = torch.cat([scores.flatten() for scores in saliency_scores.values()])

        if len(all_scores) > 0:
            # Compute threshold value (top k% of parameters)
            threshold_value = torch.quantile(all_scores, 1.0 - self.saliency_threshold)

            # Create binary masks
            for name, scores in saliency_scores.items():
                saliency_masks[name] = (scores >= threshold_value).float()

        model.train()
        self.saliency_mask = saliency_masks
        return saliency_masks

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Loss computation:
        - Forget samples: Random labeling CE loss
        - Retain samples: Standard CE loss

        Args:
            logits: Model outputs [batch_size, num_classes]
            labels: Ground truth labels [batch_size]

        Returns:
            Combined loss value
        """
        forget_mask = (labels == self.forget_class)
        retain_mask = ~forget_mask

        total_loss = 0.0
        num_components = 0

        # Forget set: random labeling (excluding the forget class itself)
        if forget_mask.any():
            # Generate random labels excluding the forget class
            num_forget = forget_mask.sum().item()
            random_labels = torch.randint(0, self.num_classes - 1, (num_forget,), device=labels.device)
            # Shift labels >= forget_class to avoid sampling the forget class
            random_labels = torch.where(random_labels >= self.forget_class,
                                       random_labels + 1, random_labels)

            forget_loss = F.cross_entropy(logits[forget_mask], random_labels, reduction='mean')
            total_loss += self.random_label_weight * forget_loss
            num_components += 1

        # Retain set: standard cross-entropy
        if retain_mask.any():
            retain_loss = F.cross_entropy(logits[retain_mask], labels[retain_mask], reduction='mean')
            total_loss += self.retain_weight * retain_loss
            num_components += 1

        # Return average if we have components, otherwise zero
        return total_loss / num_components if num_components > 0 else torch.tensor(0.0, device=logits.device)


class SCRUB(UnlearningObjective):
    """
    SCRUB: Distillation-based unlearning (Kurmanji et al. 2024)
    Uses student-teacher distillation on retain set while
    maximizing loss on forget set.
    """

    def __init__(self,
                 forget_class: int,
                 num_classes: int = 10,
                 teacher_model: Optional[nn.Module] = None,
                 distill_weight: float = 1.0,
                 forget_weight: float = 1.0,
                 temperature: float = 2.0,
                 forget_mode: str = "ce_ascent"):
        """
        Args:
            forget_class: Class to forget
            num_classes: Total number of classes
            teacher_model: Teacher model (original base model before LoRA)
            distill_weight: Weight for retain distillation loss
            forget_weight: Weight for forget maximization loss
            temperature: Temperature for distillation
            forget_mode: How to handle forget samples ("ce_ascent" or "max_entropy")
        """
        super().__init__(forget_class, num_classes)
        self.teacher_model = teacher_model
        self.distill_weight = distill_weight
        self.forget_weight = forget_weight
        self.temperature = temperature
        self.forget_mode = forget_mode

        # Set teacher to eval mode and freeze if provided
        if self.teacher_model is not None:
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False

    def set_teacher_model(self, teacher_model: nn.Module):
        """
        Set or update the teacher model after initialization

        Args:
            teacher_model: Teacher model to use for distillation
        """
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Loss computation:
        - Retain samples: KL distillation to teacher
        - Forget samples: CE ascent or max entropy

        Args:
            logits: Student model outputs [batch_size, num_classes]
            labels: Ground truth labels [batch_size]
            inputs: Input images [batch_size, C, H, W] (needed for teacher forward pass)

        Returns:
            Combined loss value
        """
        if self.teacher_model is None:
            raise ValueError("Teacher model not set. Call set_teacher_model() first.")

        forget_mask = (labels == self.forget_class)
        retain_mask = ~forget_mask

        total_loss = 0.0
        num_components = 0

        # Retain set: distillation loss
        if retain_mask.any():
            if inputs is None:
                raise ValueError("inputs required for SCRUB distillation")

            # Get teacher predictions (no gradients)
            with torch.no_grad():
                teacher_logits = self.teacher_model(inputs[retain_mask])

            # Compute KL divergence with temperature scaling
            student_log_probs = F.log_softmax(logits[retain_mask] / self.temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

            distill_loss = F.kl_div(student_log_probs, teacher_probs,
                                   reduction='batchmean') * (self.temperature ** 2)
            total_loss += self.distill_weight * distill_loss
            num_components += 1

        # Forget set: maximize loss
        if forget_mask.any():
            if self.forget_mode == "ce_ascent":
                # Cross-entropy ascent: negative CE loss
                forget_loss = -F.cross_entropy(logits[forget_mask],
                                              labels[forget_mask], reduction='mean')
            elif self.forget_mode == "max_entropy":
                # Maximize entropy (push toward uniform distribution)
                probs = F.softmax(logits[forget_mask], dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
                forget_loss = -entropy  # Maximize entropy = minimize negative entropy
            else:
                raise ValueError(f"Unknown forget_mode: {self.forget_mode}")

            total_loss += self.forget_weight * forget_loss
            num_components += 1

        # Return average if we have components, otherwise zero
        return total_loss / num_components if num_components > 0 else torch.tensor(0.0, device=logits.device)


class OrbitUnlearning(UnlearningObjective):
    """ORBIT (oracle-aligned, representation-dispersive unlearning) -- reconstruction.

    Diagnostic stress-test objective from the paper, re-implemented from its
    four-term specification. The original objective code was never committed, so
    this is a fidelity-caveated *reconstruction* (in the spirit of the BalDRO/RURK
    recon rows); it will not reproduce the orphaned ORBIT result numbers, but it
    makes ORBIT runnable from committed code.

    All four terms act on the FORGET batch (the trainer supplies retain CE
    separately, weighted by retain_lambda):
      (1) hinge-margin suppression of the forget-class logit below the top
          non-forget logit by a margin gamma;
      (2) masked dark-knowledge distillation from a frozen retain-only ORACLE on
          the non-forget classes (the forget dimension is excluded from the KL);
      (3) feature alignment pulling forget-input representations toward the
          oracle's feature space (cosine);
      (4) a dispersion penalty pushing forget features apart (mean pairwise
          cosine) to weaken prototype / linear recovery channels.

    The oracle is supplied through set_teacher_model() (the trainer passes the
    frozen retain-only oracle); the student model is attached via set_model() for
    feature extraction.
    """

    def __init__(self,
                 forget_class: int,
                 num_classes: int = 10,
                 oracle_model: Optional[nn.Module] = None,
                 margin_weight: float = 1.0,
                 distill_weight: float = 1.0,
                 align_weight: float = 0.5,
                 disperse_weight: float = 0.5,
                 gamma: float = 1.0,
                 temperature: float = 2.0):
        super().__init__(forget_class, num_classes)
        self.oracle_model = oracle_model
        self.margin_weight = margin_weight
        self.distill_weight = distill_weight
        self.align_weight = align_weight
        self.disperse_weight = disperse_weight
        self.gamma = gamma
        self.temperature = temperature
        self.student_model = None
        if self.oracle_model is not None:
            self._freeze(self.oracle_model)

    @staticmethod
    def _freeze(m: nn.Module) -> None:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    def set_model(self, model: nn.Module) -> None:
        """Attach the student (LoRA) model, used for forget-feature extraction."""
        self.student_model = model

    def set_teacher_model(self, teacher_model: nn.Module) -> None:
        """Attach the frozen retain-only oracle (passed by the trainer)."""
        self.oracle_model = teacher_model
        self._freeze(self.oracle_model)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        t = self.forget_class

        # (1) hinge-margin suppression: drive the forget logit below the top
        # non-forget logit by at least gamma.
        forget_logit = logits[:, t]
        other = logits.clone()
        other[:, t] = float("-inf")
        top_other = other.max(dim=1).values
        margin = forget_logit - top_other  # > 0 means forget class still wins
        total = self.margin_weight * F.relu(margin + self.gamma).mean()

        # (2) masked dark-knowledge distillation from the oracle on non-forget
        # classes (exclude the forget dimension from the KL target).
        if self.oracle_model is not None and inputs is not None:
            with torch.no_grad():
                oracle_logits = self.oracle_model(inputs)
            keep = [j for j in range(self.num_classes) if j != t]
            s = logits[:, keep] / self.temperature
            o = oracle_logits[:, keep] / self.temperature
            distill = F.kl_div(F.log_softmax(s, dim=-1), F.softmax(o, dim=-1),
                               reduction="batchmean") * (self.temperature ** 2)
            total = total + self.distill_weight * distill

        # (3)+(4) feature alignment toward the oracle + dispersion of forget feats.
        if (self.student_model is not None and self.oracle_model is not None
                and inputs is not None):
            sf = F.normalize(_extract_features(self.student_model, inputs), dim=1)
            with torch.no_grad():
                of = F.normalize(_extract_features(self.oracle_model, inputs), dim=1)
            align = (1.0 - (sf * of).sum(dim=1)).mean()
            total = total + self.align_weight * align
            if sf.size(0) > 1:
                sim = sf @ sf.t()
                n = sim.size(0)
                disperse = (sim.sum() - sim.diagonal().sum()) / (n * (n - 1))
                total = total + self.disperse_weight * disperse

        return total


class SmoothedMarginUnlearning(UnlearningObjective):
    """
    Theorem 4 of theory_appendix.tex: a Gaussian-smoothed target-margin hinge
    penalty for forget examples, combined with standard CE on retain examples.

    For each forget input x with true label t (the forget class), the loss
    samples m Gaussian noise vectors delta_1..delta_m ~ N(0, sigma^2 I) and
    applies the hinge to the Monte-Carlo smoothed target margin

        max(0, mean_k m_t(x + delta_k; theta) + gamma)

    where m_t(x; theta) = f_t(x) - max_{j != t} f_j(x) is the target margin.
    The strict Cohen-Rosenfeld-Kolter certificate is for an unclipped Gaussian
    smoothed detector; clipping noisy training inputs to the image cube is an
    implementation choice and must be stated separately in certification audits.

    Retain side: classic CE (the smoothing on the forget set already protects
    retain accuracy because retain examples are not perturbed by this term).

    Forwards-compatible with the existing trainer: the smoothing is *inside*
    the loss (not in the data loader), so the trainer remains unchanged.

    Args:
        forget_class:    int, target class to forget.
        num_classes:     int, K.
        sigma:           float, noise std in the objective's input space
                         (pixel space when the model wrapper normalizes).
        n_noise:         int, number of noise samples per forward pass.
        gamma:           float, safety margin (>= 0). Theorem 4 uses gamma as
                         the offset that pushes the smoothed margin to <= -gamma.
        forget_weight:   weight on the smoothed hinge term.
        retain_weight:   weight on the CE retain term.
        clip_min/clip_max: optional input-range clip applied after noise.
                          For 224x224 ViT inputs in [0,1], pass (0.0, 1.0).
    """

    def __init__(
        self,
        forget_class: int,
        num_classes: int = 10,
        sigma: float = 0.10,
        n_noise: int = 4,
        gamma: float = 1.0,
        forget_weight: float = 1.0,
        retain_weight: float = 1.0,
        clip_min: Optional[float] = 0.0,
        clip_max: Optional[float] = 1.0,
    ):
        super().__init__(forget_class, num_classes)
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        if n_noise < 1:
            raise ValueError("n_noise must be >= 1")
        self.sigma = float(sigma)
        self.n_noise = int(n_noise)
        self.gamma = float(gamma)
        self.forget_weight = float(forget_weight)
        self.retain_weight = float(retain_weight)
        self.clip_min = clip_min
        self.clip_max = clip_max
        # We need the model in forward() to re-run on noisy inputs. The
        # trainer passes inputs= explicitly; we capture model via a setter.
        self._model: Optional[nn.Module] = None

    def set_model(self, model: nn.Module) -> None:
        """Attach the live model so forward() can run noisy forward passes."""
        self._model = model

    def _noisy_margin(self, inputs: torch.Tensor) -> torch.Tensor:
        """Returns mean-over-noise margin m_t(x + delta) for each x in inputs."""
        assert self._model is not None, \
            "SmoothedMarginUnlearning: call set_model(trainer.model) first."
        B = inputs.shape[0]
        # Expand and noise: [B*m, ...]
        x_rep = inputs.unsqueeze(1).expand(B, self.n_noise, *inputs.shape[1:])
        x_rep = x_rep.contiguous().view(B * self.n_noise, *inputs.shape[1:])
        noise = torch.randn_like(x_rep) * self.sigma
        x_noisy = x_rep + noise
        if self.clip_min is not None and self.clip_max is not None:
            x_noisy = x_noisy.clamp(self.clip_min, self.clip_max)
        logits = self._model(x_noisy)  # [B*m, K]
        target_logit = logits[:, self.forget_class]
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[:, self.forget_class] = False
        runnerup = logits.masked_fill(~mask, float("-inf")).max(dim=1).values
        margin = target_logit - runnerup  # [B*m]
        margin = margin.view(B, self.n_noise).mean(dim=1)  # [B]
        return margin

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs is None:
            raise ValueError(
                "SmoothedMarginUnlearning requires `inputs`; the trainer must "
                "pass inputs=forget_inputs when invoking the objective."
            )

        forget_mask = (labels == self.forget_class)
        retain_mask = ~forget_mask
        total_loss = torch.tensor(0.0, device=logits.device)
        n_components = 0

        # Forget side: smoothed hinge on the target margin
        if forget_mask.any():
            margin_smooth = self._noisy_margin(inputs[forget_mask])
            hinge = F.relu(margin_smooth + self.gamma)
            total_loss = total_loss + self.forget_weight * hinge.mean()
            n_components += 1

        # Retain side: standard CE on clean retain inputs
        if retain_mask.any():
            retain_loss = F.cross_entropy(
                logits[retain_mask], labels[retain_mask], reduction="mean"
            )
            total_loss = total_loss + self.retain_weight * retain_loss
            n_components += 1

        return total_loss / n_components if n_components > 0 else total_loss


def create_unlearning_objective(objective_name: str, forget_class: int,
                               num_classes: int = 10, **kwargs) -> UnlearningObjective:
    """
    Factory function to create unlearning objectives

    Args:
        objective_name: Name of objective ("ce_ascent", "uniform_kl", "feature_scrub",
                        "negative_grad", "salun", "scrub", "smoothed_margin")
        forget_class: Class to forget
        num_classes: Total number of classes
        **kwargs: Additional objective parameters (retain_weight, target_class_weight, etc.)
    """
    # Map YAML weight names to code parameter names for backward compatibility
    if 'target_class_weight' in kwargs:
        kwargs['target_weight'] = kwargs.pop('target_class_weight')
    if 'retain_class_weight' in kwargs:
        kwargs['retain_weight'] = kwargs.pop('retain_class_weight')
    # ce_ascent config includes a boolean flag that's implicit in the class
    kwargs.pop('ascent', None)

    if objective_name == "ce_ascent":
        return CrossEntropyAscent(forget_class, num_classes, **kwargs)

    elif objective_name == "uniform_kl":
        return UniformKL(forget_class, num_classes, **kwargs)

    elif objective_name == "feature_scrub":
        return FeatureScrub(forget_class, num_classes, **kwargs)

    elif objective_name == "negative_grad":
        return NegativeGradient(forget_class, num_classes, **kwargs)

    elif objective_name == "salun":
        return SalUn(forget_class, num_classes, **kwargs)

    elif objective_name == "scrub":
        return SCRUB(forget_class, num_classes, **kwargs)

    elif objective_name == "smoothed_margin":
        return SmoothedMarginUnlearning(forget_class, num_classes, **kwargs)

    elif objective_name == "orbit":
        return OrbitUnlearning(forget_class, num_classes, **kwargs)

    # Restored adapted-method objectives ("X-style"; see paper Adaptation appendix).
    elif objective_name == "sga":
        return SmoothedGradientAscent(forget_class, num_classes, **kwargs)

    elif objective_name == "bal_dro":
        return BalDROUnlearning(forget_class, num_classes, **kwargs)

    elif objective_name == "falw":
        return FaLWUnlearning(forget_class, num_classes, **kwargs)

    elif objective_name == "rurk":
        return RURKUnlearning(forget_class, num_classes, **kwargs)

    else:
        raise ValueError(f"Unknown unlearning objective: {objective_name}")


def get_forget_retain_loss(objective: UnlearningObjective,
                          model: nn.Module,
                          forget_batch: Tuple[torch.Tensor, torch.Tensor],
                          retain_batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                          retain_weight: float = 1.0) -> torch.Tensor:
    """
    Compute combined forget + retain loss

    Args:
        objective: Unlearning objective
        model: Model to evaluate
        forget_batch: (inputs, labels) for forget class
        retain_batch: (inputs, labels) for retain classes
        retain_weight: Weight for retain loss

    Returns:
        Combined loss
    """
    forget_inputs, forget_labels = forget_batch
    forget_outputs = model(forget_inputs)

    # Forget loss. Some objectives (SCRUB, smoothed margin) need the inputs for
    # teacher or noisy forwards; older objectives accept logits/labels only.
    try:
        forget_loss = objective(forget_outputs, forget_labels, inputs=forget_inputs)
    except TypeError:
        forget_loss = objective(forget_outputs, forget_labels)

    total_loss = forget_loss

    # Add retain loss if provided
    if retain_batch is not None:
        retain_inputs, retain_labels = retain_batch
        retain_outputs = model(retain_inputs)

        # Standard cross-entropy for retain classes
        retain_loss = F.cross_entropy(retain_outputs, retain_labels)
        total_loss = total_loss + retain_weight * retain_loss

    return total_loss


if __name__ == "__main__":
    # Test objectives
    batch_size = 4
    num_classes = 10
    forget_class = 0

    # Dummy data
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))

    # Test CE ascent
    ce_ascent = CrossEntropyAscent(forget_class, num_classes)
    loss = ce_ascent(logits, labels)
    print(f"CE Ascent loss: {loss.item():.4f}")

    # Test Uniform KL
    uniform_kl = UniformKL(forget_class, num_classes)
    loss = uniform_kl(logits, labels)
    print(f"Uniform KL loss: {loss.item():.4f}")

    # Test factory function
    objective = create_unlearning_objective("ce_ascent", forget_class, num_classes)
    loss = objective(logits, labels)
    print(f"Factory CE Ascent loss: {loss.item():.4f}")
