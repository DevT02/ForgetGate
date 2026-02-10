"""
Unlearning objectives for ForgetGate-V
Implements various loss functions for machine unlearning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


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


class RetainOnlyCE(UnlearningObjective):
    """Cross-entropy on retain classes only (forget class ignored)."""

    def __init__(self, forget_class: int, num_classes: int = 10):
        super().__init__(forget_class, num_classes)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        forget_mask = (labels == self.forget_class)
        retain_mask = ~forget_mask
        retain_loss = torch.where(retain_mask, ce_loss, torch.zeros_like(ce_loss))
        # Avoid divide-by-zero if batch is all forget-class.
        denom = retain_mask.sum().clamp_min(1)
        return retain_loss.sum() / denom


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
    Saliency-based Unlearning (Chundawat et al. 2023)
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

            # Accumulate gradient magnitudes for all trainable parameters
            for name, param in model.named_parameters():
                if param.requires_grad:
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
                 retain_ce_weight: float = 1.0,
                 extra_retain_steps: int = 0,
                 temperature: float = 2.0,
                 forget_mode: str = "ce_ascent"):
        """
        Args:
            forget_class: Class to forget
            num_classes: Total number of classes
            teacher_model: Teacher model (original base model before LoRA)
            distill_weight: Weight for retain distillation loss (alpha)
            forget_weight: Weight for forget maximization loss (beta)
            retain_ce_weight: Weight for retain CE loss (gamma)
            temperature: Temperature for distillation
            forget_mode: How to handle forget samples ("ce_ascent" or "max_entropy")
        """
        super().__init__(forget_class, num_classes)
        self.teacher_model = teacher_model
        self.distill_weight = distill_weight
        self.forget_weight = forget_weight
        self.retain_ce_weight = retain_ce_weight
        self.extra_retain_steps = max(int(extra_retain_steps), 0)
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

    def _distill_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)

    def compute_forget_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        # SCRUB: maximize KL on forget samples (negative distill loss)
        return -self._distill_loss(student_logits, teacher_logits)

    def compute_retain_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                            labels: torch.Tensor) -> torch.Tensor:
        distill = self._distill_loss(student_logits, teacher_logits)
        ce = F.cross_entropy(student_logits, labels, reduction='mean')
        return (self.distill_weight * distill) + (self.retain_ce_weight * ce)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Combined SCRUB loss (used for validation):
        alpha * KL(retain) + gamma * CE(retain) - KL(forget)
        """
        if self.teacher_model is None:
            raise ValueError("Teacher model not set. Call set_teacher_model() first.")
        if inputs is None:
            raise ValueError("inputs required for SCRUB distillation")

        forget_mask = (labels == self.forget_class)
        retain_mask = ~forget_mask

        total_loss = 0.0
        num_components = 0

        with torch.no_grad():
            teacher_logits = self.teacher_model(inputs)

        if retain_mask.any():
            retain_loss = self.compute_retain_loss(
                logits[retain_mask], teacher_logits[retain_mask], labels[retain_mask]
            )
            total_loss += retain_loss
            num_components += 1

        if forget_mask.any():
            forget_loss = self.compute_forget_loss(
                logits[forget_mask], teacher_logits[forget_mask], labels[forget_mask]
            )
            total_loss += self.forget_weight * forget_loss
            num_components += 1

        return total_loss / num_components if num_components > 0 else torch.tensor(0.0, device=logits.device)


class BalDRO_SCRUB(BalDROUnlearning):
    """
    BalDRO + SCRUB Hybrid:
    Combines BalDRO's hard-sample mining with SCRUB's feature ascent.
    """

    def __init__(self,
                 forget_class: int,
                 num_classes: int = 10,
                 teacher_model: Optional[nn.Module] = None,
                 retain_weight: float = 0.1,
                 target_weight: float = 1.0,
                 beta: float = 1.0,
                 top_fraction: float = 1.0,
                 distill_weight: float = 1.0,  # Alpha (retain distillation)
                 forget_weight: float = 1.0,   # Beta (forget maximization)
                 retain_ce_weight: float = 1.0, # Gamma (retain CE)
                 temperature: float = 2.0):
        super().__init__(forget_class, num_classes, retain_weight, target_weight, beta, top_fraction)
        self.teacher_model = teacher_model
        self.distill_weight = distill_weight
        self.forget_weight = forget_weight
        self.retain_ce_weight = retain_ce_weight
        self.temperature = temperature

        # Set teacher to eval mode and freeze if provided
        if self.teacher_model is not None:
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False

    def set_teacher_model(self, teacher_model: nn.Module):
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def _distill_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        # Re-implement distill loss (or mixin, but copy is safe)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)

    def compute_forget_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute forget loss: BalDRO Term + SCRUB Term
        We want to MAXIMIZE both.
        Trainer minimizes (forget_weight * forget_loss).
        So return -(BalDRO_Term + KL_Term).
        """
        # BalDRO Term: Maximized CE on hard samples
        # Assuming student_logits are for forget class samples ONLY
        ce_loss = F.cross_entropy(student_logits, labels, reduction='none') if labels is not None else torch.tensor(0.0)
        
        # If labels are None (shouldn't happen with updated trainer), we can't compute BalDRO properly.
        # But for 'forget_batch', labels are all 'forget_class'.
        # Assuming labels are provided.

        if self.top_fraction < 1.0:
            k = max(1, int(ce_loss.numel() * self.top_fraction))
            forget_losses = torch.topk(ce_loss, k=k, largest=True).values
        else:
            forget_losses = ce_loss

        baldro_term = self._log_mean_exp(forget_losses) # This is positive CE-like term

        # SCRUB Term: Maximize KL(student || teacher)
        scrub_term = self._distill_loss(student_logits, teacher_logits) # This is positive KL

        # We return negative sum because trainer minimizes result
        # Note: SCRUB implementation returns -KL.
        # So we return -(BalDRO + KL).
        return -(self.target_weight * baldro_term + self.forget_weight * scrub_term)

    def compute_retain_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                            labels: torch.Tensor) -> torch.Tensor:
        """
        Compute retain loss: CE + KL(student || teacher)
        Minimize both.
        """
        distill = self._distill_loss(student_logits, teacher_logits)
        ce = F.cross_entropy(student_logits, labels, reduction='mean')
        return (self.distill_weight * distill) + (self.retain_ce_weight * ce)



def create_unlearning_objective(objective_name: str, forget_class: int,
                               num_classes: int = 10, **kwargs) -> UnlearningObjective:
    """
    Factory function to create unlearning objectives

    Args:
        objective_name: Name of objective ("ce_ascent", "uniform_kl", "feature_scrub", "negative_grad")
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

    elif objective_name in ("sga", "smoothed_ga"):
        return SmoothedGradientAscent(forget_class, num_classes, **kwargs)

    elif objective_name in ("bal_dro", "baldro"):
        return BalDROUnlearning(forget_class, num_classes, **kwargs)

    elif objective_name in ("falw", "fa_lw"):
        return FaLWUnlearning(forget_class, num_classes, **kwargs)

    elif objective_name in ("rurk", "robust_unlearn"):
        return RURKUnlearning(forget_class, num_classes, **kwargs)

    elif objective_name == "uniform_kl":
        return UniformKL(forget_class, num_classes, **kwargs)

    elif objective_name in ("retain_only", "noisy_retain"):
        return RetainOnlyCE(forget_class, num_classes)

    elif objective_name == "feature_scrub":
        return FeatureScrub(forget_class, num_classes, **kwargs)

    elif objective_name == "negative_grad":
        return NegativeGradient(forget_class, num_classes, **kwargs)

    elif objective_name == "salun":
        return SalUn(forget_class, num_classes, **kwargs)

    elif objective_name == "scrub":
        return SCRUB(forget_class, num_classes, **kwargs)

    elif objective_name == "baldro_scrub":
        return BalDRO_SCRUB(forget_class, num_classes, **kwargs)

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

    # Forget loss
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


class DistillationDefense(UnlearningObjective):
    """
    Distillation-based defense wrapper for unlearning objectives.
    
    Adds teacher-student distillation loss to ANY unlearning objective,
    forcing feature alignment to prevent probe-based leakage attacks.
    
    Based on analysis of why SCRUB (Kurmanji 2023) is probe-resistant:
    SCRUB forces student features to match a frozen teacher, which
    implicitly scrubs class-specific information from representations.
    """
    
    def __init__(self, 
                 base_objective: UnlearningObjective,
                 distill_weight: float = 1.0,
                 temperature: float = 4.0,
                 apply_to_forget: bool = True,
                 apply_to_retain: bool = True):
        """
        Args:
            base_objective: The underlying unlearning objective to wrap
            distill_weight: Weight for distillation loss (λ in L = L_base + λ*L_distill)
            temperature: KL distillation temperature
            apply_to_forget: Whether to apply distillation on forget samples
            apply_to_retain: Whether to apply distillation on retain samples
        """
        super().__init__(base_objective.forget_class, base_objective.num_classes)
        self.base_objective = base_objective
        self.distill_weight = distill_weight
        self.temperature = temperature
        self.apply_to_forget = apply_to_forget
        self.apply_to_retain = apply_to_retain
        self.teacher_model = None
    
    def set_teacher(self, teacher_model: nn.Module):
        """Set the frozen teacher model for distillation."""
        self.teacher_model = teacher_model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
    
    def _kl_distill_loss(self, 
                         student_logits: torch.Tensor, 
                         teacher_logits: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence distillation loss."""
        T = self.temperature
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        # KL(teacher || student) = sum teacher * (log teacher - log student)
        kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T * T)
        return kl_loss
    
    def forward(self, 
                logits: torch.Tensor, 
                labels: torch.Tensor,
                inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute defended unlearning loss = base_loss + distill_weight * distill_loss
        
        Args:
            logits: Student model outputs
            labels: Ground truth labels
            inputs: Original inputs (needed for teacher forward pass)
        """
        # Compute base unlearning loss
        base_loss = self.base_objective(logits, labels)
        
        # If no teacher or no inputs, return base loss only
        if self.teacher_model is None or inputs is None:
            return base_loss
        
        # Compute distillation loss
        with torch.no_grad():
            teacher_logits = self.teacher_model(inputs)
        
        # Separate forget and retain samples
        forget_mask = (labels == self.forget_class)
        retain_mask = ~forget_mask
        
        distill_loss = torch.tensor(0.0, device=logits.device)
        count = 0
        
        # Apply distillation to forget samples (align with teacher)
        if self.apply_to_forget and forget_mask.any():
            forget_distill = self._kl_distill_loss(
                logits[forget_mask], teacher_logits[forget_mask]
            )
            distill_loss = distill_loss + forget_distill
            count += 1
        
        # Apply distillation to retain samples (preserve teacher behavior)
        if self.apply_to_retain and retain_mask.any():
            retain_distill = self._kl_distill_loss(
                logits[retain_mask], teacher_logits[retain_mask]
            )
            distill_loss = distill_loss + retain_distill
            count += 1
        
        if count > 0:
            distill_loss = distill_loss / count
        
        return base_loss + self.distill_weight * distill_loss


def wrap_with_distillation_defense(objective: UnlearningObjective,
                                    distill_weight: float = 1.0,
                                    temperature: float = 4.0) -> DistillationDefense:
    """
    Factory function to wrap any unlearning objective with distillation defense.
    
    Usage:
        base = BalDROUnlearning(forget_class=0)
        defended = wrap_with_distillation_defense(base, distill_weight=1.0)
        defended.set_teacher(teacher_model)
    """
    return DistillationDefense(
        base_objective=objective,
        distill_weight=distill_weight,
        temperature=temperature
    )


class FeatureOrthogonalizationDefense(UnlearningObjective):
    """
    Feature-level defense: project forget-class features orthogonal to class centroid.
    
    Forces forget-class features to be dissimilar from their original direction,
    making probe attacks harder.
    """
    
    def __init__(self, 
                 base_objective: UnlearningObjective,
                 ortho_weight: float = 1.0,
                 centroid: Optional[torch.Tensor] = None):
        super().__init__(base_objective.forget_class, base_objective.num_classes)
        self.base_objective = base_objective
        self.ortho_weight = ortho_weight
        self.register_buffer('centroid', centroid)
        self.feature_extractor = None
    
    def set_centroid(self, centroid: torch.Tensor):
        """Set the forget-class feature centroid to orthogonalize against."""
        self.centroid = centroid.detach()
    
    def set_feature_extractor(self, model: nn.Module):
        """Set model for feature extraction."""
        self.feature_extractor = model
    
    def forward(self, 
                logits: torch.Tensor, 
                labels: torch.Tensor,
                features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        L = L_base + ortho_weight * L_ortho
        
        L_ortho = mean(cos_sim(forget_features, centroid)^2)
        """
        base_loss = self.base_objective(logits, labels)
        
        if self.centroid is None or features is None:
            return base_loss
        
        forget_mask = (labels == self.forget_class)
        if not forget_mask.any():
            return base_loss
        
        forget_features = features[forget_mask]
        centroid = self.centroid.to(features.device)
        
        # Cosine similarity - we want this to be 0 (orthogonal)
        cos_sim = F.cosine_similarity(
            forget_features, 
            centroid.unsqueeze(0).expand(forget_features.size(0), -1)
        )
        ortho_loss = (cos_sim ** 2).mean()
        
        return base_loss + self.ortho_weight * ortho_loss


class AdversarialFeatureMasking(UnlearningObjective):
    """
    Adversarial feature masking defense.
    
    Identifies top-k feature dimensions correlated with forget class
    and adds noise/masks them during training to prevent leakage.
    """
    
    def __init__(self,
                 base_objective: UnlearningObjective,
                 mask_weight: float = 1.0,
                 top_k: int = 32,
                 noise_std: float = 1.0):
        super().__init__(base_objective.forget_class, base_objective.num_classes)
        self.base_objective = base_objective
        self.mask_weight = mask_weight
        self.top_k = top_k
        self.noise_std = noise_std
        self.register_buffer('forget_feature_directions', None)
    
    def compute_forget_directions(self, 
                                   forget_features: torch.Tensor,
                                   retain_features: torch.Tensor):
        """Compute feature directions most discriminative for forget class."""
        forget_mean = forget_features.mean(dim=0)
        retain_mean = retain_features.mean(dim=0)
        direction = forget_mean - retain_mean
        direction = direction / (direction.norm() + 1e-8)
        
        # Find top-k dimensions with largest difference
        diff = (forget_mean - retain_mean).abs()
        _, top_indices = torch.topk(diff, self.top_k)
        
        self.forget_feature_directions = top_indices
    
    def forward(self,
                logits: torch.Tensor,
                labels: torch.Tensor,
                features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        L = L_base + mask_weight * L_mask
        
        L_mask encourages forget-class features to have noise in discriminative dims.
        """
        base_loss = self.base_objective(logits, labels)
        
        if self.forget_feature_directions is None or features is None:
            return base_loss
        
        forget_mask = (labels == self.forget_class)
        if not forget_mask.any():
            return base_loss
        
        forget_features = features[forget_mask]
        mask_dims = self.forget_feature_directions.to(features.device)
        
        # Encourage variance in discriminative dimensions (make them noisy)
        masked_feats = forget_features[:, mask_dims]
        # We want high variance = low signal -> use negative variance as loss
        variance = masked_feats.var(dim=0).mean()
        mask_loss = -variance  # Minimize this = maximize variance
        
        return base_loss + self.mask_weight * mask_loss


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
