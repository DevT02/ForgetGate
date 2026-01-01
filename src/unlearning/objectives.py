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
