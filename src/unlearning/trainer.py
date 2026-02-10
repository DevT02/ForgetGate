"""
Training loop for LoRA-based unlearning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Tuple, Callable
import os
import json
from tqdm import tqdm
import numpy as np
from datetime import datetime

from .objectives import create_unlearning_objective, UnlearningObjective
from ..models.peft_lora import save_lora_adapter, print_trainable_parameters


class UnlearningTrainer:
    """Trainer for LoRA-based machine unlearning"""

    def __init__(self,
                 model: nn.Module,
                 objective: UnlearningObjective,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 normal_loader: Optional[DataLoader] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 retain_lambda: float = 1.0,
                 robust_retain: bool = False,
                 robust_retain_eps: float = 8/255,
                 robust_retain_alpha: float = 2/255,
                 robust_retain_steps: int = 5,
                 grad_noise_std: float = 0.0,
                 gu_projection: bool = False,
                 grad_surgery: bool = False,
                 projection_eps: float = 1e-12,
                 orthogonal_reg: float = 0.0,
                 gu_beta: float = 1.0,
                 scrub_rewind: bool = False,
                 scrub_rewind_on_train: bool = True,
                 scrub_full_schedule: bool = False,
                 scrub_max_steps: Optional[int] = None):
        """
        Args:
            model: Model with LoRA applied
            objective: Unlearning objective
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            retain_lambda: Weight for retain loss (objective handles forget loss internally)
            robust_retain: Whether to use adversarial examples for retain loss
            robust_retain_eps: Epsilon for robust retain (default: 8/255)
            robust_retain_alpha: Step size for robust retain (default: 2/255)
            robust_retain_steps: Number of PGD steps for robust retain (default: 5)
            gu_projection: Project forget gradients to be orthogonal to retain gradients
            grad_surgery: Apply conflict-aware projection (PCGrad-style)
        """
        self.model = model.to(device)
        self.objective = objective
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.normal_loader = normal_loader
        self.device = device
        self.retain_lambda = retain_lambda

        # Robust retain preservation config
        self.robust_retain = robust_retain
        self.robust_retain_eps = robust_retain_eps
        self.robust_retain_alpha = robust_retain_alpha
        self.robust_retain_steps = robust_retain_steps
        self.grad_noise_std = grad_noise_std
        self.gu_projection = gu_projection
        self.grad_surgery = grad_surgery
        self.projection_eps = projection_eps
        self.orthogonal_reg = orthogonal_reg
        self.gu_beta = gu_beta
        self.scrub_rewind = scrub_rewind
        self.scrub_rewind_on_train = scrub_rewind_on_train
        self.scrub_full_schedule = scrub_full_schedule
        self.scrub_max_steps = scrub_max_steps
        if self.gu_projection and self.grad_surgery:
            raise ValueError("gu_projection and grad_surgery are mutually exclusive")
        if (self.gu_projection or self.grad_surgery) and self.orthogonal_reg > 0:
            raise ValueError("orthogonal_reg is not compatible with gu_projection/grad_surgery")

        # Create PGD attack for robust retain if enabled
        if self.robust_retain:
            from ..attacks.pgd import PGDAttack
            self.robust_retain_attack = PGDAttack(
                model=self.model,
                eps=robust_retain_eps,
                alpha=robust_retain_alpha,
                steps=robust_retain_steps,
                random_start=True,
                norm='l_inf'
            )

        # Training state
        self.current_epoch = 0
        self.best_val_metric = float('inf')
        self.training_history = []

    def _trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def _compute_grads(self, loss: torch.Tensor, params: List[nn.Parameter]) -> List[Optional[torch.Tensor]]:
        grads = torch.autograd.grad(
            loss, params, retain_graph=True, allow_unused=True
        )
        return list(grads)

    def _get_h_diag(self, param: nn.Parameter) -> Optional[torch.Tensor]:
        if not hasattr(self, "optimizer"):
            return None
        state = self.optimizer.state.get(param, {})
        exp_avg_sq = state.get("exp_avg_sq")
        if exp_avg_sq is None:
            return None
        eps = getattr(self.optimizer, "eps", 1e-8)
        if isinstance(self.optimizer, torch.optim.AdamW) or isinstance(self.optimizer, torch.optim.Adam):
            return 1.0 / (torch.sqrt(exp_avg_sq) + eps)
        return None

    def _project_forget_grads(self,
                              forget_grads: List[Optional[torch.Tensor]],
                              retain_grads: List[Optional[torch.Tensor]],
                              params: List[nn.Parameter],
                              mode: str,
                              scale_override: Optional[float] = None) -> List[Optional[torch.Tensor]]:
        dot = 0.0
        denom = 0.0
        for g_f, g_r, param in zip(forget_grads, retain_grads, params):
            if g_f is None or g_r is None:
                continue
            h = self._get_h_diag(param)
            if h is None:
                dot += torch.sum(g_f * g_r)
                denom += torch.sum(g_r * g_r)
            else:
                dot += torch.sum(g_f * h * g_r)
                denom += torch.sum(g_r * h * g_r)

        denom_val = float(denom)
        if denom_val <= self.projection_eps:
            return forget_grads

        if mode == "grad_surgery":
            if float(dot) >= 0:
                return forget_grads

        scale = dot / (denom + self.projection_eps)
        if scale_override is not None:
            scale = scale * scale_override
        projected = []
        for g_f, g_r in zip(forget_grads, retain_grads):
            if g_f is None:
                projected.append(None)
                continue
            if g_r is None:
                projected.append(g_f)
                continue
            projected.append(g_f - scale * g_r)
        return projected

    def _compute_unseen_stats(self) -> Optional[Dict[int, Tuple[float, float]]]:
        if self.val_loader is None:
            return None
        self.model.eval()
        per_class = {}
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logits = self.model(inputs)
                probs = torch.softmax(logits, dim=-1)
                p_true = probs.gather(1, labels.view(-1, 1)).squeeze(1)
                for p, y in zip(p_true, labels):
                    y_int = int(y)
                    if y_int not in per_class:
                        per_class[y_int] = []
                    per_class[y_int].append(float(p))
        stats = {}
        for k, vals in per_class.items():
            if len(vals) == 0:
                continue
            arr = torch.tensor(vals, dtype=torch.float32)
            stats[k] = (float(arr.mean()), float(arr.std(unbiased=False) + 1e-8))
        return stats

    def set_teacher_model(self, teacher_model: nn.Module):
        """
        Set teacher model for distillation-based objectives (e.g., SCRUB)

        Args:
            teacher_model: Teacher model to use for distillation
        """
        if hasattr(self.objective, 'set_teacher_model'):
            self.objective.set_teacher_model(teacher_model)
        else:
            print(f"Warning: Objective {type(self.objective).__name__} does not support teacher models")

    def configure_optimizer(self,
                           lr: float = 1e-3,
                           weight_decay: float = 0.01,
                           optimizer_name: str = "adamw") -> optim.Optimizer:
        """Configure optimizer"""

        if optimizer_name.lower() == "adamw":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name.lower() == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name.lower() == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        self.optimizer = optimizer
        return optimizer

    def configure_scheduler(self,
                           scheduler_name: str = "cosine",
                           max_epochs: Optional[int] = None,
                           warmup_steps: int = 100,
                           min_lr: float = 1e-6) -> optim.lr_scheduler._LRScheduler:
        """Configure learning rate scheduler"""

        # Set max_epochs if provided, otherwise use existing value
        if max_epochs is not None:
            self.max_epochs = max_epochs

        if scheduler_name.lower() == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.max_epochs,
                eta_min=min_lr
            )
        elif scheduler_name.lower() == "step":
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif scheduler_name.lower() == "linear":
            scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=min_lr/self.optimizer.param_groups[0]['lr'],
                total_iters=self.max_epochs
            )
        else:
            # No scheduler
            scheduler = None

        # Add warmup if specified
        if warmup_steps > 0 and scheduler is not None:
            scheduler = optim.lr_scheduler.SequentialLR(
                self.optimizer,
                [optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_steps),
                 scheduler],
                [warmup_steps]
            )

        self.scheduler = scheduler
        return scheduler

    def _maybe_update_saliency_mask(self):
        """Compute/update SalUn saliency masks if configured."""
        if not hasattr(self.objective, "compute_saliency"):
            return
        if not hasattr(self.train_loader, "forget_loader"):
            return

        step_counter = getattr(self.objective, "step_counter", 0)
        compute_every = getattr(self.objective, "compute_saliency_every", 1)
        if self.objective.saliency_mask is None or (step_counter % compute_every == 0):
            self.objective.compute_saliency(self.model, self.train_loader.forget_loader)

        if hasattr(self.objective, "step_counter"):
            self.objective.step_counter += 1

    def _apply_saliency_mask(self):
        """Mask gradients using SalUn saliency masks."""
        saliency_mask = getattr(self.objective, "saliency_mask", None)
        if not saliency_mask:
            return

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            mask = saliency_mask.get(name)
            if mask is None:
                continue
            param.grad.mul_(mask.to(device=param.grad.device, dtype=param.grad.dtype))

    def _apply_grad_noise(self):
        """Optional Gaussian gradient noise for certified-style baselines."""
        if not self.grad_noise_std or self.grad_noise_std <= 0:
            return
        for param in self.model.parameters():
            if param.grad is None:
                continue
            noise = torch.randn_like(param.grad) * self.grad_noise_std
            param.grad.add_(noise)

    def _evaluate_accuracy(self, loader: DataLoader) -> float:
        """Compute accuracy on a provided loader."""
        if loader is None:
            return 0.0
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        self.model.train()
        return (correct / total) if total > 0 else 0.0

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        # SalUn: update saliency masks periodically using forget_loader
        self._maybe_update_saliency_mask()

        normal_iter = None
        if self.normal_loader is not None:
            normal_iter = iter(self.normal_loader)
        elif hasattr(self.train_loader, "retain_loader"):
            normal_iter = iter(self.train_loader.retain_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Handle combined loader (forget_batch, retain_batch) or single batch
            if isinstance(batch, tuple) and len(batch) == 2:
                forget_part, retain_part = batch

                retain_inputs, retain_labels = retain_part
                retain_inputs, retain_labels = retain_inputs.to(self.device), retain_labels.to(self.device)

                # Compute retain loss (on adversarial examples if robust_retain is enabled)
                if self.robust_retain:
                    # Generate adversarial examples for retain set
                    adv_retain_inputs = self.robust_retain_attack(retain_inputs, retain_labels)
                    retain_outputs = self.model(adv_retain_inputs)
                else:
                    retain_outputs = self.model(retain_inputs)

                retain_loss_raw = nn.functional.cross_entropy(retain_outputs, retain_labels)
                retain_loss = retain_loss_raw * self.retain_lambda

                if forget_part is not None:
                    forget_inputs, forget_labels = forget_part
                    forget_inputs, forget_labels = forget_inputs.to(self.device), forget_labels.to(self.device)

                    # Forward pass on forget batch - use full objective (handles forget ascent + retain descent with retain_weight)
                    forget_outputs = self.model(forget_inputs)

                    # SCRUB: alternating max (forget) and min (retain) steps
                    if hasattr(self.objective, "compute_forget_loss") and hasattr(self.objective, "compute_retain_loss"):
                        obj_name = self.objective.__class__.__name__.lower()
                        if "scrub" in obj_name and "feature" not in obj_name:
                            if getattr(self.objective, "teacher_model", None) is None:
                                raise ValueError("Distillation-based SCRUB requires teacher_model to be set.")
                            # Forget (maximize KL): gradient ascent = minimize negative KL
                            self.optimizer.zero_grad()
                            with torch.no_grad():
                                teacher_forget = self.objective.teacher_model(forget_inputs)
                            forget_loss = self.objective.compute_forget_loss(forget_outputs, teacher_forget, labels=forget_labels)
                            # compute_forget_loss already returns -KL; minimize directly
                            (self.objective.forget_weight * forget_loss).backward()
                            self._apply_saliency_mask()
                            self._apply_grad_noise()
                            self.optimizer.step()

                            # Retain (minimize KL + CE)
                            self.optimizer.zero_grad()
                            if self.robust_retain:
                                retain_for_inputs = self.robust_retain_attack(retain_inputs, retain_labels)
                            else:
                                retain_for_inputs = retain_inputs
                            retain_outputs = self.model(retain_for_inputs)
                            with torch.no_grad():
                                teacher_retain = self.objective.teacher_model(retain_for_inputs)
                            retain_loss = self.objective.compute_retain_loss(
                                retain_outputs, teacher_retain, retain_labels
                            )
                            retain_loss = retain_loss * self.retain_lambda
                            retain_loss.backward()
                            self._apply_saliency_mask()
                            self._apply_grad_noise()
                            self.optimizer.step()

                            # Optional extra retain steps (SCRUB schedule)
                            extra_steps = getattr(self.objective, "extra_retain_steps", 0)
                            for _ in range(extra_steps):
                                self.optimizer.zero_grad()
                                if self.robust_retain:
                                    retain_for_inputs = self.robust_retain_attack(retain_inputs, retain_labels)
                                else:
                                    retain_for_inputs = retain_inputs
                                retain_outputs = self.model(retain_for_inputs)
                                with torch.no_grad():
                                    teacher_retain = self.objective.teacher_model(retain_for_inputs)
                                retain_loss = self.objective.compute_retain_loss(
                                    retain_outputs, teacher_retain, retain_labels
                                )
                                retain_loss = retain_loss * self.retain_lambda
                                retain_loss.backward()
                                self._apply_saliency_mask()
                                self._apply_grad_noise()
                                self.optimizer.step()

                            # Logging loss (approx)
                            loss = (retain_loss + (self.objective.forget_weight * forget_loss)).detach()
                            epoch_loss += loss.item()
                            num_batches += 1
                            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                            continue

                    # Check if objective accepts inputs parameter (for SCRUB distillation)
                    try:
                        forget_loss = self.objective(forget_outputs, forget_labels, inputs=forget_inputs)
                    except TypeError:
                        # Objective doesn't accept inputs, call without it
                        forget_loss = self.objective(forget_outputs, forget_labels)

                    if hasattr(self.objective, "combine_losses"):
                        if hasattr(self.objective, "compute_forget_loss"):
                            forget_loss_raw = self.objective.compute_forget_loss(forget_outputs, forget_labels)
                        else:
                            forget_loss_raw = nn.functional.cross_entropy(forget_outputs, forget_labels)
                        if hasattr(self.objective, "compute_normal_loss"):
                            normal_loss_raw = self.objective.compute_normal_loss(retain_outputs, retain_labels)
                        else:
                            # If SGA requests multiple "normal" batches, average across them.
                            k = int(getattr(self.objective, "num_normal_samples", 1))
                            normal_loss_raw = retain_loss_raw
                            if k > 1 and normal_iter is not None:
                                extra_losses = [retain_loss_raw]
                                for _ in range(k - 1):
                                    try:
                                        nx, ny = next(normal_iter)
                                    except StopIteration:
                                        if self.normal_loader is not None:
                                            normal_iter = iter(self.normal_loader)
                                        elif hasattr(self.train_loader, "retain_loader"):
                                            normal_iter = iter(self.train_loader.retain_loader)
                                        nx, ny = next(normal_iter)
                                    nx, ny = nx.to(self.device), ny.to(self.device)
                                    nout = self.model(nx)
                                    extra_losses.append(nn.functional.cross_entropy(nout, ny))
                                normal_loss_raw = sum(extra_losses) / float(len(extra_losses))
                        loss = self.objective.combine_losses(forget_loss_raw, normal_loss_raw)
                    else:
                        loss = forget_loss + retain_loss
                else:
                    loss = retain_loss
            else:
                # Single batch (fallback)
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Check if objective accepts inputs parameter (for SCRUB distillation)
                try:
                    loss = self.objective(outputs, labels, inputs=inputs)
                except TypeError:
                    # Objective doesn't accept inputs, call without it
                    loss = self.objective(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            if ((self.gu_projection or self.grad_surgery) or self.orthogonal_reg > 0) and isinstance(batch, tuple) and len(batch) == 2 and forget_part is not None:
                params = self._trainable_params()
                retain_grads = self._compute_grads(retain_loss, params)
                forget_grads = self._compute_grads(forget_loss, params)

                if self.gu_projection:
                    forget_grads = self._project_forget_grads(forget_grads, retain_grads, params, mode="gu")
                if self.grad_surgery:
                    forget_grads = self._project_forget_grads(forget_grads, retain_grads, params, mode="grad_surgery")
                if self.orthogonal_reg > 0:
                    forget_grads = self._project_forget_grads(
                        forget_grads, retain_grads, params, mode="orthogonal_reg", scale_override=self.orthogonal_reg
                    )

                for param, g_f, g_r in zip(params, forget_grads, retain_grads):
                    if g_f is None and g_r is None:
                        continue
                    grad = None
                    if g_f is not None and g_r is not None:
                        if self.gu_projection:
                            grad = g_f + (self.gu_beta * g_r)
                        else:
                            grad = g_f + g_r
                    elif g_f is not None:
                        grad = g_f
                    else:
                        grad = (self.gu_beta * g_r) if self.gu_projection else g_r
                    param.grad = grad
            else:
                loss.backward()
            self._apply_saliency_mask()
            self._apply_grad_noise()
            self.optimizer.step()

            # Update statistics
            epoch_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_loss /= num_batches
        return {'train_loss': epoch_loss}

    def _scrub_max_epoch(self, forget_loader: DataLoader) -> Dict[str, float]:
        """SCRUB max phase: maximize KL on forget set (minimize -KL)."""
        if self.objective.teacher_model is None:
            raise ValueError("SCRUB requires teacher_model to be set.")
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        for inputs, labels in forget_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            with torch.no_grad():
                teacher_logits = self.objective.teacher_model(inputs)
            loss = self.objective.compute_forget_loss(outputs, teacher_logits)
            loss = self.objective.forget_weight * loss
            self.optimizer.zero_grad()
            loss.backward()
            self._apply_saliency_mask()
            self._apply_grad_noise()
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / max(1, num_batches)
        return {"scrub_max_loss": avg_loss}

    def _scrub_min_epoch(self, retain_loader: DataLoader) -> Dict[str, float]:
        """SCRUB min phase: minimize KL+CE on retain set."""
        if self.objective.teacher_model is None:
            raise ValueError("SCRUB requires teacher_model to be set.")
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        for inputs, labels in retain_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if self.robust_retain:
                inputs = self.robust_retain_attack(inputs, labels)
            outputs = self.model(inputs)
            with torch.no_grad():
                teacher_logits = self.objective.teacher_model(inputs)
            loss = self.objective.compute_retain_loss(outputs, teacher_logits, labels)
            loss = loss * self.retain_lambda
            self.optimizer.zero_grad()
            loss.backward()
            self._apply_saliency_mask()
            self._apply_grad_noise()
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / max(1, num_batches)
        return {"scrub_min_loss": avg_loss}

    def validate(self) -> Dict[str, float]:
        """Validate model"""
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        forget_correct = 0
        forget_total = 0
        retain_correct = 0
        retain_total = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)

                # Check if objective accepts inputs parameter (for SCRUB distillation)
                try:
                    loss = self.objective(outputs, labels, inputs=inputs)
                except TypeError:
                    # Objective doesn't accept inputs, call without it
                    loss = self.objective(outputs, labels)

                val_loss += loss.item()

                # Accuracy calculations
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

                # Forget class accuracy
                forget_mask = (labels == self.objective.forget_class)
                forget_correct += (predicted[forget_mask] == labels[forget_mask]).sum().item()
                forget_total += forget_mask.sum().item()

                # Retain class accuracy
                retain_mask = (labels != self.objective.forget_class)
                retain_correct += (predicted[retain_mask] == labels[retain_mask]).sum().item()
                retain_total += retain_mask.sum().item()

        num_batches = len(self.val_loader)
        val_loss /= num_batches
        val_acc = correct / total
        forget_acc = forget_correct / forget_total if forget_total > 0 else 0.0
        retain_acc = retain_correct / retain_total if retain_total > 0 else 0.0

        return {
            'val_loss': val_loss,
            'val_acc': val_acc,
            'forget_acc': forget_acc,
            'retain_acc': retain_acc
        }

    def train(self,
              max_epochs: int = 50,
              early_stopping_patience: int = 10,
              save_every: int = 10,
              checkpoint_dir: Optional[str] = None,
              monitor_metric: str = "forget_acc",
              mode: str = "min") -> Dict[str, List[float]]:
        """
        Main training loop

        Args:
            max_epochs: Maximum number of epochs
            early_stopping_patience: Early stopping patience
            save_every: Save checkpoint every N epochs
            checkpoint_dir: Directory to save checkpoints
            monitor_metric: Metric to monitor for early stopping
            mode: "min" or "max" for the monitor metric

        Returns:
            Training history
        """
        self.max_epochs = max_epochs

        # Create checkpoint directory
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        print(f"Starting unlearning training for {max_epochs} epochs")
        print_trainable_parameters(self.model)

        best_metric = float('inf') if mode == "min" else float('-inf')
        patience_counter = 0
        scrub_rewind_records = []

        for epoch in range(max_epochs):
            self.current_epoch = epoch

            if hasattr(self.objective, "set_class_stats"):
                stats = self._compute_unseen_stats()
                if stats:
                    class_mu = {k: v[0] for k, v in stats.items()}
                    class_sigma = {k: v[1] for k, v in stats.items()}
                    self.objective.set_class_stats(class_mu, class_sigma)

            # Train epoch
            if self.scrub_full_schedule and self.objective.__class__.__name__.lower() == "scrub":
                if not hasattr(self.train_loader, "forget_loader") or not hasattr(self.train_loader, "retain_loader"):
                    raise ValueError("SCRUB full schedule requires train_loader.forget_loader and retain_loader")
                self._maybe_update_saliency_mask()
                max_steps = self.scrub_max_steps if self.scrub_max_steps is not None else max_epochs
                train_metrics = {}
                if epoch < max_steps:
                    train_metrics.update(self._scrub_max_epoch(self.train_loader.forget_loader))
                else:
                    train_metrics["scrub_max_loss"] = 0.0
                train_metrics.update(self._scrub_min_epoch(self.train_loader.retain_loader))
                train_metrics["train_loss"] = train_metrics.get("scrub_max_loss", 0.0) + train_metrics.get("scrub_min_loss", 0.0)
            else:
                train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_metrics['epoch'] = epoch

            # Store history
            self.training_history.append(epoch_metrics)

            # SCRUB rewind bookkeeping
            if self.scrub_rewind:
                forget_train_acc = 0.0
                if hasattr(self.train_loader, "forget_loader"):
                    forget_train_acc = self._evaluate_accuracy(self.train_loader.forget_loader)
                forget_val_acc = val_metrics.get("forget_acc", 0.0)
                record = {
                    "epoch": epoch,
                    "forget_train_acc": forget_train_acc,
                    "forget_val_acc": forget_val_acc,
                    "checkpoint_path": None
                }
                if checkpoint_dir:
                    ckpt_path = os.path.join(checkpoint_dir, f"scrub_rewind_epoch_{epoch}.pt")
                    torch.save(self.model.state_dict(), ckpt_path)
                    record["checkpoint_path"] = ckpt_path
                scrub_rewind_records.append(record)

            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()

            # Early stopping check
            if monitor_metric in val_metrics:
                current_metric = val_metrics[monitor_metric]

                if mode == "min":
                    is_better = current_metric < best_metric
                else:
                    is_better = current_metric > best_metric

                if is_better:
                    best_metric = current_metric
                    patience_counter = 0

                    # Save best model
                    if checkpoint_dir:
                        self.save_checkpoint(os.path.join(checkpoint_dir, "best_model"))
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            # Periodic checkpointing
            if checkpoint_dir and epoch % save_every == 0:
                self.save_checkpoint(os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}"))

            # Print progress
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in epoch_metrics.items() if isinstance(v, (int, float))])
            print(f"Epoch {epoch}: {metrics_str}")

        # Save final model
        if checkpoint_dir:
            self.save_checkpoint(os.path.join(checkpoint_dir, "final_model"))

        # SCRUB rewind: select checkpoint closest to reference forget error
        if self.scrub_rewind and scrub_rewind_records:
            reference = scrub_rewind_records[-1].get("forget_val_acc", 0.0)
            key = "forget_train_acc" if self.scrub_rewind_on_train else "forget_val_acc"
            best = min(scrub_rewind_records, key=lambda r: abs(r.get(key, 0.0) - reference))
            if best.get("checkpoint_path") and os.path.exists(best["checkpoint_path"]):
                print(f"[SCRUB rewind] Loading epoch {best['epoch']} (ref={reference:.4f}, {key}={best.get(key,0.0):.4f})")
                state = torch.load(best["checkpoint_path"], map_location=self.device)
                self.model.load_state_dict(state)

        return self.training_history

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        os.makedirs(path, exist_ok=True)

        # Save LoRA adapter or full model weights
        if hasattr(self.model, "save_pretrained"):
            save_lora_adapter(self.model, path)
        else:
            torch.save(
                {"model_state_dict": self.model.state_dict()},
                os.path.join(path, "model_state_dict.pt"),
            )

        # Save training state
        state = {
            'epoch': self.current_epoch,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_metric': self.best_val_metric,
            'training_history': self.training_history
        }

        torch.save(state, os.path.join(path, "training_state.pt"))

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        # Load training state
        state_path = os.path.join(path, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path)
            self.current_epoch = state['epoch']
            self.optimizer.load_state_dict(state['optimizer_state'])
            if self.scheduler and state['scheduler_state']:
                self.scheduler.load_state_dict(state['scheduler_state'])
            self.best_val_metric = state['best_val_metric']
            self.training_history = state['training_history']

        # LoRA adapter loading handled elsewhere; full model loading is separate.

    def save_training_history(self, path: str):
        """Save training history to JSON"""
        with open(path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_history = []
            for entry in self.training_history:
                serializable_entry = {}
                for k, v in entry.items():
                    if isinstance(v, np.ndarray):
                        serializable_entry[k] = v.tolist()
                    elif isinstance(v, (np.float32, np.float64)):
                        serializable_entry[k] = float(v)
                    elif isinstance(v, (np.int32, np.int64)):
                        serializable_entry[k] = int(v)
                    else:
                        serializable_entry[k] = v
                serializable_history.append(serializable_entry)

            json.dump(serializable_history, f, indent=2)


def create_unlearning_trainer(model: nn.Module,
                             objective_name: str,
                             forget_class: int,
                             train_loader: DataLoader,
                             val_loader: Optional[DataLoader] = None,
                             normal_loader: Optional[DataLoader] = None,
                             num_classes: int = 10,
                             device: str = "cuda",
                             retain_lambda: float = 1.0,
                             objective_kwargs: Dict = None,
                             robust_retain: bool = False,
                             robust_retain_eps: float = 8/255,
                             robust_retain_alpha: float = 2/255,
                             robust_retain_steps: int = 5,
                             grad_noise_std: float = 0.0,
                             gu_projection: bool = False,
                             grad_surgery: bool = False,
                             orthogonal_reg: float = 0.0,
                             gu_beta: float = 1.0,
                             scrub_rewind: bool = False,
                             scrub_rewind_on_train: bool = True,
                             scrub_full_schedule: bool = False,
                             scrub_max_steps: Optional[int] = None) -> UnlearningTrainer:
    """
    Factory function to create unlearning trainer

    Args:
        model: Model with LoRA applied
        objective_name: Name of unlearning objective
        forget_class: Class to forget
        train_loader: Training data loader
        val_loader: Validation data loader
        num_classes: Number of classes
        device: Training device
        retain_lambda: Weight for retain loss
        objective_kwargs: Additional kwargs for objective creation
        robust_retain: Whether to use adversarial examples for retain loss
        robust_retain_eps: Epsilon for robust retain preservation
        robust_retain_alpha: Step size for robust retain preservation
        robust_retain_steps: Number of PGD steps for robust retain
        gu_projection: Project forget gradients to be orthogonal to retain gradients
        grad_surgery: Apply conflict-aware projection (PCGrad-style)

    Returns:
        Configured UnlearningTrainer
    """
    if objective_kwargs is None:
        objective_kwargs = {}

    objective = create_unlearning_objective(objective_name, forget_class, num_classes, **objective_kwargs)
    trainer = UnlearningTrainer(
        model, objective, train_loader, val_loader, normal_loader, device, retain_lambda,
        robust_retain=robust_retain,
        robust_retain_eps=robust_retain_eps,
        robust_retain_alpha=robust_retain_alpha,
        robust_retain_steps=robust_retain_steps,
        grad_noise_std=grad_noise_std,
        gu_projection=gu_projection,
        grad_surgery=grad_surgery,
        orthogonal_reg=orthogonal_reg,
        gu_beta=gu_beta,
        scrub_rewind=scrub_rewind,
        scrub_rewind_on_train=scrub_rewind_on_train,
        scrub_full_schedule=scrub_full_schedule,
        scrub_max_steps=scrub_max_steps
    )

    if hasattr(objective, "set_model"):
        objective.set_model(model)

    return trainer


if __name__ == "__main__":
    # Test trainer setup (would need actual model and data)
    print("UnlearningTrainer class defined successfully")
    print("Run actual training with: python scripts/2_train_unlearning_lora.py")
