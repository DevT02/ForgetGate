"""
VPT (Visual Prompt Tuning) resurrection attack
Learns prompts/patches that restore performance on forgotten classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import Dict, Optional, List, Tuple, Union
import numpy as np
from tqdm import tqdm
import os
import json
import itertools

from ..models.vit import ViTWithVPT
from ..models.cnn import CNNWithVPT


class VPTResurrectionAttack:
    """VPT-based resurrection attack that learns prompts to restore forgotten class performance"""

    def __init__(self,
                 target_model: nn.Module,
                 forget_class: int,
                 prompt_config: Dict,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Args:
            target_model: Model to attack (unlearned model)
            forget_class: Class to resurrect
            prompt_config: VPT prompt configuration
            device: Device to run attack on
        """
        self.target_model = target_model.to(device)
        self.forget_class = forget_class
        self.prompt_config = prompt_config
        self.device = device

        # Freeze target model parameters
        for param in self.target_model.parameters():
            param.requires_grad = False

        # Sanity check: ensure target model is actually frozen (only prompt params should be trainable)
        assert sum(p.requires_grad for p in self.target_model.parameters()) == 0, \
            "Target model parameters should be frozen during VPT training"

        # Create VPT model wrapper
        self.vpt_model = self._create_vpt_model()
        self.vpt_model.to(self.device)

        # Attack statistics
        self.attack_history = []

    def _unwrap_to_core(self, m: nn.Module) -> nn.Module:
        """Try to peel off PEFT / wrapper layers until we hit the underlying timm/torchvision model."""
        # PEFT often provides get_base_model()
        if hasattr(m, "get_base_model"):
            try:
                m = m.get_base_model()
            except TypeError:
                pass

        # PEFT PeftModel commonly has .base_model
        if hasattr(m, "base_model"):
            m = m.base_model

        # Your wrappers (ViTWrapper/CNNWrapper) store the real model in .model
        if hasattr(m, "model"):
            m = m.model

        return m

    def _create_vpt_model(self) -> Union[ViTWithVPT, CNNWithVPT]:
        """Create VPT model wrapper based on base model type (PEFT-aware)"""
        core = self._unwrap_to_core(self.target_model)

        # ViT if it has patch_embed
        if hasattr(core, "patch_embed"):
            return ViTWithVPT(self.target_model, self.prompt_config)

        return CNNWithVPT(self.target_model, self.prompt_config)

    def _get_forget_samples(self, data_loader: DataLoader, max_samples: Optional[int] = None) -> DataLoader:
        """Extract forget class samples from data loader"""
        forget_inputs = []
        forget_labels = []
        collected = 0

        with torch.no_grad():
            for inputs, labels in data_loader:
                # Find forget class samples
                forget_mask = (labels == self.forget_class)
                if forget_mask.any():
                    x = inputs[forget_mask].cpu()
                    y = labels[forget_mask].cpu()
                    forget_inputs.append(x)
                    forget_labels.append(y)
                    collected += x.size(0)

                    if max_samples and collected >= max_samples:
                        break

        if not forget_inputs:
            raise ValueError(f"No samples found for forget class {self.forget_class}")

        # Concatenate batches
        forget_inputs = torch.cat(forget_inputs, dim=0)
        forget_labels = torch.cat(forget_labels, dim=0)

        # Limit samples if specified
        if max_samples and len(forget_inputs) > max_samples:
            indices = torch.randperm(len(forget_inputs))[:max_samples]
            forget_inputs = forget_inputs[indices]
            forget_labels = forget_labels[indices]

        # Create new data loader
        dataset = torch.utils.data.TensorDataset(forget_inputs, forget_labels)
        forget_loader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)

        return forget_loader

    def _get_retain_samples(self, data_loader: DataLoader, max_samples: Optional[int] = None) -> DataLoader:
        """Extract retain class samples from data loader (non-forget classes)"""
        retain_inputs = []
        retain_labels = []

        with torch.no_grad():
            for inputs, labels in data_loader:
                # Find retain class samples (not forget class)
                retain_mask = (labels != self.forget_class)
                if retain_mask.any():
                    retain_inputs.append(inputs[retain_mask].cpu())
                    retain_labels.append(labels[retain_mask].cpu())

                if max_samples and sum(len(x) for x in retain_inputs) >= max_samples:
                    break

        if not retain_inputs:
            raise ValueError(f"No retain samples found (all samples are forget class {self.forget_class})")

        # Concatenate batches
        retain_inputs = torch.cat(retain_inputs, dim=0)
        retain_labels = torch.cat(retain_labels, dim=0)

        # Limit samples if specified
        if max_samples and len(retain_inputs) > max_samples:
            indices = torch.randperm(len(retain_inputs))[:max_samples]
            retain_inputs = retain_inputs[indices]
            retain_labels = retain_labels[indices]

        # Create new data loader
        dataset = torch.utils.data.TensorDataset(retain_inputs, retain_labels)
        retain_loader = DataLoader(dataset, batch_size=min(64, len(dataset)), shuffle=True)

        return retain_loader

    def train_resurrection_prompt(self,
                                 data_loader: Optional[DataLoader] = None,
                                 train_loader: Optional[DataLoader] = None,
                                 val_loader: Optional[DataLoader] = None,
                                 train_forget_loader: Optional[DataLoader] = None,
                                 train_retain_loader: Optional[DataLoader] = None,
                                 val_forget_loader: Optional[DataLoader] = None,
                                 epochs: int = 100,
                                 lr: float = 1e-2,
                                 weight_decay: float = 0.0,
                                 patience: int = 20,
                                 eval_every: int = 10,
                                 lambda_retain: float = 1.0,
                                 T: float = 1.0) -> Dict[str, List[float]]:
        """
        Train VPT prompt to resurrect forgotten class performance

        Preferred: explicit train_forget_loader + train_retain_loader (avoids CPU caching)
        Fallback: data_loader/train_loader (backward compatible)

        Args:
            data_loader: Data loader with mixed forget+retain samples (backward compatible)
            train_loader: Training data loader with mixed samples (preferred fallback)
            val_loader: Validation data loader for evaluation/early stopping
            train_forget_loader: Explicit forget-class training loader (preferred)
            train_retain_loader: Explicit retain-class training loader (preferred)
            val_forget_loader: Explicit forget-class validation loader
            epochs: Number of training epochs
            lr: Learning rate
            weight_decay: Weight decay
            patience: Early stopping patience
            eval_every: Evaluate every N epochs
            lambda_retain: Weight for retain KL loss (0.0 = no retain regularization)
            T: Temperature for KL divergence

        Returns:
            Training history
        """
        # Preferred path: explicit loaders (avoids CPU caching + retain subsampling)
        if train_forget_loader is None or train_retain_loader is None:
            # Backward compatible path: derive from a mixed loader (older behavior)
            if data_loader is None:
                data_loader = train_loader
            if data_loader is None:
                raise TypeError(
                    "train_resurrection_prompt requires either explicit "
                    "`train_forget_loader`+`train_retain_loader` or `data_loader/train_loader`."
                )
            train_forget_loader = self._get_forget_samples(data_loader, max_samples=2000)
            train_retain_loader = self._get_retain_samples(data_loader, max_samples=4000)

        # Validation forget loader
        if val_forget_loader is not None:
            eval_forget_loader = val_forget_loader
        elif val_loader is not None:
            eval_forget_loader = self._get_forget_samples(val_loader, max_samples=1000)
        else:
            eval_forget_loader = train_forget_loader

        # Setup optimizer (optimize only trainable params)
        prompt_params = [p for p in self.vpt_model.parameters() if p.requires_grad]
        if len(prompt_params) == 0:
            names = [(n, p.requires_grad) for n, p in self.vpt_model.named_parameters()]
            raise RuntimeError(f"No trainable VPT parameters found. named_parameters={names}")

        optimizer = optim.Adam(prompt_params, lr=lr, weight_decay=weight_decay)

        print(f"Training VPT resurrection attack on forget={len(train_forget_loader.dataset)} samples")
        print(f"Retain regularization on retain={len(train_retain_loader.dataset)} samples")
        if eval_forget_loader is not None:
            print(f"Validating on forget={len(eval_forget_loader.dataset)} samples")
        print(f"Optimizing {len(prompt_params)} prompt parameters")

        # Debug: Compare unprompted vs prompted predictions on first forget batch
        print("\n--- VPT Debug: Checking if prompt affects predictions ---")
        self.vpt_model.eval()
        with torch.no_grad():
            # Get first batch from forget loader
            first_batch = next(iter(train_forget_loader))
            inputs, labels = first_batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Unprompted predictions (target model)
            unprompted_logits = self.target_model(inputs)
            unprompted_preds = unprompted_logits.argmax(dim=1)

            # Prompted predictions (VPT model)
            prompted_logits = self.vpt_model(inputs)
            prompted_preds = prompted_logits.argmax(dim=1)

            # Calculate forget class logit differences
            forget_logits_unprompted = unprompted_logits[:, self.forget_class]
            forget_logits_prompted = prompted_logits[:, self.forget_class]

            print(f"Unprompted predictions for forget class {self.forget_class}: {unprompted_preds[:10].tolist()}")
            print(f"Prompted predictions for forget class {self.forget_class}: {prompted_preds[:10].tolist()}")
            print(f"Forget class logit change: {forget_logits_prompted[:5] - forget_logits_unprompted[:5]}")
            print(f"Accuracy change: unprompted={(unprompted_preds == self.forget_class).float().mean():.3f}, "
                  f"prompted={(prompted_preds == self.forget_class).float().mean():.3f}")
        print("--- End VPT Debug ---\n")

        best_acc = 0.0
        best_model_state = None
        patience_counter = 0

        for epoch in range(epochs):
            # Training step
            train_metrics = self._train_epoch(train_forget_loader, train_retain_loader, optimizer,
                                             lambda_retain=lambda_retain, T=T)

            # Periodic evaluation
            if epoch % eval_every == 0:
                eval_metrics_raw = self._evaluate_resurrection(eval_forget_loader)

                # If val_loader is provided, also expose val_* keys
                if val_loader is not None:
                    eval_metrics = {f"val_{k}": v for k, v in eval_metrics_raw.items()}
                    # Keep old names for compatibility with existing logging/early stopping
                    eval_metrics["resurrection_acc"] = eval_metrics["val_resurrection_acc"]
                    eval_metrics["resurrection_loss"] = eval_metrics["val_resurrection_loss"]
                else:
                    eval_metrics = eval_metrics_raw

                # Combine metrics
                epoch_metrics = {**train_metrics, **eval_metrics, 'epoch': epoch}
                self.attack_history.append(epoch_metrics)

                current_acc = epoch_metrics.get('resurrection_acc', 0.0)

                # Early stopping and best checkpoint saving
                if current_acc > best_acc:
                    best_acc = current_acc
                    # Save best model state (only prompt parameters)
                    best_model_state = {
                        name: param.clone()
                        for name, param in self.vpt_model.named_parameters()
                        if param.requires_grad
                    }
                    patience_counter = 0
                    print(f"New best accuracy: {best_acc:.4f}")
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

                metrics_str = " | ".join(
                    [f"{k}: {v:.4f}" for k, v in epoch_metrics.items()
                     if isinstance(v, (int, float)) and not np.isnan(v)]
                )
                print(f"Epoch {epoch}: {metrics_str}")

        # Restore best checkpoint
        if best_model_state is not None:
            print(f"Restoring best model with accuracy {best_acc:.4f}")
            for name, param in self.vpt_model.named_parameters():
                if param.requires_grad and name in best_model_state:
                    param.data.copy_(best_model_state[name])

        return self.attack_history

    def _train_epoch(self, forget_loader: DataLoader, retain_loader: DataLoader,
                    optimizer: optim.Optimizer, lambda_retain: float = 1.0, T: float = 1.0) -> Dict[str, float]:
        """Train for one epoch with forget CE loss + retain KL regularization"""
        self.vpt_model.train()
        epoch_loss = 0.0
        num_batches = 0

        # Use itertools.cycle to handle different dataset sizes
        retain_iter = itertools.cycle(retain_loader)

        for xf, yf in forget_loader:
            # Get retain batch (cycle if retain_loader is shorter)
            xr, yr = next(retain_iter)

            xf, yf = xf.to(self.device), yf.to(self.device)
            xr, yr = xr.to(self.device), yr.to(self.device)

            # Forget: make forget class correct (margin loss for better resurrection)
            logits_f = self.vpt_model(xf)
            true_logit = logits_f.gather(1, yf.unsqueeze(1)).squeeze(1)

            # Create mask to exclude the true class
            mask = torch.ones_like(logits_f, dtype=torch.bool)
            mask.scatter_(1, yf.unsqueeze(1), False)
            max_other = logits_f.masked_fill(~mask, -1e9).max(dim=1).values

            # Margin loss: maximize (true_logit - max_other)
            loss_forget = -(true_logit - max_other).mean()

            # Retain: keep behavior close to unprompted target model (KL regularization)
            with torch.no_grad():
                teacher_logits = self.target_model(xr)  # frozen, no prompt

            student_logits = self.vpt_model(xr)  # with prompt

            # KL divergence: student || teacher
            loss_retain = F.kl_div(
                F.log_softmax(student_logits / T, dim=1),
                F.softmax(teacher_logits / T, dim=1),
                reduction="batchmean"
            ) * (T * T)

            # Combined loss
            loss = loss_forget + lambda_retain * loss_retain

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        epoch_loss /= max(num_batches, 1)
        return {
            'train_loss': epoch_loss,
            'loss_forget': loss_forget.item() if 'loss_forget' in locals() else 0.0,
            'loss_retain': loss_retain.item() if 'loss_retain' in locals() else 0.0
        }

    def _evaluate_resurrection(self, forget_loader: DataLoader) -> Dict[str, float]:
        """Evaluate resurrection success"""
        self.vpt_model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for inputs, labels in forget_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.vpt_model(inputs)
                loss = F.cross_entropy(outputs, labels)

                total_loss += loss.item()

                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(forget_loader)

        return {
            'resurrection_acc': accuracy,
            'resurrection_loss': avg_loss
        }

    def attack_sample(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply trained VPT attack to a single sample

        Args:
            input_tensor: Input image tensor [C, H, W]

        Returns:
            Attacked output
        """
        self.vpt_model.eval()

        with torch.no_grad():
            # Add batch dimension if needed
            if input_tensor.dim() == 3:
                input_tensor = input_tensor.unsqueeze(0)

            input_tensor = input_tensor.to(self.device)
            output = self.vpt_model(input_tensor)

        return output

    def save_attack_prompt(self, save_path: str):
        """Save trained VPT prompt"""
        os.makedirs(save_path, exist_ok=True)

        # Save prompt parameters
        prompt_state = {}
        for name, param in self.vpt_model.named_parameters():
            if 'prompt' in name.lower():
                prompt_state[name] = param.cpu()

        torch.save(prompt_state, os.path.join(save_path, "vpt_prompt.pt"))

        # Save attack config and history
        attack_info = {
            'forget_class': self.forget_class,
            'prompt_config': self.prompt_config,
            'attack_history': self.attack_history
        }

        with open(os.path.join(save_path, "attack_info.json"), 'w') as f:
            json.dump(attack_info, f, indent=2)

    def load_attack_prompt(self, load_path: str):
        """Load trained VPT prompt"""
        prompt_path = os.path.join(load_path, "vpt_prompt.pt")

        if os.path.exists(prompt_path):
            prompt_state = torch.load(prompt_path)
            self.vpt_model.load_state_dict(prompt_state, strict=False)

        # Load attack info
        info_path = os.path.join(load_path, "attack_info.json")
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                attack_info = json.load(f)
            self.attack_history = attack_info.get('attack_history', [])

    def get_prompt_statistics(self) -> Dict[str, float]:
        """Get statistics about the trained prompt"""
        prompt_params = []
        for name, param in self.vpt_model.named_parameters():
            if 'prompt' in name.lower():
                prompt_params.append(param)

        if not prompt_params:
            return {}

        total_params = sum(p.numel() for p in prompt_params)
        total_norm = sum(torch.norm(p).item() for p in prompt_params)

        return {
            'total_prompt_params': total_params,
            'prompt_norm': total_norm,
            'num_prompt_tensors': len(prompt_params)
        }


class JointVPTAdversarialAttack:
    """Combined VPT + adversarial perturbation attack"""

    def __init__(self,
                 target_model: nn.Module,
                 forget_class: int,
                 prompt_config: Dict,
                 adv_config: Dict,
                 device: str = "cuda"):
        """
        Args:
            target_model: Model to attack
            forget_class: Class to resurrect
            prompt_config: VPT configuration
            adv_config: Adversarial attack configuration
            device: Device to run on
        """
        self.vpt_attack = VPTResurrectionAttack(target_model, forget_class, prompt_config, device)
        self.adv_config = adv_config
        self.forget_class = forget_class
        self.device = device

    def attack_sample(self, input_tensor: torch.Tensor, target_label: Optional[int] = None) -> torch.Tensor:
        """
        Apply joint VPT + adversarial attack

        Args:
            input_tensor: Input image tensor
            target_label: Target label for resurrection

        Returns:
            Attacked output
        """
        # First apply VPT
        vpt_output = self.vpt_attack.attack_sample(input_tensor)

        # Then apply adversarial perturbation (simplified PGD)
        adv_input = self._apply_adversarial_perturbation(input_tensor, target_label)

        # Get final output
        with torch.no_grad():
            final_output = self.vpt_attack.vpt_model(adv_input)

        return final_output

    def _apply_adversarial_perturbation(self, input_tensor: torch.Tensor,
                                      target_label: Optional[int] = None) -> torch.Tensor:
        """Apply adversarial perturbation (simplified implementation)"""
        # This would integrate with torchattacks or custom PGD
        # For now, return input unchanged
        return input_tensor


if __name__ == "__main__":
    # Test VPT attack setup
    print("VPTResurrectionAttack class defined successfully")
    print("Run actual attack with: python scripts/3_train_vpt_resurrector.py")
