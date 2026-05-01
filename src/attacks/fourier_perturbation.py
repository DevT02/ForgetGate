"""
Imperceptible Fourier-basis universal perturbation for ForgetGate.
"""

import itertools
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from ..models.normalize import create_imagenet_normalizer
from .universal_patch import (
    _aggregate_model_losses,
    _aggregate_target_success,
    _apply_binary_probe,
    _as_model_list,
    _fit_binary_probe,
    _normalize_features,
    _split_normalized_model,
    _targeted_margin_loss,
)


def _resolve_feature_model(model: nn.Module) -> Optional[nn.Module]:
    queue = [model]
    seen = set()
    fallback = None

    while queue:
        current = queue.pop(0)
        if current is None:
            continue
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)

        if hasattr(current, "forward_features"):
            if fallback is None:
                fallback = current
            if hasattr(current, "blocks"):
                return current

        for attr in ("module", "base_model", "model"):
            child = getattr(current, attr, None)
            if isinstance(child, nn.Module):
                queue.append(child)

        if hasattr(current, "get_base_model"):
            try:
                child = current.get_base_model()
            except TypeError:
                child = None
            if isinstance(child, nn.Module):
                queue.append(child)

        for child in current.children():
            if isinstance(child, nn.Module):
                queue.append(child)

    return fallback


def _extract_features_and_block_reps(
    wrapped_model: nn.Module,
    inputs: torch.Tensor,
    block_indices: Sequence[int],
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    if hasattr(wrapped_model, "apply_perturbation") and hasattr(wrapped_model, "base_model"):
        inputs = wrapped_model.apply_perturbation(inputs)
        wrapped_model = wrapped_model.base_model

    normalizer, inner_model = _split_normalized_model(wrapped_model)
    normalized_inputs = normalizer(inputs)
    feature_model = _resolve_feature_model(inner_model)
    if feature_model is None:
        raise ValueError("Fourier attack feature guidance requires a model with forward_features support.")

    block_outputs: Dict[int, torch.Tensor] = {}
    handles = []
    blocks = getattr(feature_model, "blocks", None)
    if blocks is not None:
        for block_idx in sorted(set(int(idx) for idx in block_indices)):
            if block_idx < 0 or block_idx >= len(blocks):
                continue
            handles.append(
                blocks[block_idx].register_forward_hook(
                    lambda _module, _inputs, output, idx=block_idx: block_outputs.__setitem__(
                        idx, output[0] if isinstance(output, (tuple, list)) else output
                    )
                )
            )

    try:
        features = feature_model.forward_features(normalized_inputs)
    finally:
        for handle in handles:
            handle.remove()

    if isinstance(features, (tuple, list)):
        features = features[0]
    if features.ndim == 3:
        final_features = features[:, 0]
    elif features.ndim > 2:
        final_features = features.flatten(start_dim=1)
    else:
        final_features = features

    block_reps: Dict[int, torch.Tensor] = {}
    for block_idx, tokens in block_outputs.items():
        if tokens.ndim == 3 and tokens.size(1) > 1:
            cls_token = tokens[:, 0]
            patch_mean = tokens[:, 1:].mean(dim=1)
            block_reps[block_idx] = _normalize_features(cls_token + patch_mean)
        else:
            block_reps[block_idx] = _normalize_features(tokens.flatten(start_dim=1))

    return final_features, block_outputs, block_reps


def _frequency_magnitudes(size: int, device: torch.device) -> torch.Tensor:
    indices = torch.arange(size, device=device)
    neg_indices = torch.remainder(-indices, size)
    return torch.minimum(indices, neg_indices).float()


def _nearest_target_distance(grid: torch.Tensor, targets: Sequence[int]) -> torch.Tensor:
    if not targets:
        return torch.full_like(grid, float("inf"), dtype=torch.float32)
    target_tensor = torch.tensor(list(targets), device=grid.device, dtype=torch.float32)
    expand_shape = [1] * grid.ndim + [target_tensor.numel()]
    return (grid.unsqueeze(-1) - target_tensor.view(*expand_shape)).abs().min(dim=-1).values


class FourierPerturbationModel(nn.Module):
    """Wrap a model with a frequency-constrained additive perturbation."""

    def __init__(self, base_model: nn.Module, attack_config: Dict):
        super().__init__()
        self.base_model = base_model
        self.attack_config = dict(attack_config)
        self.image_size = int(self.attack_config.get("image_size", 224))
        self.patch_stride = int(self.attack_config.get("patch_stride", 16))
        self.frequency_bandwidth = float(self.attack_config.get("frequency_bandwidth", 1.0))
        self.epsilon = float(self.attack_config.get("epsilon", 8.0 / 255.0))
        self.random_roll_max = int(self.attack_config.get("random_roll_max", 0))
        self.basis_mode = str(self.attack_config.get("basis_mode", "patch_grid")).lower()
        self.harmonics = [int(v) for v in self.attack_config.get("harmonics", [1, 2, 3])]

        init_scale = float(self.attack_config.get("init_scale", 1e-3))
        device = next(base_model.parameters()).device
        height = self.image_size
        width = self.image_size
        freq_shape = (1, 3, height, width // 2 + 1)

        self.spectral_real = nn.Parameter(torch.zeros(freq_shape, device=device))
        self.spectral_imag = nn.Parameter(torch.zeros(freq_shape, device=device))
        nn.init.normal_(self.spectral_real, mean=0.0, std=init_scale)
        nn.init.normal_(self.spectral_imag, mean=0.0, std=init_scale)

        mask = self._build_frequency_mask(height, width, device)
        self.register_buffer("frequency_mask", mask)

    def _build_frequency_mask(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        ky_mag = _frequency_magnitudes(height, device).view(height, 1)
        kx = torch.arange(width // 2 + 1, device=device).float().view(1, width // 2 + 1)

        base_y = float(height) / float(self.patch_stride)
        base_x = float(width) / float(self.patch_stride)
        target_y = [
            int(round(base_y * harmonic))
            for harmonic in self.harmonics
            if int(round(base_y * harmonic)) > 0 and int(round(base_y * harmonic)) <= height // 2
        ]
        target_x = [
            int(round(base_x * harmonic))
            for harmonic in self.harmonics
            if int(round(base_x * harmonic)) > 0 and int(round(base_x * harmonic)) <= width // 2
        ]

        near_y = _nearest_target_distance(ky_mag, target_y) <= self.frequency_bandwidth
        near_x = _nearest_target_distance(kx, target_x) <= self.frequency_bandwidth

        if self.basis_mode == "diagonal":
            mask_2d = near_y & near_x
        elif self.basis_mode == "hybrid":
            mask_2d = (near_y | near_x) | (near_y & near_x)
        else:
            mask_2d = near_y | near_x

        low_freq_radius = float(self.attack_config.get("low_freq_radius", 1.5))
        if bool(self.attack_config.get("exclude_low_freq", True)):
            low_freq = torch.sqrt(ky_mag.pow(2) + kx.pow(2)) <= low_freq_radius
            mask_2d = mask_2d & (~low_freq)

        mask_2d[0, 0] = False
        return mask_2d.unsqueeze(0).unsqueeze(0).expand(1, 3, -1, -1).float()

    def get_complex_spectrum(self) -> torch.Tensor:
        return torch.complex(self.spectral_real, self.spectral_imag) * self.frequency_mask

    def get_delta(self) -> torch.Tensor:
        spectrum = self.get_complex_spectrum()
        delta = torch.fft.irfft2(spectrum, s=(self.image_size, self.image_size), norm="ortho")
        delta = delta - delta.mean(dim=(2, 3), keepdim=True)
        spatial_gain = float(self.attack_config.get("spatial_gain", 24.0))
        return self.epsilon * torch.tanh(spatial_gain * delta)

    def apply_perturbation(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = inputs.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"Fourier attack expected {self.image_size}x{self.image_size} inputs, got {height}x{width}"
            )

        delta = self.get_delta().to(device=inputs.device, dtype=inputs.dtype)
        if self.training and self.random_roll_max > 0:
            deltas = []
            for _ in range(batch_size):
                shift_y = int(torch.randint(-self.random_roll_max, self.random_roll_max + 1, (1,), device=inputs.device).item())
                shift_x = int(torch.randint(-self.random_roll_max, self.random_roll_max + 1, (1,), device=inputs.device).item())
                deltas.append(torch.roll(delta, shifts=(shift_y, shift_x), dims=(2, 3)))
            delta_batch = torch.cat(deltas, dim=0)
        else:
            delta_batch = delta.expand(batch_size, -1, -1, -1)

        return torch.clamp(inputs + delta_batch, 0.0, 1.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base_model(self.apply_perturbation(inputs))


class UniversalFourierAttack:
    """Train and evaluate a universal Fourier-basis perturbation."""

    def __init__(
        self,
        target_models: Union[nn.Module, Sequence[nn.Module]],
        oracle_model: Optional[nn.Module],
        forget_class: int,
        attack_config: Dict,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.forget_class = int(forget_class)
        self.attack_config = dict(attack_config)
        self.attack_history: List[Dict] = []
        self.feature_mode = str(self.attack_config.get("feature_mode", "")).lower().strip()
        self.harp_state: Optional[Dict] = None

        target_models = _as_model_list(target_models)
        self.target_clean_models = [
            nn.Sequential(create_imagenet_normalizer(), model.to(device)).to(device)
            for model in target_models
        ]
        for model in self.target_clean_models:
            for param in model.parameters():
                param.requires_grad = False

        self.oracle_clean_model = None
        if oracle_model is not None:
            self.oracle_clean_model = nn.Sequential(
                create_imagenet_normalizer(), oracle_model.to(device)
            ).to(device)
            for param in self.oracle_clean_model.parameters():
                param.requires_grad = False

        self.perturbed_target_models = [
            FourierPerturbationModel(model, self.attack_config).to(device)
            for model in self.target_clean_models
        ]
        for extra_model in self.perturbed_target_models[1:]:
            extra_model.spectral_real = self.perturbed_target_models[0].spectral_real
            extra_model.spectral_imag = self.perturbed_target_models[0].spectral_imag

        self.perturbed_target_model = self.perturbed_target_models[0]

        self.oracle_perturbed_model = None
        if self.oracle_clean_model is not None:
            self.oracle_perturbed_model = FourierPerturbationModel(
                self.oracle_clean_model, self.attack_config
            ).to(device)
            self.oracle_perturbed_model.spectral_real = self.perturbed_target_model.spectral_real
            self.oracle_perturbed_model.spectral_imag = self.perturbed_target_model.spectral_imag

        for model in self.target_clean_models:
            model.eval()
        for model in self.perturbed_target_models:
            model.eval()
            model.base_model.eval()
        if self.oracle_clean_model is not None:
            self.oracle_clean_model.eval()
        if self.oracle_perturbed_model is not None:
            self.oracle_perturbed_model.eval()
            self.oracle_perturbed_model.base_model.eval()

    def _collect_model_features(
        self,
        model: nn.Module,
        loader: DataLoader,
        block_indices: Sequence[int],
        max_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]:
        final_features = []
        labels = []
        block_features: Dict[int, List[torch.Tensor]] = {int(idx): [] for idx in block_indices}
        collected = 0

        model.eval()
        with torch.no_grad():
            for inputs, batch_labels in loader:
                inputs = inputs.to(self.device)
                batch_labels = batch_labels.to(self.device)
                feats, _block_tokens, block_reps = _extract_features_and_block_reps(
                    model, inputs, block_indices
                )
                final_features.append(feats.detach())
                labels.append(batch_labels.detach())
                for block_idx in block_indices:
                    rep = block_reps.get(int(block_idx))
                    if rep is not None:
                        block_features[int(block_idx)].append(rep.detach())
                collected += inputs.size(0)
                if collected >= max_samples:
                    break

        if not final_features:
            raise ValueError("Failed to collect features for Fourier guidance.")

        final_tensor = torch.cat(final_features, dim=0)[:max_samples]
        label_tensor = torch.cat(labels, dim=0)[:max_samples]
        block_tensors: Dict[int, torch.Tensor] = {}
        for block_idx, reps in block_features.items():
            if reps:
                block_tensors[block_idx] = torch.cat(reps, dim=0)[:max_samples]
        return final_tensor, label_tensor, block_tensors

    def _prepare_harp_guidance(self, forget_loader: DataLoader, retain_loader: DataLoader) -> Dict:
        block_indices = [int(idx) for idx in self.attack_config.get("token_blocks", [0, 5, 11])]
        max_forget = int(self.attack_config.get("max_forget_samples", 1024))
        max_retain = int(self.attack_config.get("max_retain_samples", 2048))
        probe_epochs = int(self.attack_config.get("probe_epochs", 200))
        probe_lr = float(self.attack_config.get("probe_lr", 1e-2))

        model_states = []
        for model in self.target_clean_models:
            forget_feats, _forget_labels, forget_blocks = self._collect_model_features(
                model, forget_loader, block_indices, max_forget
            )
            retain_feats, retain_labels, retain_blocks = self._collect_model_features(
                model, retain_loader, block_indices, max_retain
            )

            forget_feats_norm = _normalize_features(forget_feats)
            retain_feats_norm = _normalize_features(retain_feats)
            forget_proto = _normalize_features(forget_feats_norm.mean(dim=0, keepdim=True)).squeeze(0)
            retain_proto_labels = []
            retain_proto_list = []
            for label in sorted(int(lbl) for lbl in retain_labels.unique().tolist() if int(lbl) != self.forget_class):
                mask = retain_labels == label
                if mask.any():
                    retain_proto_labels.append(label)
                    retain_proto_list.append(
                        _normalize_features(retain_feats_norm[mask].mean(dim=0, keepdim=True)).squeeze(0)
                    )

            final_probe = _fit_binary_probe(
                positive_features=forget_feats,
                negative_features=retain_feats,
                epochs=probe_epochs,
                lr=probe_lr,
            )

            block_probes = {}
            for block_idx in block_indices:
                if block_idx not in forget_blocks or block_idx not in retain_blocks:
                    continue
                block_probes[block_idx] = _fit_binary_probe(
                    positive_features=forget_blocks[block_idx],
                    negative_features=retain_blocks[block_idx],
                    epochs=probe_epochs,
                    lr=probe_lr,
                )

            model_states.append(
                {
                    "forget_proto": forget_proto.detach(),
                    "retain_proto_labels": torch.tensor(retain_proto_labels, device=self.device, dtype=torch.long),
                    "retain_proto_matrix": (
                        torch.stack(retain_proto_list, dim=0).detach()
                        if retain_proto_list
                        else torch.empty(0, forget_feats.size(1), device=self.device)
                    ),
                    "final_probe": final_probe,
                    "block_probes": block_probes,
                }
            )

        summary = {
            "mode": "harp",
            "block_indices": block_indices,
            "max_forget_samples": max_forget,
            "max_retain_samples": max_retain,
            "probe_epochs": probe_epochs,
            "probe_lr": probe_lr,
            "target_probe_train_acc": [float(state["final_probe"]["train_acc"]) for state in model_states],
        }
        self.harp_state = {
            "summary": summary,
            "block_indices": block_indices,
            "models": model_states,
        }
        return self.harp_state

    def _compute_harp_losses(self, inputs_forget: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not self.harp_state:
            zero = torch.tensor(0.0, device=self.device)
            return {
                "loss_proto": zero,
                "loss_final_probe": zero,
                "loss_token_probe": zero,
                "loss_sink": zero,
            }

        proto_margin = float(self.attack_config.get("prototype_margin", 0.1))
        final_probe_margin = float(self.attack_config.get("final_probe_margin", 1.0))
        token_probe_margin = float(self.attack_config.get("token_probe_margin", 1.0))
        block_indices = self.harp_state["block_indices"]

        loss_proto_terms = []
        loss_final_probe_terms = []
        loss_token_probe_terms = []
        loss_sink_terms = []

        for perturbed_model, state in zip(self.perturbed_target_models, self.harp_state["models"]):
            final_features, block_tokens, block_reps = _extract_features_and_block_reps(
                perturbed_model, inputs_forget, block_indices
            )
            final_features_norm = _normalize_features(final_features)

            final_scores = _apply_binary_probe(final_features_norm, state["final_probe"])
            loss_final_probe_terms.append(F.relu(final_probe_margin - final_scores).mean())

            retain_proto_matrix = state["retain_proto_matrix"]
            if retain_proto_matrix.numel() > 0:
                d_forget = torch.cdist(final_features_norm, state["forget_proto"].unsqueeze(0), p=2).squeeze(1)
                d_retain = torch.cdist(final_features_norm, retain_proto_matrix, p=2).min(dim=1).values
                loss_proto_terms.append(F.relu(proto_margin + d_forget - d_retain).mean())

            for block_idx, probe_state in state["block_probes"].items():
                rep = block_reps.get(int(block_idx))
                tokens = block_tokens.get(int(block_idx))
                if rep is not None:
                    scores = _apply_binary_probe(rep, probe_state)
                    loss_token_probe_terms.append(F.relu(token_probe_margin - scores).mean())
                if tokens is not None and tokens.ndim == 3 and tokens.size(1) > 1:
                    cls_token = tokens[:, 0]
                    patch_mean = tokens[:, 1:].mean(dim=1)
                    loss_sink_terms.append((1.0 - F.cosine_similarity(cls_token, patch_mean, dim=1)).mean())

        zero = torch.tensor(0.0, device=self.device)
        return {
            "loss_proto": torch.stack(loss_proto_terms).mean() if loss_proto_terms else zero,
            "loss_final_probe": torch.stack(loss_final_probe_terms).mean() if loss_final_probe_terms else zero,
            "loss_token_probe": torch.stack(loss_token_probe_terms).mean() if loss_token_probe_terms else zero,
            "loss_sink": torch.stack(loss_sink_terms).mean() if loss_sink_terms else zero,
        }

    def _evaluate_resurrection(self, forget_loader: DataLoader) -> Dict[str, float]:
        self.perturbed_target_model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for inputs, labels in forget_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.perturbed_target_model(inputs)
                loss = F.cross_entropy(outputs, labels)
                total_loss += loss.item()
                predicted = outputs.argmax(dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        return {
            "resurrection_acc": correct / total if total > 0 else 0.0,
            "resurrection_loss": total_loss / max(len(forget_loader), 1),
        }

    def _train_one_epoch(
        self,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        optimizer: optim.Optimizer,
        lambda_retain: float,
        temperature: float,
        oracle_contrast_weight: float,
    ) -> Dict[str, float]:
        for model in self.perturbed_target_models:
            model.train()
            model.base_model.eval()
        if self.oracle_perturbed_model is not None:
            self.oracle_perturbed_model.train()
            self.oracle_perturbed_model.base_model.eval()
        retain_iter = itertools.cycle(retain_loader)

        epoch_loss = 0.0
        loss_forget_last = 0.0
        loss_retain_last = 0.0
        loss_oracle_last = 0.0
        loss_proto_last = 0.0
        loss_final_probe_last = 0.0
        loss_token_probe_last = 0.0
        loss_sink_last = 0.0
        loss_delta_l2_last = 0.0
        loss_spectral_l1_last = 0.0
        target_forget_prob_last = 0.0
        oracle_forget_prob_last = 0.0
        num_batches = 0

        prototype_weight = float(self.attack_config.get("prototype_weight", 0.0))
        final_probe_weight = float(self.attack_config.get("final_probe_weight", 0.0))
        token_probe_weight = float(self.attack_config.get("token_probe_weight", 0.0))
        sink_weight = float(self.attack_config.get("sink_weight", 0.0))
        delta_l2_weight = float(self.attack_config.get("delta_l2_weight", 0.0))
        spectral_l1_weight = float(self.attack_config.get("spectral_l1_weight", 0.0))
        forget_objective = str(self.attack_config.get("forget_objective", "targeted_margin")).lower()
        target_margin = float(self.attack_config.get("target_margin", 1.0))
        forget_aggregation = str(self.attack_config.get("forget_aggregation", "mean")).lower()
        forget_aggregation_temperature = float(self.attack_config.get("forget_aggregation_temperature", 10.0))
        oracle_contrast_margin = float(self.attack_config.get("oracle_contrast_margin", 0.05))

        for inputs_forget, labels_forget in forget_loader:
            inputs_forget = inputs_forget.to(self.device)
            labels_forget = labels_forget.to(self.device)
            inputs_retain, _ = next(retain_iter)
            inputs_retain = inputs_retain.to(self.device)

            forget_outputs = [model(inputs_forget) for model in self.perturbed_target_models]
            if forget_objective == "targeted_margin":
                forget_losses = [_targeted_margin_loss(outputs, labels_forget, target_margin) for outputs in forget_outputs]
            else:
                forget_losses = [F.cross_entropy(outputs, labels_forget) for outputs in forget_outputs]
            loss_forget = _aggregate_model_losses(
                forget_losses,
                mode=forget_aggregation,
                temperature=forget_aggregation_temperature,
            )

            clean_retain_outputs = []
            with torch.no_grad():
                for model in self.target_clean_models:
                    clean_retain_outputs.append(model(inputs_retain))

            perturbed_retain_outputs = [model(inputs_retain) for model in self.perturbed_target_models]
            loss_retain_terms = []
            for clean_out, perturbed_out in zip(clean_retain_outputs, perturbed_retain_outputs):
                loss_retain_terms.append(
                    F.kl_div(
                        F.log_softmax(perturbed_out / temperature, dim=1),
                        F.softmax(clean_out / temperature, dim=1),
                        reduction="batchmean",
                    )
                    * (temperature * temperature)
                )
            loss_retain = torch.stack(loss_retain_terms).mean()

            target_probs = [F.softmax(outputs, dim=1)[:, self.forget_class].mean() for outputs in forget_outputs]
            target_forget_prob = _aggregate_target_success(
                target_probs,
                mode=forget_aggregation,
                temperature=forget_aggregation_temperature,
            )

            loss_oracle = torch.tensor(0.0, device=self.device)
            oracle_forget_prob = torch.tensor(0.0, device=self.device)
            if self.oracle_perturbed_model is not None and oracle_contrast_weight > 0.0:
                oracle_outputs = self.oracle_perturbed_model(inputs_forget)
                oracle_probs = F.softmax(oracle_outputs, dim=1)[:, self.forget_class]
                oracle_forget_prob = oracle_probs.mean()
                loss_oracle = F.relu(
                    oracle_contrast_margin + oracle_forget_prob - target_forget_prob
                )

            harp_losses = {
                "loss_proto": torch.tensor(0.0, device=self.device),
                "loss_final_probe": torch.tensor(0.0, device=self.device),
                "loss_token_probe": torch.tensor(0.0, device=self.device),
                "loss_sink": torch.tensor(0.0, device=self.device),
            }
            if self.feature_mode == "harp":
                harp_losses = self._compute_harp_losses(inputs_forget)

            delta = self.perturbed_target_model.get_delta()
            delta_l2 = delta.pow(2).mean()
            spectral_l1 = self.perturbed_target_model.get_complex_spectrum().abs().mean()

            total_loss = (
                loss_forget
                + lambda_retain * loss_retain
                + oracle_contrast_weight * loss_oracle
                + prototype_weight * harp_losses["loss_proto"]
                + final_probe_weight * harp_losses["loss_final_probe"]
                + token_probe_weight * harp_losses["loss_token_probe"]
                + sink_weight * harp_losses["loss_sink"]
                + delta_l2_weight * delta_l2
                + spectral_l1_weight * spectral_l1
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            loss_forget_last = loss_forget.item()
            loss_retain_last = loss_retain.item()
            loss_oracle_last = loss_oracle.item()
            loss_proto_last = harp_losses["loss_proto"].item()
            loss_final_probe_last = harp_losses["loss_final_probe"].item()
            loss_token_probe_last = harp_losses["loss_token_probe"].item()
            loss_sink_last = harp_losses["loss_sink"].item()
            loss_delta_l2_last = delta_l2.item()
            loss_spectral_l1_last = spectral_l1.item()
            target_forget_prob_last = target_forget_prob.item()
            oracle_forget_prob_last = oracle_forget_prob.item()
            num_batches += 1

        return {
            "train_loss": epoch_loss / max(num_batches, 1),
            "loss_forget": loss_forget_last,
            "loss_retain": loss_retain_last,
            "loss_oracle": loss_oracle_last,
            "loss_proto": loss_proto_last,
            "loss_final_probe": loss_final_probe_last,
            "loss_token_probe": loss_token_probe_last,
            "loss_sink": loss_sink_last,
            "loss_delta_l2": loss_delta_l2_last,
            "loss_spectral_l1": loss_spectral_l1_last,
            "target_forget_prob": target_forget_prob_last,
            "oracle_forget_prob": oracle_forget_prob_last,
            "oracle_contrast_gap": target_forget_prob_last - oracle_forget_prob_last,
        }

    def train_attack(
        self,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        val_forget_loader: DataLoader,
        guidance_forget_loader: Optional[DataLoader] = None,
        guidance_retain_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        lr: float = 5e-2,
        lambda_retain: float = 1.0,
        temperature: float = 1.0,
        eval_every: int = 10,
        oracle_contrast_weight: float = 1.0,
    ) -> List[Dict]:
        if self.feature_mode == "harp":
            guidance_state = self._prepare_harp_guidance(
                forget_loader=guidance_forget_loader or forget_loader,
                retain_loader=guidance_retain_loader or retain_loader,
            )
            print(f"Fourier guidance enabled: {guidance_state['summary']}")

        optimizer = optim.Adam(
            [
                self.perturbed_target_model.spectral_real,
                self.perturbed_target_model.spectral_imag,
            ],
            lr=lr,
        )
        best_acc = -1.0
        best_loss = float("inf")
        best_epoch = -1
        best_state = None

        print(f"Training Fourier attack on forget={len(forget_loader.dataset)} samples")
        print(f"Retain regularization on retain={len(retain_loader.dataset)} samples")
        print(f"Validating on forget={len(val_forget_loader.dataset)} samples")

        for epoch in range(epochs):
            metrics = self._train_one_epoch(
                forget_loader=forget_loader,
                retain_loader=retain_loader,
                optimizer=optimizer,
                lambda_retain=lambda_retain,
                temperature=temperature,
                oracle_contrast_weight=oracle_contrast_weight,
            )

            if epoch % eval_every == 0 or epoch == epochs - 1:
                eval_metrics = self._evaluate_resurrection(val_forget_loader)
                metrics.update(eval_metrics)
                metrics["epoch"] = float(epoch)
                self.attack_history.append(metrics)
                print(
                    f"Epoch {epoch}: "
                    f"train_loss: {metrics['train_loss']:.4f} | "
                    f"loss_forget: {metrics['loss_forget']:.4f} | "
                    f"loss_retain: {metrics['loss_retain']:.4f} | "
                    f"loss_oracle: {metrics['loss_oracle']:.4f} | "
                    f"loss_proto: {metrics['loss_proto']:.4f} | "
                    f"loss_final_probe: {metrics['loss_final_probe']:.4f} | "
                    f"loss_token_probe: {metrics['loss_token_probe']:.4f} | "
                    f"loss_sink: {metrics['loss_sink']:.4f} | "
                    f"loss_delta_l2: {metrics['loss_delta_l2']:.4f} | "
                    f"loss_spectral_l1: {metrics['loss_spectral_l1']:.4f} | "
                    f"target_forget_prob: {metrics['target_forget_prob']:.4f} | "
                    f"oracle_forget_prob: {metrics['oracle_forget_prob']:.4f} | "
                    f"oracle_contrast_gap: {metrics['oracle_contrast_gap']:.4f} | "
                    f"resurrection_acc: {metrics['resurrection_acc']:.4f} | "
                    f"resurrection_loss: {metrics['resurrection_loss']:.4f}"
                )

                current_acc = float(metrics["resurrection_acc"])
                current_loss = float(metrics["resurrection_loss"])
                if (
                    current_acc > best_acc
                    or (
                        abs(current_acc - best_acc) <= 1e-12
                        and current_loss < best_loss
                    )
                ):
                    best_acc = current_acc
                    best_loss = current_loss
                    best_epoch = epoch
                    best_state = {
                        "spectral_real": self.perturbed_target_model.spectral_real.detach().cpu().clone(),
                        "spectral_imag": self.perturbed_target_model.spectral_imag.detach().cpu().clone(),
                    }
                    print(
                        f"New best checkpoint: accuracy={best_acc:.4f} "
                        f"loss={best_loss:.4f} epoch={best_epoch}"
                    )

        if best_state is None:
            best_state = {
                "spectral_real": self.perturbed_target_model.spectral_real.detach().cpu().clone(),
                "spectral_imag": self.perturbed_target_model.spectral_imag.detach().cpu().clone(),
            }
            best_acc = max(best_acc, 0.0)
            best_loss = float(self.attack_history[-1]["resurrection_loss"]) if self.attack_history else float("inf")
            best_epoch = int(self.attack_history[-1]["epoch"]) if self.attack_history else 0

        print(
            f"Restoring best Fourier state with accuracy {best_acc:.4f} "
            f"loss {best_loss:.4f} from epoch {best_epoch}"
        )
        self.perturbed_target_model.spectral_real.data.copy_(best_state["spectral_real"].to(self.device))
        self.perturbed_target_model.spectral_imag.data.copy_(best_state["spectral_imag"].to(self.device))
        restored_metrics = self._evaluate_resurrection(val_forget_loader)
        restored_metrics["epoch"] = float(best_epoch if best_epoch >= 0 else 0.0)
        restored_metrics["restored_best"] = 1.0
        restored_metrics["selected_by"] = "resurrection_acc_then_loss"
        self.attack_history.append(restored_metrics)
        return self.attack_history

    def save_attack_artifact(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        torch.save(
            {
                "spectral_real": self.perturbed_target_model.spectral_real.detach().cpu(),
                "spectral_imag": self.perturbed_target_model.spectral_imag.detach().cpu(),
            },
            os.path.join(save_path, "fourier_state.pt"),
        )
        attack_info = {
            "artifact_type": "universal_fourier",
            "forget_class": self.forget_class,
            "attack_config": self.attack_config,
            "attack_history": self.attack_history,
        }
        if self.harp_state is not None:
            attack_info["harp_summary"] = self.harp_state.get("summary", {})
        with open(os.path.join(save_path, "attack_info.json"), "w", encoding="utf-8") as handle:
            json.dump(attack_info, handle, indent=2)

    def load_attack_artifact(self, load_path: str):
        state_path = os.path.join(load_path, "fourier_state.pt")
        state = torch.load(state_path, map_location=self.device)
        self.perturbed_target_model.spectral_real.data.copy_(state["spectral_real"].to(self.device))
        self.perturbed_target_model.spectral_imag.data.copy_(state["spectral_imag"].to(self.device))

        info_path = os.path.join(load_path, "attack_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as handle:
                info = json.load(handle)
            self.attack_history = info.get("attack_history", [])

    def get_attack_statistics(self) -> Dict[str, float]:
        delta = self.perturbed_target_model.get_delta().detach()
        spectrum = self.perturbed_target_model.get_complex_spectrum().detach()
        mask = self.perturbed_target_model.frequency_mask.detach()
        return {
            "epsilon": float(self.perturbed_target_model.epsilon),
            "image_size": int(self.perturbed_target_model.image_size),
            "patch_stride": int(self.perturbed_target_model.patch_stride),
            "mask_density": float(mask.mean().item()),
            "delta_mean_abs": float(delta.abs().mean().item()),
            "delta_max_abs": float(delta.abs().max().item()),
            "delta_l2": float(torch.linalg.vector_norm(delta).item()),
            "spectrum_mean_abs": float(spectrum.abs().mean().item()),
            "spectrum_max_abs": float(spectrum.abs().max().item()),
        }
